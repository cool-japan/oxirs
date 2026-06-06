//! # HDT Triples Section
//!
//! Adjacency-list representation of RDF triple IDs.  In the HDT format the
//! triple store is split into two bit-vectors (`Y` and `Z`) that encode
//! predicate and object adjacency lists, plus two parallel integer arrays
//! (`array_y` and `array_z`) holding the actual predicate and object IDs.
//!
//! ## Model
//!
//! We implement the _dense_ (non-compressed) variant: each subject in
//! 1..N is associated with a contiguous run of (predicate, object) pairs.
//! The bitmap arrays indicate where each subject's run ends (bit set = last
//! predicate for that subject, etc.).
//!
//! The on-disk format parsed here is deliberately minimal so that hand-crafted
//! test payloads can exercise the full iterator without needing real HDT files.
//!
//! ### Binary layout (simplified, little-endian)
//! ```text
//! [4 bytes]  count_sy  — number of (subject, predicate) pairs
//! [4 bytes]  count_z   — number of triples total
//! [count_sy × 4 LE u32]  array_y   — predicate IDs
//! [count_z  × 4 LE u32]  array_z   — object IDs
//! [count_sy × 4 LE u32]  bitmap_y_raw (one u32; non-zero = boundary)
//! [count_z  × 4 LE u32]  bitmap_z_raw (one u32; non-zero = boundary)
//! ```
//!
//! For the iterator we use a simple flat approach: we store parallel arrays of
//! (subject_id, predicate_id, object_id) derived during parsing.

use super::HdtError;

// ---------------------------------------------------------------------------
// Bitmap utility functions
// ---------------------------------------------------------------------------

/// Access bit `pos` in a packed `&[u64]` array.
///
/// Bits are stored LSB-first within each 64-bit word.
/// Returns `false` if `pos` is out of range.
pub fn bitmap_access(bitmap: &[u64], pos: usize) -> bool {
    let word_idx = pos / 64;
    let bit_idx = pos % 64;
    bitmap
        .get(word_idx)
        .map(|w| (w >> bit_idx) & 1 == 1)
        .unwrap_or(false)
}

/// Count the number of `1` bits in `bitmap[0..word_boundary]` up to (but not
/// including) bit position `pos`.
///
/// This is the standard *popcount-rank* operation used to navigate HDT
/// adjacency lists: `bitmap_rank(bm, k)` answers "how many 1-bits are there
/// in positions 0 … k-1?".
pub fn bitmap_rank(bitmap: &[u64], pos: usize) -> u64 {
    if pos == 0 {
        return 0;
    }
    let full_words = pos / 64;
    let remainder = pos % 64;
    let mut count: u64 = 0;
    for &w in bitmap.iter().take(full_words) {
        count += w.count_ones() as u64;
    }
    if remainder > 0 {
        if let Some(&w) = bitmap.get(full_words) {
            // Mask off the bits at `remainder` and above
            let mask = (1u64 << remainder).wrapping_sub(1);
            count += (w & mask).count_ones() as u64;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// TriplesBitmap
// ---------------------------------------------------------------------------

/// SPO-ordered triple store using bitmap adjacency lists.
///
/// This is the in-memory representation of the HDT *bitmap triples* section.
/// Subject, predicate, and object adjacency boundaries are stored as packed
/// `u64` bitmaps; parallel integer arrays hold the actual IDs.
#[derive(Debug, Clone)]
pub struct TriplesBitmap {
    /// Y-plane bitmap: 1-bit at position i means position i is the last
    /// predicate for its subject.
    pub bitmap_y: Vec<u64>,
    /// Z-plane bitmap: 1-bit at position i means position i is the last
    /// object for its (subject, predicate) pair.
    pub bitmap_z: Vec<u64>,
    /// Predicate IDs in adjacency-list order (one per (s,p) slot).
    pub array_y: Vec<u32>,
    /// Object IDs in adjacency-list order (one per triple).
    pub array_z: Vec<u32>,
}

impl TriplesBitmap {
    /// Read a bitmap from `data`.
    ///
    /// The format is: `[4-byte LE u32 count][count × 8-byte LE u64 words]`.
    ///
    /// # Errors
    /// Returns `HdtError::TripleDecodeError` if `data` is truncated.
    pub fn read_bitmap(data: &[u8]) -> Result<Vec<u64>, HdtError> {
        if data.len() < 4 {
            return Err(HdtError::TripleDecodeError {
                msg: "bitmap: truncated before count field".to_owned(),
            });
        }
        let count = u32::from_le_bytes(data[..4].try_into().map_err(|_| HdtError::TripleDecodeError {
            msg: "bitmap: cannot read count".to_owned(),
        })?) as usize;
        let needed = 4 + count * 8;
        if data.len() < needed {
            return Err(HdtError::TripleDecodeError {
                msg: format!("bitmap: need {} bytes, have {}", needed, data.len()),
            });
        }
        let mut words = Vec::with_capacity(count);
        for i in 0..count {
            let off = 4 + i * 8;
            let w = u64::from_le_bytes(
                data[off..off + 8]
                    .try_into()
                    .map_err(|_| HdtError::TripleDecodeError {
                        msg: format!("bitmap: cannot read word {}", i),
                    })?,
            );
            words.push(w);
        }
        Ok(words)
    }

    /// Iterate all triples in SPO order by traversing the adjacency-list bitmaps.
    ///
    /// Subject IDs are implicit: subject 1 owns all (s,p) slots until the first
    /// 1-bit in `bitmap_y`, subject 2 owns the next run, and so on.
    pub fn iterate_spo(&self) -> impl Iterator<Item = (u32, u32, u32)> + '_ {
        let triples = materialise_triples_from_bitmap(
            &self.array_y,
            &self.array_z,
            &self.bitmap_y,
            &self.bitmap_z,
        );
        triples.into_iter()
    }

    /// Collect all triples with the given subject ID.
    pub fn lookup_by_subject(&self, s_id: u32) -> Vec<(u32, u32, u32)> {
        self.iterate_spo()
            .filter(|(s, _, _)| *s == s_id)
            .collect()
    }

    /// Collect all triples with the given predicate ID.
    pub fn lookup_by_predicate(&self, p_id: u32) -> Vec<(u32, u32, u32)> {
        self.iterate_spo()
            .filter(|(_, p, _)| *p == p_id)
            .collect()
    }
}

/// Materialise `(subject_id, predicate_id, object_id)` from packed-bitmap
/// adjacency lists.
fn materialise_triples_from_bitmap(
    array_y: &[u32],
    array_z: &[u32],
    bitmap_y: &[u64],
    bitmap_z: &[u64],
) -> Vec<(u32, u32, u32)> {
    let mut triples = Vec::new();
    let mut subject_id: u32 = 1;
    let mut z_index: usize = 0;

    for (sy_idx, &pred_id) in array_y.iter().enumerate() {
        // Collect all object IDs for this (s, p) slot
        loop {
            if z_index >= array_z.len() {
                break;
            }
            let obj_id = array_z[z_index];
            triples.push((subject_id, pred_id, obj_id));
            let is_last_z = bitmap_access(bitmap_z, z_index);
            z_index += 1;
            if is_last_z {
                break;
            }
        }
        // Advance subject when the Y-bitmap fires
        if bitmap_access(bitmap_y, sy_idx) {
            subject_id += 1;
        }
    }
    triples
}

// ---------------------------------------------------------------------------
// HdtTriplesSection  (backward-compatible flat representation)
// ---------------------------------------------------------------------------

/// Owns all adjacency-list data for a parsed HDT triples section.
///
/// After parsing the raw arrays we materialise the full triple ID tuples into
/// `triples` for straightforward iteration.
#[derive(Debug, Clone)]
pub struct HdtTriplesSection {
    /// Y-plane bitmap (one bit per (s,p) pair; 1 = last predicate for this subject).
    pub bitmap_y: Vec<u64>,
    /// Z-plane bitmap (one bit per triple; 1 = last object for this (s,p) pair).
    pub bitmap_z: Vec<u64>,
    /// Predicate IDs in adjacency-list order.
    pub array_y: Vec<u32>,
    /// Object IDs in adjacency-list order.
    pub array_z: Vec<u32>,
    /// Materialised (subject_id, predicate_id, object_id) triples.
    triples: Vec<(u32, u32, u32)>,
}

impl HdtTriplesSection {
    /// Parse a simplified HDT triples section from a byte slice.
    ///
    /// # Format
    /// ```text
    /// [4 LE u32]  count_sy  — number of (subj, pred) pairs  (= len of array_y)
    /// [4 LE u32]  count_z   — total triples                 (= len of array_z)
    /// [count_sy × 4 LE u32]  array_y   — predicate IDs
    /// [count_z  × 4 LE u32]  array_z   — object IDs
    /// [count_sy × 4 LE u32]  bitmap_y_raw (one u32 per entry; non-zero = boundary)
    /// [count_z  × 4 LE u32]  bitmap_z_raw (one u32 per entry; non-zero = boundary)
    /// ```
    ///
    /// # Errors
    /// Returns `HdtError::TripleDecodeError` if data is truncated or the format
    /// is unrecognised.
    pub fn parse(data: &[u8]) -> Result<Self, HdtError> {
        let mut off = 0usize;

        let count_sy = read_u32_le(data, &mut off)
            .ok_or_else(|| HdtError::TripleDecodeError { msg: "cannot read count_sy".to_owned() })?
            as usize;
        let count_z = read_u32_le(data, &mut off)
            .ok_or_else(|| HdtError::TripleDecodeError { msg: "cannot read count_z".to_owned() })?
            as usize;

        // array_y: predicate IDs (one per (s,p) slot)
        let mut array_y = Vec::with_capacity(count_sy);
        for i in 0..count_sy {
            let v = read_u32_le(data, &mut off).ok_or_else(|| HdtError::TripleDecodeError {
                msg: format!("array_y truncated at index {}", i),
            })?;
            array_y.push(v);
        }

        // array_z: object IDs (one per triple)
        let mut array_z = Vec::with_capacity(count_z);
        for i in 0..count_z {
            let v = read_u32_le(data, &mut off).ok_or_else(|| HdtError::TripleDecodeError {
                msg: format!("array_z truncated at index {}", i),
            })?;
            array_z.push(v);
        }

        // bitmap_y_raw: boundary flags for subject adjacency list (u32 per entry)
        let mut bitmap_y_raw = Vec::with_capacity(count_sy);
        for i in 0..count_sy {
            let v = read_u32_le(data, &mut off).ok_or_else(|| HdtError::TripleDecodeError {
                msg: format!("bitmap_y truncated at index {}", i),
            })?;
            bitmap_y_raw.push(v);
        }

        // bitmap_z_raw: boundary flags for predicate adjacency list (u32 per entry)
        let mut bitmap_z_raw = Vec::with_capacity(count_z);
        for i in 0..count_z {
            let v = read_u32_le(data, &mut off).ok_or_else(|| HdtError::TripleDecodeError {
                msg: format!("bitmap_z truncated at index {}", i),
            })?;
            bitmap_z_raw.push(v);
        }

        // Convert raw u32 boundary flags to packed u64 bitmaps.
        let bitmap_y = pack_bitmap_from_raw(&bitmap_y_raw);
        let bitmap_z = pack_bitmap_from_raw(&bitmap_z_raw);

        // Materialise triple ID tuples using the packed bitmaps.
        let triples =
            materialise_triples_from_bitmap(&array_y, &array_z, &bitmap_y, &bitmap_z);

        Ok(HdtTriplesSection {
            bitmap_y,
            bitmap_z,
            array_y,
            array_z,
            triples,
        })
    }

    /// Iterate over `(subject_id, predicate_id, object_id)` tuples.
    pub fn iter_ids(&self) -> impl Iterator<Item = (u32, u32, u32)> + '_ {
        self.triples.iter().copied()
    }

    /// Convert to a `TriplesBitmap` for richer lookup operations.
    pub fn to_bitmap(&self) -> TriplesBitmap {
        TriplesBitmap {
            bitmap_y: self.bitmap_y.clone(),
            bitmap_z: self.bitmap_z.clone(),
            array_y: self.array_y.clone(),
            array_z: self.array_z.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Read a little-endian `u32` from `data` at `*offset`, advancing it by 4.
fn read_u32_le(data: &[u8], offset: &mut usize) -> Option<u32> {
    if *offset + 4 > data.len() {
        return None;
    }
    let bytes: [u8; 4] = data[*offset..*offset + 4].try_into().ok()?;
    *offset += 4;
    Some(u32::from_le_bytes(bytes))
}

/// Pack a slice of per-element boundary flags (non-zero = set) into a
/// `Vec<u64>` with one bit per element (LSB first in each word).
fn pack_bitmap_from_raw(flags: &[u32]) -> Vec<u64> {
    let num_words = (flags.len() + 63) / 64;
    let mut words = vec![0u64; num_words];
    for (i, &flag) in flags.iter().enumerate() {
        if flag != 0 {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    words
}
