//! Delta compression for RDF triple storage.
//!
//! This module stores a sequence of triples as *deltas* from a base snapshot.
//! The technique exploits the fact that RDF graphs change incrementally—most
//! write transactions add or remove only a small number of triples relative to
//! the existing dataset.
//!
//! # Encoding scheme
//!
//! ```text
//! TripleDeltaStore
//!  ├── base_snapshot: Vec<EncodedTriple>   (full set at snapshot time)
//!  └── deltas: Vec<TripleDelta>            (ordered list of change sets)
//!       ├── TripleDelta::Insertion(triple)
//!       └── TripleDelta::Deletion(triple)
//! ```
//!
//! When the number of deltas exceeds a configurable threshold the store
//! *materialises* the current logical state into a new base snapshot and
//! clears the delta log, keeping memory usage bounded.
//!
//! # Wire format for a single delta entry (12 bytes)
//!
//! ```text
//! ┌──────────┬──────────┬──────────┬──────────┐
//! │ tag (1B) │  Δs (3B) │  Δp (3B) │  Δo (3B) │
//! └──────────┴──────────┴──────────┴──────────┘
//! ```
//!
//! The tag byte is `0x00` for insertion and `0x01` for deletion.  Each 24-bit
//! component encodes the *signed zig-zag delta* from the previous triple's
//! corresponding component, allowing representation of changes up to ±8 MB in
//! the node-ID space without fallback to full 8-byte encoding.  Larger deltas
//! use a 9-byte fallback (tag `0x02`/`0x03`) where the full 8-byte value
//! follows the tag.

use crate::error::{Result, TdbError};
use crate::index::btree_index::EncodedTriple;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default threshold: compact (materialise) after this many delta entries.
pub const DEFAULT_COMPACTION_THRESHOLD: usize = 10_000;

/// Zig-zag encoded values that fit in 24 bits (unsigned).
const ZZ24_MAX: u64 = (1 << 24) - 1;

// ---------------------------------------------------------------------------
// TripleDelta
// ---------------------------------------------------------------------------

/// A single delta record: either an insertion or deletion of one triple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TripleDelta {
    /// Record that this triple was added.
    Insertion(EncodedTriple),
    /// Record that this triple was removed.
    Deletion(EncodedTriple),
}

impl TripleDelta {
    /// Return the triple affected by this delta.
    pub fn triple(&self) -> &EncodedTriple {
        match self {
            Self::Insertion(t) | Self::Deletion(t) => t,
        }
    }

    /// Return `true` if this is an insertion.
    pub fn is_insertion(&self) -> bool {
        matches!(self, Self::Insertion(_))
    }

    /// Return `true` if this is a deletion.
    pub fn is_deletion(&self) -> bool {
        matches!(self, Self::Deletion(_))
    }
}

// ---------------------------------------------------------------------------
// Wire encoding helpers
// ---------------------------------------------------------------------------

/// Zig-zag encode a signed 64-bit value to an unsigned 64-bit value.
fn zig_zag_encode(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Zig-zag decode an unsigned 64-bit value to a signed 64-bit value.
fn zig_zag_decode(v: u64) -> i64 {
    ((v >> 1) ^ ((v & 1).wrapping_neg())) as i64
}

/// Compute the signed delta between two u64 node IDs.
fn delta_i64(current: u64, prev: u64) -> i64 {
    (current as i64).wrapping_sub(prev as i64)
}

/// Apply a signed delta to a u64 base value.
fn apply_delta(base: u64, delta: i64) -> u64 {
    (base as i64).wrapping_add(delta) as u64
}

/// Write a 3-byte little-endian value.
fn write_u24(buf: &mut Vec<u8>, v: u32) {
    buf.push((v & 0xFF) as u8);
    buf.push(((v >> 8) & 0xFF) as u8);
    buf.push(((v >> 16) & 0xFF) as u8);
}

/// Read a 3-byte little-endian value.
fn read_u24(buf: &[u8]) -> u32 {
    (buf[0] as u32) | ((buf[1] as u32) << 8) | ((buf[2] as u32) << 16)
}

// Tag bytes for encoded delta entries
const TAG_INSERT_SHORT: u8 = 0x00; // Insertion, all three deltas fit in 24 bits
const TAG_DELETE_SHORT: u8 = 0x01; // Deletion, all three deltas fit in 24 bits
const TAG_INSERT_FULL: u8 = 0x02; // Insertion, at least one delta needs 64 bits (24 bytes)
const TAG_DELETE_FULL: u8 = 0x03; // Deletion, at least one delta needs 64 bits (24 bytes)

// ---------------------------------------------------------------------------
// EncodedDeltaLog
// ---------------------------------------------------------------------------

/// A compact binary log of triple deltas.
///
/// Each entry is serialised using delta-of-delta encoding relative to the
/// previous entry, minimising byte usage for near-sequential node IDs.
#[derive(Debug, Clone, Default)]
pub struct EncodedDeltaLog {
    data: Vec<u8>,
    count: usize,
}

impl EncodedDeltaLog {
    /// Create an empty delta log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of delta entries in the log.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Return `true` if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Byte size of the encoded log.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Append a delta entry. `prev` is the previous entry's triple (for
    /// computing deltas); pass `None` for the very first entry.
    pub fn push(&mut self, delta: &TripleDelta, prev: Option<&EncodedTriple>) {
        let triple = delta.triple();
        let (ps, pp, po) = prev.map_or((0u64, 0u64, 0u64), |p| (p.s, p.p, p.o));

        let ds = zig_zag_encode(delta_i64(triple.s, ps));
        let dp = zig_zag_encode(delta_i64(triple.p, pp));
        let do_ = zig_zag_encode(delta_i64(triple.o, po));

        let is_insertion = delta.is_insertion();

        if ds <= ZZ24_MAX && dp <= ZZ24_MAX && do_ <= ZZ24_MAX {
            // Short encoding: 1 + 3 + 3 + 3 = 10 bytes
            let tag = if is_insertion {
                TAG_INSERT_SHORT
            } else {
                TAG_DELETE_SHORT
            };
            self.data.push(tag);
            write_u24(&mut self.data, ds as u32);
            write_u24(&mut self.data, dp as u32);
            write_u24(&mut self.data, do_ as u32);
        } else {
            // Full encoding: 1 + 8 + 8 + 8 = 25 bytes
            let tag = if is_insertion {
                TAG_INSERT_FULL
            } else {
                TAG_DELETE_FULL
            };
            self.data.push(tag);
            self.data.extend_from_slice(&triple.s.to_le_bytes());
            self.data.extend_from_slice(&triple.p.to_le_bytes());
            self.data.extend_from_slice(&triple.o.to_le_bytes());
        }

        self.count += 1;
    }

    /// Decode all delta entries from the log.
    pub fn decode_all(&self) -> Result<Vec<TripleDelta>> {
        let mut result = Vec::with_capacity(self.count);
        let mut pos = 0usize;
        let data = &self.data;
        let mut prev = EncodedTriple::new(0, 0, 0);

        while pos < data.len() {
            if pos >= data.len() {
                break;
            }
            let tag = data[pos];
            pos += 1;

            let (triple, is_insertion) = match tag {
                TAG_INSERT_SHORT | TAG_DELETE_SHORT => {
                    if pos + 9 > data.len() {
                        return Err(TdbError::Deserialization(
                            "delta log truncated (short entry)".to_string(),
                        ));
                    }
                    let ds = zig_zag_decode(read_u24(&data[pos..pos + 3]) as u64);
                    let dp = zig_zag_decode(read_u24(&data[pos + 3..pos + 6]) as u64);
                    let do_ = zig_zag_decode(read_u24(&data[pos + 6..pos + 9]) as u64);
                    pos += 9;
                    let s = apply_delta(prev.s, ds);
                    let p = apply_delta(prev.p, dp);
                    let o = apply_delta(prev.o, do_);
                    let t = EncodedTriple::new(s, p, o);
                    (t, tag == TAG_INSERT_SHORT)
                }
                TAG_INSERT_FULL | TAG_DELETE_FULL => {
                    if pos + 24 > data.len() {
                        return Err(TdbError::Deserialization(
                            "delta log truncated (full entry)".to_string(),
                        ));
                    }
                    let s = u64::from_le_bytes(data[pos..pos + 8].try_into().map_err(|_| {
                        TdbError::Deserialization("invalid delta entry".to_string())
                    })?);
                    let p =
                        u64::from_le_bytes(data[pos + 8..pos + 16].try_into().map_err(|_| {
                            TdbError::Deserialization("invalid delta entry".to_string())
                        })?);
                    let o =
                        u64::from_le_bytes(data[pos + 16..pos + 24].try_into().map_err(|_| {
                            TdbError::Deserialization("invalid delta entry".to_string())
                        })?);
                    pos += 24;
                    (EncodedTriple::new(s, p, o), tag == TAG_INSERT_FULL)
                }
                other => {
                    return Err(TdbError::Deserialization(format!(
                        "unknown delta tag: 0x{:02x}",
                        other
                    )));
                }
            };

            prev = triple;
            if is_insertion {
                result.push(TripleDelta::Insertion(triple));
            } else {
                result.push(TripleDelta::Deletion(triple));
            }
        }

        Ok(result)
    }

    /// Clear the log.
    pub fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// TripleDeltaStore
// ---------------------------------------------------------------------------

/// Configuration for [`TripleDeltaStore`].
#[derive(Debug, Clone)]
pub struct DeltaStoreConfig {
    /// Number of delta entries after which the store automatically compacts.
    pub compaction_threshold: usize,
}

impl Default for DeltaStoreConfig {
    fn default() -> Self {
        Self {
            compaction_threshold: DEFAULT_COMPACTION_THRESHOLD,
        }
    }
}

/// Statistics snapshot for a [`TripleDeltaStore`].
#[derive(Debug, Clone)]
pub struct DeltaStoreStats {
    /// Number of triples in the base snapshot.
    pub base_triples: usize,
    /// Number of raw delta entries in the log.
    pub delta_entries: usize,
    /// Total logical triples (materialised count).
    pub logical_triples: usize,
    /// Raw byte size of the encoded delta log.
    pub delta_bytes: usize,
    /// Number of compaction operations performed.
    pub compaction_count: u64,
}

/// A triple store that uses a base snapshot plus a delta log for space-efficient
/// incremental updates.
///
/// # Usage
///
/// ```rust,ignore
/// let mut store = TripleDeltaStore::new(DeltaStoreConfig::default());
/// store.insert(EncodedTriple::new(1, 2, 3))?;
/// store.insert(EncodedTriple::new(4, 5, 6))?;
/// store.delete(&EncodedTriple::new(1, 2, 3))?;
///
/// let triples = store.materialise()?;
/// assert_eq!(triples.len(), 1);
/// ```
pub struct TripleDeltaStore {
    config: DeltaStoreConfig,
    /// Base snapshot: sorted vector of triples at the time of last compaction.
    base_snapshot: Vec<EncodedTriple>,
    /// Encoded delta log appended since last compaction.
    delta_log: EncodedDeltaLog,
    /// Most-recently appended triple (for computing deltas).
    last_triple: Option<EncodedTriple>,
    /// Count of compaction operations performed.
    compaction_count: u64,
}

impl TripleDeltaStore {
    /// Create a new, empty delta store with the given configuration.
    pub fn new(config: DeltaStoreConfig) -> Self {
        Self {
            config,
            base_snapshot: Vec::new(),
            delta_log: EncodedDeltaLog::new(),
            last_triple: None,
            compaction_count: 0,
        }
    }

    /// Create a delta store from an existing snapshot (e.g. loaded from disk).
    pub fn from_snapshot(config: DeltaStoreConfig, snapshot: Vec<EncodedTriple>) -> Self {
        Self {
            config,
            base_snapshot: snapshot,
            delta_log: EncodedDeltaLog::new(),
            last_triple: None,
            compaction_count: 0,
        }
    }

    /// Insert a triple. Appends an insertion delta to the log.
    pub fn insert(&mut self, triple: EncodedTriple) -> Result<()> {
        let delta = TripleDelta::Insertion(triple);
        self.delta_log.push(&delta, self.last_triple.as_ref());
        self.last_triple = Some(triple);
        self.maybe_compact()
    }

    /// Delete a triple. Appends a deletion delta to the log.
    pub fn delete(&mut self, triple: &EncodedTriple) -> Result<()> {
        let delta = TripleDelta::Deletion(*triple);
        self.delta_log.push(&delta, self.last_triple.as_ref());
        self.last_triple = Some(*triple);
        self.maybe_compact()
    }

    /// Materialise the current logical triple set by replaying the base snapshot
    /// plus all deltas.  Returns a sorted, deduplicated list of triples.
    pub fn materialise(&self) -> Result<Vec<EncodedTriple>> {
        let mut set: HashSet<EncodedTriple> = self.base_snapshot.iter().copied().collect();

        for delta in self.delta_log.decode_all()? {
            match delta {
                TripleDelta::Insertion(t) => {
                    set.insert(t);
                }
                TripleDelta::Deletion(t) => {
                    set.remove(&t);
                }
            }
        }

        let mut result: Vec<EncodedTriple> = set.into_iter().collect();
        result.sort_by_key(|t| (t.s, t.p, t.o));
        Ok(result)
    }

    /// Return the number of raw delta entries in the log.
    pub fn delta_count(&self) -> usize {
        self.delta_log.len()
    }

    /// Return the size in bytes of the encoded delta log.
    pub fn delta_bytes(&self) -> usize {
        self.delta_log.byte_size()
    }

    /// Return the number of triples in the base snapshot.
    pub fn base_count(&self) -> usize {
        self.base_snapshot.len()
    }

    /// Collect statistics.
    pub fn stats(&self) -> Result<DeltaStoreStats> {
        let logical_triples = self.materialise()?.len();
        Ok(DeltaStoreStats {
            base_triples: self.base_snapshot.len(),
            delta_entries: self.delta_log.len(),
            logical_triples,
            delta_bytes: self.delta_log.byte_size(),
            compaction_count: self.compaction_count,
        })
    }

    /// Force compaction: materialise the current state and reset the delta log.
    pub fn compact(&mut self) -> Result<()> {
        let materialised = self.materialise()?;
        self.base_snapshot = materialised;
        self.delta_log.clear();
        self.last_triple = None;
        self.compaction_count += 1;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn maybe_compact(&mut self) -> Result<()> {
        if self.delta_log.len() >= self.config.compaction_threshold {
            self.compact()
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make(s: u64, p: u64, o: u64) -> EncodedTriple {
        EncodedTriple::new(s, p, o)
    }

    // --- EncodedDeltaLog ---

    #[test]
    fn test_delta_log_roundtrip_insert() {
        let mut log = EncodedDeltaLog::new();
        let t = make(1, 2, 3);
        log.push(&TripleDelta::Insertion(t), None);
        assert_eq!(log.len(), 1);
        let decoded = log.decode_all().unwrap();
        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].is_insertion());
        assert_eq!(decoded[0].triple(), &t);
    }

    #[test]
    fn test_delta_log_roundtrip_delete() {
        let mut log = EncodedDeltaLog::new();
        let t = make(10, 20, 30);
        log.push(&TripleDelta::Deletion(t), None);
        let decoded = log.decode_all().unwrap();
        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].is_deletion());
        assert_eq!(decoded[0].triple(), &t);
    }

    #[test]
    fn test_delta_log_multiple_entries() {
        let mut log = EncodedDeltaLog::new();
        let triples = vec![make(1, 1, 1), make(2, 2, 2), make(3, 3, 3)];
        let mut prev: Option<EncodedTriple> = None;
        for &t in &triples {
            log.push(&TripleDelta::Insertion(t), prev.as_ref());
            prev = Some(t);
        }
        assert_eq!(log.len(), 3);
        let decoded = log.decode_all().unwrap();
        assert_eq!(decoded.len(), 3);
        for (i, d) in decoded.iter().enumerate() {
            assert!(d.is_insertion());
            assert_eq!(d.triple(), &triples[i]);
        }
    }

    #[test]
    fn test_delta_log_large_node_ids_use_full_encoding() {
        let mut log = EncodedDeltaLog::new();
        // Large IDs that won't fit in 24-bit zig-zag delta from 0
        let t = make(u64::MAX / 2, u64::MAX / 3, u64::MAX / 4);
        log.push(&TripleDelta::Insertion(t), None);
        let decoded = log.decode_all().unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].triple(), &t);
    }

    #[test]
    fn test_delta_log_mixed_inserts_deletes() {
        let mut log = EncodedDeltaLog::new();
        let t1 = make(1, 1, 1);
        let t2 = make(2, 2, 2);
        log.push(&TripleDelta::Insertion(t1), None);
        log.push(&TripleDelta::Deletion(t2), Some(&t1));
        let decoded = log.decode_all().unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(decoded[0].is_insertion());
        assert!(decoded[1].is_deletion());
        assert_eq!(decoded[0].triple(), &t1);
        assert_eq!(decoded[1].triple(), &t2);
    }

    // --- TripleDeltaStore ---

    #[test]
    fn test_store_insert_and_materialise() {
        let mut store = TripleDeltaStore::new(DeltaStoreConfig::default());
        store.insert(make(1, 2, 3)).unwrap();
        store.insert(make(4, 5, 6)).unwrap();
        let result = store.materialise().unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_store_delete() {
        let mut store = TripleDeltaStore::new(DeltaStoreConfig::default());
        store.insert(make(1, 2, 3)).unwrap();
        store.insert(make(4, 5, 6)).unwrap();
        store.delete(&make(1, 2, 3)).unwrap();
        let result = store.materialise().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], make(4, 5, 6));
    }

    #[test]
    fn test_store_idempotent_inserts() {
        let mut store = TripleDeltaStore::new(DeltaStoreConfig::default());
        store.insert(make(1, 1, 1)).unwrap();
        store.insert(make(1, 1, 1)).unwrap(); // duplicate
        let result = store.materialise().unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_store_delete_nonexistent() {
        let mut store = TripleDeltaStore::new(DeltaStoreConfig::default());
        // Deleting something that was never inserted is a no-op
        store.delete(&make(99, 99, 99)).unwrap();
        let result = store.materialise().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_store_compaction_triggered() {
        // Use a low threshold so compaction is triggered
        let config = DeltaStoreConfig {
            compaction_threshold: 5,
        };
        let mut store = TripleDeltaStore::new(config);

        for i in 0..10u64 {
            store.insert(make(i, i, i)).unwrap();
        }

        // After auto-compaction the delta log should be short
        assert!(store.compaction_count > 0);
        // But the materialised state must still be correct
        let result = store.materialise().unwrap();
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_store_manual_compact() {
        let mut store = TripleDeltaStore::new(DeltaStoreConfig::default());
        store.insert(make(1, 2, 3)).unwrap();
        store.insert(make(4, 5, 6)).unwrap();
        store.delete(&make(1, 2, 3)).unwrap();

        assert!(store.delta_count() > 0);

        store.compact().unwrap();
        assert_eq!(store.delta_count(), 0);
        assert_eq!(store.base_count(), 1);

        let result = store.materialise().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], make(4, 5, 6));
    }

    #[test]
    fn test_store_from_snapshot() {
        let snapshot = vec![make(1, 1, 1), make(2, 2, 2)];
        let mut store = TripleDeltaStore::from_snapshot(DeltaStoreConfig::default(), snapshot);
        store.insert(make(3, 3, 3)).unwrap();
        let result = store.materialise().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_store_stats() {
        let mut store = TripleDeltaStore::new(DeltaStoreConfig::default());
        store.insert(make(1, 2, 3)).unwrap();
        store.insert(make(4, 5, 6)).unwrap();
        let stats = store.stats().unwrap();
        assert_eq!(stats.logical_triples, 2);
        assert_eq!(stats.delta_entries, 2);
        assert_eq!(stats.compaction_count, 0);
    }

    #[test]
    fn test_zig_zag_roundtrip() {
        for v in [-1_000_000i64, -1, 0, 1, 1_000_000, i64::MAX / 2] {
            let encoded = zig_zag_encode(v);
            assert_eq!(zig_zag_decode(encoded), v);
        }
    }

    #[test]
    fn test_delta_log_clear() {
        let mut log = EncodedDeltaLog::new();
        log.push(&TripleDelta::Insertion(make(1, 2, 3)), None);
        assert_eq!(log.len(), 1);
        log.clear();
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
    }
}
