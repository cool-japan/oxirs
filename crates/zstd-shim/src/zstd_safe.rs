//! Low-level `zstd_safe`-compatible helpers implemented in Pure Rust.
//!
//! These mirror the small subset of the `zstd_safe` surface used by downstream
//! crates: a compression-bound estimator and a frame-content-size parser.

/// Maximum number of bytes the compressed output may require for `src_size`
/// input bytes.
///
/// This mirrors the canonical `ZSTD_COMPRESSBOUND` macro from `zstd.h`:
///
/// ```text
/// #define ZSTD_COMPRESSBOUND(srcSize) \
///     ((srcSize) + ((srcSize) >> 8) + \
///      (((srcSize) < (128 << 10)) ? (((128 << 10) - (srcSize)) >> 11) : 0))
/// ```
///
/// An extra `+64` safety margin is added to stay conservative across encoder
/// implementations.
pub fn compress_bound(src_size: usize) -> usize {
    const ZSTD_BLOCKSIZE_MAX: usize = 128 * 1024;
    let margin = if src_size < ZSTD_BLOCKSIZE_MAX { (ZSTD_BLOCKSIZE_MAX - src_size) >> 11 } else { 0 };
    src_size
        .saturating_add(src_size >> 8)
        .saturating_add(margin)
        .saturating_add(64)
}

/// Error returned when a frame's content size cannot be determined.
#[derive(Debug)]
pub struct ContentSizeError;

impl std::fmt::Display for ContentSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "zstd frame content size unavailable")
    }
}

impl std::error::Error for ContentSizeError {}

/// Parse the `Frame_Content_Size` field from a STANDARD Zstandard frame header.
///
/// Returns `Ok(Some(size))` when the size is present, `Ok(None)` when the frame
/// header does not carry a content size, and `Err(ContentSizeError)` when the
/// input is too short, has the wrong magic, or is otherwise malformed.
///
/// The implementation follows the Zstandard frame format specification (RFC
/// 8878) and bounds-checks every byte access, so it never panics.
pub fn get_frame_content_size(src: &[u8]) -> Result<Option<u64>, ContentSizeError> {
    if src.len() < 5 {
        return Err(ContentSizeError);
    }

    // Magic number: 0xFD2FB528 little-endian => bytes [0x28, 0xB5, 0x2F, 0xFD].
    let m0 = *src.first().ok_or(ContentSizeError)?;
    let m1 = *src.get(1).ok_or(ContentSizeError)?;
    let m2 = *src.get(2).ok_or(ContentSizeError)?;
    let m3 = *src.get(3).ok_or(ContentSizeError)?;
    if [m0, m1, m2, m3] != [0x28, 0xB5, 0x2F, 0xFD] {
        return Err(ContentSizeError);
    }

    let fhd = *src.get(4).ok_or(ContentSizeError)?;
    let fcs_flag = fhd >> 6;
    let single_segment = (fhd >> 5) & 1 == 1;
    let did_flag = fhd & 3;

    let fcs_size: usize = match fcs_flag {
        0 => if single_segment { 1 } else { 0 },
        1 => 2,
        2 => 4,
        3 => 8,
        _ => return Err(ContentSizeError),
    };

    let window_bytes: usize = if single_segment { 0 } else { 1 };

    let did_size: usize = match did_flag {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 4,
        _ => return Err(ContentSizeError),
    };

    let offset = 5 + window_bytes + did_size;

    if fcs_size == 0 {
        return Ok(None);
    }

    let value = match fcs_size {
        1 => {
            let b0 = *src.get(offset).ok_or(ContentSizeError)?;
            b0 as u64
        }
        2 => {
            let b0 = *src.get(offset).ok_or(ContentSizeError)?;
            let b1 = *src.get(offset + 1).ok_or(ContentSizeError)?;
            u16::from_le_bytes([b0, b1]) as u64 + 256
        }
        4 => {
            let b0 = *src.get(offset).ok_or(ContentSizeError)?;
            let b1 = *src.get(offset + 1).ok_or(ContentSizeError)?;
            let b2 = *src.get(offset + 2).ok_or(ContentSizeError)?;
            let b3 = *src.get(offset + 3).ok_or(ContentSizeError)?;
            u32::from_le_bytes([b0, b1, b2, b3]) as u64
        }
        8 => {
            let b0 = *src.get(offset).ok_or(ContentSizeError)?;
            let b1 = *src.get(offset + 1).ok_or(ContentSizeError)?;
            let b2 = *src.get(offset + 2).ok_or(ContentSizeError)?;
            let b3 = *src.get(offset + 3).ok_or(ContentSizeError)?;
            let b4 = *src.get(offset + 4).ok_or(ContentSizeError)?;
            let b5 = *src.get(offset + 5).ok_or(ContentSizeError)?;
            let b6 = *src.get(offset + 6).ok_or(ContentSizeError)?;
            let b7 = *src.get(offset + 7).ok_or(ContentSizeError)?;
            u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
        }
        _ => return Err(ContentSizeError),
    };

    Ok(Some(value))
}

/// A growable byte-sink destination for [`crate::bulk`]'s buffer-taking
/// `compress_to_buffer`/`decompress_to_buffer` methods.
///
/// The real `zstd_safe::WriteBuf` is an `unsafe trait` built around raw
/// pointers into possibly-uninitialized memory (`as_mut_ptr` +
/// `filled_until`). This is a **safe** reimplementation of just the
/// observable contract — "write starting at the current fill position,
/// growing storage as needed" plus "how much spare capacity is already
/// provisioned" for the decompression-bomb guard — scoped to what downstream
/// consumers of this shim (parquet, tantivy, pulsar, wasmtime) actually pass:
/// a plain `Vec<u8>`, or a `Cursor` wrapping one that's been positioned via
/// `Cursor::set_position` to mark "start writing here" against an
/// already-`reserve`d buffer (parquet's `Codec::compress`/`decompress` do
/// exactly this to write compressed pages at an offset inside a shared
/// buffer). Growth uses safe `Vec::resize` + slice copy rather than an
/// unsafe uninitialized-memory write, at a small, acceptable extra-copy cost.
pub trait WriteBuf {
    /// Capacity available for writing at (and beyond) the current write
    /// position without the destination needing to reallocate — i.e. how
    /// many bytes the caller already provisioned. Used as the
    /// decompression-bomb guard bound in [`crate::bulk::Decompressor`].
    fn spare_capacity(&self) -> usize;

    /// Write `data` starting at the destination's current write position
    /// (appending, for a plain growable buffer; at `Cursor::position()` for
    /// a `Cursor`), growing the underlying storage if `data` extends past
    /// what was already provisioned. Advances the write position by
    /// `data.len()`.
    fn write_at_pos(&mut self, data: &[u8]);
}

impl WriteBuf for Vec<u8> {
    fn spare_capacity(&self) -> usize {
        self.capacity().saturating_sub(self.len())
    }

    fn write_at_pos(&mut self, data: &[u8]) {
        self.extend_from_slice(data);
    }
}

impl WriteBuf for &mut Vec<u8> {
    fn spare_capacity(&self) -> usize {
        (**self).capacity().saturating_sub((**self).len())
    }

    fn write_at_pos(&mut self, data: &[u8]) {
        (**self).extend_from_slice(data);
    }
}

/// Write `data` into `vec` at byte offset `pos`, growing `vec` (zero-padding
/// any gap before `pos`, matching the real trait's implicit "already
/// reserved, may be unfilled" contract without leaving anything actually
/// uninitialized) so that `vec.len() >= pos + data.len()` afterward.
fn write_vec_at_pos(vec: &mut Vec<u8>, pos: usize, data: &[u8]) {
    let end = pos.saturating_add(data.len());
    if vec.len() < pos {
        vec.resize(pos, 0);
    }
    if vec.len() < end {
        vec.resize(end, 0);
    }
    vec[pos..end].copy_from_slice(data);
}

impl WriteBuf for std::io::Cursor<&mut Vec<u8>> {
    fn spare_capacity(&self) -> usize {
        let pos = self.position() as usize;
        self.get_ref().capacity().saturating_sub(pos)
    }

    fn write_at_pos(&mut self, data: &[u8]) {
        let pos = self.position() as usize;
        write_vec_at_pos(self.get_mut(), pos, data);
        self.set_position((pos as u64).saturating_add(data.len() as u64));
    }
}

impl WriteBuf for std::io::Cursor<Vec<u8>> {
    fn spare_capacity(&self) -> usize {
        let pos = self.position() as usize;
        self.get_ref().capacity().saturating_sub(pos)
    }

    fn write_at_pos(&mut self, data: &[u8]) {
        let pos = self.position() as usize;
        write_vec_at_pos(self.get_mut(), pos, data);
        self.set_position((pos as u64).saturating_add(data.len() as u64));
    }
}

#[cfg(test)]
mod write_buf_tests {
    use super::WriteBuf;
    use std::io::Cursor;

    #[test]
    fn vec_write_at_pos_appends() {
        let mut v: Vec<u8> = vec![1, 2, 3];
        v.write_at_pos(&[4, 5]);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn cursor_owned_vec_write_at_pos_appends_when_positioned_at_end() {
        let mut cur = Cursor::new(vec![1u8, 2, 3]);
        cur.set_position(3);
        cur.write_at_pos(&[4, 5]);
        assert_eq!(cur.into_inner(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn cursor_borrowed_vec_write_at_pos_appends_when_positioned_at_end() {
        // Mirrors parquet's Codec::compress/decompress exactly: reserve, wrap
        // the existing Vec in a Cursor, position it at the pre-reserve length.
        let mut buf: Vec<u8> = vec![0xAA, 0xBB];
        let offset = buf.len();
        buf.reserve(16);
        {
            let mut cur = Cursor::new(&mut buf);
            cur.set_position(offset as u64);
            cur.write_at_pos(&[1, 2, 3]);
        }
        assert_eq!(buf, vec![0xAA, 0xBB, 1, 2, 3]);
    }

    #[test]
    fn cursor_borrowed_vec_write_at_pos_overwrites_in_place_when_repositioned() {
        let mut buf: Vec<u8> = vec![0, 0, 0, 0, 0];
        {
            let mut cur = Cursor::new(&mut buf);
            cur.set_position(1);
            cur.write_at_pos(&[9, 9]);
        }
        assert_eq!(buf, vec![0, 9, 9, 0, 0]);
    }

    #[test]
    fn spare_capacity_reflects_reservation_from_cursor_position() {
        let mut buf: Vec<u8> = Vec::with_capacity(10);
        buf.extend_from_slice(&[1, 2]);
        let offset = buf.len();
        let mut cur = Cursor::new(&mut buf);
        cur.set_position(offset as u64);
        assert_eq!(cur.spare_capacity(), 8);
    }
}
