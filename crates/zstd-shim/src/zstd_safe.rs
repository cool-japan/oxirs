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
