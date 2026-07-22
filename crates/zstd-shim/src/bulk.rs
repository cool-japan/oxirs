//! Bulk (one-shot) compression/decompression surface compatible with `zstd::bulk`.
//!
//! All functions are backed by [`oxiarc_zstd`] and produce/consume STANDARD
//! Zstandard frames. Every `oxiarc_zstd` error is mapped through
//! [`std::io::Error::other`].

use std::marker::PhantomData;

/// Reject a frame whose declared `Frame_Content_Size` exceeds `capacity`
/// *before* any decompression work happens.
///
/// This closes the most common decompression-bomb vector: a small,
/// maliciously-crafted frame that declares a huge decompressed size is
/// rejected purely from its header, without ever allocating the full output
/// buffer. Frames that omit the content-size field (`Ok(None)`) or that fail
/// to parse (`Err`, e.g. truncated/non-standard headers) are passed through
/// unchanged so the underlying decoder can produce the authoritative error;
/// this check is a fast-path guard, not a full frame validator.
fn reject_oversized_frame(data: &[u8], capacity: usize) -> std::io::Result<()> {
    if let Ok(Some(declared)) = crate::zstd_safe::get_frame_content_size(data) {
        let capacity_u64 = capacity as u64;
        if declared > capacity_u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "declared decompressed size {declared} exceeds capacity bound {capacity} bytes; refusing to decode (decompression-bomb guard)"
                ),
            ));
        }
    }
    Ok(())
}

/// Compress `source` into the caller-provided `destination` slice.
///
/// Returns the number of bytes written. Errors with
/// [`std::io::ErrorKind::WriteZero`] if `destination` is too small to hold the
/// compressed frame. Non-positive `level` values fall back to
/// [`crate::DEFAULT_COMPRESSION_LEVEL`].
pub fn compress_to_buffer(source: &[u8], destination: &mut [u8], level: i32) -> std::io::Result<usize> {
    let effective = if level <= 0 { crate::DEFAULT_COMPRESSION_LEVEL } else { level };
    let out = oxiarc_zstd::compress_with_level(source, effective).map_err(std::io::Error::other)?;
    if out.len() > destination.len() {
        return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "destination buffer too small"));
    }
    destination[..out.len()].copy_from_slice(&out);
    Ok(out.len())
}

/// Decompress `source` into the caller-provided `destination` slice.
///
/// Returns the number of bytes written. Errors with
/// [`std::io::ErrorKind::WriteZero`] if `destination` is too small to hold the
/// decompressed payload, and with [`std::io::ErrorKind::InvalidData`] up
/// front (before any decompression work) if the frame's declared decompressed
/// size already exceeds `destination.len()` (decompression-bomb guard).
pub fn decompress_to_buffer(source: &[u8], destination: &mut [u8]) -> std::io::Result<usize> {
    reject_oversized_frame(source, destination.len())?;
    let out = oxiarc_zstd::decode_all(source).map_err(std::io::Error::other)?;
    if out.len() > destination.len() {
        return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "destination buffer too small"));
    }
    destination[..out.len()].copy_from_slice(&out);
    Ok(out.len())
}

/// Compress `data` into a freshly allocated buffer.
///
/// Non-positive `level` values fall back to [`crate::DEFAULT_COMPRESSION_LEVEL`].
pub fn compress(data: &[u8], level: i32) -> std::io::Result<Vec<u8>> {
    let effective = if level <= 0 { crate::DEFAULT_COMPRESSION_LEVEL } else { level };
    oxiarc_zstd::compress_with_level(data, effective).map_err(std::io::Error::other)
}

/// Decompress `data` into a freshly allocated buffer.
///
/// `capacity` is the caller's declared upper bound on the decompressed size.
/// If the frame's declared `Frame_Content_Size` exceeds `capacity`, this
/// returns [`std::io::ErrorKind::InvalidData`] immediately, before any
/// decompression work happens (decompression-bomb guard). Frames that omit
/// the content-size field are decoded without a header-derived bound.
pub fn decompress(data: &[u8], capacity: usize) -> std::io::Result<Vec<u8>> {
    reject_oversized_frame(data, capacity)?;
    oxiarc_zstd::decode_all(data).map_err(std::io::Error::other)
}

/// Reusable bulk compressor compatible with `zstd::bulk::Compressor`.
pub struct Compressor<'a> {
    level: i32,
    _marker: PhantomData<&'a ()>,
}

impl Compressor<'static> {
    /// Create a compressor with the given compression `level`.
    pub fn new(level: i32) -> std::io::Result<Self> {
        Ok(Self { level, _marker: PhantomData })
    }
}

impl<'a> Compressor<'a> {
    fn effective_level(&self) -> i32 {
        if self.level <= 0 { crate::DEFAULT_COMPRESSION_LEVEL } else { self.level }
    }

    /// Compress `source` and APPEND the resulting frame to `destination`.
    ///
    /// Returns the number of bytes appended.
    pub fn compress_to_buffer(&mut self, source: &[u8], destination: &mut Vec<u8>) -> std::io::Result<usize> {
        let out = oxiarc_zstd::compress_with_level(source, self.effective_level()).map_err(std::io::Error::other)?;
        destination.extend_from_slice(&out);
        Ok(out.len())
    }

    /// Compress `data` into a freshly allocated buffer.
    pub fn compress(&mut self, data: &[u8]) -> std::io::Result<Vec<u8>> {
        oxiarc_zstd::compress_with_level(data, self.effective_level()).map_err(std::io::Error::other)
    }
}

/// Reusable bulk decompressor compatible with `zstd::bulk::Decompressor`.
pub struct Decompressor<'a> {
    _marker: PhantomData<&'a ()>,
}

impl Decompressor<'static> {
    /// Create a decompressor.
    pub fn new() -> std::io::Result<Self> {
        Ok(Self { _marker: PhantomData })
    }
}

impl Default for Decompressor<'static> {
    fn default() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<'a> Decompressor<'a> {
    /// Decompress `source` and APPEND the resulting payload to `destination`.
    ///
    /// Returns the number of bytes appended. The spare capacity already
    /// reserved in `destination` (`destination.capacity() - destination.len()`)
    /// is treated as the caller's bound: if the frame's declared
    /// `Frame_Content_Size` exceeds it, this returns
    /// [`std::io::ErrorKind::InvalidData`] immediately, before any
    /// decompression work happens (decompression-bomb guard). Callers that
    /// want to accept arbitrarily large output should `reserve` a generous
    /// bound up front, mirroring the real `zstd::bulk::Decompressor` contract.
    pub fn decompress_to_buffer(&mut self, source: &[u8], destination: &mut Vec<u8>) -> std::io::Result<usize> {
        let spare_capacity = destination.capacity().saturating_sub(destination.len());
        reject_oversized_frame(source, spare_capacity)?;
        let out = oxiarc_zstd::decode_all(source).map_err(std::io::Error::other)?;
        destination.extend_from_slice(&out);
        Ok(out.len())
    }

    /// Decompress `data` into a freshly allocated buffer.
    ///
    /// `capacity` is the caller's declared upper bound on the decompressed
    /// size. If the frame's declared `Frame_Content_Size` exceeds `capacity`,
    /// this returns [`std::io::ErrorKind::InvalidData`] immediately, before
    /// any decompression work happens (decompression-bomb guard).
    pub fn decompress(&mut self, data: &[u8], capacity: usize) -> std::io::Result<Vec<u8>> {
        reject_oversized_frame(data, capacity)?;
        oxiarc_zstd::decode_all(data).map_err(std::io::Error::other)
    }

    /// Return the decompressed size declared in the frame header, if available.
    pub fn upper_bound(data: &[u8]) -> Option<usize> {
        crate::zstd_safe::get_frame_content_size(data).ok().flatten().and_then(|n| usize::try_from(n).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_for(len: usize) -> Vec<u8> {
        let payload = (0..len).map(|i| (i % 251) as u8).collect::<Vec<u8>>();
        oxiarc_zstd::compress_with_level(&payload, 3).expect("compress")
    }

    #[test]
    fn regression_decompress_rejects_undersized_capacity_before_decoding() {
        // A frame that honestly declares a 10_000-byte payload must be
        // rejected up front when the caller only bounds it to 10 bytes,
        // instead of first materializing the full 10_000-byte buffer.
        let frame = frame_for(10_000);
        let err = decompress(&frame, 10).expect_err("must reject oversized declared size");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("decompression-bomb"), "unexpected message: {err}");
    }

    #[test]
    fn regression_decompress_allows_exact_capacity() {
        let payload_len = 4096;
        let frame = frame_for(payload_len);
        let out = decompress(&frame, payload_len).expect("must allow exact capacity");
        assert_eq!(out.len(), payload_len);
    }

    #[test]
    fn regression_decompress_to_buffer_slice_rejects_before_decoding() {
        let frame = frame_for(50_000);
        let mut dst = vec![0u8; 4];
        let err = decompress_to_buffer(&frame, &mut dst).expect_err("must reject oversized declared size");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn regression_decompressor_decompress_rejects_before_decoding() {
        let frame = frame_for(1_000_000);
        let mut dec = Decompressor::new().expect("new");
        let err = dec.decompress(&frame, 64).expect_err("must reject oversized declared size");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn regression_decompressor_decompress_to_buffer_rejects_when_under_reserved() {
        let frame = frame_for(200_000);
        let mut dec = Decompressor::new().expect("new");
        let mut dst = Vec::with_capacity(16);
        let err = dec
            .decompress_to_buffer(&frame, &mut dst)
            .expect_err("must reject oversized declared size before decoding");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        // Rejected before any bytes were appended.
        assert!(dst.is_empty());
    }

    #[test]
    fn regression_decompressor_decompress_to_buffer_succeeds_when_reserved() {
        let payload_len = 8192;
        let frame = frame_for(payload_len);
        let mut dec = Decompressor::new().expect("new");
        let mut dst = Vec::with_capacity(payload_len);
        let n = dec.decompress_to_buffer(&frame, &mut dst).expect("must succeed with sufficient reservation");
        assert_eq!(n, payload_len);
        assert_eq!(dst.len(), payload_len);
    }

    #[test]
    fn regression_unknown_content_size_frame_not_rejected_by_header_check() {
        // A malformed / non-standard header (fails to parse as a frame) must
        // not be preemptively rejected by the header-only guard; the
        // authoritative error comes from the underlying decoder instead.
        let bogus = vec![0u8; 3];
        assert!(reject_oversized_frame(&bogus, 0).is_ok());
        // The subsequent real decode still fails, just not via the guard.
        assert!(decompress(&bogus, 0).is_err());
    }
}
