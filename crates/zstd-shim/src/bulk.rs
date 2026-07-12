//! Bulk (one-shot) compression/decompression surface compatible with `zstd::bulk`.
//!
//! All functions are backed by [`oxiarc_zstd`] and produce/consume STANDARD
//! Zstandard frames. Every `oxiarc_zstd` error is mapped through
//! [`std::io::Error::other`].

use std::marker::PhantomData;

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
/// decompressed payload.
pub fn decompress_to_buffer(source: &[u8], destination: &mut [u8]) -> std::io::Result<usize> {
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
/// The `_capacity` argument is accepted for API compatibility with `zstd::bulk`
/// but is not required by the underlying implementation.
pub fn decompress(data: &[u8], _capacity: usize) -> std::io::Result<Vec<u8>> {
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
    /// Returns the number of bytes appended.
    pub fn decompress_to_buffer(&mut self, source: &[u8], destination: &mut Vec<u8>) -> std::io::Result<usize> {
        let out = oxiarc_zstd::decode_all(source).map_err(std::io::Error::other)?;
        destination.extend_from_slice(&out);
        Ok(out.len())
    }

    /// Decompress `data` into a freshly allocated buffer.
    ///
    /// The `_capacity` argument is accepted for API compatibility but is not
    /// required by the underlying implementation.
    pub fn decompress(&mut self, data: &[u8], _capacity: usize) -> std::io::Result<Vec<u8>> {
        oxiarc_zstd::decode_all(data).map_err(std::io::Error::other)
    }

    /// Return the decompressed size declared in the frame header, if available.
    pub fn upper_bound(data: &[u8]) -> Option<usize> {
        crate::zstd_safe::get_frame_content_size(data).ok().flatten().and_then(|n| usize::try_from(n).ok())
    }
}
