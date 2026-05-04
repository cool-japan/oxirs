//! Compression codec implementations for oxirs-cluster.
//!
//! Provides a `Compressor` trait with four built-in implementations:
//! - `IdentityCodec` — no-op, always available
//! - `RleCodec`      — pure-Rust run-length encoding for repetitive data
//! - `Lz4Codec`      — fast LZ4 via oxiarc-lz4 (Pure Rust)
//! - `ZstdCodec`     — balanced Zstandard via oxiarc-zstd (Pure Rust)

use thiserror::Error;

/// Error type for all codec operations.
#[derive(Debug, Error)]
pub enum CompressionError {
    /// Compression produced an error.
    #[error("Compression failed: {0}")]
    CompressFailed(String),
    /// Decompression produced an error.
    #[error("Decompression failed: {0}")]
    DecompressFailed(String),
    /// The requested codec name is not in the registry.
    #[error("Unknown codec: {0}")]
    UnknownCodec(String),
}

/// Trait that every codec must implement.
///
/// Implementations must be `Send + Sync` so they can be stored in an `Arc`
/// and shared across threads.
pub trait Compressor: Send + Sync {
    /// Short stable identifier for this codec (e.g. `"identity"`, `"rle"`).
    fn name(&self) -> &'static str;

    /// Compress `data` and return the compressed bytes.
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;

    /// Decompress `data` and return the original bytes.
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;
}

// ---------------------------------------------------------------------------
// Identity codec
// ---------------------------------------------------------------------------

/// Identity (no-op) codec.  The safe default — data is returned unchanged.
pub struct IdentityCodec;

impl Compressor for IdentityCodec {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Ok(data.to_vec())
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        Ok(data.to_vec())
    }
}

// ---------------------------------------------------------------------------
// RLE codec
// ---------------------------------------------------------------------------

/// Simple run-length encoding codec (pure Rust, no external crate).
///
/// # Wire format
/// A sequence of `[count: u8][byte: u8]` pairs where `count ∈ [1, 255]`.
/// Runs longer than 255 bytes are split into multiple pairs.
pub struct RleCodec;

impl Compressor for RleCodec {
    fn name(&self) -> &'static str {
        "rle"
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(data.len().min(data.len() * 2 / 3 + 16));
        let mut i = 0usize;

        while i < data.len() {
            let byte = data[i];
            let mut count: u8 = 1;
            while i + (count as usize) < data.len()
                && data[i + (count as usize)] == byte
                && count < u8::MAX
            {
                count += 1;
            }
            out.push(count);
            out.push(byte);
            i += count as usize;
        }
        Ok(out)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if data.len() % 2 != 0 {
            return Err(CompressionError::DecompressFailed(
                "RLE data length must be even".into(),
            ));
        }

        let capacity: usize = data.chunks_exact(2).map(|c| c[0] as usize).sum();
        let mut out = Vec::with_capacity(capacity);

        for chunk in data.chunks_exact(2) {
            let count = chunk[0] as usize;
            let byte = chunk[1];
            out.extend(std::iter::repeat(byte).take(count));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// LZ4 codec (oxiarc-lz4 — Pure Rust)
// ---------------------------------------------------------------------------

/// LZ4 codec backed by `oxiarc-lz4` (Pure Rust, no C).
///
/// Optimised for **fast** compression and decompression at moderate ratios.
pub struct Lz4Codec;

impl Compressor for Lz4Codec {
    fn name(&self) -> &'static str {
        "lz4"
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        oxiarc_lz4::compress(data)
            .map_err(|e| CompressionError::CompressFailed(format!("LZ4: {e}")))
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        // Allow up to 256 MiB decompressed output
        oxiarc_lz4::decompress(data, 256 * 1024 * 1024)
            .map_err(|e| CompressionError::DecompressFailed(format!("LZ4: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Zstd codec (oxiarc-zstd — Pure Rust)
// ---------------------------------------------------------------------------

/// Zstandard codec backed by `oxiarc-zstd` (Pure Rust, no C).
///
/// Compression level 3 is the default (balanced speed/ratio).
pub struct ZstdCodec {
    level: i32,
}

impl ZstdCodec {
    /// Create a new Zstd codec with a specific compression level (1-21).
    pub fn new(level: i32) -> Self {
        ZstdCodec { level }
    }

    /// Create with the default level (3).
    pub fn default_level() -> Self {
        ZstdCodec { level: 3 }
    }
}

impl Compressor for ZstdCodec {
    fn name(&self) -> &'static str {
        "zstd"
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        oxiarc_zstd::encode_all(data, self.level)
            .map_err(|e| CompressionError::CompressFailed(format!("Zstd: {e}")))
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        oxiarc_zstd::decode_all(data)
            .map_err(|e| CompressionError::DecompressFailed(format!("Zstd: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- IdentityCodec ---

    #[test]
    fn identity_empty() {
        let c = IdentityCodec;
        let d = c.compress(&[]).unwrap();
        assert_eq!(c.decompress(&d).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn identity_round_trip() {
        let c = IdentityCodec;
        let data: Vec<u8> = (0u8..=255).collect();
        assert_eq!(c.decompress(&c.compress(&data).unwrap()).unwrap(), data);
    }

    // --- RleCodec ---

    #[test]
    fn rle_empty() {
        let c = RleCodec;
        assert_eq!(c.compress(&[]).unwrap(), Vec::<u8>::new());
        assert_eq!(c.decompress(&[]).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn rle_single_run() {
        let c = RleCodec;
        let data = vec![0u8; 100];
        let enc = c.compress(&data).unwrap();
        assert_eq!(enc, vec![100u8, 0u8]);
        assert_eq!(c.decompress(&enc).unwrap(), data);
    }

    #[test]
    fn rle_round_trip_all_unique() {
        let c = RleCodec;
        let data: Vec<u8> = (0u8..=127).collect();
        assert_eq!(c.decompress(&c.compress(&data).unwrap()).unwrap(), data);
    }

    #[test]
    fn rle_compresses_repetitive() {
        let c = RleCodec;
        let data = vec![42u8; 200];
        let enc = c.compress(&data).unwrap();
        // Should be encoded as two pairs: [255, 42, 42, 42] (split at 255 + 200-255 overflow)
        // i.e. [200, 42] would be the case if count < 255
        // 200 < 255 so single pair [200, 42]
        assert_eq!(enc.len(), 2);
        assert!(enc.len() < data.len());
        assert_eq!(c.decompress(&enc).unwrap(), data);
    }

    #[test]
    fn rle_decompression_odd_input_error() {
        let c = RleCodec;
        let result = c.decompress(&[1u8]);
        assert!(result.is_err());
    }

    // --- Lz4Codec ---

    #[test]
    fn lz4_empty() {
        let c = Lz4Codec;
        let enc = c.compress(&[]).unwrap();
        assert_eq!(c.decompress(&enc).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn lz4_round_trip_repetitive() {
        let c = Lz4Codec;
        let data = b"hello world ".repeat(500);
        let enc = c.compress(&data).unwrap();
        assert!(enc.len() < data.len());
        assert_eq!(c.decompress(&enc).unwrap(), data);
    }

    #[test]
    fn lz4_round_trip_diverse() {
        let c = Lz4Codec;
        let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        assert_eq!(c.decompress(&c.compress(&data).unwrap()).unwrap(), data);
    }

    // --- ZstdCodec ---

    #[test]
    fn zstd_empty() {
        let c = ZstdCodec::default_level();
        let enc = c.compress(&[]).unwrap();
        assert_eq!(c.decompress(&enc).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn zstd_round_trip_repetitive() {
        let c = ZstdCodec::new(3);
        let data = b"oxirs cluster ".repeat(500);
        let enc = c.compress(&data).unwrap();
        assert!(enc.len() < data.len());
        assert_eq!(c.decompress(&enc).unwrap(), data);
    }

    #[test]
    fn zstd_round_trip_diverse() {
        let c = ZstdCodec::default_level();
        let data: Vec<u8> = (0u8..=255).cycle().take(2048).collect();
        assert_eq!(c.decompress(&c.compress(&data).unwrap()).unwrap(), data);
    }
}
