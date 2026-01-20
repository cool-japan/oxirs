//! Zstandard (Zstd) compression algorithm implementation
//!
//! Zstandard is a modern compression algorithm that provides:
//! - High compression ratios (often better than gzip)
//! - Fast compression and decompression speeds
//! - Tunable compression levels (1-22)
//! - Dictionary support for small data compression
//!
//! # Performance Characteristics
//!
//! - Compression speed: 200-500 MB/s (level 3)
//! - Decompression speed: 500-1500 MB/s
//! - Compression ratio: 2.5-4x on typical data
//!
//! # Use Cases
//!
//! - Archival storage with high compression ratios
//! - Network transmission where bandwidth is limited
//! - General-purpose compression for RDF data

use super::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
    MAX_COMPRESSION_INPUT_SIZE,
};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::time::Instant;

/// Zstandard compression algorithm
///
/// Uses the zstd crate for high-performance compression with excellent ratios.
#[derive(Debug, Clone)]
pub struct ZstdCompressor {
    /// Compression level (1-22, higher = better compression but slower)
    level: i32,
}

impl ZstdCompressor {
    /// Create a new Zstd compressor with default compression level (3)
    pub fn new() -> Self {
        Self { level: 3 }
    }

    /// Create a new Zstd compressor with specified compression level
    ///
    /// # Arguments
    ///
    /// * `level` - Compression level (1-22). Recommended ranges:
    ///   - 1-3: Fast compression, lower ratios
    ///   - 4-9: Balanced compression (recommended for most use cases)
    ///   - 10-19: High compression, slower
    ///   - 20-22: Maximum compression, very slow
    pub fn with_level(level: i32) -> Self {
        Self {
            level: level.clamp(1, 22),
        }
    }

    /// Create a fast compression instance (level 1)
    pub fn fast() -> Self {
        Self::with_level(1)
    }

    /// Create a balanced compression instance (level 3)
    pub fn balanced() -> Self {
        Self::with_level(3)
    }

    /// Create a high compression instance (level 9)
    pub fn high() -> Self {
        Self::with_level(9)
    }

    /// Create a maximum compression instance (level 19)
    pub fn max() -> Self {
        Self::with_level(19)
    }

    /// Create an ultra compression instance (level 22)
    pub fn ultra() -> Self {
        Self::with_level(22)
    }
}

impl Default for ZstdCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionAlgorithm for ZstdCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        if data.is_empty() {
            return Ok(CompressedData {
                data: Vec::new(),
                metadata: CompressionMetadata {
                    algorithm: AdvancedCompressionType::Adaptive, // Will add Zstd type
                    original_size: 0,
                    compressed_size: 0,
                    compression_time_us: 0,
                    metadata: HashMap::new(),
                },
            });
        }

        if data.len() > MAX_COMPRESSION_INPUT_SIZE {
            bail!(
                "Input data too large: {} bytes (max: {})",
                data.len(),
                MAX_COMPRESSION_INPUT_SIZE
            );
        }

        let start = Instant::now();

        // Compress using zstd
        let mut encoder = zstd::Encoder::new(Vec::new(), self.level)?;
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;

        let compression_time = start.elapsed();

        let mut metadata_map = HashMap::new();
        metadata_map.insert("level".to_string(), self.level.to_string());
        metadata_map.insert(
            "compression_ratio".to_string(),
            format!("{:.2}", compressed.len() as f64 / data.len() as f64),
        );

        let compressed_size = compressed.len() as u64;

        Ok(CompressedData {
            data: compressed,
            metadata: CompressionMetadata {
                algorithm: AdvancedCompressionType::Adaptive, // Will add Zstd type
                original_size: data.len() as u64,
                compressed_size,
                compression_time_us: compression_time.as_micros() as u64,
                metadata: metadata_map,
            },
        })
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        if compressed.data.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();

        // Decompress using zstd
        let mut decoder = zstd::Decoder::new(&compressed.data[..])?;
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;

        let decompression_time = start.elapsed();

        // Verify decompressed size matches metadata
        if decompressed.len() != compressed.metadata.original_size as usize {
            bail!(
                "Decompressed size mismatch: expected {}, got {}",
                compressed.metadata.original_size,
                decompressed.len()
            );
        }

        log::trace!(
            "Zstd decompression: {} bytes -> {} bytes in {:?}",
            compressed.data.len(),
            decompressed.len(),
            decompression_time
        );

        Ok(decompressed)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::Adaptive // Will add Zstd type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_compress_empty() {
        let compressor = ZstdCompressor::new();
        let result = compressor.compress(&[]).unwrap();
        assert_eq!(result.data.len(), 0);
        assert_eq!(result.metadata.original_size, 0);
        assert_eq!(result.metadata.compressed_size, 0);
    }

    #[test]
    fn test_zstd_compress_decompress() {
        let compressor = ZstdCompressor::new();
        let original = b"Hello, World! This is a test of Zstandard compression. ".repeat(100);

        let compressed = compressor.compress(&original).unwrap();
        assert!(compressed.data.len() < original.len());
        assert_eq!(compressed.metadata.original_size, original.len() as u64);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_zstd_compression_levels() {
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(50);

        let fast = ZstdCompressor::fast();
        let balanced = ZstdCompressor::balanced();
        let high = ZstdCompressor::high();
        let max = ZstdCompressor::max();

        let fast_result = fast.compress(&data).unwrap();
        let balanced_result = balanced.compress(&data).unwrap();
        let high_result = high.compress(&data).unwrap();
        let max_result = max.compress(&data).unwrap();

        // All should decompress to the same data
        assert_eq!(fast.decompress(&fast_result).unwrap(), data);
        assert_eq!(balanced.decompress(&balanced_result).unwrap(), data);
        assert_eq!(high.decompress(&high_result).unwrap(), data);
        assert_eq!(max.decompress(&max_result).unwrap(), data);

        // Higher compression levels should produce smaller or equal output
        println!("Fast (1): {} bytes", fast_result.data.len());
        println!("Balanced (3): {} bytes", balanced_result.data.len());
        println!("High (9): {} bytes", high_result.data.len());
        println!("Max (19): {} bytes", max_result.data.len());

        // Verify compression improves with higher levels
        assert!(balanced_result.data.len() <= fast_result.data.len());
        assert!(high_result.data.len() <= balanced_result.data.len());
        assert!(max_result.data.len() <= high_result.data.len());
    }

    #[test]
    fn test_zstd_highly_compressible_data() {
        let compressor = ZstdCompressor::new();
        let data = vec![b'A'; 10000]; // Highly repetitive data

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        // Should achieve excellent compression on repetitive data
        assert!(compressed.data.len() < data.len() / 100);
        println!(
            "Zstd highly compressible: {} -> {} bytes ({:.2}x)",
            data.len(),
            compressed.data.len(),
            data.len() as f64 / compressed.data.len() as f64
        );
    }

    #[test]
    fn test_zstd_incompressible_data() {
        let compressor = ZstdCompressor::new();
        // Use scirs2_core::random for random data generation
        use scirs2_core::random::{rng, Random, Rng};

        let mut rng = rng();
        let data: Vec<u8> = (0..1000).map(|_| rng.random_range(0..256) as u8).collect();

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        // Random data should not compress well, might even expand slightly
        println!(
            "Zstd incompressible: {} -> {} bytes",
            data.len(),
            compressed.data.len()
        );
    }

    #[test]
    fn test_zstd_too_large_input() {
        let compressor = ZstdCompressor::new();
        let data = vec![0u8; MAX_COMPRESSION_INPUT_SIZE + 1];

        let result = compressor.compress(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_zstd_metadata() {
        let compressor = ZstdCompressor::with_level(9);
        let data = b"Testing metadata collection with Zstandard".repeat(10);

        let compressed = compressor.compress(&data).unwrap();

        assert_eq!(compressed.metadata.metadata.get("level").unwrap(), "9");
        assert!(compressed
            .metadata
            .metadata
            .contains_key("compression_ratio"));
        assert!(compressed.metadata.compression_time_us > 0);
    }

    #[test]
    fn test_zstd_rdf_data_compression() {
        let compressor = ZstdCompressor::balanced();
        // Simulate RDF triple data with typical patterns
        let rdf_data = b"<http://example.org/resource/1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Class> .\n"
            .repeat(100);

        let compressed = compressor.compress(&rdf_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, rdf_data);

        // RDF data with repetitive URIs should compress very well
        let ratio = rdf_data.len() as f64 / compressed.data.len() as f64;
        println!("RDF compression ratio: {:.2}x", ratio);
        assert!(ratio > 3.0); // Should achieve >3x compression
    }

    #[test]
    fn test_zstd_level_clamping() {
        let too_low = ZstdCompressor::with_level(0);
        let too_high = ZstdCompressor::with_level(100);

        assert_eq!(too_low.level, 1);
        assert_eq!(too_high.level, 22);
    }
}
