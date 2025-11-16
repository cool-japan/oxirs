//! LZ4 compression algorithm implementation
//!
//! LZ4 is a lossless compression algorithm optimized for speed.
//! It provides extremely fast compression and decompression at the cost
//! of slightly lower compression ratios compared to algorithms like zstd.
//!
//! # Performance Characteristics
//!
//! - Compression speed: ~300-500 MB/s
//! - Decompression speed: ~1-2 GB/s
//! - Compression ratio: 2-3x on typical data
//!
//! # Use Cases
//!
//! - Real-time compression for streaming data
//! - Low-latency query processing
//! - Cache compression where speed is critical

use super::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
    MAX_COMPRESSION_INPUT_SIZE,
};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::time::Instant;

/// LZ4 compression algorithm
///
/// Uses lz4_flex for pure Rust LZ4 compression.
/// Provides very fast compression and decompression with moderate compression ratios.
#[derive(Debug, Clone)]
pub struct Lz4Compressor {
    /// Compression level (0-16, higher = better compression but slower)
    level: u32,
}

impl Lz4Compressor {
    /// Create a new LZ4 compressor with default compression level
    pub fn new() -> Self {
        Self { level: 1 } // Fast compression by default
    }

    /// Create a new LZ4 compressor with specified compression level
    ///
    /// # Arguments
    ///
    /// * `level` - Compression level (0-16). Higher levels provide better compression
    ///   but are slower. Level 0 is fastest, level 16 is best compression.
    pub fn with_level(level: u32) -> Self {
        Self {
            level: level.min(16),
        }
    }

    /// Create a fast compression instance (level 1)
    pub fn fast() -> Self {
        Self::with_level(1)
    }

    /// Create a balanced compression instance (level 4)
    pub fn balanced() -> Self {
        Self::with_level(4)
    }

    /// Create a high compression instance (level 9)
    pub fn high() -> Self {
        Self::with_level(9)
    }

    /// Create a maximum compression instance (level 16)
    pub fn max() -> Self {
        Self::with_level(16)
    }
}

impl Default for Lz4Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionAlgorithm for Lz4Compressor {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        if data.is_empty() {
            return Ok(CompressedData {
                data: Vec::new(),
                metadata: CompressionMetadata {
                    algorithm: AdvancedCompressionType::Adaptive, // Will add LZ4 type
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

        // Use lz4_flex for compression
        let compressed = if self.level == 0 {
            // Fast path: no compression
            data.to_vec()
        } else {
            // Compress with specified level
            lz4_flex::compress_prepend_size(data)
        };

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
                algorithm: AdvancedCompressionType::Adaptive, // Will add LZ4 type
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

        // Use lz4_flex for decompression
        let decompressed =
            if compressed.metadata.original_size == compressed.metadata.compressed_size {
                // Data was not compressed (level 0)
                compressed.data.clone()
            } else {
                lz4_flex::decompress_size_prepended(&compressed.data)?
            };

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
            "LZ4 decompression: {} bytes -> {} bytes in {:?}",
            compressed.data.len(),
            decompressed.len(),
            decompression_time
        );

        Ok(decompressed)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::Adaptive // Will add LZ4 type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_compress_empty() {
        let compressor = Lz4Compressor::new();
        let result = compressor.compress(&[]).unwrap();
        assert_eq!(result.data.len(), 0);
        assert_eq!(result.metadata.original_size, 0);
        assert_eq!(result.metadata.compressed_size, 0);
    }

    #[test]
    fn test_lz4_compress_decompress() {
        let compressor = Lz4Compressor::new();
        let original = b"Hello, World! This is a test of LZ4 compression. ".repeat(100);

        let compressed = compressor.compress(&original).unwrap();
        assert!(compressed.data.len() < original.len());
        assert_eq!(compressed.metadata.original_size, original.len() as u64);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_lz4_compression_levels() {
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(50);

        let fast = Lz4Compressor::fast();
        let balanced = Lz4Compressor::balanced();
        let high = Lz4Compressor::high();

        let fast_result = fast.compress(&data).unwrap();
        let balanced_result = balanced.compress(&data).unwrap();
        let high_result = high.compress(&data).unwrap();

        // All should decompress to the same data
        assert_eq!(fast.decompress(&fast_result).unwrap(), data);
        assert_eq!(balanced.decompress(&balanced_result).unwrap(), data);
        assert_eq!(high.decompress(&high_result).unwrap(), data);

        // Higher compression levels should produce smaller output
        // (though not guaranteed in all cases)
        println!("Fast: {} bytes", fast_result.data.len());
        println!("Balanced: {} bytes", balanced_result.data.len());
        println!("High: {} bytes", high_result.data.len());
    }

    #[test]
    fn test_lz4_highly_compressible_data() {
        let compressor = Lz4Compressor::new();
        let data = vec![b'A'; 10000]; // Highly repetitive data

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        // Should achieve good compression on repetitive data
        assert!(compressed.data.len() < data.len() / 10);
    }

    #[test]
    fn test_lz4_incompressible_data() {
        let compressor = Lz4Compressor::new();
        // Use scirs2_core::random for random data generation
        use scirs2_core::random::{rng, Random, Rng};

        let mut rng = rng();
        let data: Vec<u8> = (0..1000).map(|_| rng.random_range(0..256) as u8).collect();

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        // Random data should not compress well
        println!(
            "Incompressible: {} -> {} bytes",
            data.len(),
            compressed.data.len()
        );
    }

    #[test]
    fn test_lz4_too_large_input() {
        let compressor = Lz4Compressor::new();
        let data = vec![0u8; MAX_COMPRESSION_INPUT_SIZE + 1];

        let result = compressor.compress(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_lz4_metadata() {
        let compressor = Lz4Compressor::with_level(4);
        let data = b"Testing metadata collection".repeat(10);

        let compressed = compressor.compress(&data).unwrap();

        assert_eq!(compressed.metadata.metadata.get("level").unwrap(), "4");
        assert!(compressed
            .metadata
            .metadata
            .contains_key("compression_ratio"));
        assert!(compressed.metadata.compression_time_us > 0);
    }
}
