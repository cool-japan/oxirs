//! Brotli compression algorithm implementation
//!
//! Brotli is a compression algorithm developed by Google that provides:
//! - Excellent compression ratios (often 20-25% better than gzip)
//! - Good decompression speed
//! - Tunable compression levels (0-11)
//! - Built-in dictionary support
//!
//! # Performance Characteristics
//!
//! - Compression speed: 10-100 MB/s (level dependent)
//! - Decompression speed: 200-400 MB/s
//! - Compression ratio: 3-5x on typical data
//!
//! # Use Cases
//!
//! - Web content compression (HTTP/2, HTTP/3)
//! - Archival storage requiring maximum compression
//! - Network transmission where bandwidth is expensive
//! - RDF data storage with emphasis on space efficiency

use super::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
    MAX_COMPRESSION_INPUT_SIZE,
};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::time::Instant;

/// Brotli compression algorithm
///
/// Uses the brotli crate for high-quality compression with excellent ratios.
#[derive(Debug, Clone)]
pub struct BrotliCompressor {
    /// Compression level (0-11, higher = better compression but slower)
    level: u32,
    /// Window size (10-24, higher = better compression but more memory)
    window_size: u32,
}

impl BrotliCompressor {
    /// Create a new Brotli compressor with default settings (level 6, window 22)
    pub fn new() -> Self {
        Self {
            level: 6,
            window_size: 22,
        }
    }

    /// Create a new Brotli compressor with specified compression level
    ///
    /// # Arguments
    ///
    /// * `level` - Compression level (0-11). Recommended ranges:
    ///   - 0-3: Fast compression, lower ratios
    ///   - 4-6: Balanced compression (recommended)
    ///   - 7-9: High compression, slower
    ///   - 10-11: Maximum compression, very slow
    pub fn with_level(level: u32) -> Self {
        Self {
            level: level.min(11),
            window_size: 22,
        }
    }

    /// Create a Brotli compressor with custom level and window size
    ///
    /// # Arguments
    ///
    /// * `level` - Compression level (0-11)
    /// * `window_size` - Window size (10-24). Larger windows improve compression
    ///   but require more memory during compression and decompression
    pub fn with_level_and_window(level: u32, window_size: u32) -> Self {
        Self {
            level: level.min(11),
            window_size: window_size.clamp(10, 24),
        }
    }

    /// Create a fast compression instance (level 1)
    pub fn fast() -> Self {
        Self::with_level(1)
    }

    /// Create a balanced compression instance (level 6)
    pub fn balanced() -> Self {
        Self::with_level(6)
    }

    /// Create a high compression instance (level 9)
    pub fn high() -> Self {
        Self::with_level(9)
    }

    /// Create a maximum compression instance (level 11)
    pub fn max() -> Self {
        Self::with_level(11)
    }
}

impl Default for BrotliCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionAlgorithm for BrotliCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        if data.is_empty() {
            return Ok(CompressedData {
                data: Vec::new(),
                metadata: CompressionMetadata {
                    algorithm: AdvancedCompressionType::Adaptive, // Will add Brotli type
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

        // Compress using brotli
        let mut compressed = Vec::new();
        let mut encoder = brotli::CompressorWriter::new(
            &mut compressed,
            4096, // buffer size
            self.level,
            self.window_size,
        );
        encoder.write_all(data)?;
        drop(encoder); // Ensure flush

        let compression_time = start.elapsed();

        let mut metadata_map = HashMap::new();
        metadata_map.insert("level".to_string(), self.level.to_string());
        metadata_map.insert("window_size".to_string(), self.window_size.to_string());
        metadata_map.insert(
            "compression_ratio".to_string(),
            format!("{:.2}", compressed.len() as f64 / data.len() as f64),
        );

        let compressed_size = compressed.len() as u64;

        Ok(CompressedData {
            data: compressed,
            metadata: CompressionMetadata {
                algorithm: AdvancedCompressionType::Adaptive, // Will add Brotli type
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

        // Decompress using brotli
        let mut decoder = brotli::Decompressor::new(&compressed.data[..], 4096);
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
            "Brotli decompression: {} bytes -> {} bytes in {:?}",
            compressed.data.len(),
            decompressed.len(),
            decompression_time
        );

        Ok(decompressed)
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::Adaptive // Will add Brotli type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brotli_compress_empty() {
        let compressor = BrotliCompressor::new();
        let result = compressor.compress(&[]).unwrap();
        assert_eq!(result.data.len(), 0);
        assert_eq!(result.metadata.original_size, 0);
        assert_eq!(result.metadata.compressed_size, 0);
    }

    #[test]
    fn test_brotli_compress_decompress() {
        let compressor = BrotliCompressor::new();
        let original = b"Hello, World! This is a test of Brotli compression. ".repeat(100);

        let compressed = compressor.compress(&original).unwrap();
        assert!(compressed.data.len() < original.len());
        assert_eq!(compressed.metadata.original_size, original.len() as u64);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_brotli_compression_levels() {
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(50);

        let fast = BrotliCompressor::fast();
        let balanced = BrotliCompressor::balanced();
        let high = BrotliCompressor::high();

        let fast_result = fast.compress(&data).unwrap();
        let balanced_result = balanced.compress(&data).unwrap();
        let high_result = high.compress(&data).unwrap();

        // All should decompress to the same data
        assert_eq!(fast.decompress(&fast_result).unwrap(), data);
        assert_eq!(balanced.decompress(&balanced_result).unwrap(), data);
        assert_eq!(high.decompress(&high_result).unwrap(), data);

        // Higher compression levels should produce smaller or equal output
        println!("Fast (1): {} bytes", fast_result.data.len());
        println!("Balanced (6): {} bytes", balanced_result.data.len());
        println!("High (9): {} bytes", high_result.data.len());

        // Verify compression improves with higher levels
        assert!(balanced_result.data.len() <= fast_result.data.len());
        assert!(high_result.data.len() <= balanced_result.data.len());
    }

    #[test]
    fn test_brotli_window_sizes() {
        let data = b"Testing window size effects on compression. ".repeat(100);

        let small_window = BrotliCompressor::with_level_and_window(6, 16);
        let large_window = BrotliCompressor::with_level_and_window(6, 24);

        let small_result = small_window.compress(&data).unwrap();
        let large_result = large_window.compress(&data).unwrap();

        // Both should decompress correctly
        assert_eq!(small_window.decompress(&small_result).unwrap(), data);
        assert_eq!(large_window.decompress(&large_result).unwrap(), data);

        println!("Window 16: {} bytes", small_result.data.len());
        println!("Window 24: {} bytes", large_result.data.len());
    }

    #[test]
    fn test_brotli_highly_compressible_data() {
        let compressor = BrotliCompressor::new();
        let data = vec![b'A'; 10000]; // Highly repetitive data

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        // Should achieve excellent compression on repetitive data
        assert!(compressed.data.len() < data.len() / 100);
        println!(
            "Brotli highly compressible: {} -> {} bytes ({:.2}x)",
            data.len(),
            compressed.data.len(),
            data.len() as f64 / compressed.data.len() as f64
        );
    }

    #[test]
    fn test_brotli_incompressible_data() {
        let compressor = BrotliCompressor::new();
        // Use scirs2_core::random for random data generation
        use scirs2_core::random::{rng, Random, Rng};

        let mut rng = rng();
        let data: Vec<u8> = (0..1000).map(|_| rng.random_range(0..256) as u8).collect();

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        // Random data should not compress well
        println!(
            "Brotli incompressible: {} -> {} bytes",
            data.len(),
            compressed.data.len()
        );
    }

    #[test]
    fn test_brotli_too_large_input() {
        let compressor = BrotliCompressor::new();
        let data = vec![0u8; MAX_COMPRESSION_INPUT_SIZE + 1];

        let result = compressor.compress(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_brotli_metadata() {
        let compressor = BrotliCompressor::with_level_and_window(9, 20);
        let data = b"Testing metadata collection with Brotli".repeat(10);

        let compressed = compressor.compress(&data).unwrap();

        assert_eq!(compressed.metadata.metadata.get("level").unwrap(), "9");
        assert_eq!(
            compressed.metadata.metadata.get("window_size").unwrap(),
            "20"
        );
        assert!(compressed
            .metadata
            .metadata
            .contains_key("compression_ratio"));
        assert!(compressed.metadata.compression_time_us > 0);
    }

    #[test]
    fn test_brotli_rdf_data_compression() {
        let compressor = BrotliCompressor::balanced();
        // Simulate RDF triple data with typical patterns
        let rdf_data = b"<http://example.org/resource/1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Class> .\n"
            .repeat(100);

        let compressed = compressor.compress(&rdf_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, rdf_data);

        // RDF data with repetitive URIs should compress excellently with Brotli
        let ratio = rdf_data.len() as f64 / compressed.data.len() as f64;
        println!("Brotli RDF compression ratio: {:.2}x", ratio);
        assert!(ratio > 4.0); // Should achieve >4x compression
    }

    #[test]
    fn test_brotli_small_data() {
        let compressor = BrotliCompressor::new();
        let data = b"Small";

        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        println!(
            "Brotli small data: {} -> {} bytes",
            data.len(),
            compressed.data.len()
        );
    }

    #[test]
    fn test_brotli_level_clamping() {
        let too_high = BrotliCompressor::with_level(100);
        assert_eq!(too_high.level, 11);
    }

    #[test]
    fn test_brotli_window_clamping() {
        let too_small = BrotliCompressor::with_level_and_window(6, 5);
        let too_large = BrotliCompressor::with_level_and_window(6, 30);

        assert_eq!(too_small.window_size, 10);
        assert_eq!(too_large.window_size, 24);
    }
}
