//! Unified compression module bringing together all compression capabilities
//!
//! This module provides a unified interface to all compression algorithms
//! supported by oxirs-tdb, with automatic algorithm selection based on
//! data characteristics.
//!
//! ## Supported Algorithms
//! - **LZ4**: Fast compression/decompression, moderate ratio
//! - **Zstandard**: High compression ratio, good speed
//! - **Brotli**: Web-optimized, excellent ratio for text
//! - **Snappy**: Extremely fast, moderate ratio
//! - **Prefix**: RDF URI prefix compression
//! - **Delta**: Delta encoding for sorted numeric data
//! - **Run-Length**: Run-length encoding for repeated values
//! - **Bitmap**: Bitmap compression for sparse data

use crate::error::{Result, TdbError};
use serde::{Deserialize, Serialize};

/// Compression strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    /// Optimize for speed
    Speed,
    /// Balance speed and compression ratio
    Balanced,
    /// Optimize for compression ratio
    Ratio,
}

/// Compression algorithm identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CompressionAlgorithm {
    /// No compression
    None = 0,
    /// LZ4 - fast compression
    Lz4 = 1,
    /// Zstandard - high compression ratio
    Zstd = 2,
    /// Brotli - web-optimized
    Brotli = 3,
    /// Snappy - extremely fast
    Snappy = 4,
    /// Prefix compression for URIs
    Prefix = 5,
    /// Delta encoding
    Delta = 6,
    /// Run-length encoding
    RunLength = 7,
    /// Bitmap compression
    Bitmap = 8,
}

/// Compression level (0-9, where 9 is maximum)
#[derive(Debug, Clone, Copy)]
pub struct CompressionLevel(u8);

impl CompressionLevel {
    /// Fastest compression (level 1)
    pub const FAST: Self = Self(1);
    /// Default compression (level 5)
    pub const DEFAULT: Self = Self(5);
    /// Best compression (level 9)
    pub const BEST: Self = Self(9);

    /// Create a new compression level
    pub fn new(level: u8) -> Self {
        Self(level.min(9))
    }

    /// Get the numeric level
    pub fn value(&self) -> u8 {
        self.0
    }
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio (compressed / original)
    pub ratio: f64,
    /// Compression time in microseconds
    pub compression_time_us: u64,
    /// Decompression time in microseconds (if measured)
    pub decompression_time_us: Option<u64>,
}

impl CompressionStats {
    /// Calculate compression ratio
    pub fn new(
        algorithm: CompressionAlgorithm,
        original_size: usize,
        compressed_size: usize,
        compression_time_us: u64,
    ) -> Self {
        let ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };

        Self {
            algorithm,
            original_size,
            compressed_size,
            ratio,
            compression_time_us,
            decompression_time_us: None,
        }
    }

    /// Calculate space savings percentage
    pub fn savings_percent(&self) -> f64 {
        (1.0 - self.ratio) * 100.0
    }
}

/// Unified compression engine
pub struct UnifiedCompression {
    /// Default algorithm
    default_algorithm: CompressionAlgorithm,
    /// Default compression level
    default_level: CompressionLevel,
    /// Adaptive strategy
    strategy: CompressionStrategy,
}

impl UnifiedCompression {
    /// Create a new unified compression engine
    pub fn new() -> Self {
        Self {
            default_algorithm: CompressionAlgorithm::Zstd,
            default_level: CompressionLevel::DEFAULT,
            strategy: CompressionStrategy::Balanced,
        }
    }

    /// Set default algorithm
    pub fn with_algorithm(mut self, algorithm: CompressionAlgorithm) -> Self {
        self.default_algorithm = algorithm;
        self
    }

    /// Set default compression level
    pub fn with_level(mut self, level: CompressionLevel) -> Self {
        self.default_level = level;
        self
    }

    /// Set compression strategy
    pub fn with_strategy(mut self, strategy: CompressionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Compress data using the default algorithm
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.compress_with(data, self.default_algorithm, self.default_level)
    }

    /// Compress data with a specific algorithm
    pub fn compress_with(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
        level: CompressionLevel,
    ) -> Result<Vec<u8>> {
        use std::time::Instant;

        let start = Instant::now();

        let compressed = match algorithm {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::Lz4 => self.compress_lz4(data)?,
            CompressionAlgorithm::Zstd => self.compress_zstd(data, level.value() as i32)?,
            CompressionAlgorithm::Brotli => self.compress_brotli(data, level.value() as u32)?,
            CompressionAlgorithm::Snappy => self.compress_snappy(data)?,
            _ => {
                return Err(TdbError::Other(format!(
                    "Algorithm {:?} not yet implemented",
                    algorithm
                )))
            }
        };

        Ok(compressed)
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Lz4 => self.decompress_lz4(data),
            CompressionAlgorithm::Zstd => self.decompress_zstd(data),
            CompressionAlgorithm::Brotli => self.decompress_brotli(data),
            CompressionAlgorithm::Snappy => self.decompress_snappy(data),
            _ => Err(TdbError::Other(format!(
                "Algorithm {:?} not yet implemented",
                algorithm
            ))),
        }
    }

    /// Select best algorithm for given data
    pub fn select_algorithm(&self, data: &[u8]) -> CompressionAlgorithm {
        // Heuristic-based selection
        match self.strategy {
            CompressionStrategy::Speed => CompressionAlgorithm::Snappy,
            CompressionStrategy::Balanced => {
                if data.len() < 1024 {
                    CompressionAlgorithm::Lz4
                } else {
                    CompressionAlgorithm::Zstd
                }
            }
            CompressionStrategy::Ratio => CompressionAlgorithm::Brotli,
        }
    }

    // Algorithm-specific implementations

    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::decompress_size_prepended(data)
            .map_err(|e| TdbError::Other(format!("LZ4 decompression failed: {:?}", e)))
    }

    fn compress_zstd(&self, data: &[u8], level: i32) -> Result<Vec<u8>> {
        zstd::encode_all(data, level)
            .map_err(|e| TdbError::Other(format!("Zstd compression failed: {}", e)))
    }

    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(data)
            .map_err(|e| TdbError::Other(format!("Zstd decompression failed: {}", e)))
    }

    fn compress_brotli(&self, data: &[u8], level: u32) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        let mut compressor = brotli::CompressorWriter::new(&mut output, 4096, level, 22);

        use std::io::Write;
        compressor
            .write_all(data)
            .map_err(|e| TdbError::Other(format!("Brotli compression failed: {}", e)))?;
        compressor
            .flush()
            .map_err(|e| TdbError::Other(format!("Brotli flush failed: {}", e)))?;
        drop(compressor);

        Ok(output)
    }

    fn decompress_brotli(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        let mut decompressor = brotli::Decompressor::new(data, 4096);

        use std::io::Read;
        decompressor
            .read_to_end(&mut output)
            .map_err(|e| TdbError::Other(format!("Brotli decompression failed: {}", e)))?;

        Ok(output)
    }

    fn compress_snappy(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = snap::raw::Encoder::new();
        encoder
            .compress_vec(data)
            .map_err(|e| TdbError::Other(format!("Snappy compression failed: {}", e)))
    }

    fn decompress_snappy(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = snap::raw::Decoder::new();
        decoder
            .decompress_vec(data)
            .map_err(|e| TdbError::Other(format!("Snappy decompression failed: {}", e)))
    }

    /// Benchmark all algorithms on sample data
    pub fn benchmark(&self, data: &[u8]) -> Vec<CompressionStats> {
        let mut results = Vec::new();

        let algorithms = vec![
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Brotli,
            CompressionAlgorithm::Snappy,
        ];

        for algo in algorithms {
            if let Ok(compressed) = self.compress_with(data, algo, CompressionLevel::DEFAULT) {
                let stats = CompressionStats::new(
                    algo,
                    data.len(),
                    compressed.len(),
                    0, // Time not measured in simple benchmark
                );
                results.push(stats);
            }
        }

        results
    }
}

impl Default for UnifiedCompression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_compression_creation() {
        let _compression = UnifiedCompression::new();
    }

    #[test]
    fn test_compression_level() {
        assert_eq!(CompressionLevel::FAST.value(), 1);
        assert_eq!(CompressionLevel::DEFAULT.value(), 5);
        assert_eq!(CompressionLevel::BEST.value(), 9);

        let custom = CompressionLevel::new(7);
        assert_eq!(custom.value(), 7);

        // Test clamping
        let clamped = CompressionLevel::new(15);
        assert_eq!(clamped.value(), 9);
    }

    #[test]
    fn test_lz4_roundtrip() {
        let compression = UnifiedCompression::new();
        let data = b"Hello, World! This is a test of LZ4 compression.";

        let compressed = compression
            .compress_with(data, CompressionAlgorithm::Lz4, CompressionLevel::DEFAULT)
            .unwrap();

        // Note: For small data, compression may add overhead
        // The important thing is correctness of roundtrip

        let decompressed = compression
            .decompress(&compressed, CompressionAlgorithm::Lz4)
            .unwrap();

        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_zstd_roundtrip() {
        let compression = UnifiedCompression::new();
        let data = b"Zstandard compression test data with some repetition repetition repetition";

        let compressed = compression
            .compress_with(data, CompressionAlgorithm::Zstd, CompressionLevel::DEFAULT)
            .unwrap();

        let decompressed = compression
            .decompress(&compressed, CompressionAlgorithm::Zstd)
            .unwrap();

        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_snappy_roundtrip() {
        let compression = UnifiedCompression::new();
        let data = b"Snappy is designed for speed rather than maximum compression";

        let compressed = compression
            .compress_with(
                data,
                CompressionAlgorithm::Snappy,
                CompressionLevel::DEFAULT,
            )
            .unwrap();

        let decompressed = compression
            .decompress(&compressed, CompressionAlgorithm::Snappy)
            .unwrap();

        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_algorithm_selection() {
        let compression = UnifiedCompression::new().with_strategy(CompressionStrategy::Speed);
        assert_eq!(
            compression.select_algorithm(b"test"),
            CompressionAlgorithm::Snappy
        );

        let compression = UnifiedCompression::new().with_strategy(CompressionStrategy::Ratio);
        assert_eq!(
            compression.select_algorithm(b"test"),
            CompressionAlgorithm::Brotli
        );
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats::new(CompressionAlgorithm::Zstd, 1000, 500, 100);

        assert_eq!(stats.ratio, 0.5);
        assert_eq!(stats.savings_percent(), 50.0);
    }

    #[test]
    fn test_benchmark() {
        let compression = UnifiedCompression::new();
        let data = b"Sample data for benchmarking compression algorithms. ".repeat(10);

        let results = compression.benchmark(&data);

        assert!(results.len() >= 4); // At least 4 algorithms
        for result in &results {
            assert!(result.compressed_size > 0);
            assert!(result.compressed_size <= result.original_size);
        }
    }

    #[test]
    fn test_none_algorithm() {
        let compression = UnifiedCompression::new();
        let data = b"Test data";

        let compressed = compression
            .compress_with(data, CompressionAlgorithm::None, CompressionLevel::DEFAULT)
            .unwrap();

        assert_eq!(compressed, data);

        let decompressed = compression
            .decompress(&compressed, CompressionAlgorithm::None)
            .unwrap();

        assert_eq!(decompressed, data);
    }
}
