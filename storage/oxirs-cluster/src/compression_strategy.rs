//! Compression strategy for cluster data with auto-selection
//!
//! This module provides intelligent compression strategies that automatically
//! select the best compression algorithm based on data characteristics,
//! access patterns, and system load.

use crate::error::{ClusterError, Result};
use scirs2_core::profiling::Profiler;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Compression strategy manager with auto-selection capabilities
pub struct CompressionStrategy {
    config: CompressionConfig,
    #[allow(dead_code)]
    profiler: Arc<Profiler>,
    metrics: Arc<CompressionMetrics>,
}

/// Configuration for compression behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Default compression algorithm to use
    pub default_algorithm: Algorithm,
    /// Enable automatic algorithm selection based on data characteristics
    pub auto_select: bool,
    /// Minimum data size for compression (smaller data won't be compressed)
    pub compression_threshold_bytes: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            default_algorithm: Algorithm::Zstd,
            auto_select: true,
            compression_threshold_bytes: 1024, // 1KB minimum
        }
    }
}

/// Available compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    /// No compression
    None,
    /// LZ4: Fast compression (~500 MB/s), ~60% compression ratio
    Lz4,
    /// Zstd: Balanced compression (~200 MB/s), ~70% compression ratio
    Zstd,
    /// LZMA: High compression (~50 MB/s), ~80% compression ratio
    Lzma,
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::None => write!(f, "none"),
            Algorithm::Lz4 => write!(f, "lz4"),
            Algorithm::Zstd => write!(f, "zstd"),
            Algorithm::Lzma => write!(f, "lzma"),
        }
    }
}

/// Data access patterns for algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Frequently accessed data - prioritize decompression speed
    Hot,
    /// Occasionally accessed data - balance speed and compression
    Warm,
    /// Rarely accessed data - prioritize compression ratio
    Cold,
}

/// Compressed data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedData {
    /// Compressed data bytes
    pub data: Vec<u8>,
    /// Original uncompressed size
    pub original_size: usize,
    /// Algorithm used for compression
    pub algorithm: Algorithm,
}

/// Compression performance metrics
pub struct CompressionMetrics {
    total_bytes_in: AtomicU64,
    total_bytes_out: AtomicU64,
    compression_time_ns: AtomicU64,
    decompression_time_ns: AtomicU64,
    compression_count: AtomicU64,
    decompression_count: AtomicU64,
}

impl Default for CompressionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            total_bytes_in: AtomicU64::new(0),
            total_bytes_out: AtomicU64::new(0),
            compression_time_ns: AtomicU64::new(0),
            decompression_time_ns: AtomicU64::new(0),
            compression_count: AtomicU64::new(0),
            decompression_count: AtomicU64::new(0),
        }
    }

    /// Record compression operation
    pub fn record_compression(
        &self,
        input_size: usize,
        output_size: usize,
        _algorithm: Algorithm,
        elapsed_ns: u64,
    ) {
        self.total_bytes_in
            .fetch_add(input_size as u64, Ordering::Relaxed);
        self.total_bytes_out
            .fetch_add(output_size as u64, Ordering::Relaxed);
        self.compression_time_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);
        self.compression_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record decompression operation
    pub fn record_decompression(&self, elapsed_ns: u64) {
        self.decompression_time_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);
        self.decompression_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        let bytes_in = self.total_bytes_in.load(Ordering::Relaxed) as f64;
        let bytes_out = self.total_bytes_out.load(Ordering::Relaxed) as f64;
        if bytes_in > 0.0 {
            1.0 - (bytes_out / bytes_in)
        } else {
            0.0
        }
    }

    /// Get average compression throughput in MB/s
    pub fn compression_throughput_mbps(&self) -> f64 {
        let bytes = self.total_bytes_in.load(Ordering::Relaxed) as f64;
        let time_sec = self.compression_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000_000.0;
        if time_sec > 0.0 {
            (bytes / 1_000_000.0) / time_sec
        } else {
            0.0
        }
    }

    /// Get average decompression throughput in MB/s
    pub fn decompression_throughput_mbps(&self) -> f64 {
        let bytes = self.total_bytes_out.load(Ordering::Relaxed) as f64;
        let time_sec =
            self.decompression_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000_000.0;
        if time_sec > 0.0 {
            (bytes / 1_000_000.0) / time_sec
        } else {
            0.0
        }
    }
}

impl CompressionStrategy {
    /// Create a new compression strategy
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            profiler: Arc::new(Profiler::new()),
            metrics: Arc::new(CompressionMetrics::new()),
        }
    }

    /// Auto-select compression algorithm based on data characteristics
    pub fn select_algorithm(&self, data: &[u8], access_pattern: AccessPattern) -> Algorithm {
        if !self.config.auto_select {
            return self.config.default_algorithm;
        }

        // Criteria 1: Data size - Small data won't benefit from compression
        if data.len() < self.config.compression_threshold_bytes {
            return Algorithm::None;
        }

        // Criteria 2: Compressibility - Random data won't compress well
        let compressibility = self.estimate_compressibility(data);
        if compressibility < 0.1 {
            // Nearly random data, compression won't help
            return Algorithm::None;
        }

        // Criteria 3: Access pattern determines priority (speed vs ratio)
        match access_pattern {
            AccessPattern::Hot => Algorithm::Lz4,   // Fast decompression
            AccessPattern::Warm => Algorithm::Zstd, // Balanced
            AccessPattern::Cold => Algorithm::Lzma, // Best ratio
        }
    }

    /// Compress data with selected or auto-selected algorithm
    pub fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        self.compress_with_pattern(data, AccessPattern::Warm)
    }

    /// Compress data with specific access pattern
    pub fn compress_with_pattern(
        &self,
        data: &[u8],
        access_pattern: AccessPattern,
    ) -> Result<CompressedData> {
        let algorithm = self.select_algorithm(data, access_pattern);

        let start = std::time::Instant::now();
        let compressed = match algorithm {
            Algorithm::None => data.to_vec(),
            Algorithm::Lz4 => self.compress_lz4(data)?,
            Algorithm::Zstd => self.compress_zstd(data)?,
            Algorithm::Lzma => self.compress_lzma(data)?,
        };
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        // Update metrics
        self.metrics.record_compression(
            data.len(),
            compressed.len(),
            algorithm,
            elapsed_ns,
        );

        Ok(CompressedData {
            data: compressed,
            original_size: data.len(),
            algorithm,
        })
    }

    /// Decompress data
    pub fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        let result = match compressed.algorithm {
            Algorithm::None => Ok(compressed.data.clone()),
            Algorithm::Lz4 => self.decompress_lz4(&compressed.data, compressed.original_size),
            Algorithm::Zstd => self.decompress_zstd(&compressed.data, compressed.original_size),
            Algorithm::Lzma => self.decompress_lzma(&compressed.data),
        };

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.metrics.record_decompression(elapsed_ns);

        result
    }

    /// Estimate compressibility using Shannon entropy
    /// Returns value between 0.0 (random) and 1.0 (highly compressible)
    pub fn estimate_compressibility(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        // Count byte frequencies
        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }

        // Calculate Shannon entropy
        let len = data.len() as f64;
        let mut entropy = 0.0;
        for &count in &counts {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        // Normalize: 8 bits = random (max entropy), 0 bits = constant (min entropy)
        // Return compressibility score (inverse of normalized entropy)
        1.0 - (entropy / 8.0)
    }

    // LZ4 implementation using lz4_flex
    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        use lz4_flex::compress_prepend_size;
        Ok(compress_prepend_size(data))
    }

    fn decompress_lz4(&self, data: &[u8], _expected_size: usize) -> Result<Vec<u8>> {
        use lz4_flex::decompress_size_prepended;
        decompress_size_prepended(data)
            .map_err(|e| ClusterError::Compression(format!("LZ4 decompression failed: {e}")))
    }

    // Zstd implementation
    fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::bulk::compress(data, 3)
            .map_err(|e| ClusterError::Compression(format!("Zstd compression failed: {e}")))
    }

    fn decompress_zstd(&self, data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        // Use expected_size * 4 as buffer size hint (can grow if needed)
        zstd::bulk::decompress(data, expected_size * 4)
            .map_err(|e| ClusterError::Compression(format!("Zstd decompression failed: {e}")))
    }

    // LZMA implementation using xz2
    fn compress_lzma(&self, data: &[u8]) -> Result<Vec<u8>> {
        use std::io::Write;
        use xz2::write::XzEncoder;

        let mut encoder = XzEncoder::new(Vec::new(), 6);
        encoder
            .write_all(data)
            .map_err(|e| ClusterError::Compression(format!("LZMA compression failed: {e}")))?;
        encoder
            .finish()
            .map_err(|e| ClusterError::Compression(format!("LZMA compression failed: {e}")))
    }

    fn decompress_lzma(&self, data: &[u8]) -> Result<Vec<u8>> {
        use std::io::Read;
        use xz2::read::XzDecoder;

        let mut decoder = XzDecoder::new(data);
        let mut result = Vec::new();
        decoder
            .read_to_end(&mut result)
            .map_err(|e| ClusterError::Compression(format!("LZMA decompression failed: {e}")))?;
        Ok(result)
    }

    /// Get compression metrics
    pub fn metrics(&self) -> &Arc<CompressionMetrics> {
        &self.metrics
    }

    /// Get configuration
    pub fn config(&self) -> &CompressionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_strategy_creation() {
        let config = CompressionConfig::default();
        let strategy = CompressionStrategy::new(config);
        assert!(strategy.config().auto_select);
    }

    #[test]
    fn test_algorithm_selection_small_data() {
        let config = CompressionConfig {
            default_algorithm: Algorithm::Zstd,
            auto_select: true,
            compression_threshold_bytes: 1024,
        };
        let strategy = CompressionStrategy::new(config);

        // Small data should not be compressed
        let small_data = vec![0u8; 512];
        let algorithm = strategy.select_algorithm(&small_data, AccessPattern::Warm);
        assert_eq!(algorithm, Algorithm::None);
    }

    #[test]
    fn test_algorithm_selection_by_pattern() {
        let config = CompressionConfig::default();
        let strategy = CompressionStrategy::new(config);

        let data = vec![65u8; 10000]; // Highly compressible data

        let hot_algo = strategy.select_algorithm(&data, AccessPattern::Hot);
        assert_eq!(hot_algo, Algorithm::Lz4);

        let warm_algo = strategy.select_algorithm(&data, AccessPattern::Warm);
        assert_eq!(warm_algo, Algorithm::Zstd);

        let cold_algo = strategy.select_algorithm(&data, AccessPattern::Cold);
        assert_eq!(cold_algo, Algorithm::Lzma);
    }

    #[test]
    fn test_compressibility_estimation() {
        let config = CompressionConfig::default();
        let strategy = CompressionStrategy::new(config);

        // Highly compressible data (all same byte)
        let compressible = vec![42u8; 1000];
        let score1 = strategy.estimate_compressibility(&compressible);
        assert!(score1 > 0.9, "Expected high compressibility score");

        // Random data (less compressible)
        use scirs2_core::random::{Random, RngCore};
        let mut rng = Random::seed(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()));
        let mut random_data = vec![0u8; 1000];
        rng.fill_bytes(&mut random_data);
        let score2 = strategy.estimate_compressibility(&random_data);
        assert!(
            score2 < 0.2,
            "Expected low compressibility score for random data"
        );
    }

    #[test]
    fn test_lz4_round_trip() {
        let config = CompressionConfig {
            default_algorithm: Algorithm::Lz4,
            auto_select: false,
            compression_threshold_bytes: 0,
        };
        let strategy = CompressionStrategy::new(config);

        let original = b"Hello, world! This is a test string for LZ4 compression.".repeat(100);
        let compressed = strategy.compress(&original).expect("Compression failed");
        assert!(compressed.data.len() < original.len());
        assert_eq!(compressed.algorithm, Algorithm::Lz4);

        let decompressed = strategy
            .decompress(&compressed)
            .expect("Decompression failed");
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_zstd_round_trip() {
        let config = CompressionConfig {
            default_algorithm: Algorithm::Zstd,
            auto_select: false,
            compression_threshold_bytes: 0,
        };
        let strategy = CompressionStrategy::new(config);

        let original = b"Hello, world! This is a test string for Zstd compression.".repeat(100);
        let compressed = strategy.compress(&original).expect("Compression failed");
        assert!(compressed.data.len() < original.len());
        assert_eq!(compressed.algorithm, Algorithm::Zstd);

        let decompressed = strategy
            .decompress(&compressed)
            .expect("Decompression failed");
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_lzma_round_trip() {
        let config = CompressionConfig {
            default_algorithm: Algorithm::Lzma,
            auto_select: false,
            compression_threshold_bytes: 0,
        };
        let strategy = CompressionStrategy::new(config);

        let original = b"Hello, world! This is a test string for LZMA compression.".repeat(100);
        let compressed = strategy.compress(&original).expect("Compression failed");
        assert!(compressed.data.len() < original.len());
        assert_eq!(compressed.algorithm, Algorithm::Lzma);

        let decompressed = strategy
            .decompress(&compressed)
            .expect("Decompression failed");
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_compression_metrics() {
        let config = CompressionConfig::default();
        let strategy = CompressionStrategy::new(config);

        let original = b"Test data for metrics".repeat(100);
        let _compressed = strategy.compress(&original).expect("Compression failed");

        let metrics = strategy.metrics();
        assert!(metrics.compression_ratio() > 0.0);
        assert!(metrics.compression_throughput_mbps() > 0.0);
    }

    #[test]
    fn test_none_algorithm() {
        let config = CompressionConfig {
            default_algorithm: Algorithm::None,
            auto_select: false,
            compression_threshold_bytes: 0,
        };
        let strategy = CompressionStrategy::new(config);

        let original = b"Test data";
        let compressed = strategy.compress(original).expect("Compression failed");
        assert_eq!(compressed.data, original);
        assert_eq!(compressed.algorithm, Algorithm::None);

        let decompressed = strategy
            .decompress(&compressed)
            .expect("Decompression failed");
        assert_eq!(decompressed, original);
    }
}
