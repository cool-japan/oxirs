//! Adaptive compression algorithm selection
//!
//! This module automatically selects the best compression algorithm based on
//! runtime performance metrics, data characteristics, and compression goals.

use super::AdvancedCompressionType;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Simplified compression type for adaptive selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression
    None,
    /// LZ4 compression (very fast)
    Lz4,
    /// Zstandard compression (excellent ratio)
    Zstd,
    /// Snappy compression (fast with good ratio)
    Snappy,
    /// Brotli compression (high compression ratio)
    Brotli,
    /// Zlib/DEFLATE compression
    Zlib,
}

impl From<CompressionType> for AdvancedCompressionType {
    fn from(ct: CompressionType) -> Self {
        match ct {
            CompressionType::None => AdvancedCompressionType::RunLength, // Fallback
            CompressionType::Lz4 => AdvancedCompressionType::Lz4,
            CompressionType::Zstd => AdvancedCompressionType::Zstd,
            CompressionType::Snappy => AdvancedCompressionType::Snappy,
            CompressionType::Brotli => AdvancedCompressionType::Brotli,
            CompressionType::Zlib => AdvancedCompressionType::RunLength, // Fallback
        }
    }
}

/// Compression selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Maximize compression ratio (smallest size)
    MaximizeRatio,
    /// Minimize compression time (fastest)
    MinimizeTime,
    /// Balance between ratio and time
    Balanced,
    /// Adaptive based on data size
    Adaptive,
}

/// Compression performance metrics for an algorithm
#[derive(Debug)]
pub struct AlgorithmMetrics {
    /// Total bytes compressed
    pub bytes_compressed: AtomicU64,
    /// Total compression time (microseconds)
    pub total_time_us: AtomicU64,
    /// Total compressed size
    pub total_compressed_size: AtomicU64,
    /// Number of compression operations
    pub operations: AtomicU64,
}

impl Default for AlgorithmMetrics {
    fn default() -> Self {
        Self {
            bytes_compressed: AtomicU64::new(0),
            total_time_us: AtomicU64::new(0),
            total_compressed_size: AtomicU64::new(0),
            operations: AtomicU64::new(0),
        }
    }
}

impl AlgorithmMetrics {
    /// Record a compression operation
    pub fn record(&self, input_size: usize, compressed_size: usize, duration: Duration) {
        self.bytes_compressed
            .fetch_add(input_size as u64, Ordering::Relaxed);
        self.total_compressed_size
            .fetch_add(compressed_size as u64, Ordering::Relaxed);
        self.total_time_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
        self.operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get average compression ratio (original_size / compressed_size)
    pub fn avg_ratio(&self) -> f64 {
        let total_in = self.bytes_compressed.load(Ordering::Relaxed) as f64;
        let total_out = self.total_compressed_size.load(Ordering::Relaxed) as f64;
        if total_out == 0.0 {
            1.0
        } else {
            total_in / total_out
        }
    }

    /// Get average compression throughput (MB/s)
    pub fn avg_throughput_mbps(&self) -> f64 {
        let bytes = self.bytes_compressed.load(Ordering::Relaxed) as f64;
        let time_s = self.total_time_us.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        if time_s == 0.0 {
            0.0
        } else {
            (bytes / (1024.0 * 1024.0)) / time_s
        }
    }

    /// Get average time per operation (microseconds)
    pub fn avg_time_us(&self) -> f64 {
        let ops = self.operations.load(Ordering::Relaxed);
        if ops == 0 {
            0.0
        } else {
            self.total_time_us.load(Ordering::Relaxed) as f64 / ops as f64
        }
    }
}

/// Adaptive compression selector
pub struct AdaptiveCompressionSelector {
    /// Selection strategy
    strategy: SelectionStrategy,
    /// Metrics for each compression algorithm
    metrics: Vec<(AdvancedCompressionType, Arc<AlgorithmMetrics>)>,
    /// Minimum data size to enable compression
    min_compression_size: usize,
}

impl AdaptiveCompressionSelector {
    /// Create a new adaptive selector
    pub fn new(strategy: SelectionStrategy) -> Self {
        let compression_types = vec![
            AdvancedCompressionType::Lz4,
            AdvancedCompressionType::Zstd,
            AdvancedCompressionType::Snappy,
            AdvancedCompressionType::Brotli,
            AdvancedCompressionType::RunLength, // Using RunLength as Zlib alternative
        ];

        let metrics = compression_types
            .into_iter()
            .map(|ctype| (ctype, Arc::new(AlgorithmMetrics::default())))
            .collect();

        Self {
            strategy,
            metrics,
            min_compression_size: 128, // Don't compress data < 128 bytes
        }
    }

    /// Select the best compression algorithm for the given data
    pub fn select(&self, data_size: usize) -> AdvancedCompressionType {
        // Don't compress small data - use RunLength as "no compression" fallback
        if data_size < self.min_compression_size {
            return AdvancedCompressionType::RunLength;
        }

        match self.strategy {
            SelectionStrategy::MaximizeRatio => self.select_max_ratio(),
            SelectionStrategy::MinimizeTime => self.select_min_time(),
            SelectionStrategy::Balanced => self.select_balanced(),
            SelectionStrategy::Adaptive => self.select_adaptive(data_size),
        }
    }

    /// Select algorithm with best compression ratio
    fn select_max_ratio(&self) -> AdvancedCompressionType {
        self.metrics
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.avg_ratio()
                    .partial_cmp(&b.avg_ratio())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(ctype, _)| *ctype)
            .unwrap_or(AdvancedCompressionType::Zstd)
    }

    /// Select algorithm with best throughput
    fn select_min_time(&self) -> AdvancedCompressionType {
        self.metrics
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.avg_throughput_mbps()
                    .partial_cmp(&b.avg_throughput_mbps())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(ctype, _)| *ctype)
            .unwrap_or(AdvancedCompressionType::Lz4)
    }

    /// Select algorithm with balanced ratio and speed
    fn select_balanced(&self) -> AdvancedCompressionType {
        self.metrics
            .iter()
            .max_by(|(_, a), (_, b)| {
                let score_a = a.avg_ratio() * a.avg_throughput_mbps().sqrt();
                let score_b = b.avg_ratio() * b.avg_throughput_mbps().sqrt();
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(ctype, _)| *ctype)
            .unwrap_or(AdvancedCompressionType::Snappy)
    }

    /// Select algorithm based on data size
    fn select_adaptive(&self, data_size: usize) -> AdvancedCompressionType {
        match data_size {
            0..=1024 => AdvancedCompressionType::Lz4, // Small: fast compression
            1025..=10240 => AdvancedCompressionType::Snappy, // Medium: balanced
            10241..=102400 => AdvancedCompressionType::Zstd, // Large: good ratio
            _ => AdvancedCompressionType::Zstd,       // Very large: best ratio
        }
    }

    /// Record compression metrics
    pub fn record_compression(
        &self,
        compression_type: AdvancedCompressionType,
        input_size: usize,
        compressed_size: usize,
        duration: Duration,
    ) {
        if let Some((_, metrics)) = self.metrics.iter().find(|(ct, _)| *ct == compression_type) {
            metrics.record(input_size, compressed_size, duration);
        }
    }

    /// Get metrics for an algorithm
    pub fn get_metrics(
        &self,
        compression_type: AdvancedCompressionType,
    ) -> Option<Arc<AlgorithmMetrics>> {
        self.metrics
            .iter()
            .find(|(ct, _)| *ct == compression_type)
            .map(|(_, m)| Arc::clone(m))
    }

    /// Get comprehensive statistics
    pub fn stats(&self) -> AdaptiveCompressionStats {
        let algorithm_stats = self
            .metrics
            .iter()
            .map(|(ctype, metrics)| AlgorithmStats {
                compression_type: *ctype,
                avg_ratio: metrics.avg_ratio(),
                avg_throughput_mbps: metrics.avg_throughput_mbps(),
                operations: metrics.operations.load(Ordering::Relaxed),
                total_bytes: metrics.bytes_compressed.load(Ordering::Relaxed),
            })
            .collect();

        AdaptiveCompressionStats {
            strategy: self.strategy,
            algorithm_stats,
        }
    }
}

/// Statistics for adaptive compression
#[derive(Debug)]
pub struct AdaptiveCompressionStats {
    /// Current selection strategy
    pub strategy: SelectionStrategy,
    /// Per-algorithm statistics
    pub algorithm_stats: Vec<AlgorithmStats>,
}

/// Statistics for a single algorithm
#[derive(Debug)]
pub struct AlgorithmStats {
    /// Compression type
    pub compression_type: AdvancedCompressionType,
    /// Average compression ratio
    pub avg_ratio: f64,
    /// Average throughput (MB/s)
    pub avg_throughput_mbps: f64,
    /// Number of operations
    pub operations: u64,
    /// Total bytes compressed
    pub total_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_metrics() {
        let metrics = AlgorithmMetrics::default();

        metrics.record(1000, 500, Duration::from_micros(100));
        metrics.record(2000, 800, Duration::from_micros(200));

        assert_eq!(metrics.operations.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.bytes_compressed.load(Ordering::Relaxed), 3000);
        assert_eq!(metrics.total_compressed_size.load(Ordering::Relaxed), 1300);

        // Ratio = 3000 / 1300 ≈ 2.31
        assert!((metrics.avg_ratio() - 2.31).abs() < 0.01);

        // Avg time = 300 / 2 = 150 us
        assert_eq!(metrics.avg_time_us(), 150.0);
    }

    #[test]
    fn test_adaptive_selector_creation() {
        let selector = AdaptiveCompressionSelector::new(SelectionStrategy::Balanced);

        assert_eq!(selector.strategy, SelectionStrategy::Balanced);
        assert_eq!(selector.metrics.len(), 5); // 5 compression types
    }

    #[test]
    fn test_small_data_not_compressed() {
        let selector = AdaptiveCompressionSelector::new(SelectionStrategy::MaximizeRatio);

        let selected = selector.select(50); // Small data
        assert_eq!(selected, AdvancedCompressionType::RunLength); // Fallback for small data
    }

    #[test]
    fn test_adaptive_selection_by_size() {
        let selector = AdaptiveCompressionSelector::new(SelectionStrategy::Adaptive);

        // Small data: Lz4
        assert_eq!(selector.select(500), AdvancedCompressionType::Lz4);

        // Medium data: Snappy
        assert_eq!(selector.select(5000), AdvancedCompressionType::Snappy);

        // Large data: Zstd
        assert_eq!(selector.select(50000), AdvancedCompressionType::Zstd);

        // Very large data: Zstd
        assert_eq!(selector.select(500000), AdvancedCompressionType::Zstd);
    }

    #[test]
    fn test_record_and_select() {
        let selector = AdaptiveCompressionSelector::new(SelectionStrategy::MaximizeRatio);

        // Record some operations with different ratios
        selector.record_compression(
            AdvancedCompressionType::Lz4,
            1000,
            800,
            Duration::from_micros(50),
        );
        selector.record_compression(
            AdvancedCompressionType::Zstd,
            1000,
            400,
            Duration::from_micros(200),
        );

        // Zstd has better ratio (1000/400 = 2.5 vs 1000/800 = 1.25)
        let selected = selector.select_max_ratio();
        assert_eq!(selected, AdvancedCompressionType::Zstd);
    }

    #[test]
    fn test_min_time_selection() {
        let selector = AdaptiveCompressionSelector::new(SelectionStrategy::MinimizeTime);

        // Record operations with different speeds
        selector.record_compression(
            AdvancedCompressionType::Lz4,
            10000,
            5000,
            Duration::from_micros(100), // Fast
        );
        selector.record_compression(
            AdvancedCompressionType::Zstd,
            10000,
            3000,
            Duration::from_micros(500), // Slow but better ratio
        );

        // Should select Lz4 (faster)
        let selected = selector.select_min_time();
        assert_eq!(selected, AdvancedCompressionType::Lz4);
    }

    #[test]
    fn test_get_metrics() {
        let selector = AdaptiveCompressionSelector::new(SelectionStrategy::Balanced);

        selector.record_compression(
            AdvancedCompressionType::Lz4,
            1000,
            500,
            Duration::from_micros(100),
        );

        let metrics = selector.get_metrics(AdvancedCompressionType::Lz4).unwrap();
        assert_eq!(metrics.operations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_stats() {
        let selector = AdaptiveCompressionSelector::new(SelectionStrategy::Balanced);

        selector.record_compression(
            AdvancedCompressionType::Lz4,
            1000,
            500,
            Duration::from_micros(100),
        );
        selector.record_compression(
            AdvancedCompressionType::Zstd,
            2000,
            600,
            Duration::from_micros(300),
        );

        let stats = selector.stats();
        assert_eq!(stats.strategy, SelectionStrategy::Balanced);
        assert!(stats.algorithm_stats.len() >= 2);

        // Find Lz4 stats
        let lz4_stats = stats
            .algorithm_stats
            .iter()
            .find(|s| s.compression_type == AdvancedCompressionType::Lz4)
            .unwrap();

        assert_eq!(lz4_stats.operations, 1);
        assert_eq!(lz4_stats.total_bytes, 1000);
    }
}
