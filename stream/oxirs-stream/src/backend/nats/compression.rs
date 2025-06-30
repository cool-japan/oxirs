//! # NATS Compression Module
//!
//! Advanced compression algorithms for NATS messaging with adaptive selection,
//! machine learning-based optimization, and intelligent caching.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

#[cfg(feature = "compression")]
use flate2::{read::GzDecoder, write::GzEncoder, Compression as GzCompression};
#[cfg(feature = "compression")]
use lz4::{Decoder as Lz4Decoder, EncoderBuilder as Lz4EncoderBuilder};
#[cfg(feature = "compression")]
use snap::{read::FrameDecoder as SnapDecoder, write::FrameEncoder as SnapEncoder};
#[cfg(feature = "compression")]
use zstd::{Decoder as ZstdDecoder, Encoder as ZstdEncoder};

use std::io::{Read, Write};

/// Compression algorithms supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Lz4,
    Snappy,
    Zstd,
    Adaptive, // Automatically select best algorithm
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub algorithm: CompressionAlgorithm,
    pub level: u8,
    pub min_size_threshold: usize,
    pub enable_adaptive_selection: bool,
    pub enable_ml_optimization: bool,
    pub cache_enabled: bool,
    pub cache_size_limit: usize,
    pub benchmark_interval_seconds: u64,
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub algorithm: CompressionAlgorithm,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_time_ms: u64,
    pub decompression_time_ms: u64,
    pub cpu_efficiency: f64,
}

/// Algorithm performance metrics
#[derive(Debug, Clone)]
pub struct AlgorithmMetrics {
    pub total_compressions: u64,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
    pub total_compression_time_ms: u64,
    pub total_decompression_time_ms: u64,
    pub average_ratio: f64,
    pub average_speed_mbps: f64,
}

/// Machine learning model for compression optimization
#[derive(Debug, Clone)]
struct CompressionMLModel {
    algorithm_scores: HashMap<CompressionAlgorithm, f64>,
    content_type_preferences: HashMap<String, CompressionAlgorithm>,
    size_based_thresholds: Vec<(usize, CompressionAlgorithm)>,
    learning_rate: f64,
    update_count: u64,
}

/// Advanced compression manager
pub struct CompressionManager {
    config: CompressionConfig,
    algorithm_metrics: HashMap<CompressionAlgorithm, AlgorithmMetrics>,
    compression_cache: HashMap<Vec<u8>, (CompressionAlgorithm, Vec<u8>)>,
    ml_model: Option<CompressionMLModel>,
    last_benchmark: std::time::Instant,
}

impl CompressionManager {
    /// Create new compression manager
    pub fn new(config: CompressionConfig) -> Self {
        let ml_model = if config.enable_ml_optimization {
            Some(CompressionMLModel::new())
        } else {
            None
        };

        Self {
            config,
            algorithm_metrics: HashMap::new(),
            compression_cache: HashMap::new(),
            ml_model,
            last_benchmark: std::time::Instant::now(),
        }
    }

    /// Compress data with optimal algorithm selection
    pub fn compress(&mut self, data: &[u8]) -> Result<(Vec<u8>, CompressionStats)> {
        if data.len() < self.config.min_size_threshold {
            return Ok((
                data.to_vec(),
                CompressionStats {
                    algorithm: CompressionAlgorithm::None,
                    original_size: data.len(),
                    compressed_size: data.len(),
                    compression_ratio: 1.0,
                    compression_time_ms: 0,
                    decompression_time_ms: 0,
                    cpu_efficiency: 1.0,
                },
            ));
        }

        // Check cache first
        if self.config.cache_enabled {
            if let Some((algorithm, compressed)) = self.compression_cache.get(data) {
                debug!("Cache hit for compression: {:?}", algorithm);
                return Ok((
                    compressed.clone(),
                    CompressionStats {
                        algorithm: algorithm.clone(),
                        original_size: data.len(),
                        compressed_size: compressed.len(),
                        compression_ratio: data.len() as f64 / compressed.len() as f64,
                        compression_time_ms: 0,
                        decompression_time_ms: 0,
                        cpu_efficiency: 1.0,
                    },
                ));
            }
        }

        // Select optimal algorithm
        let algorithm = self.select_optimal_algorithm(data);
        
        let start_time = std::time::Instant::now();
        let compressed_data = self.compress_with_algorithm(data, &algorithm)?;
        let compression_time = start_time.elapsed();

        // Test decompression time
        let start_decomp_time = std::time::Instant::now();
        let _decompressed = self.decompress_with_algorithm(&compressed_data, &algorithm)?;
        let decompression_time = start_decomp_time.elapsed();

        let stats = CompressionStats {
            algorithm: algorithm.clone(),
            original_size: data.len(),
            compressed_size: compressed_data.len(),
            compression_ratio: data.len() as f64 / compressed_data.len() as f64,
            compression_time_ms: compression_time.as_millis() as u64,
            decompression_time_ms: decompression_time.as_millis() as u64,
            cpu_efficiency: self.calculate_cpu_efficiency(&stats),
        };

        // Update metrics
        self.update_metrics(&algorithm, &stats);

        // Update ML model
        if let Some(ref mut model) = self.ml_model {
            model.update(&algorithm, data, &stats);
        }

        // Cache result
        if self.config.cache_enabled && self.compression_cache.len() < self.config.cache_size_limit {
            self.compression_cache.insert(data.to_vec(), (algorithm.clone(), compressed_data.clone()));
        }

        // Periodic benchmarking
        if self.last_benchmark.elapsed().as_secs() > self.config.benchmark_interval_seconds {
            self.run_benchmark(data);
            self.last_benchmark = std::time::Instant::now();
        }

        info!("Compressed {} bytes to {} bytes using {:?} (ratio: {:.2})", 
              data.len(), compressed_data.len(), algorithm, stats.compression_ratio);

        Ok((compressed_data, stats))
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8], algorithm: &CompressionAlgorithm) -> Result<Vec<u8>> {
        if *algorithm == CompressionAlgorithm::None {
            return Ok(data.to_vec());
        }

        self.decompress_with_algorithm(data, algorithm)
    }

    /// Select optimal algorithm based on data characteristics and ML model
    fn select_optimal_algorithm(&self, data: &[u8]) -> CompressionAlgorithm {
        match self.config.algorithm {
            CompressionAlgorithm::Adaptive => {
                // Use ML model if available
                if let Some(ref model) = self.ml_model {
                    if let Some(algorithm) = model.predict_best_algorithm(data) {
                        return algorithm;
                    }
                }

                // Fallback to heuristic-based selection
                self.heuristic_algorithm_selection(data)
            }
            ref algo => algo.clone(),
        }
    }

    /// Heuristic-based algorithm selection
    fn heuristic_algorithm_selection(&self, data: &[u8]) -> CompressionAlgorithm {
        let size = data.len();
        
        // Analyze data characteristics
        let entropy = self.calculate_entropy(data);
        let repetition_factor = self.calculate_repetition_factor(data);
        
        // Size-based selection
        if size < 1024 {
            CompressionAlgorithm::Lz4 // Fast for small data
        } else if size < 10240 {
            if entropy < 0.7 {
                CompressionAlgorithm::Zstd // Good compression for low entropy
            } else {
                CompressionAlgorithm::Snappy // Fast for high entropy
            }
        } else {
            if repetition_factor > 0.3 {
                CompressionAlgorithm::Zstd // Excellent for repetitive data
            } else if entropy > 0.9 {
                CompressionAlgorithm::Lz4 // Fast for random data
            } else {
                CompressionAlgorithm::Gzip // Balanced choice
            }
        }
    }

    /// Calculate Shannon entropy of data
    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }

        let length = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &counts {
            if count > 0 {
                let probability = count as f64 / length;
                entropy -= probability * probability.log2();
            }
        }

        entropy / 8.0 // Normalize to 0-1 range
    }

    /// Calculate repetition factor
    fn calculate_repetition_factor(&self, data: &[u8]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mut repetitions = 0;
        for i in 1..data.len() {
            if data[i] == data[i - 1] {
                repetitions += 1;
            }
        }

        repetitions as f64 / (data.len() - 1) as f64
    }

    /// Compress with specific algorithm
    fn compress_with_algorithm(&self, data: &[u8], algorithm: &CompressionAlgorithm) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            
            #[cfg(feature = "compression")]
            CompressionAlgorithm::Gzip => {
                let mut encoder = GzEncoder::new(Vec::new(), GzCompression::new(self.config.level as u32));
                encoder.write_all(data)?;
                encoder.finish().map_err(|e| anyhow!("Gzip compression failed: {}", e))
            }

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Lz4 => {
                let mut encoder = Lz4EncoderBuilder::new()
                    .level(self.config.level as u32)
                    .build(Vec::new())?;
                encoder.write_all(data)?;
                let (compressed, _) = encoder.finish();
                Ok(compressed)
            }

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Snappy => {
                let mut encoder = SnapEncoder::new(Vec::new());
                encoder.write_all(data)?;
                encoder.into_inner().map_err(|e| anyhow!("Snappy compression failed: {}", e))
            }

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Zstd => {
                let mut encoder = ZstdEncoder::new(Vec::new(), self.config.level as i32)?;
                encoder.write_all(data)?;
                encoder.finish().map_err(|e| anyhow!("Zstd compression failed: {}", e))
            }

            #[cfg(not(feature = "compression"))]
            _ => Ok(data.to_vec()),

            CompressionAlgorithm::Adaptive => {
                // Should not reach here
                Ok(data.to_vec())
            }
        }
    }

    /// Decompress with specific algorithm
    fn decompress_with_algorithm(&self, data: &[u8], algorithm: &CompressionAlgorithm) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Gzip => {
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Lz4 => {
                let mut decoder = Lz4Decoder::new(data)?;
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Snappy => {
                let mut decoder = SnapDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }

            #[cfg(feature = "compression")]
            CompressionAlgorithm::Zstd => {
                let mut decoder = ZstdDecoder::new(data)?;
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }

            #[cfg(not(feature = "compression"))]
            _ => Ok(data.to_vec()),

            CompressionAlgorithm::Adaptive => {
                // Should not reach here
                Ok(data.to_vec())
            }
        }
    }

    /// Calculate CPU efficiency score
    fn calculate_cpu_efficiency(&self, stats: &CompressionStats) -> f64 {
        let total_time = stats.compression_time_ms + stats.decompression_time_ms;
        if total_time == 0 {
            return 1.0;
        }

        let bytes_per_ms = stats.original_size as f64 / total_time as f64;
        let compression_benefit = stats.compression_ratio;
        
        // Combine throughput and compression benefit
        (bytes_per_ms * compression_benefit).min(1.0)
    }

    /// Update algorithm metrics
    fn update_metrics(&mut self, algorithm: &CompressionAlgorithm, stats: &CompressionStats) {
        let metrics = self.algorithm_metrics
            .entry(algorithm.clone())
            .or_insert_with(|| AlgorithmMetrics {
                total_compressions: 0,
                total_original_bytes: 0,
                total_compressed_bytes: 0,
                total_compression_time_ms: 0,
                total_decompression_time_ms: 0,
                average_ratio: 0.0,
                average_speed_mbps: 0.0,
            });

        metrics.total_compressions += 1;
        metrics.total_original_bytes += stats.original_size as u64;
        metrics.total_compressed_bytes += stats.compressed_size as u64;
        metrics.total_compression_time_ms += stats.compression_time_ms;
        metrics.total_decompression_time_ms += stats.decompression_time_ms;

        // Update averages
        metrics.average_ratio = metrics.total_original_bytes as f64 / metrics.total_compressed_bytes as f64;
        
        let total_time_s = (metrics.total_compression_time_ms + metrics.total_decompression_time_ms) as f64 / 1000.0;
        let total_mb = metrics.total_original_bytes as f64 / (1024.0 * 1024.0);
        metrics.average_speed_mbps = if total_time_s > 0.0 { total_mb / total_time_s } else { 0.0 };
    }

    /// Run performance benchmark
    fn run_benchmark(&self, sample_data: &[u8]) {
        info!("Running compression algorithm benchmark");
        
        let algorithms = vec![
            CompressionAlgorithm::Gzip,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Snappy,
            CompressionAlgorithm::Zstd,
        ];

        for algorithm in algorithms {
            if let Ok(compressed) = self.compress_with_algorithm(sample_data, &algorithm) {
                let ratio = sample_data.len() as f64 / compressed.len() as f64;
                debug!("Benchmark {:?}: ratio {:.2}", algorithm, ratio);
            }
        }
    }

    /// Get compression statistics
    pub fn get_algorithm_metrics(&self) -> &HashMap<CompressionAlgorithm, AlgorithmMetrics> {
        &self.algorithm_metrics
    }

    /// Clear compression cache
    pub fn clear_cache(&mut self) {
        self.compression_cache.clear();
        info!("Compression cache cleared");
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (self.compression_cache.len(), self.config.cache_size_limit)
    }
}

impl CompressionMLModel {
    fn new() -> Self {
        let mut algorithm_scores = HashMap::new();
        algorithm_scores.insert(CompressionAlgorithm::Gzip, 0.5);
        algorithm_scores.insert(CompressionAlgorithm::Lz4, 0.5);
        algorithm_scores.insert(CompressionAlgorithm::Snappy, 0.5);
        algorithm_scores.insert(CompressionAlgorithm::Zstd, 0.5);

        Self {
            algorithm_scores,
            content_type_preferences: HashMap::new(),
            size_based_thresholds: vec![
                (1024, CompressionAlgorithm::Lz4),
                (10240, CompressionAlgorithm::Snappy),
                (102400, CompressionAlgorithm::Zstd),
            ],
            learning_rate: 0.01,
            update_count: 0,
        }
    }

    fn predict_best_algorithm(&self, data: &[u8]) -> Option<CompressionAlgorithm> {
        // Simple algorithm selection based on learned scores and data size
        let size = data.len();
        
        // Check size-based thresholds first
        for (threshold, algorithm) in &self.size_based_thresholds {
            if size <= *threshold {
                return Some(algorithm.clone());
            }
        }

        // Find algorithm with highest score
        self.algorithm_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(algo, _)| algo.clone())
    }

    fn update(&mut self, algorithm: &CompressionAlgorithm, _data: &[u8], stats: &CompressionStats) {
        // Update algorithm score based on performance
        let performance_score = stats.cpu_efficiency * stats.compression_ratio;
        
        if let Some(current_score) = self.algorithm_scores.get_mut(algorithm) {
            *current_score = *current_score * (1.0 - self.learning_rate) + performance_score * self.learning_rate;
        }

        self.update_count += 1;

        // Adapt learning rate
        if self.update_count % 100 == 0 {
            self.learning_rate *= 0.95;
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Adaptive,
            level: 6,
            min_size_threshold: 256,
            enable_adaptive_selection: true,
            enable_ml_optimization: true,
            cache_enabled: true,
            cache_size_limit: 1000,
            benchmark_interval_seconds: 300,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_manager_creation() {
        let config = CompressionConfig::default();
        let manager = CompressionManager::new(config);
        
        assert_eq!(manager.compression_cache.len(), 0);
    }

    #[test]
    fn test_entropy_calculation() {
        let config = CompressionConfig::default();
        let manager = CompressionManager::new(config);
        
        // Test with uniform data (high entropy)
        let uniform_data = vec![0u8; 100];
        let entropy = manager.calculate_entropy(&uniform_data);
        assert!(entropy < 0.1); // Should be very low entropy
        
        // Test with random data
        let random_data: Vec<u8> = (0..100).map(|i| (i * 7) as u8).collect();
        let entropy2 = manager.calculate_entropy(&random_data);
        assert!(entropy2 > entropy); // Should be higher entropy
    }

    #[test]
    fn test_repetition_factor() {
        let config = CompressionConfig::default();
        let manager = CompressionManager::new(config);
        
        let repetitive_data = vec![1u8, 1, 2, 2, 3, 3];
        let factor = manager.calculate_repetition_factor(&repetitive_data);
        assert!(factor > 0.0);
        
        let non_repetitive_data = vec![1u8, 2, 3, 4, 5, 6];
        let factor2 = manager.calculate_repetition_factor(&non_repetitive_data);
        assert_eq!(factor2, 0.0);
    }

    #[test]
    fn test_algorithm_selection() {
        let config = CompressionConfig::default();
        let manager = CompressionManager::new(config);
        
        let small_data = vec![1u8; 100];
        let algorithm = manager.heuristic_algorithm_selection(&small_data);
        assert_eq!(algorithm, CompressionAlgorithm::Lz4);
        
        let large_data = vec![1u8; 100000];
        let algorithm2 = manager.heuristic_algorithm_selection(&large_data);
        assert_eq!(algorithm2, CompressionAlgorithm::Zstd); // High repetition
    }
}