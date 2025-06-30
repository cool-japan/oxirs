//! Adaptive compression techniques for vector data
//!
//! This module provides intelligent compression strategies that adapt to data characteristics,
//! optimizing compression ratio and decompression speed based on vector patterns and usage.

use crate::{Vector, VectorData, VectorError, compression::{VectorCompressor, CompressionMethod, create_compressor}};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Statistics about vector data characteristics
#[derive(Debug, Clone)]
pub struct VectorStats {
    pub dimensions: usize,
    pub mean: f32,
    pub std_dev: f32,
    pub min_val: f32,
    pub max_val: f32,
    pub entropy: f32,
    pub sparsity: f32,  // Fraction of near-zero values
    pub correlation: f32, // Average correlation between dimensions
}

impl VectorStats {
    /// Calculate statistics for a vector
    pub fn from_vector(vector: &Vector) -> Result<Self, VectorError> {
        let values = vector.as_f32();
        let n = values.len();
        
        if n == 0 {
            return Err(VectorError::InvalidDimensions("Empty vector".to_string()));
        }

        // Basic statistics
        let sum: f32 = values.iter().sum();
        let mean = sum / n as f32;
        
        let variance: f32 = values.iter().map(|x| (x - mean).powi(2)).sum() / n as f32;
        let std_dev = variance.sqrt();
        
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Entropy estimation (simplified)
        let mut histogram = [0u32; 256];
        let range = max_val - min_val;
        if range > 0.0 {
            for &val in values {
                let bucket = ((val - min_val) / range * 255.0).clamp(0.0, 255.0) as usize;
                histogram[bucket] += 1;
            }
        }
        
        let entropy = histogram.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f32 / n as f32;
                -p * p.log2()
            })
            .sum();

        // Sparsity (fraction of values near zero)
        let threshold = std_dev * 0.1;
        let sparse_count = values.iter().filter(|&&x| x.abs() < threshold).count();
        let sparsity = sparse_count as f32 / n as f32;

        // Simple correlation estimate (average pairwise correlation in windows)
        let mut correlation = 0.0;
        if n > 1 {
            let window_size = (n / 10).max(2).min(20);
            let mut corr_sum = 0.0;
            let mut corr_count = 0;
            
            for i in 0..(n - window_size) {
                let window1 = &values[i..i + window_size];
                let window2 = &values[i + 1..i + window_size + 1];
                
                let mean1: f32 = window1.iter().sum::<f32>() / window_size as f32;
                let mean2: f32 = window2.iter().sum::<f32>() / window_size as f32;
                
                let covariance: f32 = window1.iter().zip(window2).map(|(a, b)| (a - mean1) * (b - mean2)).sum();
                let var1: f32 = window1.iter().map(|x| (x - mean1).powi(2)).sum();
                let var2: f32 = window2.iter().map(|x| (x - mean2).powi(2)).sum();
                
                if var1 > 0.0 && var2 > 0.0 {
                    let corr = covariance / (var1.sqrt() * var2.sqrt());
                    corr_sum += corr.abs();
                    corr_count += 1;
                }
            }
            
            if corr_count > 0 {
                correlation = corr_sum / corr_count as f32;
            }
        }

        Ok(VectorStats {
            dimensions: n,
            mean,
            std_dev,
            min_val,
            max_val,
            entropy,
            sparsity,
            correlation,
        })
    }

    /// Calculate aggregate statistics from multiple vectors
    pub fn from_vectors(vectors: &[Vector]) -> Result<Self, VectorError> {
        if vectors.is_empty() {
            return Err(VectorError::InvalidDimensions("No vectors provided".to_string()));
        }

        let individual_stats: Result<Vec<_>, _> = vectors.iter()
            .map(|v| Self::from_vector(v))
            .collect();
        let stats = individual_stats?;

        let n = stats.len() as f32;
        
        Ok(VectorStats {
            dimensions: stats[0].dimensions,
            mean: stats.iter().map(|s| s.mean).sum::<f32>() / n,
            std_dev: stats.iter().map(|s| s.std_dev).sum::<f32>() / n,
            min_val: stats.iter().map(|s| s.min_val).fold(f32::INFINITY, |a, b| a.min(b)),
            max_val: stats.iter().map(|s| s.max_val).fold(f32::NEG_INFINITY, |a, b| a.max(b)),
            entropy: stats.iter().map(|s| s.entropy).sum::<f32>() / n,
            sparsity: stats.iter().map(|s| s.sparsity).sum::<f32>() / n,
            correlation: stats.iter().map(|s| s.correlation).sum::<f32>() / n,
        })
    }
}

/// Performance metrics for compression methods
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    pub method: CompressionMethod,
    pub compression_ratio: f32,
    pub compression_time: Duration,
    pub decompression_time: Duration,
    pub reconstruction_error: f32,
    pub usage_count: u64,
    pub avg_performance_score: f32,
}

impl CompressionMetrics {
    pub fn new(method: CompressionMethod) -> Self {
        Self {
            method,
            compression_ratio: 1.0,
            compression_time: Duration::ZERO,
            decompression_time: Duration::ZERO,
            reconstruction_error: 0.0,
            usage_count: 0,
            avg_performance_score: 0.0,
        }
    }

    /// Calculate performance score (higher is better)
    pub fn calculate_score(&self, priorities: &CompressionPriorities) -> f32 {
        let ratio_score = self.compression_ratio.min(0.9); // Cap at 90% compression
        let speed_score = 1.0 / (1.0 + self.compression_time.as_millis() as f32 / 1000.0);
        let accuracy_score = 1.0 / (1.0 + self.reconstruction_error);
        
        priorities.compression_weight * ratio_score +
        priorities.speed_weight * speed_score +
        priorities.accuracy_weight * accuracy_score
    }

    /// Update metrics with new measurement
    pub fn update(&mut self, compression_ratio: f32, comp_time: Duration, decomp_time: Duration, error: f32, priorities: &CompressionPriorities) {
        let alpha = 0.1; // Learning rate for exponential moving average
        
        self.compression_ratio = self.compression_ratio * (1.0 - alpha) + compression_ratio * alpha;
        self.compression_time = Duration::from_nanos(
            (self.compression_time.as_nanos() as f32 * (1.0 - alpha) + comp_time.as_nanos() as f32 * alpha) as u64
        );
        self.decompression_time = Duration::from_nanos(
            (self.decompression_time.as_nanos() as f32 * (1.0 - alpha) + decomp_time.as_nanos() as f32 * alpha) as u64
        );
        self.reconstruction_error = self.reconstruction_error * (1.0 - alpha) + error * alpha;
        self.usage_count += 1;
        
        self.avg_performance_score = self.calculate_score(priorities);
    }
}

/// Priorities for compression strategy selection
#[derive(Debug, Clone)]
pub struct CompressionPriorities {
    pub compression_weight: f32,  // Importance of compression ratio
    pub speed_weight: f32,        // Importance of compression/decompression speed
    pub accuracy_weight: f32,     // Importance of reconstruction accuracy
}

impl Default for CompressionPriorities {
    fn default() -> Self {
        Self {
            compression_weight: 0.4,
            speed_weight: 0.3,
            accuracy_weight: 0.3,
        }
    }
}

/// Multi-level compression strategy
#[derive(Debug, Clone)]
pub struct MultiLevelCompression {
    pub levels: Vec<CompressionMethod>,
    pub thresholds: Vec<f32>, // Quality thresholds for each level
}

impl MultiLevelCompression {
    pub fn new() -> Self {
        Self {
            levels: vec![
                CompressionMethod::None,
                CompressionMethod::Quantization { bits: 16 },
                CompressionMethod::Quantization { bits: 8 },
                CompressionMethod::Pca { components: 0 }, // Will be set adaptively
                CompressionMethod::Zstd { level: 3 },
            ],
            thresholds: vec![0.0, 0.1, 0.3, 0.6, 0.8],
        }
    }

    /// Select compression level based on requirements
    pub fn select_level(&self, required_compression: f32) -> &CompressionMethod {
        for (i, &threshold) in self.thresholds.iter().enumerate() {
            if required_compression <= threshold {
                return &self.levels[i];
            }
        }
        self.levels.last().unwrap()
    }
}

/// Adaptive compression engine that learns optimal strategies
pub struct AdaptiveCompressor {
    /// Current compression priorities
    priorities: CompressionPriorities,
    /// Performance metrics for each compression method
    metrics: Arc<RwLock<HashMap<String, CompressionMetrics>>>,
    /// Multi-level compression strategies
    multi_level: MultiLevelCompression,
    /// Cache of trained compressors
    compressor_cache: Arc<RwLock<HashMap<String, Box<dyn VectorCompressor + Send + Sync>>>>,
    /// Statistics cache for similar vectors
    stats_cache: Arc<RwLock<HashMap<String, (VectorStats, Instant)>>>,
    /// Learning parameters
    exploration_rate: f32,
    cache_ttl: Duration,
}

impl AdaptiveCompressor {
    pub fn new() -> Self {
        Self::new_with_priorities(CompressionPriorities::default())
    }

    pub fn new_with_priorities(priorities: CompressionPriorities) -> Self {
        Self {
            priorities,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            multi_level: MultiLevelCompression::new(),
            compressor_cache: Arc::new(RwLock::new(HashMap::new())),
            stats_cache: Arc::new(RwLock::new(HashMap::new())),
            exploration_rate: 0.1,
            cache_ttl: Duration::from_secs(3600), // 1 hour cache TTL
        }
    }

    /// Analyze vector characteristics and recommend compression strategy
    pub fn analyze_and_recommend(&mut self, vectors: &[Vector]) -> Result<CompressionMethod, VectorError> {
        let stats = VectorStats::from_vectors(vectors)?;
        let stats_key = self.generate_stats_key(&stats);

        // Check cache first
        {
            let cache = self.stats_cache.read().unwrap();
            if let Some((cached_stats, timestamp)) = cache.get(&stats_key) {
                if timestamp.elapsed() < self.cache_ttl {
                    return Ok(self.recommend_from_stats(cached_stats));
                }
            }
        }

        // Cache the stats
        {
            let mut cache = self.stats_cache.write().unwrap();
            cache.insert(stats_key, (stats.clone(), Instant::now()));
        }

        Ok(self.recommend_from_stats(&stats))
    }

    /// Recommend compression method based on vector statistics
    fn recommend_from_stats(&self, stats: &VectorStats) -> CompressionMethod {
        // High sparsity -> prefer quantization or PCA
        if stats.sparsity > 0.7 {
            return CompressionMethod::Quantization { bits: 4 };
        }

        // High correlation -> PCA works well
        if stats.correlation > 0.6 && stats.dimensions > 20 {
            let components = (stats.dimensions as f32 * 0.7) as usize;
            return CompressionMethod::Pca { components };
        }

        // Low entropy -> Zstd compression is effective
        if stats.entropy < 4.0 {
            return CompressionMethod::Zstd { level: 9 };
        }

        // High variance -> quantization with more bits
        if stats.std_dev > stats.mean.abs() {
            return CompressionMethod::Quantization { bits: 12 };
        }

        // Default: moderate quantization
        CompressionMethod::Quantization { bits: 8 }
    }

    /// Compress vector with adaptive strategy selection
    pub fn compress_adaptive(&mut self, vector: &Vector) -> Result<Vec<u8>, VectorError> {
        let stats = VectorStats::from_vector(vector)?;
        let method = self.recommend_from_stats(&stats);
        
        // Check if we should explore alternative methods
        if self.should_explore() {
            let alternative = self.get_alternative_method(&method);
            return self.compress_with_method(vector, &alternative);
        }

        self.compress_with_method(vector, &method)
    }

    /// Compress with specific method and update metrics
    pub fn compress_with_method(&mut self, vector: &Vector, method: &CompressionMethod) -> Result<Vec<u8>, VectorError> {
        let method_key = format!("{:?}", method);
        let compressor = self.get_or_create_compressor(method)?;
        
        let start_time = Instant::now();
        let compressed = compressor.compress(vector)?;
        let compression_time = start_time.elapsed();
        
        // Measure reconstruction error
        let decompressed = compressor.decompress(&compressed, vector.dimensions)?;
        let error = self.calculate_reconstruction_error(vector, &decompressed)?;
        
        let compression_ratio = compressed.len() as f32 / (vector.dimensions * 4) as f32; // Assuming f32 vectors
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            let metric = metrics.entry(method_key).or_insert_with(|| CompressionMetrics::new(method.clone()));
            metric.update(compression_ratio, compression_time, Duration::ZERO, error, &self.priorities);
        }

        Ok(compressed)
    }

    /// Multi-level compression for extreme compression ratios
    pub fn compress_multi_level(&mut self, vector: &Vector, target_ratio: f32) -> Result<Vec<u8>, VectorError> {
        let mut current_vector = vector.clone();
        let mut compression_steps = Vec::new();
        let mut total_ratio = 1.0;

        while total_ratio > target_ratio && compression_steps.len() < 3 {
            let remaining_ratio = target_ratio / total_ratio;
            let method = self.multi_level.select_level(remaining_ratio);
            
            let compressor = self.get_or_create_compressor(method)?;
            let compressed = compressor.compress(&current_vector)?;
            
            let step_ratio = compressed.len() as f32 / (current_vector.dimensions * 4) as f32;
            total_ratio *= step_ratio;
            
            compression_steps.push((method.clone(), compressed));
            
            // Prepare for next level if needed
            if total_ratio > target_ratio {
                current_vector = compressor.decompress(&compressed, current_vector.dimensions)?;
            }
        }

        // Serialize the compression steps
        self.serialize_multi_level_result(compression_steps)
    }

    /// Get best performing compression method based on current metrics
    pub fn get_best_method(&self) -> CompressionMethod {
        let metrics = self.metrics.read().unwrap();
        let best = metrics.values()
            .max_by(|a, b| a.avg_performance_score.partial_cmp(&b.avg_performance_score).unwrap());
        
        best.map(|m| m.method.clone())
            .unwrap_or(CompressionMethod::Quantization { bits: 8 })
    }

    /// Get compression performance statistics
    pub fn get_performance_stats(&self) -> HashMap<String, CompressionMetrics> {
        self.metrics.read().unwrap().clone()
    }

    /// Update compression priorities
    pub fn update_priorities(&mut self, priorities: CompressionPriorities) {
        self.priorities = priorities;
        
        // Recalculate scores for all metrics
        let mut metrics = self.metrics.write().unwrap();
        for metric in metrics.values_mut() {
            metric.avg_performance_score = metric.calculate_score(&self.priorities);
        }
    }

    /// Clear caches and reset learning
    pub fn reset(&mut self) {
        self.metrics.write().unwrap().clear();
        self.compressor_cache.write().unwrap().clear();
        self.stats_cache.write().unwrap().clear();
    }

    // Private helper methods

    fn get_or_create_compressor(&self, method: &CompressionMethod) -> Result<Box<dyn VectorCompressor>, VectorError> {
        let method_key = format!("{:?}", method);
        
        {
            let cache = self.compressor_cache.read().unwrap();
            if cache.contains_key(&method_key) {
                // Note: We can't return a reference here due to trait object limitations
                // So we create a new instance
            }
        }

        // Create new compressor (existing create_compressor function)
        let compressor = create_compressor(method);
        
        // Cache it (though we can't use it directly due to trait object limitations)
        {
            let mut cache = self.compressor_cache.write().unwrap();
            // Note: This is a placeholder for caching logic
            // In practice, we might need to redesign this for trait objects
        }

        Ok(compressor)
    }

    fn calculate_reconstruction_error(&self, original: &Vector, reconstructed: &Vector) -> Result<f32, VectorError> {
        let orig_values = original.as_f32();
        let recon_values = reconstructed.as_f32();
        
        if orig_values.len() != recon_values.len() {
            return Err(VectorError::InvalidDimensions("Dimension mismatch".to_string()));
        }

        let mse: f32 = orig_values.iter()
            .zip(recon_values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / orig_values.len() as f32;

        Ok(mse.sqrt()) // RMSE
    }

    fn generate_stats_key(&self, stats: &VectorStats) -> String {
        format!("{}_{:.2}_{:.2}_{:.2}_{:.2}", 
                stats.dimensions, 
                stats.entropy, 
                stats.sparsity, 
                stats.correlation,
                stats.std_dev)
    }

    fn should_explore(&self) -> bool {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen::<f32>() < self.exploration_rate
    }

    fn get_alternative_method(&self, current: &CompressionMethod) -> CompressionMethod {
        match current {
            CompressionMethod::None => CompressionMethod::Quantization { bits: 8 },
            CompressionMethod::Quantization { bits } => {
                if *bits > 8 {
                    CompressionMethod::Quantization { bits: bits - 2 }
                } else {
                    CompressionMethod::Pca { components: 16 }
                }
            },
            CompressionMethod::Pca { components } => {
                CompressionMethod::Zstd { level: 6 }
            },
            CompressionMethod::Zstd { level } => {
                if *level < 15 {
                    CompressionMethod::Zstd { level: level + 3 }
                } else {
                    CompressionMethod::Quantization { bits: 4 }
                }
            },
            _ => CompressionMethod::None,
        }
    }

    fn serialize_multi_level_result(&self, steps: Vec<(CompressionMethod, Vec<u8>)>) -> Result<Vec<u8>, VectorError> {
        use std::io::Write;
        
        let mut result = Vec::new();
        
        // Write number of steps
        result.write_all(&(steps.len() as u32).to_le_bytes())?;
        
        // Write each step
        for (method, data) in steps {
            // Serialize method (simplified)
            let method_id = match method {
                CompressionMethod::None => 0u8,
                CompressionMethod::Zstd { .. } => 1u8,
                CompressionMethod::Quantization { .. } => 2u8,
                CompressionMethod::Pca { .. } => 3u8,
                CompressionMethod::ProductQuantization { .. } => 4u8,
            };
            result.push(method_id);
            
            // Write data length and data
            result.write_all(&(data.len() as u32).to_le_bytes())?;
            result.extend_from_slice(&data);
        }
        
        Ok(result)
    }
}

impl Default for AdaptiveCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_stats() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = VectorStats::from_vector(&vector).unwrap();
        
        assert_eq!(stats.dimensions, 5);
        assert_eq!(stats.mean, 3.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_adaptive_compression() {
        let vectors = vec![
            Vector::new(vec![1.0, 2.0, 3.0, 4.0]),
            Vector::new(vec![2.0, 3.0, 4.0, 5.0]),
            Vector::new(vec![3.0, 4.0, 5.0, 6.0]),
        ];

        let mut compressor = AdaptiveCompressor::new();
        let recommended = compressor.analyze_and_recommend(&vectors).unwrap();
        
        // Should recommend some compression method
        assert!(!matches!(recommended, CompressionMethod::None));
    }

    #[test]
    fn test_compression_metrics() {
        let method = CompressionMethod::Quantization { bits: 8 };
        let mut metrics = CompressionMetrics::new(method);
        let priorities = CompressionPriorities::default();
        
        metrics.update(0.5, Duration::from_millis(10), Duration::from_millis(5), 0.01, &priorities);
        
        assert!(metrics.avg_performance_score > 0.0);
        assert_eq!(metrics.usage_count, 1);
    }

    #[test]
    fn test_multi_level_compression() {
        let mut compressor = AdaptiveCompressor::new();
        let vector = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        
        let compressed = compressor.compress_multi_level(&vector, 0.1).unwrap();
        
        // Should achieve significant compression
        assert!(compressed.len() < vector.dimensions * 4);
    }

    #[test]
    fn test_stats_aggregation() {
        let vectors = vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![3.0, 4.0]),
            Vector::new(vec![5.0, 6.0]),
        ];

        let stats = VectorStats::from_vectors(&vectors).unwrap();
        assert_eq!(stats.dimensions, 2);
        assert!(stats.mean > 0.0);
    }
}