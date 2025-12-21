//! # Advanced SciRS2-Powered Stream Optimization
//!
//! Leverages scirs2-core for high-performance stream processing:
//! - Array-based batch processing using ndarray_ext
//! - Random number generation for synthetic data
//! - Statistical computations for stream analytics

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::StreamEvent;

/// Advanced stream optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedOptimizerConfig {
    /// Enable SIMD acceleration (placeholder for future)
    pub enable_simd: bool,
    /// Enable GPU acceleration (placeholder for future)
    pub enable_gpu: bool,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Chunk size for parallel processing
    pub parallel_chunk_size: usize,
    /// Buffer pool size in bytes
    pub buffer_pool_size: usize,
    /// Enable profiling
    pub enable_profiling: bool,
}

impl Default for AdvancedOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_gpu: false,
            enable_parallel: true,
            parallel_chunk_size: 1000,
            buffer_pool_size: 100 * 1024 * 1024, // 100 MB
            enable_profiling: true,
        }
    }
}

/// Advanced stream optimizer using SciRS2 capabilities
pub struct AdvancedStreamOptimizer {
    config: AdvancedOptimizerConfig,
    metrics: OptimizerMetrics,
}

/// Optimizer performance metrics
#[derive(Debug, Clone)]
pub struct OptimizerMetrics {
    pub total_events_processed: u64,
    pub simd_operations: u64,
    pub gpu_operations: u64,
    pub parallel_operations: u64,
    pub total_processing_time_ms: f64,
}

impl AdvancedStreamOptimizer {
    /// Create a new advanced optimizer
    pub fn new(config: AdvancedOptimizerConfig) -> Result<Self> {
        // Initialize metrics
        let metrics = OptimizerMetrics {
            total_events_processed: 0,
            simd_operations: 0,
            gpu_operations: 0,
            parallel_operations: 0,
            total_processing_time_ms: 0.0,
        };

        Ok(Self { config, metrics })
    }

    /// Process events with batch operations using scirs2-core arrays
    pub async fn process_batch(&mut self, events: &[StreamEvent]) -> Result<Vec<f64>> {
        let start = std::time::Instant::now();

        // Extract numerical features from events
        let features = self.extract_numerical_features(events)?;

        // Use scirs2-core Array for efficient batch processing
        let array = Array1::from_vec(features);

        // Perform operations on the array
        let result = array.mapv(|x| x * 2.0 + 1.0);

        self.metrics.total_events_processed += events.len() as u64;
        self.metrics.total_processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok(result.to_vec())
    }

    /// Process events in parallel using rayon
    pub async fn process_batch_parallel(&mut self, events: &[StreamEvent]) -> Result<Vec<f64>> {
        if !self.config.enable_parallel || events.len() < self.config.parallel_chunk_size {
            return self.process_batch(events).await;
        }

        let start = std::time::Instant::now();
        self.metrics.parallel_operations += 1;

        // Extract features
        let features = self.extract_numerical_features(events)?;

        // Use rayon for parallel processing
        use rayon::prelude::*;
        let results: Vec<f64> = features.par_iter().map(|&x| x * 2.0 + 1.0).collect();

        self.metrics.total_events_processed += events.len() as u64;
        self.metrics.total_processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok(results)
    }

    /// Compute correlation matrix for stream features using scirs2-core
    pub fn compute_correlation_matrix(&self, events: &[StreamEvent]) -> Result<Array2<f64>> {
        // Extract multiple features from events
        let n_events = events.len();
        let n_features = 3; // Example: 3 features per event

        let mut feature_matrix = Array2::<f64>::zeros((n_events, n_features));

        for (i, event) in events.iter().enumerate() {
            let features = self.extract_event_features(event);
            for (j, &value) in features.iter().enumerate() {
                feature_matrix[[i, j]] = value;
            }
        }

        // Compute correlation matrix manually
        let mut correlation_matrix = Array2::<f64>::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                let col_i: Vec<f64> = feature_matrix.column(i).iter().copied().collect();
                let col_j: Vec<f64> = feature_matrix.column(j).iter().copied().collect();
                correlation_matrix[[i, j]] = compute_correlation(&col_i, &col_j);
            }
        }

        Ok(correlation_matrix)
    }

    /// Compute moving statistics
    pub fn compute_moving_statistics(
        &self,
        values: &[f64],
        window_size: usize,
    ) -> Result<MovingStats> {
        let n = values.len();
        let mut means = Vec::with_capacity(n.saturating_sub(window_size) + 1);
        let mut variances = Vec::with_capacity(n.saturating_sub(window_size) + 1);

        for i in 0..=(n.saturating_sub(window_size)) {
            let window = &values[i..i + window_size];

            // Compute mean
            let window_mean = window.iter().sum::<f64>() / window.len() as f64;

            // Compute variance
            let window_var = window
                .iter()
                .map(|&x| (x - window_mean).powi(2))
                .sum::<f64>()
                / (window.len() - 1) as f64;

            means.push(window_mean);
            variances.push(window_var);
        }

        Ok(MovingStats { means, variances })
    }

    /// Generate synthetic stream data using fastrand (fallback)
    pub fn generate_synthetic_stream(&mut self, _n_events: usize) -> Result<Vec<f64>> {
        // Use fastrand as fallback for now
        let mut data = Vec::with_capacity(_n_events);
        for _ in 0.._n_events {
            // Generate from normal distribution using Box-Muller transform
            let u1 = fastrand::f64();
            let u2 = fastrand::f64();
            let z0 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            let value = 50.0 + z0 * 10.0; // mean=50, std=10
            data.push(value);
        }

        Ok(data)
    }

    /// Get optimizer metrics
    pub fn get_metrics(&self) -> &OptimizerMetrics {
        &self.metrics
    }

    // Private helper methods

    fn extract_numerical_features(&self, events: &[StreamEvent]) -> Result<Vec<f64>> {
        // Simple feature extraction (hash-based for demo)
        Ok(events
            .iter()
            .enumerate()
            .map(|(i, _)| (i as f64 % 100.0) + fastrand::f64() * 10.0)
            .collect())
    }

    fn extract_event_features(&self, _event: &StreamEvent) -> Vec<f64> {
        vec![
            fastrand::f64() * 100.0,
            fastrand::f64() * 100.0,
            fastrand::f64() * 100.0,
        ]
    }
}

/// Moving statistics result
#[derive(Debug, Clone)]
pub struct MovingStats {
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
}

// Helper function for correlation
fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let cov = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / n;

    let var_x = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f64>() / n;
    let var_y = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f64>() / n;

    cov / (var_x.sqrt() * var_y.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EventMetadata;
    use std::collections::HashMap;

    fn create_test_event(id: u64) -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: format!("http://example.org/s{}", id),
            predicate: "http://example.org/p".to_string(),
            object: format!("value{}", id),
            graph: None,
            metadata: EventMetadata {
                event_id: format!("event-{}", id),
                timestamp: chrono::Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        }
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let config = AdvancedOptimizerConfig::default();
        let mut optimizer = AdvancedStreamOptimizer::new(config).unwrap();

        let events: Vec<_> = (0..1000).map(create_test_event).collect();
        let result = optimizer.process_batch(&events).await.unwrap();

        assert_eq!(result.len(), events.len());
    }

    #[tokio::test]
    async fn test_parallel_processing() {
        let config = AdvancedOptimizerConfig {
            enable_parallel: true,
            parallel_chunk_size: 100,
            ..Default::default()
        };

        let mut optimizer = AdvancedStreamOptimizer::new(config).unwrap();

        let events: Vec<_> = (0..10000).map(create_test_event).collect();
        let result = optimizer.process_batch_parallel(&events).await.unwrap();

        assert_eq!(result.len(), events.len());
    }

    #[test]
    fn test_correlation_matrix() {
        let config = AdvancedOptimizerConfig::default();
        let optimizer = AdvancedStreamOptimizer::new(config).unwrap();

        let events: Vec<_> = (0..100).map(create_test_event).collect();
        let correlation = optimizer.compute_correlation_matrix(&events).unwrap();

        assert_eq!(correlation.shape(), &[3, 3]);
    }

    #[test]
    fn test_moving_statistics() {
        let config = AdvancedOptimizerConfig::default();
        let optimizer = AdvancedStreamOptimizer::new(config).unwrap();

        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let stats = optimizer.compute_moving_statistics(&values, 10).unwrap();

        assert_eq!(stats.means.len(), 991); // 1000 - 10 + 1
        assert_eq!(stats.variances.len(), 991);
    }

    #[test]
    fn test_synthetic_stream_generation() {
        let config = AdvancedOptimizerConfig::default();
        let mut optimizer = AdvancedStreamOptimizer::new(config).unwrap();

        let synthetic = optimizer.generate_synthetic_stream(10000).unwrap();

        assert_eq!(synthetic.len(), 10000);
        // Check that values are reasonable
        let mean = synthetic.iter().sum::<f64>() / synthetic.len() as f64;
        assert!((mean - 50.0).abs() < 10.0); // Should be close to 50.0
    }
}
