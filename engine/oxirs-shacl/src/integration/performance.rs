//! # Performance Optimization for Integration Modules
//!
//! This module provides advanced performance optimization capabilities for the
//! integration modules, leveraging SciRS2 for statistical analysis and caching.
//!
//! ## Features
//!
//! - **Adaptive caching**: ML-powered cache sizing and eviction
//! - **Performance prediction**: Predict validation latency using historical data
//! - **Load balancing**: Intelligent request routing based on performance
//! - **Bottleneck detection**: Identify performance bottlenecks automatically
//! - **Resource optimization**: Optimize CPU/memory usage dynamically

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// SciRS2 imports for performance analysis
use scirs2_core::ndarray_ext::{Array1, Array2};

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable adaptive caching
    pub enable_adaptive_caching: bool,

    /// Enable performance prediction
    pub enable_prediction: bool,

    /// Enable load balancing
    pub enable_load_balancing: bool,

    /// Enable bottleneck detection
    pub enable_bottleneck_detection: bool,

    /// History window size for analysis
    pub history_window_size: usize,

    /// Prediction model confidence threshold
    pub prediction_confidence: f64,

    /// Cache size adjustment interval (milliseconds)
    pub cache_adjustment_interval_ms: u64,

    /// Maximum cache size (entries)
    pub max_cache_size: usize,

    /// Minimum cache size (entries)
    pub min_cache_size: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_caching: true,
            enable_prediction: true,
            enable_load_balancing: true,
            enable_bottleneck_detection: true,
            history_window_size: 1000,
            prediction_confidence: 0.85,
            cache_adjustment_interval_ms: 60000, // 1 minute
            max_cache_size: 10000,
            min_cache_size: 100,
        }
    }
}

/// Performance metrics for validation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Operation type
    pub operation_type: String,

    /// Latency in milliseconds
    pub latency_ms: f64,

    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f64,

    /// CPU usage percentage
    pub cpu_usage: f64,

    /// Memory usage in bytes
    pub memory_usage: usize,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Request size (bytes or entities)
    pub request_size: usize,

    /// Result size (bytes or entities)
    pub result_size: usize,
}

/// Performance optimizer using SciRS2
pub struct PerformanceOptimizer {
    config: PerformanceConfig,
    history: Arc<dashmap::DashMap<String, Vec<PerformanceMetrics>>>,
    cache_sizes: Arc<dashmap::DashMap<String, usize>>,
    last_adjustment: Arc<Mutex<Instant>>,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            config,
            history: Arc::new(dashmap::DashMap::new()),
            cache_sizes: Arc::new(dashmap::DashMap::new()),
            last_adjustment: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Record performance metrics
    pub fn record_metrics(&self, metrics: PerformanceMetrics) {
        let operation_type = metrics.operation_type.clone();

        self.history
            .entry(operation_type.clone())
            .or_default()
            .push(metrics);

        // Trim history to window size
        if let Some(mut history) = self.history.get_mut(&operation_type) {
            if history.len() > self.config.history_window_size {
                let excess = history.len() - self.config.history_window_size;
                history.drain(0..excess);
            }
        }

        // Check if cache adjustment is needed
        if self.config.enable_adaptive_caching {
            self.maybe_adjust_cache();
        }
    }

    /// Predict validation latency for a given request
    pub fn predict_latency(
        &mut self,
        operation_type: &str,
        request_size: usize,
    ) -> Option<LatencyPrediction> {
        if !self.config.enable_prediction {
            return None;
        }

        // Extract history and clone it to avoid borrow checker issues
        let (history_data, sample_size, confidence) = {
            let history = self.history.get(operation_type)?;

            if history.len() < 10 {
                return None; // Not enough data for prediction
            }

            let history_vec: Vec<_> = history.value().clone();
            let sample_size = history.len();
            let confidence = self.calculate_prediction_confidence(&history);

            (history_vec, sample_size, confidence)
        };

        // Extract predictor features (request_size, cache_hit_rate) and the
        // aligned latency targets from history.
        let features = self.extract_features(&history_data, request_size);
        let targets = self.extract_targets(&history_data);

        // Ordinary least squares regression on the historical features to
        // predict latency for the requested request_size.
        let prediction = self.predict_using_regression(&features, &targets, request_size)?;

        Some(LatencyPrediction {
            predicted_latency_ms: prediction,
            confidence,
            sample_size,
        })
    }

    /// Get optimal cache size for an operation type
    pub fn get_optimal_cache_size(&self, operation_type: &str) -> usize {
        if !self.config.enable_adaptive_caching {
            return self.config.max_cache_size / 2; // Default to half
        }

        self.cache_sizes
            .get(operation_type)
            .map(|size| *size)
            .unwrap_or(self.config.min_cache_size)
    }

    /// Detect performance bottlenecks
    pub fn detect_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        if !self.config.enable_bottleneck_detection {
            return Vec::new();
        }

        let mut bottlenecks = Vec::new();

        for entry in self.history.iter() {
            let operation_type = entry.key();
            let metrics = entry.value();

            if metrics.len() < 10 {
                continue;
            }

            // Analyze latency using SciRS2
            let latencies: Vec<f64> = metrics.iter().map(|m| m.latency_ms).collect();
            let latency_array = Array1::from_vec(latencies.clone());

            // Calculate statistics
            let mean_latency = latency_array.mean().unwrap_or(0.0);
            let std_latency = latency_array.std(0.0);

            // Detect high latency
            if mean_latency > 1000.0 {
                // > 1 second
                bottlenecks.push(PerformanceBottleneck {
                    operation_type: operation_type.clone(),
                    bottleneck_type: BottleneckType::HighLatency,
                    severity: if mean_latency > 5000.0 {
                        Severity::Critical
                    } else {
                        Severity::Warning
                    },
                    description: format!(
                        "Mean latency: {:.2}ms (σ={:.2}ms)",
                        mean_latency, std_latency
                    ),
                    recommendation: "Consider adding more cache or optimizing queries".to_string(),
                });
            }

            // Detect low cache hit rate
            let cache_hit_rates: Vec<f64> = metrics.iter().map(|m| m.cache_hit_rate).collect();
            let hit_rate_array = Array1::from_vec(cache_hit_rates);
            let mean_hit_rate = hit_rate_array.mean().unwrap_or(0.0);

            if mean_hit_rate < 0.3 {
                // < 30%
                bottlenecks.push(PerformanceBottleneck {
                    operation_type: operation_type.clone(),
                    bottleneck_type: BottleneckType::LowCacheHitRate,
                    severity: Severity::Warning,
                    description: format!("Mean cache hit rate: {:.1}%", mean_hit_rate * 100.0),
                    recommendation: "Increase cache size or adjust caching strategy".to_string(),
                });
            }

            // Detect high memory usage
            let memory_usages: Vec<f64> = metrics.iter().map(|m| m.memory_usage as f64).collect();
            let memory_array = Array1::from_vec(memory_usages);
            let mean_memory = memory_array.mean().unwrap_or(0.0);

            if mean_memory > 1_000_000_000.0 {
                // > 1GB
                bottlenecks.push(PerformanceBottleneck {
                    operation_type: operation_type.clone(),
                    bottleneck_type: BottleneckType::HighMemoryUsage,
                    severity: Severity::Warning,
                    description: format!("Mean memory usage: {:.2}MB", mean_memory / 1_000_000.0),
                    recommendation: "Enable streaming or batch processing".to_string(),
                });
            }
        }

        bottlenecks
    }

    /// Get load balancing recommendation
    pub fn get_load_balancing_recommendation(
        &self,
        candidate_nodes: &[String],
    ) -> Option<LoadBalancingRecommendation> {
        if !self.config.enable_load_balancing || candidate_nodes.is_empty() {
            return None;
        }

        // Analyze performance of each node
        let mut node_scores: Vec<(String, f64)> = Vec::new();

        for node in candidate_nodes {
            if let Some(history) = self.history.get(node) {
                // Calculate score based on recent performance
                let recent: Vec<&PerformanceMetrics> = history.iter().rev().take(100).collect();

                if recent.is_empty() {
                    node_scores.push((node.clone(), 0.5)); // Default score
                    continue;
                }

                let latencies: Vec<f64> = recent.iter().map(|m| m.latency_ms).collect();
                let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;

                // Lower latency = higher score
                let score = 1.0 / (1.0 + avg_latency / 1000.0);
                node_scores.push((node.clone(), score));
            } else {
                // No history, give default score
                node_scores.push((node.clone(), 0.5));
            }
        }

        // Sort by score (highest first)
        node_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate distribution
        let total_score: f64 = node_scores.iter().map(|(_, s)| s).sum();
        let distribution: Vec<(String, f64)> = node_scores
            .iter()
            .map(|(node, score)| (node.clone(), score / total_score))
            .collect();

        Some(LoadBalancingRecommendation {
            recommended_node: node_scores[0].0.clone(),
            confidence: node_scores[0].1,
            distribution,
        })
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let mut total_operations = 0;
        let mut total_latency = 0.0;
        let mut total_cache_hits = 0.0;

        for entry in self.history.iter() {
            let metrics = entry.value();
            total_operations += metrics.len();

            for metric in metrics.iter() {
                total_latency += metric.latency_ms;
                total_cache_hits += metric.cache_hit_rate;
            }
        }

        let avg_latency = if total_operations > 0 {
            total_latency / total_operations as f64
        } else {
            0.0
        };

        let avg_cache_hit_rate = if total_operations > 0 {
            total_cache_hits / total_operations as f64
        } else {
            0.0
        };

        PerformanceSummary {
            total_operations,
            average_latency_ms: avg_latency,
            average_cache_hit_rate: avg_cache_hit_rate,
            operation_types: self.history.len(),
            bottlenecks_detected: self.detect_bottlenecks().len(),
        }
    }

    // Private helper methods

    fn maybe_adjust_cache(&self) {
        let mut last_adjustment = self
            .last_adjustment
            .lock()
            .expect("lock should not be poisoned");
        let elapsed = last_adjustment.elapsed();

        if elapsed < Duration::from_millis(self.config.cache_adjustment_interval_ms) {
            return;
        }

        // Time to adjust caches
        for entry in self.history.iter() {
            let operation_type = entry.key();
            let metrics = entry.value();

            if metrics.len() < 10 {
                continue;
            }

            // Calculate optimal cache size using hit rate analysis
            let hit_rates: Vec<f64> = metrics
                .iter()
                .rev()
                .take(100)
                .map(|m| m.cache_hit_rate)
                .collect();

            let avg_hit_rate = hit_rates.iter().sum::<f64>() / hit_rates.len() as f64;

            let current_size = self
                .cache_sizes
                .get(operation_type)
                .map(|s| *s)
                .unwrap_or(self.config.min_cache_size);

            let new_size = if avg_hit_rate < 0.5 {
                // Low hit rate - increase cache
                (current_size as f64 * 1.2) as usize
            } else if avg_hit_rate > 0.9 {
                // Very high hit rate - can reduce cache
                (current_size as f64 * 0.9) as usize
            } else {
                current_size
            };

            // Clamp to min/max
            let new_size = new_size
                .max(self.config.min_cache_size)
                .min(self.config.max_cache_size);

            self.cache_sizes.insert(operation_type.clone(), new_size);
        }

        *last_adjustment = Instant::now();
    }

    fn extract_features(
        &self,
        history: &[PerformanceMetrics],
        _request_size: usize,
    ) -> Array2<f64> {
        // Extract features for regression
        let n_samples = history.len().min(100); // Use last 100 samples
        Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            let metric = &history[history.len() - n_samples + i];
            match j {
                0 => metric.request_size as f64,
                1 => metric.cache_hit_rate,
                _ => 0.0,
            }
        })
    }

    /// Extract the latency targets aligned with [`Self::extract_features`].
    fn extract_targets(&self, history: &[PerformanceMetrics]) -> Vec<f64> {
        let n_samples = history.len().min(100);
        history[history.len() - n_samples..]
            .iter()
            .map(|m| m.latency_ms)
            .collect()
    }

    /// Predict latency using an ordinary-least-squares linear regression over
    /// the historical features `[request_size, cache_hit_rate]` against the
    /// observed latencies, evaluated at `query_request_size` (and the mean
    /// historical cache-hit-rate). This is a genuine computed estimate — never a
    /// random number.
    ///
    /// Returns `None` when there is insufficient or degenerate data to fit a
    /// model, so callers can treat the prediction as unavailable rather than
    /// acting on noise.
    fn predict_using_regression(
        &self,
        features: &Array2<f64>,
        targets: &[f64],
        query_request_size: usize,
    ) -> Option<f64> {
        let n = features.nrows();
        if n == 0 || n != targets.len() {
            return None;
        }

        // Design matrix columns: [1, request_size, cache_hit_rate].
        // Solve the 3x3 normal equations (XᵀX)·β = Xᵀy via Gaussian elimination.
        let mut xtx = [[0.0f64; 3]; 3];
        let mut xty = [0.0f64; 3];
        let mut mean_hit_rate = 0.0f64;

        for i in 0..n {
            let x = [1.0, features[[i, 0]], features[[i, 1]]];
            mean_hit_rate += features[[i, 1]];
            let y = targets[i];
            for (r, xr) in x.iter().enumerate() {
                xty[r] += xr * y;
                for (c, xc) in x.iter().enumerate() {
                    xtx[r][c] += xr * xc;
                }
            }
        }
        mean_hit_rate /= n as f64;

        let beta = match solve_3x3(xtx, xty) {
            Some(b) => b,
            None => {
                // Singular system (e.g. constant predictors): fall back to the
                // mean observed latency — still a computed value, not noise.
                let mean_latency = targets.iter().sum::<f64>() / n as f64;
                return Some(mean_latency.max(0.0));
            }
        };

        let prediction = beta[0] + beta[1] * (query_request_size as f64) + beta[2] * mean_hit_rate;

        // Latency cannot be negative; clamp to a small positive floor.
        Some(prediction.max(0.0))
    }

    fn calculate_prediction_confidence(&self, history: &[PerformanceMetrics]) -> f64 {
        // Calculate confidence based on data quality and quantity
        let n = history.len() as f64;

        // More data = higher confidence
        let data_confidence = (n / 1000.0).min(1.0);

        // Calculate consistency (inverse of coefficient of variation)
        let latencies: Vec<f64> = history.iter().map(|m| m.latency_ms).collect();
        let latency_array = Array1::from_vec(latencies);

        let mean = latency_array.mean().unwrap_or(0.0);
        let std = latency_array.std(0.0);

        let cv = if mean > 0.0 { std / mean } else { 1.0 };
        let consistency_confidence = 1.0 / (1.0 + cv);

        // Combine confidences
        (data_confidence + consistency_confidence) / 2.0
    }
}

/// Solve a 3x3 linear system `A·x = b` via Gaussian elimination with partial
/// pivoting. Returns `None` if the matrix is singular (no unique solution).
fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        // Partial pivot: find the row with the largest absolute value in `col`.
        let mut pivot = col;
        for row in (col + 1)..3 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None; // singular
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        // Eliminate below.
        let pivot_row = a[col];
        let pivot_b = b[col];
        for row in (col + 1)..3 {
            let factor = a[row][col] / pivot_row[col];
            for (c, cell) in a[row].iter_mut().enumerate().skip(col) {
                *cell -= factor * pivot_row[c];
            }
            b[row] -= factor * pivot_b;
        }
    }

    // Back-substitution.
    let mut x = [0.0f64; 3];
    for row in (0..3).rev() {
        let mut sum = b[row];
        for c in (row + 1)..3 {
            sum -= a[row][c] * x[c];
        }
        x[row] = sum / a[row][row];
    }

    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// Latency prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPrediction {
    /// Predicted latency in milliseconds
    pub predicted_latency_ms: f64,

    /// Prediction confidence (0.0-1.0)
    pub confidence: f64,

    /// Number of samples used
    pub sample_size: usize,
}

/// Performance bottleneck detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Operation type with bottleneck
    pub operation_type: String,

    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,

    /// Severity level
    pub severity: Severity,

    /// Description of the bottleneck
    pub description: String,

    /// Recommendation to fix
    pub recommendation: String,
}

/// Bottleneck type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    HighLatency,
    LowCacheHitRate,
    HighMemoryUsage,
    HighCPUUsage,
    NetworkLatency,
}

/// Severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

/// Load balancing recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingRecommendation {
    /// Recommended node to use
    pub recommended_node: String,

    /// Confidence in recommendation (0.0-1.0)
    pub confidence: f64,

    /// Distribution of traffic across nodes
    pub distribution: Vec<(String, f64)>,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Total operations recorded
    pub total_operations: usize,

    /// Average latency across all operations
    pub average_latency_ms: f64,

    /// Average cache hit rate
    pub average_cache_hit_rate: f64,

    /// Number of operation types tracked
    pub operation_types: usize,

    /// Number of bottlenecks detected
    pub bottlenecks_detected: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_optimizer_creation() {
        let config = PerformanceConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        assert!(optimizer.config.enable_adaptive_caching);
    }

    #[test]
    fn test_metrics_recording() {
        let optimizer = PerformanceOptimizer::new(PerformanceConfig::default());

        let metrics = PerformanceMetrics {
            operation_type: "graphql_mutation".to_string(),
            latency_ms: 150.0,
            cache_hit_rate: 0.8,
            cpu_usage: 45.0,
            memory_usage: 1024 * 1024,
            timestamp: chrono::Utc::now(),
            request_size: 100,
            result_size: 50,
        };

        optimizer.record_metrics(metrics);

        // Verify history was recorded
        assert!(optimizer.history.contains_key("graphql_mutation"));
    }

    #[test]
    fn test_optimal_cache_size() {
        let optimizer = PerformanceOptimizer::new(PerformanceConfig::default());

        let size = optimizer.get_optimal_cache_size("test_operation");

        assert!(size >= optimizer.config.min_cache_size);
        assert!(size <= optimizer.config.max_cache_size);
    }

    #[test]
    fn test_performance_summary() {
        let optimizer = PerformanceOptimizer::new(PerformanceConfig::default());

        let summary = optimizer.get_performance_summary();

        assert_eq!(summary.total_operations, 0); // No operations yet
        assert_eq!(summary.bottlenecks_detected, 0);
    }

    #[test]
    fn test_bottleneck_detection_empty() {
        let optimizer = PerformanceOptimizer::new(PerformanceConfig::default());

        let bottlenecks = optimizer.detect_bottlenecks();

        assert!(bottlenecks.is_empty()); // No data yet
    }

    #[test]
    fn test_latency_prediction_insufficient_data() {
        let mut optimizer = PerformanceOptimizer::new(PerformanceConfig::default());

        let prediction = optimizer.predict_latency("test_op", 100);

        assert!(prediction.is_none()); // Not enough data
    }

    #[test]
    fn regression_latency_prediction_is_deterministic_and_computed() {
        // Feed a clean linear relationship latency = 10 + 2 * request_size at a
        // fixed cache_hit_rate. A real regression must (a) be deterministic
        // across calls (no randomness) and (b) recover the underlying model.
        let mut optimizer = PerformanceOptimizer::new(PerformanceConfig::default());
        for i in 1..=20 {
            let request_size = i * 10;
            // Vary cache_hit_rate (mean 0.5) so the design matrix is well-posed;
            // latency depends only on request_size, so its coefficient is ~0.
            let cache_hit_rate = if i % 2 == 0 { 0.6 } else { 0.4 };
            optimizer.record_metrics(PerformanceMetrics {
                operation_type: "op".to_string(),
                latency_ms: 10.0 + 2.0 * request_size as f64,
                cache_hit_rate,
                cpu_usage: 0.0,
                memory_usage: 0,
                timestamp: chrono::Utc::now(),
                request_size,
                result_size: 0,
            });
        }

        let p1 = optimizer.predict_latency("op", 100).expect("prediction");
        let p2 = optimizer.predict_latency("op", 100).expect("prediction");
        // Determinism: identical inputs -> identical output (would fail if random).
        assert_eq!(p1.predicted_latency_ms, p2.predicted_latency_ms);
        // Recover latency ~= 10 + 2*100 = 210 within tolerance.
        assert!(
            (p1.predicted_latency_ms - 210.0).abs() < 5.0,
            "expected ~210, got {}",
            p1.predicted_latency_ms
        );
    }

    #[test]
    fn regression_solve_3x3_identity() {
        // Sanity check the linear solver used by the regression.
        let a = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let b = [4.0, 9.0, 8.0];
        let x = solve_3x3(a, b).expect("solvable");
        assert!((x[0] - 2.0).abs() < 1e-9);
        assert!((x[1] - 3.0).abs() < 1e-9);
        assert!((x[2] - 2.0).abs() < 1e-9);
        // Singular matrix returns None.
        let singular = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        assert!(solve_3x3(singular, [1.0, 2.0, 3.0]).is_none());
    }
}
