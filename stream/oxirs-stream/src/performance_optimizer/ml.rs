//! Machine Learning Components for Performance Optimization
//!
//! This module provides ML-based performance prediction and optimization capabilities.
//! It includes predictive models for batch size optimization, resource allocation,
//! and throughput forecasting to achieve optimal streaming performance.

use crate::performance_optimizer::{ProcessingStats, TuningDecision};
use anyhow::Result;
use nalgebra::{DMatrix, DVector, Vector2};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

/// Performance metrics for ML training and prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Throughput in events per second
    pub throughput: f64,
    /// Average latency in milliseconds
    pub latency: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory usage in MB
    pub memory_usage: f64,
    /// Batch size used
    pub batch_size: f64,
    /// Number of parallel workers
    pub parallel_workers: f64,
    /// Error rate percentage
    pub error_rate: f64,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
}

impl PerformanceMetrics {
    /// Create metrics from processing stats
    pub fn from_stats(stats: &ProcessingStats, config_params: ConfigParams) -> Self {
        let total_events = stats.total_events.load(Ordering::Relaxed) as f64;
        let _total_time = stats.total_processing_time_ms.load(Ordering::Relaxed) as f64;
        let error_count = stats.error_count.load(Ordering::Relaxed) as f64;

        let throughput = stats.throughput_eps.load(Ordering::Relaxed) as f64;
        let latency = stats.avg_processing_time_ms.load(Ordering::Relaxed) as f64;
        let error_rate = if total_events > 0.0 {
            error_count / total_events * 100.0
        } else {
            0.0
        };

        Self {
            throughput,
            latency,
            cpu_utilization: config_params.cpu_utilization,
            memory_usage: config_params.memory_usage,
            batch_size: config_params.batch_size,
            parallel_workers: config_params.parallel_workers,
            error_rate,
            timestamp: SystemTime::now(),
        }
    }

    /// Convert to feature vector for ML models
    pub fn to_feature_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.batch_size,
            self.parallel_workers,
            self.cpu_utilization,
            self.memory_usage,
            self.error_rate,
        ])
    }

    /// Convert to target vector (throughput, latency)
    pub fn to_target_vector(&self) -> Vector2<f64> {
        Vector2::new(self.throughput, self.latency)
    }
}

/// Configuration parameters for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigParams {
    pub batch_size: f64,
    pub parallel_workers: f64,
    pub cpu_utilization: f64,
    pub memory_usage: f64,
}

/// Linear regression model for performance prediction
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    /// Model weights
    weights: DMatrix<f64>,
    /// Model bias
    bias: Vector2<f64>,
    /// Training history
    training_samples: usize,
    /// Model accuracy metrics
    r_squared: f64,
}

impl LinearRegressionModel {
    /// Create a new linear regression model
    pub fn new(feature_count: usize) -> Self {
        Self {
            weights: DMatrix::zeros(2, feature_count), // 2 outputs: throughput, latency
            bias: Vector2::zeros(),
            training_samples: 0,
            r_squared: 0.0,
        }
    }

    /// Train the model with new data
    pub fn train(&mut self, features: &[DVector<f64>], targets: &[Vector2<f64>]) -> Result<()> {
        if features.is_empty() || features.len() != targets.len() {
            return Err(anyhow::anyhow!("Invalid training data"));
        }

        let n_samples = features.len();
        let n_features = features[0].len();

        // Build feature matrix
        let mut x = DMatrix::zeros(n_samples, n_features + 1); // +1 for bias
        for (i, feature) in features.iter().enumerate() {
            for j in 0..n_features {
                x[(i, j)] = feature[j];
            }
            x[(i, n_features)] = 1.0; // Bias term
        }

        // Build target matrix
        let mut y = DMatrix::zeros(n_samples, 2);
        for (i, target) in targets.iter().enumerate() {
            y[(i, 0)] = target[0]; // Throughput
            y[(i, 1)] = target[1]; // Latency
        }

        // Solve normal equations: (X^T * X)^-1 * X^T * Y
        if let Some(xtx_inv) = (x.transpose() * &x).try_inverse() {
            let coefficients = xtx_inv * x.transpose() * y;

            // Extract weights and bias
            for i in 0..2 {
                for j in 0..n_features {
                    self.weights[(i, j)] = coefficients[(j, i)];
                }
                self.bias[i] = coefficients[(n_features, i)];
            }

            self.training_samples += n_samples;
            self.calculate_r_squared(features, targets);
        } else {
            return Err(anyhow::anyhow!("Failed to solve normal equations"));
        }

        Ok(())
    }

    /// Predict performance metrics
    pub fn predict(&self, features: &DVector<f64>) -> Vector2<f64> {
        if features.len() != self.weights.ncols() {
            return Vector2::zeros();
        }

        &self.weights * features + self.bias
    }

    /// Calculate R-squared for model accuracy
    fn calculate_r_squared(&mut self, features: &[DVector<f64>], targets: &[Vector2<f64>]) {
        if targets.is_empty() {
            return;
        }

        let mut total_variance = 0.0;
        let mut residual_variance = 0.0;

        // Calculate mean of targets
        let mean_throughput: f64 = targets.iter().map(|t| t[0]).sum::<f64>() / targets.len() as f64;
        let mean_latency: f64 = targets.iter().map(|t| t[1]).sum::<f64>() / targets.len() as f64;

        for (features, target) in features.iter().zip(targets.iter()) {
            let prediction = self.predict(features);

            // Residual sum of squares
            residual_variance +=
                (target[0] - prediction[0]).powi(2) + (target[1] - prediction[1]).powi(2);

            // Total sum of squares
            total_variance +=
                (target[0] - mean_throughput).powi(2) + (target[1] - mean_latency).powi(2);
        }

        self.r_squared = if total_variance > 0.0 {
            1.0 - (residual_variance / total_variance)
        } else {
            0.0
        };
    }

    /// Get model accuracy
    pub fn accuracy(&self) -> f64 {
        self.r_squared
    }

    /// Get training sample count
    pub fn sample_count(&self) -> usize {
        self.training_samples
    }
}

/// ML-based performance predictor
pub struct PerformancePredictor {
    /// Primary prediction model
    model: LinearRegressionModel,
    /// Training data history
    metrics_history: VecDeque<PerformanceMetrics>,
    /// Maximum history size
    max_history_size: usize,
    /// Minimum samples for training
    min_training_samples: usize,
    /// Last training time
    last_training: Option<SystemTime>,
    /// Training interval
    training_interval: Duration,
}

impl PerformancePredictor {
    /// Create a new performance predictor
    pub fn new() -> Self {
        Self {
            model: LinearRegressionModel::new(5), // 5 features
            metrics_history: VecDeque::new(),
            max_history_size: 1000,
            min_training_samples: 10,
            last_training: None,
            training_interval: Duration::from_secs(30),
        }
    }

    /// Add performance metrics for training
    pub fn add_metrics(&mut self, metrics: PerformanceMetrics) {
        self.metrics_history.push_back(metrics);

        // Maintain history size
        while self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }

        // Train model if conditions are met
        if self.should_retrain() {
            let _ = self.retrain();
        }
    }

    /// Check if model should be retrained
    fn should_retrain(&self) -> bool {
        self.metrics_history.len() >= self.min_training_samples
            && (self.last_training.is_none()
                || SystemTime::now()
                    .duration_since(self.last_training.unwrap())
                    .unwrap_or(Duration::from_secs(0))
                    >= self.training_interval)
    }

    /// Retrain the model with latest data
    fn retrain(&mut self) -> Result<()> {
        if self.metrics_history.len() < self.min_training_samples {
            return Ok(());
        }

        let features: Vec<DVector<f64>> = self
            .metrics_history
            .iter()
            .map(|m| m.to_feature_vector())
            .collect();

        let targets: Vec<Vector2<f64>> = self
            .metrics_history
            .iter()
            .map(|m| m.to_target_vector())
            .collect();

        self.model.train(&features, &targets)?;
        self.last_training = Some(SystemTime::now());

        Ok(())
    }

    /// Predict optimal configuration
    pub fn predict_optimal_config(&self, current_config: ConfigParams) -> Result<TuningDecision> {
        if self.model.sample_count() < self.min_training_samples {
            return Err(anyhow::anyhow!("Insufficient training data"));
        }

        let mut best_config = current_config.clone();
        let mut best_score = f64::NEG_INFINITY;

        // Test different configurations
        for batch_multiplier in [0.8, 0.9, 1.0, 1.1, 1.2] {
            for worker_delta in [-1.0, 0.0, 1.0] {
                let test_config = ConfigParams {
                    batch_size: (current_config.batch_size * batch_multiplier).clamp(10.0, 5000.0),
                    parallel_workers: (current_config.parallel_workers + worker_delta)
                        .clamp(1.0, 32.0),
                    cpu_utilization: current_config.cpu_utilization,
                    memory_usage: current_config.memory_usage,
                };

                let features = DVector::from_vec(vec![
                    test_config.batch_size,
                    test_config.parallel_workers,
                    test_config.cpu_utilization,
                    test_config.memory_usage,
                    0.0, // Assume no errors for prediction
                ]);

                let prediction = self.model.predict(&features);
                let throughput = prediction[0];
                let latency = prediction[1];

                // Score function: maximize throughput, minimize latency
                let score = throughput - (latency * 0.1); // Weight latency penalty

                if score > best_score {
                    best_score = score;
                    best_config = test_config;
                }
            }
        }

        // Generate tuning decision
        let decision = if best_config.batch_size != current_config.batch_size {
            TuningDecision {
                parameter: "batch_size".to_string(),
                old_value: current_config.batch_size,
                new_value: best_config.batch_size,
                reason: format!("ML prediction suggests optimal batch size for throughput improvement (model accuracy: {:.2})", self.model.accuracy()),
                expected_improvement: (best_score / 1000.0).clamp(0.0, 1.0),
                confidence: self.model.accuracy().max(0.5),
            }
        } else if best_config.parallel_workers != current_config.parallel_workers {
            TuningDecision {
                parameter: "parallel_workers".to_string(),
                old_value: current_config.parallel_workers,
                new_value: best_config.parallel_workers,
                reason: format!("ML prediction suggests optimal worker count for latency reduction (model accuracy: {:.2})", self.model.accuracy()),
                expected_improvement: (best_score / 1000.0).clamp(0.0, 1.0),
                confidence: self.model.accuracy().max(0.5),
            }
        } else {
            return Err(anyhow::anyhow!(
                "No beneficial configuration changes predicted"
            ));
        };

        Ok(decision)
    }

    /// Get prediction model statistics
    pub fn model_stats(&self) -> ModelStats {
        ModelStats {
            accuracy: self.model.accuracy(),
            training_samples: self.model.sample_count(),
            history_size: self.metrics_history.len(),
            last_training: self.last_training,
        }
    }
}

impl Default for PerformancePredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Model statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub accuracy: f64,
    pub training_samples: usize,
    pub history_size: usize,
    pub last_training: Option<SystemTime>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_creation() {
        let stats = ProcessingStats::default();
        let config = ConfigParams {
            batch_size: 100.0,
            parallel_workers: 4.0,
            cpu_utilization: 75.0,
            memory_usage: 512.0,
        };

        let metrics = PerformanceMetrics::from_stats(&stats, config);
        assert_eq!(metrics.batch_size, 100.0);
        assert_eq!(metrics.parallel_workers, 4.0);
    }

    #[test]
    fn test_linear_regression_model() {
        let mut model = LinearRegressionModel::new(2);

        // Use more diverse data points to ensure a well-conditioned matrix
        let features = vec![
            DVector::from_vec(vec![1.0, 1.0]),
            DVector::from_vec(vec![2.0, 4.0]),
            DVector::from_vec(vec![3.0, 1.0]),
            DVector::from_vec(vec![1.0, 3.0]),
            DVector::from_vec(vec![4.0, 2.0]),
        ];

        let targets = vec![
            Vector2::new(10.0, 5.0),
            Vector2::new(25.0, 12.0),
            Vector2::new(15.0, 8.0),
            Vector2::new(18.0, 9.0),
            Vector2::new(22.0, 11.0),
        ];

        assert!(model.train(&features, &targets).is_ok());
        assert!(model.sample_count() > 0);

        let prediction = model.predict(&DVector::from_vec(vec![4.0, 5.0]));
        assert!(prediction[0] > 0.0);
        assert!(prediction[1] > 0.0);
    }

    #[test]
    fn test_performance_predictor() {
        let mut predictor = PerformancePredictor::new();

        let config = ConfigParams {
            batch_size: 100.0,
            parallel_workers: 4.0,
            cpu_utilization: 75.0,
            memory_usage: 512.0,
        };

        // Add training data
        for _i in 0..15 {
            let stats = ProcessingStats::default();
            let metrics = PerformanceMetrics::from_stats(&stats, config.clone());
            predictor.add_metrics(metrics);
        }

        let stats = predictor.model_stats();
        assert!(stats.history_size > 0);
    }

    #[test]
    fn test_feature_vector_conversion() {
        let config = ConfigParams {
            batch_size: 100.0,
            parallel_workers: 4.0,
            cpu_utilization: 75.0,
            memory_usage: 512.0,
        };

        let stats = ProcessingStats::default();
        let metrics = PerformanceMetrics::from_stats(&stats, config);
        let features = metrics.to_feature_vector();

        assert_eq!(features.len(), 5);
        assert_eq!(features[0], 100.0); // batch_size
        assert_eq!(features[1], 4.0); // parallel_workers
    }
}
