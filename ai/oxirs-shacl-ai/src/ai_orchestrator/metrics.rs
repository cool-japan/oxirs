//! Metrics and statistics for AI orchestration
//!
//! This module contains all metrics, statistics, and performance tracking
//! capabilities for the AI orchestration system.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Comprehensive orchestrator statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AiOrchestratorStats {
    /// Total number of learning sessions
    pub total_learning_sessions: usize,
    /// Total training time across all models
    pub total_training_time: Duration,
    /// Number of shapes learned
    pub shapes_learned: usize,
    /// Number of constraints discovered
    pub constraints_discovered: usize,
    /// Average model accuracy
    pub average_accuracy: f64,
    /// Total validation predictions made
    pub total_predictions: usize,
    /// Prediction accuracy rate
    pub prediction_accuracy: f64,
    /// Quality assessment sessions
    pub quality_assessments: usize,
    /// Optimization sessions
    pub optimization_sessions: usize,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Memory usage statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Average memory usage in MB
    pub average_memory_mb: f64,
    /// Current memory usage in MB
    pub current_memory_mb: f64,
    /// Memory allocations count
    pub allocations_count: usize,
}

/// Performance metrics for orchestration
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average response time in milliseconds
    pub average_response_time_ms: f64,
    /// Throughput (operations per second)
    pub throughput_ops_per_sec: f64,
    /// Success rate (percentage)
    pub success_rate_percent: f64,
    /// Error rate (percentage)
    pub error_rate_percent: f64,
    /// Resource utilization metrics
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_utilization_percent: f64,
    /// Memory utilization percentage
    pub memory_utilization_percent: f64,
    /// Disk I/O utilization
    pub disk_io_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
}

/// Learning performance statistics
#[derive(Debug, Default, Clone)]
pub struct LearningPerformanceStats {
    /// Model-specific performance
    pub model_performance: std::collections::HashMap<String, ModelPerformance>,
    /// Overall learning metrics
    pub overall_metrics: OverallLearningMetrics,
    /// Confidence distribution
    pub confidence_distribution: ConfidenceDistribution,
}

/// Performance metrics for individual models
#[derive(Debug, Default, Clone)]
pub struct ModelPerformance {
    /// Training accuracy
    pub training_accuracy: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Training time
    pub training_time: Duration,
    /// Inference time per sample
    pub inference_time_per_sample: Duration,
    /// Memory usage during training
    pub training_memory_mb: f64,
    /// Model complexity score
    pub complexity_score: f64,
}

/// Overall learning metrics across all models
#[derive(Debug, Default, Clone)]
pub struct OverallLearningMetrics {
    /// Best achieved accuracy
    pub best_accuracy: f64,
    /// Average accuracy across models
    pub average_accuracy: f64,
    /// Standard deviation of accuracies
    pub accuracy_std_dev: f64,
    /// Total training time
    pub total_training_time: Duration,
    /// Number of successful training runs
    pub successful_training_runs: usize,
    /// Number of failed training runs
    pub failed_training_runs: usize,
}

/// Distribution of confidence scores
#[derive(Debug, Default, Clone)]
pub struct ConfidenceDistribution {
    /// Histogram bins for confidence scores
    pub confidence_bins: Vec<f64>,
    /// Count of predictions in each bin
    pub bin_counts: Vec<usize>,
    /// Mean confidence score
    pub mean_confidence: f64,
    /// Standard deviation of confidence scores
    pub confidence_std_dev: f64,
}

impl AiOrchestratorStats {
    /// Create new statistics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Update learning session statistics
    pub fn update_learning_session(
        &mut self,
        session_duration: Duration,
        shapes_count: usize,
        constraints_count: usize,
    ) {
        self.total_learning_sessions += 1;
        self.total_training_time += session_duration;
        self.shapes_learned += shapes_count;
        self.constraints_discovered += constraints_count;
    }

    /// Update prediction statistics
    pub fn update_prediction_stats(
        &mut self,
        predictions_count: usize,
        correct_predictions: usize,
    ) {
        self.total_predictions += predictions_count;
        if self.total_predictions > 0 {
            let total_correct = (self.prediction_accuracy
                * (self.total_predictions - predictions_count) as f64)
                as usize
                + correct_predictions;
            self.prediction_accuracy = total_correct as f64 / self.total_predictions as f64;
        }
    }

    /// Update memory statistics
    pub fn update_memory_stats(&mut self, current_memory_mb: f64) {
        self.memory_stats.current_memory_mb = current_memory_mb;

        if current_memory_mb > self.memory_stats.peak_memory_mb {
            self.memory_stats.peak_memory_mb = current_memory_mb;
        }

        // Update running average
        let total_measurements = self.memory_stats.allocations_count + 1;
        self.memory_stats.average_memory_mb = (self.memory_stats.average_memory_mb
            * self.memory_stats.allocations_count as f64
            + current_memory_mb)
            / total_measurements as f64;

        self.memory_stats.allocations_count = total_measurements;
    }

    /// Get learning efficiency score
    pub fn get_learning_efficiency(&self) -> f64 {
        if self.total_learning_sessions == 0 {
            return 0.0;
        }

        let shapes_per_session = self.shapes_learned as f64 / self.total_learning_sessions as f64;
        let avg_session_time_secs =
            self.total_training_time.as_secs_f64() / self.total_learning_sessions as f64;

        if avg_session_time_secs > 0.0 {
            shapes_per_session / avg_session_time_secs
        } else {
            0.0
        }
    }

    /// Generate summary report
    pub fn generate_summary(&self) -> String {
        format!(
            "AI Orchestrator Summary:\n\
             - Learning Sessions: {}\n\
             - Shapes Learned: {}\n\
             - Constraints Discovered: {}\n\
             - Average Accuracy: {:.2}%\n\
             - Prediction Accuracy: {:.2}%\n\
             - Learning Efficiency: {:.4} shapes/sec\n\
             - Peak Memory Usage: {:.2} MB",
            self.total_learning_sessions,
            self.shapes_learned,
            self.constraints_discovered,
            self.average_accuracy * 100.0,
            self.prediction_accuracy * 100.0,
            self.get_learning_efficiency(),
            self.memory_stats.peak_memory_mb
        )
    }
}
