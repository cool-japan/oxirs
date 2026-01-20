//! Core types for learned indexes

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type for learned index operations
pub type LearnedIndexResult<T> = std::result::Result<T, LearnedIndexError>;

/// Errors for learned index operations
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum LearnedIndexError {
    #[error("Model not trained")]
    ModelNotTrained,

    #[error("Training failed: {message}")]
    TrainingFailed { message: String },

    #[error("Prediction out of bounds: predicted={predicted}, actual_size={actual_size}")]
    PredictionOutOfBounds {
        predicted: usize,
        actual_size: usize,
    },

    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Insufficient data: need at least {min_required}, got {actual}")]
    InsufficientData { min_required: usize, actual: usize },

    #[error("Internal error: {message}")]
    InternalError { message: String },
}

/// Prediction bounds for error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionBounds {
    /// Predicted position
    pub predicted: usize,

    /// Lower bound (min error)
    pub lower_bound: usize,

    /// Upper bound (max error)
    pub upper_bound: usize,

    /// Error magnitude
    pub error_magnitude: usize,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

impl PredictionBounds {
    /// Create new prediction bounds
    pub fn new(predicted: usize, lower: usize, upper: usize, confidence: f32) -> Self {
        let error_magnitude = upper.saturating_sub(lower);
        Self {
            predicted,
            lower_bound: lower,
            upper_bound: upper,
            error_magnitude,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Get search range for binary search fallback
    pub fn search_range(&self) -> std::ops::Range<usize> {
        self.lower_bound..self.upper_bound
    }
}

/// Training example for learned index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input features (e.g., vector or key)
    pub features: Vec<f32>,

    /// Target position in sorted order
    pub target_position: usize,

    /// Optional weight for importance sampling
    pub weight: f32,
}

impl TrainingExample {
    pub fn new(features: Vec<f32>, target_position: usize) -> Self {
        Self {
            features,
            target_position,
            weight: 1.0,
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

/// Statistics for learned index performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStatistics {
    /// Total predictions made
    pub total_predictions: usize,

    /// Predictions within error bounds
    pub predictions_within_bounds: usize,

    /// Average prediction error
    pub avg_prediction_error: f64,

    /// Max prediction error observed
    pub max_prediction_error: usize,

    /// Average search range size
    pub avg_search_range_size: f64,

    /// Total lookups performed
    pub total_lookups: usize,

    /// Average lookup time (microseconds)
    pub avg_lookup_time_us: f64,
}

impl IndexStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_prediction(&mut self, predicted: usize, actual: usize, within_bounds: bool) {
        self.total_predictions += 1;

        if within_bounds {
            self.predictions_within_bounds += 1;
        }

        let error = predicted.abs_diff(actual);

        // Update average error
        let n = self.total_predictions as f64;
        self.avg_prediction_error = (self.avg_prediction_error * (n - 1.0) + error as f64) / n;

        // Update max error
        if error > self.max_prediction_error {
            self.max_prediction_error = error;
        }
    }

    pub fn record_lookup(&mut self, search_range_size: usize, lookup_time_us: f64) {
        self.total_lookups += 1;

        let n = self.total_lookups as f64;
        self.avg_search_range_size =
            (self.avg_search_range_size * (n - 1.0) + search_range_size as f64) / n;

        self.avg_lookup_time_us = (self.avg_lookup_time_us * (n - 1.0) + lookup_time_us) / n;
    }

    pub fn accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            self.predictions_within_bounds as f64 / self.total_predictions as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_bounds() {
        let bounds = PredictionBounds::new(100, 95, 105, 0.9);

        assert_eq!(bounds.predicted, 100);
        assert_eq!(bounds.lower_bound, 95);
        assert_eq!(bounds.upper_bound, 105);
        assert_eq!(bounds.error_magnitude, 10);
        assert_eq!(bounds.confidence, 0.9);

        let range = bounds.search_range();
        assert_eq!(range.start, 95);
        assert_eq!(range.end, 105);
    }

    #[test]
    fn test_training_example() {
        let example = TrainingExample::new(vec![1.0, 2.0, 3.0], 42).with_weight(0.5);

        assert_eq!(example.features, vec![1.0, 2.0, 3.0]);
        assert_eq!(example.target_position, 42);
        assert_eq!(example.weight, 0.5);
    }

    #[test]
    fn test_index_statistics() {
        let mut stats = IndexStatistics::new();

        stats.record_prediction(100, 102, true);
        stats.record_prediction(200, 195, true);
        stats.record_prediction(300, 250, false);

        assert_eq!(stats.total_predictions, 3);
        assert_eq!(stats.predictions_within_bounds, 2);
        assert!(stats.avg_prediction_error > 0.0);
        assert_eq!(stats.max_prediction_error, 50);
        assert!((stats.accuracy() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_lookup_statistics() {
        let mut stats = IndexStatistics::new();

        stats.record_lookup(10, 5.0);
        stats.record_lookup(20, 10.0);

        assert_eq!(stats.total_lookups, 2);
        assert_eq!(stats.avg_search_range_size, 15.0);
        assert_eq!(stats.avg_lookup_time_us, 7.5);
    }
}
