//! Shape learning and automatic constraint discovery
//!
//! This module implements AI-powered shape learning from RDF data, providing
//! automatic constraint discovery, pattern recognition, and shape optimization.

pub mod learner;
pub mod performance;
pub mod types;

// Re-export key types for convenience
pub use types::{
    LearningConfig, LearningQueryResult, LearningStatistics, ShapeExample, ShapeTrainingData,
    TemporalPatterns,
};

pub use learner::ShapeLearner;

pub use performance::{
    analyze_pattern_statistics, calculate_performance_metrics, LearningPerformanceMetrics,
    PatternStatistics,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_shape_learner_creation() {
        let learner = ShapeLearner::new();
        assert!(learner.config().enable_shape_generation);
        assert_eq!(learner.config().min_confidence, 0.8);
        assert_eq!(learner.config().max_shapes, 100);
    }

    #[test]
    fn test_learning_config() {
        let config = LearningConfig {
            enable_shape_generation: false,
            min_support: 0.2,
            min_confidence: 0.9,
            max_shapes: 50,
            enable_training: false,
            algorithm_params: HashMap::new(),
            enable_reinforcement_learning: false,
            rl_config: None,
        };

        let learner = ShapeLearner::with_config(config.clone());
        assert_eq!(learner.config().min_support, 0.2);
        assert_eq!(learner.config().min_confidence, 0.9);
        assert_eq!(learner.config().max_shapes, 50);
        assert!(!learner.config().enable_training);
    }

    #[test]
    fn test_learning_statistics() {
        let stats = LearningStatistics {
            total_shapes_learned: 5,
            failed_shapes: 1,
            total_constraints_discovered: 20,
            temporal_constraints_discovered: 3,
            classes_analyzed: 3,
            model_trained: true,
            last_training_accuracy: 0.95,
        };

        assert_eq!(stats.total_shapes_learned, 5);
        assert_eq!(stats.failed_shapes, 1);
        assert_eq!(stats.total_constraints_discovered, 20);
        assert_eq!(stats.temporal_constraints_discovered, 3);
        assert!(stats.model_trained);
        assert_eq!(stats.last_training_accuracy, 0.95);
    }

    #[test]
    fn test_temporal_patterns_detection() {
        let patterns = TemporalPatterns {
            has_regular_intervals: true,
            has_seasonal_pattern: false,
            is_strictly_increasing: true,
            total_values: 10,
            date_range_days: 365,
        };

        assert!(patterns.has_regular_intervals);
        assert!(!patterns.has_seasonal_pattern);
        assert!(patterns.is_strictly_increasing);
        assert_eq!(patterns.total_values, 10);
        assert_eq!(patterns.date_range_days, 365);
    }

    #[test]
    fn test_temporal_patterns_default() {
        let patterns = TemporalPatterns::default();

        assert!(!patterns.has_regular_intervals);
        assert!(!patterns.has_seasonal_pattern);
        assert!(!patterns.is_strictly_increasing);
        assert_eq!(patterns.total_values, 0);
        assert_eq!(patterns.date_range_days, 0);
    }
}
