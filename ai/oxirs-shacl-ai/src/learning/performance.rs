//! Performance metrics and analysis for shape learning

use crate::patterns::{Pattern, PatternType};
use serde::{Deserialize, Serialize};

/// Performance metrics for learning efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPerformanceMetrics {
    pub success_rate: f64,
    pub constraint_density: f64,
    pub temporal_constraint_ratio: f64,
    pub shapes_per_class: f64,
    pub training_accuracy: f64,
}

/// Statistics for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStatistics {
    pub total_patterns: usize,
    pub datatype_patterns: usize,
    pub cardinality_patterns: usize,
    pub temporal_patterns: usize,
    pub range_patterns: usize,
    pub diversity_score: f64,
}

/// Analyze pattern statistics for learning optimization
pub fn analyze_pattern_statistics(patterns: &[Pattern]) -> PatternStatistics {
    let mut datatype_patterns = 0;
    let mut cardinality_patterns = 0;
    let mut temporal_patterns = 0;
    let mut range_patterns = 0;

    for pattern in patterns {
        match pattern.pattern_type() {
            PatternType::Datatype => {
                datatype_patterns += 1;
            }
            PatternType::Cardinality => {
                cardinality_patterns += 1;
            }
            PatternType::Temporal => {
                temporal_patterns += 1;
            }
            PatternType::Range => {
                range_patterns += 1;
            }
            _ => {}
        }
    }

    let total_patterns = patterns.len();
    let diversity_score = if total_patterns > 0 {
        let unique_types = [
            datatype_patterns,
            cardinality_patterns,
            temporal_patterns,
            range_patterns,
        ]
        .iter()
        .filter(|&&count| count > 0)
        .count() as f64;
        unique_types / 4.0 // 4 main pattern types
    } else {
        0.0
    };

    PatternStatistics {
        total_patterns,
        datatype_patterns,
        cardinality_patterns,
        temporal_patterns,
        range_patterns,
        diversity_score,
    }
}

/// Calculate performance metrics for a learning session
pub fn calculate_performance_metrics(
    success_rate: f64,
    total_constraints: usize,
    temporal_constraints: usize,
    total_shapes: usize,
    total_classes: usize,
    training_accuracy: f64,
) -> LearningPerformanceMetrics {
    let constraint_density = if total_shapes > 0 {
        total_constraints as f64 / total_shapes as f64
    } else {
        0.0
    };

    let temporal_constraint_ratio = if total_constraints > 0 {
        temporal_constraints as f64 / total_constraints as f64
    } else {
        0.0
    };

    let shapes_per_class = if total_classes > 0 {
        total_shapes as f64 / total_classes as f64
    } else {
        0.0
    };

    LearningPerformanceMetrics {
        success_rate,
        constraint_density,
        temporal_constraint_ratio,
        shapes_per_class,
        training_accuracy,
    }
}
