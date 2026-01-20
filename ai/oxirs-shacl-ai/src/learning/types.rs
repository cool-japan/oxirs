//! Types and configurations for shape learning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ml::reinforcement::RLConfig;

/// Learning query result types
#[derive(Debug, Clone)]
pub enum LearningQueryResult {
    /// SELECT query results with variable bindings
    Select {
        variables: Vec<String>,
        bindings: Vec<HashMap<String, oxirs_core::model::Term>>,
    },
    /// ASK query boolean result
    Ask(bool),
    /// Empty result
    Empty,
}

/// Configuration for shape learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Enable automatic shape generation
    pub enable_shape_generation: bool,

    /// Minimum support threshold for patterns
    pub min_support: f64,

    /// Minimum confidence threshold for constraints
    pub min_confidence: f64,

    /// Maximum number of shapes to generate
    pub max_shapes: usize,

    /// Enable model training
    pub enable_training: bool,

    /// Learning algorithm parameters
    pub algorithm_params: HashMap<String, f64>,

    /// Enable reinforcement learning for constraint discovery optimization
    pub enable_reinforcement_learning: bool,

    /// Reinforcement learning configuration
    pub rl_config: Option<RLConfig>,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enable_shape_generation: true,
            min_support: 0.1,
            min_confidence: 0.8,
            max_shapes: 100,
            enable_training: true,
            algorithm_params: HashMap::new(),
            enable_reinforcement_learning: false,
            rl_config: None,
        }
    }
}

/// Learning statistics for tracking progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStatistics {
    /// Total number of shapes successfully learned
    pub total_shapes_learned: usize,

    /// Number of failed shape learning attempts
    pub failed_shapes: usize,

    /// Total constraints discovered
    pub total_constraints_discovered: usize,

    /// Number of temporal constraints discovered
    pub temporal_constraints_discovered: usize,

    /// Number of classes analyzed
    pub classes_analyzed: usize,

    /// Whether model has been trained
    pub model_trained: bool,

    /// Last training accuracy achieved
    pub last_training_accuracy: f64,
}

impl Default for LearningStatistics {
    fn default() -> Self {
        Self {
            total_shapes_learned: 0,
            failed_shapes: 0,
            total_constraints_discovered: 0,
            temporal_constraints_discovered: 0,
            classes_analyzed: 0,
            model_trained: false,
            last_training_accuracy: 0.0,
        }
    }
}

/// Training data for shape learning
#[derive(Debug, Clone)]
pub struct ShapeTrainingData {
    /// Input features for training
    pub features: Vec<Vec<f64>>,

    /// Target labels for training
    pub labels: Vec<String>,

    /// Validation features
    pub validation_features: Vec<Vec<f64>>,

    /// Validation labels
    pub validation_labels: Vec<String>,
}

/// Training examples for shape learning
#[derive(Debug, Clone)]
pub struct ShapeExample {
    /// Example class IRI
    pub class_iri: String,

    /// Associated properties with their frequencies
    pub properties: HashMap<String, f64>,

    /// Constraints that should be learned
    pub expected_constraints: Vec<String>,
}

/// Temporal pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemporalPatterns {
    /// Whether regular intervals are detected
    pub has_regular_intervals: bool,

    /// Whether seasonal patterns are detected
    pub has_seasonal_pattern: bool,

    /// Whether values are strictly increasing
    pub is_strictly_increasing: bool,

    /// Total number of temporal values
    pub total_values: usize,

    /// Date range in days
    pub date_range_days: i64,
}
