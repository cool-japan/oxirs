//! Common types for AI orchestration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Advanced model selection strategies for dynamic orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    /// Performance-based selection using historical metrics
    PerformanceBased,
    /// Adaptive selection based on data characteristics
    DataAdaptive,
    /// Ensemble-weighted selection with dynamic weights
    EnsembleWeighted,
    /// Reinforcement learning-based selection
    ReinforcementLearning,
    /// Meta-learning approach for model selection
    MetaLearning,
    /// Hybrid approach combining multiple strategies
    Hybrid(Vec<ModelSelectionStrategy>),
}

impl Default for ModelSelectionStrategy {
    fn default() -> Self {
        ModelSelectionStrategy::PerformanceBased
    }
}

/// Data characteristics for adaptive model selection
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub graph_size: usize,
    pub complexity_score: f64,
    pub sparsity_ratio: f64,
    pub hierarchy_depth: u32,
    pub pattern_diversity: f64,
    pub semantic_richness: f64,
}

impl Default for DataCharacteristics {
    fn default() -> Self {
        Self {
            graph_size: 0,
            complexity_score: 0.0,
            sparsity_ratio: 0.5,
            hierarchy_depth: 1,
            pattern_diversity: 0.0,
            semantic_richness: 0.0,
        }
    }
}

/// Model performance metrics for selection
#[derive(Debug, Clone)]
pub struct ModelPerformanceMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub training_time: Duration,
    pub inference_time: Duration,
    pub memory_usage: f64,
    pub confidence_calibration: f64,
    pub robustness_score: f64,
}

/// Statistics for model selection
#[derive(Debug, Default, Clone)]
pub struct ModelSelectionStats {
    pub total_selections: usize,
    pub successful_selections: usize,
    pub average_performance_improvement: f64,
    pub selection_time: Duration,
}

/// Performance requirements for model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    pub min_accuracy: f64,
    pub max_inference_time: Duration,
    pub max_memory_usage: f64,
    pub min_confidence: f64,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            min_accuracy: 0.8,
            max_inference_time: Duration::from_millis(100),
            max_memory_usage: 1000.0,
            min_confidence: 0.7,
        }
    }
}

/// Model selection result
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    pub selected_models: Vec<String>,
    pub selection_confidence: f64,
    pub expected_performance: ModelPerformanceMetrics,
    pub selection_rationale: String,
}

/// Model type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// Decision tree-based models
    DecisionTree,
    /// Neural network models
    NeuralNetwork,
    /// Graph neural network models
    GraphNeuralNetwork,
    /// Association rule learning models
    AssociationRules,
    /// Ensemble models
    Ensemble,
    /// Pattern recognition models
    PatternRecognition,
    /// Specialized models for specific tasks
    Specialized(SpecializedModel),
}

/// Specialized model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecializedModel {
    /// Shape optimization models
    ShapeOptimization,
    /// Constraint discovery models
    ConstraintDiscovery,
    /// Quality assessment models
    QualityAssessment,
    /// Validation prediction models
    ValidationPrediction,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    /// Low effort implementations
    Low,
    /// Medium effort implementations
    Medium,
    /// High effort implementations
    High,
    /// Complex implementations requiring significant resources
    Complex,
}

/// Model performance comparison
#[derive(Debug, Clone)]
pub struct ModelPerformance {
    pub model_name: String,
    pub metrics: ModelPerformanceMetrics,
    pub confidence_distribution: ConfidenceDistribution,
    pub validation_history: Vec<f64>,
}

impl Default for ModelPerformance {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            metrics: ModelPerformanceMetrics {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                training_time: Duration::new(0, 0),
                inference_time: Duration::new(0, 0),
                memory_usage: 0.0,
                confidence_calibration: 0.0,
                robustness_score: 0.0,
            },
            confidence_distribution: ConfidenceDistribution::default(),
            validation_history: Vec::new(),
        }
    }
}

/// Confidence distribution for model predictions
#[derive(Debug, Clone)]
pub struct ConfidenceDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: HashMap<u8, f64>,
}

impl Default for ConfidenceDistribution {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 1.0,
            percentiles: HashMap::new(),
        }
    }
}