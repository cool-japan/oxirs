//! Common types for AI orchestration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Advanced model selection strategies for dynamic orchestration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    /// Performance-based selection using historical metrics
    #[default]
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
/// A shape inferred by the AI orchestrator with confidence
#[derive(Debug, Clone)]
pub struct ConfidentShape {
    /// Shape identifier IRI
    pub shape_iri: String,
    /// Confidence score in [0.0, 1.0]
    pub confidence: f64,
    /// Target class IRI
    pub target_class: String,
    /// SHACL constraint triples (predicate → constraint body)
    pub constraints: std::collections::HashMap<String, String>,
    /// Evidence triples that led to this shape
    pub evidence_count: usize,
}

/// Quality analysis for learned shapes
#[derive(Debug, Clone)]
pub struct QualityAnalysis {
    /// Average confidence across all shapes
    pub avg_confidence: f64,
    /// Fraction of shapes above the confidence threshold
    pub high_confidence_fraction: f64,
    /// Estimated false positive rate
    pub estimated_fpr: f64,
    /// Estimated false negative rate
    pub estimated_fnr: f64,
    /// Overall quality score in [0.0, 1.0]
    pub quality_score: f64,
    /// Human-readable quality verdict
    pub verdict: String,
}

/// Optimization recommendation from the orchestrator
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Short title for the recommendation
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Expected improvement in quality score
    pub expected_improvement: f64,
    /// Estimated effort to implement
    pub effort: ImplementationEffort,
    /// Priority (higher = more important)
    pub priority: u8,
}

/// Predictive insights from the orchestrator
#[derive(Debug, Clone)]
pub struct PredictiveInsights {
    /// Predicted number of validation violations on new data
    pub predicted_violations: usize,
    /// Confidence in the violation prediction
    pub violation_confidence: f64,
    /// Predicted shapes that are most likely to trigger violations
    pub high_risk_shapes: Vec<String>,
    /// Recommended actions to reduce violations
    pub recommendations: Vec<String>,
}

/// Metadata about a comprehensive learning run
#[derive(Debug, Clone)]
pub struct LearningMetadata {
    /// Duration of the learning process in milliseconds
    pub duration_ms: u64,
    /// Number of triples analysed
    pub triples_analysed: usize,
    /// Number of models used
    pub models_used: usize,
    /// Version of the orchestrator
    pub orchestrator_version: String,
}

/// Comprehensive result from the AI orchestrator
#[derive(Debug, Clone)]
pub struct ComprehensiveLearningResult {
    /// Learned shapes sorted by confidence
    pub shapes: Vec<ConfidentShape>,
    /// Quality analysis of the learned shapes
    pub quality: QualityAnalysis,
    /// Optimization recommendations for further improvement
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Predictive insights for future data
    pub predictions: PredictiveInsights,
    /// Metadata about this learning run
    pub metadata: LearningMetadata,
}

/// Aggregated statistics for the AI orchestrator
#[derive(Debug, Clone, Default)]
pub struct ShapeOrchestratorStats {
    /// Total number of learning runs executed
    pub total_runs: usize,
    /// Total shapes discovered across all runs
    pub total_shapes_discovered: usize,
    /// Average shapes per run
    pub avg_shapes_per_run: f64,
    /// Average quality score across all runs
    pub avg_quality_score: f64,
    /// Number of successful runs
    pub successful_runs: usize,
}

/// Metrics for a single orchestration run
#[derive(Debug, Clone)]
pub struct OrchestrationMetrics {
    /// Shape count discovered in this run
    pub shape_count: usize,
    /// Average confidence of discovered shapes
    pub avg_confidence: f64,
    /// Overall quality score
    pub quality_score: f64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Models used in this run
    pub models_used: Vec<String>,
}
