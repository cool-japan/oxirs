//! Configuration types for advanced validation strategies

use serde::{Deserialize, Serialize};

/// Advanced validation strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedValidationConfig {
    /// Strategy selection approach
    pub strategy_selection: StrategySelectionApproach,

    /// Context awareness level
    pub context_awareness_level: ContextAwarenessLevel,

    /// Enable multi-objective optimization
    pub enable_multi_objective_optimization: bool,

    /// Enable adaptive constraint weighting
    pub enable_adaptive_constraint_weighting: bool,

    /// Enable semantic validation enhancement
    pub enable_semantic_enhancement: bool,

    /// Maximum strategies to consider simultaneously
    pub max_concurrent_strategies: usize,

    /// Strategy performance monitoring window (in validations)
    pub performance_window_size: usize,

    /// Minimum confidence threshold for strategy selection
    pub min_strategy_confidence: f64,

    /// Enable cross-validation for strategy effectiveness
    pub enable_cross_validation: bool,

    /// Dynamic strategy adaptation interval (in minutes)
    pub adaptation_interval_minutes: u64,

    /// Enable validation result explanation
    pub enable_result_explanation: bool,

    /// Enable uncertainty quantification
    pub enable_uncertainty_quantification: bool,
}

impl Default for AdvancedValidationConfig {
    fn default() -> Self {
        Self {
            strategy_selection: StrategySelectionApproach::AdaptiveMLBased,
            context_awareness_level: ContextAwarenessLevel::High,
            enable_multi_objective_optimization: true,
            enable_adaptive_constraint_weighting: true,
            enable_semantic_enhancement: true,
            max_concurrent_strategies: 4,
            performance_window_size: 1000,
            min_strategy_confidence: 0.75,
            enable_cross_validation: true,
            adaptation_interval_minutes: 15,
            enable_result_explanation: true,
            enable_uncertainty_quantification: true,
        }
    }
}

/// Strategy selection approaches
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategySelectionApproach {
    /// Static pre-configured strategy
    Static,
    /// Rule-based strategy selection
    RuleBased,
    /// Machine learning-based selection
    MLBased,
    /// Adaptive ML with continuous learning
    AdaptiveMLBased,
    /// Multi-armed bandit approach
    MultiArmedBandit,
    /// Ensemble of multiple strategies
    Ensemble,
    /// Quantum-enhanced strategy selection
    QuantumEnhanced,
}

/// Context awareness levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextAwarenessLevel {
    /// Basic context (data size, shape complexity)
    Basic,
    /// Medium context (includes domain knowledge)
    Medium,
    /// High context (includes semantic relationships)
    High,
    /// Ultra context (includes temporal and causal relationships)
    Ultra,
}

/// Computational complexity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputationalComplexity {
    Constant,
    Logarithmic,
    Linear,
    LogLinear,
    Quadratic,
    Cubic,
    Exponential,
}

/// Domain types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainType {
    Healthcare,
    Finance,
    Education,
    Government,
    Manufacturing,
    Retail,
    Energy,
    Transportation,
    Generic,
}

/// Priority levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PriorityLevel {
    Low,
    Normal,
    High,
    Critical,
}

/// Types of uncertainty sources
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintySourceType {
    DataQuality,
    ModelLimitations,
    ParameterUncertainty,
    StructuralUncertainty,
    ContextVariability,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    DataQualityImprovement,
    ShapeOptimization,
    PerformanceTuning,
    ValidationStrategy,
    ConstraintRefinement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnforcementLevel {
    Advisory,
    Warning,
    Error,
    Critical,
}