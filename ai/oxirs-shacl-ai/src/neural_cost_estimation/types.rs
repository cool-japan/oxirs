//! Core data types for neural cost estimation

use crate::{
    ml::{GraphData, ModelError, ModelMetrics},
    neural_patterns::{NeuralPattern, NeuralPatternRecognizer},
    neural_transformer_pattern_integration::{
        NeuralTransformerConfig, NeuralTransformerPatternIntegration,
    },
    quantum_enhanced_pattern_optimizer::{QuantumEnhancedPatternOptimizer, QuantumOptimizerConfig},
    realtime_adaptive_query_optimizer::{
        AdaptiveOptimizerConfig, OptimizationPlanType, PerformanceMetrics, QueryPerformanceRecord,
    },
    Result, ShaclAiError,
};

use ndarray::{Array1, Array2, Array3, Array4, Axis};
use oxirs_core::{
    model::{Term, Variable},
    query::{
        algebra::{AlgebraTriplePattern, TermPattern as AlgebraTermPattern},
        pattern_optimizer::{
            IndexStats, IndexType, OptimizedPatternPlan, PatternOptimizer, PatternStrategy,
        },
    },
    OxirsError, Store,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::config::*;

/// Cost prediction result
#[derive(Debug, Clone)]
pub struct CostPrediction {
    pub estimated_cost: f64,
    pub execution_time: Duration,
    pub resource_usage: ResourceUsage,
    pub uncertainty: f64,
    pub confidence: f64,
    pub contributing_factors: Vec<ContributingFactor>,
}

/// Resource usage estimation
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_io: f64,
    pub network_io: f64,
    pub cache_usage: f64,
}

/// Contributing factor to cost prediction
#[derive(Debug, Clone)]
pub struct ContributingFactor {
    pub factor_type: FactorType,
    pub importance: f64,
    pub description: String,
}

/// Types of contributing factors
#[derive(Debug, Clone)]
pub enum FactorType {
    PatternComplexity,
    JoinCardinality,
    IndexEfficiency,
    DataSize,
    SystemLoad,
    CacheState,
    LayerActivation(usize),
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStatistics {
    pub total_epochs: usize,
    pub current_loss: f64,
    pub current_accuracy: f64,
    pub average_loss: f64,
    pub average_accuracy: f64,
    pub convergence_rate: f64,
}

impl Default for TrainingStatistics {
    fn default() -> Self {
        Self {
            total_epochs: 0,
            current_loss: 0.0,
            current_accuracy: 0.0,
            average_loss: 0.0,
            average_accuracy: 0.0,
            convergence_rate: 0.0,
        }
    }
}

/// Training record
#[derive(Debug, Clone)]
pub struct TrainingRecord {
    pub epoch: usize,
    pub loss: f64,
    pub accuracy: f64,
    pub learning_rate: f64,
    pub timestamp: SystemTime,
}

/// Neural network layer types
#[derive(Debug, Clone)]
pub enum LayerType {
    Dense,
    Residual,
    Attention,
    Dropout,
}

/// Optimizer types
#[derive(Debug, Clone)]
pub enum OptimizerType {
    SGD,
    SGDMomentum,
    Adam,
    AdaGrad,
    RMSprop,
}

/// Adam optimizer parameters
#[derive(Debug)]
pub struct AdamParams {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub m: Vec<Array2<f64>>, // First moment estimates
    pub v: Vec<Array2<f64>>, // Second moment estimates
    pub t: usize,            // Time step
}

/// Pattern structure features
#[derive(Debug, Clone)]
pub struct PatternStructureFeatures {
    pub pattern_count: f64,
    pub variable_count: f64,
    pub constant_count: f64,
    pub predicate_variety: f64,
    pub pattern_depth: f64,
    pub structural_complexity: f64,
}

/// Index usage statistics
#[derive(Debug, Clone)]
pub struct IndexUsageStats {
    pub usage_frequency: f64,
    pub average_performance: f64,
    pub selectivity_distribution: Array1<f64>,
    pub cache_hit_rate: f64,
}

/// Join complexity features
#[derive(Debug, Clone)]
pub struct JoinComplexityFeatures {
    pub join_count: f64,
    pub join_cardinality: f64,
    pub join_selectivity: f64,
    pub cross_product_potential: f64,
    pub join_order_complexity: f64,
}

/// Resource types
#[derive(Debug, Clone)]
pub enum ResourceType {
    CPU,
    Memory,
    Disk,
    Network,
    Cache,
}

/// Resource monitor
#[derive(Debug)]
pub struct ResourceMonitor {
    pub resource_type: ResourceType,
    pub current_usage: f64,
    pub average_usage: f64,
    pub peak_usage: f64,
}

/// Model types for ensemble
#[derive(Debug, Clone)]
pub enum ModelType {
    DeepNeural,
    TreeBased,
    LinearRegression,
    SupportVector,
    Gaussian,
}

/// Runtime statistics for neural cost estimation
#[derive(Debug, Clone)]
pub struct NeuralCostEstimationStats {
    pub total_predictions: usize,
    pub average_prediction_time: Duration,
    pub accuracy_score: f64,
    pub uncertainty_score: f64,
    pub cache_hit_rate: f64,
    pub model_update_count: usize,
    pub total_training_time: Duration,
}

impl Default for NeuralCostEstimationStats {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            average_prediction_time: Duration::from_millis(0),
            accuracy_score: 0.0,
            uncertainty_score: 0.0,
            cache_hit_rate: 0.0,
            model_update_count: 0,
            total_training_time: Duration::from_millis(0),
        }
    }
}