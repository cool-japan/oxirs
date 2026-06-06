//! Machine Learning-Driven Query Optimization — facade module
//!
//! This module re-exports the full public API from the split sub-modules.

pub use crate::ml_optimizer_planner::MLOptimizer;
pub use crate::ml_optimizer_types::{
    AnomalyDetection, AnomalyType, CacheRecommendation, CachingStrategy, JoinOrderAlternative,
    JoinOrderOptimization, LinearRegressionModel, MLConfig, MLStatistics, NeuralNetworkModel,
    PerformanceOutcome, QueryFeatures, SourceAlternative, SourceSelectionPrediction,
    TrainingSample,
};
