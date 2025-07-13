//! AI Orchestrator for Comprehensive Shape Learning
//!
//! This module orchestrates all AI capabilities to provide intelligent,
//! comprehensive SHACL shape learning and validation optimization.

pub mod config;
pub mod core;
pub mod metrics;
pub mod model_selection;
pub mod types;

// Re-export main types and functions
pub use config::{
    AiOrchestratorConfig, ModelSelectionStrategy,
    PerformanceRequirements as ConfigPerformanceRequirements,
};
pub use core::{
    AiOrchestrator, ComprehensiveLearningResult,
    LearningPerformanceStats as CoreLearningPerformanceStats, OptimizationRecommendation,
    PredictiveInsights,
};
pub use metrics::{
    AiOrchestratorStats, ConfidenceDistribution,
    LearningPerformanceStats as MetricsLearningPerformanceStats,
};
pub use model_selection::*;
pub use types::{
    AdaptiveLearningInsights, ConfidentShape, DataCharacteristics, LearningMetadata,
    ModelPerformanceMetrics, ModelSelectionResult, OrchestrationMetrics, PerformanceRequirements,
    QualityAnalysis, SelectedModel,
};
