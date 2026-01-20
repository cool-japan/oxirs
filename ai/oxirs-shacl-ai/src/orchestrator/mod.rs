//! AI Orchestrator for Comprehensive Shape Learning
//!
//! This module orchestrates all AI capabilities to provide intelligent,
//! comprehensive SHACL shape learning and validation optimization.

pub mod config;
pub mod data_analysis;
pub mod model_selection;
pub mod core;
pub mod types;

// Re-export main types and functions
pub use config::{AiOrchestratorConfig, PerformanceRequirements};
pub use data_analysis::{DataCharacteristics, DataStatistics, ModelPerformanceMetrics};
pub use model_selection::{AdvancedModelSelector, ModelSelectionStrategy, ModelSelectionResult};
pub use core::AiOrchestrator;
pub use types::{
    ComprehensiveLearningResult, ConfidentShape, QualityAnalysis, ImplementationEffort,
    ModelType, SpecializedModel, OptimizationRecommendation, PredictiveInsights,
    LearningMetadata, AiOrchestratorStats, OrchestrationMetrics,
};