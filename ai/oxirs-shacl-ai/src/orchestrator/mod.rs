//! AI Orchestrator for Comprehensive Shape Learning
//!
//! This module orchestrates all AI capabilities to provide intelligent,
//! comprehensive SHACL shape learning and validation optimization.

pub mod config;
pub mod core;
pub mod data_analysis;
pub mod model_selection;
pub mod types;

// Re-export main types and functions
pub use config::{PerformanceRequirements, ShapeOrchestratorConfig};
pub use core::ShapeOrchestrator;
pub use data_analysis::{
    DataAnalyzer, DataCharacteristics, DataStatistics, ModelPerformanceMetrics,
};
pub use model_selection::{AdvancedModelSelector, ModelSelectionResult, ModelSelectionStrategy};
pub use types::{
    ComprehensiveLearningResult, ConfidentShape, ImplementationEffort, LearningMetadata, ModelType,
    OptimizationRecommendation, OrchestrationMetrics, PredictiveInsights, QualityAnalysis,
    ShapeOrchestratorStats, SpecializedModel,
};
