//! Quality Assessment Module for SHACL-AI
//!
//! This module provides comprehensive quality assessment capabilities including
//! basic quality assessment, extended multi-dimensional analysis, and AI-powered metrics.

pub mod ai_metrics;
pub mod core;
pub mod enhancement;
pub mod extended_dimensions;
pub mod issue_detection;

// Re-export core quality types
pub use core::{
    ImplementationEffort, QualityAlgorithms, QualityAssessmentData, QualityAssessor, QualityConfig,
    QualityDimension, QualityIssue, QualityIssueCategory, QualityIssueSeverity,
    QualityRecommendation, QualityRecommendationCategory, QualityReport, QualityScores,
    QualityStatistics, QualityThresholds, QualityTrainingData, QualityTrend,
    RecommendationPriority,
};

// Re-export extended quality assessment types
pub use extended_dimensions::{
    ConceptCoherence, ContextualQualityAssessment, CorrelationAnalysis, DistributionAnalysis,
    EntropyCalculation, ExtendedQualityConfig, ExtendedQualityDimensionsAssessor,
    InformationContentMeasure, IntrinsicQualityAssessment, KnowledgeCompleteness,
    LogicalConsistency, MultiDimensionalQualityAssessment, OutlierDetectionResult,
    QualityDimensionResult, RedundancyAssessment, RelationshipValidity, RelevanceMethod,
    SemanticDensity, SemanticQualityMeasures, StatisticalQualityMeasures, TaxonomyConsistency,
};

// Re-export AI-powered quality metrics types
pub use ai_metrics::{
    AdvancedSemanticMetrics, AdvancedStatisticalMetrics, AiMetricsConfig, AiMetricsStatistics,
    AiQualityMetricsEngine, AiQualityMetricsResult, ComputationMetadata, ConfidenceScores,
    FeatureImportance, MachineLearningPredictions, QualityPredictions,
};

// Re-export quality issue detection types
pub use issue_detection::{
    AlertSeverity, AlertType, AnomalyDetectionResult, DegradationDetectionResult,
    IssueDetectionConfig, IssueDetectionResult, IssueDetectionStatistics, IssueRecommendation,
    IssueSummary, ProactiveAlert, QualityIssueDetector, QualitySnapshot, TrendDirection,
};

// Re-export quality enhancement types
pub use enhancement::{
    ActionStatus, AutomationModel, CostBenefitModel, DataEnhancementModel, EnhancementAction,
    EnhancementActionType, EnhancementCategory, EnhancementConfig, EnhancementRecommendation,
    EnhancementStatistics, EnhancementStrategy, ImpactPredictionModel,
    ImplementationEffort as EnhancementImplementationEffort, Priority, ProcessOptimizationModel,
    QualityEnhancementEngine, RecommendationModels,
};
