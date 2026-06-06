//! Quality Assessment Core — facade module.
//!
//! This is a thin re-export facade. The implementation is split across:
//! - `core_types` — structs, enums, quality dimension types, scoring types
//! - `core_metrics` — quality metric computation, dimension scoring, aggregation
pub use super::core_metrics::QualityAssessor;
pub use super::core_types::{
    ImplementationEffort, QualityAlgorithms, QualityAssessmentData, QualityConfig,
    QualityDimension, QualityExample, QualityIssue, QualityIssueCategory, QualityIssueSeverity,
    QualityRecommendation, QualityRecommendationCategory, QualityReport, QualityScores,
    QualityStatistics, QualityThresholds, QualityTrainingData, QualityTrainingMetadata,
    QualityTrend, RecommendationPriority,
};
