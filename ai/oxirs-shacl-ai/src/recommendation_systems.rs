//! Recommendation Systems for Shape Improvements and Validation Strategy Optimization
//!
//! This module provides intelligent recommendations for:
//! - Shape improvement suggestions
//! - Validation strategy optimization
//! - Tool and process recommendations
//! - Training and investment guidance
//!
//! ## Module Layout (sibling files in `src/`)
//!
//! - `recommendation_systems_types` — all data types (structs, enums)
//! - `recommendation_systems_engine` — `RecommendationEngine` implementation
//! - `recommendation_systems_tests` — unit tests
//!
//! Sub-modules are declared in the crate's `lib.rs` (sibling-file pattern).

// Flat re-exports so callers can continue using `recommendation_systems::Foo`
pub use crate::recommendation_systems_engine::RecommendationEngine;
pub use crate::recommendation_systems_types::{
    EffectivenessTracker, EffortComplexity, EstimatedImpact, ImpactCategory, ImplementationEffort,
    ImplementationRoadmap, ImplementationStep, Milestone, Recommendation, RecommendationConfig,
    RecommendationModel, RecommendationOutcome, RecommendationPattern, RecommendationPriority,
    RecommendationRecord, RecommendationReport, RecommendationSummary, RecommendationType,
    ResourceRequirement, RiskLevel, RoadmapPhase, SuccessMetric, UserFeedback,
};
