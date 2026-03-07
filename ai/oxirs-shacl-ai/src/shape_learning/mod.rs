//! Shape Learning Module for v0.3.0 Advanced SHACL AI
//!
//! Provides machine learning-based shape discovery and validation for RDF datasets.
//! This module implements statistical mining, pattern detection, constraint learning
//! from examples, and AI-assisted constraint validation.
//!
//! # Architecture
//!
//! ```text
//! shape_learning/
//! ├── mod.rs              - Module declarations and re-exports
//! ├── shape_miner.rs      - Statistical shape mining from RDF data
//! ├── pattern_detector.rs - Graph structural pattern detection
//! ├── constraint_learner.rs - Learn SHACL constraints from positive/negative examples
//! └── shape_validator_ai.rs - AI-assisted constraint validation
//! ```

pub mod constraint_learner;
pub mod pattern_detector;
pub mod shape_miner;
pub mod shape_validator_ai;

// Re-export all public types for ergonomic access
pub use constraint_learner::{
    ConstraintLearner, ConstraintLearningReport, ConstraintLearningStats, LearnedConstraintResult,
};
pub use pattern_detector::{
    DetectedPattern, GraphPattern, PatternDetectionConfig, PatternDetectionReport, PatternDetector,
    PatternKind,
};
pub use shape_miner::{
    MinedShape, NodeKind, PropertyConstraint, ShapeMiner, ShapeMinerConfig, ShapeMiningReport,
    ShapeMiningStats,
};
pub use shape_validator_ai::{
    AiValidationReport, ConstraintValidationScore, ShapeValidatorAi, ShapeValidatorAiConfig,
    ValidationFinding, ValidationFindingKind,
};
