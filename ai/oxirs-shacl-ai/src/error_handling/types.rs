//! Core types for error handling

use oxirs_shacl::{Severity, ValidationViolation as Violation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Error types in taxonomy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    ConstraintViolation,
    DataTypeError,
    CardinalityError,
    RangeError,
    PatternError,
    Other(String),
}

/// Error impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorImpact {
    pub business_impact: f64,
    pub data_quality_impact: f64,
    pub performance_impact: f64,
    pub user_experience_impact: f64,
}

/// Error priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ErrorPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Resolution difficulty estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionDifficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Comprehensive error classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorClassificationResult {
    /// Primary error type from taxonomy
    pub error_type: ErrorType,

    /// Detailed error subtype
    pub error_subtype: String,

    /// Severity classification
    pub severity: ErrorSeverity,

    /// Impact assessment
    pub impact: ErrorImpact,

    /// Priority assignment
    pub priority: ErrorPriority,

    /// Resolution difficulty estimate
    pub resolution_difficulty: ResolutionDifficulty,

    /// Business criticality assessment
    pub business_criticality: f64,

    /// Classification confidence
    pub confidence: f64,

    /// Affected entities count
    pub affected_entities: usize,

    /// Recommended actions
    pub recommended_actions: Vec<String>,
}
