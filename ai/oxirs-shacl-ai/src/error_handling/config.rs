//! Configuration for error handling system

use serde::{Deserialize, Serialize};

/// Configuration for error handling system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    /// Enable machine learning-based error classification
    pub enable_ml_classification: bool,

    /// Minimum confidence threshold for repair suggestions
    pub min_repair_confidence: f64,

    /// Maximum number of repair suggestions per error
    pub max_repair_suggestions: usize,

    /// Enable automated impact assessment
    pub enable_impact_assessment: bool,

    /// Enable prevention strategy generation
    pub enable_prevention_strategies: bool,

    /// Severity threshold for critical errors
    pub critical_severity_threshold: f64,

    /// Business impact weight in priority calculation
    pub business_impact_weight: f64,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            enable_ml_classification: true,
            min_repair_confidence: 0.7,
            max_repair_suggestions: 5,
            enable_impact_assessment: true,
            enable_prevention_strategies: true,
            critical_severity_threshold: 0.8,
            business_impact_weight: 0.3,
        }
    }
}
