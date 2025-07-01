//! Main error handling engine

use crate::error_handling::{
    classification::ErrorClassifier,
    config::ErrorHandlingConfig,
    impact::ErrorImpactAssessor,
    prevention::PreventionStrategyGenerator,
    repair::RepairSuggestionEngine,
};

/// Intelligent error handling system for SHACL validation
#[derive(Debug)]
pub struct IntelligentErrorHandler {
    /// Error classifier for categorizing validation errors
    error_classifier: ErrorClassifier,

    /// Repair suggestion engine
    repair_engine: RepairSuggestionEngine,

    /// Error impact assessor
    impact_assessor: ErrorImpactAssessor,

    /// Prevention strategy generator
    prevention_generator: PreventionStrategyGenerator,

    /// Configuration
    config: ErrorHandlingConfig,
}

impl IntelligentErrorHandler {
    /// Create a new intelligent error handler
    pub fn new() -> Self {
        Self::with_config(ErrorHandlingConfig::default())
    }

    /// Create a new intelligent error handler with custom configuration
    pub fn with_config(config: ErrorHandlingConfig) -> Self {
        Self {
            error_classifier: ErrorClassifier::new(),
            repair_engine: RepairSuggestionEngine::new(),
            impact_assessor: ErrorImpactAssessor::new(),
            prevention_generator: PreventionStrategyGenerator::new(),
            config,
        }
    }
}

impl Default for IntelligentErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}