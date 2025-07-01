//! Tests for advanced error reporting functionality

use oxirs_shacl_ai::error_handling::{
    ActionableInsight, AdvancedErrorReport, ErrorHandlingConfig, ExecutiveSummary,
    IntelligentErrorHandler, SmartErrorAnalysis,
};
use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intelligent_error_handler_creation() {
        let handler = IntelligentErrorHandler::new();
        assert_eq!(handler.config.min_repair_confidence, 0.7);
        assert!(handler.config.enable_ml_classification);
        assert!(handler.config.enable_impact_assessment);
    }

    #[test]
    fn test_error_handler_with_custom_config() {
        let config = ErrorHandlingConfig {
            enable_ml_classification: false,
            min_repair_confidence: 0.9,
            max_repair_suggestions: 3,
            enable_impact_assessment: false,
            enable_prevention_strategies: false,
            critical_severity_threshold: 0.95,
            business_impact_weight: 0.5,
        };

        let handler = IntelligentErrorHandler::with_config(config.clone());
        assert_eq!(handler.config.min_repair_confidence, 0.9);
        assert!(!handler.config.enable_ml_classification);
        assert_eq!(handler.config.max_repair_suggestions, 3);
    }

    #[test]
    fn test_error_handler_default() {
        let handler = IntelligentErrorHandler::default();
        assert_eq!(handler.config.min_repair_confidence, 0.7);
        assert!(handler.config.enable_ml_classification);
    }

    #[test]
    fn test_config_default() {
        let config = ErrorHandlingConfig::default();
        assert_eq!(config.min_repair_confidence, 0.7);
        assert_eq!(config.max_repair_suggestions, 5);
        assert!(config.enable_ml_classification);
        assert!(config.enable_impact_assessment);
        assert!(config.enable_prevention_strategies);
        assert_eq!(config.critical_severity_threshold, 0.8);
        assert_eq!(config.business_impact_weight, 0.3);
    }

    // Note: More comprehensive tests would require mock validation reports
    // and proper integration with oxirs-shacl, but these basic tests
    // validate the core structure and configuration functionality
}
