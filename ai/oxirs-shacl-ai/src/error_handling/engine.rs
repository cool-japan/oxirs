//! Main error handling engine

use crate::error_handling::{
    classification::ErrorClassifier,
    config::ErrorHandlingConfig,
    impact::ErrorImpactAssessor,
    prevention::PreventionStrategyGenerator,
    repair::RepairSuggestionEngine,
    types::{ErrorType, RepairSuggestion, RepairType, SmartErrorAnalysis},
};
use crate::{Result, ShaclAiError};
use oxirs_core::Store;
use oxirs_shacl::{Shape, ValidationReport};
use std::collections::HashMap;

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

    /// Get the current configuration
    pub fn config(&self) -> &ErrorHandlingConfig {
        &self.config
    }

    /// Process validation errors with intelligent analysis and repair suggestions
    pub fn process_validation_errors(
        &self,
        validation_report: &ValidationReport,
        store: &dyn Store,
        shapes: &[Shape],
    ) -> Result<SmartErrorAnalysis> {
        tracing::debug!(
            "Processing {} validation violations for intelligent analysis",
            validation_report.violations.len()
        );

        // Classify the primary error type based on violations
        let error_classification = self.classify_primary_error_type(validation_report)?;

        // Perform root cause analysis
        let root_cause_analysis = self.analyze_root_causes(validation_report, store, shapes)?;

        // Generate intelligent repair suggestions
        let fix_suggestions = self.generate_repair_suggestions(validation_report, store, shapes)?;

        // Calculate confidence score based on analysis depth and data quality
        let confidence_score =
            self.calculate_confidence_score(validation_report, &fix_suggestions)?;

        // Find similar cases from historical data
        let similar_cases = self.find_similar_cases(validation_report)?;

        Ok(SmartErrorAnalysis {
            error_classification,
            root_cause_analysis,
            fix_suggestions,
            confidence_score,
            similar_cases,
        })
    }

    /// Classify the primary error type from validation violations
    fn classify_primary_error_type(
        &self,
        validation_report: &ValidationReport,
    ) -> Result<ErrorType> {
        if validation_report.violations.is_empty() {
            return Ok(ErrorType::default());
        }

        // Analyze violation patterns to determine primary error type
        let mut cardinality_count = 0;
        let mut datatype_count = 0;
        let mut pattern_count = 0;
        let mut range_count = 0;

        for violation in &validation_report.violations {
            let message = violation
                .message()
                .as_ref()
                .map(|m| m.to_lowercase())
                .unwrap_or_default();
            if message.contains("cardinality") || message.contains("min") || message.contains("max")
            {
                cardinality_count += 1;
            } else if message.contains("datatype") || message.contains("type") {
                datatype_count += 1;
            } else if message.contains("pattern") || message.contains("regex") {
                pattern_count += 1;
            } else if message.contains("range") || message.contains("value") {
                range_count += 1;
            }
        }

        // Return the most frequent error type
        let counts = [
            cardinality_count,
            datatype_count,
            pattern_count,
            range_count,
        ];
        let max_count = counts.iter().max().unwrap_or(&0);

        if *max_count == 0 {
            Ok(ErrorType::ConstraintViolation)
        } else if cardinality_count == *max_count {
            Ok(ErrorType::CardinalityError)
        } else if datatype_count == *max_count {
            Ok(ErrorType::DataTypeError)
        } else if pattern_count == *max_count {
            Ok(ErrorType::PatternError)
        } else {
            Ok(ErrorType::RangeError)
        }
    }

    /// Analyze root causes of validation failures
    fn analyze_root_causes(
        &self,
        validation_report: &ValidationReport,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<Vec<String>> {
        let mut root_causes = Vec::new();

        // Analyze common root cause patterns
        if validation_report.violations.len() > 10 {
            root_causes.push(
                "High volume of violations suggests systematic data quality issues".to_string(),
            );
        }

        // Check for missing required properties
        let missing_property_violations = validation_report
            .violations
            .iter()
            .filter(|v| {
                let message = v
                    .message()
                    .as_ref()
                    .map(|m| m.to_lowercase())
                    .unwrap_or_default();
                message.contains("required") || message.contains("missing")
            })
            .count();

        if missing_property_violations > 0 {
            root_causes.push(format!(
                "Missing required properties detected in {} violations - likely incomplete data ingestion",
                missing_property_violations
            ));
        }

        // Check for type mismatches
        let type_mismatch_violations = validation_report
            .violations
            .iter()
            .filter(|v| {
                let message = v
                    .message()
                    .as_ref()
                    .map(|m| m.to_lowercase())
                    .unwrap_or_default();
                message.contains("datatype") || message.contains("type")
            })
            .count();

        if type_mismatch_violations > 0 {
            root_causes.push(format!(
                "Data type mismatches in {} violations - possible data transformation errors",
                type_mismatch_violations
            ));
        }

        // Check for pattern violations
        let pattern_violations = validation_report
            .violations
            .iter()
            .filter(|v| {
                let message = v
                    .message()
                    .as_ref()
                    .map(|m| m.to_lowercase())
                    .unwrap_or_default();
                message.contains("pattern")
            })
            .count();

        if pattern_violations > 0 {
            root_causes.push(format!(
                "Pattern constraint violations in {} cases - data format inconsistencies",
                pattern_violations
            ));
        }

        if root_causes.is_empty() {
            root_causes
                .push("General constraint violations - manual review recommended".to_string());
        }

        Ok(root_causes)
    }

    /// Generate intelligent repair suggestions
    fn generate_repair_suggestions(
        &self,
        validation_report: &ValidationReport,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<Vec<RepairSuggestion>> {
        let mut suggestions = Vec::new();

        // Generate suggestions based on violation patterns
        for violation in &validation_report.violations {
            let message = violation
                .message()
                .as_ref()
                .map(|m| m.to_lowercase())
                .unwrap_or_default();

            if message.contains("required") || message.contains("missing") {
                suggestions.push(RepairSuggestion {
                    repair_type: RepairType::DataCorrection,
                    description: "Add missing required properties to the data".to_string(),
                    confidence: 0.85,
                    effort_estimate: 0.3,
                    success_probability: 0.9,
                    automated: true,
                });
            }

            if message.contains("datatype") || message.contains("type") {
                suggestions.push(RepairSuggestion {
                    repair_type: RepairType::DataTypeConversion,
                    description: "Convert data values to expected data types".to_string(),
                    confidence: 0.8,
                    effort_estimate: 0.4,
                    success_probability: 0.85,
                    automated: true,
                });
            }

            if message.contains("cardinality") {
                suggestions.push(RepairSuggestion {
                    repair_type: RepairType::CardinalityAdjustment,
                    description: "Adjust property cardinalities to match actual data patterns"
                        .to_string(),
                    confidence: 0.75,
                    effort_estimate: 0.6,
                    success_probability: 0.8,
                    automated: false,
                });
            }

            if message.contains("pattern") {
                suggestions.push(RepairSuggestion {
                    repair_type: RepairType::PatternFix,
                    description: "Update data format to match required patterns".to_string(),
                    confidence: 0.7,
                    effort_estimate: 0.5,
                    success_probability: 0.75,
                    automated: true,
                });
            }
        }

        // Add general suggestions if no specific ones found
        if suggestions.is_empty() {
            suggestions.push(RepairSuggestion {
                repair_type: RepairType::ValidationRuleUpdate,
                description:
                    "Review and update validation rules to better match data characteristics"
                        .to_string(),
                confidence: 0.6,
                effort_estimate: 0.8,
                success_probability: 0.7,
                automated: false,
            });
        }

        // Remove duplicates and sort by confidence
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.dedup_by(|a, b| a.repair_type.to_string() == b.repair_type.to_string());

        Ok(suggestions)
    }

    /// Calculate confidence score for the analysis
    fn calculate_confidence_score(
        &self,
        validation_report: &ValidationReport,
        fix_suggestions: &[RepairSuggestion],
    ) -> Result<f64> {
        let mut confidence = 0.5; // Base confidence

        // Increase confidence based on number of violations (more data = better analysis)
        let violation_count = validation_report.violations.len();
        confidence += (violation_count as f64 * 0.01).min(0.3);

        // Increase confidence based on fix suggestion quality
        if !fix_suggestions.is_empty() {
            let avg_suggestion_confidence: f64 =
                fix_suggestions.iter().map(|s| s.confidence).sum::<f64>()
                    / fix_suggestions.len() as f64;
            confidence += avg_suggestion_confidence * 0.2;
        }

        // Increase confidence if violations are consistent (similar patterns)
        let unique_messages: std::collections::HashSet<_> = validation_report
            .violations
            .iter()
            .map(|v| v.message())
            .collect();
        if unique_messages.len() < violation_count / 2 {
            confidence += 0.1; // Patterns are consistent
        }

        Ok(confidence.min(1.0))
    }

    /// Find similar cases from historical data
    fn find_similar_cases(&self, validation_report: &ValidationReport) -> Result<Vec<String>> {
        // For now, return mock similar cases
        // In a real implementation, this would query a database of historical validation reports
        let mut similar_cases = Vec::new();

        if !validation_report.violations.is_empty() {
            similar_cases.push(
                "Case #1234: Similar constraint violations resolved by data type corrections"
                    .to_string(),
            );
            similar_cases.push(
                "Case #5678: Comparable validation patterns fixed with schema updates".to_string(),
            );
        }

        Ok(similar_cases)
    }
}

impl Default for IntelligentErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}
