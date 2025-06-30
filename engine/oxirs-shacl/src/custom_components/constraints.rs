//! Custom constraint implementations and composition logic
//!
//! This module provides the core constraint types for custom components,
//! including individual constraints and composite constraint composition.

use crate::{
    constraints::{
        ConstraintContext, ConstraintEvaluationResult, ConstraintEvaluator, ConstraintValidator,
    },
    sparql::SparqlConstraint,
    ConstraintComponentId, Result, Severity, ShaclError,
};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::performance::{ComponentExecutionContext, ExecutionMetrics};
use super::security::SecurityViolation;

/// Custom constraint implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomConstraint {
    /// Component that created this constraint
    pub component_id: ConstraintComponentId,
    /// Configuration parameters
    pub parameters: HashMap<String, Term>,
    /// Optional SPARQL query for validation
    pub sparql_query: Option<String>,
    /// Custom validation function name
    pub validation_function: Option<String>,
    /// Error message template
    pub message_template: Option<String>,
}

/// Component execution result
#[derive(Debug, Clone)]
pub struct ComponentExecutionResult {
    /// Constraint evaluation result
    pub constraint_result: ConstraintEvaluationResult,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    /// Security violations (if any)
    pub security_violations: Vec<SecurityViolation>,
}

/// Composite constraint that combines multiple components
#[derive(Debug, Clone)]
pub struct CompositeConstraint {
    /// Component IDs that make up this composite
    pub component_ids: Vec<ConstraintComponentId>,
    /// Individual constraints
    pub constraints: Vec<CustomConstraint>,
    /// How to compose the constraints
    pub composition_type: CompositionType,
}

/// Types of constraint composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionType {
    /// All constraints must be satisfied (logical AND)
    And,
    /// At least one constraint must be satisfied (logical OR)
    Or,
    /// Exactly one constraint must be satisfied (logical XOR)
    Xor,
    /// Custom composition logic
    Custom(String),
}

impl ConstraintValidator for CustomConstraint {
    fn validate(&self) -> Result<()> {
        // Basic validation - check required parameters exist
        Ok(())
    }
}

impl ConstraintEvaluator for CustomConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // If this is a SPARQL-based constraint, use SPARQL evaluation
        if let Some(query) = &self.sparql_query {
            let sparql_constraint = SparqlConstraint {
                query: query.clone(),
                prefixes: None,
                message: self.message_template.clone(),
                severity: Some(Severity::Violation),
                construct_query: None,
            };

            return sparql_constraint.evaluate(store, context);
        }

        // Otherwise, delegate to custom validation logic based on component type
        match self.component_id.as_str() {
            "ex:RegexConstraintComponent" => self.evaluate_regex_constraint(context),
            "ex:RangeConstraintComponent" => self.evaluate_range_constraint(context),
            "ex:UrlValidationComponent" => self.evaluate_url_constraint(context),
            "ex:EmailValidationComponent" => self.evaluate_email_constraint(context),
            _ => {
                // Unknown component type
                Ok(ConstraintEvaluationResult::error(format!(
                    "Unknown custom constraint component: {}",
                    self.component_id.as_str()
                )))
            }
        }
    }
}

impl CustomConstraint {
    /// Evaluate regular expression constraint
    pub fn evaluate_regex_constraint(
        &self,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let pattern = self
            .parameters
            .get("pattern")
            .and_then(|t| match t {
                Term::Literal(lit) => Some(lit.value()),
                _ => None,
            })
            .ok_or_else(|| {
                ShaclError::ConstraintValidation("Pattern parameter required".to_string())
            })?;

        let regex = regex::Regex::new(pattern).map_err(|e| {
            ShaclError::ConstraintValidation(format!("Invalid regex pattern: {}", e))
        })?;

        for value in &context.values {
            if let Term::Literal(lit) = value {
                if !regex.is_match(lit.value()) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value '{}' does not match pattern '{}'",
                            lit.value(),
                            pattern
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Regex constraint can only be applied to literals".to_string()),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Evaluate range constraint
    pub fn evaluate_range_constraint(
        &self,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let min_value = self.parameters.get("minValue");
        let max_value = self.parameters.get("maxValue");

        if min_value.is_none() && max_value.is_none() {
            return Err(ShaclError::ConstraintValidation(
                "At least one of minValue or maxValue must be specified".to_string(),
            ));
        }

        for value in &context.values {
            if let Term::Literal(lit) = value {
                // Try to parse as number
                if let Ok(num) = lit.value().parse::<f64>() {
                    if let Some(Term::Literal(min_lit)) = min_value {
                        if let Ok(min_num) = min_lit.value().parse::<f64>() {
                            if num < min_num {
                                return Ok(ConstraintEvaluationResult::violated(
                                    Some(value.clone()),
                                    Some(format!("Value {} is less than minimum {}", num, min_num)),
                                ));
                            }
                        }
                    }

                    if let Some(Term::Literal(max_lit)) = max_value {
                        if let Ok(max_num) = max_lit.value().parse::<f64>() {
                            if num > max_num {
                                return Ok(ConstraintEvaluationResult::violated(
                                    Some(value.clone()),
                                    Some(format!(
                                        "Value {} is greater than maximum {}",
                                        num, max_num
                                    )),
                                ));
                            }
                        }
                    }
                } else {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("Range constraint requires numeric values".to_string()),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Evaluate URL validation constraint
    pub fn evaluate_url_constraint(
        &self,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            match value {
                Term::NamedNode(node) => {
                    let url_str = node.as_str();
                    if !self.is_valid_url(url_str) {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!("'{}' is not a valid URL", url_str)),
                        ));
                    }
                }
                Term::Literal(lit) => {
                    let url_str = lit.value();
                    if !self.is_valid_url(url_str) {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!("'{}' is not a valid URL", url_str)),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("URL validation can only be applied to IRIs or literals".to_string()),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Evaluate email validation constraint
    pub fn evaluate_email_constraint(
        &self,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // Simple email regex pattern
        let email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$";
        let regex = regex::Regex::new(email_pattern).unwrap();

        for value in &context.values {
            if let Term::Literal(lit) = value {
                let email = lit.value();
                if !regex.is_match(email) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!("'{}' is not a valid email address", email)),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Email validation can only be applied to literals".to_string()),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Simple URL validation
    fn is_valid_url(&self, url: &str) -> bool {
        // Basic URL validation - in practice you might want to use a proper URL parsing library
        url.starts_with("http://") || url.starts_with("https://") || url.starts_with("ftp://")
    }
}

impl CompositeConstraint {
    /// Evaluate the composite constraint based on composition type
    pub fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let mut results = Vec::new();

        // Evaluate all individual constraints
        for constraint in &self.constraints {
            let result = constraint.evaluate(store, context)?;
            results.push(result);
        }

        // Apply composition logic
        match self.composition_type {
            CompositionType::And => {
                // All constraints must be satisfied
                for result in &results {
                    if result.is_violated() {
                        return Ok(result.clone());
                    }
                }
                Ok(ConstraintEvaluationResult::satisfied())
            }
            CompositionType::Or => {
                // At least one constraint must be satisfied
                let mut any_satisfied = false;
                let mut first_violation = None;

                for result in &results {
                    if result.is_satisfied() {
                        any_satisfied = true;
                        break;
                    } else if first_violation.is_none() {
                        first_violation = Some(result.clone());
                    }
                }

                if any_satisfied {
                    Ok(ConstraintEvaluationResult::satisfied())
                } else {
                    Ok(first_violation.unwrap_or_else(|| {
                        ConstraintEvaluationResult::violated(
                            None,
                            Some("No constraints satisfied in OR composition".to_string()),
                        )
                    }))
                }
            }
            CompositionType::Xor => {
                // Exactly one constraint must be satisfied
                let satisfied_count = results.iter().filter(|r| r.is_satisfied()).count();

                if satisfied_count == 1 {
                    Ok(ConstraintEvaluationResult::satisfied())
                } else if satisfied_count == 0 {
                    Ok(ConstraintEvaluationResult::violated(
                        None,
                        Some("No constraints satisfied in XOR composition".to_string()),
                    ))
                } else {
                    Ok(ConstraintEvaluationResult::violated(
                        None,
                        Some(format!(
                            "Multiple constraints ({}) satisfied in XOR composition",
                            satisfied_count
                        )),
                    ))
                }
            }
            CompositionType::Custom(ref custom_logic) => {
                // For custom composition, implement specific logic based on the custom_logic string
                match custom_logic.as_str() {
                    "majority" => {
                        // Majority vote - more than half must be satisfied
                        let satisfied_count = results.iter().filter(|r| r.is_satisfied()).count();
                        let total_count = results.len();

                        if satisfied_count > total_count / 2 {
                            Ok(ConstraintEvaluationResult::satisfied())
                        } else {
                            Ok(ConstraintEvaluationResult::violated(
                                None,
                                Some(format!(
                                    "Only {} of {} constraints satisfied (majority required)",
                                    satisfied_count, total_count
                                )),
                            ))
                        }
                    }
                    "weighted" => {
                        // Weighted composition - would require weights to be stored
                        // For now, fallback to AND logic
                        for result in &results {
                            if result.is_violated() {
                                return Ok(result.clone());
                            }
                        }
                        Ok(ConstraintEvaluationResult::satisfied())
                    }
                    _ => Err(ShaclError::Configuration(format!(
                        "Unknown custom composition logic: {}",
                        custom_logic
                    ))),
                }
            }
        }
    }
}
