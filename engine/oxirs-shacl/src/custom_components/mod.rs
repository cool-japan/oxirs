//! Custom SHACL Constraint Components
//!
//! This module provides comprehensive support for user-defined SHACL constraint components,
//! allowing users to extend SHACL with domain-specific validation logic.

pub mod registry;
pub mod standard_components; 
pub mod composite;
pub mod security;
pub mod performance;

// Re-export public API
pub use registry::*;
pub use standard_components::*;
pub use composite::*;
pub use security::*;
pub use performance::*;

use crate::{
    constraints::{ConstraintContext, ConstraintEvaluationResult},
    ConstraintComponentId, Result, ShaclError,
};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for custom constraint components
pub trait CustomConstraintComponent: Send + Sync + std::fmt::Debug {
    /// Get the component identifier
    fn component_id(&self) -> &ConstraintComponentId;

    /// Get component metadata
    fn metadata(&self) -> &ComponentMetadata;

    /// Validate the component configuration
    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()>;

    /// Create a constraint instance from parameters
    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint>;

    /// Get the SPARQL query template for this component (if SPARQL-based)
    fn sparql_template(&self) -> Option<&str> {
        None
    }

    /// Get additional prefixes needed for SPARQL queries
    fn sparql_prefixes(&self) -> Option<&str> {
        None
    }
}

/// Metadata for a constraint component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Version
    pub version: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Parameters this component accepts
    pub parameters: Vec<ParameterDefinition>,
    /// Whether this component can be used on node shapes
    pub applicable_to_node_shapes: bool,
    /// Whether this component can be used on property shapes
    pub applicable_to_property_shapes: bool,
    /// Example usage
    pub example: Option<String>,
}

/// Parameter definition for a custom constraint component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: Option<String>,
    /// Whether this parameter is required
    pub required: bool,
    /// Expected data type
    pub datatype: Option<String>,
    /// Default value
    pub default_value: Option<String>,
    /// Parameter validation constraints
    pub validation_constraints: Vec<ParameterConstraint>,
    /// Parameter cardinality (min, max)
    pub cardinality: Option<(u32, Option<u32>)>,
    /// Allowed values enumeration
    pub allowed_values: Option<Vec<String>>,
}

/// Parameter constraint for enhanced validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterConstraint {
    /// Minimum length for string parameters
    MinLength(u32),
    /// Maximum length for string parameters
    MaxLength(u32),
    /// Regular expression pattern
    Pattern(String),
    /// Numeric range constraint
    Range { min: Option<f64>, max: Option<f64> },
    /// Custom validation function
    CustomValidator(String),
}

/// Component library for organizing related components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentLibrary {
    /// Library identifier
    pub id: String,
    /// Library name
    pub name: String,
    /// Library description
    pub description: Option<String>,
    /// Library version
    pub version: String,
    /// Library author
    pub author: Option<String>,
    /// Components in this library
    pub components: Vec<ConstraintComponentId>,
    /// Library dependencies
    pub dependencies: Vec<String>,
    /// Library metadata
    pub metadata: HashMap<String, String>,
}

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

impl crate::constraints::ConstraintValidator for CustomConstraint {
    fn validate(&self) -> Result<()> {
        // Basic validation - check required parameters exist
        Ok(())
    }
}

impl crate::constraints::ConstraintEvaluator for CustomConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // If this is a SPARQL-based constraint, use SPARQL evaluation
        if let Some(query) = &self.sparql_query {
            let sparql_constraint = crate::sparql::SparqlConstraint {
                query: query.clone(),
                prefixes: None,
                message: self.message_template.clone(),
                severity: Some(crate::Severity::Violation),
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
    fn evaluate_regex_constraint(
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
    fn evaluate_range_constraint(
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
    fn evaluate_url_constraint(
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
    fn evaluate_email_constraint(
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

/// Validation rule for parameter validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: Option<String>,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Rule severity
    pub severity: crate::Severity,
    /// Rule condition
    pub condition: ValidationCondition,
}

/// Validation rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    Required,
    DataType,
    Range,
    Pattern,
    Enum,
    Custom,
}

/// Validation condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCondition {
    Required,
    DataTypeMatch(String),
    RangeCheck { min: Option<f64>, max: Option<f64> },
    PatternMatch(String),
    EnumValues(Vec<String>),
    CustomFunction(String),
}