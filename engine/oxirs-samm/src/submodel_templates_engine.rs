//! Submodel Templates — Instantiation Engine
//!
//! Template instantiation engine: variable substitution, constraint validation,
//! and submodel generation.

use std::collections::HashMap;

use super::submodel_templates_types::{
    TemplateConstraint, TemplateValidationError, ValidationSeverity,
};

/// Validate a single [`TemplateConstraint`] against a map of element values.
///
/// Returns `Some(error)` when the constraint is violated, `None` otherwise.
pub fn check_constraint(
    constraint: &TemplateConstraint,
    elements: &HashMap<String, String>,
) -> Option<TemplateValidationError> {
    match constraint {
        TemplateConstraint::FixedValue { element, value } => {
            if let Some(actual) = elements.get(element) {
                if actual != value {
                    return Some(TemplateValidationError {
                        element_path: element.clone(),
                        message: format!("Expected fixed value '{value}', got '{actual}'"),
                        severity: ValidationSeverity::Error,
                    });
                }
            }
            None
        }
        TemplateConstraint::Enumeration {
            element,
            allowed_values,
        } => {
            if let Some(actual) = elements.get(element) {
                if !allowed_values.contains(actual) {
                    return Some(TemplateValidationError {
                        element_path: element.clone(),
                        message: format!(
                            "Value '{actual}' not in allowed values: {allowed_values:?}"
                        ),
                        severity: ValidationSeverity::Error,
                    });
                }
            }
            None
        }
        TemplateConstraint::NumericRange { element, min, max } => {
            if let Some(actual) = elements.get(element) {
                if let Ok(val) = actual.parse::<f64>() {
                    if let Some(mn) = min {
                        if val < *mn {
                            return Some(TemplateValidationError {
                                element_path: element.clone(),
                                message: format!("Value {val} is below minimum {mn}"),
                                severity: ValidationSeverity::Error,
                            });
                        }
                    }
                    if let Some(mx) = max {
                        if val > *mx {
                            return Some(TemplateValidationError {
                                element_path: element.clone(),
                                message: format!("Value {val} exceeds maximum {mx}"),
                                severity: ValidationSeverity::Error,
                            });
                        }
                    }
                }
            }
            None
        }
        TemplateConstraint::StringLength {
            element,
            min_length,
            max_length,
        } => {
            if let Some(actual) = elements.get(element) {
                if let Some(mn) = min_length {
                    if actual.len() < *mn {
                        return Some(TemplateValidationError {
                            element_path: element.clone(),
                            message: format!(
                                "String length {} is below minimum {mn}",
                                actual.len()
                            ),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
                if let Some(mx) = max_length {
                    if actual.len() > *mx {
                        return Some(TemplateValidationError {
                            element_path: element.clone(),
                            message: format!("String length {} exceeds maximum {mx}", actual.len()),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
            }
            None
        }
        TemplateConstraint::Pattern { element, pattern } => {
            if let Some(actual) = elements.get(element) {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if !re.is_match(actual) {
                        return Some(TemplateValidationError {
                            element_path: element.clone(),
                            message: format!("Value '{actual}' does not match pattern '{pattern}'"),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
            }
            None
        }
        TemplateConstraint::ConditionalRequired {
            condition_element,
            required_element,
        } => {
            if elements.contains_key(condition_element) && !elements.contains_key(required_element)
            {
                return Some(TemplateValidationError {
                    element_path: required_element.clone(),
                    message: format!(
                        "Element '{required_element}' is required when '{condition_element}' is present"
                    ),
                    severity: ValidationSeverity::Error,
                });
            }
            None
        }
    }
}

/// Perform simple variable substitution in a template string.
///
/// Replaces `{{key}}` placeholders in `template_str` with the corresponding
/// values from `bindings`.  Unmatched placeholders are left as-is.
pub fn substitute_variables(template_str: &str, bindings: &HashMap<String, String>) -> String {
    let mut result = template_str.to_owned();
    for (key, value) in bindings {
        let placeholder = format!("{{{{{key}}}}}");
        result = result.replace(&placeholder, value);
    }
    result
}

/// Instantiate a submodel element collection from a template and variable bindings.
///
/// Returns a `HashMap<String, String>` representing the instantiated elements
/// (id_short → value) after variable substitution has been applied.
pub fn instantiate_elements(
    required_ids: &[String],
    optional_ids: &[String],
    bindings: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut result = HashMap::new();

    for id in required_ids.iter().chain(optional_ids.iter()) {
        if let Some(value) = bindings.get(id) {
            result.insert(id.clone(), substitute_variables(value, bindings));
        }
    }

    result
}
