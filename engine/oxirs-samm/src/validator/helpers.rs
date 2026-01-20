//! Validation helper functions for common patterns
//!
//! This module provides convenient functions for validating common patterns
//! in SAMM models without requiring full SHACL validation.

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, ModelElement, Property};
use crate::utils::urn;

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Error - must be fixed
    Error,
    /// Warning - should be reviewed
    Warning,
    /// Info - informational only
    Info,
}

/// A validation issue found during quick validation
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity level
    pub severity: Severity,
    /// Issue message
    pub message: String,
    /// Element URN where issue was found
    pub element_urn: Option<String>,
    /// Suggested fix
    pub suggestion: Option<String>,
}

impl ValidationIssue {
    /// Create a new error issue
    pub fn error(message: String, element_urn: Option<String>) -> Self {
        Self {
            severity: Severity::Error,
            message,
            element_urn,
            suggestion: None,
        }
    }

    /// Create a new warning issue
    pub fn warning(message: String, element_urn: Option<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message,
            element_urn,
            suggestion: None,
        }
    }

    /// Create a new info issue
    pub fn info(message: String, element_urn: Option<String>) -> Self {
        Self {
            severity: Severity::Info,
            message,
            element_urn,
            suggestion: None,
        }
    }

    /// Add a suggestion to this issue
    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggestion = Some(suggestion);
        self
    }
}

/// Quick validation result
#[derive(Debug, Clone)]
pub struct QuickValidationResult {
    /// List of issues found
    pub issues: Vec<ValidationIssue>,
    /// Whether there are any errors
    pub has_errors: bool,
    /// Whether there are any warnings
    pub has_warnings: bool,
}

impl QuickValidationResult {
    /// Create a new empty result
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
            has_errors: false,
            has_warnings: false,
        }
    }

    /// Add an issue to the result
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        match issue.severity {
            Severity::Error => self.has_errors = true,
            Severity::Warning => self.has_warnings = true,
            Severity::Info => {}
        }
        self.issues.push(issue);
    }

    /// Check if validation passed (no errors)
    pub fn is_valid(&self) -> bool {
        !self.has_errors
    }

    /// Get only errors
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect()
    }

    /// Get only warnings
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .collect()
    }
}

impl Default for QuickValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Perform quick validation checks on an aspect
///
/// This is faster than full SHACL validation but checks common issues.
///
/// # Example
/// ```
/// use oxirs_samm::metamodel::Aspect;
/// use oxirs_samm::validator::helpers::quick_validate;
///
/// let aspect = Aspect::new("urn:samm:test:1.0.0#MyAspect".to_string());
/// let result = quick_validate(&aspect);
///
/// if !result.is_valid() {
///     for error in result.errors() {
///         println!("Error: {}", error.message);
///     }
/// }
/// ```
pub fn quick_validate(aspect: &Aspect) -> QuickValidationResult {
    let mut result = QuickValidationResult::new();

    // Check aspect URN
    if let Err(e) = urn::validate_urn(aspect.urn()) {
        result.add_issue(ValidationIssue::error(
            format!("Invalid aspect URN: {}", e),
            Some(aspect.urn().to_string()),
        ));
    }

    // Check if aspect has at least one property
    if aspect.properties().is_empty() && aspect.operations().is_empty() {
        result.add_issue(
            ValidationIssue::warning(
                "Aspect has no properties or operations".to_string(),
                Some(aspect.urn().to_string()),
            )
            .with_suggestion(
                "Add at least one property or operation to make the aspect useful".to_string(),
            ),
        );
    }

    // Validate all properties
    for property in aspect.properties() {
        validate_property(&mut result, property);
    }

    // Check for duplicate property names
    let mut seen_names = std::collections::HashSet::new();
    for property in aspect.properties() {
        let name = property.name();
        if !seen_names.insert(name.clone()) {
            result.add_issue(
                ValidationIssue::error(
                    format!("Duplicate property name: {}", name),
                    Some(property.urn().to_string()),
                )
                .with_suggestion(format!("Rename one of the properties named '{}'", name)),
            );
        }
    }

    result
}

/// Validate a single property
fn validate_property(result: &mut QuickValidationResult, property: &Property) {
    // Check property URN
    if let Err(e) = urn::validate_urn(property.urn()) {
        result.add_issue(ValidationIssue::error(
            format!("Invalid property URN: {}", e),
            Some(property.urn().to_string()),
        ));
    }

    // Warn if property has no characteristic
    if property.characteristic.is_none() {
        result.add_issue(
            ValidationIssue::warning(
                format!("Property '{}' has no characteristic", property.name()),
                Some(property.urn().to_string()),
            )
            .with_suggestion(
                "Add a characteristic to define the property's data type and constraints"
                    .to_string(),
            ),
        );
    }

    // Check if property name follows naming conventions
    let prop_name = property.name();
    if !is_camel_case(&prop_name) {
        result.add_issue(
            ValidationIssue::info(
                format!("Property name '{}' is not in camelCase", prop_name),
                Some(property.urn().to_string()),
            )
            .with_suggestion(format!(
                "Consider renaming to '{}'",
                to_camel_case_suggestion(&prop_name)
            )),
        );
    }
}

/// Check if a string is in camelCase
fn is_camel_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    // First character should be lowercase
    let chars: Vec<char> = s.chars().collect();
    if !chars[0].is_lowercase() {
        return false;
    }

    // Should not contain underscores or hyphens
    if s.contains('_') || s.contains('-') {
        return false;
    }

    true
}

/// Suggest a camelCase version of a string
fn to_camel_case_suggestion(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }

    let parts: Vec<&str> = s.split(['_', '-', ' ']).collect();
    let mut result = String::new();

    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if i == 0 {
            // First part stays lowercase
            result.push_str(&part.to_lowercase());
        } else {
            // Capitalize first letter of subsequent parts
            let mut chars = part.chars();
            if let Some(first) = chars.next() {
                result.push(
                    first
                        .to_uppercase()
                        .next()
                        .expect("to_uppercase() always returns at least one character"),
                );
                result.push_str(&chars.as_str().to_lowercase());
            }
        }
    }

    result
}

/// Check if an aspect has required properties
///
/// # Example
/// ```
/// use oxirs_samm::metamodel::Aspect;
/// use oxirs_samm::validator::helpers::has_required_properties;
///
/// let aspect = Aspect::new("urn:samm:test:1.0.0#MyAspect".to_string());
/// assert!(!has_required_properties(&aspect));
/// ```
pub fn has_required_properties(aspect: &Aspect) -> bool {
    aspect.properties().iter().any(|p| !p.optional)
}

/// Check if an aspect has optional properties
pub fn has_optional_properties(aspect: &Aspect) -> bool {
    aspect.properties().iter().any(|p| p.optional)
}

/// Count properties by optional/required
pub fn count_by_optionality(aspect: &Aspect) -> (usize, usize) {
    let required = aspect.properties().iter().filter(|p| !p.optional).count();
    let optional = aspect.properties().iter().filter(|p| p.optional).count();
    (required, optional)
}

/// Validate that all properties have unique URNs
pub fn validate_unique_urns(aspect: &Aspect) -> Result<()> {
    let mut seen = std::collections::HashSet::new();

    for property in aspect.properties() {
        let urn = property.urn();
        if !seen.insert(urn) {
            return Err(SammError::ValidationError(format!(
                "Duplicate property URN: {}",
                urn
            )));
        }
    }

    Ok(())
}

/// Check if aspect name follows SAMM naming conventions
///
/// Aspect names should be PascalCase.
pub fn validate_aspect_name(aspect: &Aspect) -> Result<()> {
    let name = aspect.name();

    if name.is_empty() {
        return Err(SammError::ValidationError(
            "Aspect name cannot be empty".to_string(),
        ));
    }

    // First character should be uppercase
    if !name
        .chars()
        .next()
        .expect("name should not be empty (validated earlier)")
        .is_uppercase()
    {
        return Err(SammError::ValidationError(format!(
            "Aspect name '{}' should start with an uppercase letter (PascalCase)",
            name
        )));
    }

    // Should not contain underscores or hyphens
    if name.contains('_') || name.contains('-') {
        return Err(SammError::ValidationError(format!(
            "Aspect name '{}' should not contain underscores or hyphens (use PascalCase)",
            name
        )));
    }

    Ok(())
}

/// Validate that property names follow SAMM conventions
///
/// Property names should be camelCase.
pub fn validate_property_names(aspect: &Aspect) -> Result<()> {
    for property in aspect.properties() {
        let name = property.name();

        if name.is_empty() {
            return Err(SammError::ValidationError(format!(
                "Property in aspect '{}' has empty name",
                aspect.name()
            )));
        }

        // First character should be lowercase
        if !name
            .chars()
            .next()
            .expect("property name should not be empty")
            .is_lowercase()
        {
            return Err(SammError::ValidationError(format!(
                "Property name '{}' should start with a lowercase letter (camelCase)",
                name
            )));
        }

        // Should not contain underscores or hyphens
        if name.contains('_') || name.contains('-') {
            return Err(SammError::ValidationError(format!(
                "Property name '{}' should not contain underscores or hyphens (use camelCase)",
                name
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Property};

    #[test]
    fn test_quick_validate_empty_aspect() {
        let aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let result = quick_validate(&aspect);

        assert!(result.has_warnings);
        assert!(!result.has_errors);
    }

    #[test]
    fn test_quick_validate_with_properties() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new(
            "urn:samm:test:1.0.0#testProperty".to_string(),
        ));

        let result = quick_validate(&aspect);
        assert!(result.is_valid());
    }

    #[test]
    fn test_is_camel_case() {
        assert!(is_camel_case("camelCase"));
        assert!(is_camel_case("myProperty"));
        assert!(!is_camel_case("PascalCase"));
        assert!(!is_camel_case("snake_case"));
        assert!(!is_camel_case("kebab-case"));
    }

    #[test]
    fn test_to_camel_case_suggestion() {
        assert_eq!(to_camel_case_suggestion("snake_case"), "snakeCase");
        assert_eq!(to_camel_case_suggestion("kebab-case"), "kebabCase");
        assert_eq!(to_camel_case_suggestion("Space Case"), "spaceCase");
        assert_eq!(
            to_camel_case_suggestion("alreadyCamelCase"),
            "alreadycamelcase"
        );
    }

    #[test]
    fn test_has_required_properties() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        assert!(!has_required_properties(&aspect));

        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));
        assert!(has_required_properties(&aspect));

        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect2".to_string());
        aspect2.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()).as_optional());
        assert!(!has_required_properties(&aspect2));
    }

    #[test]
    fn test_count_by_optionality() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop2".to_string()).as_optional());

        let (required, optional) = count_by_optionality(&aspect);
        assert_eq!(required, 1);
        assert_eq!(optional, 1);
    }

    #[test]
    fn test_validate_aspect_name() {
        let good_aspect = Aspect::new("urn:samm:test:1.0.0#GoodName".to_string());
        assert!(validate_aspect_name(&good_aspect).is_ok());

        let bad_aspect = Aspect::new("urn:samm:test:1.0.0#badName".to_string());
        assert!(validate_aspect_name(&bad_aspect).is_err());

        let snake_aspect = Aspect::new("urn:samm:test:1.0.0#Bad_Name".to_string());
        assert!(validate_aspect_name(&snake_aspect).is_err());
    }

    #[test]
    fn test_validate_property_names() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#goodName".to_string()));

        assert!(validate_property_names(&aspect).is_ok());

        let mut bad_aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        bad_aspect.add_property(Property::new("urn:samm:test:1.0.0#BadName".to_string()));

        assert!(validate_property_names(&bad_aspect).is_err());
    }

    #[test]
    fn test_validate_unique_urns() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop2".to_string()));

        assert!(validate_unique_urns(&aspect).is_ok());

        // Note: add_property doesn't allow duplicates, so this test would need manual construction
        // Just verifying the function exists and works with valid input
    }
}
