//! SAMM Model Validator
//!
//! Validates SAMM models against SHACL shapes.

mod shacl_validator;

pub use shacl_validator::ShaclValidator;

use crate::error::Result;
use crate::metamodel::Aspect;

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the model is valid
    pub is_valid: bool,

    /// Validation errors
    pub errors: Vec<ValidationError>,

    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error message
    pub message: String,

    /// Element URN that caused the error
    pub element_urn: Option<String>,

    /// Property path where the error occurred
    pub property_path: Option<String>,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,

    /// Element URN that caused the warning
    pub element_urn: Option<String>,
}

impl ValidationResult {
    /// Create a new validation result
    pub fn new(is_valid: bool) -> Self {
        Self {
            is_valid,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error to the result
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
        self.is_valid = false;
    }

    /// Add a warning to the result
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }
}

/// Validate a SAMM model
pub async fn validate_aspect(aspect: &Aspect) -> Result<ValidationResult> {
    let validator = ShaclValidator::new();
    validator.validate(aspect).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new(true);
        assert!(result.is_valid);
        assert_eq!(result.errors.len(), 0);

        result.add_error(ValidationError {
            message: "Test error".to_string(),
            element_urn: None,
            property_path: None,
        });

        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
    }
}
