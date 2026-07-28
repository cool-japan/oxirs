//! SHACL Validator for SAMM Models
//!
//! This module provides basic structural validation for SAMM models.
//! Full SHACL validation integration is planned for future releases.

use crate::aspect_validator::{AspectModel, AspectValidator, Severity as ConstraintSeverity};
use crate::constraint_validator::validate_property_example_values;
use crate::error::Result;
use crate::metamodel::{Aspect, ModelElement};
use crate::validator::{ValidationError, ValidationResult, ValidationWarning};

/// SHACL validator for SAMM models
///
/// Currently performs basic structural validation.
/// Full SHACL validation will be added in future releases.
pub struct ShaclValidator {
    // Future: SAMM SHACL shapes will be stored here
}

impl ShaclValidator {
    /// Create a new SHACL validator
    pub fn new() -> Self {
        Self {}
    }

    /// Validate a SAMM model against SHACL shapes
    ///
    /// Currently performs basic structural validation.
    pub async fn validate(&self, aspect: &Aspect) -> Result<ValidationResult> {
        let mut validation_result = ValidationResult::new(true);

        // Perform basic structural validation
        self.validate_structure(aspect, &mut validation_result)?;

        Ok(validation_result)
    }

    /// Perform basic structural validation
    fn validate_structure(&self, aspect: &Aspect, result: &mut ValidationResult) -> Result<()> {
        // Check that aspect has at least one property or operation
        if aspect.properties().is_empty() && aspect.operations().is_empty() {
            result.add_warning(ValidationWarning {
                message: "Aspect has no properties or operations".to_string(),
                element_urn: Some(aspect.urn().to_string()),
            });
        }

        // Check that all properties have characteristics
        for property in aspect.properties() {
            if property.characteristic.is_none() {
                result.add_error(ValidationError {
                    message: "Property must have a characteristic".to_string(),
                    element_urn: Some(property.urn().to_string()),
                    property_path: Some("samm:characteristic".to_string()),
                });
            }
        }

        // Check URN format
        if !aspect.urn().starts_with("urn:samm:") {
            result.add_warning(ValidationWarning {
                message: "Aspect URN should start with 'urn:samm:'".to_string(),
                element_urn: Some(aspect.urn().to_string()),
            });
        }

        // Check preferred names exist
        if aspect.metadata.preferred_names.is_empty() {
            result.add_warning(ValidationWarning {
                message: "Aspect should have at least one preferred name".to_string(),
                element_urn: Some(aspect.urn().to_string()),
            });
        }

        // Check property URN naming convention
        for property in aspect.properties() {
            let name = property.name();
            if name.chars().next().is_some_and(|c| c.is_uppercase()) {
                result.add_warning(ValidationWarning {
                    message: format!("Property '{}' should start with lowercase", name),
                    element_urn: Some(property.urn().to_string()),
                });
            }
        }

        // Check for duplicate property URNs
        let mut seen_urns = std::collections::HashSet::new();
        for property in aspect.properties() {
            let urn = property.urn();
            if !seen_urns.insert(urn) {
                result.add_error(ValidationError {
                    message: format!("Duplicate property URN: {}", urn),
                    element_urn: Some(property.urn().to_string()),
                    property_path: None,
                });
            }
        }

        // Constraint-compatibility validation: catches a
        // `samm-c:RangeConstraint` on a non-numeric characteristic,
        // `samm-c:LengthConstraint`/`samm-c:RegularExpressionConstraint` on
        // a non-string characteristic, an inverted min/max range or length,
        // a boolean characteristic with a range constraint, and cyclic
        // entity `extends` chains. This bridges the real parsed model into
        // the `AspectValidator` constraint-validation subsystem, which is
        // otherwise never invoked from the crate's own validation pipeline.
        let aspect_model = AspectModel::from_metamodel(aspect);
        for message in &AspectValidator::validate(&aspect_model).messages {
            match message.severity {
                ConstraintSeverity::Error => result.add_error(ValidationError {
                    message: message.message.clone(),
                    element_urn: Some(message.element.clone()),
                    property_path: message.path.clone(),
                }),
                ConstraintSeverity::Warning => result.add_warning(ValidationWarning {
                    message: message.message.clone(),
                    element_urn: Some(message.element.clone()),
                }),
                ConstraintSeverity::Info => {
                    // Informational-only findings (e.g. "no description")
                    // are not surfaced as SHACL validation errors/warnings.
                }
            }
        }

        // Example-value constraint validation: catches a `samm:exampleValue`
        // that violates the property's own declared SAMM constraints
        // (Range/Length/RegularExpression/Encoding), using the real
        // ConstraintValidator/SammConstraint value-checking subsystem. This
        // is the one place in a parsed Aspect model where genuine
        // instance-like data (as opposed to schema definitions) exists to
        // check constraints against.
        for property in aspect.properties() {
            for (example, results) in validate_property_example_values(property) {
                for violation in results.iter().filter(|r| !r.is_valid()) {
                    if let Some(reason) = violation.reason() {
                        result.add_warning(ValidationWarning {
                            message: format!(
                                "Example value '{}' for property '{}' violates a declared constraint: {}",
                                example,
                                property.name(),
                                reason
                            ),
                            element_urn: Some(property.urn().to_string()),
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for ShaclValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Characteristic, CharacteristicKind, Property};

    #[test]
    fn test_validator_creation() {
        let _validator = ShaclValidator::new();
    }

    #[tokio::test]
    async fn test_basic_validation() {
        let validator = ShaclValidator::new();

        // Create a simple valid aspect
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "Test Aspect".to_string());

        // Add a property with characteristic
        let mut characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#TestCharacteristic".to_string(),
            CharacteristicKind::Trait,
        );
        characteristic.data_type = Some("xsd:string".to_string());

        let property = Property::new("urn:samm:org.example:1.0.0#testProperty".to_string())
            .with_characteristic(characteristic);
        aspect.add_property(property);

        // Validate
        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());

        let validation_result = result.expect("validation should succeed");
        // Should be valid (basic structure checks pass)
        assert!(validation_result.is_valid || validation_result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_validation_missing_characteristic() {
        let validator = ShaclValidator::new();

        // Create aspect with property but no characteristic
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        let property = Property::new("urn:samm:org.example:1.0.0#testProperty".to_string());
        aspect.add_property(property);

        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());

        let validation_result = result.expect("validation should succeed");
        // Should have an error about missing characteristic
        assert!(!validation_result.is_valid);
        assert!(!validation_result.errors.is_empty());
        assert!(validation_result.errors[0]
            .message
            .contains("characteristic"));
    }

    #[tokio::test]
    async fn test_validation_empty_aspect() {
        let validator = ShaclValidator::new();

        // Create empty aspect
        let aspect = Aspect::new("urn:samm:org.example:1.0.0#EmptyAspect".to_string());

        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());

        let validation_result = result.expect("validation should succeed");
        // Should have warnings about no properties/operations and no preferred name
        assert!(!validation_result.warnings.is_empty());
    }

    #[tokio::test]
    async fn test_validation_property_naming() {
        let validator = ShaclValidator::new();

        // Create aspect with improperly named property (starts with uppercase)
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "Test Aspect".to_string());

        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#TestCharacteristic".to_string(),
            CharacteristicKind::Trait,
        );

        let property = Property::new("urn:samm:org.example:1.0.0#TestProperty".to_string())
            .with_characteristic(characteristic);
        aspect.add_property(property);

        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());

        let validation_result = result.expect("validation should succeed");
        // Should have a warning about property naming
        assert!(!validation_result.warnings.is_empty());
        assert!(validation_result
            .warnings
            .iter()
            .any(|w| w.message.contains("lowercase")));
    }

    #[tokio::test]
    async fn test_validation_duplicate_properties() {
        let validator = ShaclValidator::new();

        // Create aspect with duplicate property URNs
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "Test Aspect".to_string());

        let characteristic1 = Characteristic::new(
            "urn:samm:org.example:1.0.0#TestCharacteristic".to_string(),
            CharacteristicKind::Trait,
        );
        let characteristic2 = Characteristic::new(
            "urn:samm:org.example:1.0.0#TestCharacteristic".to_string(),
            CharacteristicKind::Trait,
        );

        let property1 = Property::new("urn:samm:org.example:1.0.0#testProperty".to_string())
            .with_characteristic(characteristic1);
        aspect.add_property(property1);

        let property2 = Property::new("urn:samm:org.example:1.0.0#testProperty".to_string())
            .with_characteristic(characteristic2);
        aspect.add_property(property2);

        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());

        let validation_result = result.expect("validation should succeed");
        // Should have an error about duplicate URN
        assert!(!validation_result.is_valid);
        assert!(validation_result
            .errors
            .iter()
            .any(|e| e.message.contains("Duplicate")));
    }

    #[tokio::test]
    async fn regression_validate_wires_range_constraint_compatibility_checks() {
        use crate::metamodel::{BoundDefinition, Constraint};

        let validator = ShaclValidator::new();

        // A real, TTL-parser-shaped Aspect whose characteristic declares a
        // RangeConstraint on a non-numeric (string) data type — a genuine
        // SAMM modelling error that only the AspectValidator
        // constraint-compatibility subsystem can detect.
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#BadModel".to_string());
        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#BadChar".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string())
        .with_constraint(Constraint::RangeConstraint {
            min_value: Some("0".to_string()),
            max_value: Some("100".to_string()),
            lower_bound_definition: BoundDefinition::AtLeast,
            upper_bound_definition: BoundDefinition::AtLeast,
        });
        let property = Property::new("urn:samm:org.example:1.0.0#bad".to_string())
            .with_characteristic(characteristic);
        aspect.add_property(property);

        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());
        let validation_result = result.expect("validation should succeed");

        assert!(
            !validation_result.is_valid,
            "a RangeConstraint on a non-numeric characteristic must fail validation"
        );
        assert!(
            validation_result.errors.iter().any(|e| e
                .message
                .contains("RangeConstraint applied to non-numeric type")),
            "errors={:?}",
            validation_result.errors
        );
    }

    #[tokio::test]
    async fn regression_validate_accepts_valid_range_constraint_on_numeric_type() {
        use crate::metamodel::{BoundDefinition, Constraint};

        let validator = ShaclValidator::new();

        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#GoodModel".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "GoodModel".to_string());
        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#PercentageChar".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string())
        .with_constraint(Constraint::RangeConstraint {
            min_value: Some("0".to_string()),
            max_value: Some("100".to_string()),
            lower_bound_definition: BoundDefinition::AtLeast,
            upper_bound_definition: BoundDefinition::AtLeast,
        });
        let property = Property::new("urn:samm:org.example:1.0.0#percentage".to_string())
            .with_characteristic(characteristic);
        aspect.add_property(property);

        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());
        let validation_result = result.expect("validation should succeed");

        assert!(
            !validation_result
                .errors
                .iter()
                .any(|e| e.message.contains("RangeConstraint")),
            "a valid numeric RangeConstraint must not be flagged: {:?}",
            validation_result.errors
        );
    }

    #[tokio::test]
    async fn regression_validate_flags_example_value_violating_range_constraint() {
        use crate::metamodel::{BoundDefinition, Constraint};

        let validator = ShaclValidator::new();

        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Reading".to_string());
        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#PercentageChar".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string())
        .with_constraint(Constraint::RangeConstraint {
            min_value: Some("0".to_string()),
            max_value: Some("100".to_string()),
            lower_bound_definition: BoundDefinition::AtLeast,
            upper_bound_definition: BoundDefinition::AtLeast,
        });
        let mut property = Property::new("urn:samm:org.example:1.0.0#percentage".to_string())
            .with_characteristic(characteristic);
        // An exampleValue that violates the property's own declared range.
        property.example_values = vec!["150".to_string()];
        aspect.add_property(property);

        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());
        let validation_result = result.expect("validation should succeed");

        assert!(
            validation_result
                .warnings
                .iter()
                .any(|w| w.message.contains("Example value '150'")
                    && w.message.contains("violates a declared constraint")),
            "warnings={:?}",
            validation_result.warnings
        );
    }

    #[tokio::test]
    async fn regression_validate_does_not_flag_conforming_example_value() {
        use crate::metamodel::{BoundDefinition, Constraint};

        let validator = ShaclValidator::new();

        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Reading".to_string());
        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#PercentageChar".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string())
        .with_constraint(Constraint::RangeConstraint {
            min_value: Some("0".to_string()),
            max_value: Some("100".to_string()),
            lower_bound_definition: BoundDefinition::AtLeast,
            upper_bound_definition: BoundDefinition::AtLeast,
        });
        let mut property = Property::new("urn:samm:org.example:1.0.0#percentage".to_string())
            .with_characteristic(characteristic);
        property.example_values = vec!["42".to_string()];
        aspect.add_property(property);

        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());
        let validation_result = result.expect("validation should succeed");

        assert!(
            !validation_result
                .warnings
                .iter()
                .any(|w| w.message.contains("violates a declared constraint")),
            "warnings={:?}",
            validation_result.warnings
        );
    }
}
