//! SHACL Validator for SAMM Models

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, ModelElement};
use crate::serializer::TurtleSerializer;
use crate::validator::{ValidationError, ValidationResult, ValidationWarning};
use oxirs_shacl::ValidationReport;
use oxirs_core::Store;
use oxrdf::Triple;
use oxttl::TurtleParser;

/// SHACL validator for SAMM models
pub struct ShaclValidator {
    /// SAMM shapes store (optional)
    shapes_store: Option<Store>,
}

impl ShaclValidator {
    /// Create a new SHACL validator
    pub fn new() -> Self {
        Self {
            shapes_store: None,
        }
    }

    /// Create a new SHACL validator with pre-loaded shapes
    pub fn with_shapes(shapes_store: Store) -> Self {
        Self {
            shapes_store: Some(shapes_store),
        }
    }

    /// Validate a SAMM model against SHACL shapes
    pub async fn validate(&self, aspect: &Aspect) -> Result<ValidationResult> {
        // 1. Convert Aspect to RDF store
        let _data_store = self.aspect_to_store(aspect)?;

        // 2. Get or load SAMM SHACL shapes
        let _shapes_store = match &self.shapes_store {
            Some(store) => store.clone(),
            None => self.load_default_shapes().await?,
        };

        // 3. Run SHACL validation
        // TODO: Implement full SHACL validation when oxirs-shacl API is stable
        // For now, perform basic structural validation

        // 4. Create validation result
        let mut validation_result = ValidationResult::new(true);

        // Perform basic structural validation
        self.validate_structure(aspect, &mut validation_result)?;

        Ok(validation_result)
    }

    /// Convert Aspect to RDF store
    fn aspect_to_store(&self, aspect: &Aspect) -> Result<Store> {
        // Serialize aspect to Turtle
        let serializer = TurtleSerializer::new();
        let turtle_content = serializer.serialize_to_string(aspect)?;

        // Create a new store
        let store = Store::new();

        // Parse Turtle into Store
        let parser = TurtleParser::new();
        for triple_result in parser.for_reader(turtle_content.as_bytes()) {
            let triple = triple_result.map_err(|e| {
                SammError::ParseError(format!("Failed to parse serialized aspect: {}", e))
            })?;
            store.insert(&Triple::from(triple));
        }

        Ok(store)
    }

    /// Load default SAMM SHACL shapes
    async fn load_default_shapes(&self) -> Result<Store> {
        // For now, return empty store until we embed SAMM shapes
        // In production, this should load embedded SAMM 2.3.0 SHACL shapes
        tracing::warn!("Using empty shapes store - SAMM shapes not yet embedded");

        // Future: Load embedded SAMM SHACL shapes from resources
        // This would include shapes for:
        // - Aspect structure validation
        // - Property cardinality constraints
        // - Characteristic type constraints
        // - Entity structure validation
        // - Operation parameter validation

        Ok(Store::new())
    }

    /// Load SAMM SHACL shapes from Turtle string
    pub async fn load_shapes_from_string(&mut self, turtle_content: &str) -> Result<()> {
        let store = Store::new();
        let parser = TurtleParser::new();

        for triple_result in parser.for_reader(turtle_content.as_bytes()) {
            let triple = triple_result.map_err(|e| {
                SammError::ParseError(format!("Failed to parse SHACL shapes: {}", e))
            })?;
            store.insert(&Triple::from(triple));
        }

        self.shapes_store = Some(store);
        Ok(())
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
            if name.chars().next().map_or(false, |c| c.is_uppercase()) {
                result.add_warning(ValidationWarning {
                    message: format!("Property '{}' should start with lowercase", name),
                    element_urn: Some(property.urn().to_string()),
                });
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
        aspect.metadata.add_preferred_name("en".to_string(), "Test Aspect".to_string());

        // Add a property with characteristic
        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#TestCharacteristic".to_string(),
            CharacteristicKind::Trait {
                base_characteristic: None,
                constraints: vec![],
            },
        );

        let mut property = Property::new("urn:samm:org.example:1.0.0#testProperty".to_string());
        property.set_characteristic(Some(characteristic));
        aspect.add_property(property);

        // Validate
        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());

        let validation_result = result.unwrap();
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

        let validation_result = result.unwrap();
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

        let validation_result = result.unwrap();
        // Should have warnings about no properties/operations and no preferred name
        assert!(!validation_result.warnings.is_empty());
    }

    #[tokio::test]
    async fn test_load_shapes_from_string() {
        let mut validator = ShaclValidator::new();

        let shapes_ttl = r#"
            @prefix sh: <http://www.w3.org/ns/shacl#> .
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .

            samm:AspectShape a sh:NodeShape ;
                sh:targetClass samm:Aspect ;
                sh:property [
                    sh:path samm:properties ;
                    sh:minCount 1 ;
                ] .
        "#;

        let result = validator.load_shapes_from_string(shapes_ttl).await;
        assert!(result.is_ok());
        assert!(validator.shapes_store.is_some());
    }

    #[tokio::test]
    async fn test_validation_property_naming() {
        let validator = ShaclValidator::new();

        // Create aspect with improperly named property (starts with uppercase)
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        aspect.metadata.add_preferred_name("en".to_string(), "Test Aspect".to_string());

        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#TestCharacteristic".to_string(),
            CharacteristicKind::Trait {
                base_characteristic: None,
                constraints: vec![],
            },
        );

        let mut property = Property::new("urn:samm:org.example:1.0.0#TestProperty".to_string());
        property.set_characteristic(Some(characteristic));
        aspect.add_property(property);

        let result = validator.validate(&aspect).await;
        assert!(result.is_ok());

        let validation_result = result.unwrap();
        // Should have a warning about property naming
        assert!(!validation_result.warnings.is_empty());
        assert!(validation_result.warnings.iter().any(|w| w.message.contains("lowercase")));
    }
}
