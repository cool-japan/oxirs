//! AAS to SAMM converter
//!
//! This module provides conversion from AAS (Asset Administration Shell) data structures
//! to SAMM (Semantic Aspect Meta Model) Aspect models.

use super::models::{AasEnvironment, LangString, Submodel, SubmodelElement};
use crate::error::{Result, SammError};
use crate::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, Entity, ModelElement, Operation,
    Property,
};

/// Convert AAS environment to SAMM Aspect models
///
/// # Arguments
///
/// * `env` - AAS environment
/// * `submodel_indices` - List of submodel indices to convert (empty = all)
///
/// # Returns
///
/// * `Result<Vec<Aspect>>` - List of converted SAMM Aspect models
pub fn convert_environment_to_aspects(
    env: &AasEnvironment,
    submodel_indices: Vec<usize>,
) -> Result<Vec<Aspect>> {
    let mut aspects = Vec::new();

    // Determine which submodels to convert
    let indices_to_convert: Vec<usize> = if submodel_indices.is_empty() {
        // Convert all submodels
        (0..env.submodels.len()).collect()
    } else {
        // Convert only specified submodels
        submodel_indices
    };

    // Convert each selected submodel to an Aspect
    for idx in indices_to_convert {
        if idx >= env.submodels.len() {
            return Err(SammError::ParseError(format!(
                "Submodel index {} out of range (only {} submodels available)",
                idx,
                env.submodels.len()
            )));
        }

        let submodel = &env.submodels[idx];
        let aspect = convert_submodel_to_aspect(submodel)?;
        aspects.push(aspect);
    }

    Ok(aspects)
}

/// Convert a single AAS Submodel to a SAMM Aspect
fn convert_submodel_to_aspect(submodel: &Submodel) -> Result<Aspect> {
    // Extract name from idShort or ID
    let name = submodel.id_short.clone().unwrap_or_else(|| {
        // Extract last part of ID as name
        submodel
            .id
            .split(':')
            .next_back()
            .unwrap_or("UnknownAspect")
            .to_string()
    });

    // Create URN from submodel ID with # separator for name
    let urn = if submodel.id.starts_with("urn:") && submodel.id.contains('#') {
        submodel.id.clone()
    } else if submodel.id.starts_with("urn:") {
        format!("{}#{}", submodel.id, name)
    } else {
        format!("urn:aas:submodel:{}#{}", submodel.id, name)
    };

    // Create metadata
    let mut metadata = ElementMetadata::new(urn);

    // Add preferred names from description
    if let Some(descriptions) = &submodel.description {
        for lang_str in descriptions {
            metadata.add_preferred_name(lang_str.language.clone(), name.clone());
            metadata.add_description(lang_str.language.clone(), lang_str.text.clone());
        }
    }

    // If no descriptions, add English as default
    if metadata.get_preferred_name("en").is_none() {
        metadata.add_preferred_name("en".to_string(), name.clone());
    }

    // Create Aspect
    let mut aspect = Aspect {
        metadata,
        properties: Vec::new(),
        operations: Vec::new(),
        events: Vec::new(),
    };

    // Convert submodel elements to properties, operations, etc.
    for element in &submodel.submodel_elements {
        match element {
            SubmodelElement::Property(prop) => {
                let property = convert_aas_property_to_samm(prop)?;
                aspect.add_property(property);
            }
            SubmodelElement::SubmodelElementCollection(collection) => {
                // Convert collection to an Entity
                let entity = convert_collection_to_entity(collection)?;

                // Create a property that references this entity
                let entity_property = create_entity_property(&entity);
                aspect.add_property(entity_property);

                tracing::debug!("Converted collection '{}' to entity", entity.name());
            }
            SubmodelElement::Operation(op) => {
                let operation = convert_aas_operation_to_samm(op)?;
                aspect.add_operation(operation);
            }
            SubmodelElement::Unknown => {
                // Skip unknown elements
            }
        }
    }

    Ok(aspect)
}

/// Convert AAS Property to SAMM Property
fn convert_aas_property_to_samm(aas_prop: &super::models::Property) -> Result<Property> {
    let id_short = aas_prop
        .id_short
        .clone()
        .unwrap_or_else(|| "unknownProperty".to_string());

    // Create URN for property with # separator
    let urn = format!("urn:aas:property#{}", id_short);

    // Create metadata
    let mut metadata = ElementMetadata::new(urn);

    // Add descriptions
    if let Some(descriptions) = &aas_prop.description {
        for lang_str in descriptions {
            metadata.add_preferred_name(lang_str.language.clone(), id_short.clone());
            metadata.add_description(lang_str.language.clone(), lang_str.text.clone());
        }
    }

    // Determine data type from value_type
    let data_type = aas_prop.value_type.clone().or_else(|| {
        // Try to infer from value
        aas_prop.value.as_ref().map(|_| "xsd:string".to_string())
    });

    // Create characteristic
    let characteristic = data_type.map(|dtype| {
        Characteristic::new(
            format!("urn:aas:characteristic#{}Characteristic", id_short),
            CharacteristicKind::Trait,
        )
        .with_data_type(map_aas_type_to_xsd(&dtype))
    });

    // Create property
    let property = Property {
        metadata,
        characteristic,
        example_values: Vec::new(),
        optional: false,
        is_collection: false,
        payload_name: None,
        is_abstract: false,
        extends: None,
    };

    Ok(property)
}

/// Convert AAS Operation to SAMM Operation
fn convert_aas_operation_to_samm(aas_op: &super::models::Operation) -> Result<Operation> {
    let id_short = aas_op
        .id_short
        .clone()
        .unwrap_or_else(|| "unknownOperation".to_string());

    // Create URN for operation with # separator
    let urn = format!("urn:aas:operation#{}", id_short);

    // Create metadata
    let mut metadata = ElementMetadata::new(urn);

    // Add descriptions
    if let Some(descriptions) = &aas_op.description {
        for lang_str in descriptions {
            metadata.add_preferred_name(lang_str.language.clone(), id_short.clone());
            metadata.add_description(lang_str.language.clone(), lang_str.text.clone());
        }
    }

    // Create operation with input/output parameters
    let mut operation = Operation {
        metadata,
        input: Vec::new(),
        output: None,
    };

    // Convert input variables to SAMM properties
    for input_var in &aas_op.input_variables {
        match convert_aas_property_to_samm(&input_var.value) {
            Ok(property) => operation.add_input(property),
            Err(e) => {
                tracing::warn!("Failed to convert input variable: {}", e);
            }
        }
    }

    // Convert output variables to SAMM properties (take first one if available)
    if let Some(output_var) = aas_op.output_variables.first() {
        match convert_aas_property_to_samm(&output_var.value) {
            Ok(property) => {
                operation.output = Some(property);
            }
            Err(e) => {
                tracing::warn!("Failed to convert output variable: {}", e);
            }
        }
    }

    Ok(operation)
}

/// Convert AAS SubmodelElementCollection to SAMM Entity
fn convert_collection_to_entity(
    collection: &super::models::SubmodelElementCollection,
) -> Result<Entity> {
    let id_short = collection
        .id_short
        .clone()
        .unwrap_or_else(|| "unknownEntity".to_string());

    // Create URN for entity with # separator
    let urn = format!("urn:aas:entity#{}", id_short);

    // Create metadata
    let mut metadata = ElementMetadata::new(urn);

    // Add descriptions
    if let Some(descriptions) = &collection.description {
        for lang_str in descriptions {
            metadata.add_preferred_name(lang_str.language.clone(), id_short.clone());
            metadata.add_description(lang_str.language.clone(), lang_str.text.clone());
        }
    }

    // Create entity and convert nested elements to properties
    let mut properties = Vec::new();
    for element in &collection.value {
        if let SubmodelElement::Property(prop) = element {
            let property = convert_aas_property_to_samm(prop)?;
            properties.push(property);
        }
    }

    let entity = Entity {
        metadata,
        properties,
        extends: None,
        is_abstract: false,
    };

    Ok(entity)
}

/// Create a property that references an entity
fn create_entity_property(entity: &Entity) -> Property {
    let entity_name = entity.name();
    let urn = format!("urn:aas:property#{}", entity_name);

    let mut metadata = ElementMetadata::new(urn);
    metadata.add_preferred_name("en".to_string(), entity_name.to_string());

    // Create a characteristic that references the entity type
    let characteristic = Characteristic::new(
        format!("urn:aas:characteristic#{}Characteristic", entity_name),
        CharacteristicKind::Trait,
    )
    .with_data_type(entity.urn().to_string());

    Property {
        metadata,
        characteristic: Some(characteristic),
        example_values: Vec::new(),
        optional: false,
        is_collection: false,
        payload_name: None,
        is_abstract: false,
        extends: None,
    }
}

/// Map AAS data types to XSD data types
fn map_aas_type_to_xsd(aas_type: &str) -> String {
    // AAS uses "xs:" prefix, SAMM uses "xsd:" prefix
    if aas_type.starts_with("xs:") {
        aas_type.replace("xs:", "xsd:")
    } else if aas_type.starts_with("xsd:") {
        aas_type.to_string()
    } else {
        // Default to string if unknown
        "xsd:string".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aas_parser::models::*;
    use crate::metamodel::ModelElement;

    #[test]
    fn test_convert_simple_submodel() {
        let submodel = Submodel {
            id: "urn:aas:submodel:test:1".to_string(),
            id_short: Some("TestSubmodel".to_string()),
            model_type: "Submodel".to_string(),
            semantic_id: None,
            description: Some(vec![LangString {
                language: "en".to_string(),
                text: "Test submodel description".to_string(),
            }]),
            submodel_elements: vec![],
        };

        let aspect = convert_submodel_to_aspect(&submodel).expect("conversion should succeed");

        assert_eq!(aspect.name(), "TestSubmodel");
        assert_eq!(
            aspect.metadata().urn,
            "urn:aas:submodel:test:1#TestSubmodel"
        );
        assert_eq!(
            aspect.metadata().get_description("en"),
            Some("Test submodel description")
        );
    }

    #[test]
    fn test_convert_submodel_with_property() {
        let submodel = Submodel {
            id: "urn:aas:submodel:test:1".to_string(),
            id_short: Some("TestSubmodel".to_string()),
            model_type: "Submodel".to_string(),
            semantic_id: None,
            description: None,
            submodel_elements: vec![SubmodelElement::Property(super::super::models::Property {
                id_short: Some("temperature".to_string()),
                semantic_id: None,
                description: Some(vec![LangString {
                    language: "en".to_string(),
                    text: "Temperature value".to_string(),
                }]),
                value_type: Some("xs:float".to_string()),
                value: Some("25.5".to_string()),
            })],
        };

        let aspect = convert_submodel_to_aspect(&submodel).expect("conversion should succeed");

        assert_eq!(aspect.properties().len(), 1);
        assert_eq!(aspect.properties()[0].name(), "temperature");
    }

    #[test]
    fn test_map_aas_type_to_xsd() {
        assert_eq!(map_aas_type_to_xsd("xs:string"), "xsd:string");
        assert_eq!(map_aas_type_to_xsd("xs:int"), "xsd:int");
        assert_eq!(map_aas_type_to_xsd("xsd:float"), "xsd:float");
        assert_eq!(map_aas_type_to_xsd("unknown"), "xsd:string");
    }
}
