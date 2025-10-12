//! AAS Environment data structures
//!
//! Based on AAS V3.0 specification (IDTA-01001-3-0)

use super::type_mapper::{map_xsd_to_aas_data_type_def_xsd, DataTypeDefXsd};
use crate::error::SammError;
use crate::metamodel::{
    Aspect, ModelElement, Operation as SammOperation, Property as SammProperty,
};
use serde::{Deserialize, Serialize};

/// AAS Environment (root structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Environment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset_administration_shells: Option<Vec<AssetAdministrationShell>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submodels: Option<Vec<Submodel>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concept_descriptions: Option<Vec<ConceptDescription>>,
}

/// Asset Administration Shell
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetAdministrationShell {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    pub model_type: String, // "AssetAdministrationShell"
    pub asset_information: AssetInformation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submodels: Option<Vec<Reference>>,
}

/// Asset Information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetInformation {
    pub asset_kind: AssetKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_asset_id: Option<String>,
}

/// Asset Kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetKind {
    Instance,
    NotApplicable,
    Role,
    Type,
}

/// Submodel
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Submodel {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    pub model_type: String, // "Submodel"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ModellingKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<Reference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submodel_elements: Option<Vec<SubmodelElement>>,
}

/// Modelling Kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModellingKind {
    Instance,
    Template,
}

/// Submodel Element (union type)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SubmodelElement {
    Property(Property),
    Operation(Operation),
    // TODO: Add other element types (Entity, Collection, etc.)
}

/// Property
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Property {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    pub model_type: String, // "Property"
    pub value_type: String, // DataTypeDefXsd as string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<Reference>,
}

/// Operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Operation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    pub model_type: String, // "Operation"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_variables: Option<Vec<OperationVariable>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_variables: Option<Vec<OperationVariable>>,
}

/// Operation Variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationVariable {
    pub value: Box<SubmodelElement>,
}

/// Reference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Reference {
    #[serde(rename = "type")]
    pub ref_type: ReferenceType,
    pub keys: Vec<Key>,
}

/// Reference Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReferenceType {
    ExternalReference,
    ModelReference,
}

/// Key
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Key {
    #[serde(rename = "type")]
    pub key_type: KeyType,
    pub value: String,
}

/// Key Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyType {
    AssetAdministrationShell,
    ConceptDescription,
    Submodel,
    SubmodelElement,
    GlobalReference,
    Property,
    Operation,
}

/// Concept Description
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConceptDescription {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    pub model_type: String, // "ConceptDescription"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedded_data_specifications: Option<Vec<EmbeddedDataSpecification>>,
}

/// Embedded Data Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddedDataSpecification {
    pub data_specification: Reference,
    pub data_specification_content: DataSpecificationIec61360,
}

/// Data Specification IEC 61360
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataSpecificationIec61360 {
    pub model_type: String, // "DataSpecificationIec61360"
    pub preferred_name: Vec<LangString>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_name: Option<Vec<LangString>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub definition: Option<Vec<LangString>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_type: Option<String>,
}

/// Lang String
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangString {
    pub language: String,
    pub text: String,
}

/// Build AAS Environment from SAMM Aspect
pub fn build_aas_environment(aspect: &Aspect) -> Result<Environment, SammError> {
    let aspect_urn = aspect.metadata.urn.clone();
    let aspect_name = aspect.name();

    // Create Submodel ID
    let submodel_id = format!("{}/submodel", aspect_urn);

    // Build Submodel Elements
    let mut submodel_elements = Vec::new();

    // Add Properties
    for prop in aspect.properties() {
        let property_element = build_property(prop)?;
        submodel_elements.push(SubmodelElement::Property(property_element));
    }

    // Add Operations
    for op in aspect.operations() {
        let operation_element = build_operation(op)?;
        submodel_elements.push(SubmodelElement::Operation(operation_element));
    }

    // Create Submodel
    let submodel = Submodel {
        id: submodel_id.clone(),
        id_short: Some(aspect_name.clone()),
        model_type: "Submodel".to_string(),
        kind: Some(ModellingKind::Template),
        semantic_id: Some(Reference {
            ref_type: ReferenceType::ExternalReference,
            keys: vec![Key {
                key_type: KeyType::GlobalReference,
                value: aspect_urn.clone(),
            }],
        }),
        submodel_elements: Some(submodel_elements),
    };

    // Create AssetAdministrationShell
    let aas = AssetAdministrationShell {
        id: aspect_urn.clone(),
        id_short: Some("defaultAdminShell".to_string()),
        model_type: "AssetAdministrationShell".to_string(),
        asset_information: AssetInformation {
            asset_kind: AssetKind::Type,
            global_asset_id: Some(aspect_urn.clone()),
        },
        submodels: Some(vec![Reference {
            ref_type: ReferenceType::ModelReference,
            keys: vec![Key {
                key_type: KeyType::Submodel,
                value: submodel_id.clone(),
            }],
        }]),
    };

    // Create Environment
    Ok(Environment {
        asset_administration_shells: Some(vec![aas]),
        submodels: Some(vec![submodel]),
        concept_descriptions: None, // TODO: Implement ConceptDescriptions
    })
}

fn build_property(prop: &SammProperty) -> Result<Property, SammError> {
    let data_type = if let Some(characteristic) = &prop.characteristic {
        if let Some(dt) = &characteristic.data_type {
            let aas_type = map_xsd_to_aas_data_type_def_xsd(dt);
            aas_type.to_xsd_string().to_string()
        } else {
            "xs:string".to_string()
        }
    } else {
        "xs:string".to_string()
    };

    Ok(Property {
        id_short: Some(prop.name().clone()),
        model_type: "Property".to_string(),
        value_type: data_type,
        value: None,
        semantic_id: Some(Reference {
            ref_type: ReferenceType::ExternalReference,
            keys: vec![Key {
                key_type: KeyType::GlobalReference,
                value: prop.metadata.urn.clone(),
            }],
        }),
    })
}

fn build_operation(op: &SammOperation) -> Result<Operation, SammError> {
    Ok(Operation {
        id_short: Some(op.name().clone()),
        model_type: "Operation".to_string(),
        input_variables: None,  // TODO: Implement from op.input
        output_variables: None, // TODO: Implement from op.output
    })
}
