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
///
/// The Environment is the root container for all AAS data structures according to the
/// Asset Administration Shell V3.0 specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Environment {
    /// Collection of Asset Administration Shells in this environment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset_administration_shells: Option<Vec<AssetAdministrationShell>>,
    /// Collection of Submodels in this environment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submodels: Option<Vec<Submodel>>,
    /// Collection of Concept Descriptions in this environment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concept_descriptions: Option<Vec<ConceptDescription>>,
}

/// Asset Administration Shell
///
/// Represents a digital twin according to the AAS specification. An AAS aggregates
/// submodels and provides asset information.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetAdministrationShell {
    /// Unique identifier of the AAS
    pub id: String,
    /// Short identifier for display purposes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    /// Model type identifier (always "AssetAdministrationShell")
    pub model_type: String,
    /// Information about the asset this AAS represents
    pub asset_information: AssetInformation,
    /// References to submodels contained in this AAS
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submodels: Option<Vec<Reference>>,
}

/// Asset Information
///
/// Describes the asset that an AAS represents, including its kind and global identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetInformation {
    /// Kind of the asset (Instance, Type, etc.)
    pub asset_kind: AssetKind,
    /// Global unique identifier of the asset
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_asset_id: Option<String>,
}

/// Asset Kind
///
/// Enumeration of asset kinds according to AAS V3.0 specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetKind {
    /// Instance of an asset type
    Instance,
    /// Not applicable
    NotApplicable,
    /// Role or function
    Role,
    /// Asset type (template)
    Type,
}

/// Submodel
///
/// A Submodel defines a specific aspect of an asset and contains structured data through submodel elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Submodel {
    /// Unique identifier of the submodel
    pub id: String,
    /// Short identifier for display purposes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    /// Model type identifier (always "Submodel")
    pub model_type: String,
    /// Modelling kind (Instance or Template)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ModellingKind>,
    /// Semantic identifier referencing an external concept
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<Reference>,
    /// Collection of submodel elements (properties, operations, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submodel_elements: Option<Vec<SubmodelElement>>,
}

/// Modelling Kind
///
/// Specifies whether a model element is an instance or a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModellingKind {
    /// Instance with actual data
    Instance,
    /// Template defining structure
    Template,
}

/// Submodel Element (union type)
///
/// Union of all possible submodel element types. Currently supports Property and Operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SubmodelElement {
    /// Property element with a single value
    Property(Property),
    /// Operation that can be invoked
    Operation(Operation),
    // TODO: Add other element types (Entity, Collection, etc.)
}

/// Property
///
/// Represents a single data property with a typed value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Property {
    /// Short identifier for display purposes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    /// Model type identifier (always "Property")
    pub model_type: String,
    /// XSD data type of the property value
    pub value_type: String,
    /// Actual value of the property (as string)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Semantic identifier referencing an external concept
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<Reference>,
}

/// Operation
///
/// Represents an operation that can be invoked with input and output parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Operation {
    /// Short identifier for display purposes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    /// Model type identifier (always "Operation")
    pub model_type: String,
    /// Input parameters of the operation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_variables: Option<Vec<OperationVariable>>,
    /// Output parameters of the operation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_variables: Option<Vec<OperationVariable>>,
}

/// Operation Variable
///
/// Wrapper for operation input/output parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationVariable {
    /// The submodel element representing the variable
    pub value: Box<SubmodelElement>,
}

/// Reference
///
/// A reference to a model element or external resource, composed of a type and a chain of keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Reference {
    /// Type of the reference (Model or External)
    #[serde(rename = "type")]
    pub ref_type: ReferenceType,
    /// Chain of keys identifying the referenced element
    pub keys: Vec<Key>,
}

/// Reference Type
///
/// Specifies whether a reference points to a model element or an external resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReferenceType {
    /// Reference to an external resource (e.g., concept dictionary)
    ExternalReference,
    /// Reference to a model element within the AAS
    ModelReference,
}

/// Key
///
/// A single key in a reference chain, identifying a specific element by type and value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Key {
    /// Type of the key (what kind of element it identifies)
    #[serde(rename = "type")]
    pub key_type: KeyType,
    /// Identifier value (typically a URN or ID)
    pub value: String,
}

/// Key Type
///
/// Enumeration of possible key types for references.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyType {
    /// Reference to an Asset Administration Shell
    AssetAdministrationShell,
    /// Reference to a Concept Description
    ConceptDescription,
    /// Reference to a Submodel
    Submodel,
    /// Reference to a Submodel Element
    SubmodelElement,
    /// Reference to an external global identifier
    GlobalReference,
    /// Reference to a Property element
    Property,
    /// Reference to an Operation element
    Operation,
}

/// Concept Description
///
/// Provides semantic information about model elements using standardized data specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConceptDescription {
    /// Unique identifier of the concept description
    pub id: String,
    /// Short identifier for display purposes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,
    /// Model type identifier (always "ConceptDescription")
    pub model_type: String,
    /// Embedded data specifications (e.g., IEC 61360)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedded_data_specifications: Option<Vec<EmbeddedDataSpecification>>,
}

/// Embedded Data Specification
///
/// Container for a data specification and its content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddedDataSpecification {
    /// Reference to the data specification template
    pub data_specification: Reference,
    /// Actual content of the data specification
    pub data_specification_content: DataSpecificationIec61360,
}

/// Data Specification IEC 61360
///
/// Data specification according to IEC 61360 standard for property dictionaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataSpecificationIec61360 {
    /// Model type identifier (always "DataSpecificationIec61360")
    pub model_type: String,
    /// Preferred name in multiple languages
    pub preferred_name: Vec<LangString>,
    /// Short name in multiple languages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_name: Option<Vec<LangString>>,
    /// Definition/description in multiple languages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub definition: Option<Vec<LangString>>,
    /// Data type according to IEC 61360
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_type: Option<String>,
}

/// Lang String
///
/// A text string with an associated language code (e.g., "en", "de", "ja").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangString {
    /// ISO 639 language code (e.g., "en", "de", "ja")
    pub language: String,
    /// The text content in the specified language
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
