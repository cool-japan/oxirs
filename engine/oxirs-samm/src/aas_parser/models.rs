//! AAS data models
//!
//! This module defines the data structures for representing Asset Administration Shell (AAS)
//! environments, based on the AAS specification.

use serde::{Deserialize, Serialize};

/// AAS Environment (root structure)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AasEnvironment {
    /// Asset Administration Shells
    #[serde(default)]
    pub asset_administration_shells: Vec<AssetAdministrationShell>,

    /// Submodels
    #[serde(default)]
    pub submodels: Vec<Submodel>,

    /// Concept Descriptions
    #[serde(default)]
    pub concept_descriptions: Vec<ConceptDescription>,
}

/// Asset Administration Shell
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetAdministrationShell {
    /// Unique identifier
    pub id: String,

    /// Short identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,

    /// Model type (always "AssetAdministrationShell")
    pub model_type: String,

    /// Asset information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset_information: Option<AssetInformation>,

    /// References to submodels
    #[serde(default)]
    pub submodels: Vec<Reference>,
}

/// Asset Information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetInformation {
    /// Asset kind (Instance or Type)
    pub asset_kind: String,

    /// Global asset identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_asset_id: Option<String>,
}

/// Submodel
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Submodel {
    /// Unique identifier
    pub id: String,

    /// Short identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,

    /// Model type (always "Submodel")
    pub model_type: String,

    /// Semantic ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<Reference>,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Vec<LangString>>,

    /// Submodel elements
    #[serde(default)]
    pub submodel_elements: Vec<SubmodelElement>,
}

/// Submodel Element (union type)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "modelType")]
pub enum SubmodelElement {
    /// Property element
    Property(Property),

    /// Collection element
    SubmodelElementCollection(SubmodelElementCollection),

    /// Operation element
    Operation(Operation),

    /// Other element types (not yet supported)
    #[serde(other)]
    Unknown,
}

/// Property (data element)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Property {
    /// Short identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,

    /// Semantic ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<Reference>,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Vec<LangString>>,

    /// Value type (e.g., "xs:string", "xs:int")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_type: Option<String>,

    /// Value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

/// Submodel Element Collection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmodelElementCollection {
    /// Short identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,

    /// Semantic ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<Reference>,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Vec<LangString>>,

    /// Nested submodel elements
    #[serde(default)]
    pub value: Vec<SubmodelElement>,
}

/// Operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Operation {
    /// Short identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,

    /// Semantic ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<Reference>,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Vec<LangString>>,

    /// Input variables
    #[serde(default)]
    pub input_variables: Vec<OperationVariable>,

    /// Output variables
    #[serde(default)]
    pub output_variables: Vec<OperationVariable>,
}

/// Operation Variable
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationVariable {
    /// Value (property reference)
    pub value: Property,
}

/// Reference (to other elements)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Reference {
    /// Reference type (e.g., "ModelReference", "ExternalReference")
    #[serde(rename = "type")]
    pub ref_type: String,

    /// Keys
    pub keys: Vec<Key>,
}

/// Key (part of a reference)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Key {
    /// Key type (e.g., "Submodel", "ConceptDescription")
    #[serde(rename = "type")]
    pub key_type: String,

    /// Value
    pub value: String,
}

/// Language String (localized text)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LangString {
    /// Language code (e.g., "en", "de")
    pub language: String,

    /// Text
    pub text: String,
}

/// Concept Description (semantic definition)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConceptDescription {
    /// Unique identifier
    pub id: String,

    /// Short identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_short: Option<String>,

    /// Model type (always "ConceptDescription")
    pub model_type: String,

    /// Embedded data specifications
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedded_data_specifications: Option<Vec<EmbeddedDataSpecification>>,
}

/// Embedded Data Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddedDataSpecification {
    /// Data specification reference
    pub data_specification: Reference,

    /// Data specification content
    pub data_specification_content: DataSpecificationIec61360,
}

/// Data Specification IEC 61360 (semantic content)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataSpecificationIec61360 {
    /// Preferred name
    pub preferred_name: Vec<LangString>,

    /// Short name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_name: Option<Vec<LangString>>,

    /// Definition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub definition: Option<Vec<LangString>>,

    /// Data type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_type: Option<String>,
}
