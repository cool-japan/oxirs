//! DTDL Parser - Azure Digital Twins to SAMM Converter
//!
//! Parses DTDL (Digital Twins Definition Language) v3 Interface definitions
//! and converts them to SAMM Aspect models.
//!
//! # Features
//!
//! - Parse DTDL v3 JSON Interface definitions
//! - Convert DTMI to SAMM URN format
//! - Map DTDL schemas to XSD data types
//! - Support for Properties, Telemetry, Commands
//! - Comprehensive error handling
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_samm::dtdl_parser::parse_dtdl_interface;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let dtdl_json = r#"{
//!   "@context": "dtmi:dtdl:context;3",
//!   "@id": "dtmi:com:example:Movement;1",
//!   "@type": "Interface",
//!   "displayName": "Movement",
//!   "contents": []
//! }"#;
//!
//! let aspect = parse_dtdl_interface(dtdl_json)?;
//! println!("Aspect: {}", aspect.name());
//! # Ok(())
//! # }
//! ```

use crate::error::SammError;
use crate::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, Event, Operation, Property,
};
use serde::{Deserialize, Serialize};

/// DTDL Interface definition (subset for parsing)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DtdlInterface {
    #[serde(rename = "@context")]
    context: String,

    #[serde(rename = "@id")]
    id: String,

    #[serde(rename = "@type")]
    type_: DtdlType,

    #[serde(rename = "displayName", skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    comment: Option<String>,

    #[serde(default)]
    contents: Vec<DtdlContent>,
}

/// DTDL type field
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
enum DtdlType {
    Single(String),
    Multiple(Vec<String>),
}

/// DTDL content element (Property, Telemetry, Command, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DtdlContent {
    #[serde(rename = "@type")]
    type_: String,

    name: String,

    #[serde(rename = "displayName", skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<DtdlSchema>,

    #[serde(rename = "writable", skip_serializing_if = "Option::is_none")]
    writable: Option<bool>,
}

/// DTDL schema type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum DtdlSchema {
    /// Primitive type (string, integer, float, etc.)
    Primitive(String),
    /// Complex object schema
    Object(serde_json::Value),
}

/// Parse DTDL Interface JSON to SAMM Aspect
///
/// Converts a DTDL v3 Interface definition to a SAMM Aspect model.
///
/// # Arguments
///
/// * `json` - DTDL Interface JSON string
///
/// # Returns
///
/// SAMM Aspect model
///
/// # Example
///
/// ```rust,ignore
/// use oxirs_samm::dtdl_parser::parse_dtdl_interface;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dtdl = r#"{
///   "@context": "dtmi:dtdl:context;3",
///   "@id": "dtmi:com:example:Movement;1",
///   "@type": "Interface",
///   "displayName": "Movement",
///   "contents": []
/// }"#;
///
/// let aspect = parse_dtdl_interface(dtdl)?;
/// assert_eq!(aspect.name(), "Movement");
/// # Ok(())
/// # }
/// ```
pub fn parse_dtdl_interface(json: &str) -> Result<Aspect, SammError> {
    // Parse JSON
    let interface: DtdlInterface = serde_json::from_str(json)
        .map_err(|e| SammError::ParseError(format!("Invalid DTDL JSON: {}", e)))?;

    // Validate DTDL version
    if !interface.context.contains("dtmi:dtdl:context") {
        return Err(SammError::ParseError(format!(
            "Invalid DTDL context: {}. Expected dtmi:dtdl:context",
            interface.context
        )));
    }

    // Verify it's an Interface
    match &interface.type_ {
        DtdlType::Single(t) if t == "Interface" => {}
        DtdlType::Multiple(types) if types.contains(&"Interface".to_string()) => {}
        _ => {
            return Err(SammError::ParseError(format!(
                "Not a DTDL Interface. Got type: {:?}",
                interface.type_
            )));
        }
    }

    // Convert DTMI to SAMM URN
    let urn = dtmi_to_urn(&interface.id)?;

    // Create Aspect
    let mut aspect = Aspect::new(urn);

    // Set metadata
    if let Some(display_name) = interface.display_name {
        aspect
            .metadata
            .add_preferred_name("en".to_string(), display_name);
    }

    if let Some(description) = interface.description {
        aspect
            .metadata
            .add_description("en".to_string(), description);
    }

    if let Some(comment) = interface.comment {
        // Extract see_refs from comment if it contains "See also:"
        if comment.starts_with("See also:") {
            let refs = comment
                .strip_prefix("See also:")
                .unwrap_or("")
                .trim()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>();
            aspect.metadata.see_refs.extend(refs);
        }
    }

    // Parse contents
    for content in interface.contents {
        match content.type_.as_str() {
            "Property" | "Telemetry" => {
                let property = parse_property_content(&content, &interface.id)?;
                aspect.add_property(property);
            }
            "Command" => {
                let operation = parse_command_content(&content, &interface.id)?;
                aspect.add_operation(operation);
            }
            _ => {
                // Skip unsupported content types for now
                // (Relationship, Component will be added in Phase 3)
            }
        }
    }

    Ok(aspect)
}

/// Parse DTDL Property/Telemetry to SAMM Property
fn parse_property_content(
    content: &DtdlContent,
    interface_id: &str,
) -> Result<Property, SammError> {
    // Build property URN
    let property_urn = build_property_urn(interface_id, &content.name)?;

    let mut property = Property::new(property_urn);

    // Set metadata
    if let Some(display_name) = &content.display_name {
        property
            .metadata
            .add_preferred_name("en".to_string(), display_name.clone());
    }

    if let Some(description) = &content.description {
        property
            .metadata
            .add_description("en".to_string(), description.clone());
    }

    // Set optional based on type (Telemetry is read-only/optional)
    property.optional = content.type_ == "Telemetry";

    // Parse schema to characteristic
    if let Some(schema) = &content.schema {
        let characteristic = parse_schema_to_characteristic(schema, interface_id, &content.name)?;
        property.characteristic = Some(characteristic);
    }

    Ok(property)
}

/// Parse DTDL Command to SAMM Operation
fn parse_command_content(
    content: &DtdlContent,
    interface_id: &str,
) -> Result<Operation, SammError> {
    // Build operation URN
    let operation_urn = build_operation_urn(interface_id, &content.name)?;

    let mut operation = Operation::new(operation_urn);

    // Set metadata
    if let Some(display_name) = &content.display_name {
        operation
            .metadata
            .add_preferred_name("en".to_string(), display_name.clone());
    }

    if let Some(description) = &content.description {
        operation
            .metadata
            .add_description("en".to_string(), description.clone());
    }

    Ok(operation)
}

/// Parse DTDL schema to SAMM Characteristic
fn parse_schema_to_characteristic(
    schema: &DtdlSchema,
    interface_id: &str,
    property_name: &str,
) -> Result<Characteristic, SammError> {
    let schema_str = match schema {
        DtdlSchema::Primitive(s) => s.clone(),
        DtdlSchema::Object(_) => "object".to_string(),
    };

    // Build characteristic URN
    let char_urn = format!(
        "{}#{}Characteristic",
        interface_id.split(';').next().unwrap_or(interface_id),
        to_pascal_case(property_name)
    );

    let mut characteristic = Characteristic::new(char_urn, CharacteristicKind::Trait);

    // Map DTDL schema to XSD data type
    characteristic.data_type = Some(map_dtdl_to_xsd_type(&schema_str));

    Ok(characteristic)
}

/// Convert DTMI to SAMM URN
///
/// Converts a DTMI (Digital Twin Model Identifier) to SAMM URN format.
///
/// # Format
///
/// ```text
/// dtmi:com:example:Movement;1
/// → urn:samm:com.example:1.0.0#Movement
/// ```
///
/// # Arguments
///
/// * `dtmi` - DTMI string
///
/// # Returns
///
/// SAMM URN string
fn dtmi_to_urn(dtmi: &str) -> Result<String, SammError> {
    // dtmi:com:example:Movement;1
    // → urn:samm:com.example:1.0.0#Movement

    if !dtmi.starts_with("dtmi:") {
        return Err(SammError::ParseError(format!(
            "Invalid DTMI: {}. Expected format: dtmi:namespace:name;version",
            dtmi
        )));
    }

    let without_prefix = dtmi
        .strip_prefix("dtmi:")
        .expect("DTMI should start with 'dtmi:' prefix (validated earlier)");

    // Split by ';' to get version
    let parts: Vec<&str> = without_prefix.split(';').collect();
    if parts.len() != 2 {
        return Err(SammError::ParseError(format!(
            "Invalid DTMI format: {}. Expected ';' separator for version",
            dtmi
        )));
    }

    let path = parts[0];
    let major_version = parts[1];

    // Validate path is not empty
    if path.is_empty() {
        return Err(SammError::ParseError(format!(
            "Invalid DTMI: {}. Path cannot be empty",
            dtmi
        )));
    }

    // Split path by ':' to get namespace and name
    let path_parts: Vec<&str> = path.split(':').collect();
    if path_parts.len() < 2 {
        return Err(SammError::ParseError(format!(
            "Invalid DTMI path: {}. Expected at least namespace:name",
            path
        )));
    }

    let name = path_parts.last().expect("collection should not be empty");
    let namespace_parts = &path_parts[..path_parts.len() - 1];

    // Convert namespace (com:example → com.example)
    let namespace = namespace_parts.join(".");

    // Build version string (1 → 1.0.0)
    let version = format!("{}.0.0", major_version);

    // Build SAMM URN
    let urn = format!("urn:samm:{}:{}#{}", namespace, version, name);

    Ok(urn)
}

/// Build property URN from interface DTMI and property name
fn build_property_urn(interface_dtmi: &str, property_name: &str) -> Result<String, SammError> {
    let base_urn = dtmi_to_urn(interface_dtmi)?;
    let namespace_version = base_urn
        .strip_prefix("urn:samm:")
        .and_then(|s| s.split('#').next())
        .ok_or_else(|| SammError::ParseError("Invalid URN structure".to_string()))?;

    Ok(format!("urn:samm:{}#{}", namespace_version, property_name))
}

/// Build operation URN from interface DTMI and operation name
fn build_operation_urn(interface_dtmi: &str, operation_name: &str) -> Result<String, SammError> {
    build_property_urn(interface_dtmi, operation_name)
}

/// Map DTDL schema type to XSD data type
fn map_dtdl_to_xsd_type(dtdl_type: &str) -> String {
    // Handle complex DTDL schema URIs
    let base_type = if dtdl_type.starts_with("dtmi:dtdl:instance:Schema:") {
        dtdl_type
            .strip_prefix("dtmi:dtdl:instance:Schema:")
            .and_then(|s| s.split(';').next())
            .unwrap_or("Object")
    } else {
        dtdl_type
    };

    let xsd_type = match base_type {
        // Numeric types
        "integer" | "int" => "xsd:int",
        "long" => "xsd:long",
        "float" => "xsd:float",
        "double" => "xsd:double",

        // Boolean
        "boolean" | "bool" => "xsd:boolean",

        // String
        "string" => "xsd:string",

        // Date/Time types
        "dateTime" => "xsd:dateTime",
        "date" => "xsd:date",
        "time" => "xsd:time",
        "duration" => "xsd:duration",

        // Complex types
        "Object" => "xsd:string", // Complex object, map to string for now
        "Array" => "xsd:string",  // Array, map to string for now
        "Map" => "xsd:string",    // Map, map to string for now

        // Default
        _ => "xsd:string",
    };

    xsd_type.to_string()
}

/// Convert snake_case or camelCase to PascalCase
fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for ch in s.chars() {
        if ch == '_' || ch == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(
                ch.to_uppercase()
                    .next()
                    .expect("to_uppercase() always returns at least one character"),
            );
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::ModelElement;

    #[test]
    fn test_dtmi_to_urn_conversion() {
        assert_eq!(
            dtmi_to_urn("dtmi:com:example:Movement;1").expect("operation should succeed"),
            "urn:samm:com.example:1.0.0#Movement"
        );

        assert_eq!(
            dtmi_to_urn("dtmi:org:eclipse:esmf:Aspect;2").expect("operation should succeed"),
            "urn:samm:org.eclipse.esmf:2.0.0#Aspect"
        );

        assert_eq!(
            dtmi_to_urn("dtmi:io:github:oxirs:TestAspect;0").expect("operation should succeed"),
            "urn:samm:io.github.oxirs:0.0.0#TestAspect"
        );
    }

    #[test]
    fn test_dtmi_to_urn_invalid() {
        assert!(dtmi_to_urn("invalid:dtmi").is_err());
        assert!(dtmi_to_urn("dtmi:no-semicolon").is_err());
        assert!(dtmi_to_urn("dtmi:;1").is_err());
    }

    #[test]
    fn test_pascal_case_conversion() {
        assert_eq!(to_pascal_case("speed"), "Speed");
        assert_eq!(to_pascal_case("current_speed"), "CurrentSpeed");
        assert_eq!(to_pascal_case("currentSpeed"), "CurrentSpeed");
        assert_eq!(to_pascal_case("GPS_coordinates"), "GPSCoordinates");
    }

    #[test]
    fn test_dtdl_to_xsd_mapping() {
        assert_eq!(map_dtdl_to_xsd_type("string"), "xsd:string");
        assert_eq!(map_dtdl_to_xsd_type("integer"), "xsd:int");
        assert_eq!(map_dtdl_to_xsd_type("float"), "xsd:float");
        assert_eq!(map_dtdl_to_xsd_type("double"), "xsd:double");
        assert_eq!(map_dtdl_to_xsd_type("boolean"), "xsd:boolean");
        assert_eq!(map_dtdl_to_xsd_type("dateTime"), "xsd:dateTime");
        assert_eq!(
            map_dtdl_to_xsd_type("dtmi:dtdl:instance:Schema:Object;3"),
            "xsd:string"
        );
    }

    #[test]
    fn test_parse_minimal_interface() {
        let dtdl = r#"{
            "@context": "dtmi:dtdl:context;3",
            "@id": "dtmi:com:example:Movement;1",
            "@type": "Interface",
            "displayName": "Movement"
        }"#;

        let aspect = parse_dtdl_interface(dtdl).expect("DTDL parsing should succeed");
        assert_eq!(aspect.name(), "Movement");
        assert_eq!(aspect.metadata.urn, "urn:samm:com.example:1.0.0#Movement");
    }

    #[test]
    fn test_parse_interface_with_description() {
        let dtdl = r#"{
            "@context": "dtmi:dtdl:context;3",
            "@id": "dtmi:com:example:Movement;1",
            "@type": "Interface",
            "displayName": "Movement",
            "description": "Vehicle movement tracking"
        }"#;

        let aspect = parse_dtdl_interface(dtdl).expect("DTDL parsing should succeed");
        let desc = aspect.metadata.get_description("en");
        assert_eq!(desc, Some("Vehicle movement tracking"));
    }

    #[test]
    fn test_parse_interface_with_property() {
        let dtdl = r#"{
            "@context": "dtmi:dtdl:context;3",
            "@id": "dtmi:com:example:Movement;1",
            "@type": "Interface",
            "displayName": "Movement",
            "contents": [
                {
                    "@type": "Property",
                    "name": "speed",
                    "displayName": "speed",
                    "description": "Current speed",
                    "schema": "float"
                }
            ]
        }"#;

        let aspect = parse_dtdl_interface(dtdl).expect("DTDL parsing should succeed");
        assert_eq!(aspect.properties().len(), 1);

        let prop = &aspect.properties()[0];
        assert_eq!(prop.name(), "speed");
        assert!(!prop.optional); // Property is writable, not optional
        assert!(prop.characteristic.is_some());

        let char = prop
            .characteristic
            .as_ref()
            .expect("reference should be available");
        assert_eq!(char.data_type, Some("xsd:float".to_string()));
    }

    #[test]
    fn test_parse_interface_with_telemetry() {
        let dtdl = r#"{
            "@context": "dtmi:dtdl:context;3",
            "@id": "dtmi:com:example:Sensor;1",
            "@type": "Interface",
            "displayName": "Sensor",
            "contents": [
                {
                    "@type": "Telemetry",
                    "name": "temperature",
                    "schema": "double"
                }
            ]
        }"#;

        let aspect = parse_dtdl_interface(dtdl).expect("DTDL parsing should succeed");
        assert_eq!(aspect.properties().len(), 1);

        let prop = &aspect.properties()[0];
        assert_eq!(prop.name(), "temperature");
        assert!(prop.optional); // Telemetry is read-only
    }

    #[test]
    fn test_parse_interface_with_command() {
        let dtdl = r#"{
            "@context": "dtmi:dtdl:context;3",
            "@id": "dtmi:com:example:Movement;1",
            "@type": "Interface",
            "displayName": "Movement",
            "contents": [
                {
                    "@type": "Command",
                    "name": "emergencyStop",
                    "displayName": "Emergency Stop",
                    "description": "Stops the vehicle immediately"
                }
            ]
        }"#;

        let aspect = parse_dtdl_interface(dtdl).expect("DTDL parsing should succeed");
        assert_eq!(aspect.operations().len(), 1);

        let op = &aspect.operations()[0];
        assert_eq!(op.name(), "emergencyStop");
        let display_name = op.metadata.get_preferred_name("en");
        assert_eq!(display_name, Some("Emergency Stop"));
    }

    #[test]
    fn test_parse_invalid_json() {
        let invalid = "{ invalid json }";
        assert!(parse_dtdl_interface(invalid).is_err());
    }

    #[test]
    fn test_parse_non_interface() {
        let dtdl = r#"{
            "@context": "dtmi:dtdl:context;3",
            "@id": "dtmi:com:example:Thing;1",
            "@type": "NotAnInterface"
        }"#;

        assert!(parse_dtdl_interface(dtdl).is_err());
    }
}
