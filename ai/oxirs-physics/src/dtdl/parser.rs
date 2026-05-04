//! DTDL v3 JSON-LD Parser
//!
//! Parses a DTDL v3 JSON document (an Interface) into strongly-typed Rust
//! structs, handling the DTDL v3 peculiarity where the `@type` field may be
//! either a plain string or an array of strings (semantic type tagging).
//!
//! # Approach
//!
//! Rather than using `#[serde(tag)]` (which requires a scalar discriminant)
//! we parse into a raw `serde_json::Value` first and then walk the tree
//! manually.  This gives us full control over all DTDL edge cases without
//! fighting serde's internal machinery.

use super::types::{
    primary_type, DtdlCommandElement, DtdlComponentElement, DtdlContent, DtdlInterface,
    DtdlPropertyElement, DtdlRelationshipElement, DtdlSchema, DtdlTelemetryElement,
    DtdlValidationError, Dtmi,
};
use serde_json::Value;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced during DTDL parsing.
#[derive(Debug, Error)]
pub enum DtdlParseError {
    /// The input string is not valid JSON.
    #[error("invalid JSON: {0}")]
    InvalidJson(String),

    /// The JSON is valid but does not conform to the DTDL Interface schema.
    #[error("invalid DTDL structure: {0}")]
    InvalidStructure(String),

    /// A DTMI field value failed validation.
    #[error("invalid DTMI: {0}")]
    InvalidDtmi(String),
}

impl From<DtdlValidationError> for DtdlParseError {
    fn from(e: DtdlValidationError) -> Self {
        Self::InvalidDtmi(e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a DTDL v3 Interface JSON document.
///
/// # Errors
///
/// Returns [`DtdlParseError`] if the input is not valid JSON, does not
/// represent a DTDL Interface, or contains an invalid DTMI.
///
/// # Version policy
///
/// Both v2 (`dtmi:dtdl:context;2`) and v3 (`dtmi:dtdl:context;3`) context
/// URIs are accepted.  The function does **not** fail on v2 documents; the
/// caller can inspect [`DtdlInterface::context`] if the version matters.
pub fn parse_dtdl_interface(json: &str) -> Result<DtdlInterface, DtdlParseError> {
    let root: Value =
        serde_json::from_str(json).map_err(|e| DtdlParseError::InvalidJson(e.to_string()))?;

    parse_interface_value(&root)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a `serde_json::Value` that must be a DTDL Interface object.
fn parse_interface_value(root: &Value) -> Result<DtdlInterface, DtdlParseError> {
    let obj = root
        .as_object()
        .ok_or_else(|| DtdlParseError::InvalidStructure("document must be a JSON object".into()))?;

    // @type must be "Interface" (possibly inside an array)
    let type_val = obj.get("@type").cloned().unwrap_or(Value::Null);
    let primary = primary_type(&type_val);
    if primary != Some("Interface") {
        return Err(DtdlParseError::InvalidStructure(format!(
            "expected @type 'Interface', got {}",
            type_val
        )));
    }

    // @id is required
    let id_str = obj
        .get("@id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DtdlParseError::InvalidStructure("missing required field '@id'".into()))?;

    let id = Dtmi(id_str.to_owned());
    id.validate()
        .map_err(|e| DtdlParseError::InvalidDtmi(e.to_string()))?;

    let context = obj
        .get("@context")
        .cloned()
        .unwrap_or(Value::String("dtmi:dtdl:context;3".into()));

    let display_name = obj
        .get("displayName")
        .or_else(|| obj.get("display_name"))
        .cloned();

    let description = obj
        .get("description")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    let comment = obj
        .get("comment")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    let contents = parse_contents(obj.get("contents"))?;

    let schemas = obj
        .get("schemas")
        .and_then(|v| v.as_array())
        .map(|arr| arr.to_vec());

    let extends = obj.get("extends").cloned();

    Ok(DtdlInterface {
        context,
        element_type: type_val,
        id,
        display_name,
        description,
        comment,
        contents,
        schemas,
        extends,
    })
}

/// Parse the optional `contents` array into a `Vec<DtdlContent>`.
fn parse_contents(value: Option<&Value>) -> Result<Option<Vec<DtdlContent>>, DtdlParseError> {
    let arr = match value {
        None => return Ok(None),
        Some(Value::Array(a)) => a,
        Some(other) => {
            return Err(DtdlParseError::InvalidStructure(format!(
                "'contents' must be an array, got {other}"
            )))
        }
    };

    let mut result = Vec::with_capacity(arr.len());
    for item in arr {
        match parse_content_item(item) {
            Ok(Some(content)) => result.push(content),
            Ok(None) => { /* unknown type — skip gracefully */ }
            Err(e) => return Err(e),
        }
    }
    Ok(Some(result))
}

/// Parse a single content item into the appropriate [`DtdlContent`] variant.
///
/// Returns `Ok(None)` for unrecognised element types (forward compatibility).
fn parse_content_item(item: &Value) -> Result<Option<DtdlContent>, DtdlParseError> {
    let obj = item.as_object().ok_or_else(|| {
        DtdlParseError::InvalidStructure("content item must be a JSON object".into())
    })?;

    let type_val = obj.get("@type").cloned().unwrap_or(Value::Null);
    let primary = primary_type(&type_val).unwrap_or("").to_lowercase();

    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();

    let id = parse_optional_dtmi(obj, "@id")?;

    let description = obj
        .get("description")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    match primary.as_str() {
        "telemetry" => {
            let schema = parse_schema(obj.get("schema"))?;
            let unit = obj.get("unit").and_then(|v| v.as_str()).map(str::to_owned);
            let comment = obj
                .get("comment")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            Ok(Some(DtdlContent::Telemetry(DtdlTelemetryElement {
                element_type: type_val,
                id,
                name,
                schema,
                unit,
                description,
                comment,
            })))
        }
        "property" => {
            let schema = parse_schema(obj.get("schema"))?;
            let writable = obj.get("writable").and_then(|v| v.as_bool());
            let unit = obj.get("unit").and_then(|v| v.as_str()).map(str::to_owned);
            Ok(Some(DtdlContent::Property(DtdlPropertyElement {
                element_type: type_val,
                id,
                name,
                schema,
                writable,
                unit,
                description,
            })))
        }
        "command" => Ok(Some(DtdlContent::Command(DtdlCommandElement {
            element_type: type_val,
            name,
            description,
        }))),
        "component" => {
            let schema_str = obj.get("schema").and_then(|v| v.as_str()).ok_or_else(|| {
                DtdlParseError::InvalidStructure(
                    "Component element requires 'schema' (DTMI string)".into(),
                )
            })?;
            let schema_dtmi = Dtmi(schema_str.to_owned());
            schema_dtmi
                .validate()
                .map_err(|e| DtdlParseError::InvalidDtmi(e.to_string()))?;
            Ok(Some(DtdlContent::Component(DtdlComponentElement {
                element_type: type_val,
                name,
                schema: schema_dtmi,
            })))
        }
        "relationship" => {
            let target = parse_optional_dtmi(obj, "target")?;
            Ok(Some(DtdlContent::Relationship(DtdlRelationshipElement {
                element_type: type_val,
                name,
                target,
                description,
            })))
        }
        _ => Ok(None),
    }
}

/// Extract an optional DTMI from the given field key.
fn parse_optional_dtmi(
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<Dtmi>, DtdlParseError> {
    match obj.get(key).and_then(|v| v.as_str()) {
        None => Ok(None),
        Some(s) => {
            let dtmi = Dtmi(s.to_owned());
            dtmi.validate()
                .map_err(|e| DtdlParseError::InvalidDtmi(e.to_string()))?;
            Ok(Some(dtmi))
        }
    }
}

/// Parse a schema value into [`DtdlSchema`].
///
/// Supports plain strings (`"double"`) and object schemas (stored verbatim
/// as their `@id` or the serialised form).
fn parse_schema(value: Option<&Value>) -> Result<DtdlSchema, DtdlParseError> {
    match value {
        None | Some(Value::Null) => Ok(DtdlSchema("string".into())),
        Some(Value::String(s)) => Ok(DtdlSchema(s.clone())),
        Some(Value::Object(obj)) => {
            // Complex inline schema — use the @id if present, else a placeholder
            let id = obj
                .get("@id")
                .and_then(|v| v.as_str())
                .unwrap_or("complexSchema");
            Ok(DtdlSchema(id.to_owned()))
        }
        Some(other) => Err(DtdlParseError::InvalidStructure(format!(
            "schema must be a string or object, got {other}"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_interface() {
        let json = r#"{
            "@context": "dtmi:dtdl:context;3",
            "@type": "Interface",
            "@id": "dtmi:example:Simple;1",
            "displayName": "Simple"
        }"#;
        let iface = parse_dtdl_interface(json).expect("should parse");
        assert_eq!(iface.id.0, "dtmi:example:Simple;1");
    }

    #[test]
    fn parse_invalid_json_errors() {
        let result = parse_dtdl_interface("{ not json }");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DtdlParseError::InvalidJson(_)));
    }

    #[test]
    fn parse_wrong_type_errors() {
        let json = r#"{"@type": "Property", "@id": "dtmi:x:y;1"}"#;
        let result = parse_dtdl_interface(json);
        assert!(result.is_err());
    }

    #[test]
    fn parse_missing_id_errors() {
        let json = r#"{"@type": "Interface"}"#;
        let result = parse_dtdl_interface(json);
        assert!(result.is_err());
    }

    #[test]
    fn parse_invalid_dtmi_errors() {
        let json = r#"{"@type": "Interface", "@id": "bad-id"}"#;
        let result = parse_dtdl_interface(json);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DtdlParseError::InvalidDtmi(_)));
    }

    #[test]
    fn parse_telemetry_array_type() {
        // @type is an array (DTDL v3 semantic type tagging)
        let json = r#"{
            "@context": "dtmi:dtdl:context;3",
            "@type": "Interface",
            "@id": "dtmi:example:Temp;1",
            "contents": [
                {
                    "@type": ["Telemetry", "Temperature"],
                    "name": "temp",
                    "schema": "double",
                    "unit": "Celsius"
                }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("should parse");
        let contents = iface.contents.expect("should have contents");
        assert_eq!(contents.len(), 1);
        assert!(matches!(contents[0], DtdlContent::Telemetry(_)));
        if let DtdlContent::Telemetry(ref t) = contents[0] {
            assert_eq!(t.name, "temp");
            assert_eq!(t.schema.0, "double");
            assert_eq!(t.unit.as_deref(), Some("Celsius"));
        }
    }

    #[test]
    fn parse_property_element() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Device;1",
            "contents": [
                { "@type": "Property", "name": "setPoint", "schema": "double", "writable": true }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let contents = iface.contents.expect("contents");
        assert!(matches!(contents[0], DtdlContent::Property(_)));
        if let DtdlContent::Property(ref p) = contents[0] {
            assert_eq!(p.writable, Some(true));
        }
    }

    #[test]
    fn parse_command_element() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Device;1",
            "contents": [
                { "@type": "Command", "name": "reboot" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let contents = iface.contents.expect("contents");
        assert!(matches!(contents[0], DtdlContent::Command(_)));
    }

    #[test]
    fn parse_relationship_element() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Building;1",
            "contents": [
                { "@type": "Relationship", "name": "contains", "target": "dtmi:test:Room;1" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let contents = iface.contents.expect("contents");
        assert!(matches!(contents[0], DtdlContent::Relationship(_)));
        if let DtdlContent::Relationship(ref r) = contents[0] {
            assert_eq!(
                r.target.as_ref().map(|d| d.0.as_str()),
                Some("dtmi:test:Room;1")
            );
        }
    }

    #[test]
    fn parse_unknown_content_type_skipped() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Device;1",
            "contents": [
                { "@type": "FutureElement", "name": "futureThing" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let contents = iface.contents.expect("contents");
        // Unknown type skipped gracefully
        assert_eq!(contents.len(), 0);
    }

    #[test]
    fn parse_v2_context_accepted() {
        let json = r#"{
            "@context": "dtmi:dtdl:context;2",
            "@type": "Interface",
            "@id": "dtmi:com:example:Thermostat;1",
            "displayName": "Thermostat"
        }"#;
        let iface = parse_dtdl_interface(json).expect("v2 context should be accepted");
        assert_eq!(iface.id.version(), Some(1));
    }
}
