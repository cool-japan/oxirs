//! DTDL v3 Core Types
//!
//! Digital Twin Model Identifier (DTMI) and typed element structs for
//! DTDL v3 (Interface, Telemetry, Property, Command, Component, Relationship).
//!
//! # DTMI Format
//!
//! ```text
//! dtmi:<path-segment>[:<path-segment>]*;<version>
//! ```
//!
//! where `<version>` is a positive integer and each path segment is
//! alphanumeric + underscore only.
//!
//! # Design Notes
//!
//! DTDL v3 allows the `@type` field to be **either** a plain string
//! (`"Telemetry"`) or an array that includes a semantic type annotation
//! (`["Telemetry", "Temperature"]`).  All `element_type` fields therefore
//! use `serde_json::Value` for deserialization and expose helpers to
//! extract the primary type string.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// DTMI
// ─────────────────────────────────────────────────────────────────────────────

/// Digital Twin Model Identifier, e.g. `"dtmi:example:Thermostat;1"`.
///
/// # Format
///
/// ```text
/// dtmi:<seg>[:<seg>...]*;<version>
/// ```
///
/// Each `<seg>` must be non-empty and consist of alphanumeric characters
/// and underscores only.  `<version>` must be a non-negative integer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dtmi(pub String);

impl Dtmi {
    /// Validate DTMI format per the DTDL v3 specification.
    pub fn validate(&self) -> Result<(), DtdlValidationError> {
        let s = &self.0;

        if !s.starts_with("dtmi:") {
            return Err(DtdlValidationError::InvalidDtmi {
                dtmi: s.clone(),
                reason: "must start with 'dtmi:'",
            });
        }

        let semicolon_pos = s
            .rfind(';')
            .ok_or_else(|| DtdlValidationError::InvalidDtmi {
                dtmi: s.clone(),
                reason: "missing version number after ';'",
            })?;

        let version_str = &s[semicolon_pos + 1..];
        version_str
            .parse::<u32>()
            .map_err(|_| DtdlValidationError::InvalidDtmi {
                dtmi: s.clone(),
                reason: "version must be a non-negative integer",
            })?;

        // Path: the part between "dtmi:" and ";<version>"
        let path = &s[5..semicolon_pos];
        if path.is_empty() {
            return Err(DtdlValidationError::InvalidDtmi {
                dtmi: s.clone(),
                reason: "path must not be empty",
            });
        }

        for segment in path.split(':') {
            if segment.is_empty()
                || !segment
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return Err(DtdlValidationError::InvalidDtmi {
                    dtmi: s.clone(),
                    reason: "path segments must be non-empty alphanumeric/underscore",
                });
            }
        }

        Ok(())
    }

    /// Extract the version number from the DTMI, if well-formed.
    pub fn version(&self) -> Option<u32> {
        self.0
            .rfind(';')
            .and_then(|pos| self.0[pos + 1..].parse().ok())
    }

    /// Return the path portion (between `"dtmi:"` and `";<version>"`).
    pub fn path(&self) -> Option<&str> {
        let s = &self.0;
        let start = s.strip_prefix("dtmi:")?;
        let semi = start.rfind(';')?;
        Some(&start[..semi])
    }
}

impl std::fmt::Display for Dtmi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Element-type helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the primary element type from a DTDL `@type` value.
///
/// DTDL v3 allows `@type` to be either:
/// - a plain string: `"Telemetry"`
/// - an array: `["Telemetry", "Temperature"]`
///
/// This function returns the first string in either form, or `None` if the
/// value is neither.
pub fn primary_type(value: &serde_json::Value) -> Option<&str> {
    match value {
        serde_json::Value::String(s) => Some(s.as_str()),
        serde_json::Value::Array(arr) => arr.first().and_then(|v| v.as_str()),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DTDL element types
// ─────────────────────────────────────────────────────────────────────────────

/// DTDL v3 schema type as a plain string.
///
/// Primitive schemas: `"boolean"`, `"date"`, `"dateTime"`, `"double"`,
/// `"duration"`, `"float"`, `"integer"`, `"long"`, `"string"`, `"time"`.
/// Complex schema references are represented as DTMI strings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DtdlSchema(pub String);

impl DtdlSchema {
    /// Map to an XSD datatype URI.
    pub fn to_xsd_uri(&self) -> &'static str {
        match self.0.to_lowercase().as_str() {
            "boolean" => "http://www.w3.org/2001/XMLSchema#boolean",
            "date" => "http://www.w3.org/2001/XMLSchema#date",
            "datetime" => "http://www.w3.org/2001/XMLSchema#dateTime",
            "double" => "http://www.w3.org/2001/XMLSchema#double",
            "duration" => "http://www.w3.org/2001/XMLSchema#duration",
            "float" => "http://www.w3.org/2001/XMLSchema#float",
            "integer" => "http://www.w3.org/2001/XMLSchema#integer",
            "long" => "http://www.w3.org/2001/XMLSchema#long",
            "string" => "http://www.w3.org/2001/XMLSchema#string",
            "time" => "http://www.w3.org/2001/XMLSchema#time",
            _ => "http://www.w3.org/2001/XMLSchema#anyType",
        }
    }
}

/// DTDL v3 Telemetry element (time-series observable value).
#[derive(Debug, Clone)]
pub struct DtdlTelemetryElement {
    /// Raw `@type` value (string or array).
    pub element_type: serde_json::Value,
    /// Optional `@id` DTMI.
    pub id: Option<Dtmi>,
    /// Name of the telemetry field (camelCase per DTDL spec).
    pub name: String,
    /// Schema / data type.
    pub schema: DtdlSchema,
    /// Optional unit annotation (e.g. `"Celsius"`, `"watt"`).
    pub unit: Option<String>,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Optional comment.
    pub comment: Option<String>,
}

/// DTDL v3 Property element (settable configuration / state).
#[derive(Debug, Clone)]
pub struct DtdlPropertyElement {
    /// Raw `@type` value.
    pub element_type: serde_json::Value,
    /// Optional `@id` DTMI.
    pub id: Option<Dtmi>,
    /// Property name.
    pub name: String,
    /// Schema / data type.
    pub schema: DtdlSchema,
    /// Whether the twin client can write this property.
    pub writable: Option<bool>,
    /// Optional unit annotation.
    pub unit: Option<String>,
    /// Optional description.
    pub description: Option<String>,
}

/// DTDL v3 Command element (request/response operation).
#[derive(Debug, Clone)]
pub struct DtdlCommandElement {
    /// Raw `@type` value.
    pub element_type: serde_json::Value,
    /// Command name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
}

/// DTDL v3 Component element (composition via another Interface DTMI).
#[derive(Debug, Clone)]
pub struct DtdlComponentElement {
    /// Raw `@type` value.
    pub element_type: serde_json::Value,
    /// Component name.
    pub name: String,
    /// DTMI of the referenced Interface.
    pub schema: Dtmi,
}

/// DTDL v3 Relationship element.
#[derive(Debug, Clone)]
pub struct DtdlRelationshipElement {
    /// Raw `@type` value.
    pub element_type: serde_json::Value,
    /// Relationship name.
    pub name: String,
    /// Optional DTMI of the target Interface.
    pub target: Option<Dtmi>,
    /// Optional description.
    pub description: Option<String>,
}

/// Union of all DTDL v3 content element variants.
#[derive(Debug, Clone)]
pub enum DtdlContent {
    /// Time-series telemetry.
    Telemetry(DtdlTelemetryElement),
    /// Read/write property.
    Property(DtdlPropertyElement),
    /// Command operation.
    Command(DtdlCommandElement),
    /// Sub-component reference.
    Component(DtdlComponentElement),
    /// Directed relationship to other twins.
    Relationship(DtdlRelationshipElement),
}

/// DTDL v3 Interface — top-level model document.
#[derive(Debug, Clone)]
pub struct DtdlInterface {
    /// `@context` value (string or array per JSON-LD).
    pub context: serde_json::Value,
    /// Must be `"Interface"` (or an array containing `"Interface"`).
    pub element_type: serde_json::Value,
    /// DTMI identifier.
    pub id: Dtmi,
    /// Optional display name (string or language-map object).
    pub display_name: Option<serde_json::Value>,
    /// Optional description.
    pub description: Option<String>,
    /// Optional comment.
    pub comment: Option<String>,
    /// Content elements (Telemetry, Property, Command, Component, Relationship).
    pub contents: Option<Vec<DtdlContent>>,
    /// Optional complex schema definitions.
    pub schemas: Option<Vec<serde_json::Value>>,
    /// Optional base interface(s) this Interface extends.
    pub extends: Option<serde_json::Value>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by DTDL validation.
#[derive(Debug, Error)]
pub enum DtdlValidationError {
    /// The DTMI string is malformed.
    #[error("invalid DTMI '{dtmi}': {reason}")]
    InvalidDtmi { dtmi: String, reason: &'static str },

    /// A required field is missing or empty.
    #[error("missing required field: {field}")]
    MissingField { field: &'static str },

    /// The DTDL version is not supported.
    #[error("unsupported DTDL version '{version}'; only v3 is fully supported")]
    UnsupportedVersion { version: String },

    /// Schema type mismatch.
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtmi_valid_simple() {
        let d = Dtmi("dtmi:example:Thermostat;1".into());
        assert!(d.validate().is_ok());
    }

    #[test]
    fn dtmi_valid_multi_segment() {
        let d = Dtmi("dtmi:com:example:devices:Sensor;42".into());
        assert!(d.validate().is_ok());
    }

    #[test]
    fn dtmi_version_extracted() {
        let d = Dtmi("dtmi:example:Foo;7".into());
        assert_eq!(d.version(), Some(7));
    }

    #[test]
    fn dtmi_path_extracted() {
        let d = Dtmi("dtmi:example:Foo;1".into());
        assert_eq!(d.path(), Some("example:Foo"));
    }

    #[test]
    fn dtmi_missing_prefix_fails() {
        let d = Dtmi("http://example.org/Foo;1".into());
        assert!(d.validate().is_err());
    }

    #[test]
    fn dtmi_missing_version_fails() {
        let d = Dtmi("dtmi:example:Foo".into());
        assert!(d.validate().is_err());
    }

    #[test]
    fn dtmi_non_integer_version_fails() {
        let d = Dtmi("dtmi:example:Foo;abc".into());
        assert!(d.validate().is_err());
    }

    #[test]
    fn dtmi_empty_segment_fails() {
        let d = Dtmi("dtmi::Foo;1".into());
        assert!(d.validate().is_err());
    }

    #[test]
    fn dtmi_special_chars_in_segment_fail() {
        let d = Dtmi("dtmi:example:Foo-Bar;1".into());
        assert!(d.validate().is_err());
    }

    #[test]
    fn dtmi_empty_path_fails() {
        let d = Dtmi("dtmi:;1".into());
        assert!(d.validate().is_err());
    }

    #[test]
    fn dtdl_schema_to_xsd_double() {
        let s = DtdlSchema("double".into());
        assert_eq!(s.to_xsd_uri(), "http://www.w3.org/2001/XMLSchema#double");
    }

    #[test]
    fn dtdl_schema_to_xsd_string() {
        let s = DtdlSchema("string".into());
        assert_eq!(s.to_xsd_uri(), "http://www.w3.org/2001/XMLSchema#string");
    }

    #[test]
    fn dtdl_schema_unknown_is_anytype() {
        let s = DtdlSchema("complexEnumRef".into());
        assert_eq!(s.to_xsd_uri(), "http://www.w3.org/2001/XMLSchema#anyType");
    }

    #[test]
    fn primary_type_string() {
        let v = serde_json::json!("Telemetry");
        assert_eq!(primary_type(&v), Some("Telemetry"));
    }

    #[test]
    fn primary_type_array() {
        let v = serde_json::json!(["Telemetry", "Temperature"]);
        assert_eq!(primary_type(&v), Some("Telemetry"));
    }

    #[test]
    fn primary_type_null_returns_none() {
        let v = serde_json::Value::Null;
        assert_eq!(primary_type(&v), None);
    }
}
