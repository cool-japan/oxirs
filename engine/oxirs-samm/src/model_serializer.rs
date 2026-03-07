//! SAMM aspect model serialization to multiple formats.
//!
//! Provides `ModelSerializer` which can serialize a `AspectModel` into JSON,
//! YAML, Turtle (minimal), and compact key=value formats, and can round-trip
//! through JSON by basic parsing.

use crate::aspect_validator::{AspectModel, AspectProperty, Cardinality};
use std::fmt::Write as FmtWrite;

// ── Format enum ───────────────────────────────────────────────────────────────

/// Supported serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// JSON object representation.
    Json,
    /// YAML document representation.
    Yaml,
    /// Minimal Turtle/N3 representation with `samm:` prefix.
    Turtle,
    /// Compact `key=value` line-per-field representation.
    Compact,
}

impl std::fmt::Display for SerializationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationFormat::Json => write!(f, "json"),
            SerializationFormat::Yaml => write!(f, "yaml"),
            SerializationFormat::Turtle => write!(f, "turtle"),
            SerializationFormat::Compact => write!(f, "compact"),
        }
    }
}

// ── Output type ───────────────────────────────────────────────────────────────

/// The output of a serialization operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializedModel {
    /// Format that was used.
    pub format: SerializationFormat,
    /// Serialized text content.
    pub content: String,
    /// Byte-length of the content.
    pub size_bytes: usize,
}

impl SerializedModel {
    fn new(format: SerializationFormat, content: String) -> Self {
        let size_bytes = content.len();
        SerializedModel {
            format,
            content,
            size_bytes,
        }
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors returned by serialization operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializeError {
    /// The aspect model has an empty name.
    EmptyName,
    /// A property has an invalid configuration.
    InvalidProperty(String),
    /// A format-level rendering error.
    FormatError(String),
}

impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializeError::EmptyName => write!(f, "aspect model name must not be empty"),
            SerializeError::InvalidProperty(msg) => write!(f, "invalid property: {msg}"),
            SerializeError::FormatError(msg) => write!(f, "format error: {msg}"),
        }
    }
}

impl std::error::Error for SerializeError {}

// ── Serializer ────────────────────────────────────────────────────────────────

/// Serializes `AspectModel` values into multiple textual formats.
pub struct ModelSerializer;

impl ModelSerializer {
    /// Create a new `ModelSerializer`.
    pub fn new() -> Self {
        ModelSerializer
    }

    // ── Validation helper ─────────────────────────────────────────────────────

    fn validate_model(model: &AspectModel) -> Result<(), SerializeError> {
        if model.name.trim().is_empty() {
            return Err(SerializeError::EmptyName);
        }
        for prop in &model.properties {
            if prop.name.trim().is_empty() {
                return Err(SerializeError::InvalidProperty(
                    "property name must not be empty".to_string(),
                ));
            }
        }
        Ok(())
    }

    // ── JSON ──────────────────────────────────────────────────────────────────

    /// Serialize the model to a JSON string.
    ///
    /// The JSON object contains `name`, `version` (derived from description when
    /// prefixed with `version:`, otherwise `"1.0.0"`), `description`, and a
    /// `properties` array of `{name, cardinality, characteristic_ref}` objects.
    pub fn serialize_json(model: &AspectModel) -> Result<SerializedModel, SerializeError> {
        Self::validate_model(model)?;

        let version = derive_version(model);
        let description = model.description.as_deref().unwrap_or("");
        let mut out = String::new();

        write_or_err(&mut out, "{\n")?;
        write_or_err(
            &mut out,
            &format!("  \"name\": {},\n", json_string(&model.name)),
        )?;
        write_or_err(
            &mut out,
            &format!("  \"version\": {},\n", json_string(&version)),
        )?;
        write_or_err(
            &mut out,
            &format!("  \"description\": {},\n", json_string(description)),
        )?;

        write_or_err(&mut out, "  \"properties\": [")?;
        for (i, prop) in model.properties.iter().enumerate() {
            if i > 0 {
                write_or_err(&mut out, ",")?;
            }
            write_or_err(&mut out, "\n    {")?;
            write_or_err(
                &mut out,
                &format!("\n      \"name\": {},", json_string(&prop.name)),
            )?;
            write_or_err(
                &mut out,
                &format!(
                    "\n      \"cardinality\": {},",
                    json_string(cardinality_str(prop))
                ),
            )?;
            write_or_err(
                &mut out,
                &format!(
                    "\n      \"characteristic_ref\": {}",
                    json_string(&prop.characteristic_ref)
                ),
            )?;
            write_or_err(&mut out, "\n    }")?;
        }
        if model.properties.is_empty() {
            write_or_err(&mut out, "]")?;
        } else {
            write_or_err(&mut out, "\n  ]")?;
        }

        write_or_err(&mut out, "\n}\n")?;

        Ok(SerializedModel::new(SerializationFormat::Json, out))
    }

    // ── YAML ──────────────────────────────────────────────────────────────────

    /// Serialize the model to a YAML document.
    pub fn serialize_yaml(model: &AspectModel) -> Result<SerializedModel, SerializeError> {
        Self::validate_model(model)?;

        let version = derive_version(model);
        let description = model.description.as_deref().unwrap_or("");
        let mut out = String::new();

        write_or_err(&mut out, "---\n")?;
        write_or_err(&mut out, &format!("name: {}\n", yaml_scalar(&model.name)))?;
        write_or_err(&mut out, &format!("version: {}\n", yaml_scalar(&version)))?;
        write_or_err(
            &mut out,
            &format!("description: {}\n", yaml_scalar(description)),
        )?;
        write_or_err(&mut out, "properties:\n")?;

        for prop in &model.properties {
            write_or_err(
                &mut out,
                &format!("  - name: {}\n", yaml_scalar(&prop.name)),
            )?;
            write_or_err(
                &mut out,
                &format!("    cardinality: {}\n", yaml_scalar(cardinality_str(prop))),
            )?;
            write_or_err(
                &mut out,
                &format!(
                    "    characteristic_ref: {}\n",
                    yaml_scalar(&prop.characteristic_ref)
                ),
            )?;
        }

        Ok(SerializedModel::new(SerializationFormat::Yaml, out))
    }

    // ── Turtle ────────────────────────────────────────────────────────────────

    /// Serialize the model to a minimal Turtle document using the `samm:` prefix.
    pub fn serialize_turtle(model: &AspectModel) -> Result<SerializedModel, SerializeError> {
        Self::validate_model(model)?;

        let version = derive_version(model);
        let description = model.description.as_deref().unwrap_or("");
        let mut out = String::new();

        write_or_err(
            &mut out,
            "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .\n\n",
        )?;

        // Aspect node
        write_or_err(
            &mut out,
            &format!(":{} a samm:Aspect ;\n", turtle_ident(&model.name)),
        )?;
        write_or_err(
            &mut out,
            &format!("    samm:name {} ;\n", turtle_literal(&model.name)),
        )?;
        write_or_err(
            &mut out,
            &format!("    samm:version {} ;\n", turtle_literal(&version)),
        )?;
        if !description.is_empty() {
            write_or_err(
                &mut out,
                &format!("    samm:description {} ;\n", turtle_literal(description)),
            )?;
        }

        // properties list
        if model.properties.is_empty() {
            write_or_err(&mut out, "    samm:properties () .\n")?;
        } else {
            let prop_refs: Vec<String> = model
                .properties
                .iter()
                .map(|p| format!(":{}", turtle_ident(&p.name)))
                .collect();
            write_or_err(
                &mut out,
                &format!("    samm:properties ( {} ) .\n", prop_refs.join(" ")),
            )?;
        }

        // Individual property definitions
        for prop in &model.properties {
            write_or_err(&mut out, "\n")?;
            write_or_err(
                &mut out,
                &format!(":{} a samm:Property ;\n", turtle_ident(&prop.name)),
            )?;
            write_or_err(
                &mut out,
                &format!(
                    "    samm:characteristic :{} ;\n",
                    turtle_ident(&prop.characteristic_ref)
                ),
            )?;
            let optional_val = matches!(prop.cardinality, Cardinality::Optional)
                .then_some("true")
                .unwrap_or("false");
            write_or_err(&mut out, &format!("    samm:optional {} .\n", optional_val))?;
        }

        Ok(SerializedModel::new(SerializationFormat::Turtle, out))
    }

    // ── Compact ───────────────────────────────────────────────────────────────

    /// Serialize the model to compact `key=value` lines.
    pub fn serialize_compact(model: &AspectModel) -> Result<SerializedModel, SerializeError> {
        Self::validate_model(model)?;

        let version = derive_version(model);
        let description = model.description.as_deref().unwrap_or("");
        let mut out = String::new();

        write_or_err(&mut out, &format!("name={}\n", model.name))?;
        write_or_err(&mut out, &format!("version={version}\n"))?;
        write_or_err(&mut out, &format!("description={description}\n"))?;
        write_or_err(
            &mut out,
            &format!("property_count={}\n", model.properties.len()),
        )?;

        for prop in &model.properties {
            write_or_err(
                &mut out,
                &format!(
                    "property={}:{}:{}\n",
                    prop.name,
                    cardinality_str(prop),
                    prop.characteristic_ref
                ),
            )?;
        }

        Ok(SerializedModel::new(SerializationFormat::Compact, out))
    }

    // ── Round-trip ────────────────────────────────────────────────────────────

    /// Serialize to JSON then parse the result back and verify the name matches.
    ///
    /// Returns `true` when the name field decoded from JSON equals `model.name`.
    pub fn round_trip_json(model: &AspectModel) -> Result<bool, SerializeError> {
        let serialized = Self::serialize_json(model)?;
        let parsed_name = parse_json_name(&serialized.content).ok_or_else(|| {
            SerializeError::FormatError(
                "could not extract `name` field from serialized JSON".to_string(),
            )
        })?;
        Ok(parsed_name == model.name)
    }
}

impl Default for ModelSerializer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Escape a string for inclusion as a JSON string value.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Produce a safe YAML scalar (quoted when necessary).
fn yaml_scalar(s: &str) -> String {
    // Simple heuristic: quote if contains special YAML chars
    let needs_quoting = s.is_empty()
        || s.starts_with([':', '-', '#', '&', '*', '!', '|', '>', '\'', '"', '{', '['])
        || s.contains('\n')
        || s.contains(": ")
        || s.starts_with(' ')
        || s.ends_with(' ');

    if needs_quoting {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        s.to_string()
    }
}

/// Produce a valid Turtle / N3 local name for an identifier.
fn turtle_ident(s: &str) -> String {
    // Replace characters not valid in a local name with `_`
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Produce a Turtle string literal (double-quoted).
fn turtle_literal(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Return the cardinality label string for a property.
fn cardinality_str(prop: &AspectProperty) -> &'static str {
    match prop.cardinality {
        Cardinality::Optional => "optional",
        Cardinality::Mandatory => "mandatory",
    }
}

/// Derive a version string from the model description or default to `"1.0.0"`.
fn derive_version(model: &AspectModel) -> String {
    model
        .description
        .as_deref()
        .and_then(|d| d.strip_prefix("version:"))
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| "1.0.0".to_string())
}

/// Parse the `"name"` field from a hand-built JSON object (simple, no deps).
fn parse_json_name(json: &str) -> Option<String> {
    // Find `"name"` key followed by `: "value"`
    let key_pos = json.find("\"name\"")?;
    let after_key = &json[key_pos + 6..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    // Collect until unescaped closing quote
    let inner = &after_colon[1..];
    let mut value = String::new();
    let mut chars = inner.chars().peekable();
    loop {
        match chars.next()? {
            '"' => break,
            '\\' => {
                let escaped = chars.next()?;
                match escaped {
                    '"' => value.push('"'),
                    '\\' => value.push('\\'),
                    'n' => value.push('\n'),
                    'r' => value.push('\r'),
                    't' => value.push('\t'),
                    other => {
                        value.push('\\');
                        value.push(other);
                    }
                }
            }
            c => value.push(c),
        }
    }
    Some(value)
}

/// Write a string fragment into an allocated String, converting fmt errors.
fn write_or_err(out: &mut String, s: &str) -> Result<(), SerializeError> {
    out.write_str(s)
        .map_err(|e| SerializeError::FormatError(e.to_string()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aspect_validator::{AspectProperty, Cardinality};

    fn make_model(name: &str) -> AspectModel {
        AspectModel::new(name)
    }

    fn add_prop(model: &mut AspectModel, name: &str, optional: bool, char_ref: &str) {
        model.properties.push(AspectProperty {
            name: name.to_string(),
            cardinality: if optional {
                Cardinality::Optional
            } else {
                Cardinality::Mandatory
            },
            characteristic_ref: char_ref.to_string(),
        });
    }

    // ── JSON tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_json_basic_name() {
        let model = make_model("Movement");
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert_eq!(s.format, SerializationFormat::Json);
        assert!(s.content.contains("\"name\": \"Movement\""));
    }

    #[test]
    fn test_json_default_version() {
        let model = make_model("Sensor");
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\"version\": \"1.0.0\""));
    }

    #[test]
    fn test_json_description_version() {
        let mut model = make_model("Sensor");
        model.description = Some("version:2.1.0".to_string());
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\"version\": \"2.1.0\""));
    }

    #[test]
    fn test_json_empty_properties_array() {
        let model = make_model("Empty");
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\"properties\": []"));
    }

    #[test]
    fn test_json_single_property() {
        let mut model = make_model("Device");
        add_prop(&mut model, "speed", false, "SpeedCharacteristic");
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\"name\": \"speed\""));
        assert!(s.content.contains("\"cardinality\": \"mandatory\""));
        assert!(s
            .content
            .contains("\"characteristic_ref\": \"SpeedCharacteristic\""));
    }

    #[test]
    fn test_json_optional_property() {
        let mut model = make_model("Device");
        add_prop(&mut model, "label", true, "LabelChar");
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\"cardinality\": \"optional\""));
    }

    #[test]
    fn test_json_multiple_properties() {
        let mut model = make_model("Multi");
        add_prop(&mut model, "a", false, "A");
        add_prop(&mut model, "b", true, "B");
        add_prop(&mut model, "c", false, "C");
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\"name\": \"a\""));
        assert!(s.content.contains("\"name\": \"b\""));
        assert!(s.content.contains("\"name\": \"c\""));
    }

    #[test]
    fn test_json_size_bytes() {
        let model = make_model("X");
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert_eq!(s.size_bytes, s.content.len());
    }

    #[test]
    fn test_json_special_chars_in_name() {
        let model = make_model("Has\"Quote");
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        // The name should be escaped
        assert!(s.content.contains("\\\"Quote"));
    }

    #[test]
    fn test_json_description_field() {
        let mut model = make_model("Described");
        model.description = Some("A test aspect".to_string());
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\"description\": \"A test aspect\""));
    }

    // ── YAML tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_yaml_starts_with_document_marker() {
        let model = make_model("Aspect1");
        let s = ModelSerializer::serialize_yaml(&model).expect("yaml ok");
        assert!(s.content.starts_with("---\n"));
    }

    #[test]
    fn test_yaml_name_field() {
        let model = make_model("MyAspect");
        let s = ModelSerializer::serialize_yaml(&model).expect("yaml ok");
        assert!(s.content.contains("name: MyAspect"));
    }

    #[test]
    fn test_yaml_version_field_default() {
        let model = make_model("X");
        let s = ModelSerializer::serialize_yaml(&model).expect("yaml ok");
        assert!(s.content.contains("version: 1.0.0"));
    }

    #[test]
    fn test_yaml_properties_block() {
        let mut model = make_model("Block");
        add_prop(&mut model, "temp", false, "TempChar");
        let s = ModelSerializer::serialize_yaml(&model).expect("yaml ok");
        assert!(s.content.contains("properties:\n"));
        assert!(s.content.contains("  - name: temp\n"));
        assert!(s.content.contains("    cardinality: mandatory\n"));
    }

    #[test]
    fn test_yaml_optional_cardinality() {
        let mut model = make_model("Opt");
        add_prop(&mut model, "label", true, "LabelChar");
        let s = ModelSerializer::serialize_yaml(&model).expect("yaml ok");
        assert!(s.content.contains("    cardinality: optional\n"));
    }

    #[test]
    fn test_yaml_format_field() {
        let model = make_model("X");
        let s = ModelSerializer::serialize_yaml(&model).expect("yaml ok");
        assert_eq!(s.format, SerializationFormat::Yaml);
    }

    #[test]
    fn test_yaml_size_bytes() {
        let model = make_model("X");
        let s = ModelSerializer::serialize_yaml(&model).expect("yaml ok");
        assert_eq!(s.size_bytes, s.content.len());
    }

    #[test]
    fn test_yaml_empty_properties_section() {
        let model = make_model("Empty");
        let s = ModelSerializer::serialize_yaml(&model).expect("yaml ok");
        assert!(s.content.contains("properties:\n"));
        // No list items
        assert!(!s.content.contains("  - name:"));
    }

    // ── Turtle tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_turtle_prefix_declaration() {
        let model = make_model("Aspect");
        let s = ModelSerializer::serialize_turtle(&model).expect("turtle ok");
        assert!(s.content.contains("@prefix samm:"));
    }

    #[test]
    fn test_turtle_aspect_type_assertion() {
        let model = make_model("Movement");
        let s = ModelSerializer::serialize_turtle(&model).expect("turtle ok");
        assert!(s.content.contains("samm:Aspect"));
        assert!(s.content.contains(":Movement"));
    }

    #[test]
    fn test_turtle_property_definition() {
        let mut model = make_model("Sensor");
        add_prop(&mut model, "temperature", false, "TempChar");
        let s = ModelSerializer::serialize_turtle(&model).expect("turtle ok");
        assert!(s.content.contains(":temperature a samm:Property"));
    }

    #[test]
    fn test_turtle_optional_flag_false() {
        let mut model = make_model("Sensor");
        add_prop(&mut model, "speed", false, "SpeedChar");
        let s = ModelSerializer::serialize_turtle(&model).expect("turtle ok");
        assert!(s.content.contains("samm:optional false"));
    }

    #[test]
    fn test_turtle_optional_flag_true() {
        let mut model = make_model("Sensor");
        add_prop(&mut model, "label", true, "LabelChar");
        let s = ModelSerializer::serialize_turtle(&model).expect("turtle ok");
        assert!(s.content.contains("samm:optional true"));
    }

    #[test]
    fn test_turtle_format_field() {
        let model = make_model("X");
        let s = ModelSerializer::serialize_turtle(&model).expect("turtle ok");
        assert_eq!(s.format, SerializationFormat::Turtle);
    }

    #[test]
    fn test_turtle_empty_properties_list() {
        let model = make_model("Empty");
        let s = ModelSerializer::serialize_turtle(&model).expect("turtle ok");
        assert!(s.content.contains("samm:properties ()"));
    }

    #[test]
    fn test_turtle_properties_list_populated() {
        let mut model = make_model("Multi");
        add_prop(&mut model, "a", false, "A");
        add_prop(&mut model, "b", false, "B");
        let s = ModelSerializer::serialize_turtle(&model).expect("turtle ok");
        assert!(s.content.contains(":a") && s.content.contains(":b"));
    }

    // ── Compact tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_compact_name_line() {
        let model = make_model("Device");
        let s = ModelSerializer::serialize_compact(&model).expect("compact ok");
        assert!(s.content.contains("name=Device\n"));
    }

    #[test]
    fn test_compact_version_line_default() {
        let model = make_model("X");
        let s = ModelSerializer::serialize_compact(&model).expect("compact ok");
        assert!(s.content.contains("version=1.0.0\n"));
    }

    #[test]
    fn test_compact_property_count_line() {
        let mut model = make_model("X");
        add_prop(&mut model, "p1", false, "C1");
        add_prop(&mut model, "p2", true, "C2");
        let s = ModelSerializer::serialize_compact(&model).expect("compact ok");
        assert!(s.content.contains("property_count=2\n"));
    }

    #[test]
    fn test_compact_property_line_format() {
        let mut model = make_model("X");
        add_prop(&mut model, "speed", false, "SpeedChar");
        let s = ModelSerializer::serialize_compact(&model).expect("compact ok");
        assert!(s.content.contains("property=speed:mandatory:SpeedChar\n"));
    }

    #[test]
    fn test_compact_optional_property_line() {
        let mut model = make_model("X");
        add_prop(&mut model, "label", true, "LabelChar");
        let s = ModelSerializer::serialize_compact(&model).expect("compact ok");
        assert!(s.content.contains("property=label:optional:LabelChar\n"));
    }

    #[test]
    fn test_compact_format_field() {
        let model = make_model("X");
        let s = ModelSerializer::serialize_compact(&model).expect("compact ok");
        assert_eq!(s.format, SerializationFormat::Compact);
    }

    #[test]
    fn test_compact_no_properties() {
        let model = make_model("Empty");
        let s = ModelSerializer::serialize_compact(&model).expect("compact ok");
        assert!(s.content.contains("property_count=0\n"));
        assert!(!s.content.contains("property="));
    }

    // ── Round-trip tests ──────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_simple() {
        let model = make_model("SimpleAspect");
        let ok = ModelSerializer::round_trip_json(&model).expect("round-trip ok");
        assert!(ok);
    }

    #[test]
    fn test_round_trip_with_properties() {
        let mut model = make_model("WithProps");
        add_prop(&mut model, "x", false, "XChar");
        let ok = ModelSerializer::round_trip_json(&model).expect("round-trip ok");
        assert!(ok);
    }

    #[test]
    fn test_round_trip_name_preserved() {
        let model = make_model("UniqueNameForRoundTrip");
        let serialized = ModelSerializer::serialize_json(&model).expect("json ok");
        let parsed_name = super::parse_json_name(&serialized.content).expect("parse ok");
        assert_eq!(parsed_name, "UniqueNameForRoundTrip");
    }

    // ── Error tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_empty_name_json() {
        let model = make_model("");
        let err = ModelSerializer::serialize_json(&model).expect_err("should fail");
        assert_eq!(err, SerializeError::EmptyName);
    }

    #[test]
    fn test_error_empty_name_yaml() {
        let model = make_model("");
        let err = ModelSerializer::serialize_yaml(&model).expect_err("should fail");
        assert_eq!(err, SerializeError::EmptyName);
    }

    #[test]
    fn test_error_empty_name_turtle() {
        let model = make_model("");
        let err = ModelSerializer::serialize_turtle(&model).expect_err("should fail");
        assert_eq!(err, SerializeError::EmptyName);
    }

    #[test]
    fn test_error_empty_name_compact() {
        let model = make_model("");
        let err = ModelSerializer::serialize_compact(&model).expect_err("should fail");
        assert_eq!(err, SerializeError::EmptyName);
    }

    #[test]
    fn test_error_empty_property_name() {
        let mut model = make_model("Valid");
        model.properties.push(AspectProperty {
            name: "".to_string(),
            cardinality: Cardinality::Mandatory,
            characteristic_ref: "SomeChar".to_string(),
        });
        let err = ModelSerializer::serialize_json(&model).expect_err("should fail");
        assert!(matches!(err, SerializeError::InvalidProperty(_)));
    }

    #[test]
    fn test_serialize_error_display_empty_name() {
        let e = SerializeError::EmptyName;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn test_serialize_error_display_invalid_property() {
        let e = SerializeError::InvalidProperty("bad property".to_string());
        assert!(e.to_string().contains("bad property"));
    }

    #[test]
    fn test_serialize_error_display_format_error() {
        let e = SerializeError::FormatError("something went wrong".to_string());
        assert!(e.to_string().contains("something went wrong"));
    }

    #[test]
    fn test_serialization_format_display() {
        assert_eq!(SerializationFormat::Json.to_string(), "json");
        assert_eq!(SerializationFormat::Yaml.to_string(), "yaml");
        assert_eq!(SerializationFormat::Turtle.to_string(), "turtle");
        assert_eq!(SerializationFormat::Compact.to_string(), "compact");
    }

    #[test]
    fn test_default_constructor() {
        let _s = ModelSerializer;
    }

    #[test]
    fn test_round_trip_empty_name_fails() {
        let model = make_model("");
        let err = ModelSerializer::round_trip_json(&model).expect_err("should fail");
        assert_eq!(err, SerializeError::EmptyName);
    }

    #[test]
    fn test_json_newline_in_description() {
        let mut model = make_model("LineBreak");
        model.description = Some("first\nsecond".to_string());
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\\n"));
    }

    #[test]
    fn test_version_extracted_from_description() {
        let mut model = make_model("VersionExtract");
        model.description = Some("version: 3.0.0".to_string());
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\"version\": \"3.0.0\""));
    }

    #[test]
    fn test_compact_description_line() {
        let mut model = make_model("Described");
        model.description = Some("hello world".to_string());
        let s = ModelSerializer::serialize_compact(&model).expect("compact ok");
        assert!(s.content.contains("description=hello world\n"));
    }

    #[test]
    fn test_turtle_samm_name_literal() {
        let model = make_model("SomeAspect");
        let s = ModelSerializer::serialize_turtle(&model).expect("turtle ok");
        assert!(s.content.contains("samm:name \"SomeAspect\""));
    }

    #[test]
    fn test_json_backslash_escaped() {
        let model = make_model("Back\\slash");
        let s = ModelSerializer::serialize_json(&model).expect("json ok");
        assert!(s.content.contains("\\\\slash"));
    }
}
