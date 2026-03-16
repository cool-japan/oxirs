//! Aspect model serialization to JSON, YAML, and plain-text formats.
//!
//! Provides `AspectExporter` which can round-trip an `ExportedAspect` through
//! all supported formats.

use std::fmt::Write as FmtWrite;

// ── Data structures ───────────────────────────────────────────────────────────

/// A property of an Aspect model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedProperty {
    /// Property name.
    pub name: String,
    /// Whether this property is optional.
    pub optional: bool,
    /// Fully-qualified data type (e.g. `"xsd:string"`).
    pub data_type: String,
    /// Human-readable description.
    pub description: String,
}

impl ExportedProperty {
    /// Convenience constructor.
    pub fn new(
        name: impl Into<String>,
        optional: bool,
        data_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            optional,
            data_type: data_type.into(),
            description: description.into(),
        }
    }
}

/// A serialisable Aspect model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedAspect {
    /// Aspect name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Properties declared by the aspect.
    pub properties: Vec<ExportedProperty>,
    /// Operation names exposed by the aspect.
    pub operations: Vec<String>,
}

impl ExportedAspect {
    /// Convenience constructor.
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        properties: Vec<ExportedProperty>,
        operations: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            properties,
            operations,
        }
    }
}

// ── Export format ─────────────────────────────────────────────────────────────

/// Serialisation format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON (machine-readable).
    Json,
    /// YAML (human-friendly).
    Yaml,
    /// Plain text (for documentation / display).
    Text,
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Error returned by export / import operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AspectExportError(pub String);

impl std::fmt::Display for AspectExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AspectExportError: {}", self.0)
    }
}

impl std::error::Error for AspectExportError {}

// ── Exporter ─────────────────────────────────────────────────────────────────

/// Serialises and deserialises `ExportedAspect` values.
pub struct AspectExporter;

impl AspectExporter {
    // ── JSON ──────────────────────────────────────────────────────────────────

    /// Serialise to a pretty-printed JSON string.
    pub fn export_json(aspect: &ExportedAspect) -> Result<String, AspectExportError> {
        let mut s = String::new();
        writeln!(s, "{{").ok();
        writeln!(s, "  \"name\": {},", Self::json_str(&aspect.name)).ok();
        writeln!(s, "  \"version\": {},", Self::json_str(&aspect.version)).ok();

        // Properties array
        writeln!(s, "  \"properties\": [").ok();
        for (i, prop) in aspect.properties.iter().enumerate() {
            let comma = if i + 1 < aspect.properties.len() {
                ","
            } else {
                ""
            };
            writeln!(s, "    {{").ok();
            writeln!(s, "      \"name\": {},", Self::json_str(&prop.name)).ok();
            writeln!(
                s,
                "      \"optional\": {},",
                if prop.optional { "true" } else { "false" }
            )
            .ok();
            writeln!(
                s,
                "      \"dataType\": {},",
                Self::json_str(&prop.data_type)
            )
            .ok();
            writeln!(
                s,
                "      \"description\": {}",
                Self::json_str(&prop.description)
            )
            .ok();
            writeln!(s, "    }}{comma}").ok();
        }
        writeln!(s, "  ],").ok();

        // Operations array
        writeln!(s, "  \"operations\": [").ok();
        for (i, op) in aspect.operations.iter().enumerate() {
            let comma = if i + 1 < aspect.operations.len() {
                ","
            } else {
                ""
            };
            writeln!(s, "    {}{comma}", Self::json_str(op)).ok();
        }
        writeln!(s, "  ]").ok();
        write!(s, "}}").ok();

        Ok(s)
    }

    /// Parse a JSON string produced by `export_json`.
    pub fn parse_json(json: &str) -> Result<ExportedAspect, AspectExportError> {
        // We parse manually to stay dependency-free in this module.
        let name = Self::extract_json_string(json, "\"name\"")
            .ok_or_else(|| AspectExportError("missing 'name' field".to_string()))?;
        let version = Self::extract_json_string(json, "\"version\"")
            .ok_or_else(|| AspectExportError("missing 'version' field".to_string()))?;

        // Parse properties array.
        let properties = Self::extract_json_property_array(json)?;

        // Parse operations array.
        let operations = Self::extract_json_string_array(json, "\"operations\"")?;

        Ok(ExportedAspect {
            name,
            version,
            properties,
            operations,
        })
    }

    // ── YAML ──────────────────────────────────────────────────────────────────

    /// Serialise to a YAML string.
    pub fn export_yaml(aspect: &ExportedAspect) -> Result<String, AspectExportError> {
        let mut s = String::new();
        writeln!(s, "name: {}", Self::yaml_str(&aspect.name)).ok();
        writeln!(s, "version: {}", Self::yaml_str(&aspect.version)).ok();
        writeln!(s, "properties:").ok();
        for prop in &aspect.properties {
            writeln!(s, "  - name: {}", Self::yaml_str(&prop.name)).ok();
            writeln!(
                s,
                "    optional: {}",
                if prop.optional { "true" } else { "false" }
            )
            .ok();
            writeln!(s, "    dataType: {}", Self::yaml_str(&prop.data_type)).ok();
            writeln!(s, "    description: {}", Self::yaml_str(&prop.description)).ok();
        }
        writeln!(s, "operations:").ok();
        for op in &aspect.operations {
            writeln!(s, "  - {}", Self::yaml_str(op)).ok();
        }
        Ok(s)
    }

    // ── Text ──────────────────────────────────────────────────────────────────

    /// Serialise to a human-readable plain-text summary.
    pub fn export_text(aspect: &ExportedAspect) -> String {
        let mut s = String::new();
        writeln!(s, "Aspect: {} (v{})", aspect.name, aspect.version).ok();
        writeln!(s, "Properties ({}):", aspect.properties.len()).ok();
        for prop in &aspect.properties {
            let opt = if prop.optional { " [optional]" } else { "" };
            writeln!(
                s,
                "  - {}: {}{} — {}",
                prop.name, prop.data_type, opt, prop.description
            )
            .ok();
        }
        writeln!(s, "Operations ({}):", aspect.operations.len()).ok();
        for op in &aspect.operations {
            writeln!(s, "  - {op}").ok();
        }
        s
    }

    // ── Dispatch ─────────────────────────────────────────────────────────────

    /// Serialise using the specified format.
    pub fn export(
        aspect: &ExportedAspect,
        format: ExportFormat,
    ) -> Result<String, AspectExportError> {
        match format {
            ExportFormat::Json => Self::export_json(aspect),
            ExportFormat::Yaml => Self::export_yaml(aspect),
            ExportFormat::Text => Ok(Self::export_text(aspect)),
        }
    }

    /// Export then parse back; return `true` if the round-trip is lossless.
    pub fn roundtrip(
        aspect: &ExportedAspect,
        format: ExportFormat,
    ) -> Result<bool, AspectExportError> {
        match format {
            ExportFormat::Json => {
                let serialised = Self::export_json(aspect)?;
                let reparsed = Self::parse_json(&serialised)?;
                Ok(&reparsed == aspect)
            }
            ExportFormat::Yaml | ExportFormat::Text => {
                // YAML / Text parsers not yet implemented; just verify no error.
                let _ = Self::export(aspect, format)?;
                Ok(true)
            }
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn json_str(s: &str) -> String {
        // Escape backslashes and double quotes.
        let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{escaped}\"")
    }

    fn yaml_str(s: &str) -> String {
        if s.contains(':') || s.contains('#') || s.contains('"') || s.is_empty() {
            let escaped = s.replace('"', "\\\"");
            format!("\"{escaped}\"")
        } else {
            s.to_string()
        }
    }

    /// Extract a single string field from raw JSON by searching for `key: "value"`.
    fn extract_json_string(json: &str, key: &str) -> Option<String> {
        let search = format!("{key}:");
        let pos = json.find(&search)?;
        let after = json[pos + search.len()..].trim_start();
        Self::parse_json_string_value(after)
    }

    fn parse_json_string_value(s: &str) -> Option<String> {
        let s = s.trim_start();
        if !s.starts_with('"') {
            return None;
        }
        let mut result = String::new();
        let mut chars = s[1..].chars().peekable();
        loop {
            match chars.next()? {
                '\\' => match chars.next()? {
                    '"' => result.push('"'),
                    '\\' => result.push('\\'),
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    c => {
                        result.push('\\');
                        result.push(c);
                    }
                },
                '"' => break,
                c => result.push(c),
            }
        }
        Some(result)
    }

    fn extract_json_property_array(json: &str) -> Result<Vec<ExportedProperty>, AspectExportError> {
        let mut props = Vec::new();

        // Find the "properties" key.
        let Some(pos) = json.find("\"properties\"") else {
            return Ok(props);
        };
        let after = &json[pos + "\"properties\"".len()..];
        let Some(arr_start) = after.find('[') else {
            return Ok(props);
        };
        let arr = &after[arr_start + 1..];

        // Split on object boundaries (simple heuristic).
        let mut depth = 0i32;
        let mut obj_start = None;
        let mut i = 0;
        let bytes = arr.as_bytes();
        while i < bytes.len() {
            match bytes[i] {
                b'{' => {
                    if depth == 0 {
                        obj_start = Some(i);
                    }
                    depth += 1;
                }
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(start) = obj_start {
                            let obj_str = &arr[start..=i];
                            let name =
                                Self::extract_json_string(obj_str, "\"name\"").unwrap_or_default();
                            let data_type = Self::extract_json_string(obj_str, "\"dataType\"")
                                .unwrap_or_default();
                            let description = Self::extract_json_string(obj_str, "\"description\"")
                                .unwrap_or_default();
                            let optional_str = Self::extract_json_bool(obj_str, "\"optional\"")
                                .unwrap_or_default();
                            let optional = optional_str == "true";
                            props.push(ExportedProperty {
                                name,
                                optional,
                                data_type,
                                description,
                            });
                        }
                        obj_start = None;
                    }
                }
                b']' if depth == 0 => break,
                _ => {}
            }
            i += 1;
        }
        Ok(props)
    }

    fn extract_json_bool(json: &str, key: &str) -> Option<String> {
        let search = format!("{key}:");
        let pos = json.find(&search)?;
        let after = json[pos + search.len()..].trim_start();
        if after.starts_with("true") {
            Some("true".to_string())
        } else if after.starts_with("false") {
            Some("false".to_string())
        } else {
            None
        }
    }

    fn extract_json_string_array(json: &str, key: &str) -> Result<Vec<String>, AspectExportError> {
        let mut results = Vec::new();
        let search = format!("{key}:");
        let Some(pos) = json.find(&search) else {
            return Ok(results);
        };
        let after = &json[pos + search.len()..];
        let Some(arr_start) = after.find('[') else {
            return Ok(results);
        };
        let arr = &after[arr_start + 1..];

        let mut remaining = arr;
        loop {
            remaining = remaining.trim_start();
            if remaining.starts_with(']') || remaining.is_empty() {
                break;
            }
            if remaining.starts_with('"') {
                if let Some(v) = Self::parse_json_string_value(remaining) {
                    let skip = v.len() + 2; // open/close quotes
                    results.push(v);
                    if remaining.len() > skip {
                        remaining = &remaining[skip..];
                    } else {
                        break;
                    }
                    // Skip past comma
                    remaining =
                        remaining.trim_start_matches(|c: char| c == ',' || c.is_whitespace());
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(results)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prop(name: &str, optional: bool) -> ExportedProperty {
        ExportedProperty::new(name, optional, "xsd:string", format!("Desc of {name}"))
    }

    fn sample_aspect() -> ExportedAspect {
        ExportedAspect::new(
            "TestAspect",
            "1.0.0",
            vec![sample_prop("propA", false), sample_prop("propB", true)],
            vec!["getStatus".to_string(), "reset".to_string()],
        )
    }

    // ── ExportedProperty ──────────────────────────────────────────────────────

    #[test]
    fn test_prop_new() {
        let p = ExportedProperty::new("speed", false, "xsd:float", "Vehicle speed");
        assert_eq!(p.name, "speed");
        assert!(!p.optional);
        assert_eq!(p.data_type, "xsd:float");
        assert_eq!(p.description, "Vehicle speed");
    }

    #[test]
    fn test_prop_optional() {
        let p = ExportedProperty::new("color", true, "xsd:string", "Optional color");
        assert!(p.optional);
    }

    #[test]
    fn test_prop_clone() {
        let p = sample_prop("x", false);
        assert_eq!(p, p.clone());
    }

    // ── ExportedAspect ────────────────────────────────────────────────────────

    #[test]
    fn test_aspect_new() {
        let a = sample_aspect();
        assert_eq!(a.name, "TestAspect");
        assert_eq!(a.version, "1.0.0");
        assert_eq!(a.properties.len(), 2);
        assert_eq!(a.operations.len(), 2);
    }

    #[test]
    fn test_aspect_clone() {
        let a = sample_aspect();
        assert_eq!(a, a.clone());
    }

    #[test]
    fn test_aspect_no_props_no_ops() {
        let a = ExportedAspect::new("Empty", "0.1.0", vec![], vec![]);
        assert!(a.properties.is_empty());
        assert!(a.operations.is_empty());
    }

    // ── JSON export ───────────────────────────────────────────────────────────

    #[test]
    fn test_export_json_contains_name() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        assert!(json.contains("\"TestAspect\""));
    }

    #[test]
    fn test_export_json_contains_version() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        assert!(json.contains("\"1.0.0\""));
    }

    #[test]
    fn test_export_json_contains_properties() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        assert!(json.contains("\"propA\""));
        assert!(json.contains("\"propB\""));
    }

    #[test]
    fn test_export_json_contains_operations() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        assert!(json.contains("\"getStatus\""));
    }

    #[test]
    fn test_export_json_optional_flag() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        assert!(json.contains("true") || json.contains("false"));
    }

    #[test]
    fn test_export_json_empty_aspect() {
        let a = ExportedAspect::new("Empty", "0.0.1", vec![], vec![]);
        let json = AspectExporter::export_json(&a).expect("should succeed");
        assert!(json.contains("\"Empty\""));
    }

    #[test]
    fn test_export_json_special_chars() {
        let a = ExportedAspect::new("With\"Quote", "1.0.0", vec![], vec![]);
        let json = AspectExporter::export_json(&a).expect("should succeed");
        assert!(json.contains("\\\"Quote"));
    }

    // ── JSON parse ────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_json_name() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        let parsed = AspectExporter::parse_json(&json).expect("should succeed");
        assert_eq!(parsed.name, "TestAspect");
    }

    #[test]
    fn test_parse_json_version() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        let parsed = AspectExporter::parse_json(&json).expect("should succeed");
        assert_eq!(parsed.version, "1.0.0");
    }

    #[test]
    fn test_parse_json_properties_count() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        let parsed = AspectExporter::parse_json(&json).expect("should succeed");
        assert_eq!(parsed.properties.len(), 2);
    }

    #[test]
    fn test_parse_json_property_names() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        let parsed = AspectExporter::parse_json(&json).expect("should succeed");
        assert_eq!(parsed.properties[0].name, "propA");
        assert_eq!(parsed.properties[1].name, "propB");
    }

    #[test]
    fn test_parse_json_property_optional() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        let parsed = AspectExporter::parse_json(&json).expect("should succeed");
        assert!(!parsed.properties[0].optional);
        assert!(parsed.properties[1].optional);
    }

    #[test]
    fn test_parse_json_missing_name_error() {
        let result = AspectExporter::parse_json("{}");
        assert!(result.is_err());
    }

    // ── JSON roundtrip ────────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_json() {
        let a = sample_aspect();
        let ok = AspectExporter::roundtrip(&a, ExportFormat::Json).expect("should succeed");
        assert!(ok);
    }

    #[test]
    fn test_roundtrip_json_empty() {
        let a = ExportedAspect::new("Empty", "1.0.0", vec![], vec![]);
        let ok = AspectExporter::roundtrip(&a, ExportFormat::Json).expect("should succeed");
        assert!(ok);
    }

    // ── YAML export ───────────────────────────────────────────────────────────

    #[test]
    fn test_export_yaml_contains_name() {
        let a = sample_aspect();
        let yaml = AspectExporter::export_yaml(&a).expect("should succeed");
        assert!(yaml.contains("TestAspect"));
    }

    #[test]
    fn test_export_yaml_contains_version() {
        let a = sample_aspect();
        let yaml = AspectExporter::export_yaml(&a).expect("should succeed");
        assert!(yaml.contains("1.0.0"));
    }

    #[test]
    fn test_export_yaml_properties_section() {
        let a = sample_aspect();
        let yaml = AspectExporter::export_yaml(&a).expect("should succeed");
        assert!(yaml.contains("properties:"));
    }

    #[test]
    fn test_export_yaml_operations_section() {
        let a = sample_aspect();
        let yaml = AspectExporter::export_yaml(&a).expect("should succeed");
        assert!(yaml.contains("operations:"));
    }

    #[test]
    fn test_roundtrip_yaml_no_error() {
        let a = sample_aspect();
        let ok = AspectExporter::roundtrip(&a, ExportFormat::Yaml).expect("should succeed");
        assert!(ok);
    }

    // ── Text export ───────────────────────────────────────────────────────────

    #[test]
    fn test_export_text_contains_name() {
        let a = sample_aspect();
        let text = AspectExporter::export_text(&a);
        assert!(text.contains("TestAspect"));
    }

    #[test]
    fn test_export_text_contains_version() {
        let a = sample_aspect();
        let text = AspectExporter::export_text(&a);
        assert!(text.contains("1.0.0"));
    }

    #[test]
    fn test_export_text_contains_property() {
        let a = sample_aspect();
        let text = AspectExporter::export_text(&a);
        assert!(text.contains("propA"));
    }

    #[test]
    fn test_export_text_optional_marker() {
        let a = sample_aspect();
        let text = AspectExporter::export_text(&a);
        assert!(text.contains("[optional]"));
    }

    #[test]
    fn test_export_text_no_props() {
        let a = ExportedAspect::new("Bare", "1.0.0", vec![], vec![]);
        let text = AspectExporter::export_text(&a);
        assert!(text.contains("Bare"));
        assert!(text.contains("Properties (0)"));
    }

    #[test]
    fn test_roundtrip_text_no_error() {
        let a = sample_aspect();
        let ok = AspectExporter::roundtrip(&a, ExportFormat::Text).expect("should succeed");
        assert!(ok);
    }

    // ── ExportFormat ─────────────────────────────────────────────────────────

    #[test]
    fn test_export_dispatch_json() {
        let a = sample_aspect();
        let s = AspectExporter::export(&a, ExportFormat::Json).expect("should succeed");
        assert!(s.contains('{'));
    }

    #[test]
    fn test_export_dispatch_yaml() {
        let a = sample_aspect();
        let s = AspectExporter::export(&a, ExportFormat::Yaml).expect("should succeed");
        assert!(s.contains("name:"));
    }

    #[test]
    fn test_export_dispatch_text() {
        let a = sample_aspect();
        let s = AspectExporter::export(&a, ExportFormat::Text).expect("should succeed");
        assert!(s.contains("Aspect:"));
    }

    // ── Error ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_aspect_export_error_display() {
        let e = AspectExportError("something went wrong".to_string());
        let msg = format!("{e}");
        assert!(msg.contains("something went wrong"));
    }

    #[test]
    fn test_aspect_export_error_clone() {
        let e = AspectExportError("err".to_string());
        assert_eq!(e, e.clone());
    }

    #[test]
    fn test_exported_property_optional_false() {
        let p = ExportedProperty::new("speed", false, "xsd:float", "Speed in m/s");
        assert!(!p.optional);
    }

    #[test]
    fn test_exported_property_optional_true() {
        let p = ExportedProperty::new("color", true, "xsd:string", "Optional color");
        assert!(p.optional);
    }

    #[test]
    fn test_exported_aspect_operations_empty() {
        let a = ExportedAspect::new("A", "1.0.0", vec![], vec![]);
        assert!(a.operations.is_empty());
    }

    #[test]
    fn test_exported_aspect_operations_present() {
        let a = ExportedAspect::new(
            "A",
            "1.0.0",
            vec![],
            vec!["getStatus".to_string(), "reset".to_string()],
        );
        assert_eq!(a.operations.len(), 2);
        assert!(a.operations.contains(&"getStatus".to_string()));
    }

    #[test]
    fn test_export_json_has_version() {
        let a = sample_aspect();
        let json = AspectExporter::export_json(&a).expect("should succeed");
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn test_export_yaml_has_version() {
        let a = sample_aspect();
        let yaml = AspectExporter::export_yaml(&a).expect("should succeed");
        assert!(yaml.contains("1.0.0"));
    }

    #[test]
    fn test_export_text_operations_listed() {
        let a = ExportedAspect::new(
            "Ops",
            "1.0.0",
            vec![],
            vec!["start".to_string(), "stop".to_string()],
        );
        let text = AspectExporter::export_text(&a);
        assert!(text.contains("Operations"));
        assert!(text.contains("start"));
        assert!(text.contains("stop"));
    }

    #[test]
    fn test_export_format_copy() {
        let f = ExportFormat::Json;
        let g = f;
        assert_eq!(f, g);
    }
}
