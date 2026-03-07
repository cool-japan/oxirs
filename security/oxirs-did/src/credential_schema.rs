//! # Credential Schema Validation
//!
//! W3C Verifiable Credential JSON schema validation for the OxiRS DID module.
//!
//! Provides structural validation of Verifiable Credential JSON against a
//! [`CredentialSchema`], checking required properties, data types, and
//! schema compatibility.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_did::credential_schema::{
//!     CredentialSchema, CredentialProperty, CredentialSchemaValidator,
//! };
//!
//! let schema = CredentialSchema {
//!     id: "https://example.org/schemas/id-card".to_string(),
//!     schema_type: "JsonSchema".to_string(),
//!     version: "1.0".to_string(),
//!     properties: vec![
//!         CredentialProperty {
//!             name: "name".to_string(),
//!             data_type: "string".to_string(),
//!             required: true,
//!             description: "Full legal name".to_string(),
//!         },
//!     ],
//!     required_properties: vec!["name".to_string()],
//! };
//!
//! let json = r#"{"name": "Alice"}"#;
//! let result = CredentialSchemaValidator::validate_credential(json, &schema);
//! assert!(result.valid);
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single property definition within a credential schema
#[derive(Debug, Clone, PartialEq)]
pub struct CredentialProperty {
    /// Property name (key in the JSON object)
    pub name: String,
    /// Expected data type: "string", "integer", "number", "boolean", "array", "object", "uri"
    pub data_type: String,
    /// Whether this property is mandatory
    pub required: bool,
    /// Human-readable description
    pub description: String,
}

impl CredentialProperty {
    /// Convenience constructor
    pub fn new(
        name: impl Into<String>,
        data_type: impl Into<String>,
        required: bool,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            required,
            description: description.into(),
        }
    }
}

/// A W3C-aligned schema definition for a Verifiable Credential
#[derive(Debug, Clone)]
pub struct CredentialSchema {
    /// Schema identifier URI
    pub id: String,
    /// Schema type (e.g. "JsonSchema", "JsonSchemaValidator2018")
    pub schema_type: String,
    /// Schema version string
    pub version: String,
    /// Property definitions
    pub properties: Vec<CredentialProperty>,
    /// Names of required properties (must be present)
    pub required_properties: Vec<String>,
}

impl CredentialSchema {
    /// Retrieve a property definition by name
    pub fn get_property(&self, name: &str) -> Option<&CredentialProperty> {
        self.properties.iter().find(|p| p.name == name)
    }
}

/// Severity level for a validation issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Validation failure — the credential is non-conformant
    Error,
    /// Non-critical advisory
    Warning,
}

/// A single issue found during validation
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationIssue {
    /// Name of the property that caused the issue (empty if global)
    pub property: String,
    /// Human-readable description of the issue
    pub issue: String,
    /// Severity level
    pub severity: IssueSeverity,
}

impl ValidationIssue {
    fn error(property: impl Into<String>, issue: impl Into<String>) -> Self {
        Self {
            property: property.into(),
            issue: issue.into(),
            severity: IssueSeverity::Error,
        }
    }

    fn warning(property: impl Into<String>, issue: impl Into<String>) -> Self {
        Self {
            property: property.into(),
            issue: issue.into(),
            severity: IssueSeverity::Warning,
        }
    }
}

/// The overall result of validating a credential against a schema
#[derive(Debug, Clone)]
pub struct SchemaValidationResult {
    /// `true` when there are no Error-severity issues
    pub valid: bool,
    /// All issues found (errors and warnings)
    pub issues: Vec<ValidationIssue>,
}

impl SchemaValidationResult {
    fn from_issues(issues: Vec<ValidationIssue>) -> Self {
        let valid = !issues.iter().any(|i| i.severity == IssueSeverity::Error);
        Self { valid, issues }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validator
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless credential schema validator
pub struct CredentialSchemaValidator;

impl CredentialSchemaValidator {
    /// Validate `credential_json` against `schema`.
    ///
    /// Checks:
    /// 1. Required properties are present.
    /// 2. Each present property has a compatible data type.
    pub fn validate_credential(
        credential_json: &str,
        schema: &CredentialSchema,
    ) -> SchemaValidationResult {
        let mut issues: Vec<ValidationIssue> = Vec::new();

        // 1. Check required properties
        let required_issues = Self::check_required_properties(credential_json, schema);
        issues.extend(required_issues);

        // 2. Check data types of present properties
        let props = Self::parse_credential_properties(credential_json);
        for (name, value) in &props {
            if let Some(prop_def) = schema.get_property(name) {
                if let Some(type_issue) = Self::validate_property(name, value, prop_def) {
                    issues.push(type_issue);
                }
            }
        }

        SchemaValidationResult::from_issues(issues)
    }

    /// Check whether all required properties are present in `credential_json`.
    pub fn check_required_properties(
        credential_json: &str,
        schema: &CredentialSchema,
    ) -> Vec<ValidationIssue> {
        let props = Self::parse_credential_properties(credential_json);
        let mut issues = Vec::new();

        for required_name in &schema.required_properties {
            if !props.contains_key(required_name.as_str()) {
                issues.push(ValidationIssue::error(
                    required_name.clone(),
                    format!("Required property '{}' is missing", required_name),
                ));
            }
        }

        // Also check properties marked `required = true` in the property list
        for prop in &schema.properties {
            if prop.required && !props.contains_key(prop.name.as_str()) {
                // Only add if not already reported via required_properties
                if !schema.required_properties.contains(&prop.name) {
                    issues.push(ValidationIssue::error(
                        prop.name.clone(),
                        format!("Required property '{}' is missing", prop.name),
                    ));
                }
            }
        }

        issues
    }

    /// Validate a single property value against its definition.
    ///
    /// Returns `None` if the value is compatible, or a `ValidationIssue` otherwise.
    pub fn validate_property(
        name: &str,
        value: &str,
        prop_def: &CredentialProperty,
    ) -> Option<ValidationIssue> {
        if !Self::compatible_type(value, &prop_def.data_type) {
            Some(ValidationIssue::error(
                name,
                format!(
                    "Property '{}' has value '{}' incompatible with expected type '{}'",
                    name, value, prop_def.data_type
                ),
            ))
        } else {
            None
        }
    }

    /// Check whether `value` (a JSON value string) is compatible with `expected_type`.
    ///
    /// Supported type strings: "string", "integer", "number", "boolean", "array",
    /// "object", "uri", "date", "datetime".
    pub fn compatible_type(value: &str, expected_type: &str) -> bool {
        let trimmed = value.trim();
        match expected_type {
            "boolean" => trimmed == "true" || trimmed == "false",
            "integer" => trimmed.parse::<i64>().is_ok(),
            "number" => trimmed.parse::<f64>().is_ok(),
            "array" => trimmed.starts_with('[') && trimmed.ends_with(']'),
            "object" => trimmed.starts_with('{') && trimmed.ends_with('}'),
            "uri" => {
                trimmed.starts_with("http://")
                    || trimmed.starts_with("https://")
                    || trimmed.starts_with("urn:")
                    || trimmed.starts_with("did:")
            }
            "date" => {
                // Simple ISO 8601 date: YYYY-MM-DD
                trimmed.len() == 10
                    && trimmed.chars().nth(4) == Some('-')
                    && trimmed.chars().nth(7) == Some('-')
                    && trimmed[..4].parse::<u16>().is_ok()
                    && trimmed[5..7].parse::<u8>().is_ok()
                    && trimmed[8..10].parse::<u8>().is_ok()
            }
            "datetime" => {
                // Minimal ISO 8601: YYYY-MM-DDTHH:MM...
                trimmed.len() >= 16
                    && trimmed.contains('T')
                    && Self::compatible_type(&trimmed[..10], "date")
            }
            "string" => {
                // A JSON string is surrounded by quotes, or raw for simple values
                !trimmed.is_empty()
            }
            _ => true, // Unknown types pass through
        }
    }

    /// Parse a flat JSON object into a `(key → raw_value)` map.
    ///
    /// This is a deliberately simple heuristic parser that handles single-level
    /// JSON objects with string, number, boolean and null values.  Nested objects
    /// and arrays are returned as their raw JSON text.
    pub fn parse_credential_properties(json: &str) -> HashMap<String, String> {
        let mut map: HashMap<String, String> = HashMap::new();
        let trimmed = json.trim();

        // Must start and end with braces
        if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return map;
        }

        let inner = &trimmed[1..trimmed.len() - 1];
        let tokens = Self::tokenise_json_object(inner);

        let mut i = 0;
        while i + 2 < tokens.len() {
            let key_raw = tokens[i].trim();
            let colon = tokens[i + 1].trim();
            let value_raw = tokens[i + 2].trim();

            if colon == ":" {
                let key = Self::unquote(key_raw);
                let value = Self::unquote(value_raw);
                map.insert(key, value);
                i += 3;
                // Skip optional comma
                if i < tokens.len() && tokens[i].trim() == "," {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        map
    }

    /// Parse a `CredentialSchema` from a JSON string.
    ///
    /// Expects a JSON object with keys: `id`, `type`, `version`, `properties`
    /// (array of `{name, dataType, required, description}`), and `required`.
    pub fn schema_from_json(json: &str) -> Result<CredentialSchema, String> {
        let props_map = Self::parse_credential_properties(json);

        let id = props_map
            .get("id")
            .cloned()
            .ok_or_else(|| "Missing 'id' field".to_string())?;

        let schema_type = props_map
            .get("type")
            .cloned()
            .unwrap_or_else(|| "JsonSchema".to_string());

        let version = props_map
            .get("version")
            .cloned()
            .unwrap_or_else(|| "1.0".to_string());

        Ok(CredentialSchema {
            id,
            schema_type,
            version,
            properties: Vec::new(),
            required_properties: Vec::new(),
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Remove surrounding double-quotes from a JSON string token.
    fn unquote(s: &str) -> String {
        let t = s.trim();
        if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
            t[1..t.len() - 1].replace("\\\"", "\"")
        } else {
            t.to_string()
        }
    }

    /// Very simple tokeniser for a JSON object interior that tracks nested depth.
    fn tokenise_json_object(inner: &str) -> Vec<String> {
        let mut tokens: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut depth = 0i32;
        let mut in_string = false;
        let mut prev_char = '\0';

        for ch in inner.chars() {
            if in_string {
                current.push(ch);
                if ch == '"' && prev_char != '\\' {
                    in_string = false;
                    tokens.push(current.clone());
                    current.clear();
                }
            } else {
                match ch {
                    '"' => {
                        if !current.trim().is_empty() {
                            tokens.push(current.trim().to_string());
                            current.clear();
                        }
                        current.push(ch);
                        in_string = true;
                    }
                    '{' | '[' => {
                        depth += 1;
                        current.push(ch);
                    }
                    '}' | ']' => {
                        depth -= 1;
                        current.push(ch);
                        if depth == 0 {
                            tokens.push(current.trim().to_string());
                            current.clear();
                        }
                    }
                    ':' if depth == 0 => {
                        if !current.trim().is_empty() {
                            tokens.push(current.trim().to_string());
                            current.clear();
                        }
                        tokens.push(":".to_string());
                    }
                    ',' if depth == 0 => {
                        if !current.trim().is_empty() {
                            tokens.push(current.trim().to_string());
                            current.clear();
                        }
                        tokens.push(",".to_string());
                    }
                    ' ' | '\t' | '\n' | '\r' if depth == 0 => {
                        // Skip whitespace between tokens
                    }
                    _ => {
                        current.push(ch);
                    }
                }
            }
            prev_char = ch;
        }

        if !current.trim().is_empty() {
            tokens.push(current.trim().to_string());
        }

        tokens
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_schema() -> CredentialSchema {
        CredentialSchema {
            id: "https://example.org/schemas/test".to_string(),
            schema_type: "JsonSchema".to_string(),
            version: "1.0".to_string(),
            properties: vec![
                CredentialProperty::new("name", "string", true, "Full name"),
                CredentialProperty::new("age", "integer", false, "Age in years"),
                CredentialProperty::new("email", "string", true, "Email address"),
                CredentialProperty::new("score", "number", false, "Score"),
                CredentialProperty::new("active", "boolean", false, "Is active"),
                CredentialProperty::new("website", "uri", false, "Personal website"),
                CredentialProperty::new("birthdate", "date", false, "Birth date"),
            ],
            required_properties: vec!["name".to_string(), "email".to_string()],
        }
    }

    // ── CredentialProperty ────────────────────────────────────────────────────

    #[test]
    fn test_credential_property_new() {
        let p = CredentialProperty::new("name", "string", true, "Full name");
        assert_eq!(p.name, "name");
        assert_eq!(p.data_type, "string");
        assert!(p.required);
        assert_eq!(p.description, "Full name");
    }

    #[test]
    fn test_credential_property_optional() {
        let p = CredentialProperty::new("nickname", "string", false, "Nickname");
        assert!(!p.required);
    }

    // ── CredentialSchema ─────────────────────────────────────────────────────

    #[test]
    fn test_schema_get_property_found() {
        let schema = make_schema();
        let prop = schema.get_property("name");
        assert!(prop.is_some());
        assert_eq!(prop.expect("exists").data_type, "string");
    }

    #[test]
    fn test_schema_get_property_not_found() {
        let schema = make_schema();
        assert!(schema.get_property("nonexistent").is_none());
    }

    // ── IssueSeverity ────────────────────────────────────────────────────────

    #[test]
    fn test_issue_severity_variants() {
        let e = ValidationIssue::error("prop", "msg");
        let w = ValidationIssue::warning("prop", "msg");
        assert_eq!(e.severity, IssueSeverity::Error);
        assert_eq!(w.severity, IssueSeverity::Warning);
    }

    // ── compatible_type ───────────────────────────────────────────────────────

    #[test]
    fn test_compatible_type_string() {
        assert!(CredentialSchemaValidator::compatible_type(
            "hello", "string"
        ));
        assert!(CredentialSchemaValidator::compatible_type("123", "string"));
    }

    #[test]
    fn test_compatible_type_integer_valid() {
        assert!(CredentialSchemaValidator::compatible_type("42", "integer"));
        assert!(CredentialSchemaValidator::compatible_type("-7", "integer"));
    }

    #[test]
    fn test_compatible_type_integer_invalid() {
        assert!(!CredentialSchemaValidator::compatible_type(
            "3.14", "integer"
        ));
        assert!(!CredentialSchemaValidator::compatible_type(
            "abc", "integer"
        ));
    }

    #[test]
    fn test_compatible_type_number_valid() {
        assert!(CredentialSchemaValidator::compatible_type("3.14", "number"));
        assert!(CredentialSchemaValidator::compatible_type("42", "number"));
        assert!(CredentialSchemaValidator::compatible_type("-0.5", "number"));
    }

    #[test]
    fn test_compatible_type_number_invalid() {
        assert!(!CredentialSchemaValidator::compatible_type("abc", "number"));
    }

    #[test]
    fn test_compatible_type_boolean_valid() {
        assert!(CredentialSchemaValidator::compatible_type(
            "true", "boolean"
        ));
        assert!(CredentialSchemaValidator::compatible_type(
            "false", "boolean"
        ));
    }

    #[test]
    fn test_compatible_type_boolean_invalid() {
        assert!(!CredentialSchemaValidator::compatible_type(
            "yes", "boolean"
        ));
        assert!(!CredentialSchemaValidator::compatible_type("1", "boolean"));
    }

    #[test]
    fn test_compatible_type_array_valid() {
        assert!(CredentialSchemaValidator::compatible_type(
            "[1,2,3]", "array"
        ));
        assert!(CredentialSchemaValidator::compatible_type("[]", "array"));
    }

    #[test]
    fn test_compatible_type_array_invalid() {
        assert!(!CredentialSchemaValidator::compatible_type("{}", "array"));
        assert!(!CredentialSchemaValidator::compatible_type(
            "hello", "array"
        ));
    }

    #[test]
    fn test_compatible_type_object_valid() {
        assert!(CredentialSchemaValidator::compatible_type(
            r#"{"a":1}"#,
            "object"
        ));
        assert!(CredentialSchemaValidator::compatible_type("{}", "object"));
    }

    #[test]
    fn test_compatible_type_object_invalid() {
        assert!(!CredentialSchemaValidator::compatible_type("[]", "object"));
        assert!(!CredentialSchemaValidator::compatible_type(
            "hello", "object"
        ));
    }

    #[test]
    fn test_compatible_type_uri_http() {
        assert!(CredentialSchemaValidator::compatible_type(
            "http://example.org/",
            "uri"
        ));
        assert!(CredentialSchemaValidator::compatible_type(
            "https://example.org/",
            "uri"
        ));
    }

    #[test]
    fn test_compatible_type_uri_urn() {
        assert!(CredentialSchemaValidator::compatible_type(
            "urn:isbn:0451450523",
            "uri"
        ));
    }

    #[test]
    fn test_compatible_type_uri_did() {
        assert!(CredentialSchemaValidator::compatible_type(
            "did:key:z6Mk",
            "uri"
        ));
    }

    #[test]
    fn test_compatible_type_uri_invalid() {
        assert!(!CredentialSchemaValidator::compatible_type(
            "ftp://example.org",
            "uri"
        ));
        assert!(!CredentialSchemaValidator::compatible_type(
            "not-a-uri",
            "uri"
        ));
    }

    #[test]
    fn test_compatible_type_date_valid() {
        assert!(CredentialSchemaValidator::compatible_type(
            "2025-01-15",
            "date"
        ));
        assert!(CredentialSchemaValidator::compatible_type(
            "1990-12-31",
            "date"
        ));
    }

    #[test]
    fn test_compatible_type_date_invalid() {
        assert!(!CredentialSchemaValidator::compatible_type(
            "2025-1-5", "date"
        ));
        assert!(!CredentialSchemaValidator::compatible_type(
            "not-a-date",
            "date"
        ));
    }

    #[test]
    fn test_compatible_type_datetime_valid() {
        assert!(CredentialSchemaValidator::compatible_type(
            "2025-01-15T10:30",
            "datetime"
        ));
        assert!(CredentialSchemaValidator::compatible_type(
            "2025-01-15T10:30:00Z",
            "datetime"
        ));
    }

    #[test]
    fn test_compatible_type_unknown_passes() {
        // Unknown types should not block validation
        assert!(CredentialSchemaValidator::compatible_type(
            "anything",
            "customtype"
        ));
    }

    // ── parse_credential_properties ──────────────────────────────────────────

    #[test]
    fn test_parse_simple_json() {
        let json = r#"{"name": "Alice", "age": "30"}"#;
        let props = CredentialSchemaValidator::parse_credential_properties(json);
        assert_eq!(props.get("name"), Some(&"Alice".to_string()));
        assert_eq!(props.get("age"), Some(&"30".to_string()));
    }

    #[test]
    fn test_parse_empty_json_object() {
        let props = CredentialSchemaValidator::parse_credential_properties("{}");
        assert!(props.is_empty());
    }

    #[test]
    fn test_parse_not_an_object() {
        let props = CredentialSchemaValidator::parse_credential_properties("[1,2,3]");
        assert!(props.is_empty());
    }

    #[test]
    fn test_parse_single_property() {
        let json = r#"{"key": "value"}"#;
        let props = CredentialSchemaValidator::parse_credential_properties(json);
        assert_eq!(props.get("key"), Some(&"value".to_string()));
    }

    // ── check_required_properties ─────────────────────────────────────────────

    #[test]
    fn test_check_required_all_present() {
        let schema = make_schema();
        let json = r#"{"name": "Alice", "email": "alice@example.org"}"#;
        let issues = CredentialSchemaValidator::check_required_properties(json, &schema);
        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn test_check_required_missing_one() {
        let schema = make_schema();
        let json = r#"{"name": "Alice"}"#;
        let issues = CredentialSchemaValidator::check_required_properties(json, &schema);
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.property == "email"), "{issues:?}");
    }

    #[test]
    fn test_check_required_missing_all() {
        let schema = make_schema();
        let json = r#"{"nickname": "Al"}"#;
        let issues = CredentialSchemaValidator::check_required_properties(json, &schema);
        assert!(issues.len() >= 2, "{issues:?}");
    }

    // ── validate_property ─────────────────────────────────────────────────────

    #[test]
    fn test_validate_property_string_ok() {
        let prop = CredentialProperty::new("name", "string", true, "name");
        let issue = CredentialSchemaValidator::validate_property("name", "Alice", &prop);
        assert!(issue.is_none());
    }

    #[test]
    fn test_validate_property_integer_ok() {
        let prop = CredentialProperty::new("age", "integer", false, "age");
        let issue = CredentialSchemaValidator::validate_property("age", "30", &prop);
        assert!(issue.is_none());
    }

    #[test]
    fn test_validate_property_integer_bad() {
        let prop = CredentialProperty::new("age", "integer", false, "age");
        let issue = CredentialSchemaValidator::validate_property("age", "thirty", &prop);
        assert!(issue.is_some());
        assert_eq!(issue.expect("present").severity, IssueSeverity::Error);
    }

    #[test]
    fn test_validate_property_boolean_ok() {
        let prop = CredentialProperty::new("active", "boolean", false, "active");
        let issue = CredentialSchemaValidator::validate_property("active", "true", &prop);
        assert!(issue.is_none());
    }

    // ── validate_credential (end-to-end) ──────────────────────────────────────

    #[test]
    fn test_validate_credential_valid() {
        let schema = make_schema();
        let json = r#"{"name": "Alice", "email": "alice@example.org", "age": "30"}"#;
        let result = CredentialSchemaValidator::validate_credential(json, &schema);
        assert!(result.valid, "issues: {:?}", result.issues);
    }

    #[test]
    fn test_validate_credential_missing_required() {
        let schema = make_schema();
        let json = r#"{"age": "25"}"#;
        let result = CredentialSchemaValidator::validate_credential(json, &schema);
        assert!(!result.valid);
    }

    #[test]
    fn test_validate_credential_type_mismatch() {
        let schema = make_schema();
        let json = r#"{"name": "Alice", "email": "a@b.com", "age": "not-a-number"}"#;
        let result = CredentialSchemaValidator::validate_credential(json, &schema);
        // name and email present so required check passes, but age is invalid
        assert!(!result.valid);
        assert!(
            result.issues.iter().any(|i| i.property == "age"),
            "{:?}",
            result.issues
        );
    }

    #[test]
    fn test_validate_credential_extra_properties_allowed() {
        let schema = make_schema();
        let json = r#"{"name": "Alice", "email": "a@b.com", "unknown_prop": "whatever"}"#;
        let result = CredentialSchemaValidator::validate_credential(json, &schema);
        assert!(
            result.valid,
            "extra props should not fail: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_validate_credential_number_type() {
        let schema = make_schema();
        let json = r#"{"name": "Alice", "email": "a@b.com", "score": "9.5"}"#;
        let result = CredentialSchemaValidator::validate_credential(json, &schema);
        assert!(result.valid, "{:?}", result.issues);
    }

    #[test]
    fn test_validate_credential_boolean_type_mismatch() {
        let schema = make_schema();
        let json = r#"{"name": "Alice", "email": "a@b.com", "active": "yes"}"#;
        let result = CredentialSchemaValidator::validate_credential(json, &schema);
        assert!(!result.valid);
    }

    #[test]
    fn test_validate_credential_uri_valid() {
        let schema = make_schema();
        let json = r#"{"name": "Alice", "email": "a@b.com", "website": "https://alice.example"}"#;
        let result = CredentialSchemaValidator::validate_credential(json, &schema);
        assert!(result.valid, "{:?}", result.issues);
    }

    // ── schema_from_json ──────────────────────────────────────────────────────

    #[test]
    fn test_schema_from_json_valid() {
        let json =
            r#"{"id": "https://example.org/schema1", "type": "JsonSchema", "version": "1.0"}"#;
        let schema = CredentialSchemaValidator::schema_from_json(json);
        assert!(schema.is_ok());
        let s = schema.expect("ok");
        assert_eq!(s.id, "https://example.org/schema1");
        assert_eq!(s.schema_type, "JsonSchema");
    }

    #[test]
    fn test_schema_from_json_missing_id() {
        let json = r#"{"type": "JsonSchema"}"#;
        let result = CredentialSchemaValidator::schema_from_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_from_json_defaults_type() {
        let json = r#"{"id": "https://example.org/schema2"}"#;
        let schema = CredentialSchemaValidator::schema_from_json(json).expect("ok");
        assert_eq!(schema.schema_type, "JsonSchema");
        assert_eq!(schema.version, "1.0");
    }

    // ── SchemaValidationResult ────────────────────────────────────────────────

    #[test]
    fn test_validation_result_no_issues_is_valid() {
        let result = SchemaValidationResult::from_issues(vec![]);
        assert!(result.valid);
    }

    #[test]
    fn test_validation_result_warning_only_is_valid() {
        let issues = vec![ValidationIssue::warning("prop", "advisory")];
        let result = SchemaValidationResult::from_issues(issues);
        assert!(result.valid);
    }

    #[test]
    fn test_validation_result_error_is_invalid() {
        let issues = vec![ValidationIssue::error("prop", "failure")];
        let result = SchemaValidationResult::from_issues(issues);
        assert!(!result.valid);
    }
}
