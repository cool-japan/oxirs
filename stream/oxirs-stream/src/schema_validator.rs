//! Stream message schema validation.
//!
//! Provides schema-driven validation of stream messages, including field type
//! checking, required field enforcement, format validation (email, URI, date-time,
//! UUID), strict unknown-field detection, and schema composition stubs (allOf/anyOf).
//!
//! Schemas are versioned and stored in a [`SchemaRegistry`].

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Field type
// ─────────────────────────────────────────────────────────────────────────────

/// The expected type of a schema field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
}

impl FieldType {
    /// Human-readable name used in error messages.
    pub fn name(&self) -> &'static str {
        match self {
            FieldType::String => "string",
            FieldType::Integer => "integer",
            FieldType::Float => "float",
            FieldType::Boolean => "boolean",
            FieldType::Array => "array",
            FieldType::Object => "object",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Format
// ─────────────────────────────────────────────────────────────────────────────

/// Optional string format constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldFormat {
    /// `user@example.com`
    Email,
    /// `https://example.com/path`
    Uri,
    /// ISO 8601 / RFC 3339, e.g. `2024-01-15T10:00:00Z`
    DateTime,
    /// `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
    Uuid,
}

impl FieldFormat {
    /// Validate a string value against this format pattern.
    pub fn validate(&self, value: &str) -> bool {
        match self {
            FieldFormat::Email => validate_email(value),
            FieldFormat::Uri => validate_uri(value),
            FieldFormat::DateTime => validate_datetime(value),
            FieldFormat::Uuid => validate_uuid(value),
        }
    }

    /// Short name used in error messages.
    pub fn name(&self) -> &'static str {
        match self {
            FieldFormat::Email => "email",
            FieldFormat::Uri => "uri",
            FieldFormat::DateTime => "date-time",
            FieldFormat::Uuid => "uuid",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Field definition
// ─────────────────────────────────────────────────────────────────────────────

/// Definition of a single field within a schema.
#[derive(Debug, Clone)]
pub struct FieldDefinition {
    /// Field name / key.
    pub name: String,
    /// Expected JSON-style type.
    pub field_type: FieldType,
    /// Whether the field must be present.
    pub required: bool,
    /// Optional string format constraint (applied when `field_type == String`).
    pub format: Option<FieldFormat>,
}

impl FieldDefinition {
    /// Create a new field definition.
    pub fn new(
        name: impl Into<String>,
        field_type: FieldType,
        required: bool,
        format: Option<FieldFormat>,
    ) -> Self {
        Self {
            name: name.into(),
            field_type,
            required,
            format,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema composition stubs
// ─────────────────────────────────────────────────────────────────────────────

/// Composition operator applied to sub-schemas.
#[derive(Debug, Clone)]
pub enum SchemaComposition {
    /// Message must satisfy ALL listed sub-schemas.
    AllOf(Vec<String>),
    /// Message must satisfy AT LEAST ONE listed sub-schema.
    AnyOf(Vec<String>),
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema
// ─────────────────────────────────────────────────────────────────────────────

/// A versioned schema describing the expected structure of a stream message.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Schema name.
    pub name: String,
    /// Schema version (e.g. `"1.0"`).
    pub version: String,
    /// Field definitions.
    pub fields: Vec<FieldDefinition>,
    /// When `true`, fields not declared in `fields` cause validation errors.
    pub strict_mode: bool,
    /// Optional schema composition (allOf / anyOf stub).
    pub composition: Option<SchemaComposition>,
}

impl Schema {
    /// Build a new schema.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            fields: Vec::new(),
            strict_mode: false,
            composition: None,
        }
    }

    /// Add a field definition.
    pub fn with_field(mut self, field: FieldDefinition) -> Self {
        self.fields.push(field);
        self
    }

    /// Enable strict mode (unknown fields are rejected).
    pub fn strict(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Attach a composition rule.
    pub fn with_composition(mut self, composition: SchemaComposition) -> Self {
        self.composition = Some(composition);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Typed message value (simplified JSON-like)
// ─────────────────────────────────────────────────────────────────────────────

/// A typed value extracted from a stream message.
#[derive(Debug, Clone, PartialEq)]
pub enum MessageValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<MessageValue>),
    Object(HashMap<String, MessageValue>),
}

impl MessageValue {
    /// Return the `FieldType` that matches this value variant.
    pub fn field_type(&self) -> FieldType {
        match self {
            MessageValue::String(_) => FieldType::String,
            MessageValue::Integer(_) => FieldType::Integer,
            MessageValue::Float(_) => FieldType::Float,
            MessageValue::Boolean(_) => FieldType::Boolean,
            MessageValue::Array(_) => FieldType::Array,
            MessageValue::Object(_) => FieldType::Object,
        }
    }

    /// Return the string value if this is a `String` variant.
    pub fn as_str(&self) -> Option<&str> {
        if let MessageValue::String(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }
}

/// A stream message represented as a flat map of field names to values.
pub type StreamMessage = HashMap<String, MessageValue>;

// ─────────────────────────────────────────────────────────────────────────────
// Validation result
// ─────────────────────────────────────────────────────────────────────────────

/// A single field-level validation error.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldError {
    /// Field that failed.
    pub field: String,
    /// Human-readable description.
    pub message: String,
}

impl FieldError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

/// Result of validating a single message against a schema.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// `true` iff the message passed all checks.
    pub is_valid: bool,
    /// Per-field errors (empty on success).
    pub errors: Vec<FieldError>,
}

impl ValidationResult {
    fn ok() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
        }
    }

    fn with_errors(errors: Vec<FieldError>) -> Self {
        Self {
            is_valid: errors.is_empty(),
            errors,
        }
    }

    /// Number of field errors.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validator
// ─────────────────────────────────────────────────────────────────────────────

/// Validates stream messages against a [`Schema`].
pub struct SchemaValidator;

impl SchemaValidator {
    /// Validate `message` against `schema`.
    ///
    /// Checks, in order:
    /// 1. Required field presence
    /// 2. Field type correctness
    /// 3. String format constraints
    /// 4. Unknown field detection (strict mode)
    pub fn validate(schema: &Schema, message: &StreamMessage) -> ValidationResult {
        let mut errors: Vec<FieldError> = Vec::new();

        // Build a set of known field names for fast lookup.
        let known: std::collections::HashSet<&str> =
            schema.fields.iter().map(|f| f.name.as_str()).collect();

        // Check required fields and type / format correctness.
        for field_def in &schema.fields {
            match message.get(&field_def.name) {
                None => {
                    if field_def.required {
                        errors.push(FieldError::new(
                            &field_def.name,
                            format!("required field '{}' is missing", field_def.name),
                        ));
                    }
                }
                Some(value) => {
                    // Type check
                    if value.field_type() != field_def.field_type {
                        errors.push(FieldError::new(
                            &field_def.name,
                            format!(
                                "expected type '{}', found '{}'",
                                field_def.field_type.name(),
                                value.field_type().name()
                            ),
                        ));
                    } else if let Some(fmt) = &field_def.format {
                        // Format check (only applicable to strings)
                        if let Some(s) = value.as_str() {
                            if !fmt.validate(s) {
                                errors.push(FieldError::new(
                                    &field_def.name,
                                    format!("value '{}' does not match format '{}'", s, fmt.name()),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Strict mode: unknown fields
        if schema.strict_mode {
            for key in message.keys() {
                if !known.contains(key.as_str()) {
                    errors.push(FieldError::new(
                        key,
                        format!("unknown field '{}' not allowed in strict mode", key),
                    ));
                }
            }
        }

        if errors.is_empty() {
            ValidationResult::ok()
        } else {
            ValidationResult::with_errors(errors)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema registry
// ─────────────────────────────────────────────────────────────────────────────

/// Registry key: (schema name, version).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RegistryKey {
    name: String,
    version: String,
}

/// Registry for named, versioned schemas.
pub struct SchemaRegistry {
    schemas: HashMap<RegistryKey, Schema>,
}

impl SchemaRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }

    /// Register a schema. Overwrites any existing entry with the same name+version.
    pub fn register(&mut self, schema: Schema) {
        let key = RegistryKey {
            name: schema.name.clone(),
            version: schema.version.clone(),
        };
        self.schemas.insert(key, schema);
    }

    /// Look up a schema by name and version.
    pub fn lookup(&self, name: &str, version: &str) -> Option<&Schema> {
        let key = RegistryKey {
            name: name.to_string(),
            version: version.to_string(),
        };
        self.schemas.get(&key)
    }

    /// Validate a message using a registered schema.
    ///
    /// Returns `None` if the schema is not found.
    pub fn validate(
        &self,
        schema_name: &str,
        schema_version: &str,
        message: &StreamMessage,
    ) -> Option<ValidationResult> {
        let schema = self.lookup(schema_name, schema_version)?;
        Some(SchemaValidator::validate(schema, message))
    }

    /// Number of schemas registered.
    pub fn schema_count(&self) -> usize {
        self.schemas.len()
    }

    /// List all registered (name, version) pairs.
    pub fn list(&self) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = self
            .schemas
            .keys()
            .map(|k| (k.name.clone(), k.version.clone()))
            .collect();
        pairs.sort();
        pairs
    }
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Format validation helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_email(value: &str) -> bool {
    // Simple pattern: <local>@<domain>.<tld>
    let parts: Vec<&str> = value.splitn(2, '@').collect();
    if parts.len() != 2 {
        return false;
    }
    let local = parts[0];
    let domain = parts[1];
    !local.is_empty() && domain.contains('.') && !domain.starts_with('.') && !domain.ends_with('.')
}

fn validate_uri(value: &str) -> bool {
    // Require a known scheme followed by "://"
    let schemes = ["http://", "https://", "ftp://", "urn:", "mailto:"];
    schemes.iter().any(|s| value.starts_with(s)) && value.len() > 8
}

fn validate_datetime(value: &str) -> bool {
    // Accept ISO 8601 / RFC 3339: YYYY-MM-DDTHH:MM:SS with optional timezone.
    // Minimal pattern: 19 chars for "YYYY-MM-DDTHH:MM:SS"
    if value.len() < 19 {
        return false;
    }
    let bytes = value.as_bytes();
    let date_sep1 = bytes.get(4).copied() == Some(b'-');
    let date_sep2 = bytes.get(7).copied() == Some(b'-');
    let time_sep = bytes.get(10).copied() == Some(b'T') || bytes.get(10).copied() == Some(b' ');
    let colon1 = bytes.get(13).copied() == Some(b':');
    let colon2 = bytes.get(16).copied() == Some(b':');
    date_sep1 && date_sep2 && time_sep && colon1 && colon2
}

fn validate_uuid(value: &str) -> bool {
    // xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  (36 chars with hyphens)
    if value.len() != 36 {
        return false;
    }
    let bytes = value.as_bytes();
    bytes[8] == b'-'
        && bytes[13] == b'-'
        && bytes[18] == b'-'
        && bytes[23] == b'-'
        && value
            .chars()
            .enumerate()
            .all(|(i, c)| matches!(i, 8 | 13 | 18 | 23) || c.is_ascii_hexdigit())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn string_field(name: &str, required: bool) -> FieldDefinition {
        FieldDefinition::new(name, FieldType::String, required, None)
    }

    fn int_field(name: &str, required: bool) -> FieldDefinition {
        FieldDefinition::new(name, FieldType::Integer, required, None)
    }

    fn email_field(name: &str) -> FieldDefinition {
        FieldDefinition::new(name, FieldType::String, true, Some(FieldFormat::Email))
    }

    fn uri_field(name: &str) -> FieldDefinition {
        FieldDefinition::new(name, FieldType::String, true, Some(FieldFormat::Uri))
    }

    fn datetime_field(name: &str) -> FieldDefinition {
        FieldDefinition::new(name, FieldType::String, true, Some(FieldFormat::DateTime))
    }

    fn uuid_field(name: &str) -> FieldDefinition {
        FieldDefinition::new(name, FieldType::String, true, Some(FieldFormat::Uuid))
    }

    fn basic_schema() -> Schema {
        Schema::new("event", "1.0")
            .with_field(string_field("id", true))
            .with_field(int_field("count", false))
    }

    fn msg(pairs: &[(&str, MessageValue)]) -> StreamMessage {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    // ── required field presence ──────────────────────────────────────────────

    #[test]
    fn test_valid_message_passes() {
        let schema = basic_schema();
        let message = msg(&[("id", MessageValue::String("abc".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_missing_required_field_fails() {
        let schema = basic_schema();
        let message = msg(&[]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].field, "id");
    }

    #[test]
    fn test_optional_field_may_be_absent() {
        let schema = basic_schema();
        let message = msg(&[("id", MessageValue::String("x".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    // ── type checking ────────────────────────────────────────────────────────

    #[test]
    fn test_type_mismatch_integer_vs_string() {
        let schema = Schema::new("s", "1").with_field(int_field("n", true));
        let message = msg(&[("n", MessageValue::String("not-a-number".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
        assert!(result.errors[0].message.contains("integer"));
    }

    #[test]
    fn test_boolean_type_correct() {
        let schema = Schema::new("s", "1").with_field(FieldDefinition::new(
            "flag",
            FieldType::Boolean,
            true,
            None,
        ));
        let message = msg(&[("flag", MessageValue::Boolean(true))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    #[test]
    fn test_float_type_accepted() {
        let schema = Schema::new("s", "1").with_field(FieldDefinition::new(
            "temp",
            FieldType::Float,
            true,
            None,
        ));
        let message = msg(&[("temp", MessageValue::Float(36.6))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    #[test]
    fn test_array_type_accepted() {
        let schema = Schema::new("s", "1").with_field(FieldDefinition::new(
            "tags",
            FieldType::Array,
            true,
            None,
        ));
        let message = msg(&[(
            "tags",
            MessageValue::Array(vec![MessageValue::String("a".into())]),
        )]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    #[test]
    fn test_object_type_accepted() {
        let schema = Schema::new("s", "1").with_field(FieldDefinition::new(
            "meta",
            FieldType::Object,
            true,
            None,
        ));
        let inner: HashMap<String, MessageValue> =
            [("k".to_string(), MessageValue::Integer(1))].into();
        let message = msg(&[("meta", MessageValue::Object(inner))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    // ── format validation ────────────────────────────────────────────────────

    #[test]
    fn test_valid_email_passes() {
        let schema = Schema::new("s", "1").with_field(email_field("email"));
        let message = msg(&[("email", MessageValue::String("user@example.com".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid, "{:?}", result.errors);
    }

    #[test]
    fn test_invalid_email_fails() {
        let schema = Schema::new("s", "1").with_field(email_field("email"));
        let message = msg(&[("email", MessageValue::String("not-an-email".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
        assert!(result.errors[0].message.contains("email"));
    }

    #[test]
    fn test_valid_uri_passes() {
        let schema = Schema::new("s", "1").with_field(uri_field("url"));
        let message = msg(&[(
            "url",
            MessageValue::String("https://example.com/path".into()),
        )]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    #[test]
    fn test_invalid_uri_fails() {
        let schema = Schema::new("s", "1").with_field(uri_field("url"));
        let message = msg(&[("url", MessageValue::String("not-a-uri".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_valid_datetime_passes() {
        let schema = Schema::new("s", "1").with_field(datetime_field("ts"));
        let message = msg(&[("ts", MessageValue::String("2024-01-15T10:30:00Z".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    #[test]
    fn test_invalid_datetime_fails() {
        let schema = Schema::new("s", "1").with_field(datetime_field("ts"));
        let message = msg(&[("ts", MessageValue::String("not-a-date".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_valid_uuid_passes() {
        let schema = Schema::new("s", "1").with_field(uuid_field("id"));
        let message = msg(&[(
            "id",
            MessageValue::String("550e8400-e29b-41d4-a716-446655440000".into()),
        )]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    #[test]
    fn test_invalid_uuid_fails() {
        let schema = Schema::new("s", "1").with_field(uuid_field("id"));
        let message = msg(&[("id", MessageValue::String("not-a-uuid".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
    }

    // ── strict mode ──────────────────────────────────────────────────────────

    #[test]
    fn test_strict_mode_rejects_unknown_field() {
        let schema = Schema::new("s", "1")
            .with_field(string_field("id", true))
            .strict();
        let message = msg(&[
            ("id", MessageValue::String("x".into())),
            ("extra", MessageValue::Boolean(false)),
        ]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.field == "extra"));
    }

    #[test]
    fn test_non_strict_allows_unknown_field() {
        let schema = Schema::new("s", "1").with_field(string_field("id", true));
        let message = msg(&[
            ("id", MessageValue::String("x".into())),
            ("extra", MessageValue::Boolean(false)),
        ]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    // ── schema composition stubs ─────────────────────────────────────────────

    #[test]
    fn test_all_of_composition_stored() {
        let schema = Schema::new("s", "1")
            .with_composition(SchemaComposition::AllOf(vec!["base".into(), "ext".into()]));
        assert!(matches!(
            schema.composition,
            Some(SchemaComposition::AllOf(_))
        ));
    }

    #[test]
    fn test_any_of_composition_stored() {
        let schema =
            Schema::new("s", "1").with_composition(SchemaComposition::AnyOf(vec!["opt1".into()]));
        assert!(matches!(
            schema.composition,
            Some(SchemaComposition::AnyOf(_))
        ));
    }

    // ── schema registry ──────────────────────────────────────────────────────

    #[test]
    fn test_registry_register_and_lookup() {
        let mut registry = SchemaRegistry::new();
        registry.register(basic_schema());
        assert!(registry.lookup("event", "1.0").is_some());
        assert!(registry.lookup("event", "2.0").is_none());
    }

    #[test]
    fn test_registry_validate_via_registry() {
        let mut registry = SchemaRegistry::new();
        registry.register(basic_schema());
        let message = msg(&[("id", MessageValue::String("abc".into()))]);
        let result = registry
            .validate("event", "1.0", &message)
            .expect("schema found");
        assert!(result.is_valid);
    }

    #[test]
    fn test_registry_returns_none_for_missing_schema() {
        let registry = SchemaRegistry::new();
        let message = msg(&[]);
        let result = registry.validate("nonexistent", "1.0", &message);
        assert!(result.is_none());
    }

    #[test]
    fn test_registry_schema_count() {
        let mut registry = SchemaRegistry::new();
        assert_eq!(registry.schema_count(), 0);
        registry.register(basic_schema());
        assert_eq!(registry.schema_count(), 1);
    }

    #[test]
    fn test_registry_list_all() {
        let mut registry = SchemaRegistry::new();
        registry.register(Schema::new("a", "1.0"));
        registry.register(Schema::new("b", "2.0"));
        let list = registry.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_registry_overwrite_same_version() {
        let mut registry = SchemaRegistry::new();
        registry.register(Schema::new("ev", "1.0").with_field(string_field("x", true)));
        registry.register(Schema::new("ev", "1.0").with_field(string_field("y", true)));
        let schema = registry.lookup("ev", "1.0").expect("exists");
        assert_eq!(schema.fields.len(), 1);
        assert_eq!(schema.fields[0].name, "y");
    }

    // ── multiple errors ──────────────────────────────────────────────────────

    #[test]
    fn test_multiple_required_fields_missing() {
        let schema = Schema::new("s", "1")
            .with_field(string_field("a", true))
            .with_field(string_field("b", true));
        let message = msg(&[]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
        assert_eq!(result.error_count(), 2);
    }

    #[test]
    fn test_type_error_and_missing_field() {
        let schema = Schema::new("s", "1")
            .with_field(string_field("name", true))
            .with_field(int_field("count", true));
        // 'name' present with wrong type, 'count' missing
        let message = msg(&[("name", MessageValue::Integer(42))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
        assert_eq!(result.error_count(), 2);
    }

    // ── field type names ─────────────────────────────────────────────────────

    #[test]
    fn test_field_type_names() {
        assert_eq!(FieldType::String.name(), "string");
        assert_eq!(FieldType::Integer.name(), "integer");
        assert_eq!(FieldType::Float.name(), "float");
        assert_eq!(FieldType::Boolean.name(), "boolean");
        assert_eq!(FieldType::Array.name(), "array");
        assert_eq!(FieldType::Object.name(), "object");
    }

    // ── message value helpers ─────────────────────────────────────────────────

    #[test]
    fn test_message_value_field_type_all_variants() {
        assert_eq!(
            MessageValue::String("s".into()).field_type(),
            FieldType::String
        );
        assert_eq!(MessageValue::Integer(1).field_type(), FieldType::Integer);
        assert_eq!(MessageValue::Float(1.0).field_type(), FieldType::Float);
        assert_eq!(MessageValue::Boolean(true).field_type(), FieldType::Boolean);
        assert_eq!(MessageValue::Array(vec![]).field_type(), FieldType::Array);
        let empty: HashMap<String, MessageValue> = HashMap::new();
        assert_eq!(MessageValue::Object(empty).field_type(), FieldType::Object);
    }

    #[test]
    fn test_message_value_as_str() {
        let v = MessageValue::String("hello".into());
        assert_eq!(v.as_str(), Some("hello"));
        assert_eq!(MessageValue::Integer(0).as_str(), None);
    }

    // ── format helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_email_missing_at_sign_invalid() {
        assert!(!validate_email("nodomain"));
    }

    #[test]
    fn test_email_missing_tld_invalid() {
        assert!(!validate_email("user@nodot"));
    }

    #[test]
    fn test_uri_ftp_valid() {
        assert!(validate_uri("ftp://files.example.com/pub/data.txt"));
    }

    #[test]
    fn test_uuid_wrong_length_invalid() {
        assert!(!validate_uuid("550e8400-e29b-41d4-a716-44665544000"));
    }

    #[test]
    fn test_datetime_too_short_invalid() {
        assert!(!validate_datetime("2024-01-15"));
    }

    #[test]
    fn test_validation_result_ok_has_no_errors() {
        let r = ValidationResult::ok();
        assert!(r.is_valid);
        assert_eq!(r.error_count(), 0);
    }

    #[test]
    fn test_field_format_names() {
        assert_eq!(FieldFormat::Email.name(), "email");
        assert_eq!(FieldFormat::Uri.name(), "uri");
        assert_eq!(FieldFormat::DateTime.name(), "date-time");
        assert_eq!(FieldFormat::Uuid.name(), "uuid");
    }

    #[test]
    fn test_schema_strict_flag() {
        let schema = Schema::new("s", "1").strict();
        assert!(schema.strict_mode);
    }

    #[test]
    fn test_registry_default() {
        let registry = SchemaRegistry::default();
        assert_eq!(registry.schema_count(), 0);
    }

    #[test]
    fn test_all_of_sub_schema_names_stored() {
        let composition = SchemaComposition::AllOf(vec!["a".into(), "b".into(), "c".into()]);
        if let SchemaComposition::AllOf(names) = composition {
            assert_eq!(names.len(), 3);
        }
    }

    #[test]
    fn test_any_of_sub_schema_names_stored() {
        let composition = SchemaComposition::AnyOf(vec!["x".into()]);
        if let SchemaComposition::AnyOf(names) = composition {
            assert_eq!(names.len(), 1);
        }
    }

    #[test]
    fn test_strict_mode_with_multiple_unknown_fields() {
        let schema = Schema::new("s", "1")
            .with_field(string_field("id", true))
            .strict();
        let message = msg(&[
            ("id", MessageValue::String("v".into())),
            ("x", MessageValue::Integer(1)),
            ("y", MessageValue::Integer(2)),
        ]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
        assert_eq!(result.error_count(), 2);
    }

    #[test]
    fn test_empty_schema_empty_message_valid() {
        let schema = Schema::new("empty", "1.0");
        let message: StreamMessage = HashMap::new();
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    #[test]
    fn test_optional_field_present_with_wrong_type() {
        let schema = Schema::new("s", "1").with_field(int_field("n", false));
        let message = msg(&[("n", MessageValue::Boolean(true))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(!result.is_valid);
        assert_eq!(result.errors[0].field, "n");
    }

    #[test]
    fn test_field_error_new() {
        let err = FieldError::new("field_x", "something went wrong");
        assert_eq!(err.field, "field_x");
        assert_eq!(err.message, "something went wrong");
    }

    #[test]
    fn test_schema_name_and_version() {
        let schema = Schema::new("orders", "2.5");
        assert_eq!(schema.name, "orders");
        assert_eq!(schema.version, "2.5");
    }

    #[test]
    fn test_schema_fields_count() {
        let schema = Schema::new("s", "1")
            .with_field(string_field("a", true))
            .with_field(int_field("b", false))
            .with_field(string_field("c", true));
        assert_eq!(schema.fields.len(), 3);
    }

    #[test]
    fn test_validate_email_with_subdomain() {
        let schema = Schema::new("s", "1").with_field(email_field("email"));
        let message = msg(&[("email", MessageValue::String("user@sub.example.com".into()))]);
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_https_uri_valid() {
        assert!(validate_uri("https://example.com/path?q=1"));
    }

    #[test]
    fn test_validate_ftp_uri_valid() {
        assert!(validate_uri("ftp://files.example.com/data"));
    }

    #[test]
    fn test_uuid_valid_uppercase_hex() {
        // UUIDs are case-insensitive in practice; validate_uuid checks is_ascii_hexdigit
        assert!(validate_uuid("550E8400-E29B-41D4-A716-446655440000"));
    }

    #[test]
    fn test_datetime_with_space_separator() {
        // Some systems emit "YYYY-MM-DD HH:MM:SS"
        assert!(validate_datetime("2024-03-15 08:30:00"));
    }

    #[test]
    fn test_strict_mode_empty_message_no_unknown() {
        let schema = Schema::new("s", "1")
            .with_field(string_field("id", false))
            .strict();
        // No fields present — no unknown fields, optional field absent.
        let message: StreamMessage = HashMap::new();
        let result = SchemaValidator::validate(&schema, &message);
        assert!(result.is_valid);
    }

    #[test]
    fn test_registry_list_sorted() {
        let mut registry = SchemaRegistry::new();
        registry.register(Schema::new("z_schema", "1.0"));
        registry.register(Schema::new("a_schema", "1.0"));
        let list = registry.list();
        assert_eq!(list[0].0, "a_schema");
        assert_eq!(list[1].0, "z_schema");
    }
}
