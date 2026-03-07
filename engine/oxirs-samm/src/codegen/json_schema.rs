//! SAMM to JSON Schema Generator
//!
//! Generates [JSON Schema](https://json-schema.org/) (draft-07 / 2020-12 compatible)
//! documents from SAMM Aspect Model definitions.  The generated schemas can be
//! used for API payload validation, documentation, and tooling integration.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_samm::parser::parse_aspect_model;
//! use oxirs_samm::codegen::JsonSchemaGenerator;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let aspect = parse_aspect_model("Movement.ttl").await?;
//! let generator = JsonSchemaGenerator::new()
//!     .with_descriptions()
//!     .with_examples();
//! let schema = generator.generate(&aspect)?;
//! println!("{}", serde_json::to_string_pretty(&schema)?);
//! # Ok(())
//! # }
//! ```

use serde_json::{json, Map, Value};

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, ModelElement, Property};

// ------------------------------------------------------------------ //
//  Configuration                                                       //
// ------------------------------------------------------------------ //

/// Configuration for [`JsonSchemaGenerator`].
#[derive(Debug, Clone)]
pub struct JsonSchemaOptions {
    /// Emit `description` fields when available.
    pub include_descriptions: bool,
    /// Emit `examples` arrays when example values are present.
    pub include_examples: bool,
    /// Prefer the `$defs` keyword (2020-12) over `definitions` (draft-07).
    pub use_defs_keyword: bool,
    /// Language tag used for selecting descriptions and titles.
    pub language: String,
}

impl Default for JsonSchemaOptions {
    fn default() -> Self {
        Self {
            include_descriptions: true,
            include_examples: true,
            use_defs_keyword: true,
            language: "en".to_string(),
        }
    }
}

// ------------------------------------------------------------------ //
//  Generator                                                           //
// ------------------------------------------------------------------ //

/// Generates JSON Schema documents from SAMM Aspect Models.
///
/// Use the builder methods to customise the output, then call
/// [`generate`](JsonSchemaGenerator::generate) with an [`Aspect`].
#[derive(Debug, Default, Clone)]
pub struct JsonSchemaGenerator {
    options: JsonSchemaOptions,
}

impl JsonSchemaGenerator {
    /// Create a generator with default options.
    pub fn new() -> Self {
        Self {
            options: JsonSchemaOptions::default(),
        }
    }

    /// Enable `description` fields in the output.
    pub fn with_descriptions(mut self) -> Self {
        self.options.include_descriptions = true;
        self
    }

    /// Enable `examples` arrays in the output.
    pub fn with_examples(mut self) -> Self {
        self.options.include_examples = true;
        self
    }

    /// Disable `description` fields.
    pub fn without_descriptions(mut self) -> Self {
        self.options.include_descriptions = false;
        self
    }

    /// Disable `examples` arrays.
    pub fn without_examples(mut self) -> Self {
        self.options.include_examples = false;
        self
    }

    /// Choose the language for `title` and `description` lookup.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.options.language = lang.into();
        self
    }

    // ---------------------------------------------------------------- //
    //  Public API                                                        //
    // ---------------------------------------------------------------- //

    /// Generate a JSON Schema `Value` for the given `aspect`.
    ///
    /// The returned value represents the root schema object and can be
    /// serialised with `serde_json::to_string_pretty`.
    pub fn generate(&self, aspect: &Aspect) -> Result<Value> {
        let mut root = Map::new();

        // JSON Schema meta-schema identifier
        let schema_keyword = if self.options.use_defs_keyword {
            "https://json-schema.org/draft/2020-12/schema"
        } else {
            "http://json-schema.org/draft-07/schema#"
        };
        root.insert(
            "$schema".to_string(),
            Value::String(schema_keyword.to_string()),
        );
        root.insert("$id".to_string(), Value::String(aspect.urn().to_string()));

        // Title
        let aspect_name = aspect.name();
        let title = aspect
            .metadata()
            .get_preferred_name(&self.options.language)
            .map(|s| s.to_string())
            .unwrap_or_else(|| aspect_name.clone());
        root.insert("title".to_string(), Value::String(title));

        // Description
        if self.options.include_descriptions {
            if let Some(desc) = aspect.metadata().get_description(&self.options.language) {
                root.insert("description".to_string(), Value::String(desc.to_string()));
            }
        }

        root.insert("type".to_string(), Value::String("object".to_string()));

        // Properties and required list
        let (properties_map, required) = self.build_properties(aspect.properties())?;
        root.insert("properties".to_string(), Value::Object(properties_map));
        if !required.is_empty() {
            root.insert(
                "required".to_string(),
                Value::Array(required.into_iter().map(Value::String).collect()),
            );
        }

        // No additional properties by default
        root.insert("additionalProperties".to_string(), Value::Bool(false));

        Ok(Value::Object(root))
    }

    // ---------------------------------------------------------------- //
    //  Internal helpers                                                  //
    // ---------------------------------------------------------------- //

    /// Build a `properties` object from a slice of SAMM properties.
    ///
    /// Returns `(properties_map, required_names)`.
    fn build_properties(&self, props: &[Property]) -> Result<(Map<String, Value>, Vec<String>)> {
        let mut properties_map = Map::new();
        let mut required = Vec::new();

        for prop in props {
            let name = prop_json_name(prop);
            let prop_schema = self.property_to_schema(prop)?;
            properties_map.insert(name.clone(), prop_schema);

            if !prop.optional {
                required.push(name);
            }
        }

        Ok((properties_map, required))
    }

    /// Build a JSON Schema fragment for a single SAMM property.
    fn property_to_schema(&self, prop: &Property) -> Result<Value> {
        let mut schema = Map::new();

        // Title from preferred name
        if let Some(name) = prop.metadata().get_preferred_name(&self.options.language) {
            schema.insert("title".to_string(), Value::String(name.to_string()));
        }

        // Description
        if self.options.include_descriptions {
            if let Some(desc) = prop.metadata().get_description(&self.options.language) {
                schema.insert("description".to_string(), Value::String(desc.to_string()));
            }
        }

        // Type info from characteristic
        if let Some(char) = &prop.characteristic {
            let type_schema = self.characteristic_to_schema(char)?;
            // Merge type schema into our property schema
            if let Value::Object(type_map) = type_schema {
                for (k, v) in type_map {
                    schema.insert(k, v);
                }
            }
        } else {
            // No characteristic – accept any value
            schema.insert("type".to_string(), Value::String("string".to_string()));
        }

        // Examples
        if self.options.include_examples && !prop.example_values.is_empty() {
            let examples: Vec<Value> = prop
                .example_values
                .iter()
                .map(|v| Value::String(v.clone()))
                .collect();
            schema.insert("examples".to_string(), Value::Array(examples));
        }

        Ok(Value::Object(schema))
    }

    /// Convert a SAMM [`Characteristic`] to a JSON Schema type fragment.
    fn characteristic_to_schema(&self, char: &Characteristic) -> Result<Value> {
        let schema = match char.kind() {
            CharacteristicKind::Trait => {
                // Use data type if available
                let json_type = char
                    .data_type
                    .as_deref()
                    .map(|dt| self.data_type_to_json_type(dt))
                    .unwrap_or("string");
                json!({ "type": json_type })
            }

            CharacteristicKind::Quantifiable { unit }
            | CharacteristicKind::Measurement { unit } => {
                let json_type = char
                    .data_type
                    .as_deref()
                    .map(|dt| self.data_type_to_json_type(dt))
                    .unwrap_or("number");
                json!({
                    "type": json_type,
                    "description": format!("Value in {}", unit)
                })
            }

            CharacteristicKind::Duration { unit } => {
                json!({
                    "type": "number",
                    "description": format!("Duration value in {}", unit)
                })
            }

            CharacteristicKind::Enumeration { values } => {
                json!({ "enum": values })
            }

            CharacteristicKind::State {
                values,
                default_value,
            } => {
                let mut s = json!({ "enum": values });
                if let (Some(map), Some(default)) = (s.as_object_mut(), default_value.as_deref()) {
                    map.insert("default".to_string(), Value::String(default.to_string()));
                }
                s
            }

            CharacteristicKind::Collection {
                element_characteristic,
            }
            | CharacteristicKind::List {
                element_characteristic,
            }
            | CharacteristicKind::TimeSeries {
                element_characteristic,
            } => {
                let items = if let Some(inner) = element_characteristic {
                    self.characteristic_to_schema(inner)?
                } else {
                    json!({})
                };
                json!({ "type": "array", "items": items })
            }

            CharacteristicKind::Set {
                element_characteristic,
            } => {
                let items = if let Some(inner) = element_characteristic {
                    self.characteristic_to_schema(inner)?
                } else {
                    json!({})
                };
                json!({ "type": "array", "items": items, "uniqueItems": true })
            }

            CharacteristicKind::SortedSet {
                element_characteristic,
            } => {
                let items = if let Some(inner) = element_characteristic {
                    self.characteristic_to_schema(inner)?
                } else {
                    json!({})
                };
                json!({ "type": "array", "items": items, "uniqueItems": true })
            }

            CharacteristicKind::Code => {
                json!({ "type": "string" })
            }

            CharacteristicKind::Either { left, right } => {
                let left_schema = self.characteristic_to_schema(left)?;
                let right_schema = self.characteristic_to_schema(right)?;
                json!({ "oneOf": [left_schema, right_schema] })
            }

            CharacteristicKind::SingleEntity { entity_type } => {
                // Reference to an entity definition – produce a $ref
                let ref_name = entity_type
                    .split('#')
                    .next_back()
                    .unwrap_or(entity_type.as_str());
                let defs_key = if self.options.use_defs_keyword {
                    "$defs"
                } else {
                    "definitions"
                };
                json!({ "$ref": format!("#{}/{}", defs_key, ref_name) })
            }

            CharacteristicKind::StructuredValue {
                deconstruction_rule: _,
                elements: _,
            } => {
                // Represented as a string with a custom format hint
                json!({ "type": "string", "format": "structured-value" })
            }
        };

        // Apply constraint keywords on top
        let schema = self.apply_constraints(char, schema)?;

        Ok(schema)
    }

    /// Apply SAMM constraints as JSON Schema keywords.
    fn apply_constraints(&self, char: &Characteristic, mut schema: Value) -> Result<Value> {
        use crate::metamodel::Constraint;

        for constraint in &char.constraints {
            if let Some(obj) = schema.as_object_mut() {
                match constraint {
                    Constraint::RangeConstraint {
                        min_value,
                        max_value,
                        ..
                    } => {
                        if let Some(min) = min_value {
                            if let Ok(n) = min.parse::<f64>() {
                                obj.insert("minimum".to_string(), json!(n));
                            }
                        }
                        if let Some(max) = max_value {
                            if let Ok(n) = max.parse::<f64>() {
                                obj.insert("maximum".to_string(), json!(n));
                            }
                        }
                    }
                    Constraint::LengthConstraint {
                        min_value,
                        max_value,
                    } => {
                        if let Some(min) = min_value {
                            obj.insert("minLength".to_string(), json!(min));
                        }
                        if let Some(max) = max_value {
                            obj.insert("maxLength".to_string(), json!(max));
                        }
                    }
                    Constraint::RegularExpressionConstraint { pattern } => {
                        obj.insert("pattern".to_string(), Value::String(pattern.clone()));
                    }
                    Constraint::LanguageConstraint { .. } | Constraint::LocaleConstraint { .. } => {
                        // Represented as metadata; no direct JSON Schema equivalent
                    }
                    Constraint::EncodingConstraint { encoding } => {
                        obj.insert(
                            "contentEncoding".to_string(),
                            Value::String(encoding.clone()),
                        );
                    }
                    Constraint::FixedPointConstraint { integer, scale } => {
                        // Represented as multipleOf: 1 / 10^scale (best effort)
                        let _ = (integer, scale); // used for documentation only
                    }
                }
            }
        }
        Ok(schema)
    }

    /// Map a SAMM / XSD data type string to a JSON Schema type keyword.
    pub fn data_type_to_json_type(&self, data_type: &str) -> &'static str {
        if data_type.ends_with("boolean") {
            return "boolean";
        }
        if data_type.ends_with("int")
            || data_type.ends_with("integer")
            || data_type.ends_with("long")
            || data_type.ends_with("short")
            || data_type.ends_with("byte")
            || data_type.ends_with("unsignedInt")
            || data_type.ends_with("unsignedLong")
            || data_type.ends_with("unsignedShort")
            || data_type.ends_with("unsignedByte")
            || data_type.ends_with("positiveInteger")
            || data_type.ends_with("negativeInteger")
            || data_type.ends_with("nonNegativeInteger")
            || data_type.ends_with("nonPositiveInteger")
        {
            return "integer";
        }
        if data_type.ends_with("decimal")
            || data_type.ends_with("float")
            || data_type.ends_with("double")
        {
            return "number";
        }
        "string"
    }
}

// ------------------------------------------------------------------ //
//  Utility                                                             //
// ------------------------------------------------------------------ //

/// Return the JSON field name for a property (payload name if set, otherwise
/// the local name from the URN).
fn prop_json_name(prop: &Property) -> String {
    prop.payload_name.clone().unwrap_or_else(|| prop.name())
}

// ------------------------------------------------------------------ //
//  JsonSchemaValidator                                                 //
// ------------------------------------------------------------------ //

/// A single JSON Schema validation error.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    /// JSON Pointer path to the failing instance location (e.g. `"/name"` or `""` for root).
    pub path: String,
    /// Human-readable error message.
    pub message: String,
    /// JSON Pointer path into the schema where the violation was detected (e.g. `"/type"`).
    pub schema_path: String,
}

impl ValidationError {
    /// Create a new `ValidationError`.
    pub fn new(
        path: impl Into<String>,
        message: impl Into<String>,
        schema_path: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
            schema_path: schema_path.into(),
        }
    }
}

/// Validates a JSON instance against a JSON Schema value.
///
/// Supports a practical subset of JSON Schema keywords:
///
/// - `type` — string/number/integer/boolean/array/object/null type checking
/// - `required` — checks that all listed property names are present
/// - `enum` — validates value membership
/// - `minimum` / `maximum` — numeric range constraints
/// - `minLength` / `maxLength` — string length constraints
/// - `additionalProperties: false` — rejects unknown object properties
/// - `properties` — recursive validation of nested object properties
///
/// # Example
///
/// ```rust
/// use oxirs_samm::codegen::{JsonSchemaValidator, ValidationError};
/// use serde_json::json;
///
/// let validator = JsonSchemaValidator::new();
/// let schema = json!({ "type": "object", "required": ["name"], "properties": { "name": { "type": "string" } } });
/// let instance = json!({ "name": 42 });
/// let errors = validator.validate(&schema, &instance);
/// assert!(!errors.is_empty());
/// ```
#[derive(Debug, Default, Clone)]
pub struct JsonSchemaValidator;

impl JsonSchemaValidator {
    /// Create a new validator instance.
    pub fn new() -> Self {
        Self
    }

    /// Validate `instance` against `schema`, returning all validation errors.
    ///
    /// An empty `Vec` means the instance is valid.
    pub fn validate(
        &self,
        schema: &serde_json::Value,
        instance: &serde_json::Value,
    ) -> Vec<ValidationError> {
        self.validate_with_path(schema, instance, "", "")
    }

    /// Internal recursive validation with path tracking.
    fn validate_with_path(
        &self,
        schema: &serde_json::Value,
        instance: &serde_json::Value,
        path: &str,
        schema_path: &str,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // ── type ──────────────────────────────────────────────────────────
        if let Some(type_value) = schema.get("type") {
            if let Some(type_str) = type_value.as_str() {
                let type_ok = match type_str {
                    "string" => instance.is_string(),
                    "number" => instance.is_number(),
                    "integer" => instance.as_f64().map(|n| n.fract() == 0.0).unwrap_or(false),
                    "boolean" => instance.is_boolean(),
                    "array" => instance.is_array(),
                    "object" => instance.is_object(),
                    "null" => instance.is_null(),
                    _ => true, // unknown types pass through
                };
                if !type_ok {
                    errors.push(ValidationError::new(
                        path,
                        format!(
                            "expected type '{}' but got '{}'",
                            type_str,
                            json_type_name(instance)
                        ),
                        format!("{}/type", schema_path),
                    ));
                }
            }
        }

        // ── required ──────────────────────────────────────────────────────
        if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
            if let Some(obj) = instance.as_object() {
                for req in required {
                    if let Some(field) = req.as_str() {
                        if !obj.contains_key(field) {
                            errors.push(ValidationError::new(
                                path,
                                format!("required property '{}' is missing", field),
                                format!("{}/required", schema_path),
                            ));
                        }
                    }
                }
            }
        }

        // ── enum ──────────────────────────────────────────────────────────
        if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
            if !enum_values.iter().any(|e| e == instance) {
                errors.push(ValidationError::new(
                    path,
                    "value is not one of the allowed enum values".to_string(),
                    format!("{}/enum", schema_path),
                ));
            }
        }

        // ── minimum / maximum ─────────────────────────────────────────────
        if let Some(instance_num) = instance.as_f64() {
            if let Some(min) = schema.get("minimum").and_then(|v| v.as_f64()) {
                if instance_num < min {
                    errors.push(ValidationError::new(
                        path,
                        format!("value {} is less than minimum {}", instance_num, min),
                        format!("{}/minimum", schema_path),
                    ));
                }
            }
            if let Some(max) = schema.get("maximum").and_then(|v| v.as_f64()) {
                if instance_num > max {
                    errors.push(ValidationError::new(
                        path,
                        format!("value {} is greater than maximum {}", instance_num, max),
                        format!("{}/maximum", schema_path),
                    ));
                }
            }
        }

        // ── minLength / maxLength ─────────────────────────────────────────
        if let Some(s) = instance.as_str() {
            let char_count = s.chars().count();
            if let Some(min_len) = schema.get("minLength").and_then(|v| v.as_u64()) {
                if (char_count as u64) < min_len {
                    errors.push(ValidationError::new(
                        path,
                        format!(
                            "string length {} is less than minLength {}",
                            char_count, min_len
                        ),
                        format!("{}/minLength", schema_path),
                    ));
                }
            }
            if let Some(max_len) = schema.get("maxLength").and_then(|v| v.as_u64()) {
                if (char_count as u64) > max_len {
                    errors.push(ValidationError::new(
                        path,
                        format!("string length {} exceeds maxLength {}", char_count, max_len),
                        format!("{}/maxLength", schema_path),
                    ));
                }
            }
        }

        // ── additionalProperties: false ───────────────────────────────────
        if let Some(add_props) = schema.get("additionalProperties") {
            if add_props == &serde_json::Value::Bool(false) {
                if let Some(obj) = instance.as_object() {
                    let allowed: std::collections::HashSet<&str> = schema
                        .get("properties")
                        .and_then(|p| p.as_object())
                        .map(|p| p.keys().map(|k| k.as_str()).collect())
                        .unwrap_or_default();

                    for key in obj.keys() {
                        if !allowed.contains(key.as_str()) {
                            let err_path = if path.is_empty() {
                                key.clone()
                            } else {
                                format!("{}/{}", path, key)
                            };
                            errors.push(ValidationError::new(
                                err_path,
                                format!("additional property '{}' is not allowed", key),
                                format!("{}/additionalProperties", schema_path),
                            ));
                        }
                    }
                }
            }
        }

        // ── properties (recursive) ────────────────────────────────────────
        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            if let Some(instance_obj) = instance.as_object() {
                for (prop_key, prop_schema) in properties {
                    if let Some(prop_value) = instance_obj.get(prop_key) {
                        let child_path = if path.is_empty() {
                            format!("/{}", prop_key)
                        } else {
                            format!("{}/{}", path, prop_key)
                        };
                        let child_schema_path = format!("{}/properties/{}", schema_path, prop_key);
                        let child_errors = self.validate_with_path(
                            prop_schema,
                            prop_value,
                            &child_path,
                            &child_schema_path,
                        );
                        errors.extend(child_errors);
                    }
                }
            }
        }

        errors
    }
}

/// Return a human-readable type name for a JSON value.
fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

// ------------------------------------------------------------------ //
//  Tests                                                               //
// ------------------------------------------------------------------ //

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};

    fn speed_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "Movement".to_string());
        aspect
            .metadata
            .add_description("en".to_string(), "Describes movement data".to_string());

        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#SpeedChar".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:kilometrePerHour".to_string(),
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());

        let prop =
            Property::new("urn:samm:org.example:1.0.0#speed".to_string()).with_characteristic(char);

        aspect.add_property(prop);
        aspect
    }

    #[test]
    fn test_generate_basic_schema() {
        let aspect = speed_aspect();
        let gen = JsonSchemaGenerator::new();
        let schema = gen.generate(&aspect).expect("generation should succeed");

        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["speed"].is_object());
    }

    #[test]
    fn test_schema_has_description() {
        let aspect = speed_aspect();
        let gen = JsonSchemaGenerator::new().with_descriptions();
        let schema = gen.generate(&aspect).expect("generation should succeed");
        assert_eq!(schema["description"], "Describes movement data");
    }

    #[test]
    fn test_schema_no_description() {
        let aspect = speed_aspect();
        let gen = JsonSchemaGenerator::new().without_descriptions();
        let schema = gen.generate(&aspect).expect("generation should succeed");
        assert!(schema.get("description").is_none());
    }

    #[test]
    fn test_required_non_optional_properties() {
        let aspect = speed_aspect();
        let gen = JsonSchemaGenerator::new();
        let schema = gen.generate(&aspect).expect("generation should succeed");
        let required = schema["required"]
            .as_array()
            .expect("required should be array");
        assert!(required.iter().any(|v| v == "speed"));
    }

    #[test]
    fn test_optional_not_in_required() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#Char".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        let prop = Property::new("urn:samm:org.example:1.0.0#optProp".to_string())
            .with_characteristic(char)
            .as_optional();
        aspect.add_property(prop);

        let gen = JsonSchemaGenerator::new();
        let schema = gen.generate(&aspect).expect("generation should succeed");
        // required array may be absent or not contain optProp
        if let Some(arr) = schema.get("required").and_then(|v| v.as_array()) {
            assert!(!arr.iter().any(|v| v == "optProp"));
        }
    }

    #[test]
    fn test_enumeration_generates_enum_keyword() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#StatusEnum".to_string(),
            CharacteristicKind::Enumeration {
                values: vec!["Active".to_string(), "Inactive".to_string()],
            },
        );
        let prop = Property::new("urn:samm:org.example:1.0.0#status".to_string())
            .with_characteristic(char);
        aspect.add_property(prop);

        let gen = JsonSchemaGenerator::new();
        let schema = gen.generate(&aspect).expect("generation should succeed");
        let status_prop = &schema["properties"]["status"];
        assert!(status_prop["enum"].is_array());
        let vals = status_prop["enum"].as_array().expect("enum is array");
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_collection_generates_array_type() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        let inner = Characteristic::new(
            "urn:samm:org.example:1.0.0#Inner".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#Names".to_string(),
            CharacteristicKind::List {
                element_characteristic: Some(Box::new(inner)),
            },
        );
        let prop =
            Property::new("urn:samm:org.example:1.0.0#names".to_string()).with_characteristic(char);
        aspect.add_property(prop);

        let gen = JsonSchemaGenerator::new();
        let schema = gen.generate(&aspect).expect("generation should succeed");
        assert_eq!(schema["properties"]["names"]["type"], "array");
    }

    #[test]
    fn test_data_type_to_json_type_mapping() {
        let gen = JsonSchemaGenerator::new();
        assert_eq!(
            gen.data_type_to_json_type("http://www.w3.org/2001/XMLSchema#boolean"),
            "boolean"
        );
        assert_eq!(
            gen.data_type_to_json_type("http://www.w3.org/2001/XMLSchema#int"),
            "integer"
        );
        assert_eq!(
            gen.data_type_to_json_type("http://www.w3.org/2001/XMLSchema#float"),
            "number"
        );
        assert_eq!(
            gen.data_type_to_json_type("http://www.w3.org/2001/XMLSchema#string"),
            "string"
        );
        assert_eq!(gen.data_type_to_json_type("xsd:dateTime"), "string");
    }

    #[test]
    fn test_either_generates_one_of() {
        let left = Characteristic::new(
            "urn:samm:org.example:1.0.0#Left".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        let right = Characteristic::new(
            "urn:samm:org.example:1.0.0#Right".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#int".to_string());

        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#EitherChar".to_string(),
            CharacteristicKind::Either {
                left: Box::new(left),
                right: Box::new(right),
            },
        );

        let gen = JsonSchemaGenerator::new();
        let schema = gen
            .characteristic_to_schema(&char)
            .expect("generation should succeed");
        assert!(schema["oneOf"].is_array());
        assert_eq!(schema["oneOf"].as_array().map(|a| a.len()), Some(2));
    }

    #[test]
    fn test_draft_07_schema_identifier() {
        let gen = JsonSchemaGenerator {
            options: JsonSchemaOptions {
                use_defs_keyword: false,
                ..Default::default()
            },
        };
        let aspect = speed_aspect();
        let schema = gen.generate(&aspect).expect("generation should succeed");
        assert_eq!(schema["$schema"], "http://json-schema.org/draft-07/schema#");
    }

    // ─────────────────────────────────────────────────────────────────────
    // JsonSchemaValidator tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_validator_valid_string_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "string" });
        let instance = serde_json::json!("hello");
        assert!(v.validate(&schema, &instance).is_empty());
    }

    #[test]
    fn test_validator_invalid_string_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "string" });
        let instance = serde_json::json!(42);
        let errors = v.validate(&schema, &instance);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("string"));
    }

    #[test]
    fn test_validator_valid_number_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "number" });
        assert!(v.validate(&schema, &serde_json::json!(3.5)).is_empty());
    }

    #[test]
    fn test_validator_invalid_number_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "number" });
        let errors = v.validate(&schema, &serde_json::json!("not a number"));
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_valid_integer_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "integer" });
        assert!(v.validate(&schema, &serde_json::json!(7)).is_empty());
    }

    #[test]
    fn test_validator_invalid_integer_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "integer" });
        // 3.5 has a fractional part — not an integer
        let errors = v.validate(&schema, &serde_json::json!(3.5));
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_valid_boolean_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "boolean" });
        assert!(v.validate(&schema, &serde_json::json!(true)).is_empty());
    }

    #[test]
    fn test_validator_invalid_boolean_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "boolean" });
        let errors = v.validate(&schema, &serde_json::json!("true"));
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_valid_array_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "array" });
        assert!(v
            .validate(&schema, &serde_json::json!([1, 2, 3]))
            .is_empty());
    }

    #[test]
    fn test_validator_invalid_array_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "array" });
        let errors = v.validate(&schema, &serde_json::json!({}));
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_valid_object_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "object" });
        assert!(v.validate(&schema, &serde_json::json!({"a": 1})).is_empty());
    }

    #[test]
    fn test_validator_invalid_object_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "object" });
        let errors = v.validate(&schema, &serde_json::json!([1, 2]));
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_valid_null_type() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "null" });
        assert!(v.validate(&schema, &serde_json::json!(null)).is_empty());
    }

    #[test]
    fn test_validator_required_fields_all_present() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name", "age"]
        });
        let instance = serde_json::json!({ "name": "Alice", "age": 30 });
        assert!(v.validate(&schema, &instance).is_empty());
    }

    #[test]
    fn test_validator_required_fields_missing() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name"]
        });
        let instance = serde_json::json!({ "age": 30 });
        let errors = v.validate(&schema, &instance);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.message.contains("name")));
    }

    #[test]
    fn test_validator_required_multiple_missing() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({
            "type": "object",
            "required": ["a", "b", "c"]
        });
        let instance = serde_json::json!({});
        let errors = v.validate(&schema, &instance);
        // Should have at least 3 errors (one per missing field)
        assert!(errors.len() >= 3);
    }

    #[test]
    fn test_validator_enum_valid() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "enum": ["red", "green", "blue"] });
        assert!(v.validate(&schema, &serde_json::json!("green")).is_empty());
    }

    #[test]
    fn test_validator_enum_invalid() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "enum": ["red", "green", "blue"] });
        let errors = v.validate(&schema, &serde_json::json!("purple"));
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("enum"));
    }

    #[test]
    fn test_validator_minimum_valid() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "number", "minimum": 0.0 });
        assert!(v.validate(&schema, &serde_json::json!(5)).is_empty());
    }

    #[test]
    fn test_validator_minimum_violation() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "number", "minimum": 10 });
        let errors = v.validate(&schema, &serde_json::json!(5));
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.schema_path.contains("minimum")));
    }

    #[test]
    fn test_validator_maximum_valid() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "number", "maximum": 100 });
        assert!(v.validate(&schema, &serde_json::json!(50)).is_empty());
    }

    #[test]
    fn test_validator_maximum_violation() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "number", "maximum": 10 });
        let errors = v.validate(&schema, &serde_json::json!(20));
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.schema_path.contains("maximum")));
    }

    #[test]
    fn test_validator_min_length_valid() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "string", "minLength": 3 });
        assert!(v.validate(&schema, &serde_json::json!("abcd")).is_empty());
    }

    #[test]
    fn test_validator_min_length_violation() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "string", "minLength": 5 });
        let errors = v.validate(&schema, &serde_json::json!("hi"));
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.schema_path.contains("minLength")));
    }

    #[test]
    fn test_validator_max_length_violation() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({ "type": "string", "maxLength": 3 });
        let errors = v.validate(&schema, &serde_json::json!("toolong"));
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.schema_path.contains("maxLength")));
    }

    #[test]
    fn test_validator_additional_properties_blocked() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "additionalProperties": false
        });
        let instance = serde_json::json!({ "name": "Alice", "extra": "oops" });
        let errors = v.validate(&schema, &instance);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.message.contains("extra")));
    }

    #[test]
    fn test_validator_additional_properties_allowed_when_not_false() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" } }
        });
        let instance = serde_json::json!({ "name": "Alice", "extra": "ok" });
        // No additionalProperties: false — should pass
        assert!(v.validate(&schema, &instance).is_empty());
    }

    #[test]
    fn test_validator_nested_property_type_error() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": { "type": "integer" }
                    }
                }
            }
        });
        // age should be integer but is string
        let instance = serde_json::json!({ "user": { "age": "not-a-number" } });
        let errors = v.validate(&schema, &instance);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.path.contains("age")));
    }

    #[test]
    fn test_validation_error_fields() {
        let err = ValidationError::new("/name", "required property 'name' is missing", "/required");
        assert_eq!(err.path, "/name");
        assert!(err.message.contains("name"));
        assert_eq!(err.schema_path, "/required");
    }

    #[test]
    fn test_validator_no_errors_for_valid_complex_object() {
        let v = JsonSchemaValidator::new();
        let schema = serde_json::json!({
            "type": "object",
            "required": ["id", "name"],
            "properties": {
                "id": { "type": "integer" },
                "name": { "type": "string", "minLength": 1 },
                "score": { "type": "number", "minimum": 0.0, "maximum": 100.0 }
            },
            "additionalProperties": false
        });
        let instance = serde_json::json!({
            "id": 42,
            "name": "Alice",
            "score": 95.5
        });
        let errors = v.validate(&schema, &instance);
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }
}
