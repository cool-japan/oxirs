//! SAMM Aspect instance factory and validator.
//!
//! Provides factory functions that create, validate, and serialise
//! instances of SAMM Aspect schemas.

use std::collections::HashMap;

/// Schema describing a single SAMM property.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertySchema {
    /// Property name.
    pub name: String,
    /// Expected data type string (e.g. `"string"`, `"integer"`, `"float"`, `"boolean"`).
    pub data_type: String,
    /// Whether the property may be omitted.
    pub optional: bool,
    /// Whether the property holds a list of values.
    pub collection: bool,
}

impl PropertySchema {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, data_type: impl Into<String>, optional: bool, collection: bool) -> Self {
        Self { name: name.into(), data_type: data_type.into(), optional, collection }
    }
}

/// Top-level schema describing a SAMM Aspect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AspectSchema {
    /// Aspect name.
    pub name: String,
    /// Aspect version string.
    pub version: String,
    /// Properties defined on the Aspect.
    pub properties: Vec<PropertySchema>,
}

impl AspectSchema {
    /// Construct a new schema.
    pub fn new(name: impl Into<String>, version: impl Into<String>, properties: Vec<PropertySchema>) -> Self {
        Self { name: name.into(), version: version.into(), properties }
    }
}

/// A value held by an instance property.
#[derive(Debug, Clone, PartialEq)]
pub enum InstanceValue {
    /// String value.
    String(String),
    /// Integer value.
    Integer(i64),
    /// Floating-point value.
    Float(f64),
    /// Boolean value.
    Boolean(bool),
    /// Collection of values.
    List(Vec<InstanceValue>),
    /// Absent / null value.
    Null,
}

/// An Aspect instance holding concrete property values.
#[derive(Debug, Clone, PartialEq)]
pub struct AspectInstance {
    /// Name of the schema this instance was built from.
    pub schema_name: String,
    /// Mapping from property name to value.
    pub values: HashMap<String, InstanceValue>,
}

/// Errors produced by the factory.
#[derive(Debug, Clone, PartialEq)]
pub enum FactoryError {
    /// A required (non-optional) property is absent.
    MissingProperty(String),
    /// A value has a type that does not match the schema expectation.
    TypeMismatch {
        property: String,
        expected: String,
        got: String,
    },
    /// The schema definition is itself invalid.
    InvalidSchema(String),
}

impl std::fmt::Display for FactoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactoryError::MissingProperty(p) => write!(f, "missing required property: {p}"),
            FactoryError::TypeMismatch { property, expected, got } => {
                write!(f, "type mismatch on '{property}': expected {expected}, got {got}")
            }
            FactoryError::InvalidSchema(msg) => write!(f, "invalid schema: {msg}"),
        }
    }
}

/// Stateless instance factory.
pub struct InstanceFactory;

impl InstanceFactory {
    /// Create a validated `AspectInstance` from raw values.
    ///
    /// Returns `Err` with the first fatal error encountered.
    pub fn create(
        schema: &AspectSchema,
        values: HashMap<String, InstanceValue>,
    ) -> Result<AspectInstance, FactoryError> {
        // Validate required properties are present.
        for prop in &schema.properties {
            if !prop.optional && !values.contains_key(&prop.name) {
                return Err(FactoryError::MissingProperty(prop.name.clone()));
            }
        }
        // Validate types for present values.
        for prop in &schema.properties {
            if let Some(val) = values.get(&prop.name) {
                if !Self::is_compatible_type(val, &prop.data_type) {
                    return Err(FactoryError::TypeMismatch {
                        property: prop.name.clone(),
                        expected: prop.data_type.clone(),
                        got: Self::value_type_name(val).to_string(),
                    });
                }
            }
        }
        Ok(AspectInstance { schema_name: schema.name.clone(), values })
    }

    /// Validate an existing instance against a schema, returning all errors
    /// rather than stopping at the first.
    pub fn validate(instance: &AspectInstance, schema: &AspectSchema) -> Vec<FactoryError> {
        let mut errors = Vec::new();
        for prop in &schema.properties {
            match instance.values.get(&prop.name) {
                None => {
                    if !prop.optional {
                        errors.push(FactoryError::MissingProperty(prop.name.clone()));
                    }
                }
                Some(val) => {
                    if !Self::is_compatible_type(val, &prop.data_type) {
                        errors.push(FactoryError::TypeMismatch {
                            property: prop.name.clone(),
                            expected: prop.data_type.clone(),
                            got: Self::value_type_name(val).to_string(),
                        });
                    }
                }
            }
        }
        errors
    }

    /// Create an instance with default (null/zero) values for each property.
    pub fn default_instance(schema: &AspectSchema) -> AspectInstance {
        let mut values = HashMap::new();
        for prop in &schema.properties {
            let default_val = Self::default_for_type(&prop.data_type);
            values.insert(prop.name.clone(), default_val);
        }
        AspectInstance { schema_name: schema.name.clone(), values }
    }

    /// Serialise an instance to a JSON-like string (hand-rolled, no external deps).
    pub fn to_json(instance: &AspectInstance) -> String {
        let mut pairs: Vec<String> = Vec::new();
        // Sort keys for deterministic output.
        let mut keys: Vec<&String> = instance.values.keys().collect();
        keys.sort();
        for key in keys {
            let val = &instance.values[key];
            pairs.push(format!("  \"{}\": {}", key, Self::value_to_json(val)));
        }
        format!("{{\n{}\n}}", pairs.join(",\n"))
    }

    /// Parse a JSON object string into an `AspectInstance`.
    ///
    /// Only handles the simple key-value types produced by `to_json`.
    pub fn from_json(schema: &AspectSchema, json: &str) -> Result<AspectInstance, FactoryError> {
        let trimmed = json.trim();
        if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return Err(FactoryError::InvalidSchema("Expected JSON object".to_string()));
        }
        let inner = &trimmed[1..trimmed.len() - 1];
        let mut values: HashMap<String, InstanceValue> = HashMap::new();

        for line in inner.lines() {
            let line = line.trim().trim_end_matches(',');
            if line.is_empty() {
                continue;
            }
            // Split on first ':'
            let colon_pos = line.find(':').ok_or_else(|| {
                FactoryError::InvalidSchema(format!("Malformed JSON line: {line}"))
            })?;
            let raw_key = line[..colon_pos].trim().trim_matches('"');
            let raw_val = line[colon_pos + 1..].trim();

            let value = Self::parse_json_value(raw_val)?;
            values.insert(raw_key.to_string(), value);
        }
        Self::create(schema, values)
    }

    /// Check whether `value` is compatible with the given `expected` type string.
    pub fn is_compatible_type(value: &InstanceValue, expected: &str) -> bool {
        match (value, expected) {
            (InstanceValue::String(_), "string") => true,
            (InstanceValue::Integer(_), "integer") => true,
            (InstanceValue::Float(_), "float") => true,
            (InstanceValue::Boolean(_), "boolean") => true,
            (InstanceValue::List(_), t) if t.starts_with("list") => true,
            (InstanceValue::Null, _) => true, // Null is compatible with any type.
            _ => false,
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn value_type_name(v: &InstanceValue) -> &'static str {
        match v {
            InstanceValue::String(_) => "string",
            InstanceValue::Integer(_) => "integer",
            InstanceValue::Float(_) => "float",
            InstanceValue::Boolean(_) => "boolean",
            InstanceValue::List(_) => "list",
            InstanceValue::Null => "null",
        }
    }

    fn default_for_type(data_type: &str) -> InstanceValue {
        match data_type {
            "string" => InstanceValue::String(String::new()),
            "integer" => InstanceValue::Integer(0),
            "float" => InstanceValue::Float(0.0),
            "boolean" => InstanceValue::Boolean(false),
            t if t.starts_with("list") => InstanceValue::List(Vec::new()),
            _ => InstanceValue::Null,
        }
    }

    fn value_to_json(v: &InstanceValue) -> String {
        match v {
            InstanceValue::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            InstanceValue::Integer(n) => n.to_string(),
            InstanceValue::Float(f) => format!("{f}"),
            InstanceValue::Boolean(b) => b.to_string(),
            InstanceValue::List(items) => {
                let inner: Vec<String> = items.iter().map(Self::value_to_json).collect();
                format!("[{}]", inner.join(", "))
            }
            InstanceValue::Null => "null".to_string(),
        }
    }

    fn parse_json_value(s: &str) -> Result<InstanceValue, FactoryError> {
        let s = s.trim();
        if s == "null" {
            return Ok(InstanceValue::Null);
        }
        if s == "true" {
            return Ok(InstanceValue::Boolean(true));
        }
        if s == "false" {
            return Ok(InstanceValue::Boolean(false));
        }
        if s.starts_with('"') && s.ends_with('"') {
            return Ok(InstanceValue::String(s[1..s.len() - 1].replace("\\\"", "\"")));
        }
        if s.contains('.') {
            if let Ok(f) = s.parse::<f64>() {
                return Ok(InstanceValue::Float(f));
            }
        }
        if let Ok(i) = s.parse::<i64>() {
            return Ok(InstanceValue::Integer(i));
        }
        Err(FactoryError::InvalidSchema(format!("Cannot parse JSON value: {s}")))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_with_props(props: &[(&str, &str, bool, bool)]) -> AspectSchema {
        let properties = props
            .iter()
            .map(|(n, t, o, c)| PropertySchema::new(*n, *t, *o, *c))
            .collect();
        AspectSchema::new("TestAspect", "1.0.0", properties)
    }

    fn vals(entries: &[(&str, InstanceValue)]) -> HashMap<String, InstanceValue> {
        entries.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    // ── AspectSchema / PropertySchema ────────────────────────────────────────

    #[test]
    fn test_schema_construction() {
        let s = schema_with_props(&[("age", "integer", false, false)]);
        assert_eq!(s.name, "TestAspect");
        assert_eq!(s.version, "1.0.0");
        assert_eq!(s.properties.len(), 1);
    }

    #[test]
    fn test_property_schema_optional_flag() {
        let p = PropertySchema::new("x", "string", true, false);
        assert!(p.optional);
        assert!(!p.collection);
    }

    // ── InstanceFactory::create ──────────────────────────────────────────────

    #[test]
    fn test_create_valid() {
        let schema = schema_with_props(&[("name", "string", false, false)]);
        let v = vals(&[("name", InstanceValue::String("Alice".to_string()))]);
        let inst = InstanceFactory::create(&schema, v).expect("should succeed");
        assert_eq!(inst.schema_name, "TestAspect");
    }

    #[test]
    fn test_create_missing_required() {
        let schema = schema_with_props(&[("age", "integer", false, false)]);
        let result = InstanceFactory::create(&schema, HashMap::new());
        assert!(matches!(result, Err(FactoryError::MissingProperty(_))));
    }

    #[test]
    fn test_create_optional_missing_ok() {
        let schema = schema_with_props(&[("note", "string", true, false)]);
        let result = InstanceFactory::create(&schema, HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_type_mismatch() {
        let schema = schema_with_props(&[("count", "integer", false, false)]);
        let v = vals(&[("count", InstanceValue::String("oops".to_string()))]);
        let result = InstanceFactory::create(&schema, v);
        assert!(matches!(result, Err(FactoryError::TypeMismatch { .. })));
    }

    #[test]
    fn test_create_null_compatible() {
        let schema = schema_with_props(&[("x", "string", false, false)]);
        let v = vals(&[("x", InstanceValue::Null)]);
        assert!(InstanceFactory::create(&schema, v).is_ok());
    }

    #[test]
    fn test_create_boolean() {
        let schema = schema_with_props(&[("active", "boolean", false, false)]);
        let v = vals(&[("active", InstanceValue::Boolean(true))]);
        assert!(InstanceFactory::create(&schema, v).is_ok());
    }

    #[test]
    fn test_create_float() {
        let schema = schema_with_props(&[("temp", "float", false, false)]);
        let v = vals(&[("temp", InstanceValue::Float(36.6))]);
        assert!(InstanceFactory::create(&schema, v).is_ok());
    }

    // ── InstanceFactory::validate ────────────────────────────────────────────

    #[test]
    fn test_validate_no_errors() {
        let schema = schema_with_props(&[("name", "string", false, false)]);
        let inst = AspectInstance {
            schema_name: "TestAspect".to_string(),
            values: vals(&[("name", InstanceValue::String("Bob".to_string()))]),
        };
        let errs = InstanceFactory::validate(&inst, &schema);
        assert!(errs.is_empty());
    }

    #[test]
    fn test_validate_missing_required() {
        let schema = schema_with_props(&[("name", "string", false, false), ("age", "integer", false, false)]);
        let inst = AspectInstance {
            schema_name: "TestAspect".to_string(),
            values: vals(&[("name", InstanceValue::String("Bob".to_string()))]),
        };
        let errs = InstanceFactory::validate(&inst, &schema);
        assert_eq!(errs.len(), 1);
        assert!(matches!(&errs[0], FactoryError::MissingProperty(p) if p == "age"));
    }

    #[test]
    fn test_validate_type_mismatch() {
        let schema = schema_with_props(&[("age", "integer", false, false)]);
        let inst = AspectInstance {
            schema_name: "TestAspect".to_string(),
            values: vals(&[("age", InstanceValue::Boolean(true))]),
        };
        let errs = InstanceFactory::validate(&inst, &schema);
        assert_eq!(errs.len(), 1);
        assert!(matches!(&errs[0], FactoryError::TypeMismatch { .. }));
    }

    #[test]
    fn test_validate_multiple_errors() {
        let schema = schema_with_props(&[
            ("a", "string", false, false),
            ("b", "integer", false, false),
        ]);
        let inst = AspectInstance {
            schema_name: "TestAspect".to_string(),
            values: vals(&[("a", InstanceValue::Integer(1))]),
        };
        let errs = InstanceFactory::validate(&inst, &schema);
        assert_eq!(errs.len(), 2);
    }

    // ── InstanceFactory::default_instance ───────────────────────────────────

    #[test]
    fn test_default_instance_string() {
        let schema = schema_with_props(&[("name", "string", false, false)]);
        let inst = InstanceFactory::default_instance(&schema);
        assert_eq!(inst.values["name"], InstanceValue::String(String::new()));
    }

    #[test]
    fn test_default_instance_integer() {
        let schema = schema_with_props(&[("count", "integer", false, false)]);
        let inst = InstanceFactory::default_instance(&schema);
        assert_eq!(inst.values["count"], InstanceValue::Integer(0));
    }

    #[test]
    fn test_default_instance_boolean() {
        let schema = schema_with_props(&[("flag", "boolean", false, false)]);
        let inst = InstanceFactory::default_instance(&schema);
        assert_eq!(inst.values["flag"], InstanceValue::Boolean(false));
    }

    #[test]
    fn test_default_instance_float() {
        let schema = schema_with_props(&[("ratio", "float", false, false)]);
        let inst = InstanceFactory::default_instance(&schema);
        assert_eq!(inst.values["ratio"], InstanceValue::Float(0.0));
    }

    #[test]
    fn test_default_instance_list() {
        let schema = schema_with_props(&[("items", "list<string>", false, true)]);
        let inst = InstanceFactory::default_instance(&schema);
        assert_eq!(inst.values["items"], InstanceValue::List(Vec::new()));
    }

    // ── InstanceFactory::to_json ─────────────────────────────────────────────

    #[test]
    fn test_to_json_string() {
        let inst = AspectInstance {
            schema_name: "S".to_string(),
            values: vals(&[("x", InstanceValue::String("hello".to_string()))]),
        };
        let json = InstanceFactory::to_json(&inst);
        assert!(json.contains("\"x\": \"hello\""));
    }

    #[test]
    fn test_to_json_integer() {
        let inst = AspectInstance {
            schema_name: "S".to_string(),
            values: vals(&[("n", InstanceValue::Integer(42))]),
        };
        let json = InstanceFactory::to_json(&inst);
        assert!(json.contains("\"n\": 42"));
    }

    #[test]
    fn test_to_json_boolean() {
        let inst = AspectInstance {
            schema_name: "S".to_string(),
            values: vals(&[("b", InstanceValue::Boolean(true))]),
        };
        let json = InstanceFactory::to_json(&inst);
        assert!(json.contains("\"b\": true"));
    }

    #[test]
    fn test_to_json_null() {
        let inst = AspectInstance {
            schema_name: "S".to_string(),
            values: vals(&[("n", InstanceValue::Null)]),
        };
        let json = InstanceFactory::to_json(&inst);
        assert!(json.contains("\"n\": null"));
    }

    // ── InstanceFactory::from_json ───────────────────────────────────────────

    #[test]
    fn test_from_json_roundtrip_string() {
        let schema = schema_with_props(&[("name", "string", false, false)]);
        let json = "{\n  \"name\": \"Alice\"\n}";
        let inst = InstanceFactory::from_json(&schema, json).expect("parse ok");
        assert_eq!(inst.values["name"], InstanceValue::String("Alice".to_string()));
    }

    #[test]
    fn test_from_json_roundtrip_integer() {
        let schema = schema_with_props(&[("age", "integer", false, false)]);
        let json = "{\n  \"age\": 30\n}";
        let inst = InstanceFactory::from_json(&schema, json).expect("parse ok");
        assert_eq!(inst.values["age"], InstanceValue::Integer(30));
    }

    #[test]
    fn test_from_json_roundtrip_boolean() {
        let schema = schema_with_props(&[("active", "boolean", false, false)]);
        let json = "{\n  \"active\": true\n}";
        let inst = InstanceFactory::from_json(&schema, json).expect("parse ok");
        assert_eq!(inst.values["active"], InstanceValue::Boolean(true));
    }

    #[test]
    fn test_from_json_invalid_object() {
        let schema = schema_with_props(&[]);
        let result = InstanceFactory::from_json(&schema, "not json");
        assert!(result.is_err());
    }

    // ── is_compatible_type ───────────────────────────────────────────────────

    #[test]
    fn test_compatible_string() {
        assert!(InstanceFactory::is_compatible_type(&InstanceValue::String("x".to_string()), "string"));
    }

    #[test]
    fn test_compatible_integer() {
        assert!(InstanceFactory::is_compatible_type(&InstanceValue::Integer(1), "integer"));
    }

    #[test]
    fn test_compatible_float() {
        assert!(InstanceFactory::is_compatible_type(&InstanceValue::Float(1.0), "float"));
    }

    #[test]
    fn test_compatible_boolean() {
        assert!(InstanceFactory::is_compatible_type(&InstanceValue::Boolean(false), "boolean"));
    }

    #[test]
    fn test_compatible_null_any() {
        assert!(InstanceFactory::is_compatible_type(&InstanceValue::Null, "string"));
        assert!(InstanceFactory::is_compatible_type(&InstanceValue::Null, "integer"));
    }

    #[test]
    fn test_incompatible_type() {
        assert!(!InstanceFactory::is_compatible_type(&InstanceValue::String("x".to_string()), "integer"));
    }

    #[test]
    fn test_list_compatible() {
        assert!(InstanceFactory::is_compatible_type(&InstanceValue::List(vec![]), "list<string>"));
    }

    // ── FactoryError display ─────────────────────────────────────────────────

    #[test]
    fn test_factory_error_display_missing() {
        let e = FactoryError::MissingProperty("prop1".to_string());
        assert!(e.to_string().contains("prop1"));
    }

    #[test]
    fn test_factory_error_display_mismatch() {
        let e = FactoryError::TypeMismatch {
            property: "x".to_string(),
            expected: "string".to_string(),
            got: "integer".to_string(),
        };
        assert!(e.to_string().contains("string"));
        assert!(e.to_string().contains("integer"));
    }

    #[test]
    fn test_create_all_types() {
        let schema = schema_with_props(&[
            ("s", "string", false, false),
            ("i", "integer", false, false),
            ("f", "float", false, false),
            ("b", "boolean", false, false),
        ]);
        let v = vals(&[
            ("s", InstanceValue::String("hi".to_string())),
            ("i", InstanceValue::Integer(1)),
            ("f", InstanceValue::Float(2.71)),
            ("b", InstanceValue::Boolean(true)),
        ]);
        assert!(InstanceFactory::create(&schema, v).is_ok());
    }
}
