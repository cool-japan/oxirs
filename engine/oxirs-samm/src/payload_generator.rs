/// SAMM payload generation from aspect models.
///
/// Generates JSON payloads from SAMM aspect definitions, handling nested
/// entities, collections, characteristic-based serialization (Measurement,
/// Quantifiable, Enumeration, etc.), and optional property semantics.
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Aspect model types (simplified for payload generation)
// ---------------------------------------------------------------------------

/// Data type discriminator for property values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    /// A string value.
    String,
    /// An integer value.
    Integer,
    /// A floating-point (decimal) value.
    Decimal,
    /// A boolean value.
    Boolean,
    /// A date (ISO-8601 date string).
    Date,
    /// A dateTime (ISO-8601 dateTime string).
    DateTime,
    /// A reference to a named entity.
    EntityRef(String),
}

/// Kind of characteristic that describes a property's semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharacteristicKind {
    /// A plain single-valued property.
    Trait,
    /// A measurement with a unit.
    Measurement(String),
    /// A quantifiable value with a unit.
    Quantifiable(String),
    /// A duration value with a unit.
    Duration(String),
    /// An enumeration with allowed values.
    Enumeration(Vec<String>),
    /// A collection (ordered list).
    Collection,
    /// A set (unordered, no duplicates).
    Set,
    /// A sorted set.
    SortedSet,
    /// A time series collection.
    TimeSeries,
    /// A single entity reference.
    SingleEntity,
    /// A state (enum with a default).
    State(Vec<String>, Option<String>),
    /// A code (short alphanumeric string).
    Code,
    /// A structured value (complex multi-field).
    StructuredValue,
}

/// A property definition inside an aspect or entity.
#[derive(Debug, Clone)]
pub struct PropertyDef {
    /// Property name.
    pub name: String,
    /// Data type.
    pub data_type: DataType,
    /// Characteristic kind.
    pub characteristic: CharacteristicKind,
    /// Whether the property is optional.
    pub optional: bool,
    /// Whether the property must not be null (even if optional).
    pub not_in_payload: bool,
    /// Example value (used for documentation payloads).
    pub example_value: Option<String>,
    /// Description.
    pub description: Option<String>,
}

/// An entity definition (nested complex type).
#[derive(Debug, Clone)]
pub struct EntityDef {
    /// Entity name.
    pub name: String,
    /// Properties of this entity.
    pub properties: Vec<PropertyDef>,
    /// Optional extends (parent entity name).
    pub extends: Option<String>,
}

/// An aspect model definition.
#[derive(Debug, Clone)]
pub struct AspectDef {
    /// Aspect name.
    pub name: String,
    /// Version.
    pub version: String,
    /// Top-level properties.
    pub properties: Vec<PropertyDef>,
    /// Entity definitions referenced by properties.
    pub entities: HashMap<String, EntityDef>,
}

// ---------------------------------------------------------------------------
// Payload value (intermediate representation)
// ---------------------------------------------------------------------------

/// A JSON-like value tree used as the intermediate payload representation.
#[derive(Debug, Clone, PartialEq)]
pub enum PayloadValue {
    /// Null.
    Null,
    /// A string.
    String(String),
    /// An integer.
    Integer(i64),
    /// A floating-point number.
    Decimal(f64),
    /// A boolean.
    Bool(bool),
    /// An ordered array.
    Array(Vec<PayloadValue>),
    /// A key-value object.
    Object(Vec<(String, PayloadValue)>),
}

impl PayloadValue {
    /// Serialize to a JSON string.
    pub fn to_json(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::String(s) => format!("\"{}\"", escape_json(s)),
            Self::Integer(n) => n.to_string(),
            Self::Decimal(n) => {
                if n.fract() == 0.0 {
                    format!("{n:.1}")
                } else {
                    format!("{n}")
                }
            }
            Self::Bool(b) => b.to_string(),
            Self::Array(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.to_json()).collect();
                format!("[{}]", inner.join(","))
            }
            Self::Object(fields) => {
                let inner: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", escape_json(k), v.to_json()))
                    .collect();
                format!("{{{}}}", inner.join(","))
            }
        }
    }

    /// Return true if this is a Null.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Return true if this is an Object.
    pub fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    /// Return true if this is an Array.
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// Generator errors
// ---------------------------------------------------------------------------

/// Errors from payload generation.
#[derive(Debug)]
pub enum PayloadError {
    /// Referenced entity is not defined.
    UnknownEntity(String),
    /// Infinite recursion detected in entity expansion.
    RecursionLimit(String),
    /// Enumeration has no allowed values.
    EmptyEnumeration(String),
    /// Validation failed against the aspect model.
    ValidationFailed(String),
    /// General error.
    General(String),
}

impl std::fmt::Display for PayloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownEntity(name) => write!(f, "Unknown entity: {name}"),
            Self::RecursionLimit(name) => write!(f, "Recursion limit for entity: {name}"),
            Self::EmptyEnumeration(name) => write!(f, "Empty enumeration: {name}"),
            Self::ValidationFailed(msg) => write!(f, "Validation failed: {msg}"),
            Self::General(msg) => write!(f, "Payload error: {msg}"),
        }
    }
}

impl std::error::Error for PayloadError {}

// ---------------------------------------------------------------------------
// PayloadGenerator
// ---------------------------------------------------------------------------

/// Configuration for payload generation.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Maximum recursion depth for nested entities.
    pub max_depth: usize,
    /// Whether to include optional properties.
    pub include_optional: bool,
    /// Whether to include null values for optional properties.
    pub include_nulls: bool,
    /// Whether to generate example values.
    pub example_mode: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            include_optional: true,
            include_nulls: false,
            example_mode: false,
        }
    }
}

/// Generates JSON payloads from SAMM aspect model definitions.
#[derive(Debug)]
pub struct PayloadGenerator {
    config: GeneratorConfig,
}

impl PayloadGenerator {
    /// Create a new generator with default configuration.
    pub fn new() -> Self {
        Self {
            config: GeneratorConfig::default(),
        }
    }

    /// Create a new generator with the given configuration.
    pub fn with_config(config: GeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate a payload for an aspect definition.
    pub fn generate(&self, aspect: &AspectDef) -> Result<PayloadValue, PayloadError> {
        let mut visited = std::collections::HashSet::new();
        self.generate_object(&aspect.properties, &aspect.entities, &mut visited, 0)
    }

    /// Generate an example payload (using example_value fields).
    pub fn generate_example(&self, aspect: &AspectDef) -> Result<PayloadValue, PayloadError> {
        let config = GeneratorConfig {
            example_mode: true,
            include_optional: true,
            ..self.config.clone()
        };
        let gen = PayloadGenerator::with_config(config);
        gen.generate(aspect)
    }

    /// Validate a payload against an aspect definition.
    pub fn validate(
        &self,
        payload: &PayloadValue,
        aspect: &AspectDef,
    ) -> Result<Vec<String>, PayloadError> {
        let mut errors = Vec::new();

        if let PayloadValue::Object(fields) = payload {
            let field_map: HashMap<&str, &PayloadValue> =
                fields.iter().map(|(k, v)| (k.as_str(), v)).collect();

            for prop in &aspect.properties {
                if prop.not_in_payload {
                    continue;
                }

                match field_map.get(prop.name.as_str()) {
                    None if !prop.optional => {
                        errors.push(format!("Missing required property: {}", prop.name));
                    }
                    Some(val) => {
                        if let Err(e) =
                            self.validate_value(val, &prop.data_type, &prop.characteristic)
                        {
                            errors.push(format!("Property '{}': {}", prop.name, e));
                        }
                    }
                    _ => {}
                }
            }
        } else {
            errors.push("Payload must be a JSON object".to_string());
        }

        Ok(errors)
    }

    // --- private ---

    fn generate_object(
        &self,
        properties: &[PropertyDef],
        entities: &HashMap<String, EntityDef>,
        visited: &mut std::collections::HashSet<String>,
        depth: usize,
    ) -> Result<PayloadValue, PayloadError> {
        if depth > self.config.max_depth {
            return Err(PayloadError::RecursionLimit(format!("depth {depth}")));
        }

        let mut fields = Vec::new();

        for prop in properties {
            if prop.not_in_payload {
                continue;
            }

            if prop.optional && !self.config.include_optional {
                continue;
            }

            let value = self.generate_property_value(prop, entities, visited, depth)?;

            if prop.optional && value.is_null() && !self.config.include_nulls {
                continue;
            }

            fields.push((prop.name.clone(), value));
        }

        Ok(PayloadValue::Object(fields))
    }

    fn generate_property_value(
        &self,
        prop: &PropertyDef,
        entities: &HashMap<String, EntityDef>,
        visited: &mut std::collections::HashSet<String>,
        depth: usize,
    ) -> Result<PayloadValue, PayloadError> {
        // Use example value if in example mode
        if self.config.example_mode {
            if let Some(ref example) = prop.example_value {
                return Ok(self.parse_example_value(example, &prop.data_type));
            }
        }

        match &prop.characteristic {
            CharacteristicKind::Enumeration(values) => {
                if values.is_empty() {
                    return Err(PayloadError::EmptyEnumeration(prop.name.clone()));
                }
                Ok(PayloadValue::String(values[0].clone()))
            }
            CharacteristicKind::State(values, default) => {
                let val = default
                    .as_ref()
                    .or_else(|| values.first())
                    .cloned()
                    .unwrap_or_default();
                Ok(PayloadValue::String(val))
            }
            CharacteristicKind::Collection
            | CharacteristicKind::Set
            | CharacteristicKind::SortedSet
            | CharacteristicKind::TimeSeries => {
                // Generate a single-element array
                let element =
                    self.generate_default_value(&prop.data_type, entities, visited, depth)?;
                Ok(PayloadValue::Array(vec![element]))
            }
            CharacteristicKind::Measurement(unit)
            | CharacteristicKind::Quantifiable(unit)
            | CharacteristicKind::Duration(unit) => {
                let num_value =
                    self.generate_default_value(&prop.data_type, entities, visited, depth)?;
                // For measurements, wrap in an object with value + unit
                Ok(PayloadValue::Object(vec![
                    ("value".to_string(), num_value),
                    ("unit".to_string(), PayloadValue::String(unit.clone())),
                ]))
            }
            _ => self.generate_default_value(&prop.data_type, entities, visited, depth),
        }
    }

    fn generate_default_value(
        &self,
        data_type: &DataType,
        entities: &HashMap<String, EntityDef>,
        visited: &mut std::collections::HashSet<String>,
        depth: usize,
    ) -> Result<PayloadValue, PayloadError> {
        match data_type {
            DataType::String => Ok(PayloadValue::String(String::new())),
            DataType::Integer => Ok(PayloadValue::Integer(0)),
            DataType::Decimal => Ok(PayloadValue::Decimal(0.0)),
            DataType::Boolean => Ok(PayloadValue::Bool(false)),
            DataType::Date => Ok(PayloadValue::String("2024-01-01".to_string())),
            DataType::DateTime => Ok(PayloadValue::String("2024-01-01T00:00:00Z".to_string())),
            DataType::EntityRef(entity_name) => {
                if visited.contains(entity_name) {
                    return Err(PayloadError::RecursionLimit(entity_name.clone()));
                }
                let entity = entities
                    .get(entity_name)
                    .ok_or_else(|| PayloadError::UnknownEntity(entity_name.clone()))?;
                visited.insert(entity_name.clone());
                let result =
                    self.generate_object(&entity.properties, entities, visited, depth + 1)?;
                visited.remove(entity_name);
                Ok(result)
            }
        }
    }

    fn parse_example_value(&self, example: &str, data_type: &DataType) -> PayloadValue {
        match data_type {
            DataType::Integer => example
                .parse::<i64>()
                .map(PayloadValue::Integer)
                .unwrap_or_else(|_| PayloadValue::String(example.to_string())),
            DataType::Decimal => example
                .parse::<f64>()
                .map(PayloadValue::Decimal)
                .unwrap_or_else(|_| PayloadValue::String(example.to_string())),
            DataType::Boolean => example
                .parse::<bool>()
                .map(PayloadValue::Bool)
                .unwrap_or_else(|_| PayloadValue::String(example.to_string())),
            _ => PayloadValue::String(example.to_string()),
        }
    }

    fn validate_value(
        &self,
        value: &PayloadValue,
        data_type: &DataType,
        characteristic: &CharacteristicKind,
    ) -> Result<(), String> {
        // Null is only valid for optional properties (caller handles that)
        if value.is_null() {
            return Ok(());
        }

        // Check characteristic-specific constraints
        match characteristic {
            CharacteristicKind::Enumeration(allowed) => {
                if let PayloadValue::String(s) = value {
                    if !allowed.contains(s) {
                        return Err(format!("Value '{s}' not in enumeration"));
                    }
                }
            }
            CharacteristicKind::Collection
            | CharacteristicKind::Set
            | CharacteristicKind::SortedSet
            | CharacteristicKind::TimeSeries
                if !value.is_array() =>
            {
                return Err("Expected array for collection".to_string());
            }
            _ => {}
        }

        // Check data type
        match data_type {
            DataType::Integer if !matches!(value, PayloadValue::Integer(_)) => {
                return Err("Expected integer".to_string());
            }
            DataType::Decimal
                if !matches!(value, PayloadValue::Decimal(_) | PayloadValue::Integer(_)) =>
            {
                return Err("Expected decimal".to_string());
            }
            DataType::Boolean if !matches!(value, PayloadValue::Bool(_)) => {
                return Err("Expected boolean".to_string());
            }
            _ => {}
        }

        Ok(())
    }
}

impl Default for PayloadGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_aspect() -> AspectDef {
        AspectDef {
            name: "Movement".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![
                PropertyDef {
                    name: "speed".to_string(),
                    data_type: DataType::Decimal,
                    characteristic: CharacteristicKind::Measurement(
                        "unit:metrePerSecond".to_string(),
                    ),
                    optional: false,
                    not_in_payload: false,
                    example_value: Some("42.5".to_string()),
                    description: Some("Speed of the object".to_string()),
                },
                PropertyDef {
                    name: "direction".to_string(),
                    data_type: DataType::String,
                    characteristic: CharacteristicKind::Enumeration(vec![
                        "N".to_string(),
                        "S".to_string(),
                        "E".to_string(),
                        "W".to_string(),
                    ]),
                    optional: false,
                    not_in_payload: false,
                    example_value: Some("N".to_string()),
                    description: None,
                },
                PropertyDef {
                    name: "isMoving".to_string(),
                    data_type: DataType::Boolean,
                    characteristic: CharacteristicKind::Trait,
                    optional: false,
                    not_in_payload: false,
                    example_value: Some("true".to_string()),
                    description: None,
                },
            ],
            entities: HashMap::new(),
        }
    }

    fn aspect_with_entity() -> AspectDef {
        let mut entities = HashMap::new();
        entities.insert(
            "Position".to_string(),
            EntityDef {
                name: "Position".to_string(),
                properties: vec![
                    PropertyDef {
                        name: "latitude".to_string(),
                        data_type: DataType::Decimal,
                        characteristic: CharacteristicKind::Trait,
                        optional: false,
                        not_in_payload: false,
                        example_value: Some("35.6762".to_string()),
                        description: None,
                    },
                    PropertyDef {
                        name: "longitude".to_string(),
                        data_type: DataType::Decimal,
                        characteristic: CharacteristicKind::Trait,
                        optional: false,
                        not_in_payload: false,
                        example_value: Some("139.6503".to_string()),
                        description: None,
                    },
                ],
                extends: None,
            },
        );

        AspectDef {
            name: "Location".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "position".to_string(),
                data_type: DataType::EntityRef("Position".to_string()),
                characteristic: CharacteristicKind::SingleEntity,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities,
        }
    }

    // -- PayloadValue tests --

    #[test]
    fn test_payload_value_null() {
        let v = PayloadValue::Null;
        assert!(v.is_null());
        assert!(!v.is_object());
        assert!(!v.is_array());
        assert_eq!(v.to_json(), "null");
    }

    #[test]
    fn test_payload_value_string() {
        let v = PayloadValue::String("hello".to_string());
        assert_eq!(v.to_json(), "\"hello\"");
    }

    #[test]
    fn test_payload_value_string_escape() {
        let v = PayloadValue::String("he\"llo".to_string());
        assert!(v.to_json().contains("\\\""));
    }

    #[test]
    fn test_payload_value_integer() {
        let v = PayloadValue::Integer(42);
        assert_eq!(v.to_json(), "42");
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_payload_value_decimal() {
        let v = PayloadValue::Decimal(3.14);
        assert_eq!(v.to_json(), "3.14");
    }

    #[test]
    fn test_payload_value_decimal_integer_form() {
        let v = PayloadValue::Decimal(10.0);
        assert_eq!(v.to_json(), "10.0");
    }

    #[test]
    fn test_payload_value_bool() {
        assert_eq!(PayloadValue::Bool(true).to_json(), "true");
        assert_eq!(PayloadValue::Bool(false).to_json(), "false");
    }

    #[test]
    fn test_payload_value_array() {
        let v = PayloadValue::Array(vec![PayloadValue::Integer(1), PayloadValue::Integer(2)]);
        assert!(v.is_array());
        assert_eq!(v.to_json(), "[1,2]");
    }

    #[test]
    fn test_payload_value_object() {
        let v = PayloadValue::Object(vec![(
            "name".to_string(),
            PayloadValue::String("test".to_string()),
        )]);
        assert!(v.is_object());
        assert_eq!(v.to_json(), "{\"name\":\"test\"}");
    }

    #[test]
    fn test_payload_value_nested_object() {
        let inner = PayloadValue::Object(vec![("x".to_string(), PayloadValue::Integer(1))]);
        let outer = PayloadValue::Object(vec![("inner".to_string(), inner)]);
        assert_eq!(outer.to_json(), "{\"inner\":{\"x\":1}}");
    }

    // -- Simple generation tests --

    #[test]
    fn test_generate_simple_aspect() {
        let gen = PayloadGenerator::new();
        let aspect = simple_aspect();
        let payload = gen.generate(&aspect).expect("should generate");
        assert!(payload.is_object());
    }

    #[test]
    fn test_generate_has_speed_field() {
        let gen = PayloadGenerator::new();
        let aspect = simple_aspect();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("\"speed\""));
    }

    #[test]
    fn test_generate_enumeration_uses_first_value() {
        let gen = PayloadGenerator::new();
        let aspect = simple_aspect();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("\"N\""));
    }

    #[test]
    fn test_generate_boolean_default() {
        let gen = PayloadGenerator::new();
        let aspect = simple_aspect();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("false"));
    }

    // -- Example mode --

    #[test]
    fn test_generate_example() {
        let gen = PayloadGenerator::new();
        let aspect = simple_aspect();
        let payload = gen
            .generate_example(&aspect)
            .expect("should generate example");
        let json = payload.to_json();
        assert!(json.contains("42.5"));
        assert!(json.contains("\"N\""));
        assert!(json.contains("true"));
    }

    // -- Nested entity --

    #[test]
    fn test_generate_nested_entity() {
        let gen = PayloadGenerator::new();
        let aspect = aspect_with_entity();
        let payload = gen.generate(&aspect).expect("should generate");
        assert!(payload.is_object());
        let json = payload.to_json();
        assert!(json.contains("\"position\""));
        assert!(json.contains("\"latitude\""));
        assert!(json.contains("\"longitude\""));
    }

    #[test]
    fn test_generate_example_nested_entity() {
        let gen = PayloadGenerator::new();
        let aspect = aspect_with_entity();
        let payload = gen.generate_example(&aspect).expect("example");
        let json = payload.to_json();
        assert!(json.contains("35.6762"));
        assert!(json.contains("139.6503"));
    }

    // -- Collection handling --

    #[test]
    fn test_generate_collection() {
        let aspect = AspectDef {
            name: "SensorData".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "values".to_string(),
                data_type: DataType::Decimal,
                characteristic: CharacteristicKind::Collection,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let gen = PayloadGenerator::new();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("["));
    }

    #[test]
    fn test_generate_set() {
        let aspect = AspectDef {
            name: "Tags".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "tags".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::Set,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let gen = PayloadGenerator::new();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("\"tags\""));
    }

    // -- Optional property handling --

    #[test]
    fn test_optional_included() {
        let aspect = AspectDef {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "optional_field".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::Trait,
                optional: true,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let config = GeneratorConfig {
            include_optional: true,
            ..Default::default()
        };
        let gen = PayloadGenerator::with_config(config);
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("\"optional_field\""));
    }

    #[test]
    fn test_optional_excluded() {
        let aspect = AspectDef {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "optional_field".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::Trait,
                optional: true,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let config = GeneratorConfig {
            include_optional: false,
            ..Default::default()
        };
        let gen = PayloadGenerator::with_config(config);
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(!json.contains("optional_field"));
    }

    #[test]
    fn test_not_in_payload_excluded() {
        let aspect = AspectDef {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "hidden_field".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::Trait,
                optional: false,
                not_in_payload: true,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let gen = PayloadGenerator::new();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(!json.contains("hidden_field"));
    }

    // -- Measurement characteristic --

    #[test]
    fn test_measurement_wraps_value_and_unit() {
        let gen = PayloadGenerator::new();
        let aspect = simple_aspect();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("\"unit\""));
        assert!(json.contains("metrePerSecond"));
    }

    // -- Validation --

    #[test]
    fn test_validate_valid_payload() {
        let aspect = AspectDef {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "count".to_string(),
                data_type: DataType::Integer,
                characteristic: CharacteristicKind::Trait,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let payload = PayloadValue::Object(vec![("count".to_string(), PayloadValue::Integer(5))]);
        let gen = PayloadGenerator::new();
        let errors = gen.validate(&payload, &aspect).expect("should validate");
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_missing_required() {
        let aspect = AspectDef {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "name".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::Trait,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let payload = PayloadValue::Object(vec![]);
        let gen = PayloadGenerator::new();
        let errors = gen.validate(&payload, &aspect).expect("should validate");
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Missing required"));
    }

    #[test]
    fn test_validate_wrong_type() {
        let aspect = AspectDef {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "count".to_string(),
                data_type: DataType::Integer,
                characteristic: CharacteristicKind::Trait,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let payload = PayloadValue::Object(vec![(
            "count".to_string(),
            PayloadValue::String("not_a_number".to_string()),
        )]);
        let gen = PayloadGenerator::new();
        let errors = gen.validate(&payload, &aspect).expect("should validate");
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Expected integer"));
    }

    #[test]
    fn test_validate_enum_invalid() {
        let aspect = AspectDef {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "color".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::Enumeration(vec![
                    "red".to_string(),
                    "blue".to_string(),
                ]),
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let payload = PayloadValue::Object(vec![(
            "color".to_string(),
            PayloadValue::String("green".to_string()),
        )]);
        let gen = PayloadGenerator::new();
        let errors = gen.validate(&payload, &aspect).expect("should validate");
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("not in enumeration"));
    }

    #[test]
    fn test_validate_not_object() {
        let aspect = simple_aspect();
        let payload = PayloadValue::Array(vec![]);
        let gen = PayloadGenerator::new();
        let errors = gen.validate(&payload, &aspect).expect("should validate");
        assert!(!errors.is_empty());
    }

    // -- Error cases --

    #[test]
    fn test_unknown_entity() {
        let aspect = AspectDef {
            name: "Bad".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "item".to_string(),
                data_type: DataType::EntityRef("Nonexistent".to_string()),
                characteristic: CharacteristicKind::SingleEntity,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let gen = PayloadGenerator::new();
        let result = gen.generate(&aspect);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_enumeration_error() {
        let aspect = AspectDef {
            name: "Bad".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "status".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::Enumeration(vec![]),
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let gen = PayloadGenerator::new();
        let result = gen.generate(&aspect);
        assert!(result.is_err());
    }

    #[test]
    fn test_recursion_limit() {
        let mut entities = HashMap::new();
        entities.insert(
            "Node".to_string(),
            EntityDef {
                name: "Node".to_string(),
                properties: vec![PropertyDef {
                    name: "child".to_string(),
                    data_type: DataType::EntityRef("Node".to_string()),
                    characteristic: CharacteristicKind::SingleEntity,
                    optional: false,
                    not_in_payload: false,
                    example_value: None,
                    description: None,
                }],
                extends: None,
            },
        );
        let aspect = AspectDef {
            name: "Tree".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "root".to_string(),
                data_type: DataType::EntityRef("Node".to_string()),
                characteristic: CharacteristicKind::SingleEntity,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities,
        };
        let gen = PayloadGenerator::new();
        let result = gen.generate(&aspect);
        assert!(result.is_err());
    }

    // -- Error display --

    #[test]
    fn test_error_display_unknown_entity() {
        let e = PayloadError::UnknownEntity("Foo".to_string());
        assert!(e.to_string().contains("Foo"));
    }

    #[test]
    fn test_error_display_recursion() {
        let e = PayloadError::RecursionLimit("Bar".to_string());
        assert!(e.to_string().contains("Bar"));
    }

    #[test]
    fn test_error_display_empty_enum() {
        let e = PayloadError::EmptyEnumeration("Status".to_string());
        assert!(e.to_string().contains("Status"));
    }

    #[test]
    fn test_error_display_validation() {
        let e = PayloadError::ValidationFailed("bad".to_string());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn test_error_display_general() {
        let e = PayloadError::General("misc".to_string());
        assert!(e.to_string().contains("misc"));
    }

    // -- Config defaults --

    #[test]
    fn test_default_config() {
        let config = GeneratorConfig::default();
        assert_eq!(config.max_depth, 10);
        assert!(config.include_optional);
        assert!(!config.include_nulls);
        assert!(!config.example_mode);
    }

    #[test]
    fn test_generator_default() {
        let gen = PayloadGenerator::default();
        let aspect = simple_aspect();
        let payload = gen.generate(&aspect);
        assert!(payload.is_ok());
    }

    // -- State characteristic --

    #[test]
    fn test_state_with_default() {
        let aspect = AspectDef {
            name: "Machine".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "state".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::State(
                    vec![
                        "running".to_string(),
                        "stopped".to_string(),
                        "error".to_string(),
                    ],
                    Some("stopped".to_string()),
                ),
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let gen = PayloadGenerator::new();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("\"stopped\""));
    }

    #[test]
    fn test_state_without_default() {
        let aspect = AspectDef {
            name: "Machine".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "state".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::State(
                    vec!["on".to_string(), "off".to_string()],
                    None,
                ),
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let gen = PayloadGenerator::new();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("\"on\""));
    }

    // -- Date/DateTime default values --

    #[test]
    fn test_date_default() {
        let aspect = AspectDef {
            name: "Event".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "date".to_string(),
                data_type: DataType::Date,
                characteristic: CharacteristicKind::Trait,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let gen = PayloadGenerator::new();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("2024-01-01"));
    }

    #[test]
    fn test_datetime_default() {
        let aspect = AspectDef {
            name: "Event".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "timestamp".to_string(),
                data_type: DataType::DateTime,
                characteristic: CharacteristicKind::Trait,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let gen = PayloadGenerator::new();
        let payload = gen.generate(&aspect).expect("should generate");
        let json = payload.to_json();
        assert!(json.contains("2024-01-01T00:00:00Z"));
    }

    // -- escape_json --

    #[test]
    fn test_escape_json_backslash() {
        assert!(escape_json("a\\b").contains("\\\\"));
    }

    #[test]
    fn test_escape_json_newline() {
        assert!(escape_json("a\nb").contains("\\n"));
    }

    // -- Validate optional missing is OK --

    #[test]
    fn test_validate_optional_missing_ok() {
        let aspect = AspectDef {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "opt".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::Trait,
                optional: true,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let payload = PayloadValue::Object(vec![]);
        let gen = PayloadGenerator::new();
        let errors = gen.validate(&payload, &aspect).expect("should validate");
        assert!(errors.is_empty());
    }

    // -- Validate collection type --

    #[test]
    fn test_validate_collection_wrong_type() {
        let aspect = AspectDef {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            properties: vec![PropertyDef {
                name: "items".to_string(),
                data_type: DataType::String,
                characteristic: CharacteristicKind::Collection,
                optional: false,
                not_in_payload: false,
                example_value: None,
                description: None,
            }],
            entities: HashMap::new(),
        };
        let payload = PayloadValue::Object(vec![(
            "items".to_string(),
            PayloadValue::String("not_array".to_string()),
        )]);
        let gen = PayloadGenerator::new();
        let errors = gen.validate(&payload, &aspect).expect("should validate");
        assert!(!errors.is_empty());
    }
}
