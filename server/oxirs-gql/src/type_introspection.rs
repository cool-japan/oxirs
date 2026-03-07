//! GraphQL schema introspection engine
//!
//! Implements the GraphQL introspection system (__schema, __type) that allows
//! clients to query the type system of a GraphQL server. This is a standalone,
//! JSON-producing engine that does not depend on the existing schema runtime.

use serde_json::{json, Value as JsonValue};

// ─── TypeKind ─────────────────────────────────────────────────────────────────

/// The GraphQL category of a type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    Scalar,
    Object,
    Interface,
    Union,
    Enum,
    InputObject,
    List,
    NonNull,
}

impl TypeKind {
    fn as_str(&self) -> &'static str {
        match self {
            TypeKind::Scalar => "SCALAR",
            TypeKind::Object => "OBJECT",
            TypeKind::Interface => "INTERFACE",
            TypeKind::Union => "UNION",
            TypeKind::Enum => "ENUM",
            TypeKind::InputObject => "INPUT_OBJECT",
            TypeKind::List => "LIST",
            TypeKind::NonNull => "NON_NULL",
        }
    }
}

// ─── TypeRef ──────────────────────────────────────────────────────────────────

/// A reference to a type (potentially wrapped in List / NonNull)
#[derive(Debug, Clone)]
pub struct TypeRef {
    /// The kind of the outermost wrapper
    pub kind: TypeKind,
    /// Name of a named type; `None` for List / NonNull
    pub name: Option<String>,
    /// The wrapped type for List / NonNull
    pub of_type: Option<Box<TypeRef>>,
}

impl TypeRef {
    /// Named type reference (Scalar, Object, Enum, etc.)
    pub fn named(kind: TypeKind, name: impl Into<String>) -> Self {
        TypeRef {
            kind,
            name: Some(name.into()),
            of_type: None,
        }
    }

    /// Wrap this reference as a NonNull type
    pub fn non_null(inner: TypeRef) -> Self {
        TypeRef {
            kind: TypeKind::NonNull,
            name: None,
            of_type: Some(Box::new(inner)),
        }
    }

    /// Wrap this reference as a List type
    pub fn list(inner: TypeRef) -> Self {
        TypeRef {
            kind: TypeKind::List,
            name: None,
            of_type: Some(Box::new(inner)),
        }
    }

    fn to_json(&self) -> JsonValue {
        let of_type = self
            .of_type
            .as_ref()
            .map(|t| t.to_json())
            .unwrap_or(JsonValue::Null);
        json!({
            "kind": self.kind.as_str(),
            "name": self.name,
            "ofType": of_type,
        })
    }
}

// ─── InputValue ───────────────────────────────────────────────────────────────

/// An input argument for a field or directive
#[derive(Debug, Clone)]
pub struct InputValue {
    /// Argument name
    pub name: String,
    /// Type of this argument
    pub type_ref: TypeRef,
    /// Optional default value (as a string)
    pub default_value: Option<String>,
}

impl InputValue {
    /// Create a new input value
    pub fn new(name: impl Into<String>, type_ref: TypeRef) -> Self {
        InputValue {
            name: name.into(),
            type_ref,
            default_value: None,
        }
    }

    fn to_json(&self) -> JsonValue {
        json!({
            "name": self.name,
            "type": self.type_ref.to_json(),
            "defaultValue": self.default_value,
        })
    }
}

// ─── IntrospectionField ───────────────────────────────────────────────────────

/// A field on an Object or Interface type
#[derive(Debug, Clone)]
pub struct IntrospectionField {
    /// Field name
    pub name: String,
    /// Return type reference
    pub type_ref: TypeRef,
    /// Arguments accepted by this field
    pub args: Vec<InputValue>,
    /// Optional description
    pub description: Option<String>,
    /// Whether this field is deprecated
    pub is_deprecated: bool,
    /// Deprecation reason (if `is_deprecated` is true)
    pub deprecation_reason: Option<String>,
}

impl IntrospectionField {
    /// Create a field with no arguments
    pub fn new(name: impl Into<String>, type_ref: TypeRef) -> Self {
        IntrospectionField {
            name: name.into(),
            type_ref,
            args: Vec::new(),
            description: None,
            is_deprecated: false,
            deprecation_reason: None,
        }
    }

    /// Mark this field as deprecated with a reason
    pub fn deprecated(mut self, reason: impl Into<String>) -> Self {
        self.is_deprecated = true;
        self.deprecation_reason = Some(reason.into());
        self
    }

    fn to_json(&self) -> JsonValue {
        json!({
            "name": self.name,
            "type": self.type_ref.to_json(),
            "args": self.args.iter().map(InputValue::to_json).collect::<Vec<_>>(),
            "description": self.description,
            "isDeprecated": self.is_deprecated,
            "deprecationReason": self.deprecation_reason,
        })
    }
}

// ─── IntrospectionType ────────────────────────────────────────────────────────

/// A type in the GraphQL schema
#[derive(Debug, Clone)]
pub struct IntrospectionType {
    /// Category of this type
    pub kind: TypeKind,
    /// Name (None for List / NonNull)
    pub name: Option<String>,
    /// Optional description
    pub description: Option<String>,
    /// Fields (for Object, Interface)
    pub fields: Vec<IntrospectionField>,
    /// Possible values (for Enum)
    pub enum_values: Vec<String>,
    /// Possible concrete types (for Interface, Union)
    pub possible_types: Vec<String>,
    /// Input fields (for InputObject)
    pub input_fields: Vec<InputValue>,
}

impl IntrospectionType {
    /// Create a Scalar type
    pub fn scalar(name: impl Into<String>) -> Self {
        IntrospectionType {
            kind: TypeKind::Scalar,
            name: Some(name.into()),
            description: None,
            fields: Vec::new(),
            enum_values: Vec::new(),
            possible_types: Vec::new(),
            input_fields: Vec::new(),
        }
    }

    /// Create an Object type
    pub fn object(name: impl Into<String>, fields: Vec<IntrospectionField>) -> Self {
        IntrospectionType {
            kind: TypeKind::Object,
            name: Some(name.into()),
            description: None,
            fields,
            enum_values: Vec::new(),
            possible_types: Vec::new(),
            input_fields: Vec::new(),
        }
    }

    /// Create an Enum type
    pub fn enum_type(name: impl Into<String>, values: Vec<String>) -> Self {
        IntrospectionType {
            kind: TypeKind::Enum,
            name: Some(name.into()),
            description: None,
            fields: Vec::new(),
            enum_values: values,
            possible_types: Vec::new(),
            input_fields: Vec::new(),
        }
    }

    /// Create an InputObject type
    pub fn input_object(name: impl Into<String>, input_fields: Vec<InputValue>) -> Self {
        IntrospectionType {
            kind: TypeKind::InputObject,
            name: Some(name.into()),
            description: None,
            fields: Vec::new(),
            enum_values: Vec::new(),
            possible_types: Vec::new(),
            input_fields,
        }
    }

    fn to_json(&self) -> JsonValue {
        json!({
            "kind": self.kind.as_str(),
            "name": self.name,
            "description": self.description,
            "fields": self.fields.iter().map(IntrospectionField::to_json).collect::<Vec<_>>(),
            "enumValues": if self.enum_values.is_empty() { JsonValue::Null } else {
                json!(self.enum_values)
            },
            "possibleTypes": if self.possible_types.is_empty() { JsonValue::Null } else {
                json!(self.possible_types)
            },
            "inputFields": if self.input_fields.is_empty() { JsonValue::Null } else {
                json!(self.input_fields.iter().map(InputValue::to_json).collect::<Vec<_>>())
            },
        })
    }
}

// ─── Schema ───────────────────────────────────────────────────────────────────

/// Top-level schema descriptor
#[derive(Debug, Clone)]
pub struct IntrospectionSchema {
    /// Name of the root query type
    pub query_type: String,
    /// Name of the root mutation type (if any)
    pub mutation_type: Option<String>,
    /// Name of the root subscription type (if any)
    pub subscription_type: Option<String>,
    /// All named types in this schema
    pub types: Vec<IntrospectionType>,
}

impl IntrospectionSchema {
    /// Create a minimal schema with only a query type
    pub fn new(query_type: impl Into<String>) -> Self {
        IntrospectionSchema {
            query_type: query_type.into(),
            mutation_type: None,
            subscription_type: None,
            types: Vec::new(),
        }
    }

    /// Add a type to the schema
    pub fn add_type(mut self, t: IntrospectionType) -> Self {
        self.types.push(t);
        self
    }
}

// ─── IntrospectionEngine ──────────────────────────────────────────────────────

/// Engine that executes GraphQL __schema / __type introspection queries
pub struct IntrospectionEngine {
    schema: IntrospectionSchema,
}

impl IntrospectionEngine {
    /// Construct the engine with the given schema
    pub fn new(schema: IntrospectionSchema) -> Self {
        IntrospectionEngine { schema }
    }

    /// Produce the JSON response for a `__schema` introspection query
    pub fn schema_introspection(&self) -> String {
        let mut all_types: Vec<JsonValue> = self
            .schema
            .types
            .iter()
            .map(IntrospectionType::to_json)
            .collect();
        // Append built-in scalars
        for builtin in Self::built_in_scalars() {
            all_types.push(builtin.to_json());
        }
        let response = json!({
            "data": {
                "__schema": {
                    "queryType": { "name": self.schema.query_type },
                    "mutationType": self.schema.mutation_type.as_ref().map(|n| json!({ "name": n })),
                    "subscriptionType": self.schema.subscription_type.as_ref().map(|n| json!({ "name": n })),
                    "types": all_types,
                }
            }
        });
        response.to_string()
    }

    /// Produce the JSON response for a `__type(name: "…")` introspection query.
    /// Returns `None` when the type is not found.
    pub fn type_introspection(&self, type_name: &str) -> Option<String> {
        // Check user-defined types first
        let found = self.get_type(type_name).map(|t| t.to_json()).or_else(|| {
            // Fall back to built-in scalars
            Self::built_in_scalars()
                .into_iter()
                .find(|t| t.name.as_deref() == Some(type_name))
                .map(|t| t.to_json())
        });
        found.map(|type_json| json!({ "data": { "__type": type_json } }).to_string())
    }

    /// Return the field names for a named Object / Interface type
    pub fn field_names(&self, type_name: &str) -> Vec<String> {
        self.get_type(type_name)
            .map(|t| t.fields.iter().map(|f| f.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Look up a type by name; does NOT include built-in scalars
    pub fn get_type(&self, name: &str) -> Option<&IntrospectionType> {
        self.schema
            .types
            .iter()
            .find(|t| t.name.as_deref() == Some(name))
    }

    /// The five GraphQL built-in scalar types
    pub fn built_in_scalars() -> Vec<IntrospectionType> {
        vec![
            IntrospectionType::scalar("String"),
            IntrospectionType::scalar("Int"),
            IntrospectionType::scalar("Float"),
            IntrospectionType::scalar("Boolean"),
            IntrospectionType::scalar("ID"),
        ]
    }

    /// Total number of types (user-defined + built-ins)
    pub fn type_count(&self) -> usize {
        self.schema.types.len() + Self::built_in_scalars().len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value as JsonValue;

    fn simple_engine() -> IntrospectionEngine {
        let query_type = IntrospectionType::object(
            "Query",
            vec![
                IntrospectionField::new("hero", TypeRef::named(TypeKind::Object, "Character")),
                IntrospectionField::new(
                    "search",
                    TypeRef::non_null(TypeRef::list(TypeRef::non_null(TypeRef::named(
                        TypeKind::Object,
                        "SearchResult",
                    )))),
                )
                .deprecated("Use heroSearch instead"),
            ],
        );
        let char_type = IntrospectionType::object(
            "Character",
            vec![
                IntrospectionField::new(
                    "id",
                    TypeRef::non_null(TypeRef::named(TypeKind::Scalar, "ID")),
                ),
                IntrospectionField::new("name", TypeRef::named(TypeKind::Scalar, "String")),
            ],
        );
        let episode_enum = IntrospectionType::enum_type(
            "Episode",
            vec![
                "NEWHOPE".to_string(),
                "EMPIRE".to_string(),
                "JEDI".to_string(),
            ],
        );
        let schema = IntrospectionSchema::new("Query")
            .add_type(query_type)
            .add_type(char_type)
            .add_type(episode_enum);
        IntrospectionEngine::new(schema)
    }

    // ── schema_introspection ─────────────────────────────────────────────────

    #[test]
    fn test_schema_introspection_returns_json_string() {
        let engine = simple_engine();
        let result = engine.schema_introspection();
        assert!(!result.is_empty());
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        assert!(parsed.is_object());
    }

    #[test]
    fn test_schema_introspection_has_data_key() {
        let engine = simple_engine();
        let result = engine.schema_introspection();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        assert!(parsed["data"].is_object());
    }

    #[test]
    fn test_schema_introspection_has_schema_key() {
        let engine = simple_engine();
        let result = engine.schema_introspection();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        assert!(parsed["data"]["__schema"].is_object());
    }

    #[test]
    fn test_schema_introspection_query_type_name() {
        let engine = simple_engine();
        let result = engine.schema_introspection();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        let qtype = &parsed["data"]["__schema"]["queryType"]["name"];
        assert_eq!(qtype, "Query");
    }

    #[test]
    fn test_schema_introspection_types_array() {
        let engine = simple_engine();
        let result = engine.schema_introspection();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        let types = &parsed["data"]["__schema"]["types"];
        assert!(types.is_array());
        assert!(!types.as_array().unwrap().is_empty());
    }

    #[test]
    fn test_schema_introspection_includes_builtin_scalars() {
        let engine = simple_engine();
        let result = engine.schema_introspection();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        let types = parsed["data"]["__schema"]["types"].as_array().unwrap();
        let names: Vec<&str> = types.iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"String"));
        assert!(names.contains(&"Int"));
        assert!(names.contains(&"Boolean"));
        assert!(names.contains(&"Float"));
        assert!(names.contains(&"ID"));
    }

    #[test]
    fn test_schema_introspection_no_mutation_type_by_default() {
        let engine = simple_engine();
        let result = engine.schema_introspection();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        let mut_type = &parsed["data"]["__schema"]["mutationType"];
        assert!(mut_type.is_null());
    }

    #[test]
    fn test_schema_introspection_mutation_type_included_when_set() {
        let mut_type = IntrospectionType::object("Mutation", vec![]);
        let schema = IntrospectionSchema::new("Query")
            .add_type(IntrospectionType::object("Query", vec![]))
            .add_type(mut_type);
        let mut schema_desc = schema.clone();
        schema_desc.mutation_type = Some("Mutation".to_string());
        let engine = IntrospectionEngine::new(schema_desc);
        let result = engine.schema_introspection();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        assert_eq!(
            parsed["data"]["__schema"]["mutationType"]["name"],
            "Mutation"
        );
    }

    // ── type_introspection ───────────────────────────────────────────────────

    #[test]
    fn test_type_introspection_object_found() {
        let engine = simple_engine();
        let result = engine.type_introspection("Query");
        assert!(result.is_some());
    }

    #[test]
    fn test_type_introspection_scalar_found() {
        let engine = simple_engine();
        let result = engine.type_introspection("String");
        assert!(result.is_some());
    }

    #[test]
    fn test_type_introspection_enum_found() {
        let engine = simple_engine();
        let result = engine.type_introspection("Episode");
        assert!(result.is_some());
        let parsed: JsonValue = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed["data"]["__type"]["kind"], "ENUM");
    }

    #[test]
    fn test_type_introspection_not_found_returns_none() {
        let engine = simple_engine();
        let result = engine.type_introspection("NonexistentType");
        assert!(result.is_none());
    }

    #[test]
    fn test_type_introspection_object_has_fields() {
        let engine = simple_engine();
        let result = engine.type_introspection("Character").unwrap();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        let fields = parsed["data"]["__type"]["fields"].as_array().unwrap();
        assert_eq!(fields.len(), 2);
        let field_names: Vec<&str> = fields.iter().map(|f| f["name"].as_str().unwrap()).collect();
        assert!(field_names.contains(&"id"));
        assert!(field_names.contains(&"name"));
    }

    #[test]
    fn test_type_introspection_scalar_kind() {
        let engine = simple_engine();
        let result = engine.type_introspection("String").unwrap();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["data"]["__type"]["kind"], "SCALAR");
    }

    // ── field_names ──────────────────────────────────────────────────────────

    #[test]
    fn test_field_names_returns_names_for_object() {
        let engine = simple_engine();
        let names = engine.field_names("Character");
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"id".to_string()));
        assert!(names.contains(&"name".to_string()));
    }

    #[test]
    fn test_field_names_empty_for_unknown_type() {
        let engine = simple_engine();
        let names = engine.field_names("Unknown");
        assert!(names.is_empty());
    }

    #[test]
    fn test_field_names_empty_for_enum() {
        let engine = simple_engine();
        let names = engine.field_names("Episode");
        assert!(names.is_empty());
    }

    // ── built_in_scalars ─────────────────────────────────────────────────────

    #[test]
    fn test_built_in_scalars_count_is_five() {
        let scalars = IntrospectionEngine::built_in_scalars();
        assert_eq!(scalars.len(), 5);
    }

    #[test]
    fn test_built_in_scalars_names() {
        let scalars = IntrospectionEngine::built_in_scalars();
        let names: Vec<&str> = scalars.iter().filter_map(|s| s.name.as_deref()).collect();
        assert!(names.contains(&"String"));
        assert!(names.contains(&"Int"));
        assert!(names.contains(&"Float"));
        assert!(names.contains(&"Boolean"));
        assert!(names.contains(&"ID"));
    }

    #[test]
    fn test_built_in_scalars_are_scalar_kind() {
        for scalar in IntrospectionEngine::built_in_scalars() {
            assert_eq!(scalar.kind, TypeKind::Scalar);
        }
    }

    // ── get_type ─────────────────────────────────────────────────────────────

    #[test]
    fn test_get_type_found() {
        let engine = simple_engine();
        assert!(engine.get_type("Query").is_some());
    }

    #[test]
    fn test_get_type_not_found() {
        let engine = simple_engine();
        assert!(engine.get_type("Missing").is_none());
    }

    #[test]
    fn test_get_type_returns_correct_kind() {
        let engine = simple_engine();
        let t = engine.get_type("Episode").unwrap();
        assert_eq!(t.kind, TypeKind::Enum);
    }

    // ── type_count ───────────────────────────────────────────────────────────

    #[test]
    fn test_type_count_includes_builtins() {
        let engine = simple_engine();
        // simple_engine has 3 user types + 5 built-in scalars = 8
        assert_eq!(engine.type_count(), 8);
    }

    #[test]
    fn test_type_count_empty_schema_is_five() {
        let schema = IntrospectionSchema::new("Query");
        let engine = IntrospectionEngine::new(schema);
        // Only built-ins
        assert_eq!(engine.type_count(), 5);
    }

    // ── TypeRef wrapping ─────────────────────────────────────────────────────

    #[test]
    fn test_type_ref_non_null_kind() {
        let inner = TypeRef::named(TypeKind::Scalar, "String");
        let non_null = TypeRef::non_null(inner);
        assert_eq!(non_null.kind, TypeKind::NonNull);
        assert!(non_null.name.is_none());
    }

    #[test]
    fn test_type_ref_list_kind() {
        let inner = TypeRef::named(TypeKind::Scalar, "String");
        let list = TypeRef::list(inner);
        assert_eq!(list.kind, TypeKind::List);
        assert!(list.name.is_none());
    }

    #[test]
    fn test_type_ref_nested_non_null_list() {
        // [String!]!
        let inner = TypeRef::non_null(TypeRef::named(TypeKind::Scalar, "String"));
        let list = TypeRef::list(inner);
        let outer = TypeRef::non_null(list);
        assert_eq!(outer.kind, TypeKind::NonNull);
        let list_ref = outer.of_type.as_ref().unwrap();
        assert_eq!(list_ref.kind, TypeKind::List);
    }

    #[test]
    fn test_type_ref_to_json_named() {
        let tr = TypeRef::named(TypeKind::Scalar, "String");
        let j = tr.to_json();
        assert_eq!(j["kind"], "SCALAR");
        assert_eq!(j["name"], "String");
        assert!(j["ofType"].is_null());
    }

    #[test]
    fn test_type_ref_to_json_non_null_has_of_type() {
        let inner = TypeRef::named(TypeKind::Scalar, "Int");
        let nn = TypeRef::non_null(inner);
        let j = nn.to_json();
        assert_eq!(j["kind"], "NON_NULL");
        assert!(!j["ofType"].is_null());
        assert_eq!(j["ofType"]["name"], "Int");
    }

    // ── deprecated field ─────────────────────────────────────────────────────

    #[test]
    fn test_deprecated_field_flag() {
        let engine = simple_engine();
        // "search" field on Query was marked deprecated
        let t = engine.get_type("Query").unwrap();
        let search = t.fields.iter().find(|f| f.name == "search").unwrap();
        assert!(search.is_deprecated);
        assert!(search.deprecation_reason.is_some());
    }

    #[test]
    fn test_deprecated_field_in_json() {
        let engine = simple_engine();
        let result = engine.type_introspection("Query").unwrap();
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        let fields = parsed["data"]["__type"]["fields"].as_array().unwrap();
        let search = fields.iter().find(|f| f["name"] == "search").unwrap();
        assert_eq!(search["isDeprecated"], true);
    }

    // ── empty schema ─────────────────────────────────────────────────────────

    #[test]
    fn test_empty_schema_introspection_valid_json() {
        let schema = IntrospectionSchema::new("Query");
        let engine = IntrospectionEngine::new(schema);
        let result = engine.schema_introspection();
        let parsed: Result<JsonValue, _> = serde_json::from_str(&result);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_empty_schema_type_introspection_not_found() {
        let schema = IntrospectionSchema::new("Query");
        let engine = IntrospectionEngine::new(schema);
        assert!(engine.type_introspection("Query").is_none());
    }
}
