//! # GraphQL Federation v2 Subgraph Support
//!
//! Implements Apollo Federation v2 subgraph directives: `@key`, `@requires`,
//! `@provides`, `@external`, `@shareable`, and `@override`. These directives
//! enable OxiRS-GQL to participate as a subgraph in a federated GraphQL
//! supergraph.
//!
//! ## Federation v2 Directives
//!
//! - `@key(fields: "id")` - Marks a type as an entity resolvable by key
//! - `@requires(fields: "...")` - Specifies fields needed from other subgraphs
//! - `@provides(fields: "...")` - Specifies fields this subgraph provides
//! - `@external` - Marks a field as owned by another subgraph
//! - `@shareable` - Marks a field as resolvable by multiple subgraphs
//! - `@override(from: "...")` - Overrides a field from another subgraph
//!
//! ## Example
//!
//! ```text
//! type Product @key(fields: "id") {
//!   id: ID!
//!   name: String!
//!   price: Float @external
//!   weight: Float @provides(fields: "price")
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Federation directives
// ---------------------------------------------------------------------------

/// A federation directive applied to a type or field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederationDirective {
    /// `@key(fields: "...")` - Entity key definition
    Key { fields: String, resolvable: bool },
    /// `@requires(fields: "...")` - Fields required from other subgraphs
    Requires { fields: String },
    /// `@provides(fields: "...")` - Fields this subgraph provides
    Provides { fields: String },
    /// `@external` - Field owned by another subgraph
    External,
    /// `@shareable` - Field resolvable by multiple subgraphs
    Shareable,
    /// `@override(from: "...")` - Override field from another subgraph
    Override { from: String },
    /// `@inaccessible` - Field hidden from the supergraph
    Inaccessible,
    /// `@tag(name: "...")` - Arbitrary tagging
    Tag { name: String },
}

impl fmt::Display for FederationDirective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FederationDirective::Key { fields, resolvable } => {
                if *resolvable {
                    write!(f, "@key(fields: \"{fields}\")")
                } else {
                    write!(f, "@key(fields: \"{fields}\", resolvable: false)")
                }
            }
            FederationDirective::Requires { fields } => {
                write!(f, "@requires(fields: \"{fields}\")")
            }
            FederationDirective::Provides { fields } => {
                write!(f, "@provides(fields: \"{fields}\")")
            }
            FederationDirective::External => write!(f, "@external"),
            FederationDirective::Shareable => write!(f, "@shareable"),
            FederationDirective::Override { from } => write!(f, "@override(from: \"{from}\")"),
            FederationDirective::Inaccessible => write!(f, "@inaccessible"),
            FederationDirective::Tag { name } => write!(f, "@tag(name: \"{name}\")"),
        }
    }
}

// ---------------------------------------------------------------------------
// Entity key
// ---------------------------------------------------------------------------

/// Parsed entity key for federation resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityKey {
    /// Type name (e.g., "Product")
    pub type_name: String,
    /// Key field names
    pub fields: Vec<String>,
    /// Whether this key is resolvable by this subgraph
    pub resolvable: bool,
}

impl EntityKey {
    /// Create a new entity key.
    pub fn new(type_name: impl Into<String>, fields: Vec<String>) -> Self {
        Self {
            type_name: type_name.into(),
            fields,
            resolvable: true,
        }
    }

    /// Create a non-resolvable key (stub entity).
    pub fn stub(type_name: impl Into<String>, fields: Vec<String>) -> Self {
        Self {
            type_name: type_name.into(),
            fields,
            resolvable: false,
        }
    }

    /// Check if a set of field values satisfies this key.
    pub fn matches(&self, provided_fields: &HashMap<String, String>) -> bool {
        self.fields.iter().all(|f| provided_fields.contains_key(f))
    }

    /// Parse key fields from the directive string (e.g., "id name").
    pub fn parse_fields(fields_str: &str) -> Vec<String> {
        fields_str
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Federated field
// ---------------------------------------------------------------------------

/// A field with federation metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedField {
    /// Field name
    pub name: String,
    /// Field type (GraphQL type string)
    pub field_type: String,
    /// Whether this field is nullable
    pub nullable: bool,
    /// Federation directives on this field
    pub directives: Vec<FederationDirective>,
}

impl FederatedField {
    /// Create a new federated field.
    pub fn new(name: impl Into<String>, field_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_type: field_type.into(),
            nullable: true,
            directives: Vec::new(),
        }
    }

    /// Mark as non-nullable.
    pub fn non_null(mut self) -> Self {
        self.nullable = false;
        self
    }

    /// Add a directive.
    pub fn with_directive(mut self, directive: FederationDirective) -> Self {
        self.directives.push(directive);
        self
    }

    /// Check if this field is external.
    pub fn is_external(&self) -> bool {
        self.directives
            .iter()
            .any(|d| matches!(d, FederationDirective::External))
    }

    /// Check if this field is shareable.
    pub fn is_shareable(&self) -> bool {
        self.directives
            .iter()
            .any(|d| matches!(d, FederationDirective::Shareable))
    }

    /// Check if this field is inaccessible.
    pub fn is_inaccessible(&self) -> bool {
        self.directives
            .iter()
            .any(|d| matches!(d, FederationDirective::Inaccessible))
    }

    /// Get @requires fields, if any.
    pub fn requires_fields(&self) -> Option<Vec<String>> {
        self.directives.iter().find_map(|d| {
            if let FederationDirective::Requires { fields } = d {
                Some(EntityKey::parse_fields(fields))
            } else {
                None
            }
        })
    }

    /// Get @provides fields, if any.
    pub fn provides_fields(&self) -> Option<Vec<String>> {
        self.directives.iter().find_map(|d| {
            if let FederationDirective::Provides { fields } = d {
                Some(EntityKey::parse_fields(fields))
            } else {
                None
            }
        })
    }

    /// Render the field definition with directives.
    pub fn to_sdl(&self) -> String {
        let type_str = if self.nullable {
            self.field_type.clone()
        } else {
            format!("{}!", self.field_type)
        };

        let directives: Vec<String> = self.directives.iter().map(|d| d.to_string()).collect();

        if directives.is_empty() {
            format!("  {}: {}", self.name, type_str)
        } else {
            format!("  {}: {} {}", self.name, type_str, directives.join(" "))
        }
    }
}

// ---------------------------------------------------------------------------
// Federated type
// ---------------------------------------------------------------------------

/// A GraphQL type with federation metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedType {
    /// Type name
    pub name: String,
    /// Entity keys (from @key directives)
    pub keys: Vec<EntityKey>,
    /// Fields
    pub fields: Vec<FederatedField>,
    /// Type-level directives
    pub directives: Vec<FederationDirective>,
}

impl FederatedType {
    /// Create a new federated type.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            keys: Vec::new(),
            fields: Vec::new(),
            directives: Vec::new(),
        }
    }

    /// Add a @key directive.
    pub fn with_key(mut self, fields: impl Into<String>, resolvable: bool) -> Self {
        let fields_str: String = fields.into();
        let key = if resolvable {
            EntityKey::new(&self.name, EntityKey::parse_fields(&fields_str))
        } else {
            EntityKey::stub(&self.name, EntityKey::parse_fields(&fields_str))
        };
        self.keys.push(key);
        self.directives.push(FederationDirective::Key {
            fields: fields_str,
            resolvable,
        });
        self
    }

    /// Add a field.
    pub fn with_field(mut self, field: FederatedField) -> Self {
        self.fields.push(field);
        self
    }

    /// Check if this type is an entity (has @key).
    pub fn is_entity(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Get all field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Get external field names.
    pub fn external_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| f.is_external())
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Get owned (non-external) field names.
    pub fn owned_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| !f.is_external())
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Render the type definition in SDL with federation directives.
    pub fn to_sdl(&self) -> String {
        let directives: Vec<String> = self.directives.iter().map(|d| d.to_string()).collect();
        let dir_str = if directives.is_empty() {
            String::new()
        } else {
            format!(" {}", directives.join(" "))
        };

        let fields: Vec<String> = self.fields.iter().map(|f| f.to_sdl()).collect();

        format!(
            "type {}{} {{\n{}\n}}",
            self.name,
            dir_str,
            fields.join("\n")
        )
    }
}

// ---------------------------------------------------------------------------
// Entity representation
// ---------------------------------------------------------------------------

/// An entity representation sent by the gateway for _entities resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRepresentation {
    /// The __typename field.
    #[serde(rename = "__typename")]
    pub typename: String,
    /// Key fields and their values.
    #[serde(flatten)]
    pub fields: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Subgraph schema
// ---------------------------------------------------------------------------

/// A complete subgraph schema with federation metadata.
#[derive(Debug, Clone)]
pub struct SubgraphSchema {
    /// Subgraph name
    pub name: String,
    /// Federated types in this subgraph
    pub types: HashMap<String, FederatedType>,
    /// URL of this subgraph endpoint
    pub url: Option<String>,
}

impl SubgraphSchema {
    /// Create a new subgraph schema.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            types: HashMap::new(),
            url: None,
        }
    }

    /// Set the subgraph URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Add a federated type.
    pub fn add_type(&mut self, federated_type: FederatedType) {
        self.types
            .insert(federated_type.name.clone(), federated_type);
    }

    /// Get all entity types (types with @key).
    pub fn entity_types(&self) -> Vec<&FederatedType> {
        self.types.values().filter(|t| t.is_entity()).collect()
    }

    /// Get all type names.
    pub fn type_names(&self) -> Vec<&str> {
        self.types.keys().map(|k| k.as_str()).collect()
    }

    /// Lookup a type by name.
    pub fn get_type(&self, name: &str) -> Option<&FederatedType> {
        self.types.get(name)
    }

    /// Resolve entity representations using the registered types.
    pub fn resolve_entities(
        &self,
        representations: &[EntityRepresentation],
    ) -> Vec<Result<HashMap<String, serde_json::Value>, String>> {
        representations
            .iter()
            .map(|rep| self.resolve_entity(rep))
            .collect()
    }

    /// Resolve a single entity representation.
    fn resolve_entity(
        &self,
        rep: &EntityRepresentation,
    ) -> Result<HashMap<String, serde_json::Value>, String> {
        let federated_type = self
            .types
            .get(&rep.typename)
            .ok_or_else(|| format!("Unknown entity type: {}", rep.typename))?;

        if !federated_type.is_entity() {
            return Err(format!("{} is not an entity type", rep.typename));
        }

        // Check that at least one key is satisfied
        let key_satisfied = federated_type
            .keys
            .iter()
            .any(|key| key.fields.iter().all(|f| rep.fields.contains_key(f)));

        if !key_satisfied {
            return Err(format!(
                "No matching key for entity {} with provided fields",
                rep.typename
            ));
        }

        // Return the key fields plus __typename
        let mut result = rep.fields.clone();
        result.insert(
            "__typename".to_string(),
            serde_json::Value::String(rep.typename.clone()),
        );

        Ok(result)
    }

    /// Generate the _service SDL for introspection.
    pub fn service_sdl(&self) -> String {
        let mut parts = Vec::new();

        // Federation v2 schema directive
        parts.push(
            "extend schema @link(url: \"https://specs.apollo.dev/federation/v2.0\", \
             import: [\"@key\", \"@requires\", \"@provides\", \"@external\", \"@shareable\", \"@override\", \"@inaccessible\", \"@tag\"])"
                .to_string(),
        );
        parts.push(String::new());

        // Type definitions
        for federated_type in self.types.values() {
            parts.push(federated_type.to_sdl());
            parts.push(String::new());
        }

        parts.join("\n")
    }

    /// Validate the subgraph schema for federation compliance.
    pub fn validate(&self) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for (type_name, federated_type) in &self.types {
            // Entity types must have at least one field
            if federated_type.is_entity() && federated_type.fields.is_empty() {
                errors.push(ValidationError {
                    type_name: type_name.clone(),
                    field_name: None,
                    message: "Entity type has no fields".to_string(),
                });
            }

            // Key fields must exist
            for key in &federated_type.keys {
                for field_name in &key.fields {
                    if !federated_type.fields.iter().any(|f| &f.name == field_name) {
                        errors.push(ValidationError {
                            type_name: type_name.clone(),
                            field_name: Some(field_name.clone()),
                            message: format!("Key field '{field_name}' not found in type"),
                        });
                    }
                }
            }

            // @requires fields must be @external
            for field in &federated_type.fields {
                if let Some(required_fields) = field.requires_fields() {
                    for req_field in required_fields {
                        let target_field =
                            federated_type.fields.iter().find(|f| f.name == req_field);
                        if let Some(target) = target_field {
                            if !target.is_external() {
                                errors.push(ValidationError {
                                    type_name: type_name.clone(),
                                    field_name: Some(req_field),
                                    message: "Field referenced in @requires must be @external"
                                        .to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        errors
    }
}

/// A schema validation error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Type that has the error
    pub type_name: String,
    /// Field that has the error (if applicable)
    pub field_name: Option<String>,
    /// Error message
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(field) = &self.field_name {
            write!(f, "{}.{}: {}", self.type_name, field, self.message)
        } else {
            write!(f, "{}: {}", self.type_name, self.message)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- FederationDirective Display ---

    #[test]
    fn test_key_directive_display() {
        let d = FederationDirective::Key {
            fields: "id".to_string(),
            resolvable: true,
        };
        assert_eq!(d.to_string(), "@key(fields: \"id\")");
    }

    #[test]
    fn test_key_directive_non_resolvable_display() {
        let d = FederationDirective::Key {
            fields: "id".to_string(),
            resolvable: false,
        };
        assert!(d.to_string().contains("resolvable: false"));
    }

    #[test]
    fn test_requires_directive_display() {
        let d = FederationDirective::Requires {
            fields: "price weight".to_string(),
        };
        assert_eq!(d.to_string(), "@requires(fields: \"price weight\")");
    }

    #[test]
    fn test_provides_directive_display() {
        let d = FederationDirective::Provides {
            fields: "name".to_string(),
        };
        assert_eq!(d.to_string(), "@provides(fields: \"name\")");
    }

    #[test]
    fn test_external_directive_display() {
        assert_eq!(FederationDirective::External.to_string(), "@external");
    }

    #[test]
    fn test_shareable_directive_display() {
        assert_eq!(FederationDirective::Shareable.to_string(), "@shareable");
    }

    #[test]
    fn test_override_directive_display() {
        let d = FederationDirective::Override {
            from: "products".to_string(),
        };
        assert_eq!(d.to_string(), "@override(from: \"products\")");
    }

    #[test]
    fn test_tag_directive_display() {
        let d = FederationDirective::Tag {
            name: "internal".to_string(),
        };
        assert_eq!(d.to_string(), "@tag(name: \"internal\")");
    }

    // --- EntityKey ---

    #[test]
    fn test_entity_key_new() {
        let key = EntityKey::new("Product", vec!["id".to_string()]);
        assert_eq!(key.type_name, "Product");
        assert_eq!(key.fields, vec!["id"]);
        assert!(key.resolvable);
    }

    #[test]
    fn test_entity_key_stub() {
        let key = EntityKey::stub("Product", vec!["id".to_string()]);
        assert!(!key.resolvable);
    }

    #[test]
    fn test_entity_key_matches() {
        let key = EntityKey::new("Product", vec!["id".to_string(), "sku".to_string()]);
        let mut fields = HashMap::new();
        fields.insert("id".to_string(), "123".to_string());
        fields.insert("sku".to_string(), "ABC".to_string());
        assert!(key.matches(&fields));

        // Missing field
        let mut partial = HashMap::new();
        partial.insert("id".to_string(), "123".to_string());
        assert!(!key.matches(&partial));
    }

    #[test]
    fn test_entity_key_parse_fields() {
        let fields = EntityKey::parse_fields("id name email");
        assert_eq!(fields, vec!["id", "name", "email"]);
    }

    #[test]
    fn test_entity_key_parse_single_field() {
        let fields = EntityKey::parse_fields("id");
        assert_eq!(fields, vec!["id"]);
    }

    // --- FederatedField ---

    #[test]
    fn test_field_new() {
        let field = FederatedField::new("name", "String");
        assert_eq!(field.name, "name");
        assert_eq!(field.field_type, "String");
        assert!(field.nullable);
    }

    #[test]
    fn test_field_non_null() {
        let field = FederatedField::new("id", "ID").non_null();
        assert!(!field.nullable);
    }

    #[test]
    fn test_field_with_directive() {
        let field =
            FederatedField::new("price", "Float").with_directive(FederationDirective::External);
        assert!(field.is_external());
    }

    #[test]
    fn test_field_is_shareable() {
        let field =
            FederatedField::new("name", "String").with_directive(FederationDirective::Shareable);
        assert!(field.is_shareable());
    }

    #[test]
    fn test_field_is_inaccessible() {
        let field = FederatedField::new("internal", "String")
            .with_directive(FederationDirective::Inaccessible);
        assert!(field.is_inaccessible());
    }

    #[test]
    fn test_field_requires_fields() {
        let field = FederatedField::new("displayPrice", "String").with_directive(
            FederationDirective::Requires {
                fields: "price currency".to_string(),
            },
        );
        let req = field.requires_fields();
        assert!(req.is_some());
        assert_eq!(
            req.expect("should have requires"),
            vec!["price", "currency"]
        );
    }

    #[test]
    fn test_field_provides_fields() {
        let field = FederatedField::new("reviews", "[Review]").with_directive(
            FederationDirective::Provides {
                fields: "body author".to_string(),
            },
        );
        let prov = field.provides_fields();
        assert!(prov.is_some());
        assert_eq!(prov.expect("should have provides"), vec!["body", "author"]);
    }

    #[test]
    fn test_field_to_sdl_simple() {
        let field = FederatedField::new("name", "String");
        assert_eq!(field.to_sdl(), "  name: String");
    }

    #[test]
    fn test_field_to_sdl_non_null() {
        let field = FederatedField::new("id", "ID").non_null();
        assert_eq!(field.to_sdl(), "  id: ID!");
    }

    #[test]
    fn test_field_to_sdl_with_directive() {
        let field =
            FederatedField::new("price", "Float").with_directive(FederationDirective::External);
        let sdl = field.to_sdl();
        assert!(sdl.contains("@external"));
    }

    // --- FederatedType ---

    #[test]
    fn test_type_new() {
        let t = FederatedType::new("Product");
        assert_eq!(t.name, "Product");
        assert!(!t.is_entity());
        assert!(t.keys.is_empty());
    }

    #[test]
    fn test_type_with_key() {
        let t = FederatedType::new("Product").with_key("id", true);
        assert!(t.is_entity());
        assert_eq!(t.keys.len(), 1);
        assert_eq!(t.keys[0].fields, vec!["id"]);
    }

    #[test]
    fn test_type_with_multiple_keys() {
        let t = FederatedType::new("Product")
            .with_key("id", true)
            .with_key("sku", true);
        assert_eq!(t.keys.len(), 2);
    }

    #[test]
    fn test_type_with_fields() {
        let t = FederatedType::new("Product")
            .with_key("id", true)
            .with_field(FederatedField::new("id", "ID").non_null())
            .with_field(FederatedField::new("name", "String").non_null())
            .with_field(
                FederatedField::new("price", "Float").with_directive(FederationDirective::External),
            );

        assert_eq!(t.field_names().len(), 3);
        assert_eq!(t.external_fields(), vec!["price"]);
        assert_eq!(t.owned_fields().len(), 2);
    }

    #[test]
    fn test_type_to_sdl() {
        let t = FederatedType::new("Product")
            .with_key("id", true)
            .with_field(FederatedField::new("id", "ID").non_null())
            .with_field(FederatedField::new("name", "String"));

        let sdl = t.to_sdl();
        assert!(sdl.contains("type Product"));
        assert!(sdl.contains("@key"));
        assert!(sdl.contains("id: ID!"));
        assert!(sdl.contains("name: String"));
    }

    // --- SubgraphSchema ---

    #[test]
    fn test_subgraph_schema_new() {
        let schema = SubgraphSchema::new("products");
        assert_eq!(schema.name, "products");
        assert!(schema.types.is_empty());
    }

    #[test]
    fn test_subgraph_schema_with_url() {
        let schema = SubgraphSchema::new("products").with_url("http://localhost:4001/graphql");
        assert_eq!(
            schema.url,
            Some("http://localhost:4001/graphql".to_string())
        );
    }

    #[test]
    fn test_subgraph_add_type() {
        let mut schema = SubgraphSchema::new("products");
        schema.add_type(
            FederatedType::new("Product")
                .with_key("id", true)
                .with_field(FederatedField::new("id", "ID").non_null()),
        );
        assert_eq!(schema.types.len(), 1);
        assert!(schema.get_type("Product").is_some());
    }

    #[test]
    fn test_subgraph_entity_types() {
        let mut schema = SubgraphSchema::new("products");
        schema.add_type(
            FederatedType::new("Product")
                .with_key("id", true)
                .with_field(FederatedField::new("id", "ID").non_null()),
        );
        schema.add_type(
            FederatedType::new("Category").with_field(FederatedField::new("name", "String")),
        );

        let entities = schema.entity_types();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Product");
    }

    #[test]
    fn test_subgraph_type_names() {
        let mut schema = SubgraphSchema::new("test");
        schema.add_type(FederatedType::new("A"));
        schema.add_type(FederatedType::new("B"));
        let names = schema.type_names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_subgraph_resolve_entities_success() {
        let mut schema = SubgraphSchema::new("products");
        schema.add_type(
            FederatedType::new("Product")
                .with_key("id", true)
                .with_field(FederatedField::new("id", "ID").non_null())
                .with_field(FederatedField::new("name", "String")),
        );

        let rep = EntityRepresentation {
            typename: "Product".to_string(),
            fields: {
                let mut m = HashMap::new();
                m.insert(
                    "id".to_string(),
                    serde_json::Value::String("123".to_string()),
                );
                m
            },
        };

        let results = schema.resolve_entities(&[rep]);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_subgraph_resolve_entities_unknown_type() {
        let schema = SubgraphSchema::new("products");
        let rep = EntityRepresentation {
            typename: "Unknown".to_string(),
            fields: HashMap::new(),
        };

        let results = schema.resolve_entities(&[rep]);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }

    #[test]
    fn test_subgraph_resolve_entities_missing_key() {
        let mut schema = SubgraphSchema::new("products");
        schema.add_type(
            FederatedType::new("Product")
                .with_key("id", true)
                .with_field(FederatedField::new("id", "ID").non_null()),
        );

        let rep = EntityRepresentation {
            typename: "Product".to_string(),
            fields: HashMap::new(), // Missing "id"
        };

        let results = schema.resolve_entities(&[rep]);
        assert!(results[0].is_err());
    }

    #[test]
    fn test_service_sdl() {
        let mut schema = SubgraphSchema::new("products");
        schema.add_type(
            FederatedType::new("Product")
                .with_key("id", true)
                .with_field(FederatedField::new("id", "ID").non_null())
                .with_field(FederatedField::new("name", "String")),
        );

        let sdl = schema.service_sdl();
        assert!(sdl.contains("@link"));
        assert!(sdl.contains("federation/v2.0"));
        assert!(sdl.contains("type Product"));
    }

    // --- Validation ---

    #[test]
    fn test_validate_valid_schema() {
        let mut schema = SubgraphSchema::new("products");
        schema.add_type(
            FederatedType::new("Product")
                .with_key("id", true)
                .with_field(FederatedField::new("id", "ID").non_null())
                .with_field(FederatedField::new("name", "String")),
        );

        let errors = schema.validate();
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_validate_entity_no_fields() {
        let mut schema = SubgraphSchema::new("test");
        schema.add_type(FederatedType::new("Empty").with_key("id", true));

        let errors = schema.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("no fields"));
    }

    #[test]
    fn test_validate_key_field_missing() {
        let mut schema = SubgraphSchema::new("test");
        schema.add_type(
            FederatedType::new("Product")
                .with_key("missing_field", true)
                .with_field(FederatedField::new("name", "String")),
        );

        let errors = schema.validate();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.message.contains("not found")));
    }

    #[test]
    fn test_validate_requires_external() {
        let mut schema = SubgraphSchema::new("test");
        schema.add_type(
            FederatedType::new("Product")
                .with_key("id", true)
                .with_field(FederatedField::new("id", "ID").non_null())
                .with_field(FederatedField::new("price", "Float")) // NOT @external
                .with_field(
                    FederatedField::new("displayPrice", "String").with_directive(
                        FederationDirective::Requires {
                            fields: "price".to_string(),
                        },
                    ),
                ),
        );

        let errors = schema.validate();
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| e.message.contains("must be @external")));
    }

    #[test]
    fn test_validation_error_display_with_field() {
        let err = ValidationError {
            type_name: "Product".to_string(),
            field_name: Some("price".to_string()),
            message: "must be @external".to_string(),
        };
        assert_eq!(err.to_string(), "Product.price: must be @external");
    }

    #[test]
    fn test_validation_error_display_without_field() {
        let err = ValidationError {
            type_name: "Product".to_string(),
            field_name: None,
            message: "has no fields".to_string(),
        };
        assert_eq!(err.to_string(), "Product: has no fields");
    }

    // --- Complex scenario ---

    #[test]
    fn test_federation_v2_complete_scenario() {
        let mut schema = SubgraphSchema::new("inventory").with_url("http://localhost:4002/graphql");

        // Product entity with composite key
        let product = FederatedType::new("Product")
            .with_key("id", true)
            .with_key("sku warehouse", true)
            .with_field(FederatedField::new("id", "ID").non_null())
            .with_field(FederatedField::new("sku", "String").non_null())
            .with_field(FederatedField::new("warehouse", "String").non_null())
            .with_field(
                FederatedField::new("price", "Float").with_directive(FederationDirective::External),
            )
            .with_field(
                FederatedField::new("inStock", "Boolean")
                    .with_directive(FederationDirective::Shareable),
            )
            .with_field(
                FederatedField::new("shippingEstimate", "Float").with_directive(
                    FederationDirective::Requires {
                        fields: "price".to_string(),
                    },
                ),
            );

        schema.add_type(product);

        // Validate
        let errors = schema.validate();
        assert!(errors.is_empty(), "Errors: {:?}", errors);

        // Check entity resolution
        assert_eq!(schema.entity_types().len(), 1);

        // SDL generation
        let sdl = schema.service_sdl();
        assert!(sdl.contains("@key"));
        assert!(sdl.contains("@external"));
        assert!(sdl.contains("@shareable"));
        assert!(sdl.contains("@requires"));
    }

    #[test]
    fn test_entity_representation_deserialization() {
        let json = r#"{"__typename": "Product", "id": "p1", "sku": "SKU-001"}"#;
        let rep: EntityRepresentation = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(rep.typename, "Product");
        assert!(rep.fields.contains_key("id"));
        assert!(rep.fields.contains_key("sku"));
    }

    #[test]
    fn test_multiple_resolve() {
        let mut schema = SubgraphSchema::new("test");
        schema.add_type(
            FederatedType::new("User")
                .with_key("id", true)
                .with_field(FederatedField::new("id", "ID").non_null())
                .with_field(FederatedField::new("name", "String")),
        );

        let reps = vec![
            EntityRepresentation {
                typename: "User".to_string(),
                fields: {
                    let mut m = HashMap::new();
                    m.insert(
                        "id".to_string(),
                        serde_json::Value::String("u1".to_string()),
                    );
                    m
                },
            },
            EntityRepresentation {
                typename: "User".to_string(),
                fields: {
                    let mut m = HashMap::new();
                    m.insert(
                        "id".to_string(),
                        serde_json::Value::String("u2".to_string()),
                    );
                    m
                },
            },
        ];

        let results = schema.resolve_entities(&reps);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
