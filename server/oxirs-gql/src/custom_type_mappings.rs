//! Custom type mapping system for GraphQL schema generation
//!
//! This module provides flexible type mapping capabilities:
//! - Runtime custom type registration
//! - Type transformation rules
//! - Pattern-based type matching
//! - Composite type mappings
//! - Type validation

use crate::types::{BuiltinScalars, GraphQLType, ScalarType};
use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Type alias for serialization function
pub type SerializeFn = Arc<dyn Fn(&str) -> Result<serde_json::Value> + Send + Sync>;

/// Type alias for deserialization function
pub type DeserializeFn = Arc<dyn Fn(&serde_json::Value) -> Result<String> + Send + Sync>;

/// Type alias for validation function
pub type ValidateFn = Arc<dyn Fn(&str) -> Result<()> + Send + Sync>;

/// Type alias for transformation function
pub type TransformFn = Arc<dyn Fn(&str) -> Result<String> + Send + Sync>;

/// Type mapping rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeMappingRule {
    /// Pattern to match RDF URIs (regex or exact match)
    pub pattern: String,
    /// Target GraphQL type name
    pub target_type: String,
    /// Whether pattern is a regex
    pub is_regex: bool,
    /// Priority (higher = higher priority)
    pub priority: u32,
    /// Optional transformation function name
    pub transformer: Option<String>,
    /// Whether this is a list type
    pub is_list: bool,
    /// Whether this is a non-null type
    pub is_non_null: bool,
}

impl TypeMappingRule {
    pub fn exact_match(pattern: String, target_type: String) -> Self {
        Self {
            pattern,
            target_type,
            is_regex: false,
            priority: 100,
            transformer: None,
            is_list: false,
            is_non_null: false,
        }
    }

    pub fn regex_match(pattern: String, target_type: String) -> Self {
        Self {
            pattern,
            target_type,
            is_regex: true,
            priority: 50,
            transformer: None,
            is_list: false,
            is_non_null: false,
        }
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_transformer(mut self, transformer: String) -> Self {
        self.transformer = Some(transformer);
        self
    }

    pub fn as_list(mut self) -> Self {
        self.is_list = true;
        self
    }

    pub fn as_non_null(mut self) -> Self {
        self.is_non_null = true;
        self
    }

    pub fn matches(&self, uri: &str) -> bool {
        if self.is_regex {
            if let Ok(re) = Regex::new(&self.pattern) {
                re.is_match(uri)
            } else {
                false
            }
        } else {
            self.pattern == uri
        }
    }
}

/// Custom scalar type definition
#[derive(Clone)]
pub struct CustomScalarDef {
    pub name: String,
    pub description: Option<String>,
    /// Serialization function (converts internal representation to JSON)
    pub serialize: SerializeFn,
    /// Deserialization function (converts JSON to internal representation)
    pub deserialize: DeserializeFn,
    /// Validation function
    pub validate: Option<ValidateFn>,
}

impl std::fmt::Debug for CustomScalarDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomScalarDef")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("serialize", &"<function>")
            .field("deserialize", &"<function>")
            .field(
                "validate",
                &if self.validate.is_some() {
                    "<function>"
                } else {
                    "None"
                },
            )
            .finish()
    }
}

impl CustomScalarDef {
    pub fn new(name: String, serialize: SerializeFn, deserialize: DeserializeFn) -> Self {
        Self {
            name,
            description: None,
            serialize,
            deserialize,
            validate: None,
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_validator(mut self, validator: ValidateFn) -> Self {
        self.validate = Some(validator);
        self
    }
}

/// Custom type mapper with advanced features
pub struct CustomTypeMapper {
    /// Type mapping rules (ordered by priority)
    rules: Vec<TypeMappingRule>,
    /// Custom scalar definitions
    custom_scalars: HashMap<String, CustomScalarDef>,
    /// Type aliases
    type_aliases: HashMap<String, String>,
    /// Transformation functions
    transformers: HashMap<String, TransformFn>,
}

impl CustomTypeMapper {
    pub fn new() -> Self {
        let mut mapper = Self {
            rules: Vec::new(),
            custom_scalars: HashMap::new(),
            type_aliases: HashMap::new(),
            transformers: HashMap::new(),
        };

        // Add default XSD type mappings
        mapper.add_default_mappings();

        mapper
    }

    /// Add default XSD and RDF type mappings
    fn add_default_mappings(&mut self) {
        // XSD string types
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#string".to_string(),
            "String".to_string(),
        ));

        // XSD numeric types
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#int".to_string(),
            "Int".to_string(),
        ));
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#integer".to_string(),
            "Int".to_string(),
        ));
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#long".to_string(),
            "Int".to_string(),
        ));

        // XSD decimal types
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#float".to_string(),
            "Float".to_string(),
        ));
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#double".to_string(),
            "Float".to_string(),
        ));
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
            "Float".to_string(),
        ));

        // XSD boolean
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
            "Boolean".to_string(),
        ));

        // XSD date/time types - map to custom DateTime scalar
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#dateTime".to_string(),
            "DateTime".to_string(),
        ));
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#date".to_string(),
            "DateTime".to_string(),
        ));
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#time".to_string(),
            "DateTime".to_string(),
        ));

        // XSD anyURI -> IRI custom scalar
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#anyURI".to_string(),
            "IRI".to_string(),
        ));

        // RDF types
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string(),
            "LangString".to_string(),
        ));
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2000/01/rdf-schema#Literal".to_string(),
            "Literal".to_string(),
        ));
        self.add_rule(TypeMappingRule::exact_match(
            "http://www.w3.org/2000/01/rdf-schema#Resource".to_string(),
            "IRI".to_string(),
        ));
    }

    /// Add a type mapping rule
    pub fn add_rule(&mut self, rule: TypeMappingRule) {
        self.rules.push(rule);
        // Sort by priority (descending)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Add multiple rules
    pub fn add_rules(&mut self, rules: Vec<TypeMappingRule>) {
        for rule in rules {
            self.add_rule(rule);
        }
    }

    /// Register a custom scalar type
    pub fn register_custom_scalar(&mut self, scalar: CustomScalarDef) {
        self.custom_scalars.insert(scalar.name.clone(), scalar);
    }

    /// Add a type alias
    pub fn add_alias(&mut self, alias: String, target: String) {
        self.type_aliases.insert(alias, target);
    }

    /// Register a transformation function
    pub fn register_transformer(&mut self, name: String, transformer: TransformFn) {
        self.transformers.insert(name, transformer);
    }

    /// Map RDF URI to GraphQL type
    pub fn map_type(&self, rdf_uri: &str) -> Result<GraphQLType> {
        // Find matching rule
        for rule in &self.rules {
            if rule.matches(rdf_uri) {
                let mut gql_type = self.resolve_target_type(&rule.target_type)?;

                // Apply transformation if specified
                if let Some(transformer_name) = &rule.transformer {
                    if let Some(transformer) = self.transformers.get(transformer_name) {
                        let transformed_uri = transformer(rdf_uri)?;
                        gql_type = self.resolve_target_type(&transformed_uri)?;
                    }
                }

                // Apply list wrapper if needed
                if rule.is_list {
                    gql_type = GraphQLType::List(Box::new(gql_type));
                }

                // Apply non-null wrapper if needed
                if rule.is_non_null {
                    gql_type = GraphQLType::NonNull(Box::new(gql_type));
                }

                return Ok(gql_type);
            }
        }

        // No matching rule found, default to String
        Ok(GraphQLType::Scalar(BuiltinScalars::string()))
    }

    /// Resolve target type name to GraphQL type
    fn resolve_target_type(&self, type_name: &str) -> Result<GraphQLType> {
        // Check for alias
        let resolved_name = self
            .type_aliases
            .get(type_name)
            .map(|s| s.as_str())
            .unwrap_or(type_name);

        // Check for builtin scalars
        match resolved_name {
            "String" => return Ok(GraphQLType::Scalar(BuiltinScalars::string())),
            "Int" => return Ok(GraphQLType::Scalar(BuiltinScalars::int())),
            "Float" => return Ok(GraphQLType::Scalar(BuiltinScalars::float())),
            "Boolean" => return Ok(GraphQLType::Scalar(BuiltinScalars::boolean())),
            "ID" => return Ok(GraphQLType::Scalar(BuiltinScalars::id())),
            _ => {}
        }

        // Check for custom scalars
        if let Some(custom_scalar) = self.custom_scalars.get(resolved_name) {
            return Ok(GraphQLType::Scalar(ScalarType {
                name: custom_scalar.name.clone(),
                description: custom_scalar.description.clone(),
                serialize: |_| Ok(crate::ast::Value::NullValue),
                parse_value: |_| Err(anyhow!("Parsing not implemented")),
                parse_literal: |_| Err(anyhow!("Parsing not implemented")),
            }));
        }

        // Assume it's an object type reference
        Ok(GraphQLType::Scalar(BuiltinScalars::string()))
    }

    /// Get all custom scalars for schema generation
    pub fn get_custom_scalars(&self) -> Vec<GraphQLType> {
        self.custom_scalars
            .values()
            .map(|scalar| {
                GraphQLType::Scalar(ScalarType {
                    name: scalar.name.clone(),
                    description: scalar.description.clone(),
                    serialize: |_| Ok(crate::ast::Value::NullValue),
                    parse_value: |_| Err(anyhow!("Parsing not implemented")),
                    parse_literal: |_| Err(anyhow!("Parsing not implemented")),
                })
            })
            .collect()
    }

    /// Create common custom scalars
    pub fn create_common_scalars() -> Self {
        let mut mapper = Self::new();

        // Email scalar
        let email_validator = Arc::new(|value: &str| -> Result<()> {
            if value.contains('@') {
                Ok(())
            } else {
                Err(anyhow!("Invalid email format"))
            }
        });

        let email_scalar = CustomScalarDef::new(
            "Email".to_string(),
            Arc::new(|s| Ok(serde_json::Value::String(s.to_string()))),
            Arc::new(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow!("Expected string"))
            }),
        )
        .with_description("Email address scalar type".to_string())
        .with_validator(email_validator);

        mapper.register_custom_scalar(email_scalar);

        // URL scalar
        let url_scalar = CustomScalarDef::new(
            "URL".to_string(),
            Arc::new(|s| Ok(serde_json::Value::String(s.to_string()))),
            Arc::new(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow!("Expected string"))
            }),
        )
        .with_description("URL scalar type".to_string());

        mapper.register_custom_scalar(url_scalar);

        // JSON scalar
        let json_scalar = CustomScalarDef::new(
            "JSON".to_string(),
            Arc::new(|s| serde_json::from_str(s).map_err(|e| anyhow!("Invalid JSON: {}", e))),
            Arc::new(|v| {
                serde_json::to_string(v).map_err(|e| anyhow!("JSON serialization error: {}", e))
            }),
        )
        .with_description("JSON scalar type".to_string());

        mapper.register_custom_scalar(json_scalar);

        mapper
    }
}

impl Default for CustomTypeMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_mapping_rule_exact_match() {
        let rule = TypeMappingRule::exact_match(
            "http://www.w3.org/2001/XMLSchema#string".to_string(),
            "String".to_string(),
        );

        assert!(rule.matches("http://www.w3.org/2001/XMLSchema#string"));
        assert!(!rule.matches("http://www.w3.org/2001/XMLSchema#int"));
    }

    #[test]
    fn test_type_mapping_rule_regex_match() {
        let rule =
            TypeMappingRule::regex_match(r".*XMLSchema#\w+".to_string(), "String".to_string());

        assert!(rule.matches("http://www.w3.org/2001/XMLSchema#string"));
        assert!(rule.matches("http://www.w3.org/2001/XMLSchema#int"));
        assert!(!rule.matches("http://example.org/custom"));
    }

    #[test]
    fn test_type_mapping_rule_priority() {
        let rule1 = TypeMappingRule::exact_match("test".to_string(), "Type1".to_string());
        let rule2 = TypeMappingRule::exact_match("test".to_string(), "Type2".to_string())
            .with_priority(200);

        assert_eq!(rule1.priority, 100);
        assert_eq!(rule2.priority, 200);
    }

    #[test]
    fn test_custom_type_mapper_creation() {
        let mapper = CustomTypeMapper::new();
        assert!(!mapper.rules.is_empty()); // Should have default mappings
    }

    #[test]
    fn test_custom_type_mapper_add_rule() {
        let mut mapper = CustomTypeMapper::new();
        let initial_count = mapper.rules.len();

        mapper.add_rule(
            TypeMappingRule::exact_match(
                "http://example.org/CustomType".to_string(),
                "CustomType".to_string(),
            )
            .with_priority(300),
        );

        assert_eq!(mapper.rules.len(), initial_count + 1);
        // Should be first due to high priority
        assert_eq!(mapper.rules[0].target_type, "CustomType");
    }

    #[test]
    fn test_custom_type_mapper_map_xsd_string() {
        let mapper = CustomTypeMapper::new();
        let result = mapper.map_type("http://www.w3.org/2001/XMLSchema#string");

        assert!(result.is_ok());
        let gql_type = result.unwrap();

        match gql_type {
            GraphQLType::Scalar(scalar) => assert_eq!(scalar.name, "String"),
            _ => panic!("Expected scalar type"),
        }
    }

    #[test]
    fn test_custom_type_mapper_map_xsd_int() {
        let mapper = CustomTypeMapper::new();
        let result = mapper.map_type("http://www.w3.org/2001/XMLSchema#int");

        assert!(result.is_ok());
        let gql_type = result.unwrap();

        match gql_type {
            GraphQLType::Scalar(scalar) => assert_eq!(scalar.name, "Int"),
            _ => panic!("Expected scalar type"),
        }
    }

    #[test]
    fn test_custom_type_mapper_map_unknown_type() {
        let mapper = CustomTypeMapper::new();
        let result = mapper.map_type("http://example.org/UnknownType");

        assert!(result.is_ok());
        // Should default to String
        let gql_type = result.unwrap();

        match gql_type {
            GraphQLType::Scalar(scalar) => assert_eq!(scalar.name, "String"),
            _ => panic!("Expected scalar type"),
        }
    }

    #[test]
    fn test_custom_scalar_def_creation() {
        let scalar = CustomScalarDef::new(
            "Email".to_string(),
            Arc::new(|s| Ok(serde_json::Value::String(s.to_string()))),
            Arc::new(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow!("Expected string"))
            }),
        );

        assert_eq!(scalar.name, "Email");
        assert!(scalar.description.is_none());
        assert!(scalar.validate.is_none());
    }

    #[test]
    fn test_custom_scalar_with_description() {
        let scalar = CustomScalarDef::new(
            "Email".to_string(),
            Arc::new(|s| Ok(serde_json::Value::String(s.to_string()))),
            Arc::new(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow!("Expected string"))
            }),
        )
        .with_description("Email address".to_string());

        assert_eq!(scalar.description, Some("Email address".to_string()));
    }

    #[test]
    fn test_common_scalars_creation() {
        let mapper = CustomTypeMapper::create_common_scalars();

        assert!(mapper.custom_scalars.contains_key("Email"));
        assert!(mapper.custom_scalars.contains_key("URL"));
        assert!(mapper.custom_scalars.contains_key("JSON"));
    }

    #[test]
    fn test_type_alias() {
        let mut mapper = CustomTypeMapper::new();
        mapper.add_alias("Str".to_string(), "String".to_string());

        // The alias should resolve to String
        assert!(mapper.type_aliases.contains_key("Str"));
        assert_eq!(mapper.type_aliases.get("Str").unwrap(), "String");
    }

    #[test]
    fn test_list_type_mapping() {
        let mut mapper = CustomTypeMapper::new();
        mapper.add_rule(
            TypeMappingRule::exact_match(
                "http://example.org/ListType".to_string(),
                "String".to_string(),
            )
            .as_list(),
        );

        let result = mapper.map_type("http://example.org/ListType");
        assert!(result.is_ok());

        match result.unwrap() {
            GraphQLType::List(_) => {} // Success
            _ => panic!("Expected list type"),
        }
    }

    #[test]
    fn test_non_null_type_mapping() {
        let mut mapper = CustomTypeMapper::new();
        mapper.add_rule(
            TypeMappingRule::exact_match(
                "http://example.org/RequiredType".to_string(),
                "String".to_string(),
            )
            .as_non_null(),
        );

        let result = mapper.map_type("http://example.org/RequiredType");
        assert!(result.is_ok());

        match result.unwrap() {
            GraphQLType::NonNull(_) => {} // Success
            _ => panic!("Expected non-null type"),
        }
    }
}
