// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Automatic Schema Composition for Federation
//!
//! This module provides automatic composition of GraphQL schemas from multiple
//! federated subgraphs, handling conflicts, merging types, and generating a
//! unified gateway schema.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// GraphQL type kind
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeKind {
    Object,
    Interface,
    Union,
    Enum,
    InputObject,
    Scalar,
}

/// GraphQL field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: String,
    /// Field arguments
    pub arguments: Vec<ArgumentDefinition>,
    /// Field description
    pub description: Option<String>,
    /// Source subgraph
    pub source: String,
    /// Deprecation info
    pub deprecated: Option<String>,
}

impl FieldDefinition {
    /// Create a new field definition
    pub fn new(name: String, field_type: String, source: String) -> Self {
        Self {
            name,
            field_type,
            arguments: Vec::new(),
            description: None,
            source,
            deprecated: None,
        }
    }

    /// Add an argument
    pub fn with_argument(mut self, arg: ArgumentDefinition) -> Self {
        self.arguments.push(arg);
        self
    }

    /// Check if field is compatible with another
    pub fn is_compatible_with(&self, other: &FieldDefinition) -> bool {
        self.name == other.name && self.field_type == other.field_type
    }
}

/// GraphQL argument definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentDefinition {
    /// Argument name
    pub name: String,
    /// Argument type
    pub arg_type: String,
    /// Default value
    pub default_value: Option<String>,
    /// Description
    pub description: Option<String>,
}

/// GraphQL type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefinition {
    /// Type name
    pub name: String,
    /// Type kind
    pub kind: TypeKind,
    /// Fields (for Object/Interface types)
    pub fields: Vec<FieldDefinition>,
    /// Interfaces implemented (for Object types)
    pub interfaces: Vec<String>,
    /// Possible types (for Union types)
    pub possible_types: Vec<String>,
    /// Enum values (for Enum types)
    pub enum_values: Vec<String>,
    /// Description
    pub description: Option<String>,
    /// Source subgraphs
    pub sources: HashSet<String>,
}

impl TypeDefinition {
    /// Create a new type definition
    pub fn new(name: String, kind: TypeKind, source: String) -> Self {
        let mut sources = HashSet::new();
        sources.insert(source);
        Self {
            name,
            kind,
            fields: Vec::new(),
            interfaces: Vec::new(),
            possible_types: Vec::new(),
            enum_values: Vec::new(),
            description: None,
            sources,
        }
    }

    /// Add a field
    pub fn with_field(mut self, field: FieldDefinition) -> Self {
        self.fields.push(field);
        self
    }

    /// Add an interface
    pub fn with_interface(mut self, interface: String) -> Self {
        self.interfaces.push(interface);
        self
    }

    /// Merge with another type definition
    pub fn merge(&mut self, other: TypeDefinition) -> Result<()> {
        if self.name != other.name {
            return Err(anyhow!(
                "Cannot merge types with different names: {} vs {}",
                self.name,
                other.name
            ));
        }

        if self.kind != other.kind {
            return Err(anyhow!(
                "Cannot merge types with different kinds: {:?} vs {:?}",
                self.kind,
                other.kind
            ));
        }

        // Merge fields
        for other_field in other.fields {
            if let Some(existing_field) =
                self.fields.iter_mut().find(|f| f.name == other_field.name)
            {
                if !existing_field.is_compatible_with(&other_field) {
                    return Err(anyhow!(
                        "Field '{}' has incompatible types in merged schemas",
                        other_field.name
                    ));
                }
            } else {
                self.fields.push(other_field);
            }
        }

        // Merge interfaces
        for interface in other.interfaces {
            if !self.interfaces.contains(&interface) {
                self.interfaces.push(interface);
            }
        }

        // Merge possible types (for unions)
        for possible_type in other.possible_types {
            if !self.possible_types.contains(&possible_type) {
                self.possible_types.push(possible_type);
            }
        }

        // Merge enum values
        for enum_value in other.enum_values {
            if !self.enum_values.contains(&enum_value) {
                self.enum_values.push(enum_value);
            }
        }

        // Merge sources
        self.sources.extend(other.sources);

        Ok(())
    }
}

/// Subgraph schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphSchema {
    /// Subgraph name
    pub name: String,
    /// Type definitions
    pub types: HashMap<String, TypeDefinition>,
    /// SDL (Schema Definition Language)
    pub sdl: Option<String>,
}

impl SubgraphSchema {
    /// Create a new subgraph schema
    pub fn new(name: String) -> Self {
        Self {
            name,
            types: HashMap::new(),
            sdl: None,
        }
    }

    /// Add a type definition
    pub fn add_type(&mut self, type_def: TypeDefinition) {
        self.types.insert(type_def.name.clone(), type_def);
    }

    /// Get a type by name
    pub fn get_type(&self, name: &str) -> Option<&TypeDefinition> {
        self.types.get(name)
    }
}

/// Composed schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedSchema {
    /// All type definitions
    pub types: HashMap<String, TypeDefinition>,
    /// Query type name
    pub query_type: String,
    /// Mutation type name (optional)
    pub mutation_type: Option<String>,
    /// Subscription type name (optional)
    pub subscription_type: Option<String>,
    /// Contributing subgraphs
    pub subgraphs: Vec<String>,
    /// Composition metadata
    pub metadata: HashMap<String, String>,
}

impl ComposedSchema {
    /// Create a new composed schema
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            query_type: "Query".to_string(),
            mutation_type: Some("Mutation".to_string()),
            subscription_type: Some("Subscription".to_string()),
            subgraphs: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Generate SDL representation
    pub fn to_sdl(&self) -> String {
        let mut sdl = String::new();

        // Add schema definition
        sdl.push_str("schema {\n");
        sdl.push_str(&format!("  query: {}\n", self.query_type));
        if let Some(mutation) = &self.mutation_type {
            sdl.push_str(&format!("  mutation: {}\n", mutation));
        }
        if let Some(subscription) = &self.subscription_type {
            sdl.push_str(&format!("  subscription: {}\n", subscription));
        }
        sdl.push_str("}\n\n");

        // Add type definitions
        for type_def in self.types.values() {
            sdl.push_str(&self.type_to_sdl(type_def));
            sdl.push('\n');
        }

        sdl
    }

    /// Convert a type definition to SDL
    fn type_to_sdl(&self, type_def: &TypeDefinition) -> String {
        let mut sdl = String::new();

        if let Some(desc) = &type_def.description {
            sdl.push_str(&format!("\"\"\"{}\"\"\"\n", desc));
        }

        match type_def.kind {
            TypeKind::Object => {
                sdl.push_str(&format!("type {}", type_def.name));
                if !type_def.interfaces.is_empty() {
                    sdl.push_str(&format!(" implements {}", type_def.interfaces.join(" & ")));
                }
                sdl.push_str(" {\n");
                for field in &type_def.fields {
                    sdl.push_str(&format!("  {}: {}\n", field.name, field.field_type));
                }
                sdl.push_str("}\n");
            }
            TypeKind::Interface => {
                sdl.push_str(&format!("interface {} {{\n", type_def.name));
                for field in &type_def.fields {
                    sdl.push_str(&format!("  {}: {}\n", field.name, field.field_type));
                }
                sdl.push_str("}\n");
            }
            TypeKind::Union => {
                sdl.push_str(&format!(
                    "union {} = {}\n",
                    type_def.name,
                    type_def.possible_types.join(" | ")
                ));
            }
            TypeKind::Enum => {
                sdl.push_str(&format!("enum {} {{\n", type_def.name));
                for value in &type_def.enum_values {
                    sdl.push_str(&format!("  {}\n", value));
                }
                sdl.push_str("}\n");
            }
            TypeKind::InputObject => {
                sdl.push_str(&format!("input {} {{\n", type_def.name));
                for field in &type_def.fields {
                    sdl.push_str(&format!("  {}: {}\n", field.name, field.field_type));
                }
                sdl.push_str("}\n");
            }
            TypeKind::Scalar => {
                sdl.push_str(&format!("scalar {}\n", type_def.name));
            }
        }

        sdl
    }
}

impl Default for ComposedSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Composition conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionConflict {
    /// Conflict type
    pub conflict_type: String,
    /// Type name involved
    pub type_name: String,
    /// Field name (if applicable)
    pub field_name: Option<String>,
    /// Conflicting subgraphs
    pub subgraphs: Vec<String>,
    /// Conflict description
    pub description: String,
}

/// Composition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionResult {
    /// Composed schema (if successful)
    pub schema: Option<ComposedSchema>,
    /// Conflicts encountered
    pub conflicts: Vec<CompositionConflict>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Success status
    pub success: bool,
}

/// Automatic schema composer
pub struct AutomaticSchemaComposer {
    /// Registered subgraph schemas
    schemas: Arc<RwLock<HashMap<String, SubgraphSchema>>>,
    /// Composed schema cache
    composed: Arc<RwLock<Option<ComposedSchema>>>,
    /// Composition metadata
    metadata: Arc<RwLock<HashMap<String, String>>>,
}

impl AutomaticSchemaComposer {
    /// Create a new automatic schema composer
    pub fn new() -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            composed: Arc::new(RwLock::new(None)),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a subgraph schema
    pub async fn register_subgraph(&self, schema: SubgraphSchema) -> Result<()> {
        let mut schemas = self.schemas.write().await;
        schemas.insert(schema.name.clone(), schema);

        // Invalidate composed schema
        let mut composed = self.composed.write().await;
        *composed = None;

        Ok(())
    }

    /// Unregister a subgraph schema
    pub async fn unregister_subgraph(&self, name: &str) -> Result<()> {
        let mut schemas = self.schemas.write().await;
        schemas.remove(name);

        // Invalidate composed schema
        let mut composed = self.composed.write().await;
        *composed = None;

        Ok(())
    }

    /// Compose all registered schemas
    pub async fn compose(&self) -> Result<CompositionResult> {
        let schemas = self.schemas.read().await;

        if schemas.is_empty() {
            return Ok(CompositionResult {
                schema: None,
                conflicts: Vec::new(),
                warnings: vec!["No subgraph schemas registered".to_string()],
                success: false,
            });
        }

        let mut composed = ComposedSchema::new();
        let mut conflicts = Vec::new();
        let mut warnings = Vec::new();

        // Merge all types from all subgraphs
        for (subgraph_name, subgraph) in schemas.iter() {
            composed.subgraphs.push(subgraph_name.clone());

            for (type_name, type_def) in &subgraph.types {
                if let Some(existing_type) = composed.types.get_mut(type_name) {
                    // Type exists, try to merge
                    if let Err(e) = existing_type.merge(type_def.clone()) {
                        conflicts.push(CompositionConflict {
                            conflict_type: "TYPE_MERGE_CONFLICT".to_string(),
                            type_name: type_name.clone(),
                            field_name: None,
                            subgraphs: vec![
                                existing_type
                                    .sources
                                    .iter()
                                    .next()
                                    .unwrap_or(&String::new())
                                    .clone(),
                                subgraph_name.clone(),
                            ],
                            description: e.to_string(),
                        });
                    }
                } else {
                    // New type, add it
                    composed.types.insert(type_name.clone(), type_def.clone());
                }
            }
        }

        // Validate the composed schema
        let validation_warnings = self.validate_composed_schema(&composed).await;
        warnings.extend(validation_warnings);

        let success = conflicts.is_empty();

        // Cache the composed schema if successful
        if success {
            let mut cached = self.composed.write().await;
            *cached = Some(composed.clone());
        }

        Ok(CompositionResult {
            schema: Some(composed),
            conflicts,
            warnings,
            success,
        })
    }

    /// Validate composed schema
    async fn validate_composed_schema(&self, schema: &ComposedSchema) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check if Query type exists
        if !schema.types.contains_key(&schema.query_type) {
            warnings.push(format!("Query type '{}' not found", schema.query_type));
        }

        // Check if Mutation type exists (if specified)
        if let Some(mutation_type) = &schema.mutation_type {
            if !schema.types.contains_key(mutation_type) {
                warnings.push(format!("Mutation type '{}' not found", mutation_type));
            }
        }

        // Check if Subscription type exists (if specified)
        if let Some(subscription_type) = &schema.subscription_type {
            if !schema.types.contains_key(subscription_type) {
                warnings.push(format!(
                    "Subscription type '{}' not found",
                    subscription_type
                ));
            }
        }

        // Validate field types reference existing types
        for (type_name, type_def) in &schema.types {
            for field in &type_def.fields {
                let field_type = self.extract_base_type(&field.field_type);
                if !self.is_built_in_type(&field_type) && !schema.types.contains_key(&field_type) {
                    warnings.push(format!(
                        "Field '{}.{}' references unknown type '{}'",
                        type_name, field.name, field_type
                    ));
                }
            }
        }

        warnings
    }

    /// Extract base type from GraphQL type notation
    fn extract_base_type(&self, type_str: &str) -> String {
        type_str
            .trim_end_matches('!')
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_end_matches('!')
            .to_string()
    }

    /// Check if a type is a built-in GraphQL type
    fn is_built_in_type(&self, type_name: &str) -> bool {
        matches!(type_name, "String" | "Int" | "Float" | "Boolean" | "ID")
    }

    /// Get the current composed schema
    pub async fn get_composed_schema(&self) -> Option<ComposedSchema> {
        let composed = self.composed.read().await;
        composed.clone()
    }

    /// List registered subgraphs
    pub async fn list_subgraphs(&self) -> Vec<String> {
        let schemas = self.schemas.read().await;
        schemas.keys().cloned().collect()
    }

    /// Get a specific subgraph schema
    pub async fn get_subgraph(&self, name: &str) -> Option<SubgraphSchema> {
        let schemas = self.schemas.read().await;
        schemas.get(name).cloned()
    }

    /// Set metadata
    pub async fn set_metadata(&self, key: String, value: String) -> Result<()> {
        let mut metadata = self.metadata.write().await;
        metadata.insert(key, value);
        Ok(())
    }

    /// Get metadata
    pub async fn get_metadata(&self, key: &str) -> Option<String> {
        let metadata = self.metadata.read().await;
        metadata.get(key).cloned()
    }
}

impl Default for AutomaticSchemaComposer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_definition_creation() {
        let field =
            FieldDefinition::new("id".to_string(), "ID!".to_string(), "service1".to_string());
        assert_eq!(field.name, "id");
        assert_eq!(field.field_type, "ID!");
        assert_eq!(field.source, "service1");
    }

    #[test]
    fn test_field_compatibility() {
        let field1 =
            FieldDefinition::new("name".to_string(), "String!".to_string(), "s1".to_string());
        let field2 =
            FieldDefinition::new("name".to_string(), "String!".to_string(), "s2".to_string());
        let field3 = FieldDefinition::new("name".to_string(), "Int!".to_string(), "s3".to_string());

        assert!(field1.is_compatible_with(&field2));
        assert!(!field1.is_compatible_with(&field3));
    }

    #[test]
    fn test_type_definition_creation() {
        let type_def =
            TypeDefinition::new("User".to_string(), TypeKind::Object, "service1".to_string());
        assert_eq!(type_def.name, "User");
        assert_eq!(type_def.kind, TypeKind::Object);
        assert!(type_def.sources.contains("service1"));
    }

    #[test]
    fn test_type_merge_success() {
        let mut type1 =
            TypeDefinition::new("User".to_string(), TypeKind::Object, "s1".to_string()).with_field(
                FieldDefinition::new("id".to_string(), "ID!".to_string(), "s1".to_string()),
            );

        let type2 =
            TypeDefinition::new("User".to_string(), TypeKind::Object, "s2".to_string()).with_field(
                FieldDefinition::new("name".to_string(), "String!".to_string(), "s2".to_string()),
            );

        let result = type1.merge(type2);
        assert!(result.is_ok());
        assert_eq!(type1.fields.len(), 2);
        assert_eq!(type1.sources.len(), 2);
    }

    #[test]
    fn test_type_merge_conflict() {
        let mut type1 = TypeDefinition::new("User".to_string(), TypeKind::Object, "s1".to_string());
        let type2 = TypeDefinition::new("Product".to_string(), TypeKind::Object, "s2".to_string());

        let result = type1.merge(type2);
        assert!(result.is_err());
    }

    #[test]
    fn test_subgraph_schema() {
        let mut schema = SubgraphSchema::new("service1".to_string());
        let type_def =
            TypeDefinition::new("User".to_string(), TypeKind::Object, "service1".to_string());
        schema.add_type(type_def);

        assert!(schema.get_type("User").is_some());
        assert!(schema.get_type("Product").is_none());
    }

    #[test]
    fn test_composed_schema_sdl_generation() {
        let mut composed = ComposedSchema::new();
        let type_def = TypeDefinition::new("User".to_string(), TypeKind::Object, "s1".to_string())
            .with_field(FieldDefinition::new(
                "id".to_string(),
                "ID!".to_string(),
                "s1".to_string(),
            ))
            .with_field(FieldDefinition::new(
                "name".to_string(),
                "String!".to_string(),
                "s1".to_string(),
            ));

        composed.types.insert("User".to_string(), type_def);

        let sdl = composed.to_sdl();
        assert!(sdl.contains("type User"));
        assert!(sdl.contains("id: ID!"));
        assert!(sdl.contains("name: String!"));
    }

    #[tokio::test]
    async fn test_composer_register_subgraph() {
        let composer = AutomaticSchemaComposer::new();
        let schema = SubgraphSchema::new("service1".to_string());

        composer.register_subgraph(schema).await.unwrap();
        let subgraphs = composer.list_subgraphs().await;
        assert_eq!(subgraphs.len(), 1);
        assert!(subgraphs.contains(&"service1".to_string()));
    }

    #[tokio::test]
    async fn test_composer_unregister_subgraph() {
        let composer = AutomaticSchemaComposer::new();
        let schema = SubgraphSchema::new("service1".to_string());

        composer.register_subgraph(schema).await.unwrap();
        composer.unregister_subgraph("service1").await.unwrap();

        let subgraphs = composer.list_subgraphs().await;
        assert_eq!(subgraphs.len(), 0);
    }

    #[tokio::test]
    async fn test_compose_empty_schemas() {
        let composer = AutomaticSchemaComposer::new();
        let result = composer.compose().await.unwrap();

        assert!(!result.success);
        assert!(result.schema.is_none());
        assert!(!result.warnings.is_empty());
    }

    #[tokio::test]
    async fn test_compose_single_schema() {
        let composer = AutomaticSchemaComposer::new();
        let mut schema = SubgraphSchema::new("service1".to_string());
        let type_def = TypeDefinition::new(
            "Query".to_string(),
            TypeKind::Object,
            "service1".to_string(),
        )
        .with_field(FieldDefinition::new(
            "hello".to_string(),
            "String!".to_string(),
            "service1".to_string(),
        ));
        schema.add_type(type_def);

        composer.register_subgraph(schema).await.unwrap();
        let result = composer.compose().await.unwrap();

        assert!(result.success);
        assert!(result.schema.is_some());
        assert!(result.conflicts.is_empty());
    }

    #[tokio::test]
    async fn test_compose_multiple_schemas() {
        let composer = AutomaticSchemaComposer::new();

        let mut schema1 = SubgraphSchema::new("service1".to_string());
        let type1 = TypeDefinition::new(
            "Query".to_string(),
            TypeKind::Object,
            "service1".to_string(),
        )
        .with_field(FieldDefinition::new(
            "user".to_string(),
            "User".to_string(),
            "service1".to_string(),
        ));
        schema1.add_type(type1);

        let mut schema2 = SubgraphSchema::new("service2".to_string());
        let type2 = TypeDefinition::new(
            "Query".to_string(),
            TypeKind::Object,
            "service2".to_string(),
        )
        .with_field(FieldDefinition::new(
            "product".to_string(),
            "Product".to_string(),
            "service2".to_string(),
        ));
        schema2.add_type(type2);

        composer.register_subgraph(schema1).await.unwrap();
        composer.register_subgraph(schema2).await.unwrap();

        let result = composer.compose().await.unwrap();
        assert!(result.success);

        let schema = result.schema.unwrap();
        let query_type = schema.types.get("Query").unwrap();
        assert_eq!(query_type.fields.len(), 2);
    }

    #[tokio::test]
    async fn test_get_composed_schema() {
        let composer = AutomaticSchemaComposer::new();
        let mut schema = SubgraphSchema::new("service1".to_string());
        let type_def = TypeDefinition::new(
            "Query".to_string(),
            TypeKind::Object,
            "service1".to_string(),
        );
        schema.add_type(type_def);

        composer.register_subgraph(schema).await.unwrap();
        composer.compose().await.unwrap();

        let composed = composer.get_composed_schema().await;
        assert!(composed.is_some());
    }

    #[tokio::test]
    async fn test_metadata() {
        let composer = AutomaticSchemaComposer::new();
        composer
            .set_metadata("version".to_string(), "1.0.0".to_string())
            .await
            .unwrap();

        let value = composer.get_metadata("version").await;
        assert_eq!(value, Some("1.0.0".to_string()));
    }

    #[tokio::test]
    async fn test_get_subgraph() {
        let composer = AutomaticSchemaComposer::new();
        let schema = SubgraphSchema::new("service1".to_string());
        composer.register_subgraph(schema).await.unwrap();

        let retrieved = composer.get_subgraph("service1").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "service1");
    }

    #[test]
    fn test_extract_base_type() {
        let composer = AutomaticSchemaComposer::new();
        assert_eq!(composer.extract_base_type("String!"), "String");
        assert_eq!(composer.extract_base_type("[String!]!"), "String");
        assert_eq!(composer.extract_base_type("User"), "User");
    }

    #[test]
    fn test_is_built_in_type() {
        let composer = AutomaticSchemaComposer::new();
        assert!(composer.is_built_in_type("String"));
        assert!(composer.is_built_in_type("Int"));
        assert!(composer.is_built_in_type("ID"));
        assert!(!composer.is_built_in_type("User"));
    }
}
