//! Type definitions for GraphQL Federation
//!
//! This module contains all the type definitions, enums, and data structures
//! used throughout the GraphQL federation system.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{executor::GraphQLResponse, QueryResultData, StepResult};

/// Entity data returned from federated services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityData {
    pub typename: String,
    pub fields: serde_json::Map<String, serde_json::Value>,
}

/// Composed schema for federation
#[derive(Debug, Clone)]
pub struct ComposedSchema {
    pub types: HashMap<String, GraphQLType>,
    pub query_type: String,
    pub mutation_type: Option<String>,
    pub subscription_type: Option<String>,
    pub directives: Vec<String>,
    pub entity_types: HashMap<String, EntityTypeInfo>,
    pub field_ownership: HashMap<String, FieldOwnershipType>,
}

/// GraphQL type definition
#[derive(Debug, Clone)]
pub struct GraphQLType {
    pub name: String,
    pub kind: GraphQLTypeKind,
    pub fields: HashMap<String, GraphQLField>,
}

/// GraphQL type kinds
#[derive(Debug, Clone)]
pub enum GraphQLTypeKind {
    Object,
    Interface,
    Union,
    Enum,
    InputObject,
    Scalar,
}

/// GraphQL field definition
#[derive(Debug, Clone)]
pub struct GraphQLField {
    pub name: String,
    pub field_type: String,
    pub arguments: HashMap<String, GraphQLArgument>,
}

/// GraphQL argument definition
#[derive(Debug, Clone)]
pub struct GraphQLArgument {
    pub name: String,
    pub argument_type: String,
    pub default_value: Option<serde_json::Value>,
}

/// Entity type information for federation
#[derive(Debug, Clone)]
pub struct EntityTypeInfo {
    pub key_fields: Vec<String>,
    pub owning_service: String,
    pub extending_services: Vec<String>,
}

/// Field ownership types for federation
#[derive(Debug, Clone)]
pub enum FieldOwnershipType {
    Owned(String),
    External,
    Requires(Vec<String>),
    Provides(Vec<String>),
}

/// GraphQL directive
#[derive(Debug, Clone)]
pub struct Directive {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

/// Entity reference in a federated query
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EntityReference {
    pub entity_type: String,
    pub key_fields: Vec<String>,
    pub required_fields: Vec<String>,
    pub service_id: String,
}

/// Schema capabilities discovered through introspection
#[derive(Debug, Clone)]
pub struct SchemaCapabilities {
    pub supports_federation: bool,
    pub supports_subscriptions: bool,
    pub supports_defer_stream: bool,
    pub entity_types: Vec<String>,
    pub custom_directives: Vec<String>,
    pub scalar_types: Vec<String>,
    pub estimated_complexity: f64,
}

/// Result of dynamic schema update
#[derive(Debug, Clone)]
pub struct SchemaUpdateResult {
    pub service_id: String,
    pub update_successful: bool,
    pub breaking_changes: Vec<BreakingChange>,
    pub warnings: Vec<String>,
    pub rollback_available: bool,
}

/// Breaking change detected during schema update
#[derive(Debug, Clone)]
pub struct BreakingChange {
    pub change_type: BreakingChangeType,
    pub description: String,
    pub severity: BreakingChangeSeverity,
}

/// Types of breaking changes
#[derive(Debug, Clone)]
pub enum BreakingChangeType {
    TypeRemoved,
    FieldRemoved,
    ArgumentMadeRequired,
    RequiredArgumentAdded,
    TypeChanged,
    DirectiveRemoved,
}

/// Severity of breaking changes
#[derive(Debug, Clone)]
pub enum BreakingChangeSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// GraphQL type definition with federation support
#[derive(Debug, Clone)]
pub struct GraphQLTypeDefinition {
    pub name: String,
    pub kind: String,
    pub fields: HashMap<String, GraphQLFieldDefinition>,
    pub directives: Vec<Directive>,
}

/// GraphQL field definition with federation support
#[derive(Debug, Clone)]
pub struct GraphQLFieldDefinition {
    pub name: String,
    pub field_type: String,
    pub arguments: HashMap<String, GraphQLArgument>,
    pub directives: Vec<Directive>,
}

/// Entity resolution context
#[derive(Debug, Clone)]
pub struct ResolutionContext {
    pub request_id: String,
    pub user_id: Option<String>,
    pub headers: HashMap<String, String>,
    pub timeout: std::time::Duration,
}

/// Entity dependency graph for resolution planning
#[derive(Debug, Clone)]
pub struct EntityDependencyGraph {
    pub nodes: HashMap<EntityReference, usize>,
    pub edges: Vec<(usize, usize)>,
}

/// Entity resolution plan for federation
#[derive(Debug, Clone)]
pub struct EntityResolutionPlan {
    pub steps: Vec<EntityResolutionStep>,
    pub dependencies: HashMap<String, Vec<String>>,
}

/// A step in entity resolution
#[derive(Debug, Clone)]
pub struct EntityResolutionStep {
    pub service_name: String,
    pub entity_type: String,
    pub key_fields: Vec<String>,
    pub query: String,
    pub depends_on: Vec<String>,
}

/// Resolved entity data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedEntity {
    pub entity_type: String,
    pub key_values: HashMap<String, serde_json::Value>,
    pub data: serde_json::Value,
    pub service_name: String,
}

/// GraphQL federation manager
#[derive(Debug)]
pub struct GraphQLFederation {
    pub schemas: Arc<RwLock<HashMap<String, FederatedSchema>>>,
    pub config: GraphQLFederationConfig,
}

/// Federated schema definition
#[derive(Debug, Clone)]
pub struct FederatedSchema {
    pub service_id: String,
    pub types: HashMap<String, TypeDefinition>,
    pub queries: HashMap<String, FieldDefinition>,
    pub mutations: HashMap<String, FieldDefinition>,
    pub subscriptions: HashMap<String, FieldDefinition>,
    pub directives: HashMap<String, DirectiveDefinition>,
}

/// Unified schema combining all federated schemas
#[derive(Debug, Clone)]
pub struct UnifiedSchema {
    pub types: HashMap<String, TypeDefinition>,
    pub queries: HashMap<String, FieldDefinition>,
    pub mutations: HashMap<String, FieldDefinition>,
    pub subscriptions: HashMap<String, FieldDefinition>,
    pub directives: HashMap<String, DirectiveDefinition>,
    pub schema_mapping: HashMap<String, Vec<String>>,
}

/// Type definition in schema
#[derive(Debug, Clone, PartialEq)]
pub struct TypeDefinition {
    pub name: String,
    pub description: Option<String>,
    pub kind: TypeKind,
    pub directives: Vec<Directive>,
}

/// Different kinds of GraphQL types
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Object {
        fields: HashMap<String, FieldDefinition>,
    },
    Interface {
        fields: HashMap<String, FieldDefinition>,
    },
    Union {
        possible_types: Vec<String>,
    },
    Enum {
        values: Vec<EnumValueDefinition>,
    },
    InputObject {
        fields: HashMap<String, InputFieldDefinition>,
    },
    Scalar,
}

/// Field definition in schema
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDefinition {
    pub name: String,
    pub description: Option<String>,
    pub field_type: String,
    pub arguments: HashMap<String, ArgumentDefinition>,
    pub directives: Vec<Directive>,
}

/// Argument definition for fields
#[derive(Debug, Clone, PartialEq)]
pub struct ArgumentDefinition {
    pub name: String,
    pub description: Option<String>,
    pub argument_type: String,
    pub default_value: Option<serde_json::Value>,
    pub directives: Vec<Directive>,
}

/// Input field definition
#[derive(Debug, Clone, PartialEq)]
pub struct InputFieldDefinition {
    pub name: String,
    pub description: Option<String>,
    pub field_type: String,
    pub default_value: Option<serde_json::Value>,
    pub directives: Vec<Directive>,
}

/// Enum value definition
#[derive(Debug, Clone, PartialEq)]
pub struct EnumValueDefinition {
    pub name: String,
    pub description: Option<String>,
    pub deprecated: bool,
    pub deprecation_reason: Option<String>,
    pub directives: Vec<Directive>,
}

/// Directive definition
#[derive(Debug, Clone, PartialEq)]
pub struct DirectiveDefinition {
    pub name: String,
    pub description: Option<String>,
    pub locations: Vec<DirectiveLocation>,
    pub arguments: HashMap<String, ArgumentDefinition>,
    pub repeatable: bool,
}

/// Valid locations for directives
#[derive(Debug, Clone, PartialEq)]
pub enum DirectiveLocation {
    Query,
    Mutation,
    Subscription,
    Field,
    FragmentDefinition,
    FragmentSpread,
    InlineFragment,
    VariableDefinition,
    Schema,
    Scalar,
    Object,
    FieldDefinition,
    ArgumentDefinition,
    Interface,
    Union,
    Enum,
    EnumValue,
    InputObject,
    InputFieldDefinition,
}

/// GraphQL federation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLFederationConfig {
    pub enable_schema_stitching: bool,
    pub enable_query_planning: bool,
    pub enable_entity_resolution: bool,
    pub max_query_depth: usize,
    pub max_query_complexity: usize,
    pub enable_introspection: bool,
    pub enable_subscriptions: bool,
    pub type_conflict_resolution: TypeConflictResolution,
    pub field_conflict_resolution: FieldConflictResolution,
    pub field_merge_strategy: FieldMergeStrategy,
}

impl Default for GraphQLFederationConfig {
    fn default() -> Self {
        Self {
            enable_schema_stitching: true,
            enable_query_planning: true,
            enable_entity_resolution: true,
            max_query_depth: 15,
            max_query_complexity: 1000,
            enable_introspection: true,
            enable_subscriptions: false,
            type_conflict_resolution: TypeConflictResolution::Error,
            field_conflict_resolution: FieldConflictResolution::Error,
            field_merge_strategy: FieldMergeStrategy::Union,
        }
    }
}

/// How to resolve type conflicts in federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeConflictResolution {
    Error,
    Merge,
    ServicePriority,
}

/// How to resolve field conflicts in federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldConflictResolution {
    Error,
    Namespace,
    FirstWins,
}

/// Strategy for merging fields within types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldMergeStrategy {
    Union,
    Override,
}

/// Parsed GraphQL query structure
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    pub operation_type: GraphQLOperationType,
    pub operation_name: Option<String>,
    pub selection_set: Vec<Selection>,
    pub variables: HashMap<String, VariableDefinition>,
}

/// GraphQL operation types
#[derive(Debug, Clone)]
pub enum GraphQLOperationType {
    Query,
    Mutation,
    Subscription,
}

/// Selection in a GraphQL query
#[derive(Debug, Clone)]
pub struct Selection {
    pub name: String,
    pub alias: Option<String>,
    pub arguments: HashMap<String, serde_json::Value>,
    pub selection_set: Vec<Selection>,
}

/// Variable definition in GraphQL query
#[derive(Debug, Clone)]
pub struct VariableDefinition {
    pub name: String,
    pub variable_type: String,
    pub default_value: Option<serde_json::Value>,
}

/// Field ownership analysis result
#[derive(Debug, Clone)]
pub struct FieldOwnership {
    pub field_to_service: HashMap<String, String>,
    pub service_to_fields: HashMap<String, Vec<String>>,
}

/// Service-specific query
#[derive(Debug, Clone)]
pub struct ServiceQuery {
    pub service_id: String,
    pub query: String,
    pub variables: Option<serde_json::Value>,
}

/// Result from a service query
#[derive(Debug, Clone)]
pub struct ServiceResult {
    pub service_id: String,
    pub response: GraphQLResponse,
}

/// Federation service information from introspection
#[derive(Debug, Clone)]
pub struct FederationServiceInfo {
    pub sdl: String,
    pub capabilities: FederationCapabilities,
    pub entity_types: Vec<String>,
}

/// Federation capabilities of a service
#[derive(Debug, Clone)]
pub struct FederationCapabilities {
    pub federation_version: String,
    pub supports_entities: bool,
    pub supports_entity_interfaces: bool,
    pub supports_progressive_override: bool,
}

/// Entity representation for federated resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRepresentation {
    pub typename: String,
    pub fields: serde_json::Value,
}

/// Federation directives parsed from SDL
#[derive(Debug, Clone)]
pub struct FederationDirectives {
    pub key: Option<String>,
    pub external: bool,
    pub requires: Option<String>,
    pub provides: Option<String>,
    pub extends: bool,
    pub shareable: bool,
    pub override_from: Option<String>,
    pub inaccessible: bool,
    pub tags: Vec<String>,
}

/// Query complexity analysis result
#[derive(Debug, Clone)]
pub struct QueryComplexity {
    pub total_complexity: usize,
    pub max_depth: usize,
    pub field_count: usize,
    pub estimated_execution_time: std::time::Duration,
}

/// Schema validation error
#[derive(Debug, Clone)]
pub struct SchemaValidationError {
    pub error_type: SchemaErrorType,
    pub message: String,
    pub location: Option<String>,
}

/// Types of schema validation errors
#[derive(Debug, Clone)]
pub enum SchemaErrorType {
    TypeNotFound,
    FieldNotFound,
    CircularDependency,
    InvalidDirective,
    ConflictingDefinitions,
}