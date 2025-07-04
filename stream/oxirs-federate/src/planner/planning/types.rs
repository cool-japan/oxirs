//! Type definitions for federated query planning
//!
//! This module contains all the data structures, enums, and type definitions
//! used throughout the federated query planning system.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use uuid::Uuid;

/// Execution context for query processing
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub query_id: String,
    pub execution_id: String,
    pub start_time: std::time::Instant,
    pub timeout: Option<Duration>,
    pub variables: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            query_id: Uuid::new_v4().to_string(),
            execution_id: Uuid::new_v4().to_string(),
            start_time: std::time::Instant::now(),
            timeout: Some(Duration::from_secs(30)),
            variables: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Historical performance data for optimization
#[derive(Debug, Clone)]
pub struct HistoricalPerformance {
    pub query_patterns: HashMap<String, f64>,
    pub service_performance: HashMap<String, f64>,
    pub join_performance: HashMap<String, f64>,
    pub avg_response_times: HashMap<String, Duration>,
}

/// Analysis result for query reoptimization
#[derive(Debug, Clone)]
pub struct ReoptimizationAnalysis {
    pub should_reoptimize: bool,
    pub performance_degradation: f64,
    pub suggested_changes: Vec<String>,
    pub confidence_score: f64,
}

/// Configuration for GraphQL federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLFederationConfig {
    pub enable_schema_stitching: bool,
    pub type_conflict_resolution: TypeConflictResolution,
    pub field_conflict_resolution: FieldConflictResolution,
    pub field_merge_strategy: FieldMergeStrategy,
    pub enable_query_planning: bool,
    pub enable_entity_resolution: bool,
    pub allow_breaking_changes: bool,
}

impl Default for GraphQLFederationConfig {
    fn default() -> Self {
        Self {
            enable_schema_stitching: true,
            type_conflict_resolution: TypeConflictResolution::Merge,
            field_conflict_resolution: FieldConflictResolution::Namespace,
            field_merge_strategy: FieldMergeStrategy::Union,
            enable_query_planning: true,
            enable_entity_resolution: true,
            allow_breaking_changes: false,
        }
    }
}

/// Strategies for resolving type conflicts
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TypeConflictResolution {
    Error,
    Merge,
    ServicePriority,
}

/// Strategies for resolving field conflicts
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FieldConflictResolution {
    Error,
    Namespace,
    FirstWins,
}

/// Strategies for merging fields
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FieldMergeStrategy {
    Union,
    Override,
}

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
    pub selection_set: Vec<GraphQLField>,
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

/// Represents a federated GraphQL schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSchema {
    pub service_id: String,
    pub types: HashMap<String, TypeDefinition>,
    pub queries: HashMap<String, FieldDefinition>,
    pub mutations: HashMap<String, FieldDefinition>,
    pub subscriptions: HashMap<String, FieldDefinition>,
    pub directives: HashMap<String, DirectiveDefinition>,
}

/// GraphQL type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefinition {
    pub name: String,
    pub description: Option<String>,
    pub kind: TypeKind,
    pub directives: Vec<DirectiveUsage>,
}

/// Kinds of GraphQL types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeKind {
    Scalar,
    Object {
        fields: HashMap<String, FieldDefinition>,
    },
    Interface {
        fields: HashMap<String, FieldDefinition>,
    },
    Union {
        types: Vec<String>,
    },
    Enum {
        values: Vec<String>,
    },
    InputObject {
        fields: HashMap<String, InputFieldDefinition>,
    },
}

/// GraphQL field definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub description: Option<String>,
    pub field_type: String,
    pub arguments: HashMap<String, InputFieldDefinition>,
    pub directives: Vec<DirectiveUsage>,
}

/// GraphQL input field definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputFieldDefinition {
    pub name: String,
    pub field_type: String,
    pub default_value: Option<serde_json::Value>,
    pub description: Option<String>,
}

/// GraphQL directive definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectiveDefinition {
    pub name: String,
    pub description: Option<String>,
    pub locations: Vec<DirectiveLocation>,
    pub arguments: HashMap<String, InputFieldDefinition>,
}

/// GraphQL directive usage
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectiveUsage {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

/// GraphQL directive locations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirectiveLocation {
    Query,
    Mutation,
    Subscription,
    Field,
    FragmentDefinition,
    FragmentSpread,
    InlineFragment,
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

/// Unified schema from multiple federated schemas
#[derive(Debug, Clone)]
pub struct UnifiedSchema {
    pub types: HashMap<String, TypeDefinition>,
    pub queries: HashMap<String, FieldDefinition>,
    pub mutations: HashMap<String, FieldDefinition>,
    pub subscriptions: HashMap<String, FieldDefinition>,
    pub directives: HashMap<String, DirectiveDefinition>,
    pub schema_mapping: HashMap<String, Vec<String>>, // Type -> Services
}

/// Query targeted at a specific service
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

/// GraphQL response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLResponse {
    pub data: serde_json::Value,
    pub errors: Vec<GraphQLError>,
    pub extensions: Option<serde_json::Value>,
}

/// GraphQL error structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLError {
    pub message: String,
    pub locations: Option<Vec<GraphQLLocation>>,
    pub path: Option<Vec<serde_json::Value>>,
    pub extensions: Option<serde_json::Value>,
}

/// GraphQL error location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLLocation {
    pub line: u32,
    pub column: u32,
}

/// GraphQL operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphQLOperationType {
    Query,
    Mutation,
    Subscription,
}

/// Parsed GraphQL query
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    pub operation_type: GraphQLOperationType,
    pub operation_name: Option<String>,
    pub selection_set: Vec<Selection>,
    pub variables: HashMap<String, serde_json::Value>,
}

/// GraphQL selection (field)
#[derive(Debug, Clone)]
pub struct Selection {
    pub name: String,
    pub alias: Option<String>,
    pub arguments: HashMap<String, serde_json::Value>,
    pub selection_set: Vec<Selection>,
}

/// Field ownership mapping
#[derive(Debug, Clone)]
pub struct FieldOwnership {
    pub field_to_service: HashMap<String, String>,
    pub service_to_fields: HashMap<String, Vec<String>>,
}

/// Apollo Federation directives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationDirectives {
    /// @key directive for entity identification
    pub key: Option<KeyDirective>,
    /// @external directive for fields owned by other services
    pub external: bool,
    /// @requires directive for field dependencies
    pub requires: Option<String>,
    /// @provides directive for field guarantees
    pub provides: Option<String>,
    /// @extends directive for extending types
    pub extends: bool,
    /// @shareable directive for fields that can be resolved by multiple services
    pub shareable: bool,
    /// @override directive for taking ownership of fields
    pub override_from: Option<String>,
    /// @inaccessible directive for hiding fields
    pub inaccessible: bool,
    /// @tag directive for metadata
    pub tags: Vec<String>,
}

/// Apollo Federation @key directive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDirective {
    pub fields: String,
    pub resolvable: bool,
}

/// Entity representation for Apollo Federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRepresentation {
    #[serde(rename = "__typename")]
    pub typename: String,
    #[serde(flatten)]
    pub fields: serde_json::Value,
}

/// Apollo Federation service info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationServiceInfo {
    /// Service SDL (Schema Definition Language)
    pub sdl: String,
    /// Service capabilities
    pub capabilities: FederationCapabilities,
    /// Entity types this service can resolve
    pub entity_types: Vec<String>,
}

/// Apollo Federation capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationCapabilities {
    pub federation_version: String,
    pub supports_entities: bool,
    pub supports_entity_interfaces: bool,
    pub supports_progressive_override: bool,
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

/// Types of queries for federation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryType {
    Select,
    Construct,
    Ask,
    Describe,
    Update,
    Sparql,
    GraphQL,
}

/// Query information for caching and optimization
#[derive(Debug, Clone)]
pub struct QueryInfo {
    pub query_type: QueryType,
    pub original_query: String,
    pub patterns: Vec<TriplePattern>,
    pub variables: HashSet<String>,
    pub filters: Vec<FilterExpression>,
    pub complexity: u64,
    pub estimated_cost: u64,
}

/// Triple pattern for SPARQL queries
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub pattern_string: String,
}

/// Filter expression for query optimization
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
pub struct FilterExpression {
    pub expression: String,
    pub variables: Vec<String>,
}

/// Execution plan for federated queries
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub query_id: String,
    pub steps: Vec<ExecutionStep>,
    pub estimated_total_cost: f64,
    pub max_parallelism: usize,
    pub planning_time: Duration,
    pub cache_key: Option<String>,
    pub metadata: HashMap<String, String>,
    pub parallelizable_steps: Vec<Vec<String>>,
}

/// Individual step in an execution plan
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub step_id: String,
    pub step_type: StepType,
    pub service_id: Option<String>,
    pub query_fragment: String,
    pub dependencies: Vec<String>,
    pub estimated_cost: f64,
    pub timeout: Duration,
    pub retry_config: Option<RetryConfig>,
}

/// Types of execution steps
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepType {
    ServiceQuery,
    GraphQLQuery,
    Join,
    Union,
    Filter,
    SchemaStitch,
    Aggregate,
    Sort,
    EntityResolution,
    ResultStitching,
}

impl std::fmt::Display for StepType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepType::ServiceQuery => write!(f, "ServiceQuery"),
            StepType::GraphQLQuery => write!(f, "GraphQLQuery"),
            StepType::Join => write!(f, "Join"),
            StepType::Union => write!(f, "Union"),
            StepType::Filter => write!(f, "Filter"),
            StepType::SchemaStitch => write!(f, "SchemaStitch"),
            StepType::Aggregate => write!(f, "Aggregate"),
            StepType::Sort => write!(f, "Sort"),
            StepType::EntityResolution => write!(f, "EntityResolution"),
            StepType::ResultStitching => write!(f, "ResultStitching"),
        }
    }
}

/// Retry configuration for execution steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}
