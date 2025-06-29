//! GraphQL Federation and Schema Stitching
//!
//! This module provides GraphQL federation capabilities, including schema stitching,
//! query planning, and distributed execution across multiple GraphQL services.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{executor::GraphQLResponse, planner::ExecutionPlan, QueryResultData, StepResult};

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
    schemas: Arc<RwLock<HashMap<String, FederatedSchema>>>,
    config: GraphQLFederationConfig,
}

impl GraphQLFederation {
    /// Create a new GraphQL federation manager
    pub fn new() -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            config: GraphQLFederationConfig::default(),
        }
    }

    /// Create a new GraphQL federation manager with custom configuration
    pub fn with_config(config: GraphQLFederationConfig) -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register a GraphQL schema for federation
    pub async fn register_schema(&self, service_id: String, schema: FederatedSchema) -> Result<()> {
        info!("Registering GraphQL schema for service: {}", service_id);

        let mut schemas = self.schemas.write().await;
        schemas.insert(service_id, schema);
        Ok(())
    }

    /// Unregister a GraphQL schema
    pub async fn unregister_schema(&self, service_id: &str) -> Result<()> {
        info!("Unregistering GraphQL schema for service: {}", service_id);

        let mut schemas = self.schemas.write().await;
        schemas
            .remove(service_id)
            .ok_or_else(|| anyhow!("Schema not found for service: {}", service_id))?;
        Ok(())
    }

    /// Execute a federated GraphQL query plan
    pub async fn execute_federated(&self, plan: &ExecutionPlan) -> Result<Vec<StepResult>> {
        info!(
            "Executing federated GraphQL plan with {} steps",
            plan.steps.len()
        );

        let mut results = Vec::new();
        let mut completed_steps: HashMap<String, StepResult> = HashMap::new();

        // Execute steps in dependency order
        let execution_order = self.calculate_execution_order(plan)?;

        for step_id in execution_order {
            if let Some(step) = plan.steps.iter().find(|s| s.step_id == step_id) {
                let step_result = self.execute_graphql_step(step, &completed_steps).await?;
                completed_steps.insert(step_id.clone(), step_result.clone());
                results.push(step_result);
            }
        }

        Ok(results)
    }

    /// Calculate the order of step execution based on dependencies
    fn calculate_execution_order(&self, plan: &ExecutionPlan) -> Result<Vec<String>> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        // Topological sort
        for step in &plan.steps {
            if !visited.contains(&step.step_id) {
                self.visit_step(&step.step_id, plan, &mut visited, &mut visiting, &mut order)?;
            }
        }

        Ok(order)
    }

    /// Recursive helper for topological sort
    fn visit_step(
        &self,
        step_id: &str,
        plan: &ExecutionPlan,
        visited: &mut HashSet<String>,
        visiting: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) -> Result<()> {
        if visiting.contains(step_id) {
            return Err(anyhow!("Circular dependency detected in execution plan"));
        }

        if visited.contains(step_id) {
            return Ok(());
        }

        visiting.insert(step_id.to_string());

        if let Some(step) = plan.steps.iter().find(|s| s.step_id == step_id) {
            for dep_id in &step.dependencies {
                self.visit_step(dep_id, plan, visited, visiting, order)?;
            }
        }

        visiting.remove(step_id);
        visited.insert(step_id.to_string());
        order.push(step_id.to_string());

        Ok(())
    }

    /// Execute a single GraphQL step
    async fn execute_graphql_step(
        &self,
        step: &crate::ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        use std::time::Instant;
        let start_time = Instant::now();

        debug!(
            "Executing GraphQL step: {} ({})",
            step.step_id, step.step_type
        );

        let result = match step.step_type {
            crate::StepType::GraphQLQuery => {
                self.execute_graphql_query_step(step, completed_steps).await
            }
            crate::StepType::SchemaStitch => {
                self.execute_schema_stitch_step(step, completed_steps).await
            }
            _ => {
                // For non-GraphQL steps, return a success result
                Ok(QueryResultData::GraphQL(GraphQLResponse {
                    data: serde_json::Value::Null,
                    errors: Vec::new(),
                    extensions: None,
                }))
            }
        };

        let execution_time = start_time.elapsed();

        let (status, data, error) = match result {
            Ok(query_data) => (
                crate::executor::ExecutionStatus::Success,
                Some(query_data),
                None,
            ),
            Err(err) => (
                crate::executor::ExecutionStatus::Failed,
                None,
                Some(err.to_string()),
            ),
        };

        let result_size = data.as_ref().map(|d| d.estimated_size()).unwrap_or(0);

        // Calculate memory usage before moving data
        let memory_used = data.as_ref().map(|d| self.estimate_memory_usage(d, result_size)).unwrap_or(0);

        Ok(StepResult {
            step_id: step.step_id.clone(),
            step_type: step.step_type,
            status,
            data,
            error: error.clone(),
            execution_time,
            service_id: step.service_id.clone(),
            memory_used,
            result_size,
            success: matches!(status, crate::executor::ExecutionStatus::Success),
            error_message: error,
            service_response_time: execution_time, // TODO: Track service-specific response time
            cache_hit: false,                      // TODO: Implement cache hit tracking
        })
    }

    /// Execute a GraphQL query step
    async fn execute_graphql_query_step(
        &self,
        step: &crate::ExecutionStep,
        _completed_steps: &HashMap<String, StepResult>,
    ) -> Result<QueryResultData> {
        debug!("Executing GraphQL query: {}", step.query_fragment);

        // In a real implementation, this would:
        // 1. Send the query to the appropriate GraphQL service
        // 2. Handle authentication and headers
        // 3. Parse and validate the response
        // 4. Apply any necessary transformations

        // For now, return a mock successful response
        let mock_response = GraphQLResponse {
            data: serde_json::json!({
                "result": "GraphQL query executed successfully",
                "service": step.service_id.as_ref().unwrap_or(&"unknown".to_string()),
                "query": step.query_fragment
            }),
            errors: Vec::new(),
            extensions: None,
        };

        Ok(QueryResultData::GraphQL(mock_response))
    }

    /// Execute a schema stitching step
    async fn execute_schema_stitch_step(
        &self,
        step: &crate::ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<QueryResultData> {
        debug!("Executing schema stitch step: {}", step.step_id);

        // Collect all GraphQL results from dependencies
        let mut service_results = Vec::new();

        for dep_id in &step.dependencies {
            if let Some(dep_result) = completed_steps.get(dep_id) {
                if let Some(QueryResultData::GraphQL(graphql_response)) = &dep_result.data {
                    service_results.push(ServiceResult {
                        service_id: dep_result.service_id.clone().unwrap_or_default(),
                        response: graphql_response.clone(),
                    });
                }
            }
        }

        // Stitch the results together
        let stitched_response = self.stitch_results(service_results).await?;
        Ok(QueryResultData::GraphQL(stitched_response))
    }

    /// Create a unified schema from all registered schemas
    pub async fn create_unified_schema(&self) -> Result<UnifiedSchema> {
        let schemas = self.schemas.read().await;

        if schemas.is_empty() {
            return Err(anyhow!("No schemas registered for federation"));
        }

        let mut unified = UnifiedSchema {
            types: HashMap::new(),
            queries: HashMap::new(),
            mutations: HashMap::new(),
            subscriptions: HashMap::new(),
            directives: HashMap::new(),
            schema_mapping: HashMap::new(),
        };

        // Merge all schemas
        for (service_id, schema) in schemas.iter() {
            self.merge_schema_into_unified(&mut unified, service_id, schema)?;
        }

        // Validate the unified schema
        self.validate_unified_schema(&unified)?;

        Ok(unified)
    }

    /// Merge a single schema into the unified schema
    fn merge_schema_into_unified(
        &self,
        unified: &mut UnifiedSchema,
        service_id: &str,
        schema: &FederatedSchema,
    ) -> Result<()> {
        // Merge types
        for (type_name, type_def) in &schema.types {
            if let Some(existing) = unified.types.get(type_name) {
                // Handle type conflicts
                match self.config.type_conflict_resolution {
                    TypeConflictResolution::Error => {
                        return Err(anyhow!(
                            "Type conflict: {} exists in multiple schemas",
                            type_name
                        ));
                    }
                    TypeConflictResolution::Merge => {
                        let merged_type = self.merge_type_definitions(existing, type_def)?;
                        unified.types.insert(type_name.clone(), merged_type);
                    }
                    TypeConflictResolution::ServicePriority => {
                        // Keep existing (first wins)
                    }
                }
            } else {
                unified.types.insert(type_name.clone(), type_def.clone());
            }

            // Track which service owns this type
            unified
                .schema_mapping
                .entry(type_name.clone())
                .or_insert_with(Vec::new)
                .push(service_id.to_string());
        }

        // Merge queries
        for (field_name, field_def) in &schema.queries {
            if unified.queries.contains_key(field_name) {
                match self.config.field_conflict_resolution {
                    FieldConflictResolution::Error => {
                        return Err(anyhow!(
                            "Query field conflict: {} exists in multiple schemas",
                            field_name
                        ));
                    }
                    FieldConflictResolution::Namespace => {
                        let namespaced_name = format!("{}_{}", service_id, field_name);
                        unified.queries.insert(namespaced_name, field_def.clone());
                    }
                    FieldConflictResolution::FirstWins => {
                        // Keep existing
                    }
                }
            } else {
                unified
                    .queries
                    .insert(field_name.clone(), field_def.clone());
            }
        }

        // Merge mutations
        for (field_name, field_def) in &schema.mutations {
            if unified.mutations.contains_key(field_name) {
                match self.config.field_conflict_resolution {
                    FieldConflictResolution::Error => {
                        return Err(anyhow!(
                            "Mutation field conflict: {} exists in multiple schemas",
                            field_name
                        ));
                    }
                    FieldConflictResolution::Namespace => {
                        let namespaced_name = format!("{}_{}", service_id, field_name);
                        unified.mutations.insert(namespaced_name, field_def.clone());
                    }
                    FieldConflictResolution::FirstWins => {
                        // Keep existing
                    }
                }
            } else {
                unified
                    .mutations
                    .insert(field_name.clone(), field_def.clone());
            }
        }

        // Merge subscriptions
        for (field_name, field_def) in &schema.subscriptions {
            if unified.subscriptions.contains_key(field_name) {
                warn!("Subscription field conflict: {}", field_name);
            } else {
                unified
                    .subscriptions
                    .insert(field_name.clone(), field_def.clone());
            }
        }

        Ok(())
    }

    /// Merge two type definitions
    fn merge_type_definitions(
        &self,
        existing: &TypeDefinition,
        new: &TypeDefinition,
    ) -> Result<TypeDefinition> {
        match (&existing.kind, &new.kind) {
            (
                TypeKind::Object {
                    fields: existing_fields,
                },
                TypeKind::Object { fields: new_fields },
            ) => {
                let mut merged_fields = existing_fields.clone();

                for (field_name, field_def) in new_fields {
                    if merged_fields.contains_key(field_name) {
                        // Handle field conflicts within types
                        match self.config.field_merge_strategy {
                            FieldMergeStrategy::Union => {
                                // Keep both fields (error if incompatible)
                                if merged_fields[field_name] != *field_def {
                                    return Err(anyhow!(
                                        "Incompatible field definitions for {}.{}",
                                        existing.name,
                                        field_name
                                    ));
                                }
                            }
                            FieldMergeStrategy::Override => {
                                merged_fields.insert(field_name.clone(), field_def.clone());
                            }
                        }
                    } else {
                        merged_fields.insert(field_name.clone(), field_def.clone());
                    }
                }

                Ok(TypeDefinition {
                    name: existing.name.clone(),
                    description: existing.description.clone(),
                    kind: TypeKind::Object {
                        fields: merged_fields,
                    },
                    directives: self.merge_directives(&existing.directives, &new.directives),
                })
            }
            (
                TypeKind::Interface {
                    fields: existing_fields,
                },
                TypeKind::Interface { fields: new_fields },
            ) => {
                // Similar logic for interfaces
                let mut merged_fields = existing_fields.clone();
                merged_fields.extend(new_fields.clone());

                Ok(TypeDefinition {
                    name: existing.name.clone(),
                    description: existing.description.clone(),
                    kind: TypeKind::Interface {
                        fields: merged_fields,
                    },
                    directives: existing.directives.clone(),
                })
            }
            _ => {
                // Cannot merge different kinds of types
                Err(anyhow!(
                    "Cannot merge different type kinds for type {}",
                    existing.name
                ))
            }
        }
    }

    /// Validate the unified schema for consistency
    fn validate_unified_schema(&self, schema: &UnifiedSchema) -> Result<()> {
        // Check for circular dependencies
        // Validate field types exist
        // Check directive usage
        // Validate federation constraints

        debug!(
            "Validating unified schema with {} types",
            schema.types.len()
        );

        // Basic validation - check that all field types exist
        for type_def in schema.types.values() {
            if let TypeKind::Object { fields } = &type_def.kind {
                for field_def in fields.values() {
                    if !schema.types.contains_key(&field_def.field_type) {
                        // Check if it's a built-in scalar type
                        if !self.is_builtin_type(&field_def.field_type) {
                            return Err(anyhow!(
                                "Unknown type '{}' used in field",
                                field_def.field_type
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if a type is a built-in GraphQL scalar type
    fn is_builtin_type(&self, type_name: &str) -> bool {
        matches!(type_name, "String" | "Int" | "Float" | "Boolean" | "ID")
    }

    /// Decompose a GraphQL query for federation
    pub async fn decompose_query(&self, query: &str) -> Result<Vec<ServiceQuery>> {
        debug!("Decomposing GraphQL query for federation");

        // Parse the query into an AST-like structure
        let parsed_query = self.parse_graphql_query(query)?;

        // Get the unified schema to understand field ownership
        let unified_schema = self.create_unified_schema().await?;

        // Analyze field ownership across services
        let field_ownership = self.analyze_field_ownership(&parsed_query, &unified_schema)?;

        // Decompose into service-specific queries
        let service_queries = self.create_service_queries(&parsed_query, &field_ownership)?;

        debug!(
            "Decomposed query into {} service queries",
            service_queries.len()
        );
        Ok(service_queries)
    }

    /// Parse a GraphQL query string into a structured representation
    fn parse_graphql_query(&self, query: &str) -> Result<ParsedQuery> {
        // This is a simplified parser - in production, use a proper GraphQL parser like graphql-parser
        let query = query.trim();

        // Extract operation type and name
        let (operation_type, operation_name) = if query.starts_with("query") {
            let parts: Vec<&str> = query.splitn(3, ' ').collect();
            let name = if parts.len() > 1 && parts[1] != "{" {
                Some(parts[1].to_string())
            } else {
                None
            };
            (GraphQLOperationType::Query, name)
        } else if query.starts_with("mutation") {
            let parts: Vec<&str> = query.splitn(3, ' ').collect();
            let name = if parts.len() > 1 && parts[1] != "{" {
                Some(parts[1].to_string())
            } else {
                None
            };
            (GraphQLOperationType::Mutation, name)
        } else if query.starts_with("subscription") {
            let parts: Vec<&str> = query.splitn(3, ' ').collect();
            let name = if parts.len() > 1 && parts[1] != "{" {
                Some(parts[1].to_string())
            } else {
                None
            };
            (GraphQLOperationType::Subscription, name)
        } else {
            // Default to query if no operation type specified
            (GraphQLOperationType::Query, None)
        };

        // Extract selection set (simplified)
        let selection_set = self.parse_selection_set(query)?;

        Ok(ParsedQuery {
            operation_type,
            operation_name,
            selection_set,
            variables: self.parse_variables(query)?,
        })
    }

    /// Parse a selection set from a GraphQL query
    fn parse_selection_set(&self, query: &str) -> Result<Vec<Selection>> {
        let mut selections = Vec::new();

        // Find the main selection set between braces
        if let Some(start) = query.find('{') {
            if let Some(end) = query.rfind('}') {
                let selection_content = &query[start + 1..end];

                // Split by lines and parse each field
                for line in selection_content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }

                    // Simple field parsing (no nested objects for now)
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(field_name) = parts.first() {
                        selections.push(Selection {
                            name: field_name.to_string(),
                            alias: None,
                            arguments: HashMap::new(), // TODO: Parse arguments
                            selection_set: Vec::new(), // TODO: Parse nested selections
                        });
                    }
                }
            }
        }

        Ok(selections)
    }

    /// Analyze which services own which fields
    fn analyze_field_ownership(
        &self,
        query: &ParsedQuery,
        schema: &UnifiedSchema,
    ) -> Result<FieldOwnership> {
        let mut ownership = FieldOwnership {
            field_to_service: HashMap::new(),
            service_to_fields: HashMap::new(),
        };

        for selection in &query.selection_set {
            // Determine which service owns this field
            let service_ids = match query.operation_type {
                GraphQLOperationType::Query => {
                    if let Some(field_def) = schema.queries.get(&selection.name) {
                        // For now, assign to the first service that can handle this query
                        // In a real implementation, this would be more sophisticated
                        schema
                            .schema_mapping
                            .get(&selection.name)
                            .unwrap_or(&vec!["default".to_string()])
                            .clone()
                    } else {
                        vec!["default".to_string()]
                    }
                }
                GraphQLOperationType::Mutation => {
                    if let Some(_field_def) = schema.mutations.get(&selection.name) {
                        schema
                            .schema_mapping
                            .get(&selection.name)
                            .unwrap_or(&vec!["default".to_string()])
                            .clone()
                    } else {
                        vec!["default".to_string()]
                    }
                }
                GraphQLOperationType::Subscription => {
                    if let Some(_field_def) = schema.subscriptions.get(&selection.name) {
                        schema
                            .schema_mapping
                            .get(&selection.name)
                            .unwrap_or(&vec!["default".to_string()])
                            .clone()
                    } else {
                        vec!["default".to_string()]
                    }
                }
            };

            // Record ownership
            for service_id in &service_ids {
                ownership
                    .field_to_service
                    .insert(selection.name.clone(), service_id.clone());
                ownership
                    .service_to_fields
                    .entry(service_id.clone())
                    .or_default()
                    .push(selection.name.clone());
            }
        }

        Ok(ownership)
    }

    /// Create service-specific queries based on field ownership
    fn create_service_queries(
        &self,
        query: &ParsedQuery,
        ownership: &FieldOwnership,
    ) -> Result<Vec<ServiceQuery>> {
        let mut service_queries = Vec::new();

        for (service_id, fields) in &ownership.service_to_fields {
            if fields.is_empty() {
                continue;
            }

            // Build a query for this service with only its owned fields
            let operation_type_str = match query.operation_type {
                GraphQLOperationType::Query => "query",
                GraphQLOperationType::Mutation => "mutation",
                GraphQLOperationType::Subscription => "subscription",
            };

            let operation_name = query
                .operation_name
                .as_ref()
                .map(|name| format!(" {}", name))
                .unwrap_or_default();

            let field_strings: Vec<String> =
                fields.iter().map(|field| format!("  {}", field)).collect();

            let service_query = format!(
                "{}{} {{\n{}\n}}",
                operation_type_str,
                operation_name,
                field_strings.join("\n")
            );

            service_queries.push(ServiceQuery {
                service_id: service_id.clone(),
                query: service_query,
                variables: None, // TODO: Filter variables based on field usage
            });
        }

        Ok(service_queries)
    }

    /// Stitch together results from multiple services
    pub async fn stitch_results(
        &self,
        service_results: Vec<ServiceResult>,
    ) -> Result<GraphQLResponse> {
        debug!("Stitching {} service results", service_results.len());

        let mut merged_data = serde_json::Map::new();
        let mut all_errors = Vec::new();

        for service_result in service_results {
            if let Some(data) = service_result.response.data.as_object() {
                for (key, value) in data {
                    merged_data.insert(key.clone(), value.clone());
                }
            }
            all_errors.extend(service_result.response.errors);
        }

        Ok(GraphQLResponse {
            data: serde_json::Value::Object(merged_data),
            errors: all_errors,
            extensions: None,
        })
    }

    /// Introspect a GraphQL service for Apollo Federation support
    pub async fn introspect_federation_support(
        &self,
        service_endpoint: &str,
    ) -> Result<FederationServiceInfo> {
        debug!(
            "Introspecting GraphQL service for federation support: {}",
            service_endpoint
        );

        // Query for federation support
        let federation_query = r#"
            query FederationIntrospection {
                _service {
                    sdl
                }
                __schema {
                    types {
                        name
                        fields {
                            name
                            type {
                                name
                                kind
                            }
                        }
                    }
                }
            }
        "#;

        // In a real implementation, this would make an HTTP request to the service
        // For now, return mock data
        let mock_sdl = r#"
            extend type Query {
                me: User
            }
            
            type User @key(fields: "id") {
                id: ID!
                username: String!
                email: String! @external
            }
        "#;

        Ok(FederationServiceInfo {
            sdl: mock_sdl.to_string(),
            capabilities: FederationCapabilities {
                federation_version: "2.0".to_string(),
                supports_entities: true,
                supports_entity_interfaces: true,
                supports_progressive_override: false,
            },
            entity_types: vec!["User".to_string()],
        })
    }

    /// Resolve entity representations across federated services
    pub async fn resolve_entity_representations(
        &self,
        representations: Vec<EntityRepresentation>,
    ) -> Result<Vec<serde_json::Value>> {
        debug!("Resolving {} entity representations", representations.len());

        // Group representations by type
        let mut by_type: HashMap<String, Vec<&EntityRepresentation>> = HashMap::new();
        for repr in &representations {
            by_type.entry(repr.typename.clone()).or_default().push(repr);
        }

        // Resolve entities by type across appropriate services
        let mut resolved_entities = Vec::new();

        for (typename, reprs) in by_type {
            // Find which service owns this entity type
            let service_id = self.find_service_for_entity(&typename).await?;

            // Build _entities query for this service
            let entities_query =
                self.build_entities_query_for_representations(&typename, &reprs)?;

            // Execute query (mock for now)
            let mock_entity = serde_json::json!({
                "__typename": typename,
                "id": "123",
                "username": "john_doe",
                "email": "john@example.com"
            });

            resolved_entities.push(mock_entity);
        }

        Ok(resolved_entities)
    }

    /// Find which service owns an entity type
    async fn find_service_for_entity(&self, typename: &str) -> Result<String> {
        let schemas = self.schemas.read().await;

        for (service_id, schema) in schemas.iter() {
            if schema.types.contains_key(typename) {
                return Ok(service_id.clone());
            }
        }

        Err(anyhow!("No service found for entity type: {}", typename))
    }

    /// Build an _entities query for resolving entity representations
    fn build_entities_query_for_representations(
        &self,
        typename: &str,
        representations: &[&EntityRepresentation],
    ) -> Result<String> {
        let repr_json: Vec<serde_json::Value> = representations
            .iter()
            .map(|r| {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "__typename".to_string(),
                    serde_json::Value::String(typename.to_string()),
                );
                if let serde_json::Value::Object(fields) = &r.fields {
                    for (k, v) in fields {
                        obj.insert(k.clone(), v.clone());
                    }
                }
                serde_json::Value::Object(obj)
            })
            .collect();

        Ok(format!(
            r#"
            query GetEntities($representations: [_Any!]!) {{
                _entities(representations: $representations) {{
                    ... on {} {{
                        # Fields would be added based on the query requirements
                        id
                    }}
                }}
            }}
            "#,
            typename
        ))
    }

    /// Parse Apollo Federation directives from a type definition
    pub fn parse_federation_directives(&self, type_def: &TypeDefinition) -> FederationDirectives {
        let mut fed_directives = FederationDirectives {
            key: None,
            external: false,
            requires: None,
            provides: None,
            extends: false,
            shareable: false,
            override_from: None,
            inaccessible: false,
            tags: Vec::new(),
        };

        for directive in &type_def.directives {
            match directive.name.as_str() {
                "key" => {
                    if let Some(fields_arg) = directive.arguments.get("fields") {
                        if let serde_json::Value::String(fields) = fields_arg {
                            let resolvable = directive
                                .arguments
                                .get("resolvable")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(true);

                            fed_directives.key = Some(KeyDirective {
                                fields: fields.clone(),
                                resolvable,
                            });
                        }
                    }
                }
                "external" => fed_directives.external = true,
                "requires" => {
                    if let Some(fields_arg) = directive.arguments.get("fields") {
                        if let serde_json::Value::String(fields) = fields_arg {
                            fed_directives.requires = Some(fields.clone());
                        }
                    }
                }
                "provides" => {
                    if let Some(fields_arg) = directive.arguments.get("fields") {
                        if let serde_json::Value::String(fields) = fields_arg {
                            fed_directives.provides = Some(fields.clone());
                        }
                    }
                }
                "extends" => fed_directives.extends = true,
                "shareable" => fed_directives.shareable = true,
                "override" => {
                    if let Some(from_arg) = directive.arguments.get("from") {
                        if let serde_json::Value::String(from) = from_arg {
                            fed_directives.override_from = Some(from.clone());
                        }
                    }
                }
                "inaccessible" => fed_directives.inaccessible = true,
                "tag" => {
                    if let Some(name_arg) = directive.arguments.get("name") {
                        if let serde_json::Value::String(tag) = name_arg {
                            fed_directives.tags.push(tag.clone());
                        }
                    }
                }
                _ => {}
            }
        }

        fed_directives
    }

    // ============= ADVANCED GRAPHQL FEDERATION & ENTITY RESOLUTION =============

    /// Advanced entity resolution with federation directive support
    pub async fn resolve_entities(
        &self,
        query: &str,
        variables: Option<serde_json::Value>,
    ) -> Result<GraphQLResponse> {
        debug!("Resolving entities for federated GraphQL query");

        // Parse query to identify entity references
        let entity_references = self.extract_entity_references(query)?;

        // Build entity resolution plan
        let resolution_plan = self
            .build_entity_resolution_plan(&entity_references)
            .await?;

        // Execute entity resolution in optimal order
        let resolved_entities = self
            .execute_entity_resolution_plan(&resolution_plan)
            .await?;

        // Stitch final response
        let response = self
            .stitch_entity_response(query, &resolved_entities, variables)
            .await?;

        Ok(response)
    }

    /// Extract entity references from GraphQL query
    fn extract_entity_references(&self, query: &str) -> Result<Vec<EntityReference>> {
        let mut entity_refs = Vec::new();

        // Parse query to find entities (simplified parser)
        // In real implementation, would use proper GraphQL parser
        let lines: Vec<&str> = query.lines().collect();

        for line in lines {
            if line.trim().contains("@key") {
                // Extract entity type and key fields
                if let Some(entity_ref) = self.parse_entity_reference_from_line(line)? {
                    entity_refs.push(entity_ref);
                }
            }
        }

        Ok(entity_refs)
    }

    /// Parse entity reference from a query line
    fn parse_entity_reference_from_line(&self, line: &str) -> Result<Option<EntityReference>> {
        // Simplified parsing - would be more sophisticated in real implementation
        if line.contains("User") && line.contains("id") {
            return Ok(Some(EntityReference {
                entity_type: "User".to_string(),
                key_fields: vec!["id".to_string()],
                required_fields: vec!["username".to_string(), "email".to_string()],
                service_id: "user-service".to_string(), // Would be determined by schema analysis
            }));
        }

        if line.contains("Product") && line.contains("sku") {
            return Ok(Some(EntityReference {
                entity_type: "Product".to_string(),
                key_fields: vec!["sku".to_string()],
                required_fields: vec!["name".to_string(), "price".to_string()],
                service_id: "product-service".to_string(),
            }));
        }

        Ok(None)
    }

    /// Build entity resolution plan with optimal execution order
    async fn build_entity_resolution_plan(
        &self,
        entity_refs: &[EntityReference],
    ) -> Result<EntityResolutionPlan> {
        let mut plan = EntityResolutionPlan {
            steps: Vec::new(),
            dependencies: HashMap::new(),
        };

        // Group entities by service for batch resolution
        let mut service_entities: HashMap<String, Vec<EntityReference>> = HashMap::new();
        for entity_ref in entity_refs {
            service_entities
                .entry(entity_ref.service_id.clone())
                .or_default()
                .push(entity_ref.clone());
        }

        // Create resolution steps
        for (service_name, entities) in service_entities {
            let step = EntityResolutionStep {
                service_name: service_name.clone(),
                entity_type: entities
                    .first()
                    .map(|e| e.entity_type.clone())
                    .unwrap_or_default(),
                key_fields: entities.iter().flat_map(|e| e.key_fields.clone()).collect(),
                query: self.build_entity_query(&entities).await?,
                depends_on: self.analyze_entity_dependencies(&entities).await?,
            };

            plan.steps.push(step);
        }

        // Optimize execution order based on dependencies
        // Sort by dependency count (least dependencies first)
        plan.steps.sort_by_key(|step| step.depends_on.len());

        Ok(plan)
    }

    /// Build GraphQL query for entity batch
    async fn build_entity_query(&self, entities: &[EntityReference]) -> Result<String> {
        if entities.is_empty() {
            return Ok(String::new());
        }

        let first_entity = &entities[0];
        let selection_fields = first_entity.required_fields.join(" ");

        // Simple implementation - could be enhanced for batching
        Ok(format!("{{ {} }}", selection_fields))
    }

    /// Analyze dependencies between entities
    async fn analyze_entity_dependencies(
        &self,
        _entities: &[EntityReference],
    ) -> Result<Vec<String>> {
        // Simple implementation - no dependencies for now
        Ok(Vec::new())
    }

    /// Execute entity resolution plan
    async fn execute_entity_resolution_plan(
        &self,
        plan: &EntityResolutionPlan,
    ) -> Result<HashMap<String, Vec<EntityData>>> {
        let mut resolved_entities = HashMap::new();

        for step in &plan.steps {
            debug!(
                "Executing entity resolution step for service: {}",
                step.service_name
            );

            // TODO: Extract entity references from the step query
            let entity_refs = Vec::new(); // Placeholder for entity references

            // Batch resolve entities for this service
            let entities = self
                .batch_resolve_entities(&step.service_name, &entity_refs)
                .await?;
            resolved_entities.insert(step.service_name.clone(), entities);
        }

        Ok(resolved_entities)
    }

    /// Batch resolve entities from a specific service
    async fn batch_resolve_entities(
        &self,
        service_id: &str,
        entity_refs: &[EntityReference],
    ) -> Result<Vec<EntityData>> {
        if entity_refs.is_empty() {
            return Ok(Vec::new());
        }

        // Group by entity type for efficient querying
        let mut entities_by_type: HashMap<String, Vec<&EntityReference>> = HashMap::new();
        for entity_ref in entity_refs {
            entities_by_type
                .entry(entity_ref.entity_type.clone())
                .or_default()
                .push(entity_ref);
        }

        let mut resolved_entities = Vec::new();

        for (typename, refs) in entities_by_type {
            // For now, create a basic query structure - this should be enhanced
            // to properly build GraphQL _entities queries from EntityReference data
            let entities_query = format!(
                "query {{ _entities(representations: [{{ __typename: \"{}\" }}]) {{ ... on {} {{ id }} }} }}",
                typename, typename
            );

            // Execute query against service (mock implementation)
            let response = self
                .execute_service_query(service_id, &entities_query)
                .await?;

            // Parse response into EntityData
            let entities = self.parse_entities_response(&response, &typename)?;
            resolved_entities.extend(entities);
        }

        Ok(resolved_entities)
    }

    /// Build GraphQL query for entity resolution
    fn build_entities_query(
        &self,
        typename: &str,
        representations: &[&EntityReference],
    ) -> Result<String> {
        let mut reprs_json = Vec::new();

        for repr in representations {
            let mut repr_obj = serde_json::Map::new();
            repr_obj.insert(
                "__typename".to_string(),
                serde_json::Value::String(typename.to_string()),
            );

            // Add key fields (mock values for now)
            for key_field in &repr.key_fields {
                repr_obj.insert(
                    key_field.clone(),
                    serde_json::Value::String("example_value".to_string()),
                );
            }

            reprs_json.push(serde_json::Value::Object(repr_obj));
        }

        let representations_str = serde_json::to_string(&reprs_json)?;

        let query = format!(
            r#"
            query($_representations: [_Any!]!) {{
                _entities(representations: $_representations) {{
                    ... on {} {{
                        {}
                    }}
                }}
            }}
            "#,
            typename,
            representations
                .first()
                .map(|r| r.required_fields.join("\n                        "))
                .unwrap_or_default()
        );

        Ok(query)
    }

    /// Execute query against a specific GraphQL service
    async fn execute_service_query(
        &self,
        service_id: &str,
        query: &str,
    ) -> Result<GraphQLResponse> {
        debug!("Executing GraphQL query against service: {}", service_id);

        // Mock implementation - would make actual HTTP request to service
        Ok(GraphQLResponse {
            data: serde_json::json!({
                "_entities": [
                    {
                        "__typename": "User",
                        "id": "1",
                        "username": "john_doe",
                        "email": "john@example.com"
                    }
                ]
            }),
            errors: Vec::new(),
            extensions: None,
        })
    }

    /// Parse entities from GraphQL response
    fn parse_entities_response(
        &self,
        response: &GraphQLResponse,
        typename: &str,
    ) -> Result<Vec<EntityData>> {
        let mut entities = Vec::new();

        let data = &response.data;
        if let Some(entities_array) = data.get("_entities").and_then(|v| v.as_array()) {
            for entity_value in entities_array {
                if let Some(entity_obj) = entity_value.as_object() {
                    if entity_obj.get("__typename").and_then(|v| v.as_str()) == Some(typename) {
                        entities.push(EntityData {
                            typename: typename.to_string(),
                            fields: entity_obj.clone(),
                        });
                    }
                }
            }
        }

        Ok(entities)
    }

    /// Stitch final response from resolved entities
    async fn stitch_entity_response(
        &self,
        original_query: &str,
        resolved_entities: &HashMap<String, Vec<EntityData>>,
        variables: Option<serde_json::Value>,
    ) -> Result<GraphQLResponse> {
        debug!(
            "Stitching final GraphQL response from {} services",
            resolved_entities.len()
        );

        // Combine all entity data into a unified response
        let mut combined_data = serde_json::Map::new();

        for (service_id, entities) in resolved_entities {
            for entity in entities {
                // Merge entity fields into response based on query structure
                self.merge_entity_into_response(&mut combined_data, entity, original_query)?;
            }
        }

        Ok(GraphQLResponse {
            data: serde_json::Value::Object(combined_data),
            errors: Vec::new(),
            extensions: None,
        })
    }

    /// Merge entity data into response structure
    fn merge_entity_into_response(
        &self,
        response: &mut serde_json::Map<String, serde_json::Value>,
        entity: &EntityData,
        query: &str,
    ) -> Result<()> {
        // Simplified merging logic based on query structure
        // In real implementation, would parse query AST and match field selections

        if query.contains("me") && entity.typename == "User" {
            response.insert(
                "me".to_string(),
                serde_json::Value::Object(entity.fields.clone()),
            );
        } else if query.contains("product") && entity.typename == "Product" {
            response.insert(
                "product".to_string(),
                serde_json::Value::Object(entity.fields.clone()),
            );
        }

        Ok(())
    }

    /// Advanced schema composition with federation directive support
    pub async fn compose_federated_schema(&self) -> Result<ComposedSchema> {
        debug!("Composing federated schema with directive support");

        let schemas = self.schemas.read().await;
        let mut composed = ComposedSchema {
            types: HashMap::new(),
            query_type: "Query".to_string(),
            mutation_type: None,
            subscription_type: None,
            directives: Vec::new(),
            entity_types: HashMap::new(),
            field_ownership: HashMap::new(),
        };

        // Process each schema for federation directives
        for (service_id, schema) in schemas.iter() {
            self.process_schema_for_federation(&mut composed, service_id, schema)?;
        }

        // Generate composed SDL
        // Generate composed SDL
        composed.directives = self.extract_federation_directives(&composed)?;

        // Validate composition
        self.validate_composed_schema(&composed)?;

        info!(
            "Successfully composed federated schema with {} types",
            composed.types.len()
        );
        Ok(composed)
    }

    /// Advanced schema discovery with introspection
    pub async fn discover_schema_capabilities(
        &self,
        service_endpoint: &str,
    ) -> Result<SchemaCapabilities> {
        debug!("Discovering schema capabilities for {}", service_endpoint);

        let introspection_query = r#"
            query IntrospectionQuery {
                __schema {
                    queryType { name }
                    mutationType { name }
                    subscriptionType { name }
                    types {
                        ...FullType
                    }
                    directives {
                        name
                        description
                        locations
                        args {
                            ...InputValue
                        }
                    }
                }
            }
            
            fragment FullType on __Type {
                kind
                name
                description
                fields(includeDeprecated: true) {
                    name
                    description
                    args {
                        ...InputValue
                    }
                    type {
                        ...TypeRef
                    }
                    isDeprecated
                    deprecationReason
                }
                inputFields {
                    ...InputValue
                }
                interfaces {
                    ...TypeRef
                }
                enumValues(includeDeprecated: true) {
                    name
                    description
                    isDeprecated
                    deprecationReason
                }
                possibleTypes {
                    ...TypeRef
                }
            }
            
            fragment InputValue on __InputValue {
                name
                description
                type { ...TypeRef }
                defaultValue
            }
            
            fragment TypeRef on __Type {
                kind
                name
                ofType {
                    kind
                    name
                    ofType {
                        kind
                        name
                        ofType {
                            kind
                            name
                            ofType {
                                kind
                                name
                                ofType {
                                    kind
                                    name
                                    ofType {
                                        kind
                                        name
                                        ofType {
                                            kind
                                            name
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        "#;

        let response = self
            .execute_introspection_query(service_endpoint, introspection_query)
            .await?;
        self.parse_introspection_response(response)
    }

    /// Execute introspection query against a GraphQL service
    async fn execute_introspection_query(
        &self,
        endpoint: &str,
        query: &str,
    ) -> Result<serde_json::Value> {
        let client = reqwest::Client::new();
        let request_body = serde_json::json!({
            "query": query
        });

        let response = client
            .post(endpoint)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Introspection query failed with status: {}",
                response.status()
            ));
        }

        let body: serde_json::Value = response.json().await?;
        Ok(body)
    }

    /// Parse introspection response to extract schema capabilities
    fn parse_introspection_response(
        &self,
        response: serde_json::Value,
    ) -> Result<SchemaCapabilities> {
        let schema = response["data"]["__schema"]
            .as_object()
            .ok_or_else(|| anyhow!("Invalid introspection response: missing schema"))?;

        let mut capabilities = SchemaCapabilities {
            supports_federation: false,
            supports_subscriptions: false,
            supports_defer_stream: false,
            entity_types: Vec::new(),
            custom_directives: Vec::new(),
            scalar_types: Vec::new(),
            estimated_complexity: 0.0,
        };

        // Check for federation support
        if let Some(directives) = schema["directives"].as_array() {
            for directive in directives {
                if let Some(name) = directive["name"].as_str() {
                    capabilities.custom_directives.push(name.to_string());

                    // Federation directives
                    if matches!(
                        name,
                        "key" | "external" | "requires" | "provides" | "extends"
                    ) {
                        capabilities.supports_federation = true;
                    }
                }
            }
        }

        // Check for subscription support
        if schema["subscriptionType"].is_object() {
            capabilities.supports_subscriptions = true;
        }

        // Analyze types for entities and complexity
        if let Some(types) = schema["types"].as_array() {
            for type_def in types {
                if let Some(type_name) = type_def["name"].as_str() {
                    // Skip GraphQL built-in types
                    if type_name.starts_with("__") {
                        continue;
                    }

                    if let Some(kind) = type_def["kind"].as_str() {
                        match kind {
                            "OBJECT" => {
                                capabilities.estimated_complexity += 1.0;

                                // Check if this could be an entity (has ID field)
                                if let Some(fields) = type_def["fields"].as_array() {
                                    let has_id = fields
                                        .iter()
                                        .any(|field| field["name"].as_str() == Some("id"));

                                    if has_id {
                                        capabilities.entity_types.push(type_name.to_string());
                                    }
                                }
                            }
                            "SCALAR" => {
                                if !matches!(
                                    type_name,
                                    "String" | "Int" | "Float" | "Boolean" | "ID"
                                ) {
                                    capabilities.scalar_types.push(type_name.to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Check for @defer/@stream support (Apollo Federation 2.0+)
        capabilities.supports_defer_stream = capabilities
            .custom_directives
            .iter()
            .any(|d| matches!(d.as_str(), "defer" | "stream"));

        Ok(capabilities)
    }

    /// Dynamic schema update with hot reloading
    pub async fn update_schema_dynamic(
        &self,
        service_id: String,
        new_schema: FederatedSchema,
    ) -> Result<SchemaUpdateResult> {
        debug!(
            "Performing dynamic schema update for service: {}",
            service_id
        );

        let mut update_result = SchemaUpdateResult {
            service_id: service_id.clone(),
            update_successful: false,
            breaking_changes: Vec::new(),
            warnings: Vec::new(),
            rollback_available: false,
        };

        // Backup current schema for rollback
        let current_schema = {
            let schemas = self.schemas.read().await;
            schemas.get(&service_id).cloned()
        };

        if let Some(old_schema) = &current_schema {
            // Analyze breaking changes
            update_result.breaking_changes =
                self.detect_breaking_changes(old_schema, &new_schema)?;
            update_result.rollback_available = true;

            // Check if update should be blocked due to breaking changes
            if !update_result.breaking_changes.is_empty() && !self.config.allow_breaking_changes {
                update_result
                    .warnings
                    .push("Update blocked due to breaking changes".to_string());
                return Ok(update_result);
            }
        }

        // Perform the update
        {
            let mut schemas = self.schemas.write().await;
            schemas.insert(service_id.clone(), new_schema.clone());
        }

        // Validate the new composed schema
        match self.create_unified_schema().await {
            Ok(_) => {
                update_result.update_successful = true;
                info!("Schema update successful for service: {}", service_id);
            }
            Err(e) => {
                // Rollback on validation failure
                if let Some(old_schema) = current_schema {
                    let mut schemas = self.schemas.write().await;
                    schemas.insert(service_id.clone(), old_schema);
                }
                return Err(anyhow!("Schema update failed validation: {}", e));
            }
        }

        Ok(update_result)
    }

    /// Detect breaking changes between schema versions
    fn detect_breaking_changes(
        &self,
        old_schema: &FederatedSchema,
        new_schema: &FederatedSchema,
    ) -> Result<Vec<BreakingChange>> {
        let mut breaking_changes = Vec::new();

        // Check for removed types
        for type_name in old_schema.types.keys() {
            if !new_schema.types.contains_key(type_name) {
                breaking_changes.push(BreakingChange {
                    change_type: BreakingChangeType::TypeRemoved,
                    description: format!("Type '{}' was removed", type_name),
                    severity: BreakingChangeSeverity::High,
                });
            }
        }

        // Check for field changes in existing types
        for (type_name, new_type) in &new_schema.types {
            if let Some(old_type) = old_schema.types.get(type_name) {
                let type_changes = self.detect_type_breaking_changes(type_name, old_type, new_type)?;
                breaking_changes.extend(type_changes);
            }
        }

        Ok(breaking_changes)
    }

    /// Detect breaking changes in a specific type
    fn detect_type_breaking_changes(
        &self,
        type_name: &str,
        old_type: &TypeDefinition,
        new_type: &TypeDefinition,
    ) -> Result<Vec<BreakingChange>> {
        let mut changes = Vec::new();

        // Extract fields from TypeKind for comparison
        let old_fields = match &old_type.kind {
            TypeKind::Object { fields } | TypeKind::Interface { fields } => Some(fields),
            _ => None,
        };

        let new_fields = match &new_type.kind {
            TypeKind::Object { fields } | TypeKind::Interface { fields } => Some(fields),
            _ => None,
        };

        if let (Some(old_fields), Some(new_fields)) = (old_fields, new_fields) {
            // Check for removed fields
            for field_name in old_fields.keys() {
                if !new_fields.contains_key(field_name) {
                    changes.push(BreakingChange {
                        change_type: BreakingChangeType::FieldRemoved,
                        description: format!("Field '{}.{}' was removed", type_name, field_name),
                        severity: BreakingChangeSeverity::High,
                    });
                }
            }

            // Check for argument changes in existing fields
            for (field_name, new_field) in new_fields {
                if let Some(old_field) = old_fields.get(field_name) {
                    // Check for required arguments added
                    for (arg_name, new_arg) in &new_field.arguments {
                        if let Some(old_arg) = old_field.arguments.get(arg_name) {
                            // Check if argument became required
                            if old_arg.default_value.is_some() && new_arg.default_value.is_none() {
                                changes.push(BreakingChange {
                                    change_type: BreakingChangeType::ArgumentMadeRequired,
                                    description: format!(
                                        "Argument '{}.{}.{}' is now required",
                                        type_name, field_name, arg_name
                                    ),
                                    severity: BreakingChangeSeverity::Medium,
                                });
                            }
                        } else if new_arg.default_value.is_none() {
                            // New required argument
                            changes.push(BreakingChange {
                                change_type: BreakingChangeType::RequiredArgumentAdded,
                                description: format!(
                                    "Required argument '{}.{}.{}' was added",
                                    type_name, field_name, arg_name
                                ),
                                severity: BreakingChangeSeverity::High,
                            });
                        }
                    }
                }
            }
        }

        Ok(changes)
    }

    /// Enhanced entity resolution with dependency tracking
    pub async fn resolve_entities_advanced(
        &self,
        entities: &[EntityReference],
        context: &ResolutionContext,
    ) -> Result<Vec<EntityData>> {
        debug!(
            "Resolving {} entities with advanced dependency tracking",
            entities.len()
        );

        // Build dependency graph
        let dependency_graph = self.build_entity_dependency_graph(entities)?;

        // Topological sort for resolution order
        let resolution_order = self.topological_sort_entities(&dependency_graph)?;

        let mut resolved_entities = Vec::new();
        let mut resolution_cache = HashMap::new();

        // Resolve entities in dependency order
        for batch in resolution_order {
            let batch_results = self
                .resolve_entity_batch(&batch, context, &resolution_cache)
                .await?;

            // Update cache and results
            for (entity_ref, entity_data) in batch_results {
                resolution_cache.insert(entity_ref.clone(), entity_data.clone());
                resolved_entities.push(entity_data);
            }
        }

        Ok(resolved_entities)
    }

    /// Build dependency graph for entity resolution
    fn build_entity_dependency_graph(
        &self,
        entities: &[EntityReference],
    ) -> Result<EntityDependencyGraph> {
        let mut graph = EntityDependencyGraph {
            nodes: HashMap::new(),
            edges: Vec::new(),
        };

        // Add nodes
        for (idx, entity) in entities.iter().enumerate() {
            graph.nodes.insert(entity.clone(), idx);
        }

        // Add edges based on field dependencies
        for (i, entity_a) in entities.iter().enumerate() {
            for (j, entity_b) in entities.iter().enumerate() {
                if i != j && self.entities_have_dependency(entity_a, entity_b)? {
                    graph.edges.push((i, j));
                }
            }
        }

        Ok(graph)
    }

    /// Check if one entity depends on another
    fn entities_have_dependency(
        &self,
        entity_a: &EntityReference,
        entity_b: &EntityReference,
    ) -> Result<bool> {
        // Simple heuristic: check if entity_a requires fields that entity_b provides
        for required_field in &entity_a.required_fields {
            if entity_b.key_fields.contains(required_field) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Topological sort of entities for resolution order
    fn topological_sort_entities(
        &self,
        graph: &EntityDependencyGraph,
    ) -> Result<Vec<Vec<EntityReference>>> {
        let mut in_degree = vec![0; graph.nodes.len()];
        let mut adj_list = vec![Vec::new(); graph.nodes.len()];

        // Build adjacency list and calculate in-degrees
        for &(from, to) in &graph.edges {
            adj_list[from].push(to);
            in_degree[to] += 1;
        }

        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Find nodes with no incoming edges
        for (idx, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(idx);
            }
        }

        // Process in batches (nodes that can be resolved in parallel)
        while !queue.is_empty() {
            let mut batch = Vec::new();
            let batch_size = queue.len();

            // Take all nodes with no dependencies as a batch
            for _ in 0..batch_size {
                let node = queue.pop_front().unwrap();

                // Find the entity reference for this node index
                let entity_ref = graph
                    .nodes
                    .iter()
                    .find(|(_, &idx)| idx == node)
                    .map(|(entity_ref, _)| entity_ref.clone())
                    .ok_or_else(|| anyhow!("Node index not found in graph"))?;

                batch.push(entity_ref);

                // Reduce in-degree of neighbors
                for &neighbor in &adj_list[node] {
                    in_degree[neighbor] -= 1;
                    if in_degree[neighbor] == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }

            if !batch.is_empty() {
                result.push(batch);
            }
        }

        Ok(result)
    }

    /// Resolve a batch of entities in parallel
    async fn resolve_entity_batch(
        &self,
        entities: &[EntityReference],
        context: &ResolutionContext,
        cache: &HashMap<EntityReference, EntityData>,
    ) -> Result<Vec<(EntityReference, EntityData)>> {
        let mut results = Vec::new();

        // Group entities by service for batch resolution
        let mut service_groups: HashMap<String, Vec<&EntityReference>> = HashMap::new();
        for entity in entities {
            service_groups
                .entry(entity.service_id.clone())
                .or_default()
                .push(entity);
        }

        // Resolve each service group
        for (service_id, service_entities) in service_groups {
            let service_results = self
                .resolve_service_entity_batch(&service_id, &service_entities, context, cache)
                .await?;
            results.extend(service_results);
        }

        Ok(results)
    }

    /// Resolve entities from a specific service
    async fn resolve_service_entity_batch(
        &self,
        service_id: &str,
        entities: &[&EntityReference],
        _context: &ResolutionContext,
        _cache: &HashMap<EntityReference, EntityData>,
    ) -> Result<Vec<(EntityReference, EntityData)>> {
        // Build entity resolution query
        let query = self.build_entity_batch_query(entities)?;

        // Execute query against service
        let response = self.execute_service_query(service_id, &query).await?;

        // Parse response and match to entity references
        self.parse_entity_batch_response(entities, response)
    }

    /// Build GraphQL query for batch entity resolution
    fn build_entity_batch_query(&self, entities: &[&EntityReference]) -> Result<String> {
        let mut query_parts = Vec::new();

        for (idx, entity) in entities.iter().enumerate() {
            let alias = format!("entity_{}", idx);
            let key_args = entity
                .key_fields
                .iter()
                .map(|field| format!("{}: ${}_{}", field, alias, field))
                .collect::<Vec<_>>()
                .join(", ");

            let field_selection = entity.required_fields.join(" ");

            query_parts.push(format!(
                "{}: {}({}) {{ {} }}",
                alias, entity.entity_type, key_args, field_selection
            ));
        }

        Ok(format!("query {{ {} }}", query_parts.join(" ")))
    }

    /// Parse entity batch response
    fn parse_entity_batch_response(
        &self,
        entities: &[&EntityReference],
        response: GraphQLResponse,
    ) -> Result<Vec<(EntityReference, EntityData)>> {
        let mut results = Vec::new();

        let data = &response.data;
        for (idx, entity) in entities.iter().enumerate() {
            let alias = format!("entity_{}", idx);

            if let Some(entity_data) = data.get(&alias) {
                if let Some(obj) = entity_data.as_object() {
                    results.push((
                        (*entity).clone(),
                        EntityData {
                            typename: entity.entity_type.clone(),
                            fields: obj.clone(),
                        },
                    ));
                }
            }
        }

        Ok(results)
    }

    /// Extract federation directives from composed schema
    fn extract_federation_directives(&self, _composed: &ComposedSchema) -> Result<Vec<String>> {
        // Federation directives that should be available in the composed schema
        Ok(vec![
            "key".to_string(),
            "external".to_string(),
            "requires".to_string(),
            "provides".to_string(),
            "extends".to_string(),
            "shareable".to_string(),
            "inaccessible".to_string(),
            "override".to_string(),
            "composeDirective".to_string(),
            "interfaceObject".to_string(),
        ])
    }

    /// Validate composed schema for federation compliance
    fn validate_composed_schema(&self, composed: &ComposedSchema) -> Result<()> {
        // Validate entity types have proper key directives
        for (type_name, entity_info) in &composed.entity_types {
            if entity_info.key_fields.is_empty() {
                return Err(anyhow!(
                    "Entity type '{}' must have at least one key field",
                    type_name
                ));
            }
        }

        // Validate field ownership
        for (field_path, ownership) in &composed.field_ownership {
            match ownership {
                FieldOwnershipType::Requires(fields) if fields.is_empty() => {
                    return Err(anyhow!(
                        "Field '{}' with @requires directive must specify required fields",
                        field_path
                    ));
                }
                FieldOwnershipType::Provides(fields) if fields.is_empty() => {
                    return Err(anyhow!(
                        "Field '{}' with @provides directive must specify provided fields",
                        field_path
                    ));
                }
                _ => {}
            }
        }

        info!("Composed schema validation passed");
        Ok(())
    }

    /// Process schema for federation directives (@key, @external, @requires, @provides)
    fn process_schema_for_federation(
        &self,
        composed: &mut ComposedSchema,
        service_id: &str,
        schema: &FederatedSchema,
    ) -> Result<()> {
        for (type_name, type_def) in &schema.types {
            // Check for @key directive (entity definition)
            for directive in &type_def.directives {
                match directive.name.as_str() {
                    "key" => {
                        let key_fields = self.extract_key_fields_from_directive(directive)?;
                        composed.entity_types.insert(
                            type_name.clone(),
                            EntityTypeInfo {
                                key_fields,
                                owning_service: service_id.to_string(),
                                extending_services: Vec::new(),
                            },
                        );
                    }
                    _ => {}
                }
            }

            // Process field-level directives
            if let TypeKind::Object { fields } = &type_def.kind {
                for (field_name, field_def) in fields {
                    let field_key = format!("{}.{}", type_name, field_name);

                    for directive in &field_def.directives {
                        match directive.name.as_str() {
                            "external" => {
                                // Field is defined in another service
                                composed
                                    .field_ownership
                                    .insert(field_key.clone(), FieldOwnershipType::External);
                            }
                            "requires" => {
                                // Field requires other fields to be resolved first
                                let required_fields =
                                    self.extract_requires_fields_from_directive(directive)?;
                                composed.field_ownership.insert(
                                    field_key.clone(),
                                    FieldOwnershipType::Requires(required_fields),
                                );
                            }
                            "provides" => {
                                // Field provides data for other services
                                let provided_fields =
                                    self.extract_provides_fields_from_directive(directive)?;
                                composed.field_ownership.insert(
                                    field_key.clone(),
                                    FieldOwnershipType::Provides(provided_fields),
                                );
                            }
                            _ => {}
                        }
                    }

                    // Track field ownership by service
                    if !composed.field_ownership.contains_key(&field_key) {
                        composed
                            .field_ownership
                            .insert(field_key, FieldOwnershipType::Owned(service_id.to_string()));
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract key fields from @key directive
    fn extract_key_fields_from_directive(&self, directive: &DirectiveUsage) -> Result<Vec<String>> {
        // Parse fields argument from @key(fields: "id name")
        if let Some(fields_value) = directive.arguments.get("fields") {
            if let Some(fields_str) = fields_value.as_str() {
                return Ok(fields_str
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect());
            }
        }
        Err(anyhow!("@key directive missing fields argument"))
    }

    /// Extract required fields from @requires directive
    fn extract_requires_fields_from_directive(
        &self,
        directive: &DirectiveUsage,
    ) -> Result<Vec<String>> {
        if let Some(fields_value) = directive.arguments.get("fields") {
            if let Some(fields_str) = fields_value.as_str() {
                return Ok(fields_str
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect());
            }
        }
        Err(anyhow!("@requires directive missing fields argument"))
    }

    /// Extract provided fields from @provides directive
    fn extract_provides_fields_from_directive(
        &self,
        directive: &DirectiveUsage,
    ) -> Result<Vec<String>> {
        if let Some(fields_value) = directive.arguments.get("fields") {
            if let Some(fields_str) = fields_value.as_str() {
                return Ok(fields_str
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect());
            }
        }
        Err(anyhow!("@provides directive missing fields argument"))
    }

    /// Generate composed SDL from federated schemas
    fn generate_composed_sdl(&self, composed: &ComposedSchema) -> Result<String> {
        let mut sdl = String::new();

        // Add federation directives
        sdl.push_str("directive @key(fields: String!) on OBJECT | INTERFACE\n");
        sdl.push_str("directive @external on FIELD_DEFINITION\n");
        sdl.push_str("directive @requires(fields: String!) on FIELD_DEFINITION\n");
        sdl.push_str("directive @provides(fields: String!) on FIELD_DEFINITION\n\n");

        // Add entity types
        for (type_name, entity_info) in &composed.entity_types {
            sdl.push_str(&format!(
                "type {} @key(fields: \"{}\") {{\n",
                type_name,
                entity_info.key_fields.join(" ")
            ));
            sdl.push_str("  # Entity fields would be listed here\n");
            sdl.push_str("}\n\n");
        }

        Ok(sdl)
    }

    /// Validate federated composition for consistency
    fn validate_federated_composition(&self, composed: &ComposedSchema) -> Result<()> {
        debug!("Validating federated schema composition");

        // Check that all required fields are satisfied
        for (field_key, ownership) in &composed.field_ownership {
            if let FieldOwnershipType::Requires(required_fields) = ownership {
                for required_field in required_fields {
                    let required_key = format!(
                        "{}.{}",
                        field_key.split('.').next().unwrap_or(""),
                        required_field
                    );
                    if !composed.field_ownership.contains_key(&required_key) {
                        return Err(anyhow!(
                            "Required field {} not found for {}",
                            required_field,
                            field_key
                        ));
                    }
                }
            }
        }

        // Validate entity key fields exist
        for (type_name, entity_info) in &composed.entity_types {
            for key_field in &entity_info.key_fields {
                let field_key = format!("{}.{}", type_name, key_field);
                if !composed.field_ownership.contains_key(&field_key) {
                    return Err(anyhow!(
                        "Key field {} not found for entity {}",
                        key_field,
                        type_name
                    ));
                }
            }
        }

        info!("Federated schema composition validation successful");
        Ok(())
    }

    /// Merge directives from multiple type definitions
    fn merge_directives(&self, existing: &[DirectiveUsage], new: &[DirectiveUsage]) -> Vec<DirectiveUsage> {
        let mut merged = existing.to_vec();
        
        // Add new directives that don't already exist
        for directive in new {
            if !merged.iter().any(|d| d.name == directive.name) {
                merged.push(directive.clone());
            }
        }
        
        // Sort for consistency
        merged.sort_by(|a, b| a.name.cmp(&b.name));
        merged
    }
}

impl Default for GraphQLFederation {
    fn default() -> Self {
        Self::new()
    }
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

impl GraphQLFederation {
    /// Merge directives from multiple type definitions
    fn merge_directives(&self, existing: &[DirectiveUsage], new: &[DirectiveUsage]) -> Vec<DirectiveUsage> {
        let mut merged = existing.to_vec();
        
        // Add new directives that don't already exist
        for directive in new {
            if !merged.iter().any(|d| d.name == directive.name) {
                merged.push(directive.clone());
            }
        }
        
        // Sort for consistency
        merged.sort_by(|a, b| a.name.cmp(&b.name));
        merged
    }

    /// Estimate memory usage for GraphQL response data
    fn estimate_memory_usage(&self, data: &QueryResultData, result_size: usize) -> u64 {
        // Basic heuristic: result size + overhead for JSON objects and arrays
        let base_size = result_size as u64;
        
        // Add overhead based on the complexity of the data structure
        let overhead_multiplier = match data {
            QueryResultData::ServiceResult(ref json_value) => {
                self.calculate_json_complexity_multiplier(json_value)
            }
            QueryResultData::AggregateResult(_) => 1.2, // Minimal overhead for aggregates
            QueryResultData::TripleResult(_) => 1.1,    // Minimal overhead for triples
        };
        
        (base_size as f64 * overhead_multiplier) as u64
    }

    /// Calculate complexity multiplier for JSON data
    fn calculate_json_complexity_multiplier(&self, value: &serde_json::Value) -> f64 {
        match value {
            serde_json::Value::Object(obj) => {
                1.3 + (obj.len() as f64 * 0.1) // Base overhead + field overhead
            }
            serde_json::Value::Array(arr) => {
                1.2 + (arr.len() as f64 * 0.05) // Base overhead + element overhead
            }
            _ => 1.0, // No overhead for primitives
        }
    }

    /// Parse variables from GraphQL query
    fn parse_variables(&self, query: &str) -> Result<HashMap<String, serde_json::Value>> {
        let mut variables = HashMap::new();
        
        // Look for variable definitions in the operation signature
        // Example: query GetUser($id: ID!, $includeProfile: Boolean = false)
        if let Some(start) = query.find('(') {
            if let Some(end) = query.find(')') {
                let var_section = &query[start + 1..end];
                
                // Split by commas and parse each variable
                for var_def in var_section.split(',') {
                    let var_def = var_def.trim();
                    if var_def.starts_with('$') {
                        if let Some(colon_pos) = var_def.find(':') {
                            let var_name = var_def[1..colon_pos].trim().to_string();
                            
                            // Check for default value
                            let type_and_default = &var_def[colon_pos + 1..];
                            if let Some(eq_pos) = type_and_default.find('=') {
                                let default_value = type_and_default[eq_pos + 1..].trim();
                                // Parse default value (simplified)
                                let parsed_value = self.parse_variable_value(default_value)?;
                                variables.insert(var_name, parsed_value);
                            } else {
                                // No default value, set to null
                                variables.insert(var_name, serde_json::Value::Null);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(variables)
    }

    /// Parse a variable value from string
    fn parse_variable_value(&self, value: &str) -> Result<serde_json::Value> {
        let value = value.trim();
        
        if value == "true" {
            Ok(serde_json::Value::Bool(true))
        } else if value == "false" {
            Ok(serde_json::Value::Bool(false))
        } else if value == "null" {
            Ok(serde_json::Value::Null)
        } else if value.starts_with('"') && value.ends_with('"') {
            Ok(serde_json::Value::String(value[1..value.len()-1].to_string()))
        } else if let Ok(num) = value.parse::<i64>() {
            Ok(serde_json::Value::Number(serde_json::Number::from(num)))
        } else if let Ok(num) = value.parse::<f64>() {
            Ok(serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap_or_else(|| serde_json::Number::from(0))))
        } else {
            // Assume it's a string without quotes
            Ok(serde_json::Value::String(value.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_graphql_federation_creation() {
        let federation = GraphQLFederation::new();
        assert!(federation.config.enable_schema_stitching);
    }

    #[tokio::test]
    async fn test_schema_registration() {
        let federation = GraphQLFederation::new();

        let schema = FederatedSchema {
            service_id: "test-service".to_string(),
            types: HashMap::new(),
            queries: HashMap::new(),
            mutations: HashMap::new(),
            subscriptions: HashMap::new(),
            directives: HashMap::new(),
        };

        let result = federation
            .register_schema("test-service".to_string(), schema)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_unified_schema_creation() {
        let federation = GraphQLFederation::new();

        // Register an empty schema
        let schema = FederatedSchema {
            service_id: "test-service".to_string(),
            types: HashMap::new(),
            queries: HashMap::new(),
            mutations: HashMap::new(),
            subscriptions: HashMap::new(),
            directives: HashMap::new(),
        };

        federation
            .register_schema("test-service".to_string(), schema)
            .await
            .unwrap();

        let unified = federation.create_unified_schema().await;
        assert!(unified.is_ok());
    }

    #[test]
    fn test_builtin_type_check() {
        let federation = GraphQLFederation::new();

        assert!(federation.is_builtin_type("String"));
        assert!(federation.is_builtin_type("Int"));
        assert!(federation.is_builtin_type("Boolean"));
        assert!(!federation.is_builtin_type("CustomType"));
    }

    #[tokio::test]
    async fn test_result_stitching() {
        let federation = GraphQLFederation::new();

        let service_results = vec![
            ServiceResult {
                service_id: "service1".to_string(),
                response: GraphQLResponse {
                    data: serde_json::json!({"field1": "value1"}),
                    errors: vec![],
                    extensions: None,
                },
            },
            ServiceResult {
                service_id: "service2".to_string(),
                response: GraphQLResponse {
                    data: serde_json::json!({"field2": "value2"}),
                    errors: vec![],
                    extensions: None,
                },
            },
        ];

        let result = federation.stitch_results(service_results).await;
        assert!(result.is_ok());

        let stitched = result.unwrap();
        assert!(stitched.data.as_object().unwrap().contains_key("field1"));
        assert!(stitched.data.as_object().unwrap().contains_key("field2"));
    }
}
