//! GraphQL Federation and Schema Stitching Support
//!
//! This module provides federation capabilities for OxiRS GraphQL, including:
//! - Remote schema introspection
//! - Schema merging and composition
//! - Cross-service query planning
//! - RDF dataset federation

use anyhow::{Result, Context};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::types::{Schema, GraphQLType, ObjectType, FieldType};
use crate::ast::{Document, OperationDefinition, OperationType, SelectionSet};
use crate::introspection::IntrospectionQuery;

/// Remote GraphQL service endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteEndpoint {
    /// Service identifier
    pub id: String,
    /// GraphQL endpoint URL
    pub url: String,
    /// Optional authentication header
    pub auth_header: Option<String>,
    /// Service namespace for type prefixing
    pub namespace: Option<String>,
    /// Timeout in seconds
    pub timeout_secs: u64,
}

/// Federation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// Remote endpoints to federate
    pub endpoints: Vec<RemoteEndpoint>,
    /// Enable schema caching
    pub enable_schema_cache: bool,
    /// Schema cache TTL in seconds
    pub schema_cache_ttl: u64,
    /// Enable query result caching
    pub enable_result_cache: bool,
    /// Query result cache TTL in seconds
    pub result_cache_ttl: u64,
    /// Maximum federation depth for nested queries
    pub max_federation_depth: usize,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            endpoints: Vec::new(),
            enable_schema_cache: true,
            schema_cache_ttl: 3600, // 1 hour
            enable_result_cache: true,
            result_cache_ttl: 300, // 5 minutes
            max_federation_depth: 3,
        }
    }
}

/// Schema stitching engine for merging multiple GraphQL schemas
pub struct SchemaStitcher {
    /// Local schema
    local_schema: Arc<Schema>,
    /// Remote schemas cache
    remote_schemas: Arc<RwLock<HashMap<String, CachedSchema>>>,
    /// HTTP client for remote introspection
    http_client: reqwest::Client,
}

#[derive(Debug, Clone)]
struct CachedSchema {
    schema: Schema,
    cached_at: std::time::Instant,
    ttl: std::time::Duration,
}

impl CachedSchema {
    fn is_expired(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }
}

impl SchemaStitcher {
    pub fn new(local_schema: Arc<Schema>) -> Self {
        Self {
            local_schema,
            remote_schemas: Arc::new(RwLock::new(HashMap::new())),
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Introspect a remote GraphQL endpoint
    pub async fn introspect_remote(&self, endpoint: &RemoteEndpoint) -> Result<Schema> {
        // Check cache first
        {
            let cache = self.remote_schemas.read().await;
            if let Some(cached) = cache.get(&endpoint.id) {
                if !cached.is_expired() {
                    return Ok(cached.schema.clone());
                }
            }
        }

        // Build introspection query
        let introspection_query = IntrospectionQuery::full_query();
        
        let mut request = self.http_client
            .post(&endpoint.url)
            .json(&serde_json::json!({
                "query": introspection_query,
                "variables": {}
            }));

        // Add authentication if provided
        if let Some(auth) = &endpoint.auth_header {
            request = request.header("Authorization", auth);
        }

        // Execute request
        let response = request
            .timeout(std::time::Duration::from_secs(endpoint.timeout_secs))
            .send()
            .await
            .context("Failed to send introspection request")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Remote introspection failed with status: {}",
                response.status()
            ));
        }

        let introspection_result: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse introspection response")?;

        // Parse introspection result into Schema
        let schema = self.parse_introspection_result(introspection_result, endpoint)?;

        // Cache the schema
        {
            let mut cache = self.remote_schemas.write().await;
            cache.insert(
                endpoint.id.clone(),
                CachedSchema {
                    schema: schema.clone(),
                    cached_at: std::time::Instant::now(),
                    ttl: std::time::Duration::from_secs(3600), // 1 hour cache
                },
            );
        }

        Ok(schema)
    }

    /// Parse introspection result into a Schema
    fn parse_introspection_result(
        &self,
        result: serde_json::Value,
        endpoint: &RemoteEndpoint,
    ) -> Result<Schema> {
        let data = result
            .get("data")
            .ok_or_else(|| anyhow::anyhow!("Missing data in introspection result"))?;
        
        let schema_data = data
            .get("__schema")
            .ok_or_else(|| anyhow::anyhow!("Missing __schema in introspection result"))?;

        let mut schema = Schema::new();

        // Parse types
        if let Some(types) = schema_data.get("types").and_then(|t| t.as_array()) {
            for type_def in types {
                if let Ok(gql_type) = self.parse_type_from_introspection(type_def, endpoint) {
                    schema.add_type(gql_type);
                }
            }
        }

        // Set root types
        if let Some(query_type) = schema_data.get("queryType") {
            if let Some(name) = query_type.get("name").and_then(|n| n.as_str()) {
                schema.set_query_type(self.namespace_type_name(name, endpoint));
            }
        }

        if let Some(mutation_type) = schema_data.get("mutationType") {
            if let Some(name) = mutation_type.get("name").and_then(|n| n.as_str()) {
                schema.set_mutation_type(self.namespace_type_name(name, endpoint));
            }
        }

        if let Some(subscription_type) = schema_data.get("subscriptionType") {
            if let Some(name) = subscription_type.get("name").and_then(|n| n.as_str()) {
                schema.set_subscription_type(self.namespace_type_name(name, endpoint));
            }
        }

        Ok(schema)
    }

    /// Parse a single type from introspection data
    fn parse_type_from_introspection(
        &self,
        type_def: &serde_json::Value,
        endpoint: &RemoteEndpoint,
    ) -> Result<GraphQLType> {
        let name = type_def
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| anyhow::anyhow!("Type missing name"))?;

        let kind = type_def
            .get("kind")
            .and_then(|k| k.as_str())
            .ok_or_else(|| anyhow::anyhow!("Type missing kind"))?;

        match kind {
            "OBJECT" => {
                let mut object_type = ObjectType::new(self.namespace_type_name(name, endpoint));
                
                if let Some(desc) = type_def.get("description").and_then(|d| d.as_str()) {
                    object_type = object_type.with_description(desc.to_string());
                }

                // Parse fields
                if let Some(fields) = type_def.get("fields").and_then(|f| f.as_array()) {
                    for field_def in fields {
                        if let Ok((field_name, field_type)) = self.parse_field_from_introspection(field_def, endpoint) {
                            object_type = object_type.with_field(field_name, field_type);
                        }
                    }
                }

                // Parse interfaces
                if let Some(interfaces) = type_def.get("interfaces").and_then(|i| i.as_array()) {
                    for interface in interfaces {
                        if let Some(interface_name) = interface.get("name").and_then(|n| n.as_str()) {
                            object_type = object_type.with_interface(
                                self.namespace_type_name(interface_name, endpoint)
                            );
                        }
                    }
                }

                Ok(GraphQLType::Object(object_type))
            }
            "INTERFACE" => {
                let mut interface_type = crate::types::InterfaceType::new(
                    self.namespace_type_name(name, endpoint)
                );
                
                if let Some(desc) = type_def.get("description").and_then(|d| d.as_str()) {
                    interface_type = interface_type.with_description(desc.to_string());
                }

                // Parse fields
                if let Some(fields) = type_def.get("fields").and_then(|f| f.as_array()) {
                    for field_def in fields {
                        if let Ok((field_name, field_type)) = self.parse_field_from_introspection(field_def, endpoint) {
                            interface_type = interface_type.with_field(field_name, field_type);
                        }
                    }
                }

                Ok(GraphQLType::Interface(interface_type))
            }
            "UNION" => {
                let mut union_type = crate::types::UnionType::new(
                    self.namespace_type_name(name, endpoint)
                );
                
                if let Some(desc) = type_def.get("description").and_then(|d| d.as_str()) {
                    union_type = union_type.with_description(desc.to_string());
                }

                // Parse possible types
                if let Some(possible_types) = type_def.get("possibleTypes").and_then(|p| p.as_array()) {
                    for possible_type in possible_types {
                        if let Some(type_name) = possible_type.get("name").and_then(|n| n.as_str()) {
                            union_type = union_type.with_type(
                                self.namespace_type_name(type_name, endpoint)
                            );
                        }
                    }
                }

                Ok(GraphQLType::Union(union_type))
            }
            "ENUM" => {
                let mut enum_type = crate::types::EnumType::new(
                    self.namespace_type_name(name, endpoint)
                );
                
                if let Some(desc) = type_def.get("description").and_then(|d| d.as_str()) {
                    enum_type = enum_type.with_description(desc.to_string());
                }

                // Parse enum values
                if let Some(enum_values) = type_def.get("enumValues").and_then(|e| e.as_array()) {
                    for enum_value_def in enum_values {
                        if let Some(value_name) = enum_value_def.get("name").and_then(|n| n.as_str()) {
                            let mut enum_value = crate::types::EnumValue::new(
                                value_name.to_string(),
                                crate::ast::Value::EnumValue(value_name.to_string())
                            );
                            
                            if let Some(value_desc) = enum_value_def.get("description").and_then(|d| d.as_str()) {
                                enum_value = enum_value.with_description(value_desc.to_string());
                            }
                            
                            if let Some(deprecated) = enum_value_def.get("isDeprecated").and_then(|d| d.as_bool()) {
                                if deprecated {
                                    let reason = enum_value_def.get("deprecationReason")
                                        .and_then(|r| r.as_str())
                                        .unwrap_or("Deprecated")
                                        .to_string();
                                    enum_value = enum_value.with_deprecation(reason);
                                }
                            }
                            
                            enum_type = enum_type.with_value(value_name.to_string(), enum_value);
                        }
                    }
                }

                Ok(GraphQLType::Enum(enum_type))
            }
            "INPUT_OBJECT" => {
                let mut input_type = crate::types::InputObjectType::new(
                    self.namespace_type_name(name, endpoint)
                );
                
                if let Some(desc) = type_def.get("description").and_then(|d| d.as_str()) {
                    input_type = input_type.with_description(desc.to_string());
                }

                // Parse input fields
                if let Some(input_fields) = type_def.get("inputFields").and_then(|f| f.as_array()) {
                    for field_def in input_fields {
                        if let Ok((field_name, arg_type)) = self.parse_input_field_from_introspection(field_def, endpoint) {
                            input_type = input_type.with_field(field_name, arg_type);
                        }
                    }
                }

                Ok(GraphQLType::InputObject(input_type))
            }
            "SCALAR" => {
                // Skip built-in scalars, they're already in the schema
                if matches!(name, "String" | "Int" | "Float" | "Boolean" | "ID") {
                    return Err(anyhow::anyhow!("Built-in scalar type: {}", name));
                }
                
                let mut scalar_type = crate::types::ScalarType::new(
                    self.namespace_type_name(name, endpoint)
                );
                
                if let Some(desc) = type_def.get("description").and_then(|d| d.as_str()) {
                    scalar_type = scalar_type.with_description(desc.to_string());
                }

                Ok(GraphQLType::Scalar(scalar_type))
            }
            _ => {
                Err(anyhow::anyhow!("Unsupported type kind: {}", kind))
            }
        }
    }

    /// Parse a field from introspection data
    fn parse_field_from_introspection(
        &self,
        field_def: &serde_json::Value,
        endpoint: &RemoteEndpoint,
    ) -> Result<(String, FieldType)> {
        let name = field_def
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| anyhow::anyhow!("Field missing name"))?;

        let type_def = field_def
            .get("type")
            .ok_or_else(|| anyhow::anyhow!("Field missing type"))?;
        
        let field_gql_type = self.parse_type_ref_from_introspection(type_def, endpoint)?;
        
        let mut field_type = FieldType::new(name.to_string(), field_gql_type);
        
        // Add description if present
        if let Some(desc) = field_def.get("description").and_then(|d| d.as_str()) {
            field_type = field_type.with_description(desc.to_string());
        }
        
        // Parse arguments if present
        if let Some(args) = field_def.get("args").and_then(|a| a.as_array()) {
            for arg_def in args {
                if let Ok((arg_name, arg_type)) = self.parse_argument_from_introspection(arg_def, endpoint) {
                    field_type = field_type.with_argument(arg_name, arg_type);
                }
            }
        }

        Ok((name.to_string(), field_type))
    }

    /// Parse type reference from introspection data
    fn parse_type_ref_from_introspection(
        &self,
        type_ref: &serde_json::Value,
        endpoint: &RemoteEndpoint,
    ) -> Result<GraphQLType> {
        let kind = type_ref
            .get("kind")
            .and_then(|k| k.as_str())
            .ok_or_else(|| anyhow::anyhow!("Type ref missing kind"))?;

        match kind {
            "SCALAR" => {
                let name = type_ref
                    .get("name")
                    .and_then(|n| n.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Scalar type missing name"))?;
                
                // Map to built-in scalars or create custom scalar
                match name {
                    "String" => Ok(GraphQLType::Scalar(crate::types::BuiltinScalars::string())),
                    "Int" => Ok(GraphQLType::Scalar(crate::types::BuiltinScalars::int())),
                    "Float" => Ok(GraphQLType::Scalar(crate::types::BuiltinScalars::float())),
                    "Boolean" => Ok(GraphQLType::Scalar(crate::types::BuiltinScalars::boolean())),
                    "ID" => Ok(GraphQLType::Scalar(crate::types::BuiltinScalars::id())),
                    _ => Ok(GraphQLType::Scalar(crate::types::ScalarType::new(
                        self.namespace_type_name(name, endpoint)
                    ))),
                }
            }
            "OBJECT" => {
                let name = type_ref
                    .get("name")
                    .and_then(|n| n.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Object type missing name"))?;
                
                Ok(GraphQLType::Object(ObjectType::new(
                    self.namespace_type_name(name, endpoint)
                )))
            }
            "LIST" => {
                let of_type = type_ref
                    .get("ofType")
                    .ok_or_else(|| anyhow::anyhow!("List type missing ofType"))?;
                
                let inner_type = self.parse_type_ref_from_introspection(of_type, endpoint)?;
                Ok(GraphQLType::List(Box::new(inner_type)))
            }
            "NON_NULL" => {
                let of_type = type_ref
                    .get("ofType")
                    .ok_or_else(|| anyhow::anyhow!("NonNull type missing ofType"))?;
                
                let inner_type = self.parse_type_ref_from_introspection(of_type, endpoint)?;
                Ok(GraphQLType::NonNull(Box::new(inner_type)))
            }
            _ => Err(anyhow::anyhow!("Unsupported type kind: {}", kind)),
        }
    }
    
    /// Parse argument from introspection data
    fn parse_argument_from_introspection(
        &self,
        arg_def: &serde_json::Value,
        endpoint: &RemoteEndpoint,
    ) -> Result<(String, crate::types::ArgumentType)> {
        let name = arg_def
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| anyhow::anyhow!("Argument missing name"))?;
        
        let type_def = arg_def
            .get("type")
            .ok_or_else(|| anyhow::anyhow!("Argument missing type"))?;
            
        let arg_type = self.parse_type_ref_from_introspection(type_def, endpoint)?;
        
        let mut argument = crate::types::ArgumentType::new(name.to_string(), arg_type);
        
        // Add description if present
        if let Some(desc) = arg_def.get("description").and_then(|d| d.as_str()) {
            argument = argument.with_description(desc.to_string());
        }
        
        // Add default value if present
        if let Some(default) = arg_def.get("defaultValue").and_then(|d| d.as_str()) {
            // Parse the default value string into an AST Value
            if let Ok(value) = self.parse_default_value(default) {
                argument = argument.with_default_value(value);
            }
        }
        
        Ok((name.to_string(), argument))
    }
    
    /// Parse input field from introspection data
    fn parse_input_field_from_introspection(
        &self,
        field_def: &serde_json::Value,
        endpoint: &RemoteEndpoint,
    ) -> Result<(String, crate::types::InputFieldType)> {
        let name = field_def
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| anyhow::anyhow!("Input field missing name"))?;
        
        let type_def = field_def
            .get("type")
            .ok_or_else(|| anyhow::anyhow!("Input field missing type"))?;
            
        let field_type = self.parse_type_ref_from_introspection(type_def, endpoint)?;
        
        let mut input_field = crate::types::InputFieldType::new(name.to_string(), field_type);
        
        // Add description if present
        if let Some(desc) = field_def.get("description").and_then(|d| d.as_str()) {
            input_field = input_field.with_description(desc.to_string());
        }
        
        // Add default value if present
        if let Some(default) = field_def.get("defaultValue").and_then(|d| d.as_str()) {
            // Parse the default value string into an AST Value
            if let Ok(value) = self.parse_default_value(default) {
                input_field = input_field.with_default_value(value);
            }
        }
        
        Ok((name.to_string(), input_field))
    }

    /// Parse default value string into AST Value
    fn parse_default_value(&self, default_str: &str) -> Result<crate::ast::Value> {
        // Simple parsing for common cases
        if default_str == "null" {
            Ok(crate::ast::Value::NullValue)
        } else if default_str == "true" {
            Ok(crate::ast::Value::BooleanValue(true))
        } else if default_str == "false" {
            Ok(crate::ast::Value::BooleanValue(false))
        } else if let Ok(i) = default_str.parse::<i32>() {
            Ok(crate::ast::Value::IntValue(i as i64))
        } else if let Ok(f) = default_str.parse::<f64>() {
            Ok(crate::ast::Value::FloatValue(f))
        } else if default_str.starts_with('"') && default_str.ends_with('"') {
            Ok(crate::ast::Value::StringValue(default_str[1..default_str.len()-1].to_string()))
        } else {
            // Treat as enum value
            Ok(crate::ast::Value::EnumValue(default_str.to_string()))
        }
    }

    /// Apply namespace to type name if configured
    fn namespace_type_name(&self, name: &str, endpoint: &RemoteEndpoint) -> String {
        if let Some(namespace) = &endpoint.namespace {
            format!("{}_{}", namespace, name)
        } else {
            name.to_string()
        }
    }

    /// Merge multiple schemas with conflict resolution
    pub async fn merge_schemas(
        &self,
        endpoints: &[RemoteEndpoint],
        config: &FederationConfig,
    ) -> Result<Schema> {
        let mut merged_schema = self.local_schema.as_ref().clone();
        let mut type_conflicts = HashMap::new();

        // Introspect all remote endpoints
        for endpoint in endpoints {
            match self.introspect_remote(endpoint).await {
                Ok(remote_schema) => {
                    // Merge types from remote schema
                    for (type_name, gql_type) in &remote_schema.types {
                        if merged_schema.types.contains_key(type_name) {
                            // Handle type conflict
                            type_conflicts.entry(type_name.clone())
                                .or_insert_with(Vec::new)
                                .push(endpoint.id.clone());
                        } else {
                            merged_schema.add_type(gql_type.clone());
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to introspect endpoint {}: {}", endpoint.id, e);
                    // Continue with other endpoints
                }
            }
        }

        // Resolve type conflicts
        if !type_conflicts.is_empty() {
            self.resolve_type_conflicts(&mut merged_schema, type_conflicts)?;
        }

        Ok(merged_schema)
    }

    /// Resolve type naming conflicts between schemas
    fn resolve_type_conflicts(
        &self,
        schema: &mut Schema,
        conflicts: HashMap<String, Vec<String>>,
    ) -> Result<()> {
        for (type_name, conflicting_services) in conflicts {
            tracing::warn!(
                "Type conflict for '{}' between services: {:?}",
                type_name,
                conflicting_services
            );
            
            // Strategy: Keep local type, rename remote types with service prefix
            // TODO: Implement more sophisticated conflict resolution strategies
        }
        Ok(())
    }
}

/// Query planner for federated queries
pub struct QueryPlanner {
    schema_stitcher: Arc<SchemaStitcher>,
    config: FederationConfig,
}

impl QueryPlanner {
    pub fn new(schema_stitcher: Arc<SchemaStitcher>, config: FederationConfig) -> Self {
        Self {
            schema_stitcher,
            config,
        }
    }

    /// Plan execution of a federated query
    pub async fn plan_query(&self, query: &Document) -> Result<QueryPlan> {
        let mut plan = QueryPlan::new();

        // Analyze query to determine which services are needed
        for definition in &query.definitions {
            if let crate::ast::Definition::Operation(operation) = definition {
                match operation.operation_type {
                    OperationType::Query => {
                        self.plan_selection_set(&operation.selection_set, &mut plan, 0).await?;
                    }
                    OperationType::Mutation => {
                        // TODO: Handle mutations
                    }
                    OperationType::Subscription => {
                        // TODO: Handle subscriptions
                    }
                }
            }
        }

        Ok(plan)
    }

    /// Plan execution for a selection set
    async fn plan_selection_set(
        &self,
        selection_set: &SelectionSet,
        plan: &mut QueryPlan,
        depth: usize,
    ) -> Result<()> {
        if depth > self.config.max_federation_depth {
            return Err(anyhow::anyhow!("Maximum federation depth exceeded"));
        }

        // Get merged schema to determine field origins
        let merged_schema = self.schema_stitcher.merge_schemas(&self.config.endpoints, &self.config).await?;
        
        for selection in &selection_set.selections {
            match selection {
                crate::ast::Selection::Field(field) => {
                    self.plan_field_selection(field, plan, depth, &merged_schema).await?;
                }
                crate::ast::Selection::InlineFragment(fragment) => {
                    // Handle inline fragments
                    Box::pin(self.plan_selection_set(&fragment.selection_set, plan, depth + 1)).await?;
                }
                crate::ast::Selection::FragmentSpread(spread) => {
                    // Handle fragment spreads (requires fragment definition lookup)
                    tracing::warn!("Fragment spread '{}' not fully implemented", spread.fragment_name);
                }
            }
        }
        
        Ok(())
    }
    
    /// Plan execution for a single field selection
    async fn plan_field_selection(
        &self,
        field: &crate::ast::Field,
        plan: &mut QueryPlan,
        depth: usize,
        schema: &Schema,
    ) -> Result<()> {
        let field_name = &field.name;
        
        // Determine which service owns this field
        let service_id = self.determine_field_service(field_name, schema)?;
        
        // Check if we already have a step for this service
        let existing_step = plan.steps.iter().find(|step| step.service_id == service_id);
        
        if existing_step.is_none() {
            // Create new query step for this service
            let step_id = format!("step_{}_{}", service_id, plan.steps.len());
            let query_fragment = self.build_query_fragment(field, schema)?;
            
            let step = QueryStep {
                id: step_id.clone(),
                service_id: service_id.clone(),
                query_fragment,
                variables: self.extract_field_variables(field),
                parent_extractions: Vec::new(),
            };
            
            plan.steps.push(step);
            
            // Add dependency tracking
            if plan.steps.len() > 1 {
                plan.dependencies.entry(step_id)
                    .or_insert_with(Vec::new);
            }
        }
        
        // Recursively plan nested selections
        if let Some(nested_selections) = &field.selection_set {
            Box::pin(self.plan_selection_set(nested_selections, plan, depth + 1)).await?;
        }
        
        Ok(())
    }
    
    /// Determine which service owns a field
    fn determine_field_service(&self, field_name: &str, schema: &Schema) -> Result<String> {
        // Check if field exists in local schema
        if let Some(query_type_name) = &schema.query_type {
            if let Some(GraphQLType::Object(query_type)) = schema.types.get(query_type_name) {
                if query_type.fields.contains_key(field_name) {
                    return Ok("local".to_string());
                }
            }
        }
        
        // Check remote services by namespace
        for endpoint in &self.config.endpoints {
            if let Some(namespace) = &endpoint.namespace {
                let namespaced_field = format!("{}_{}", namespace, field_name);
                if let Some(query_type_name) = &schema.query_type {
                    if let Some(GraphQLType::Object(query_type)) = schema.types.get(query_type_name) {
                        if query_type.fields.contains_key(&namespaced_field) {
                            return Ok(endpoint.id.clone());
                        }
                    }
                }
            }
        }
        
        // Default to local service if not found
        Ok("local".to_string())
    }
    
    /// Build a GraphQL query fragment for a field
    fn build_query_fragment(&self, field: &crate::ast::Field, _schema: &Schema) -> Result<String> {
        let mut fragment = String::new();
        
        // Add field name
        fragment.push_str(&field.name);
        
        // Add arguments if present
        if !field.arguments.is_empty() {
            fragment.push('(');
            let args: Vec<String> = field.arguments.iter()
                .map(|arg| format!("{}: {}", arg.name, self.value_to_string(&arg.value)))
                .collect();
            fragment.push_str(&args.join(", "));
            fragment.push(')');
        }
        
        // Add selection set if present
        if let Some(selection_set) = &field.selection_set {
            fragment.push_str(" {\n");
            for selection in &selection_set.selections {
                match selection {
                    crate::ast::Selection::Field(nested_field) => {
                        let nested_fragment = self.build_query_fragment(nested_field, _schema)?;
                        fragment.push_str("  ");
                        fragment.push_str(&nested_fragment);
                        fragment.push('\n');
                    }
                    crate::ast::Selection::InlineFragment(_) => {
                        fragment.push_str("  # inline fragment\n");
                    }
                    crate::ast::Selection::FragmentSpread(spread) => {
                        fragment.push_str(&format!("  ...{}\n", spread.fragment_name));
                    }
                }
            }
            fragment.push('}');
        }
        
        Ok(fragment)
    }
    
    /// Convert AST Value to string representation
    fn value_to_string(&self, value: &crate::ast::Value) -> String {
        match value {
            crate::ast::Value::Variable(var) => format!("${}", var.name),
            crate::ast::Value::IntValue(i) => i.to_string(),
            crate::ast::Value::FloatValue(f) => f.to_string(),
            crate::ast::Value::StringValue(s) => format!("\"{}\"", s),
            crate::ast::Value::BooleanValue(b) => b.to_string(),
            crate::ast::Value::NullValue => "null".to_string(),
            crate::ast::Value::EnumValue(e) => e.clone(),
            crate::ast::Value::ListValue(list) => {
                let items: Vec<String> = list.iter()
                    .map(|v| self.value_to_string(v))
                    .collect();
                format!("[{}]", items.join(", "))
            }
            crate::ast::Value::ObjectValue(obj) => {
                let fields: Vec<String> = obj.iter()
                    .map(|(k, v)| format!("{}: {}", k, self.value_to_string(v)))
                    .collect();
                format!("{{{}}}", fields.join(", "))
            }
        }
    }
    
    /// Extract variables used in a field
    fn extract_field_variables(&self, field: &crate::ast::Field) -> HashMap<String, serde_json::Value> {
        let mut variables = HashMap::new();
        
        // Extract from arguments
        for arg in &field.arguments {
            self.extract_variables_from_value(&arg.value, &mut variables);
        }
        
        // Extract from nested fields
        if let Some(selection_set) = &field.selection_set {
            for selection in &selection_set.selections {
                if let crate::ast::Selection::Field(nested_field) = selection {
                    let nested_vars = self.extract_field_variables(nested_field);
                    variables.extend(nested_vars);
                }
            }
        }
        
        variables
    }
    
    /// Extract variables from an AST value
    fn extract_variables_from_value(&self, value: &crate::ast::Value, variables: &mut HashMap<String, serde_json::Value>) {
        match value {
            crate::ast::Value::Variable(var) => {
                // Add placeholder - actual values would come from query variables
                variables.insert(var.name.clone(), serde_json::Value::Null);
            }
            crate::ast::Value::ListValue(list) => {
                for item in list {
                    self.extract_variables_from_value(item, variables);
                }
            }
            crate::ast::Value::ObjectValue(obj) => {
                for (_, val) in obj {
                    self.extract_variables_from_value(val, variables);
                }
            }
            _ => {} // Other value types don't contain variables
        }
    }
}

/// Execution plan for a federated query
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Steps to execute in order
    pub steps: Vec<QueryStep>,
    /// Dependencies between steps
    pub dependencies: HashMap<String, Vec<String>>,
}

impl QueryPlan {
    fn new() -> Self {
        Self {
            steps: Vec::new(),
            dependencies: HashMap::new(),
        }
    }
}

/// A single step in query execution
#[derive(Debug, Clone)]
pub struct QueryStep {
    /// Unique identifier for this step
    pub id: String,
    /// Service that will execute this step
    pub service_id: String,
    /// Query fragment to execute
    pub query_fragment: String,
    /// Variables needed for this step
    pub variables: HashMap<String, serde_json::Value>,
    /// Fields to extract from parent results
    pub parent_extractions: Vec<String>,
}

/// RDF Dataset Federation
pub struct DatasetFederation {
    /// Local store reference
    local_store: Arc<crate::RdfStore>,
    /// Remote SPARQL endpoints
    remote_endpoints: Vec<SparqlEndpoint>,
    /// HTTP client for SPARQL requests
    http_client: reqwest::Client,
}

#[derive(Debug, Clone)]
pub struct SparqlEndpoint {
    pub id: String,
    pub url: String,
    pub auth: Option<String>,
    pub timeout_secs: u64,
}

impl DatasetFederation {
    pub fn new(local_store: Arc<crate::RdfStore>) -> Self {
        Self {
            local_store,
            remote_endpoints: Vec::new(),
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Add a remote SPARQL endpoint
    pub fn add_endpoint(&mut self, endpoint: SparqlEndpoint) {
        self.remote_endpoints.push(endpoint);
    }

    /// Execute a federated SPARQL query with SERVICE clauses
    pub async fn execute_federated_query(&self, query: &str) -> Result<crate::QueryResults> {
        // Parse query to identify SERVICE clauses
        let service_patterns = self.extract_service_patterns(query)?;
        
        if service_patterns.is_empty() {
            // No federation needed, execute locally
            return self.local_store.query(query);
        }

        // Plan federated execution
        let execution_plan = self.plan_federated_execution(query, service_patterns)?;
        
        // Execute plan
        self.execute_plan(execution_plan).await
    }

    /// Extract SERVICE patterns from SPARQL query
    fn extract_service_patterns(&self, query: &str) -> Result<Vec<ServicePattern>> {
        let mut patterns = Vec::new();
        let mut in_service = false;
        let mut service_uri = String::new();
        let mut pattern_start = 0;
        let mut brace_count = 0;
        
        // Simple regex-based extraction (should use proper SPARQL parser in production)
        let service_regex = regex::Regex::new(r"SERVICE\s*<([^>]+)>\s*\{")?;
        
        for cap in service_regex.captures_iter(query) {
            if let Some(uri_match) = cap.get(1) {
                service_uri = uri_match.as_str().to_string();
                pattern_start = cap.get(0).unwrap().end();
                in_service = true;
                brace_count = 1;
                
                // Find the closing brace for this SERVICE block
                let rest = &query[pattern_start..];
                let mut pattern_end = pattern_start;
                
                for (i, ch) in rest.chars().enumerate() {
                    match ch {
                        '{' => brace_count += 1,
                        '}' => {
                            brace_count -= 1;
                            if brace_count == 0 {
                                pattern_end = pattern_start + i;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                
                if pattern_end > pattern_start {
                    let pattern = query[pattern_start..pattern_end].trim().to_string();
                    patterns.push(ServicePattern {
                        service_uri: service_uri.clone(),
                        pattern,
                        start_pos: cap.get(0).unwrap().start(),
                        end_pos: pattern_end + 1, // Include the closing brace
                    });
                }
            }
        }
        
        Ok(patterns)
    }

    /// Plan execution of federated SPARQL query
    fn plan_federated_execution(
        &self,
        query: &str,
        service_patterns: Vec<ServicePattern>,
    ) -> Result<FederatedExecutionPlan> {
        let mut steps = Vec::new();
        let mut modified_query = query.to_string();
        
        // Sort patterns by position (reverse order to modify query correctly)
        let mut sorted_patterns = service_patterns;
        sorted_patterns.sort_by(|a, b| b.start_pos.cmp(&a.start_pos));
        
        for (idx, pattern) in sorted_patterns.iter().enumerate() {
            // Find endpoint for this service URI
            let endpoint = self.remote_endpoints.iter()
                .find(|e| e.url == pattern.service_uri)
                .ok_or_else(|| anyhow::anyhow!("Unknown service URI: {}", pattern.service_uri))?;
            
            // Replace SERVICE block with a placeholder
            let placeholder = format!("{{ ?__service_result_{} }}", idx);
            modified_query.replace_range(pattern.start_pos..pattern.end_pos, &placeholder);
            
            // Create execution step
            steps.push(FederatedStep {
                step_id: format!("service_{}", idx),
                endpoint: endpoint.id.clone(),
                endpoint_url: endpoint.url.clone(),
                auth: endpoint.auth.clone(),
                query: format!("SELECT * WHERE {{ {} }}", pattern.pattern),
                result_variable: format!("?__service_result_{}", idx),
                timeout: std::time::Duration::from_secs(endpoint.timeout_secs),
            });
        }
        
        // Add final step for local query with placeholders
        steps.push(FederatedStep {
            step_id: "local_final".to_string(),
            endpoint: "local".to_string(),
            endpoint_url: String::new(),
            auth: None,
            query: modified_query,
            result_variable: String::new(),
            timeout: std::time::Duration::from_secs(60),
        });
        
        Ok(FederatedExecutionPlan { steps })
    }

    /// Execute a federated query plan
    async fn execute_plan(&self, plan: FederatedExecutionPlan) -> Result<crate::QueryResults> {
        let mut intermediate_results = HashMap::new();
        let mut final_result = None;
        
        for step in plan.steps {
            if step.endpoint == "local" {
                // Execute final query locally with substituted results
                let query = self.substitute_sparql_results(&step.query, &intermediate_results)?;
                final_result = Some(self.local_store.query(&query)?);
            } else {
                // Execute remote query
                let results = self.execute_remote_query(&step).await?;
                intermediate_results.insert(step.result_variable.clone(), results);
            }
        }
        
        final_result.ok_or_else(|| anyhow::anyhow!("No final result from federated query"))
    }
    
    /// Substitute intermediate results into SPARQL query
    fn substitute_sparql_results(
        &self,
        query: &str,
        intermediate_results: &HashMap<String, Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<String> {
        let mut substituted_query = query.to_string();
        
        for (var_name, results) in intermediate_results {
            // Find the placeholder variable in the query
            let placeholder = format!("{{ {} }}", var_name);
            
            if substituted_query.contains(&placeholder) {
                // Build VALUES clause from intermediate results
                let values_clause = self.build_values_clause(var_name, results)?;
                substituted_query = substituted_query.replace(&placeholder, &values_clause);
            }
        }
        
        Ok(substituted_query)
    }
    
    /// Build a SPARQL VALUES clause from intermediate results
    fn build_values_clause(
        &self,
        var_name: &str,
        results: &[HashMap<String, serde_json::Value>],
    ) -> Result<String> {
        if results.is_empty() {
            return Ok("".to_string());
        }
        
        // Extract variable names from first result
        let vars: Vec<String> = results[0].keys().cloned().collect();
        
        if vars.is_empty() {
            return Ok("".to_string());
        }
        
        let mut values_clause = format!("VALUES ( {} ) {{\n", 
            vars.iter().map(|v| format!("?{}", v)).collect::<Vec<_>>().join(" ")
        );
        
        // Add value rows
        for result in results {
            values_clause.push_str("  ( ");
            let values: Vec<String> = vars.iter()
                .map(|var| {
                    if let Some(value) = result.get(var) {
                        self.sparql_value_to_string(value)
                    } else {
                        "UNDEF".to_string()
                    }
                })
                .collect();
            values_clause.push_str(&values.join(" "));
            values_clause.push_str(" )\n");
        }
        
        values_clause.push('}');
        Ok(values_clause)
    }
    
    /// Convert JSON value to SPARQL value string
    fn sparql_value_to_string(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(s) => {
                // Check if it's an IRI (starts with http)
                if s.starts_with("http") {
                    format!("<{}>", s)
                } else {
                    format!("\"{}\"", s.replace('"', "\\\""))
                }
            }
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Null => "UNDEF".to_string(),
            serde_json::Value::Object(obj) => {
                // Handle RDF term objects with type and value
                if let (Some(term_type), Some(term_value)) = (obj.get("type"), obj.get("value")) {
                    match term_type.as_str() {
                        Some("uri") => format!("<{}>", term_value.as_str().unwrap_or("")),
                        Some("literal") => {
                            let value_str = term_value.as_str().unwrap_or("");
                            if let Some(datatype) = obj.get("datatype").and_then(|d| d.as_str()) {
                                format!("\"{}\"^^<{}>", value_str, datatype)
                            } else if let Some(lang) = obj.get("xml:lang").and_then(|l| l.as_str()) {
                                format!("\"{}\"@{}", value_str, lang)
                            } else {
                                format!("\"{}\"", value_str)
                            }
                        }
                        Some("bnode") => format!("_:{}", term_value.as_str().unwrap_or("blank")),
                        _ => format!("\"{}\"", term_value.as_str().unwrap_or(""))
                    }
                } else {
                    format!("\"{}\"", serde_json::to_string(obj).unwrap_or_default().replace('"', "\\\""))
                }
            }
            serde_json::Value::Array(_) => {
                // Arrays not directly supported in SPARQL VALUES
                "UNDEF".to_string()
            }
        }
    }
    
    /// Execute a query on a remote SPARQL endpoint
    async fn execute_remote_query(&self, step: &FederatedStep) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut request = self.http_client
            .post(&step.endpoint_url)
            .timeout(step.timeout)
            .header("Accept", "application/sparql-results+json")
            .form(&[(
                "query", step.query.as_str()
            )]);
        
        // Add authentication if provided
        if let Some(auth) = &step.auth {
            request = request.header("Authorization", auth);
        }
        
        let response = request.send().await
            .context("Failed to send SPARQL request")?;
        
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!(
                "Remote SPARQL query failed with status {}: {}",
                status,
                error_text
            ));
        }
        
        let results: serde_json::Value = response.json().await
            .context("Failed to parse SPARQL results")?;
        
        // Extract bindings from SPARQL JSON results
        let bindings = results
            .get("results")
            .and_then(|r| r.get("bindings"))
            .and_then(|b| b.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid SPARQL results format"))?;
        
        Ok(bindings.iter()
            .filter_map(|b| b.as_object())
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .collect())
    }
}

#[derive(Debug, Clone)]
struct ServicePattern {
    service_uri: String,
    pattern: String,
    start_pos: usize,
    end_pos: usize,
}

#[derive(Debug)]
struct FederatedExecutionPlan {
    steps: Vec<FederatedStep>,
}

#[derive(Debug, Clone)]
pub(crate) struct FederatedStep {
    step_id: String,
    endpoint: String,
    endpoint_url: String,
    auth: Option<String>,
    query: String,
    result_variable: String,
    timeout: std::time::Duration,
}

/// Cross-dataset join optimizer
pub struct JoinOptimizer {
    /// Statistics about remote endpoints
    endpoint_stats: Arc<RwLock<HashMap<String, EndpointStatistics>>>,
}

#[derive(Debug, Clone)]
pub(crate) struct EndpointStatistics {
    /// Average query response time
    avg_response_time: std::time::Duration,
    /// Estimated number of triples
    triple_count: Option<usize>,
    /// Available indexes
    indexes: Vec<String>,
    /// Last updated
    last_updated: std::time::Instant,
}

impl JoinOptimizer {
    pub fn new() -> Self {
        Self {
            endpoint_stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Optimize a federated query plan for cross-dataset joins
    pub async fn optimize_plan(&self, plan: &mut FederatedExecutionPlan) -> Result<()> {
        // Analyze join patterns in the query
        let join_patterns = self.extract_join_patterns(&plan)?;
        
        // Reorder steps based on selectivity estimates
        self.reorder_steps_by_selectivity(&mut plan.steps, &join_patterns).await?;
        
        // Apply join reduction strategies
        self.apply_join_reduction(&mut plan.steps, &join_patterns)?;
        
        Ok(())
    }
    
    /// Extract join patterns from execution plan
    fn extract_join_patterns(&self, plan: &FederatedExecutionPlan) -> Result<Vec<JoinPattern>> {
        let mut patterns = Vec::new();
        
        for (i, step) in plan.steps.iter().enumerate() {
            // Simple pattern matching for joins (should use proper SPARQL algebra)
            if step.query.contains("?") {
                let variables = self.extract_variables(&step.query);
                
                // Look for shared variables with other steps
                for (j, other_step) in plan.steps.iter().enumerate() {
                    if i != j {
                        let other_variables = self.extract_variables(&other_step.query);
                        let shared_vars: Vec<_> = variables.iter()
                            .filter(|v| other_variables.contains(v))
                            .cloned()
                            .collect();
                        
                        if !shared_vars.is_empty() {
                            patterns.push(JoinPattern {
                                left_step: i,
                                right_step: j,
                                join_variables: shared_vars,
                                estimated_cost: 1.0, // Will be updated
                            });
                        }
                    }
                }
            }
        }
        
        Ok(patterns)
    }
    
    /// Extract variables from a SPARQL query fragment
    fn extract_variables(&self, query: &str) -> Vec<String> {
        let var_regex = regex::Regex::new(r"\?(\w+)").unwrap();
        var_regex.captures_iter(query)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }
    
    /// Reorder execution steps based on selectivity estimates
    async fn reorder_steps_by_selectivity(
        &self,
        steps: &mut Vec<FederatedStep>,
        join_patterns: &[JoinPattern],
    ) -> Result<()> {
        // Estimate selectivity for each step
        let mut selectivities = Vec::new();
        
        for (i, step) in steps.iter().enumerate() {
            let selectivity = self.estimate_selectivity(step, join_patterns).await?;
            selectivities.push((i, selectivity));
        }
        
        // Sort by selectivity (lower is better - more selective)
        selectivities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Reorder steps (preserve dependencies)
        let mut new_steps = Vec::new();
        let mut processed = HashSet::new();
        
        for (idx, _) in selectivities {
            if !processed.contains(&idx) {
                new_steps.push(steps[idx].clone());
                processed.insert(idx);
            }
        }
        
        // Add any remaining steps
        for (i, step) in steps.iter().enumerate() {
            if !processed.contains(&i) {
                new_steps.push(step.clone());
            }
        }
        
        *steps = new_steps;
        Ok(())
    }
    
    /// Estimate selectivity of a query step
    async fn estimate_selectivity(
        &self,
        step: &FederatedStep,
        join_patterns: &[JoinPattern],
    ) -> Result<f64> {
        // Simple heuristic-based estimation
        let mut selectivity = 1.0;
        
        // Penalize queries with many variables
        let var_count = self.extract_variables(&step.query).len();
        selectivity *= 1.0 + (var_count as f64 * 0.1);
        
        // Favor queries with filters
        if step.query.contains("FILTER") {
            selectivity *= 0.5;
        }
        
        // Consider endpoint statistics if available
        let stats = self.endpoint_stats.read().await;
        if let Some(endpoint_stats) = stats.get(&step.endpoint) {
            // Favor faster endpoints
            selectivity *= endpoint_stats.avg_response_time.as_secs_f64() / 10.0;
        }
        
        Ok(selectivity)
    }
    
    /// Apply join reduction strategies
    fn apply_join_reduction(
        &self,
        steps: &mut Vec<FederatedStep>,
        join_patterns: &[JoinPattern],
    ) -> Result<()> {
        // Implement bind join strategy for selective patterns
        for pattern in join_patterns {
            if pattern.estimated_cost < 0.3 {
                // Apply bind join optimization
                self.convert_to_bind_join(steps, pattern)?;
            }
        }
        
        Ok(())
    }
    
    /// Convert a join pattern to use bind join strategy
    fn convert_to_bind_join(
        &self,
        steps: &mut Vec<FederatedStep>,
        pattern: &JoinPattern,
    ) -> Result<()> {
        // Bind join: execute selective query first, then bind results to second query
        if pattern.left_step < steps.len() && pattern.right_step < steps.len() {
            // Update left step to collect bindings
            steps[pattern.left_step].query = format!(
                "{} # COLLECT BINDINGS: {:?}",
                steps[pattern.left_step].query,
                pattern.join_variables
            );
            
            // Right step will receive bindings from left
            // This is a simplified representation; actual implementation would
            // modify the query to include VALUES clause with bindings
            if pattern.left_step != pattern.right_step {
                let right_query = steps[pattern.right_step].query.clone();
                steps[pattern.right_step].query = format!(
                    "{} # RECEIVE BINDINGS FROM STEP {}",
                    right_query,
                    pattern.left_step
                );
            }
        }
        
        Ok(())
    }
    
    /// Update endpoint statistics
    pub async fn update_endpoint_stats(
        &self,
        endpoint_id: &str,
        response_time: std::time::Duration,
    ) {
        let mut stats = self.endpoint_stats.write().await;
        let entry = stats.entry(endpoint_id.to_string()).or_insert(EndpointStatistics {
            avg_response_time: response_time,
            triple_count: None,
            indexes: Vec::new(),
            last_updated: std::time::Instant::now(),
        });
        
        // Update moving average
        entry.avg_response_time = (entry.avg_response_time + response_time) / 2;
        entry.last_updated = std::time::Instant::now();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct JoinPattern {
    left_step: usize,
    right_step: usize,
    join_variables: Vec<String>,
    estimated_cost: f64,
}

/// Federation manager that coordinates all federation features
pub struct FederationManager {
    config: FederationConfig,
    schema_stitcher: Arc<SchemaStitcher>,
    query_planner: Arc<QueryPlanner>,
    dataset_federation: Arc<RwLock<DatasetFederation>>,
    join_optimizer: Arc<JoinOptimizer>,
}

impl FederationManager {
    pub fn new(
        config: FederationConfig,
        local_schema: Arc<Schema>,
        local_store: Arc<crate::RdfStore>,
    ) -> Self {
        let schema_stitcher = Arc::new(SchemaStitcher::new(local_schema));
        let query_planner = Arc::new(QueryPlanner::new(
            schema_stitcher.clone(),
            config.clone(),
        ));
        let dataset_federation = Arc::new(RwLock::new(DatasetFederation::new(local_store)));
        let join_optimizer = Arc::new(JoinOptimizer::new());

        Self {
            config,
            schema_stitcher,
            query_planner,
            dataset_federation,
            join_optimizer,
        }
    }

    /// Get the merged federated schema
    pub async fn get_federated_schema(&self) -> Result<Schema> {
        self.schema_stitcher
            .merge_schemas(&self.config.endpoints, &self.config)
            .await
    }

    /// Execute a federated GraphQL query
    pub async fn execute_query(&self, query: &Document) -> Result<serde_json::Value> {
        let plan = self.query_planner.plan_query(query).await?;
        
        if plan.steps.is_empty() {
            return Ok(serde_json::json!({
                "data": {},
                "errors": []
            }));
        }
        
        // Execute query steps in dependency order
        let mut results = HashMap::new();
        let mut errors = Vec::new();
        
        // Build execution order respecting dependencies
        let execution_order = self.build_execution_order(&plan)?;
        
        for step_id in execution_order {
            if let Some(step) = plan.steps.iter().find(|s| s.id == step_id) {
                match self.execute_query_step(step, &results).await {
                    Ok(step_result) => {
                        results.insert(step.id.clone(), step_result);
                    }
                    Err(e) => {
                        errors.push(serde_json::json!({
                            "message": e.to_string(),
                            "extensions": {
                                "code": "FEDERATION_ERROR",
                                "service": step.service_id,
                                "step": step.id
                            }
                        }));
                    }
                }
            }
        }
        
        // Merge results from all steps
        let merged_data = self.merge_step_results(&plan, &results)?;
        
        Ok(serde_json::json!({
            "data": merged_data,
            "errors": if errors.is_empty() { serde_json::Value::Null } else { serde_json::Value::Array(errors) }
        }))
    }
    
    /// Build execution order respecting dependencies
    fn build_execution_order(&self, plan: &QueryPlan) -> Result<Vec<String>> {
        let mut order = Vec::new();
        let mut remaining: HashSet<String> = plan.steps.iter().map(|s| s.id.clone()).collect();
        let mut resolved = HashSet::new();
        
        while !remaining.is_empty() {
            let mut progress = false;
            
            // Find steps with all dependencies resolved
            let ready_steps: Vec<String> = remaining.iter()
                .filter(|step_id| {
                    if let Some(deps) = plan.dependencies.get(*step_id) {
                        deps.iter().all(|dep| resolved.contains(dep))
                    } else {
                        true // No dependencies
                    }
                })
                .cloned()
                .collect();
            
            if ready_steps.is_empty() {
                // Check for circular dependencies or missing dependencies
                return Err(anyhow::anyhow!("Circular dependencies or missing dependencies in query plan"));
            }
            
            // Add ready steps to order
            for step_id in ready_steps {
                order.push(step_id.clone());
                remaining.remove(&step_id);
                resolved.insert(step_id);
                progress = true;
            }
            
            if !progress {
                break;
            }
        }
        
        Ok(order)
    }
    
    /// Execute a single query step
    async fn execute_query_step(
        &self,
        step: &QueryStep,
        previous_results: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        if step.service_id == "local" {
            // Execute locally
            self.execute_local_step(step, previous_results).await
        } else {
            // Execute on remote service
            self.execute_remote_step(step, previous_results).await
        }
    }
    
    /// Execute a step on the local service
    async fn execute_local_step(
        &self,
        step: &QueryStep,
        _previous_results: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        // Build complete GraphQL query
        let query = format!("query {{ {} }}", step.query_fragment);
        
        // Execute using local GraphQL executor
        // This would normally use the main GraphQL execution engine
        // For now, return a placeholder result
        Ok(serde_json::json!({
            "local_result": format!("Executed: {}", query)
        }))
    }
    
    /// Execute a step on a remote service
    async fn execute_remote_step(
        &self,
        step: &QueryStep,
        previous_results: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        // Find endpoint configuration
        let endpoint = self.config.endpoints.iter()
            .find(|e| e.id == step.service_id)
            .ok_or_else(|| anyhow::anyhow!("Remote endpoint not found: {}", step.service_id))?;
        
        // Build complete GraphQL query
        let query = format!("query {{ {} }}", step.query_fragment);
        
        // Substitute variables from previous results if needed
        let final_query = self.substitute_variables(&query, &step.variables, previous_results)?;
        
        // Execute remote GraphQL query
        let mut request = self.schema_stitcher.http_client
            .post(&endpoint.url)
            .json(&serde_json::json!({
                "query": final_query,
                "variables": step.variables
            }));
        
        // Add authentication if provided
        if let Some(auth) = &endpoint.auth_header {
            request = request.header("Authorization", auth);
        }
        
        let response = request
            .timeout(std::time::Duration::from_secs(endpoint.timeout_secs))
            .send()
            .await
            .context("Failed to send federated GraphQL request")?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Remote GraphQL query failed with status: {}",
                response.status()
            ));
        }
        
        let result: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse remote GraphQL response")?;
        
        // Extract data from GraphQL response
        if let Some(data) = result.get("data") {
            Ok(data.clone())
        } else {
            // Handle errors from remote service
            let errors = result.get("errors").unwrap_or(&serde_json::Value::Null);
            Err(anyhow::anyhow!("Remote GraphQL query returned errors: {}", errors))
        }
    }
    
    /// Substitute variables in query string with values from previous results
    fn substitute_variables(
        &self,
        query: &str,
        variables: &HashMap<String, serde_json::Value>,
        previous_results: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        let mut substituted = query.to_string();
        
        // Simple variable substitution - in production this would use proper GraphQL variable handling
        for (var_name, _var_value) in variables {
            // Look for variable in previous results
            for (_step_id, result) in previous_results {
                if let Some(var_data) = result.get(var_name) {
                    let var_placeholder = format!("${}", var_name);
                    let var_string = match var_data {
                        serde_json::Value::String(s) => format!("\"{}\"", s),
                        serde_json::Value::Number(n) => n.to_string(),
                        serde_json::Value::Bool(b) => b.to_string(),
                        serde_json::Value::Null => "null".to_string(),
                        _ => var_data.to_string(),
                    };
                    substituted = substituted.replace(&var_placeholder, &var_string);
                }
            }
        }
        
        Ok(substituted)
    }
    
    /// Merge results from multiple query steps
    fn merge_step_results(
        &self,
        plan: &QueryPlan,
        results: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let mut merged = serde_json::Map::new();
        
        // Simple merging strategy - combine all results at top level
        for step in &plan.steps {
            if let Some(result) = results.get(&step.id) {
                if let Some(obj) = result.as_object() {
                    for (key, value) in obj {
                        // Handle namespace conflicts by prefixing with service ID
                        let final_key = if step.service_id == "local" {
                            key.clone()
                        } else {
                            format!("{}_{}", step.service_id, key)
                        };
                        merged.insert(final_key, value.clone());
                    }
                }
            }
        }
        
        Ok(serde_json::Value::Object(merged))
    }

    /// Execute a federated SPARQL query
    pub async fn execute_sparql(&self, query: &str) -> Result<crate::QueryResults> {
        let mut federation = self.dataset_federation.write().await;
        
        // Extract service patterns
        let service_patterns = federation.extract_service_patterns(query)?;
        
        if service_patterns.is_empty() {
            // No federation needed
            return federation.local_store.query(query);
        }
        
        // Plan the federated execution
        let mut plan = federation.plan_federated_execution(query, service_patterns)?;
        
        // Optimize the plan using join optimizer
        self.join_optimizer.optimize_plan(&mut plan).await?;
        
        // Execute the optimized plan
        let start_time = std::time::Instant::now();
        let result = federation.execute_plan(plan).await;
        
        // Update endpoint statistics
        if let Ok(ref _results) = result {
            let elapsed = start_time.elapsed();
            for endpoint in &federation.remote_endpoints {
                self.join_optimizer.update_endpoint_stats(&endpoint.id, elapsed).await;
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ObjectType, FieldType, BuiltinScalars};

    fn create_test_schema() -> Schema {
        let mut schema = Schema::new();
        
        let query_type = ObjectType::new("Query".to_string())
            .with_field(
                "hello".to_string(),
                FieldType::new(
                    "hello".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::string()),
                ),
            );
        
        schema.add_type(GraphQLType::Object(query_type));
        schema.set_query_type("Query".to_string());
        
        schema
    }

    #[tokio::test]
    async fn test_schema_stitcher_creation() {
        let schema = Arc::new(create_test_schema());
        let stitcher = SchemaStitcher::new(schema);
        assert!(stitcher.remote_schemas.read().await.is_empty());
    }

    #[test]
    fn test_namespace_type_name() {
        let schema = Arc::new(create_test_schema());
        let stitcher = SchemaStitcher::new(schema);
        
        let endpoint_with_namespace = RemoteEndpoint {
            id: "test".to_string(),
            url: "https://example.com/graphql".to_string(),
            auth_header: None,
            namespace: Some("remote".to_string()),
            timeout_secs: 30,
        };
        
        let namespaced = stitcher.namespace_type_name("User", &endpoint_with_namespace);
        assert_eq!(namespaced, "remote_User");
        
        let endpoint_without_namespace = RemoteEndpoint {
            id: "test2".to_string(),
            url: "https://example.com/graphql".to_string(),
            auth_header: None,
            namespace: None,
            timeout_secs: 30,
        };
        
        let not_namespaced = stitcher.namespace_type_name("User", &endpoint_without_namespace);
        assert_eq!(not_namespaced, "User");
    }

    #[test]
    fn test_parse_default_value() {
        let schema = Arc::new(create_test_schema());
        let stitcher = SchemaStitcher::new(schema);
        
        // Test null
        let null_value = stitcher.parse_default_value("null").unwrap();
        assert!(matches!(null_value, crate::ast::Value::NullValue));
        
        // Test boolean
        let true_value = stitcher.parse_default_value("true").unwrap();
        assert!(matches!(true_value, crate::ast::Value::BooleanValue(true)));
        
        // Test integer
        let int_value = stitcher.parse_default_value("42").unwrap();
        assert!(matches!(int_value, crate::ast::Value::IntValue(42)));
        
        // Test string
        let string_value = stitcher.parse_default_value("\"hello\"").unwrap();
        if let crate::ast::Value::StringValue(s) = string_value {
            assert_eq!(s, "hello");
        } else {
            panic!("Expected string value");
        }
    }

    #[tokio::test]
    async fn test_federation_config_default() {
        let config = FederationConfig::default();
        assert!(config.enable_schema_cache);
        assert_eq!(config.schema_cache_ttl, 3600);
        assert_eq!(config.max_federation_depth, 3);
    }

    #[tokio::test]
    async fn test_query_plan_creation() {
        let plan = QueryPlan::new();
        assert!(plan.steps.is_empty());
        assert!(plan.dependencies.is_empty());
    }

    #[test]
    fn test_service_pattern_extraction() {
        let store = Arc::new(crate::RdfStore::new().unwrap());
        let federation = DatasetFederation::new(store);
        
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name WHERE {
                SERVICE <https://example.com/sparql> {
                    ?person foaf:name ?name
                }
            }
        "#;
        
        let patterns = federation.extract_service_patterns(query).unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].service_uri, "https://example.com/sparql");
        assert!(patterns[0].pattern.contains("foaf:name"));
    }

    #[tokio::test]
    async fn test_join_optimizer_creation() {
        let optimizer = JoinOptimizer::new();
        let stats = optimizer.endpoint_stats.read().await;
        assert!(stats.is_empty());
    }

    #[tokio::test]
    async fn test_join_optimizer_update_stats() {
        let optimizer = JoinOptimizer::new();
        
        let response_time = std::time::Duration::from_millis(100);
        optimizer.update_endpoint_stats("test-endpoint", response_time).await;
        
        let stats = optimizer.endpoint_stats.read().await;
        assert!(stats.contains_key("test-endpoint"));
        
        let endpoint_stats = stats.get("test-endpoint").unwrap();
        assert_eq!(endpoint_stats.avg_response_time, response_time);
    }

    #[test]
    fn test_extract_variables() {
        let optimizer = JoinOptimizer::new();
        
        let query = "SELECT ?name ?age WHERE { ?person foaf:name ?name ; foaf:age ?age }";
        let vars = optimizer.extract_variables(query);
        
        assert!(vars.contains(&"name".to_string()));
        assert!(vars.contains(&"age".to_string()));
        assert!(vars.contains(&"person".to_string()));
    }

    #[tokio::test]
    async fn test_federation_manager_creation() {
        let config = FederationConfig::default();
        let schema = Arc::new(create_test_schema());
        let store = Arc::new(crate::RdfStore::new().unwrap());
        
        let manager = FederationManager::new(config, schema, store);
        
        // Just verify it creates successfully
        assert_eq!(manager.config.max_federation_depth, 3);
    }

    #[test]
    fn test_federated_step_clone() {
        let step = FederatedStep {
            step_id: "test".to_string(),
            endpoint: "endpoint1".to_string(),
            endpoint_url: "https://example.com".to_string(),
            auth: Some("Bearer token".to_string()),
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            result_variable: "?result".to_string(),
            timeout: std::time::Duration::from_secs(30),
        };
        
        let cloned = step.clone();
        assert_eq!(cloned.step_id, step.step_id);
        assert_eq!(cloned.endpoint, step.endpoint);
        assert_eq!(cloned.query, step.query);
    }

    #[test]
    fn test_join_pattern_creation() {
        let pattern = JoinPattern {
            left_step: 0,
            right_step: 1,
            join_variables: vec!["?person".to_string()],
            estimated_cost: 0.5,
        };
        
        assert_eq!(pattern.left_step, 0);
        assert_eq!(pattern.right_step, 1);
        assert_eq!(pattern.join_variables.len(), 1);
    }

    #[test]
    fn test_endpoint_statistics() {
        let stats = EndpointStatistics {
            avg_response_time: std::time::Duration::from_millis(150),
            triple_count: Some(1000000),
            indexes: vec!["spo".to_string()],
            last_updated: std::time::Instant::now(),
        };
        
        assert_eq!(stats.avg_response_time.as_millis(), 150);
        assert_eq!(stats.triple_count.unwrap(), 1000000);
        assert_eq!(stats.indexes.len(), 1);
    }
}