//! Core GraphQL Federation implementation
//!
//! This module contains the main GraphQLFederation implementation with core operations
//! including execution management, result stitching, and basic federation operations.

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    executor::types::{QueryResultData, StepResult},
    executor::GraphQLResponse,
    planner::ExecutionPlan,
};

use super::types::*;

impl GraphQLFederation {
    /// Create a new GraphQL federation manager
    pub fn new() -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            config: GraphQLFederationConfig::default(),
            cache: None,
        }
    }

    /// Create a new GraphQL federation manager with custom configuration
    pub fn with_config(config: GraphQLFederationConfig) -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            config,
            cache: None,
        }
    }

    /// Create a new GraphQL federation manager with cache
    pub fn with_cache(
        config: GraphQLFederationConfig,
        cache: Arc<crate::cache::FederationCache>,
    ) -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            config,
            cache: Some(cache),
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
                Self::visit_step(&step.step_id, plan, &mut visited, &mut visiting, &mut order)?;
            }
        }

        Ok(order)
    }

    /// Recursive helper for topological sort
    fn visit_step(
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
                Self::visit_step(dep_id, plan, visited, visiting, order)?;
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

        // Check cache for this query step
        let cache_key = self.generate_cache_key(step, completed_steps);
        let mut cache_hit = false;

        // Try to get cached result first
        let cached_result = if let Some(cache) = &self.cache {
            let cached = cache.get_query_result(&cache_key).await;
            if cached.is_some() {
                cache_hit = true;
                debug!("Cache hit for step: {}", step.step_id);
            }
            cached
        } else {
            None
        };

        let (result, service_response_time) = if let Some(cached) = cached_result {
            // Use cached result - extract the data from the enum variant
            let cached_data = match cached {
                crate::cache::QueryResultCache::GraphQL(response) => {
                    QueryResultData::GraphQL(response)
                }
                crate::cache::QueryResultCache::Sparql(results) => QueryResultData::Sparql(results),
            };
            (Ok(cached_data), Duration::from_millis(0))
        } else {
            // Execute the step
            match step.step_type {
                crate::StepType::GraphQLQuery => {
                    match self.execute_graphql_query_step(step, completed_steps).await {
                        Ok((data, service_time)) => (Ok(data), service_time),
                        Err(e) => (Err(e), Duration::from_millis(0)),
                    }
                }
                crate::StepType::SchemaStitch => {
                    // Schema stitch doesn't make service calls, so no specific service response time
                    match self.execute_schema_stitch_step(step, completed_steps).await {
                        Ok(data) => (Ok(data), Duration::from_millis(0)),
                        Err(e) => (Err(e), Duration::from_millis(0)),
                    }
                }
                _ => {
                    // For non-GraphQL steps, return a success result
                    (
                        Ok(QueryResultData::GraphQL(GraphQLResponse {
                            data: serde_json::Value::Null,
                            errors: Vec::new(),
                            extensions: None,
                        })),
                        Duration::from_millis(0),
                    )
                }
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

        // Cache successful results for future use
        if !cache_hit && matches!(status, crate::executor::ExecutionStatus::Success) {
            if let (Some(cache), Some(data)) = (&self.cache, &data) {
                match data {
                    QueryResultData::GraphQL(response) => {
                        let cache_entry = crate::cache::QueryResultCache::GraphQL(response.clone());
                        let ttl = Some(std::time::Duration::from_secs(300)); // 5 minutes default TTL
                        cache.put_query_result(&cache_key, cache_entry, ttl).await;
                    }
                    QueryResultData::Sparql(results) => {
                        let cache_entry = crate::cache::QueryResultCache::Sparql(results.clone());
                        let ttl = Some(std::time::Duration::from_secs(300)); // 5 minutes default TTL
                        cache.put_query_result(&cache_key, cache_entry, ttl).await;
                    }
                    QueryResultData::ServiceResult(_) => {
                        // Skip caching service results as they don't fit the QueryResultCache enum
                        debug!("Skipping cache for ServiceResult data type");
                    }
                }
            }
        }

        let result_size = data
            .as_ref()
            .map(|d| self.estimate_result_size(d))
            .unwrap_or(0);

        // Calculate memory usage before moving data
        let memory_used = data
            .as_ref()
            .map(|d| self.estimate_memory_usage(d, result_size as usize))
            .unwrap_or(0);

        Ok(StepResult {
            step_id: step.step_id.clone(),
            step_type: step.step_type,
            status: status.clone(),
            data,
            error: error.clone(),
            execution_time,
            service_id: step.service_id.clone(),
            memory_used,
            result_size: result_size as usize,
            success: matches!(status, crate::executor::ExecutionStatus::Success),
            error_message: error,
            service_response_time,
            cache_hit,
        })
    }

    /// Generate a cache key for a query step
    fn generate_cache_key(
        &self,
        step: &crate::ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the query fragment
        step.query_fragment.hash(&mut hasher);

        // Hash the service ID
        step.service_id.hash(&mut hasher);

        // Hash relevant completed step data that might affect this query
        for dep in &step.dependencies {
            if let Some(completed_step) = completed_steps.get(dep) {
                if let Some(data) = &completed_step.data {
                    // Create a simplified hash of the data
                    let data_str = format!("{data:?}");
                    data_str.hash(&mut hasher);
                }
            }
        }

        format!("gql_step_{}_{}", step.step_id, hasher.finish())
    }

    /// Execute a GraphQL query step
    async fn execute_graphql_query_step(
        &self,
        step: &crate::ExecutionStep,
        _completed_steps: &HashMap<String, StepResult>,
    ) -> Result<(QueryResultData, Duration)> {
        debug!("Executing GraphQL query: {}", step.query_fragment);

        // Track service-specific response time
        let service_start = std::time::Instant::now();

        // In a real implementation, this would:
        // 1. Send the query to the appropriate GraphQL service
        // 2. Handle authentication and headers
        // 3. Parse and validate the response
        // 4. Apply any necessary transformations

        // Simulate actual service call timing
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

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

        let service_response_time = service_start.elapsed();

        Ok((
            QueryResultData::GraphQL(mock_response),
            service_response_time,
        ))
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

    /// Estimate memory usage for result data
    pub fn estimate_memory_usage(&self, data: &QueryResultData, result_size: usize) -> usize {
        match data {
            QueryResultData::GraphQL(response) => {
                // Estimate based on JSON serialization size
                if let Ok(serialized) = serde_json::to_string(response) {
                    serialized.len() + result_size
                } else {
                    result_size
                }
            }
            _ => result_size,
        }
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
                        kind
                        description
                        fields {
                            name
                            type {
                                name
                                kind
                            }
                        }
                    }
                    directives {
                        name
                        description
                        locations
                        args {
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

        // Execute the introspection query
        let response = self
            .execute_introspection_query(service_endpoint, federation_query)
            .await?;

        // Parse the response to extract federation capabilities
        let capabilities = self.parse_federation_capabilities(&response)?;

        // Extract SDL from response or generate from schema
        let sdl = self.extract_or_generate_sdl(&response).await?;

        // Extract entity types from the schema
        let entity_types = self.extract_entity_types(&response)?;

        Ok(FederationServiceInfo {
            sdl,
            capabilities,
            entity_types,
        })
    }

    /// Parse variables from a GraphQL query (helper method)
    pub fn parse_variables(&self, query: &str) -> Result<HashMap<String, VariableDefinition>> {
        let mut variables = HashMap::new();

        // Simple variable parsing - look for variable definitions
        // In production, use a proper GraphQL parser
        if let Some(start) = query.find('(') {
            if let Some(end) = query.find(')') {
                let vars_section = &query[start + 1..end];

                for var_def in vars_section.split(',') {
                    let var_def = var_def.trim();
                    if var_def.starts_with('$') {
                        let parts: Vec<&str> = var_def.split(':').collect();
                        if parts.len() >= 2 {
                            let var_name = parts[0].trim().trim_start_matches('$').to_string();
                            let var_type = parts[1].trim().to_string();

                            variables.insert(
                                var_name.clone(),
                                VariableDefinition {
                                    name: var_name,
                                    variable_type: var_type,
                                    default_value: None,
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(variables)
    }

    /// Merge directives from two sets (helper method)
    pub fn merge_directives(&self, existing: &[Directive], new: &[Directive]) -> Vec<Directive> {
        let mut merged = existing.to_vec();

        for new_directive in new {
            // Check if directive already exists
            if !merged.iter().any(|d| d.name == new_directive.name) {
                merged.push(new_directive.clone());
            }
        }

        merged
    }

    /// Check if a type is a built-in GraphQL scalar type
    pub fn is_builtin_type(&self, type_name: &str) -> bool {
        matches!(type_name, "String" | "Int" | "Float" | "Boolean" | "ID")
    }

    /// Federated subscription support with real-time event propagation
    pub async fn create_federated_subscription(
        &self,
        subscription_query: &str,
        variables: Option<serde_json::Value>,
    ) -> Result<FederatedSubscriptionStream> {
        debug!("Creating federated subscription");

        // Parse the subscription query
        let parsed_query = self.parse_graphql_query(subscription_query)?;

        if !matches!(
            parsed_query.operation_type,
            GraphQLOperationType::Subscription
        ) {
            return Err(anyhow!("Query is not a subscription"));
        }

        // Decompose subscription across services
        let service_subscriptions = self.decompose_subscription(&parsed_query).await?;

        // Create federated subscription stream
        let stream = FederatedSubscriptionStream::new(
            subscription_query.to_string(),
            variables,
            service_subscriptions,
        );

        Ok(stream)
    }

    /// Decompose subscription query across federated services
    async fn decompose_subscription(
        &self,
        query: &ParsedQuery,
    ) -> Result<Vec<ServiceSubscription>> {
        let mut service_subscriptions = Vec::new();

        // Get unified schema
        let unified_schema = self.create_unified_schema().await?;

        // Analyze field ownership for subscription fields
        let field_ownership = self.analyze_field_ownership(query, &unified_schema)?;

        // Create service-specific subscriptions
        for (service_id, fields) in &field_ownership.service_to_fields {
            if !fields.is_empty() {
                let subscription_query = self.build_service_subscription_query(fields, query)?;

                // Convert VariableDefinition to actual values (using default values where available)
                let variable_values: HashMap<String, serde_json::Value> = query
                    .variables
                    .iter()
                    .filter_map(|(name, def)| {
                        def.default_value
                            .as_ref()
                            .map(|val| (name.clone(), val.clone()))
                    })
                    .collect();

                service_subscriptions.push(ServiceSubscription {
                    service_id: service_id.clone(),
                    query: subscription_query,
                    variables: variable_values,
                    stream_active: false,
                });
            }
        }

        Ok(service_subscriptions)
    }

    /// Build subscription query for a specific service
    fn build_service_subscription_query(
        &self,
        fields: &[String],
        original_query: &ParsedQuery,
    ) -> Result<String> {
        let operation_name = original_query
            .operation_name
            .as_ref()
            .map(|name| format!(" {name}"))
            .unwrap_or_default();

        // Build subscription with service-specific fields
        let subscription_query =
            format!("subscription{} {{ {} }}", operation_name, fields.join(" "));

        Ok(subscription_query)
    }

    /// Merge subscription events from multiple services
    pub async fn merge_subscription_events(
        &self,
        events: Vec<SubscriptionEvent>,
    ) -> Result<GraphQLResponse> {
        debug!("Merging {} subscription events", events.len());

        let mut merged_data = serde_json::Map::new();
        let mut all_errors = Vec::new();

        // Merge events based on their source service and field ownership
        for event in events {
            if let Some(data) = event.data.as_object() {
                for (key, value) in data {
                    // Handle conflicts by service priority or timestamp
                    if merged_data.contains_key(key) {
                        // Use most recent event for conflicts
                        if event.timestamp > self.get_field_timestamp(&merged_data, key) {
                            merged_data.insert(key.clone(), value.clone());
                        }
                    } else {
                        merged_data.insert(key.clone(), value.clone());
                    }
                }
            }

            // Convert graphql::types::GraphQLError to executor::types::GraphQLError
            let converted_errors: Vec<crate::executor::types::GraphQLError> = event
                .errors
                .into_iter()
                .map(|err| crate::executor::types::GraphQLError {
                    message: err.message,
                    locations: err.locations.map(|locs| {
                        locs.into_iter()
                            .map(|loc| crate::executor::types::GraphQLLocation {
                                line: loc.line,
                                column: loc.column,
                            })
                            .collect()
                    }),
                    path: err.path,
                })
                .collect();
            all_errors.extend(converted_errors);
        }

        Ok(GraphQLResponse {
            data: serde_json::Value::Object(merged_data),
            errors: all_errors,
            extensions: None,
        })
    }

    /// Get timestamp of a field in merged data (placeholder implementation)
    fn get_field_timestamp(
        &self,
        _data: &serde_json::Map<String, serde_json::Value>,
        _field: &str,
    ) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc::now() // Simplified - in real implementation would track field timestamps
    }

    /// Handle subscription connection management
    pub async fn manage_subscription_connections(
        &self,
        connection_id: &str,
        operation: SubscriptionOperation,
    ) -> Result<()> {
        match operation {
            SubscriptionOperation::Start { query, variables } => {
                debug!("Starting subscription connection: {}", connection_id);
                let _subscription = self
                    .create_federated_subscription(&query, variables)
                    .await?;
                // Store subscription in connection manager
                // Implementation would involve WebSocket connection management
            }
            SubscriptionOperation::Stop => {
                debug!("Stopping subscription connection: {}", connection_id);
                // Clean up subscription resources
            }
            SubscriptionOperation::ConnectionInit => {
                debug!("Initializing subscription connection: {}", connection_id);
                // Send connection ack
            }
        }

        Ok(())
    }

    /// Handle real-time event propagation across federation
    pub async fn propagate_federation_event(&self, event: FederationEvent) -> Result<()> {
        debug!("Propagating federation event: {:?}", event.event_type);

        match event.event_type {
            FederationEventType::EntityUpdate => {
                // Invalidate caches and notify dependent services
                self.handle_entity_update_propagation(&event).await?
            }
            FederationEventType::SchemaChange => {
                // Update schema and revalidate subscriptions
                self.handle_schema_change_propagation(&event).await?
            }
            FederationEventType::ServiceAvailability => {
                // Update service status and reroute queries
                self.handle_service_availability_change(&event).await?
            }
        }

        Ok(())
    }

    /// Handle entity update propagation
    async fn handle_entity_update_propagation(&self, event: &FederationEvent) -> Result<()> {
        debug!("Handling entity update propagation");

        // Extract entity information from event
        if let Some(entity_type) = event.data.get("entityType").and_then(|v| v.as_str()) {
            // Find all services that might be affected by this entity update
            let schemas = self.schemas.read().await;
            let mut affected_services = Vec::new();

            for (service_id, schema) in schemas.iter() {
                if schema.types.contains_key(entity_type) {
                    affected_services.push(service_id.clone());
                }
            }

            // Notify affected services about the entity update
            for service_id in affected_services {
                debug!("Notifying service {} about entity update", service_id);
                // Implementation would send notification to service
            }
        }

        Ok(())
    }

    /// Handle schema change propagation
    async fn handle_schema_change_propagation(&self, event: &FederationEvent) -> Result<()> {
        debug!("Handling schema change propagation");

        if let Some(service_id) = event.data.get("serviceId").and_then(|v| v.as_str()) {
            // Re-validate unified schema after change
            match self.create_unified_schema().await {
                Err(e) => {
                    warn!(
                        "Schema validation failed after change in service {}: {}",
                        service_id, e
                    );
                }
                _ => {
                    info!("Schema successfully updated for service {}", service_id);
                }
            }
        }

        Ok(())
    }

    /// Handle service availability changes
    async fn handle_service_availability_change(&self, event: &FederationEvent) -> Result<()> {
        debug!("Handling service availability change");

        if let Some(service_id) = event.data.get("serviceId").and_then(|v| v.as_str()) {
            if let Some(available) = event.data.get("available").and_then(|v| v.as_bool()) {
                if !available {
                    warn!(
                        "Service {} is no longer available - implementing fallback strategies",
                        service_id
                    );
                    // Implement circuit breaker and fallback logic
                } else {
                    info!("Service {} is now available", service_id);
                    // Resume normal operation
                }
            }
        }

        Ok(())
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

    /// Parse federation capabilities from introspection response
    fn parse_federation_capabilities(
        &self,
        response: &serde_json::Value,
    ) -> Result<FederationCapabilities> {
        let schema = response["data"]["__schema"]
            .as_object()
            .ok_or_else(|| anyhow!("Invalid introspection response: missing schema"))?;

        let mut supports_entities = false;
        let mut supports_entity_interfaces = false;
        let mut supports_progressive_override = false;
        let mut federation_version = "1.0".to_string();

        // Check for federation directives
        if let Some(directives) = schema["directives"].as_array() {
            for directive in directives {
                if let Some(name) = directive["name"].as_str() {
                    match name {
                        "key" => {
                            supports_entities = true;
                            federation_version = "2.0".to_string();
                        }
                        "external" | "requires" | "provides" => {
                            supports_entities = true;
                        }
                        "extends" => {
                            supports_entity_interfaces = true;
                        }
                        "override" => {
                            supports_progressive_override = true;
                            federation_version = "2.0".to_string();
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(FederationCapabilities {
            federation_version,
            supports_entities,
            supports_entity_interfaces,
            supports_progressive_override,
        })
    }

    /// Extract or generate SDL from introspection response
    async fn extract_or_generate_sdl(&self, response: &serde_json::Value) -> Result<String> {
        // First try to get SDL from _service field (if supported)
        if let Some(service) = response["data"]["_service"].as_object() {
            if let Some(sdl) = service["sdl"].as_str() {
                return Ok(sdl.to_string());
            }
        }

        // Fall back to generating SDL from schema introspection
        self.generate_sdl_from_introspection(response)
    }

    /// Generate SDL from introspection response
    fn generate_sdl_from_introspection(&self, response: &serde_json::Value) -> Result<String> {
        let schema = response["data"]["__schema"]
            .as_object()
            .ok_or_else(|| anyhow!("Invalid introspection response: missing schema"))?;

        let mut sdl = String::new();

        // Generate type definitions
        if let Some(types) = schema["types"].as_array() {
            for type_def in types {
                if let Some(type_name) = type_def["name"].as_str() {
                    // Skip built-in types
                    if type_name.starts_with("__") || self.is_builtin_type(type_name) {
                        continue;
                    }

                    if let Some(kind) = type_def["kind"].as_str() {
                        match kind {
                            "OBJECT" => {
                                sdl.push_str(&self.generate_object_type_sdl(type_def)?);
                            }
                            "INTERFACE" => {
                                sdl.push_str(&self.generate_interface_type_sdl(type_def)?);
                            }
                            "ENUM" => {
                                sdl.push_str(&self.generate_enum_type_sdl(type_def)?);
                            }
                            "SCALAR" => {
                                sdl.push_str(&format!("scalar {type_name}\n\n"));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(sdl)
    }

    /// Generate SDL for object type
    fn generate_object_type_sdl(&self, type_def: &serde_json::Value) -> Result<String> {
        let type_name = type_def["name"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing type name"))?;

        let mut sdl = format!("type {type_name} ");

        // Add directives (simplified - would need more sophisticated parsing)
        if let Some(description) = type_def["description"].as_str() {
            if description.contains("@key") {
                sdl.push_str("@key(fields: \"id\") ");
            }
        }

        sdl.push_str("{\n");

        // Add fields
        if let Some(fields) = type_def["fields"].as_array() {
            for field in fields {
                if let Some(field_name) = field["name"].as_str() {
                    let field_type = self.extract_type_name_from_field(field)?;
                    sdl.push_str(&format!("  {field_name}: {field_type}\n"));
                }
            }
        }

        sdl.push_str("}\n\n");
        Ok(sdl)
    }

    /// Generate SDL for interface type
    fn generate_interface_type_sdl(&self, type_def: &serde_json::Value) -> Result<String> {
        let type_name = type_def["name"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing type name"))?;

        let mut sdl = format!("interface {type_name} {{\n");

        // Add fields
        if let Some(fields) = type_def["fields"].as_array() {
            for field in fields {
                if let Some(field_name) = field["name"].as_str() {
                    let field_type = self.extract_type_name_from_field(field)?;
                    sdl.push_str(&format!("  {field_name}: {field_type}\n"));
                }
            }
        }

        sdl.push_str("}\n\n");
        Ok(sdl)
    }

    /// Generate SDL for enum type
    fn generate_enum_type_sdl(&self, type_def: &serde_json::Value) -> Result<String> {
        let type_name = type_def["name"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing type name"))?;

        let mut sdl = format!("enum {type_name} {{\n");

        // Add enum values
        if let Some(enum_values) = type_def["enumValues"].as_array() {
            for enum_value in enum_values {
                if let Some(value_name) = enum_value["name"].as_str() {
                    sdl.push_str(&format!("  {value_name}\n"));
                }
            }
        }

        sdl.push_str("}\n\n");
        Ok(sdl)
    }

    /// Extract type name from field definition
    fn extract_type_name_from_field(&self, field: &serde_json::Value) -> Result<String> {
        self.extract_type_name_recursive(&field["type"])
    }

    /// Recursively extract type name handling NON_NULL and LIST wrappers
    #[allow(clippy::only_used_in_recursion)]
    fn extract_type_name_recursive(&self, type_ref: &serde_json::Value) -> Result<String> {
        if let Some(kind) = type_ref["kind"].as_str() {
            match kind {
                "NON_NULL" => {
                    let inner_type = self.extract_type_name_recursive(&type_ref["ofType"])?;
                    Ok(format!("{inner_type}!"))
                }
                "LIST" => {
                    let inner_type = self.extract_type_name_recursive(&type_ref["ofType"])?;
                    Ok(format!("[{inner_type}]"))
                }
                _ => {
                    if let Some(name) = type_ref["name"].as_str() {
                        Ok(name.to_string())
                    } else {
                        Err(anyhow!("Missing type name"))
                    }
                }
            }
        } else {
            Err(anyhow!("Missing type kind"))
        }
    }

    /// Extract entity types from introspection response
    fn extract_entity_types(&self, response: &serde_json::Value) -> Result<Vec<String>> {
        let schema = response["data"]["__schema"]
            .as_object()
            .ok_or_else(|| anyhow!("Invalid introspection response: missing schema"))?;

        let mut entity_types = Vec::new();

        if let Some(types) = schema["types"].as_array() {
            for type_def in types {
                if let Some(type_name) = type_def["name"].as_str() {
                    // Skip built-in types
                    if type_name.starts_with("__") {
                        continue;
                    }

                    // Check if this is an entity type (simplified detection)
                    if self.is_entity_type_from_def(type_def) {
                        entity_types.push(type_name.to_string());
                    }
                }
            }
        }

        Ok(entity_types)
    }

    /// Check if a type is an entity type (has key fields) from type definition
    fn is_entity_type_from_def(&self, type_def: &serde_json::Value) -> bool {
        // Simplified check - look for @key directive in description or check for id field
        if let Some(description) = type_def["description"].as_str() {
            if description.contains("@key") {
                return true;
            }
        }

        // Check if type has an 'id' field (common pattern for entities)
        if let Some(fields) = type_def["fields"].as_array() {
            for field in fields {
                if field["name"].as_str() == Some("id") {
                    return true;
                }
            }
        }

        false
    }

    /// Parse a value from string format to JSON value (helper method)
    pub fn parse_value_from_string(&self, value: &str) -> Result<serde_json::Value> {
        let value = value.trim();

        if value == "true" {
            Ok(serde_json::Value::Bool(true))
        } else if value == "false" {
            Ok(serde_json::Value::Bool(false))
        } else if value == "null" {
            Ok(serde_json::Value::Null)
        } else if value.starts_with('"') && value.ends_with('"') {
            Ok(serde_json::Value::String(
                value[1..value.len() - 1].to_string(),
            ))
        } else if let Ok(num) = value.parse::<i64>() {
            Ok(serde_json::Value::Number(serde_json::Number::from(num)))
        } else if let Ok(num) = value.parse::<f64>() {
            Ok(serde_json::Value::Number(
                serde_json::Number::from_f64(num).unwrap_or_else(|| serde_json::Number::from(0)),
            ))
        } else {
            // Assume it's a string without quotes
            Ok(serde_json::Value::String(value.to_string()))
        }
    }

    /// Estimate the result size of query result data
    fn estimate_result_size(&self, data: &QueryResultData) -> u64 {
        match data {
            QueryResultData::Sparql(sparql_results) => {
                // Estimate based on number of results and variables
                sparql_results.results.len() as u64 * 10 // Simple heuristic
            }
            QueryResultData::GraphQL(graphql_response) => {
                // Estimate based on JSON serialization size
                serde_json::to_string(&graphql_response.data)
                    .map(|s| s.len() as u64)
                    .unwrap_or(100)
            }
            QueryResultData::ServiceResult(service_result) => {
                // Estimate based on JSON value size
                serde_json::to_string(service_result)
                    .map(|s| s.len() as u64)
                    .unwrap_or(50)
            }
        }
    }
}

impl Default for GraphQLFederation {
    fn default() -> Self {
        Self::new()
    }
}
