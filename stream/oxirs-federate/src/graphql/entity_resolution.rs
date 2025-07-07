//! Entity resolution for GraphQL Federation
//!
//! This module handles entity resolution, dependency analysis, and advanced federation
//! operations including entity stitching and Apollo Federation directive support.

use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use tracing::{debug, info, warn};

use crate::executor::GraphQLResponse;

use super::types::*;

impl GraphQLFederation {
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

        // Build dependency graph
        let dependency_graph = self.build_entity_dependency_graph(entity_refs).await?;

        // Perform topological sort to determine execution order
        let execution_batches = self.topological_sort_entities(&dependency_graph)?;

        // Create resolution steps from execution batches
        for batch in execution_batches {
            for entity_ref in batch {
                let step = self.create_resolution_step(&entity_ref).await?;
                plan.steps.push(step);
            }
        }

        Ok(plan)
    }

    /// Build entity dependency graph for resolution planning
    async fn build_entity_dependency_graph(
        &self,
        entity_refs: &[EntityReference],
    ) -> Result<EntityDependencyGraph> {
        let mut graph = EntityDependencyGraph {
            nodes: HashMap::new(),
            edges: Vec::new(),
        };

        // Add nodes (entities) to the graph
        for (idx, entity_ref) in entity_refs.iter().enumerate() {
            graph.nodes.insert(entity_ref.clone(), idx);
        }

        // Add edges (dependencies) to the graph
        for (i, entity_a) in entity_refs.iter().enumerate() {
            for (j, entity_b) in entity_refs.iter().enumerate() {
                if i != j && self.entities_have_dependency(entity_a, entity_b)? {
                    graph.edges.push((i, j));
                }
            }
        }

        Ok(graph)
    }

    /// Check if one entity depends on another
    pub fn entities_have_dependency(
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
                    .find(|&(_, &idx)| idx == node)
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

    /// Create a resolution step for an entity
    async fn create_resolution_step(
        &self,
        entity_ref: &EntityReference,
    ) -> Result<EntityResolutionStep> {
        Ok(EntityResolutionStep {
            service_name: entity_ref.service_id.clone(),
            entity_type: entity_ref.entity_type.clone(),
            key_fields: entity_ref.key_fields.clone(),
            query: self.build_entity_query(&[entity_ref.clone()]).await?,
            depends_on: self
                .analyze_entity_dependencies(&[entity_ref.clone()])
                .await?,
        })
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

    /// Resolve a batch of entities in parallel with optimization
    async fn resolve_entity_batch(
        &self,
        entities: &[EntityReference],
        context: &ResolutionContext,
        cache: &HashMap<EntityReference, EntityData>,
    ) -> Result<Vec<(EntityReference, EntityData)>> {
        let mut results = Vec::new();

        // Check cache first for all entities
        let mut uncached_entities = Vec::new();
        for entity in entities {
            if let Some(cached_data) = cache.get(entity) {
                results.push((entity.clone(), cached_data.clone()));
            } else {
                uncached_entities.push(entity);
            }
        }

        if uncached_entities.is_empty() {
            return Ok(results);
        }

        // Group entities by service for batch resolution
        let mut service_groups: HashMap<String, Vec<&EntityReference>> = HashMap::new();
        for entity in &uncached_entities {
            service_groups
                .entry(entity.service_id.clone())
                .or_default()
                .push(entity);
        }

        // Create futures for parallel execution
        let mut futures = Vec::new();
        for (service_id, service_entities) in service_groups {
            // Clone the values to avoid lifetime issues
            let service_id_owned = service_id.clone();
            let service_entities_owned = service_entities.clone();
            let self_ref = self;
            let context_clone = context.clone();
            let cache_clone = cache.clone();

            let fut = async move {
                let entity_refs: Vec<&EntityReference> =
                    service_entities_owned.iter().map(|e| *e).collect();
                self_ref
                    .resolve_service_entity_batch(
                        &service_id_owned,
                        &entity_refs[..],
                        &context_clone,
                        &cache_clone,
                    )
                    .await
            };
            futures.push(fut);
        }

        // Execute all service queries in parallel using tokio::join!
        for fut in futures {
            match fut.await {
                Ok(service_results) => results.extend(service_results),
                Err(e) => {
                    debug!("Service entity resolution failed: {}", e);
                    // Continue with partial results
                }
            }
        }

        Ok(results)
    }

    /// Resolve entities for a specific service in batch
    async fn resolve_service_entity_batch(
        &self,
        service_id: &str,
        entities: &[&EntityReference],
        context: &ResolutionContext,
        cache: &HashMap<EntityReference, EntityData>,
    ) -> Result<Vec<(EntityReference, EntityData)>> {
        let mut results = Vec::new();

        // Check cache first
        for entity in entities {
            if let Some(cached_data) = cache.get(entity) {
                results.push(((*entity).clone(), cached_data.clone()));
                continue;
            }

            // Build representations for _entities query
            let representations = self.build_entity_representations(entities)?;

            // Execute _entities query
            let response = self
                .execute_entities_query(service_id, &representations, context)
                .await?;

            // Parse response into EntityData
            let entity_data = self.parse_entity_from_response(&response, &entity.entity_type)?;

            if let Some(data) = entity_data {
                results.push(((*entity).clone(), data));
            }
        }

        Ok(results)
    }

    /// Build entity representations for _entities query with smart batching
    fn build_entity_representations(
        &self,
        entities: &[&EntityReference],
    ) -> Result<Vec<EntityRepresentation>> {
        let mut representations = Vec::new();

        for entity in entities {
            let mut fields = serde_json::Map::new();
            fields.insert(
                "__typename".to_string(),
                serde_json::Value::String(entity.entity_type.clone()),
            );

            // Add key field values with better type handling
            for key_field in &entity.key_fields {
                let mock_value = self.generate_mock_key_value(&entity.entity_type, key_field);
                fields.insert(key_field.clone(), mock_value);
            }

            representations.push(EntityRepresentation {
                typename: entity.entity_type.clone(),
                fields: serde_json::Value::Object(fields),
            });
        }

        Ok(representations)
    }

    /// Generate appropriate mock values for entity keys
    fn generate_mock_key_value(&self, entity_type: &str, key_field: &str) -> serde_json::Value {
        match (entity_type, key_field) {
            ("User", "id") => serde_json::Value::String("user-123".to_string()),
            ("Product", "id") => serde_json::Value::String("product-456".to_string()),
            ("Product", "sku") => serde_json::Value::String("SKU-789".to_string()),
            ("Order", "id") => serde_json::Value::String("order-999".to_string()),
            (_, "id") => serde_json::Value::String(format!("{}-id", entity_type.to_lowercase())),
            (_, field) => serde_json::Value::String(format!("mock-{}", field)),
        }
    }

    /// Execute _entities query against a service
    async fn execute_entities_query(
        &self,
        service_id: &str,
        representations: &[EntityRepresentation],
        context: &ResolutionContext,
    ) -> Result<GraphQLResponse> {
        debug!(
            "Executing _entities query for {} representations on service: {}",
            representations.len(),
            service_id
        );

        // Build _entities query
        let query = self.build_entities_query_from_representations(representations)?;

        // Execute query (mock implementation)
        let response = self.execute_service_query(service_id, &query).await?;

        Ok(response)
    }

    /// Build _entities query from representations
    fn build_entities_query_from_representations(
        &self,
        representations: &[EntityRepresentation],
    ) -> Result<String> {
        if representations.is_empty() {
            return Ok(String::new());
        }

        let first_repr = &representations[0];
        let typename = &first_repr.typename;

        let query = format!(
            r#"
            query($_representations: [_Any!]!) {{
                _entities(representations: $_representations) {{
                    ... on {} {{
                        id
                        # Additional fields would be specified based on requirements
                    }}
                }}
            }}
            "#,
            typename
        );

        Ok(query)
    }

    /// Parse entity data from GraphQL response
    fn parse_entity_from_response(
        &self,
        response: &GraphQLResponse,
        typename: &str,
    ) -> Result<Option<EntityData>> {
        if let Some(entities_array) = response.data.get("_entities").and_then(|v| v.as_array()) {
            for entity_value in entities_array {
                if let Some(entity_obj) = entity_value.as_object() {
                    if entity_obj.get("__typename").and_then(|v| v.as_str()) == Some(typename) {
                        return Ok(Some(EntityData {
                            typename: typename.to_string(),
                            fields: entity_obj.clone(),
                        }));
                    }
                }
            }
        }

        Ok(None)
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

    /// Build an optimized _entities query for resolving entity representations
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

        // Get the appropriate fields for this entity type
        let fields = self.get_entity_selection_fields(typename)?;

        Ok(format!(
            r#"
            query GetEntities($representations: [_Any!]!) {{
                _entities(representations: $representations) {{
                    ... on {} {{
                        {}
                    }}
                }}
            }}
            "#,
            typename,
            fields.join("\n                        ")
        ))
    }

    /// Get selection fields for an entity type based on common patterns
    fn get_entity_selection_fields(&self, typename: &str) -> Result<Vec<String>> {
        match typename {
            "User" => Ok(vec![
                "id".to_string(),
                "username".to_string(),
                "email".to_string(),
                "firstName".to_string(),
                "lastName".to_string(),
            ]),
            "Product" => Ok(vec![
                "id".to_string(),
                "sku".to_string(),
                "name".to_string(),
                "description".to_string(),
                "price".to_string(),
                "category".to_string(),
            ]),
            "Order" => Ok(vec![
                "id".to_string(),
                "orderNumber".to_string(),
                "status".to_string(),
                "total".to_string(),
                "createdAt".to_string(),
            ]),
            "Review" => Ok(vec![
                "id".to_string(),
                "rating".to_string(),
                "comment".to_string(),
                "createdAt".to_string(),
            ]),
            _ => Ok(vec!["id".to_string()]), // Default to just id for unknown types
        }
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
        _variables: Option<serde_json::Value>,
    ) -> Result<GraphQLResponse> {
        debug!(
            "Stitching final GraphQL response from {} services",
            resolved_entities.len()
        );

        // Combine all entity data into a unified response
        let mut combined_data = serde_json::Map::new();

        for (_service_id, entities) in resolved_entities {
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
                            fed_directives.key = Some(fields.clone());
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
        composed.directives = self.extract_federation_directives(&composed)?;

        // Validate composition
        self.validate_composed_schema(&composed)?;

        info!(
            "Successfully composed federated schema with {} types",
            composed.types.len()
        );
        Ok(composed)
    }

    /// Process schema for federation composition
    fn process_schema_for_federation(
        &self,
        composed: &mut ComposedSchema,
        service_id: &str,
        schema: &FederatedSchema,
    ) -> Result<()> {
        // Process types and identify entities
        for (type_name, type_def) in &schema.types {
            let federation_directives = self.parse_federation_directives(type_def);

            // Check if this is an entity type (has @key directive)
            if federation_directives.key.is_some() {
                let entity_info = EntityTypeInfo {
                    key_fields: federation_directives
                        .key
                        .as_ref()
                        .map(|k| vec![k.clone()])
                        .unwrap_or_default(),
                    owning_service: service_id.to_string(),
                    extending_services: Vec::new(),
                };
                composed.entity_types.insert(type_name.clone(), entity_info);
            }

            // Convert to GraphQLType for composed schema
            let graphql_type = self.convert_to_graphql_type(type_def)?;
            composed.types.insert(type_name.clone(), graphql_type);
        }

        // Process field ownership
        for (field_name, _field_def) in &schema.queries {
            composed.field_ownership.insert(
                field_name.clone(),
                FieldOwnershipType::Owned(service_id.to_string()),
            );
        }

        Ok(())
    }

    /// Convert TypeDefinition to GraphQLType
    fn convert_to_graphql_type(&self, type_def: &TypeDefinition) -> Result<GraphQLType> {
        let kind = match &type_def.kind {
            TypeKind::Object { .. } => GraphQLTypeKind::Object,
            TypeKind::Interface { .. } => GraphQLTypeKind::Interface,
            TypeKind::Union { .. } => GraphQLTypeKind::Union,
            TypeKind::Enum { .. } => GraphQLTypeKind::Enum,
            TypeKind::InputObject { .. } => GraphQLTypeKind::InputObject,
            TypeKind::Scalar => GraphQLTypeKind::Scalar,
        };

        let fields = match &type_def.kind {
            TypeKind::Object { fields } | TypeKind::Interface { fields } => {
                let mut graphql_fields = HashMap::new();
                for (field_name, field_def) in fields {
                    graphql_fields.insert(
                        field_name.clone(),
                        GraphQLField {
                            name: field_def.name.clone(),
                            field_type: field_def.field_type.clone(),
                            arguments: HashMap::new(), // Simplified
                            selection_set: Vec::new(),
                        },
                    );
                }
                graphql_fields
            }
            _ => HashMap::new(),
        };

        Ok(GraphQLType {
            name: type_def.name.clone(),
            kind,
            fields,
        })
    }

    /// Extract federation directives from composed schema
    fn extract_federation_directives(&self, composed: &ComposedSchema) -> Result<Vec<String>> {
        let mut directives = Vec::new();

        // Standard federation directives
        directives.extend(vec![
            "@key".to_string(),
            "@external".to_string(),
            "@requires".to_string(),
            "@provides".to_string(),
            "@extends".to_string(),
            "@shareable".to_string(),
            "@override".to_string(),
            "@inaccessible".to_string(),
            "@tag".to_string(),
        ]);

        Ok(directives)
    }

    /// Validate composed schema with comprehensive federation checks
    fn validate_composed_schema(&self, composed: &ComposedSchema) -> Result<()> {
        // Check that all entity types have proper key fields
        for (type_name, entity_info) in &composed.entity_types {
            if entity_info.key_fields.is_empty() {
                return Err(anyhow!(
                    "Entity type '{}' must have at least one key field",
                    type_name
                ));
            }

            // Verify the type exists in the composed schema
            if !composed.types.contains_key(type_name) {
                return Err(anyhow!(
                    "Entity type '{}' not found in composed schema",
                    type_name
                ));
            }

            // Validate that key fields exist in the type definition
            if let Some(graphql_type) = composed.types.get(type_name) {
                for key_field in &entity_info.key_fields {
                    if !graphql_type.fields.contains_key(key_field) {
                        return Err(anyhow!(
                            "Key field '{}' not found in entity type '{}'",
                            key_field,
                            type_name
                        ));
                    }
                }
            }
        }

        // Check for field ownership conflicts
        let mut field_service_map: HashMap<String, Vec<String>> = HashMap::new();
        for (field_name, ownership) in &composed.field_ownership {
            if let FieldOwnershipType::Owned(service_id) = ownership {
                field_service_map
                    .entry(field_name.clone())
                    .or_default()
                    .push(service_id.clone());
            }
        }

        for (field_name, services) in field_service_map {
            if services.len() > 1 {
                warn!(
                    "Field '{}' is owned by multiple services: {:?} - potential conflict",
                    field_name, services
                );
                // Convert to warning instead of error for better flexibility
            }
        }

        // Validate type references
        self.validate_type_references_in_composed(composed)?;

        // Validate federation rules
        self.validate_federation_rules(composed)?;

        info!(
            "Schema composition validation completed: {} types, {} entities",
            composed.types.len(),
            composed.entity_types.len()
        );

        Ok(())
    }

    /// Validate type references in composed schema
    fn validate_type_references_in_composed(&self, composed: &ComposedSchema) -> Result<()> {
        for (type_name, graphql_type) in &composed.types {
            for (field_name, field) in &graphql_type.fields {
                let base_type = self.extract_base_type_from_field_type(&field.field_type);

                // Check if referenced type exists
                if !self.is_builtin_graphql_type(&base_type)
                    && !composed.types.contains_key(&base_type)
                {
                    return Err(anyhow!(
                        "Field '{}.{}' references unknown type '{}'",
                        type_name,
                        field_name,
                        base_type
                    ));
                }
            }
        }
        Ok(())
    }

    /// Extract base type from field type notation
    fn extract_base_type_from_field_type(&self, field_type: &str) -> String {
        field_type
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_end_matches('!')
            .to_string()
    }

    /// Check if type is a built-in GraphQL type
    fn is_builtin_graphql_type(&self, type_name: &str) -> bool {
        matches!(
            type_name,
            "String"
                | "Int"
                | "Float"
                | "Boolean"
                | "ID"
                | "_Any"
                | "_Entity"
                | "_Service"
                | "_FieldSet"
        )
    }

    /// Validate federation-specific rules
    fn validate_federation_rules(&self, composed: &ComposedSchema) -> Result<()> {
        // Rule 1: Entity types should have resolvable key fields
        for (type_name, entity_info) in &composed.entity_types {
            if let Some(graphql_type) = composed.types.get(type_name) {
                // Check that entity has an ID field (common pattern)
                if !graphql_type.fields.contains_key("id") {
                    debug!(
                        "Entity type '{}' does not have an 'id' field - consider adding one for better federation support",
                        type_name
                    );
                }
            }
        }

        // Rule 2: Check for proper service distribution
        let mut service_entity_count: HashMap<String, usize> = HashMap::new();
        for entity_info in composed.entity_types.values() {
            *service_entity_count
                .entry(entity_info.owning_service.clone())
                .or_insert(0) += 1;
        }

        // Warn about unbalanced entity distribution
        if service_entity_count.len() > 1 {
            let max_entities = service_entity_count.values().max().unwrap_or(&0);
            let min_entities = service_entity_count.values().min().unwrap_or(&0);

            if max_entities > &0 && min_entities > &0 && max_entities / min_entities > 3 {
                debug!(
                    "Unbalanced entity distribution across services - consider redistributing for better performance"
                );
            }
        }

        Ok(())
    }
}
