//! Query processing for GraphQL Federation
//!
//! This module handles query parsing, decomposition, field ownership analysis,
//! and service-specific query generation for GraphQL federation.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::debug;

use super::types::*;
use crate::planner::planning::types::{ExecutionPlan, ExecutionStep, StepType};

impl GraphQLFederation {
    /// Decompose a GraphQL query for federation with advanced planning
    pub async fn decompose_query(&self, query: &str) -> Result<Vec<ServiceQuery>> {
        debug!("Decomposing GraphQL query for federation");

        // Parse the query into an AST-like structure
        let parsed_query = self.parse_graphql_query(query)?;

        // Validate query complexity
        let complexity = self.analyze_query_complexity(&parsed_query)?;
        if complexity.total_complexity > self.config.max_query_complexity {
            return Err(anyhow!(
                "Query complexity {} exceeds maximum allowed {}",
                complexity.total_complexity,
                self.config.max_query_complexity
            ));
        }

        // Get the unified schema to understand field ownership
        let unified_schema = self.create_unified_schema().await?;

        // Analyze field ownership and dependencies
        let field_ownership = self.analyze_field_ownership(&parsed_query, &unified_schema)?;

        // Check for entity resolution requirements
        let entity_requirements =
            self.analyze_entity_requirements(&parsed_query, &unified_schema)?;

        // Generate execution plan with entity resolution
        let execution_plan =
            self.create_execution_plan(&parsed_query, &field_ownership, &entity_requirements)?;

        // Decompose into optimized service-specific queries
        let service_queries = self.create_optimized_service_queries(
            &execution_plan,
            &parsed_query,
            &field_ownership,
        )?;

        debug!(
            "Decomposed query into {} service queries with {} entity resolution steps",
            service_queries.len(),
            entity_requirements.len()
        );
        Ok(service_queries)
    }

    /// Analyze entity resolution requirements
    fn analyze_entity_requirements(
        &self,
        query: &ParsedQuery,
        schema: &UnifiedSchema,
    ) -> Result<Vec<EntityReference>> {
        let mut entity_requirements = Vec::new();

        // Analyze selections for entity references
        self.collect_entity_requirements_from_selections(
            &query.selection_set,
            schema,
            &mut entity_requirements,
        )?;

        Ok(entity_requirements)
    }

    /// Collect entity requirements from selections
    fn collect_entity_requirements_from_selections(
        &self,
        selections: &[Selection],
        schema: &UnifiedSchema,
        entity_requirements: &mut Vec<EntityReference>,
    ) -> Result<()> {
        for selection in selections {
            // Check if this field represents an entity
            if let Some(field_def) = schema.queries.get(&selection.name) {
                // Check if the return type is an entity
                let return_type = self.extract_base_type(&field_def.field_type);
                if self.is_entity_type(&return_type, schema) {
                    entity_requirements.push(EntityReference {
                        entity_type: return_type.clone(),
                        key_fields: vec!["id".to_string()], // Default key field
                        required_fields: self.extract_required_fields(&selection.selection_set),
                        service_id: self.determine_service_for_entity(&return_type, schema)?,
                    });
                }
            }

            // Recursively check nested selections
            self.collect_entity_requirements_from_selections(
                &selection.selection_set,
                schema,
                entity_requirements,
            )?;
        }

        Ok(())
    }

    /// Check if a type is an entity type
    fn is_entity_type(&self, type_name: &str, schema: &UnifiedSchema) -> bool {
        // Check if type exists and could be an entity
        schema.types.contains_key(type_name)
            && !matches!(type_name, "String" | "Int" | "Float" | "Boolean" | "ID")
    }

    /// Extract required fields from selection set
    fn extract_required_fields(&self, selections: &[Selection]) -> Vec<String> {
        selections.iter().map(|s| s.name.clone()).collect()
    }

    /// Determine which service owns an entity type
    fn determine_service_for_entity(
        &self,
        entity_type: &str,
        schema: &UnifiedSchema,
    ) -> Result<String> {
        if let Some(services) = schema.schema_mapping.get(entity_type) {
            if let Some(service) = services.first() {
                Ok(service.clone())
            } else {
                Err(anyhow!("No service found for entity type: {}", entity_type))
            }
        } else {
            Err(anyhow!(
                "Entity type not found in schema mapping: {}",
                entity_type
            ))
        }
    }

    /// Create execution plan with entity resolution
    fn create_execution_plan(
        &self,
        query: &ParsedQuery,
        field_ownership: &FieldOwnership,
        entity_requirements: &[EntityReference],
    ) -> Result<ExecutionPlan> {
        let mut steps = Vec::new();
        let mut step_counter = 0;

        // Create steps for entity resolution if needed
        for entity_req in entity_requirements {
            steps.push(ExecutionStep {
                step_id: format!("entity_resolution_{step_counter}"),
                step_type: StepType::EntityResolution,
                service_id: Some(entity_req.service_id.clone()),
                query_fragment: format!(
                    "_entities(representations: [{{__typename: \"{}\", id: \"$id\"}}])",
                    entity_req.entity_type
                ),
                dependencies: Vec::new(),
                estimated_cost: 1.0,
                timeout: std::time::Duration::from_secs(30),
                retry_config: None,
            });
            step_counter += 1;
        }

        // Create steps for field resolution
        for (service_id, fields) in &field_ownership.service_to_fields {
            if !fields.is_empty() {
                steps.push(ExecutionStep {
                    step_id: format!("service_query_{step_counter}"),
                    step_type: StepType::GraphQLQuery,
                    service_id: Some(service_id.clone()),
                    query_fragment: fields.join(", "),
                    dependencies: Vec::new(),
                    estimated_cost: fields.len() as f64,
                    timeout: std::time::Duration::from_secs(30),
                    retry_config: None,
                });
                step_counter += 1;
            }
        }

        // Add result stitching step
        steps.push(ExecutionStep {
            step_id: "result_stitching".to_string(),
            step_type: StepType::ResultStitching,
            service_id: None,
            query_fragment: String::new(),
            dependencies: steps.iter().map(|s| s.step_id.clone()).collect(),
            estimated_cost: 0.5,
            timeout: std::time::Duration::from_secs(10),
            retry_config: None,
        });

        Ok(ExecutionPlan {
            query_id: uuid::Uuid::new_v4().to_string(),
            steps,
            estimated_total_cost: 0.0, // Will be calculated
            max_parallelism: 4,
            planning_time: std::time::Duration::from_millis(0),
            cache_key: None,
            metadata: HashMap::new(),
            parallelizable_steps: Vec::new(),
        })
    }

    /// Create optimized service queries with entity resolution
    fn create_optimized_service_queries(
        &self,
        execution_plan: &ExecutionPlan,
        query: &ParsedQuery,
        field_ownership: &FieldOwnership,
    ) -> Result<Vec<ServiceQuery>> {
        let mut service_queries = Vec::new();

        // Generate queries from execution plan steps
        for step in &execution_plan.steps {
            if let Some(service_id) = &step.service_id {
                match step.step_type {
                    StepType::GraphQLQuery => {
                        let fields = field_ownership
                            .service_to_fields
                            .get(service_id)
                            .cloned()
                            .unwrap_or_default();

                        if !fields.is_empty() {
                            let operation_type = match query.operation_type {
                                GraphQLOperationType::Query => "query",
                                GraphQLOperationType::Mutation => "mutation",
                                GraphQLOperationType::Subscription => "subscription",
                            };

                            let service_query =
                                format!("{} {{ {} }}", operation_type, fields.join(" "));

                            service_queries.push(ServiceQuery {
                                service_id: service_id.clone(),
                                query: service_query,
                                variables: None,
                            });
                        }
                    }
                    StepType::EntityResolution => {
                        // Create entity resolution query
                        let entity_query = format!(
                            "query($_representations: [_Any!]!) {{ {} }}",
                            step.query_fragment
                        );

                        service_queries.push(ServiceQuery {
                            service_id: service_id.clone(),
                            query: entity_query,
                            variables: Some(serde_json::json!({
                                "representations": []
                            })),
                        });
                    }
                    _ => {}
                }
            }
        }

        Ok(service_queries)
    }

    /// Parse a GraphQL query string into a structured representation
    pub fn parse_graphql_query(&self, query: &str) -> Result<ParsedQuery> {
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

                    // Parse field with potential arguments and nested selections
                    let selection = self.parse_field_selection(line)?;
                    if let Some(selection) = selection {
                        selections.push(selection);
                    }
                }
            }
        }

        Ok(selections)
    }

    /// Parse a single field selection
    fn parse_field_selection(&self, field_str: &str) -> Result<Option<Selection>> {
        let field_str = field_str.trim();

        // Skip empty lines and comments
        if field_str.is_empty() || field_str.starts_with('#') {
            return Ok(None);
        }

        // Handle fragment spreads and inline fragments
        if field_str.starts_with("...") {
            // TODO: Implement fragment handling
            return Ok(None);
        }

        // Parse field name and alias
        let (alias, field_name) = if let Some(colon_pos) = field_str.find(':') {
            let alias_part = field_str[..colon_pos].trim();
            let field_part = field_str[colon_pos + 1..].trim();
            let field_name = field_part.split_whitespace().next().unwrap_or(field_part);
            (Some(alias_part.to_string()), field_name.to_string())
        } else {
            let field_name = field_str.split_whitespace().next().unwrap_or(field_str);
            (None, field_name.to_string())
        };

        // Parse arguments (simplified)
        let arguments = self.parse_field_arguments(field_str)?;

        // Parse nested selection set (simplified)
        let selection_set = if field_str.contains('{') {
            self.parse_nested_selection_set(field_str)?
        } else {
            Vec::new()
        };

        Ok(Some(Selection {
            name: field_name,
            alias,
            arguments,
            selection_set,
        }))
    }

    /// Parse field arguments from a field string
    fn parse_field_arguments(&self, field_str: &str) -> Result<HashMap<String, serde_json::Value>> {
        let mut arguments = HashMap::new();

        // Find arguments between parentheses
        if let Some(start) = field_str.find('(') {
            if let Some(end) = field_str.find(')') {
                let args_content = &field_str[start + 1..end];

                // Split arguments by comma (simplified parsing)
                for arg in args_content.split(',') {
                    let arg = arg.trim();
                    if let Some(colon_pos) = arg.find(':') {
                        let arg_name = arg[..colon_pos].trim().to_string();
                        let arg_value_str = arg[colon_pos + 1..].trim();
                        let arg_value = self.parse_value_from_string(arg_value_str)?;
                        arguments.insert(arg_name, arg_value);
                    }
                }
            }
        }

        Ok(arguments)
    }

    /// Parse nested selection set from a field string
    fn parse_nested_selection_set(&self, field_str: &str) -> Result<Vec<Selection>> {
        if let Some(start) = field_str.find('{') {
            if let Some(end) = field_str.rfind('}') {
                let nested_content = &field_str[start + 1..end];
                return self.parse_selection_set(&format!("{{{nested_content}}}"));
            }
        }
        Ok(Vec::new())
    }

    /// Analyze which services own which fields
    pub fn analyze_field_ownership(
        &self,
        query: &ParsedQuery,
        schema: &UnifiedSchema,
    ) -> Result<FieldOwnership> {
        let mut ownership = FieldOwnership {
            field_to_service: HashMap::new(),
            service_to_fields: HashMap::new(),
        };

        for selection in &query.selection_set {
            self.analyze_selection_ownership(selection, query, schema, &mut ownership)?;
        }

        Ok(ownership)
    }

    /// Analyze ownership for a single selection
    fn analyze_selection_ownership(
        &self,
        selection: &Selection,
        query: &ParsedQuery,
        schema: &UnifiedSchema,
        ownership: &mut FieldOwnership,
    ) -> Result<()> {
        // Determine which service owns this field
        let service_ids = match query.operation_type {
            GraphQLOperationType::Query => {
                if let Some(_field_def) = schema.queries.get(&selection.name) {
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

        // Recursively analyze nested selections
        for nested_selection in &selection.selection_set {
            self.analyze_selection_ownership(nested_selection, query, schema, ownership)?;
        }

        Ok(())
    }

    /// Create service-specific queries based on field ownership
    pub fn create_service_queries(
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
                .map(|name| format!(" {name}"))
                .unwrap_or_default();

            // Create service-specific selections
            let service_selections =
                self.filter_selections_for_service(&query.selection_set, service_id, ownership);

            let field_strings = self.format_selections(&service_selections, 1);

            let service_query = format!(
                "{}{} {{\n{}\n}}",
                operation_type_str,
                operation_name,
                field_strings.join("\n")
            );

            // Filter variables based on field usage
            let filtered_variables =
                self.filter_variables_for_service(&query.variables, &service_selections);

            service_queries.push(ServiceQuery {
                service_id: service_id.clone(),
                query: service_query,
                variables: if filtered_variables.is_empty() {
                    None
                } else {
                    Some(serde_json::to_value(filtered_variables)?)
                },
            });
        }

        Ok(service_queries)
    }

    /// Filter selections for a specific service
    fn filter_selections_for_service(
        &self,
        selections: &[Selection],
        service_id: &str,
        ownership: &FieldOwnership,
    ) -> Vec<Selection> {
        let mut filtered = Vec::new();

        for selection in selections {
            // Check if this field belongs to the service
            if let Some(field_service) = ownership.field_to_service.get(&selection.name) {
                if field_service == service_id {
                    let mut filtered_selection = selection.clone();

                    // Recursively filter nested selections
                    filtered_selection.selection_set = self.filter_selections_for_service(
                        &selection.selection_set,
                        service_id,
                        ownership,
                    );

                    filtered.push(filtered_selection);
                }
            }
        }

        filtered
    }

    /// Format selections as GraphQL query strings
    fn format_selections(&self, selections: &[Selection], indent_level: usize) -> Vec<String> {
        let mut formatted = Vec::new();
        let indent = "  ".repeat(indent_level);

        for selection in selections {
            let mut field_str = String::new();

            // Add alias if present
            if let Some(alias) = &selection.alias {
                field_str.push_str(&format!("{alias}: "));
            }

            // Add field name
            field_str.push_str(&selection.name);

            // Add arguments if present
            if !selection.arguments.is_empty() {
                let args: Vec<String> = selection
                    .arguments
                    .iter()
                    .map(|(name, value)| format!("{}: {}", name, self.format_value(value)))
                    .collect();
                field_str.push_str(&format!("({})", args.join(", ")));
            }

            // Add nested selections if present
            if !selection.selection_set.is_empty() {
                field_str.push_str(" {");
                formatted.push(format!("{indent}{field_str}"));

                let nested_formatted =
                    self.format_selections(&selection.selection_set, indent_level + 1);
                formatted.extend(nested_formatted);

                formatted.push(format!("{indent}}}"));
            } else {
                formatted.push(format!("{indent}{field_str}"));
            }
        }

        formatted
    }

    /// Format a JSON value as GraphQL literal
    fn format_value(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(s) => format!("\"{s}\""),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Null => "null".to_string(),
            serde_json::Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| self.format_value(v)).collect();
                format!("[{}]", items.join(", "))
            }
            serde_json::Value::Object(obj) => {
                let items: Vec<String> = obj
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, self.format_value(v)))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
        }
    }

    /// Filter variables based on field usage in selections
    fn filter_variables_for_service(
        &self,
        variables: &HashMap<String, VariableDefinition>,
        selections: &[Selection],
    ) -> HashMap<String, VariableDefinition> {
        let mut used_variables = std::collections::HashSet::new();
        self.collect_used_variables(selections, &mut used_variables);

        variables
            .iter()
            .filter(|(name, _)| used_variables.contains(*name))
            .map(|(name, var_def)| (name.clone(), var_def.clone()))
            .collect()
    }

    /// Collect all variable names used in selections
    fn collect_used_variables(
        &self,
        selections: &[Selection],
        used_variables: &mut std::collections::HashSet<String>,
    ) {
        for selection in selections {
            // Check arguments for variable usage
            for value in selection.arguments.values() {
                self.collect_variables_from_value(value, used_variables);
            }

            // Recursively check nested selections
            self.collect_used_variables(&selection.selection_set, used_variables);
        }
    }

    /// Collect variable names from a JSON value
    fn collect_variables_from_value(
        &self,
        value: &serde_json::Value,
        used_variables: &mut std::collections::HashSet<String>,
    ) {
        match value {
            serde_json::Value::String(s) => {
                if s.starts_with('$') {
                    used_variables.insert(s[1..].to_string());
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.collect_variables_from_value(item, used_variables);
                }
            }
            serde_json::Value::Object(obj) => {
                for val in obj.values() {
                    self.collect_variables_from_value(val, used_variables);
                }
            }
            _ => {}
        }
    }

    /// Analyze query complexity
    pub fn analyze_query_complexity(&self, query: &ParsedQuery) -> Result<QueryComplexity> {
        let mut complexity = QueryComplexity {
            total_complexity: 0,
            max_depth: 0,
            field_count: 0,
            estimated_execution_time: std::time::Duration::from_millis(0),
        };

        // Calculate complexity metrics
        complexity.max_depth = self.calculate_max_depth(&query.selection_set, 1);
        complexity.field_count = self.count_fields(&query.selection_set);

        // Base complexity calculation
        complexity.total_complexity = complexity.field_count * 2;

        // Add depth penalty
        if complexity.max_depth > 5 {
            complexity.total_complexity += (complexity.max_depth - 5) * 10;
        }

        // Estimate execution time based on complexity
        let base_time_ms = complexity.total_complexity as u64 * 10;
        complexity.estimated_execution_time = std::time::Duration::from_millis(base_time_ms);

        Ok(complexity)
    }

    /// Calculate maximum query depth
    fn calculate_max_depth(&self, selections: &[Selection], current_depth: usize) -> usize {
        let mut max_depth = current_depth;

        for selection in selections {
            if !selection.selection_set.is_empty() {
                let nested_depth =
                    self.calculate_max_depth(&selection.selection_set, current_depth + 1);
                max_depth = max_depth.max(nested_depth);
            }
        }

        max_depth
    }

    /// Count total number of fields in query
    fn count_fields(&self, selections: &[Selection]) -> usize {
        let mut count = selections.len();

        for selection in selections {
            count += self.count_fields(&selection.selection_set);
        }

        count
    }

    /// Validate query against schema
    pub async fn validate_query(&self, query: &str) -> Result<Vec<SchemaValidationError>> {
        let mut errors = Vec::new();

        // Parse the query
        let parsed_query = match self.parse_graphql_query(query) {
            Ok(parsed) => parsed,
            Err(e) => {
                errors.push(SchemaValidationError {
                    error_type: SchemaErrorType::InvalidDirective, // Using closest available type
                    message: format!("Query parsing failed: {e}"),
                    location: None,
                });
                return Ok(errors);
            }
        };

        // Get unified schema
        let schema = match self.create_unified_schema().await {
            Ok(schema) => schema,
            Err(e) => {
                errors.push(SchemaValidationError {
                    error_type: SchemaErrorType::TypeNotFound,
                    message: format!("Could not create unified schema: {e}"),
                    location: None,
                });
                return Ok(errors);
            }
        };

        // Validate selections against schema
        self.validate_selections(
            &parsed_query.selection_set,
            &parsed_query.operation_type,
            &schema,
            &mut errors,
        );

        Ok(errors)
    }

    /// Validate selections against schema
    fn validate_selections(
        &self,
        selections: &[Selection],
        operation_type: &GraphQLOperationType,
        schema: &UnifiedSchema,
        errors: &mut Vec<SchemaValidationError>,
    ) {
        for selection in selections {
            // Check if field exists in the appropriate root type
            let field_exists = match operation_type {
                GraphQLOperationType::Query => schema.queries.contains_key(&selection.name),
                GraphQLOperationType::Mutation => schema.mutations.contains_key(&selection.name),
                GraphQLOperationType::Subscription => {
                    schema.subscriptions.contains_key(&selection.name)
                }
            };

            if !field_exists {
                errors.push(SchemaValidationError {
                    error_type: SchemaErrorType::FieldNotFound,
                    message: format!("Field '{}' not found in schema", selection.name),
                    location: Some(selection.name.clone()),
                });
            }

            // Recursively validate nested selections
            // Note: In a real implementation, this would need to track the current type context
            self.validate_selections(&selection.selection_set, operation_type, schema, errors);
        }
    }
}
