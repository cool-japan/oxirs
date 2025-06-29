//! Query Analysis and Parsing for Federated Queries
//!
//! This module handles GraphQL query parsing, analysis, and decomposition
//! for federated execution planning.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::types::*;

/// Query analysis utilities
pub struct QueryAnalyzer;

impl QueryAnalyzer {
    /// Parse a GraphQL query string into a structured representation
    pub fn parse_graphql_query(query: &str) -> Result<ParsedQuery> {
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
        let selection_set = Self::parse_selection_set(query)?;
        let variables = Self::parse_variables(query)?;

        Ok(ParsedQuery {
            operation_type,
            operation_name,
            selection_set,
            variables,
        })
    }

    /// Parse a selection set from a GraphQL query
    fn parse_selection_set(query: &str) -> Result<Vec<Selection>> {
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

    /// Parse variables from GraphQL query
    fn parse_variables(query: &str) -> Result<HashMap<String, serde_json::Value>> {
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
                                let parsed_value = Self::parse_variable_value(default_value)?;
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
    fn parse_variable_value(value: &str) -> Result<serde_json::Value> {
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

    /// Analyze which services own which fields
    pub fn analyze_field_ownership(
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
        }

        Ok(ownership)
    }

    /// Create service-specific queries based on field ownership
    pub fn create_service_queries(
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

    /// Decompose a GraphQL query for federation
    pub async fn decompose_query(
        query: &str,
        unified_schema: &UnifiedSchema,
    ) -> Result<Vec<ServiceQuery>> {
        debug!("Decomposing GraphQL query for federation");

        // Parse the query into an AST-like structure
        let parsed_query = Self::parse_graphql_query(query)?;

        // Analyze field ownership across services
        let field_ownership = Self::analyze_field_ownership(&parsed_query, unified_schema)?;

        // Decompose into service-specific queries
        let service_queries = Self::create_service_queries(&parsed_query, &field_ownership)?;

        debug!(
            "Decomposed query into {} service queries",
            service_queries.len()
        );
        Ok(service_queries)
    }

    /// Analyze query complexity for optimization
    pub fn analyze_query_complexity(query: &ParsedQuery) -> QueryComplexity {
        let mut complexity = QueryComplexity {
            field_count: 0,
            nesting_depth: 0,
            estimated_cost: 0.0,
            variables_count: query.variables.len(),
        };

        // Count fields and calculate nesting depth
        complexity.field_count = query.selection_set.len();
        complexity.nesting_depth = Self::calculate_max_depth(&query.selection_set);

        // Estimate cost based on fields and depth
        complexity.estimated_cost = (complexity.field_count as f64) * (1.0 + complexity.nesting_depth as f64 * 0.5);

        complexity
    }

    /// Calculate maximum nesting depth of selections
    fn calculate_max_depth(selections: &[Selection]) -> u32 {
        let mut max_depth = 0;
        for selection in selections {
            let depth = 1 + Self::calculate_max_depth(&selection.selection_set);
            max_depth = max_depth.max(depth);
        }
        max_depth
    }

    /// Extract query information for optimization
    pub fn extract_query_info(query: &ParsedQuery) -> QueryInfo {
        QueryInfo {
            operation_type: query.operation_type,
            field_count: query.selection_set.len(),
            has_variables: !query.variables.is_empty(),
            complexity: Self::analyze_query_complexity(query),
            estimated_execution_time: Self::estimate_execution_time(query),
        }
    }

    /// Estimate query execution time based on complexity
    fn estimate_execution_time(query: &ParsedQuery) -> std::time::Duration {
        let complexity = Self::analyze_query_complexity(query);
        
        // Base time of 10ms + 5ms per field + 2ms per nesting level
        let base_time_ms = 10.0;
        let field_time_ms = complexity.field_count as f64 * 5.0;
        let depth_time_ms = complexity.nesting_depth as f64 * 2.0;
        
        let total_ms = base_time_ms + field_time_ms + depth_time_ms;
        std::time::Duration::from_millis(total_ms as u64)
    }

    /// Check if query requires federation
    pub fn requires_federation(query: &ParsedQuery, schema: &UnifiedSchema) -> bool {
        let field_ownership = Self::analyze_field_ownership(query, schema).unwrap_or_else(|_| FieldOwnership {
            field_to_service: HashMap::new(),
            service_to_fields: HashMap::new(),
        });

        // Query requires federation if fields are owned by multiple services
        field_ownership.service_to_fields.len() > 1
    }

    /// Extract field dependencies from query
    pub fn extract_field_dependencies(query: &ParsedQuery) -> HashMap<String, Vec<String>> {
        let mut dependencies = HashMap::new();

        for selection in &query.selection_set {
            // For now, assume no dependencies (simplified)
            // In a real implementation, this would analyze @requires directives
            dependencies.insert(selection.name.clone(), Vec::new());
        }

        dependencies
    }

    /// Validate query against schema
    pub fn validate_query_against_schema(
        query: &ParsedQuery,
        schema: &UnifiedSchema,
    ) -> Result<Vec<String>> {
        let mut errors = Vec::new();

        for selection in &query.selection_set {
            match query.operation_type {
                GraphQLOperationType::Query => {
                    if !schema.queries.contains_key(&selection.name) {
                        errors.push(format!("Query field '{}' not found in schema", selection.name));
                    }
                }
                GraphQLOperationType::Mutation => {
                    if !schema.mutations.contains_key(&selection.name) {
                        errors.push(format!("Mutation field '{}' not found in schema", selection.name));
                    }
                }
                GraphQLOperationType::Subscription => {
                    if !schema.subscriptions.contains_key(&selection.name) {
                        errors.push(format!("Subscription field '{}' not found in schema", selection.name));
                    }
                }
            }
        }

        Ok(errors)
    }
}

/// Query complexity metrics
#[derive(Debug, Clone)]
pub struct QueryComplexity {
    pub field_count: usize,
    pub nesting_depth: u32,
    pub estimated_cost: f64,
    pub variables_count: usize,
}

impl Default for QueryComplexity {
    fn default() -> Self {
        Self {
            field_count: 0,
            nesting_depth: 0,
            estimated_cost: 1.0,
            variables_count: 0,
        }
    }
}

/// Query information for optimization
#[derive(Debug, Clone)]
pub struct QueryInfo {
    pub operation_type: GraphQLOperationType,
    pub field_count: usize,
    pub has_variables: bool,
    pub complexity: QueryComplexity,
    pub estimated_execution_time: std::time::Duration,
}