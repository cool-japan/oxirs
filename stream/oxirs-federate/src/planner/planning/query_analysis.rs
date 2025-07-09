//! Query Analysis and Parsing for Federated Queries
//!
//! This module handles GraphQL query parsing, analysis, and decomposition
//! for federated execution planning.

use anyhow::Result;
use std::collections::HashMap;
use tracing::debug;

use super::types::*;

/// Query analysis utilities
#[derive(Debug)]
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

                    // Parse field with potential arguments
                    if let Some(selection) = QueryAnalyzer::parse_field_selection(line)? {
                        selections.push(selection);
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
                .map(|name| format!(" {name}"))
                .unwrap_or_default();

            let field_strings: Vec<String> =
                fields.iter().map(|field| format!("  {field}")).collect();

            let service_query = format!(
                "{}{} {{\n{}\n}}",
                operation_type_str,
                operation_name,
                field_strings.join("\n")
            );

            // Filter variables based on fields used by this service
            let filtered_variables = filter_variables_by_field_usage(
                &query.variables,
                fields,
                &query.selection_set,
            );
            
            // Convert HashMap to serde_json::Value
            let variables_json = if filtered_variables.is_empty() {
                None
            } else {
                Some(serde_json::to_value(filtered_variables)?)
            };

            service_queries.push(ServiceQuery {
                service_id: service_id.clone(),
                query: service_query,
                variables: variables_json,
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
        complexity.estimated_cost =
            (complexity.field_count as f64) * (1.0 + complexity.nesting_depth as f64 * 0.5);

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
        let field_ownership =
            Self::analyze_field_ownership(query, schema).unwrap_or_else(|_| FieldOwnership {
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
                        errors.push(format!(
                            "Query field '{}' not found in schema",
                            selection.name
                        ));
                    }
                }
                GraphQLOperationType::Mutation => {
                    if !schema.mutations.contains_key(&selection.name) {
                        errors.push(format!(
                            "Mutation field '{}' not found in schema",
                            selection.name
                        ));
                    }
                }
                GraphQLOperationType::Subscription => {
                    if !schema.subscriptions.contains_key(&selection.name) {
                        errors.push(format!(
                            "Subscription field '{}' not found in schema",
                            selection.name
                        ));
                    }
                }
            }
        }

        Ok(errors)
    }

    /// Parse a single field selection from a line of GraphQL query
    fn parse_field_selection(field_str: &str) -> Result<Option<Selection>> {
        let field_str = field_str.trim();

        // Skip empty lines and comments
        if field_str.is_empty() || field_str.starts_with('#') {
            return Ok(None);
        }

        // Handle fragment spreads and inline fragments
        if field_str.starts_with("...") {
            return parse_fragment_selection(field_str);
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

        // Parse arguments (simplified) - look for parentheses
        let mut arguments = HashMap::new();
        if let Some(start) = field_str.find('(') {
            if let Some(end) = field_str.find(')') {
                let args_str = &field_str[start + 1..end];
                // Simple argument parsing - split by comma and parse key: value pairs
                for arg in args_str.split(',') {
                    let arg = arg.trim();
                    if let Some(colon_pos) = arg.find(':') {
                        let key = arg[..colon_pos].trim();
                        let value_str = arg[colon_pos + 1..].trim();
                        // Simple value parsing
                        let value = if value_str.starts_with('"') && value_str.ends_with('"') {
                            serde_json::Value::String(value_str[1..value_str.len() - 1].to_string())
                        } else if let Ok(num) = value_str.parse::<i64>() {
                            serde_json::Value::Number(serde_json::Number::from(num))
                        } else {
                            serde_json::Value::String(value_str.to_string())
                        };
                        arguments.insert(key.to_string(), value);
                    }
                }
            }
        }

        // For now, don't parse nested selection sets in this simplified implementation
        let selection_set = Vec::new();

        Ok(Some(Selection {
            name: field_name,
            alias,
            arguments,
            selection_set,
        }))
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

/// Filter variables based on their usage in specific fields
pub fn filter_variables_by_field_usage(
    original_variables: &HashMap<String, serde_json::Value>,
    service_fields: &[String],
    selection_set: &[Selection],
) -> HashMap<String, serde_json::Value> {
    let mut used_variables = std::collections::HashSet::new();
    
    // Extract variables used in the fields assigned to this service
    for field_name in service_fields {
        // Find the selection for this field
        if let Some(selection) = selection_set.iter().find(|s| &s.name == field_name) {
            // Extract variables from field arguments
            extract_variables_from_arguments(&selection.arguments, &mut used_variables);
            
            // Extract variables from nested selections
            extract_variables_from_selections(&selection.selection_set, &mut used_variables);
        }
    }
    
    // Filter the original variables to only include used ones
    let mut filtered_variables = HashMap::new();
    for var_name in used_variables {
        if let Some(var_value) = original_variables.get(&var_name) {
            filtered_variables.insert(var_name, var_value.clone());
        }
    }
    
    filtered_variables
}

/// Extract variable names from field arguments
fn extract_variables_from_arguments(
    arguments: &HashMap<String, serde_json::Value>,
    used_variables: &mut std::collections::HashSet<String>,
) {
    for value in arguments.values() {
        extract_variables_from_value(value, used_variables);
    }
}

/// Extract variable names from a JSON value (could contain variable references)
fn extract_variables_from_value(
    value: &serde_json::Value,
    used_variables: &mut std::collections::HashSet<String>,
) {
    match value {
        serde_json::Value::String(s) => {
            // Check if string is a variable reference (e.g., "$variableName")
            if s.starts_with('$') {
                let var_name = s.trim_start_matches('$');
                used_variables.insert(var_name.to_string());
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                extract_variables_from_value(item, used_variables);
            }
        }
        serde_json::Value::Object(obj) => {
            for (_, v) in obj {
                extract_variables_from_value(v, used_variables);
            }
        }
        _ => {} // Numbers, booleans, null don't contain variable references
    }
}

/// Extract variables from nested selections
fn extract_variables_from_selections(
    selections: &[Selection],
    used_variables: &mut std::collections::HashSet<String>,
) {
    for selection in selections {
        // Extract from arguments
        extract_variables_from_arguments(&selection.arguments, used_variables);
        
        // Extract from nested selections recursively
        extract_variables_from_selections(&selection.selection_set, used_variables);
    }
}

/// Parse fragment spreads and inline fragments
fn parse_fragment_selection(fragment_str: &str) -> Result<Option<Selection>> {
    let fragment_str = fragment_str.trim();
    
    if fragment_str.starts_with("...") {
        let content = &fragment_str[3..].trim(); // Remove "..."
        
        if content.starts_with("on ") {
            // Inline fragment: ... on TypeName { fields }
            parse_inline_fragment(content)
        } else if content.contains(' ') {
            // Complex fragment with conditions
            parse_conditional_fragment(content)
        } else {
            // Named fragment spread: ...FragmentName
            parse_named_fragment_spread(content)
        }
    } else {
        Ok(None)
    }
}

/// Parse inline fragment: ... on TypeName { fields }
fn parse_inline_fragment(content: &str) -> Result<Option<Selection>> {
    if !content.starts_with("on ") {
        return Ok(None);
    }
    
    let content = &content[3..]; // Remove "on "
    let parts: Vec<&str> = content.splitn(2, ' ').collect();
    
    if parts.len() < 2 {
        return Ok(None);
    }
    
    let type_name = parts[0];
    let field_content = parts[1];
    
    // Parse the field content within braces
    let mut selection_set = Vec::new();
    if let Some(start) = field_content.find('{') {
        if let Some(end) = field_content.rfind('}') {
            let fields_content = &field_content[start + 1..end];
            
            // Parse each field in the inline fragment
            for line in fields_content.lines() {
                let line = line.trim();
                if !line.is_empty() && !line.starts_with('#') {
                    if let Some(selection) = QueryAnalyzer::parse_field_selection(line)? {
                        selection_set.push(selection);
                    }
                }
            }
        }
    }
    
    // Create a virtual selection representing the inline fragment
    Ok(Some(Selection {
        name: format!("__inline_fragment_on_{type_name}"),
        alias: None,
        arguments: HashMap::new(),
        selection_set,
    }))
}

/// Parse named fragment spread: ...FragmentName
fn parse_named_fragment_spread(fragment_name: &str) -> Result<Option<Selection>> {
    let fragment_name = fragment_name.trim();
    
    if fragment_name.is_empty() {
        return Ok(None);
    }
    
    // Create a virtual selection representing the fragment spread
    // In a full implementation, this would resolve the fragment definition
    Ok(Some(Selection {
        name: format!("__fragment_spread_{fragment_name}"),
        alias: None,
        arguments: HashMap::new(),
        selection_set: Vec::new(), // Would be populated by resolving the fragment
    }))
}

/// Parse conditional fragment with directives
fn parse_conditional_fragment(content: &str) -> Result<Option<Selection>> {
    // Handle fragments with directives like: ... @include(if: $condition) { fields }
    // or: ... on TypeName @skip(if: $condition) { fields }
    
    let parts = content.split_whitespace();
    let mut type_condition = None;
    let mut directives = Vec::new();
    let mut remaining_content = String::new();
    
    // Parse the fragment components
    for part in parts {
        if part == "on" {
            // Next part should be the type name
            continue;
        } else if part.starts_with('@') {
            // Directive
            directives.push(part.to_string());
        } else if part.starts_with('{') {
            // Start of selection set
            remaining_content = content[content.find('{').unwrap()..].to_string();
            break;
        } else if type_condition.is_none() && !part.starts_with('@') {
            // Type condition
            type_condition = Some(part.to_string());
        }
    }
    
    // Parse selection set if present
    let mut selection_set = Vec::new();
    if let Some(start) = remaining_content.find('{') {
        if let Some(end) = remaining_content.rfind('}') {
            let fields_content = &remaining_content[start + 1..end];
            
            for line in fields_content.lines() {
                let line = line.trim();
                if !line.is_empty() && !line.starts_with('#') {
                    if let Some(selection) = QueryAnalyzer::parse_field_selection(line)? {
                        selection_set.push(selection);
                    }
                }
            }
        }
    }
    
    // Create fragment representation
    let fragment_name = if let Some(type_name) = type_condition {
        format!("__conditional_fragment_on_{type_name}")
    } else {
        "__conditional_fragment".to_string()
    };
    
    // Store directive information in arguments (simplified approach)
    let mut arguments = HashMap::new();
    for directive in directives {
        arguments.insert(
            "directive".to_string(),
            serde_json::Value::String(directive),
        );
    }
    
    Ok(Some(Selection {
        name: fragment_name,
        alias: None,
        arguments,
        selection_set,
    }))
}
