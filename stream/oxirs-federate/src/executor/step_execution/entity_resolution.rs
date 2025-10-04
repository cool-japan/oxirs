//! Entity resolution functions for federated queries

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use tracing::debug;
use super::super::types::*;

pub fn perform_sparql_entity_resolution(
    results: &SparqlResults,
    query_fragment: &str,
) -> Result<SparqlResults> {
    debug!("Performing SPARQL entity resolution");

    // Extract entity resolution keys from query fragment
    let resolution_keys = extract_entity_resolution_keys(query_fragment);

    if resolution_keys.is_empty() {
        debug!("No resolution keys found, returning original results");
        return Ok(results.clone());
    }

    // Group bindings by entity keys
    let mut entity_groups: HashMap<String, Vec<SparqlBinding>> = HashMap::new();

    for binding in &results.results.bindings {
        let entity_key = compute_entity_key(binding, &resolution_keys);
        entity_groups
            .entry(entity_key)
            .or_default()
            .push(binding.clone());
    }

    // Merge bindings within each entity group
    let mut resolved_bindings = Vec::new();
    for (_, group_bindings) in entity_groups {
        if group_bindings.len() == 1 {
            // Single binding, no merging needed
            resolved_bindings.push(group_bindings.into_iter().next().unwrap());
        } else {
            // Multiple bindings for same entity, merge them
            let merged_binding = merge_sparql_bindings(&group_bindings)?;
            resolved_bindings.push(merged_binding);
        }
    }

    Ok(SparqlResults {
        head: results.head.clone(),
        results: SparqlResultsData {
            bindings: resolved_bindings,
        },
    })
}

/// Perform entity resolution on GraphQL responses
pub fn perform_graphql_entity_resolution(
    response: &GraphQLResponse,
    query_fragment: &str,
) -> Result<GraphQLResponse> {
    debug!("Performing GraphQL entity resolution");

    // Extract entity resolution configuration from query fragment
    let resolution_config = extract_graphql_resolution_config(query_fragment);

    let resolved_data = resolve_graphql_entities(&response.data, &resolution_config)?;

    Ok(GraphQLResponse {
        data: resolved_data,
        errors: response.errors.clone(),
        extensions: response.extensions.clone(),
    })
}

/// Perform entity resolution on service results
pub fn perform_service_entity_resolution(
    result: &serde_json::Value,
    query_fragment: &str,
) -> Result<serde_json::Value> {
    debug!("Performing service entity resolution");

    // Extract resolution configuration
    let resolution_config = extract_service_resolution_config(query_fragment);

    let resolved_result = resolve_service_entities(result, &resolution_config)?;

    Ok(resolved_result)
}

/// Extract entity resolution keys from query fragment
fn extract_entity_resolution_keys(query_fragment: &str) -> Vec<String> {
    let mut keys = Vec::new();

    // Look for @key directives or similar annotations
    if let Some(start) = query_fragment.find("@key") {
        if let Some(end) = query_fragment[start..].find(')') {
            let key_section = &query_fragment[start..start + end];
            // Extract fields within parentheses
            if let Some(fields_start) = key_section.find('(') {
                let fields_str = &key_section[fields_start + 1..];
                for field in fields_str.split(',') {
                    let field = field.trim().trim_matches('"').trim_matches('\'');
                    if !field.is_empty() {
                        keys.push(field.to_string());
                    }
                }
            }
        }
    }

    // If no explicit keys found, use common entity identifiers
    if keys.is_empty() {
        for common_key in &["id", "uri", "identifier", "key"] {
            if query_fragment.contains(common_key) {
                keys.push(common_key.to_string());
            }
        }
    }

    keys
}

/// Compute entity key from binding
fn compute_entity_key(binding: &SparqlBinding, resolution_keys: &[String]) -> String {
    let mut key_parts = Vec::new();

    for key in resolution_keys {
        if let Some(value) = binding.get(key) {
            key_parts.push(format!("{}:{}", key, value.value));
        } else {
            key_parts.push(format!("{key}:null"));
        }
    }

    key_parts.join("|")
}

/// Merge multiple SPARQL bindings for the same entity
fn merge_sparql_bindings(bindings: &[SparqlBinding]) -> Result<SparqlBinding> {
    if bindings.is_empty() {
        return Ok(HashMap::new());
    }

    let mut merged = bindings[0].clone();

    for binding in bindings.iter().skip(1) {
        for (var, value) in binding {
            match merged.get(var) {
                Some(existing_value) => {
                    // If values differ, prefer non-null, more specific, or first value
                    if existing_value.value != value.value
                        && existing_value.value.is_empty()
                        && !value.value.is_empty()
                    {
                        merged.insert(var.clone(), value.clone());
                    }
                    // Otherwise keep existing value
                }
                None => {
                    merged.insert(var.clone(), value.clone());
                }
            }
        }
    }

    Ok(merged)
}

/// GraphQL entity resolution configuration
#[derive(Debug, Clone)]
struct GraphQLResolutionConfig {
    entity_key_fields: Vec<String>,
    merge_strategy: MergeStrategy,
}

/// Service entity resolution configuration
#[derive(Debug, Clone)]
struct ServiceResolutionConfig {
    entity_key_fields: Vec<String>,
    merge_strategy: MergeStrategy,
}

/// Entity merge strategies
#[derive(Debug, Clone)]
enum MergeStrategy {
    #[allow(dead_code)]
    PreferFirst,
    #[allow(dead_code)]
    PreferLast,
    PreferNonNull,
    #[allow(dead_code)]
    Concatenate,
}

/// Extract GraphQL resolution configuration
fn extract_graphql_resolution_config(query_fragment: &str) -> GraphQLResolutionConfig {
    let entity_key_fields = extract_entity_resolution_keys(query_fragment);

    GraphQLResolutionConfig {
        entity_key_fields,
        merge_strategy: MergeStrategy::PreferNonNull,
    }
}

/// Extract service resolution configuration
fn extract_service_resolution_config(query_fragment: &str) -> ServiceResolutionConfig {
    let entity_key_fields = extract_entity_resolution_keys(query_fragment);

    ServiceResolutionConfig {
        entity_key_fields,
        merge_strategy: MergeStrategy::PreferNonNull,
    }
}

/// Resolve entities in GraphQL data
fn resolve_graphql_entities(
    data: &serde_json::Value,
    config: &GraphQLResolutionConfig,
) -> Result<serde_json::Value> {
    match data {
        serde_json::Value::Array(arr) => {
            let resolved_items =
                resolve_entity_array(arr, &config.entity_key_fields, &config.merge_strategy)?;
            Ok(serde_json::Value::Array(resolved_items))
        }
        serde_json::Value::Object(obj) => {
            let mut resolved_obj = obj.clone();

            // Recursively resolve nested arrays and objects
            for (key, value) in obj {
                let resolved_value = resolve_graphql_entities(value, config)?;
                resolved_obj.insert(key.clone(), resolved_value);
            }

            Ok(serde_json::Value::Object(resolved_obj))
        }
        _ => Ok(data.clone()),
    }
}

/// Resolve entities in service data
fn resolve_service_entities(
    data: &serde_json::Value,
    config: &ServiceResolutionConfig,
) -> Result<serde_json::Value> {
    match data {
        serde_json::Value::Array(arr) => {
            let resolved_items =
                resolve_entity_array(arr, &config.entity_key_fields, &config.merge_strategy)?;
            Ok(serde_json::Value::Array(resolved_items))
        }
        serde_json::Value::Object(obj) => {
            let mut resolved_obj = obj.clone();

            // Recursively resolve nested arrays and objects
            for (key, value) in obj {
                let resolved_value = resolve_service_entities(value, config)?;
                resolved_obj.insert(key.clone(), resolved_value);
            }

            Ok(serde_json::Value::Object(resolved_obj))
        }
        _ => Ok(data.clone()),
    }
}

/// Resolve entities in an array by grouping and merging
fn resolve_entity_array(
    arr: &[serde_json::Value],
    key_fields: &[String],
    merge_strategy: &MergeStrategy,
) -> Result<Vec<serde_json::Value>> {
    if key_fields.is_empty() {
        return Ok(arr.to_vec());
    }

    let mut entity_groups: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

    for item in arr {
        let entity_key = compute_json_entity_key(item, key_fields);
        entity_groups
            .entry(entity_key)
            .or_default()
            .push(item.clone());
    }

    let mut resolved_entities = Vec::new();
    for (_, group) in entity_groups {
        if group.len() == 1 {
            resolved_entities.push(group.into_iter().next().unwrap());
        } else {
            let merged_entity = merge_json_entities(&group, merge_strategy)?;
            resolved_entities.push(merged_entity);
        }
    }

    Ok(resolved_entities)
}

/// Compute entity key from JSON object
fn compute_json_entity_key(item: &serde_json::Value, key_fields: &[String]) -> String {
    let mut key_parts = Vec::new();

    if let serde_json::Value::Object(obj) = item {
        for field in key_fields {
            if let Some(value) = obj.get(field) {
                key_parts.push(format!("{field}:{value}"));
            } else {
                key_parts.push(format!("{field}:null"));
            }
        }
    }

    key_parts.join("|")
}

/// Merge multiple JSON entities
fn merge_json_entities(
    entities: &[serde_json::Value],
    merge_strategy: &MergeStrategy,
) -> Result<serde_json::Value> {
    if entities.is_empty() {
        return Ok(serde_json::Value::Null);
    }

    if entities.len() == 1 {
        return Ok(entities[0].clone());
    }

    let mut merged = serde_json::Map::new();

    // Collect all keys from all entities
    let mut all_keys = HashSet::new();
    for entity in entities {
        if let serde_json::Value::Object(obj) = entity {
            for key in obj.keys() {
                all_keys.insert(key.clone());
            }
        }
    }

    // Merge each field according to strategy
    for key in all_keys {
        let values: Vec<&serde_json::Value> = entities
            .iter()
            .filter_map(|e| e.as_object().and_then(|obj| obj.get(&key)))
            .collect();

        if values.is_empty() {
            continue;
        }

        let merged_value = match merge_strategy {
            MergeStrategy::PreferFirst => values[0].clone(),
            MergeStrategy::PreferLast => values[values.len() - 1].clone(),
            MergeStrategy::PreferNonNull => {
                (*values.iter().find(|v| !v.is_null()).unwrap_or(&values[0])).clone()
            }
            MergeStrategy::Concatenate => {
                if values.iter().all(|v| v.is_string()) {
                    let concatenated = values
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    serde_json::Value::String(concatenated)
                } else if values.iter().all(|v| v.is_array()) {
                    let mut concatenated = Vec::new();
                    for value in values {
                        if let Some(arr) = value.as_array() {
                            concatenated.extend(arr.clone());
                        }
                    }
                    serde_json::Value::Array(concatenated)
                } else {
                    values[0].clone()
                }
            }
        };

        merged.insert(key, merged_value);
    }

    Ok(serde_json::Value::Object(merged))
}
