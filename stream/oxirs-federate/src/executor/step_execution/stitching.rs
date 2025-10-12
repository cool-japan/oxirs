//! Result stitching functions for federated queries

use super::super::types::*;
use super::joins::{merge_json_values, perform_graphql_join};
use super::result_processing::perform_sparql_join;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::debug;

pub fn perform_intelligent_result_stitching(
    results: &[QueryResultData],
    query_fragment: &str,
) -> Result<QueryResultData> {
    debug!(
        "Performing intelligent result stitching on {} results",
        results.len()
    );

    if results.is_empty() {
        return Err(anyhow!("No results to stitch"));
    }

    if results.len() == 1 {
        return Ok(results[0].clone());
    }

    // Determine stitching strategy based on result types
    let stitching_strategy = determine_stitching_strategy(results, query_fragment);

    match stitching_strategy {
        StitchingStrategy::SparqlUnion => stitch_sparql_results_union(results),
        StitchingStrategy::SparqlJoin => stitch_sparql_results_join(results),
        StitchingStrategy::GraphQLMerge => stitch_graphql_results_merge(results),
        StitchingStrategy::GraphQLNested => stitch_graphql_results_nested(results, query_fragment),
        StitchingStrategy::ServiceMerge => stitch_service_results_merge(results),
        StitchingStrategy::HeterogeneousStitch => {
            stitch_heterogeneous_results(results, query_fragment)
        }
    }
}

/// Result stitching strategies
#[derive(Debug, Clone, PartialEq)]
enum StitchingStrategy {
    SparqlUnion,         // Union SPARQL results
    SparqlJoin,          // Join SPARQL results on common variables
    GraphQLMerge,        // Merge GraphQL responses by combining data fields
    GraphQLNested,       // Create nested GraphQL structure
    ServiceMerge,        // Merge service results as JSON
    HeterogeneousStitch, // Stitch different result types together
}

/// Determine the best stitching strategy based on result types and query
fn determine_stitching_strategy(
    results: &[QueryResultData],
    query_fragment: &str,
) -> StitchingStrategy {
    // Count result types
    let sparql_count = results
        .iter()
        .filter(|r| matches!(r, QueryResultData::Sparql(_)))
        .count();
    let graphql_count = results
        .iter()
        .filter(|r| matches!(r, QueryResultData::GraphQL(_)))
        .count();
    let service_count = results
        .iter()
        .filter(|r| matches!(r, QueryResultData::ServiceResult(_)))
        .count();

    // Determine strategy based on result type homogeneity
    if sparql_count == results.len() {
        // All SPARQL results
        if query_fragment.contains("UNION") || query_fragment.contains("union") {
            StitchingStrategy::SparqlUnion
        } else if query_fragment.contains("JOIN") || query_fragment.contains("join") {
            StitchingStrategy::SparqlJoin
        } else {
            StitchingStrategy::SparqlUnion // Default to union for SPARQL
        }
    } else if graphql_count == results.len() {
        // All GraphQL results
        if query_fragment.contains("nested") || query_fragment.contains("fragment") {
            StitchingStrategy::GraphQLNested
        } else {
            StitchingStrategy::GraphQLMerge
        }
    } else if service_count == results.len() {
        // All service results
        StitchingStrategy::ServiceMerge
    } else {
        // Mixed result types
        StitchingStrategy::HeterogeneousStitch
    }
}

/// Stitch SPARQL results using UNION strategy
fn stitch_sparql_results_union(results: &[QueryResultData]) -> Result<QueryResultData> {
    debug!("Stitching SPARQL results using UNION strategy");

    let mut all_bindings = Vec::new();
    let mut all_vars = Vec::new();

    for result in results {
        if let QueryResultData::Sparql(sparql_result) = result {
            // Collect variables (use the union of all variables)
            for var in &sparql_result.head.vars {
                if !all_vars.contains(var) {
                    all_vars.push(var.clone());
                }
            }

            // Add all bindings
            all_bindings.extend(sparql_result.results.bindings.clone());
        }
    }

    let stitched_result = SparqlResults {
        head: SparqlHead { vars: all_vars },
        results: SparqlResultsData {
            bindings: all_bindings,
        },
    };

    Ok(QueryResultData::Sparql(stitched_result))
}

/// Stitch SPARQL results using JOIN strategy
fn stitch_sparql_results_join(results: &[QueryResultData]) -> Result<QueryResultData> {
    debug!("Stitching SPARQL results using JOIN strategy");

    if results.len() < 2 {
        return Ok(results[0].clone());
    }

    // Start with the first result and join with subsequent results
    let mut current_result = match &results[0] {
        QueryResultData::Sparql(sparql_result) => sparql_result.clone(),
        _ => return Err(anyhow!("Expected SPARQL result for JOIN stitching")),
    };

    for result in results.iter().skip(1) {
        if let QueryResultData::Sparql(sparql_result) = result {
            current_result = perform_sparql_join(&current_result, sparql_result)?;
        }
    }

    Ok(QueryResultData::Sparql(current_result))
}

/// Stitch GraphQL results using MERGE strategy
fn stitch_graphql_results_merge(results: &[QueryResultData]) -> Result<QueryResultData> {
    debug!("Stitching GraphQL results using MERGE strategy");

    let mut current_response = match &results[0] {
        QueryResultData::GraphQL(graphql_response) => graphql_response.clone(),
        _ => return Err(anyhow!("Expected GraphQL result for MERGE stitching")),
    };

    for result in results.iter().skip(1) {
        if let QueryResultData::GraphQL(graphql_response) = result {
            current_response = perform_graphql_join(&current_response, graphql_response)?;
        }
    }

    Ok(QueryResultData::GraphQL(current_response))
}

/// Stitch GraphQL results using NESTED strategy
fn stitch_graphql_results_nested(
    results: &[QueryResultData],
    query_fragment: &str,
) -> Result<QueryResultData> {
    debug!("Stitching GraphQL results using NESTED strategy");

    let mut nested_data = serde_json::Map::new();
    let mut combined_errors = Vec::new();
    let mut combined_extensions = serde_json::Map::new();

    // Create nested structure based on query fragment
    let nested_fields = extract_nested_field_names(query_fragment);

    for (i, result) in results.iter().enumerate() {
        if let QueryResultData::GraphQL(graphql_response) = result {
            // Use extracted field name or generate one
            let field_name = nested_fields
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("result{i}"));

            nested_data.insert(field_name, graphql_response.data.clone());
            combined_errors.extend(graphql_response.errors.clone());

            if let Some(serde_json::Value::Object(ext_obj)) = &graphql_response.extensions {
                for (key, value) in ext_obj {
                    combined_extensions.insert(key.clone(), value.clone());
                }
            }
        }
    }

    let stitched_response = GraphQLResponse {
        data: serde_json::Value::Object(nested_data),
        errors: combined_errors,
        extensions: if combined_extensions.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(combined_extensions))
        },
    };

    Ok(QueryResultData::GraphQL(stitched_response))
}

/// Stitch service results using MERGE strategy
fn stitch_service_results_merge(results: &[QueryResultData]) -> Result<QueryResultData> {
    debug!("Stitching service results using MERGE strategy");

    let mut merged_result = serde_json::Value::Object(serde_json::Map::new());

    for result in results {
        if let QueryResultData::ServiceResult(service_result) = result {
            merged_result = merge_json_values(&merged_result, service_result)?;
        }
    }

    Ok(QueryResultData::ServiceResult(merged_result))
}

/// Stitch heterogeneous results (different types)
fn stitch_heterogeneous_results(
    results: &[QueryResultData],
    query_fragment: &str,
) -> Result<QueryResultData> {
    debug!("Stitching heterogeneous results");

    // Convert all results to a common JSON format for stitching
    let mut converted_results = Vec::new();

    for result in results {
        let json_result = convert_result_to_json(result)?;
        converted_results.push(json_result);
    }

    // Merge all JSON results
    let mut merged = serde_json::Value::Object(serde_json::Map::new());
    for (i, json_result) in converted_results.iter().enumerate() {
        let field_name = format!("result{i}");
        if let serde_json::Value::Object(ref mut obj) = merged {
            obj.insert(field_name, json_result.clone());
        }
    }

    // Determine the best result type to return based on query fragment
    if query_fragment.contains("SELECT") || query_fragment.contains("sparql") {
        // Convert back to SPARQL if possible
        convert_json_to_sparql_result(&merged)
    } else if query_fragment.contains("query") || query_fragment.contains("mutation") {
        // Return as GraphQL
        Ok(QueryResultData::GraphQL(GraphQLResponse {
            data: merged,
            errors: Vec::new(),
            extensions: None,
        }))
    } else {
        // Return as service result
        Ok(QueryResultData::ServiceResult(merged))
    }
}

/// Convert any result type to JSON for heterogeneous stitching
fn convert_result_to_json(result: &QueryResultData) -> Result<serde_json::Value> {
    match result {
        QueryResultData::Sparql(sparql_result) => Ok(serde_json::to_value(sparql_result)?),
        QueryResultData::GraphQL(graphql_response) => Ok(serde_json::to_value(graphql_response)?),
        QueryResultData::ServiceResult(service_result) => Ok(service_result.clone()),
    }
}

/// Convert JSON back to SPARQL result format
fn convert_json_to_sparql_result(json: &serde_json::Value) -> Result<QueryResultData> {
    // Basic conversion - in practice this would be more sophisticated
    let vars = vec!["value".to_string()];
    let binding = {
        let mut map = HashMap::new();
        map.insert(
            "value".to_string(),
            SparqlValue {
                value_type: "literal".to_string(),
                value: json.to_string(),
                datatype: None,
                lang: None,
                quoted_triple: None,
            },
        );
        map
    };

    let sparql_result = SparqlResults {
        head: SparqlHead { vars },
        results: SparqlResultsData {
            bindings: vec![binding],
        },
    };

    Ok(QueryResultData::Sparql(sparql_result))
}

/// Extract nested field names from query fragment
fn extract_nested_field_names(query_fragment: &str) -> Vec<String> {
    let mut field_names = Vec::new();

    // Look for field specifications in the query fragment
    // This is a simplified parser - would be more sophisticated in practice
    if let Some(start) = query_fragment.find("fields:") {
        let fields_section = &query_fragment[start + 7..];
        if let Some(end) = fields_section.find(';') {
            let fields_str = &fields_section[..end];
            for field in fields_str.split(',') {
                let field = field.trim();
                if !field.is_empty() {
                    field_names.push(field.to_string());
                }
            }
        }
    }

    field_names
}
