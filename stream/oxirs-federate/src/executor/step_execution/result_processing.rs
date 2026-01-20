//! Result aggregation and processing functions

use super::super::types::*;
use anyhow::Result;

pub fn aggregate_graphql_response(
    response: &GraphQLResponse,
    query_fragment: &str,
) -> Result<GraphQLResponse> {
    // Basic aggregation implementation for GraphQL
    // This would typically involve combining fields or calculating aggregates

    let mut aggregated_data = response.data.clone();

    // Look for aggregation operations in the query fragment
    if query_fragment.contains("count") {
        aggregated_data = apply_count_aggregation_to_json(&aggregated_data)?;
    }

    Ok(GraphQLResponse {
        data: aggregated_data,
        errors: response.errors.clone(),
        extensions: response.extensions.clone(),
    })
}

/// Apply count aggregation to JSON data
fn apply_count_aggregation_to_json(data: &serde_json::Value) -> Result<serde_json::Value> {
    // Basic count implementation - count array elements or object keys
    match data {
        serde_json::Value::Array(arr) => {
            let count = serde_json::json!({"count": arr.len()});
            Ok(count)
        }
        serde_json::Value::Object(obj) => {
            let count = serde_json::json!({"count": obj.len()});
            Ok(count)
        }
        _ => Ok(data.clone()),
    }
}

/// Aggregate service result (JSON data)
pub fn aggregate_service_result(
    service_result: &serde_json::Value,
    query_fragment: &str,
) -> Result<serde_json::Value> {
    // Implement aggregation operations on JSON service results

    if query_fragment.contains("COUNT") {
        return apply_count_aggregation_to_json(service_result);
    }

    if query_fragment.contains("SUM") {
        return apply_sum_aggregation_to_json(service_result, query_fragment);
    }

    if query_fragment.contains("AVG") {
        return apply_avg_aggregation_to_json(service_result, query_fragment);
    }

    if query_fragment.contains("MIN") || query_fragment.contains("MAX") {
        return apply_minmax_aggregation_to_json(service_result, query_fragment);
    }

    // No aggregation specified, return as-is
    Ok(service_result.clone())
}

/// Apply SUM aggregation to JSON data
fn apply_sum_aggregation_to_json(
    data: &serde_json::Value,
    _query_fragment: &str,
) -> Result<serde_json::Value> {
    // Extract field name and sum numeric values
    match data {
        serde_json::Value::Array(arr) => {
            let sum: f64 = arr
                .iter()
                .filter_map(|item| {
                    if let serde_json::Value::Object(obj) = item {
                        obj.values()
                            .filter_map(|val| val.as_f64())
                            .sum::<f64>()
                            .into()
                    } else {
                        item.as_f64()
                    }
                })
                .sum();
            Ok(serde_json::json!({"sum": sum}))
        }
        _ => Ok(data.clone()),
    }
}

/// Apply AVG aggregation to JSON data
fn apply_avg_aggregation_to_json(
    data: &serde_json::Value,
    _query_fragment: &str,
) -> Result<serde_json::Value> {
    // Calculate average of numeric values
    match data {
        serde_json::Value::Array(arr) => {
            let values: Vec<f64> = arr
                .iter()
                .filter_map(|item| {
                    if let serde_json::Value::Object(obj) = item {
                        obj.values().filter_map(|val| val.as_f64()).next()
                    } else {
                        item.as_f64()
                    }
                })
                .collect();

            if !values.is_empty() {
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                Ok(serde_json::json!({"avg": avg}))
            } else {
                Ok(serde_json::json!({"avg": 0.0}))
            }
        }
        _ => Ok(data.clone()),
    }
}

/// Apply MIN/MAX aggregation to JSON data
fn apply_minmax_aggregation_to_json(
    data: &serde_json::Value,
    query_fragment: &str,
) -> Result<serde_json::Value> {
    let is_min = query_fragment.contains("MIN");

    match data {
        serde_json::Value::Array(arr) => {
            let values: Vec<f64> = arr
                .iter()
                .filter_map(|item| {
                    if let serde_json::Value::Object(obj) = item {
                        obj.values().filter_map(|val| val.as_f64()).next()
                    } else {
                        item.as_f64()
                    }
                })
                .collect();

            if !values.is_empty() {
                let result = if is_min {
                    values.iter().fold(f64::INFINITY, |a, &b| a.min(b))
                } else {
                    values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                };

                let key = if is_min { "min" } else { "max" };
                Ok(serde_json::json!({key: result}))
            } else {
                let key = if is_min { "min" } else { "max" };
                Ok(serde_json::json!({key: null}))
            }
        }
        _ => Ok(data.clone()),
    }
}

/// Sort service result (JSON data)
pub fn sort_service_result(
    service_result: &serde_json::Value,
    query_fragment: &str,
) -> Result<serde_json::Value> {
    // Implement sorting for JSON service results

    match service_result {
        serde_json::Value::Array(arr) => {
            let mut sorted_arr = arr.clone();

            // Extract sort criteria from query fragment
            let sort_key = extract_sort_key(query_fragment);
            let descending = query_fragment.contains("DESC");

            // Sort the array based on the specified key
            sorted_arr.sort_by(|a, b| {
                let a_val = if let Some(key) = &sort_key {
                    a.get(key).and_then(|v| v.as_str()).unwrap_or("")
                } else {
                    a.as_str().unwrap_or("")
                };

                let b_val = if let Some(key) = &sort_key {
                    b.get(key).and_then(|v| v.as_str()).unwrap_or("")
                } else {
                    b.as_str().unwrap_or("")
                };

                if descending {
                    b_val.cmp(a_val)
                } else {
                    a_val.cmp(b_val)
                }
            });

            Ok(serde_json::Value::Array(sorted_arr))
        }
        _ => Ok(service_result.clone()),
    }
}

/// Extract sort key from query fragment
fn extract_sort_key(query_fragment: &str) -> Option<String> {
    // Simple implementation to extract ORDER BY variable
    if let Some(order_pos) = query_fragment.find("ORDER BY") {
        let after_order = &query_fragment[order_pos + 8..];
        if let Some(var_match) = after_order.split_whitespace().next() {
            let clean_var = var_match.trim_start_matches('?');
            return Some(clean_var.to_string());
        }
    }
    None
}

/// Apply SPARQL filters to results based on FILTER expressions
pub(crate) fn apply_sparql_filters(
    results: &SparqlResults,
    filter_expr: &str,
) -> Result<SparqlResults> {
    // Parse simple FILTER expressions like "FILTER(?price > 100)"
    let filter_expr = filter_expr.trim();

    // Extract filter conditions
    let filter_conditions = extract_filter_conditions(filter_expr);

    let mut filtered_bindings = Vec::new();

    for binding in &results.results.bindings {
        let mut keep_binding = true;

        for condition in &filter_conditions {
            if !evaluate_filter_condition(binding, condition) {
                keep_binding = false;
                break;
            }
        }

        if keep_binding {
            filtered_bindings.push(binding.clone());
        }
    }

    Ok(SparqlResults {
        head: results.head.clone(),
        results: SparqlResultsData {
            bindings: filtered_bindings,
        },
    })
}

/// Extract filter conditions from FILTER expression
fn extract_filter_conditions(filter_expr: &str) -> Vec<FilterCondition> {
    let mut conditions = Vec::new();

    // Simple parsing for basic conditions like "?var > value" or "?var = value"
    if let Some(filter_start) = filter_expr.find("FILTER") {
        let filter_content = &filter_expr[filter_start + 6..].trim();
        if filter_content.starts_with('(') && filter_content.ends_with(')') {
            let inner = &filter_content[1..filter_content.len() - 1];

            // Parse simple comparison expressions
            for op in &[" >= ", " <= ", " > ", " < ", " = ", " != "] {
                if let Some(op_pos) = inner.find(op) {
                    let var_part = inner[..op_pos].trim().trim_start_matches('?');
                    let value_part = inner[op_pos + op.len()..].trim();

                    conditions.push(FilterCondition {
                        variable: var_part.to_string(),
                        operator: op.trim().to_string(),
                        value: value_part.to_string(),
                    });
                    break;
                }
            }
        }
    }

    conditions
}

/// Evaluate a single filter condition against a binding
fn evaluate_filter_condition(binding: &SparqlBinding, condition: &FilterCondition) -> bool {
    if let Some(sparql_value) = binding.get(&condition.variable) {
        let binding_value = &sparql_value.value;

        match condition.operator.as_str() {
            "=" => binding_value == &condition.value,
            "!=" => binding_value != &condition.value,
            ">" => {
                if let (Ok(left), Ok(right)) =
                    (binding_value.parse::<f64>(), condition.value.parse::<f64>())
                {
                    left > right
                } else {
                    binding_value > &condition.value
                }
            }
            "<" => {
                if let (Ok(left), Ok(right)) =
                    (binding_value.parse::<f64>(), condition.value.parse::<f64>())
                {
                    left < right
                } else {
                    binding_value < &condition.value
                }
            }
            ">=" => {
                if let (Ok(left), Ok(right)) =
                    (binding_value.parse::<f64>(), condition.value.parse::<f64>())
                {
                    left >= right
                } else {
                    binding_value >= &condition.value
                }
            }
            "<=" => {
                if let (Ok(left), Ok(right)) =
                    (binding_value.parse::<f64>(), condition.value.parse::<f64>())
                {
                    left <= right
                } else {
                    binding_value <= &condition.value
                }
            }
            _ => true, // Unknown operator, pass through
        }
    } else {
        false // Variable not found in binding
    }
}

/// Perform SPARQL join operation between two result sets
pub(crate) fn perform_sparql_join(
    left: &SparqlResults,
    right: &SparqlResults,
) -> Result<SparqlResults> {
    // Find common variables between the two result sets
    let left_vars: std::collections::HashSet<_> = left.head.vars.iter().collect();
    let right_vars: std::collections::HashSet<_> = right.head.vars.iter().collect();
    let common_vars: Vec<_> = left_vars.intersection(&right_vars).cloned().collect();

    // Combine all variables
    let mut all_vars = left.head.vars.clone();
    for var in &right.head.vars {
        if !all_vars.contains(var) {
            all_vars.push(var.clone());
        }
    }

    let mut joined_bindings = Vec::new();

    // Perform inner join based on common variables
    for left_binding in &left.results.bindings {
        for right_binding in &right.results.bindings {
            let mut is_joinable = true;

            // Check if common variables have compatible values
            for common_var in &common_vars {
                let left_val = left_binding.get(*common_var);
                let right_val = right_binding.get(*common_var);

                match (left_val, right_val) {
                    (Some(left_v), Some(right_v)) => {
                        if left_v.value != right_v.value {
                            is_joinable = false;
                            break;
                        }
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        // Variable exists in one but not the other - still joinable
                    }
                    (None, None) => {
                        // Variable doesn't exist in either - still joinable
                    }
                }
            }

            if is_joinable {
                // Merge the bindings
                let mut merged_binding = left_binding.clone();
                for (var, value) in right_binding {
                    merged_binding.insert(var.clone(), value.clone());
                }
                joined_bindings.push(merged_binding);
            }
        }
    }

    Ok(SparqlResults {
        head: SparqlHead { vars: all_vars },
        results: SparqlResultsData {
            bindings: joined_bindings,
        },
    })
}

/// Simple filter condition structure
#[derive(Debug, Clone)]
struct FilterCondition {
    variable: String,
    operator: String,
    value: String,
}
