//! Sorting and comparison functions for SPARQL results

use anyhow::Result;
use tracing::warn;
use super::super::types::*;
use super::aggregation::{
    perform_avg_aggregation,
    perform_count_aggregation,
    perform_group_by_aggregation,
    perform_sum_aggregation,
    perform_min_aggregation,
    perform_max_aggregation,
};

pub fn aggregate_sparql_results(
    results: &SparqlResults,
    aggregate_expr: &str,
) -> Result<SparqlResults> {
    let aggregate_expr = aggregate_expr.trim();

    // Parse the aggregate expression to identify the operation
    if aggregate_expr.contains("GROUP BY") {
        perform_group_by_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("COUNT") {
        perform_count_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("SUM") {
        perform_sum_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("AVG") {
        perform_avg_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("MIN") {
        perform_min_aggregation(results, aggregate_expr)
    } else if aggregate_expr.contains("MAX") {
        perform_max_aggregation(results, aggregate_expr)
    } else {
        warn!("Unknown aggregation type: {}", aggregate_expr);
        Ok(results.clone())
    }
}

/// Sort SPARQL results based on ORDER BY expression
pub fn sort_sparql_results(results: &SparqlResults, order_expr: &str) -> Result<SparqlResults> {
    let order_clauses = parse_order_by_expression(order_expr);

    let mut sorted_bindings = results.results.bindings.clone();

    // Sort the bindings based on the ORDER BY clauses
    sorted_bindings.sort_by(|a, b| {
        for order_clause in &order_clauses {
            let comparison = compare_bindings(a, b, &order_clause.variable);

            let result = if order_clause.descending {
                comparison.reverse()
            } else {
                comparison
            };

            if result != std::cmp::Ordering::Equal {
                return result;
            }
        }
        std::cmp::Ordering::Equal
    });

    Ok(SparqlResults {
        head: results.head.clone(),
        results: SparqlResultsData {
            bindings: sorted_bindings,
        },
    })
}

// Implementation of all the helper functions for aggregation and sorting...
// (This would include all the helper functions from the original file)

/// Order clause for sorting
#[derive(Debug, Clone)]
pub struct OrderClause {
    pub variable: String,
    pub descending: bool,
}

/// Parse ORDER BY expression into order clauses
pub fn parse_order_by_expression(expr: &str) -> Vec<OrderClause> {
    let mut clauses = Vec::new();

    if let Some(order_start) = expr.find("ORDER BY") {
        let order_part = &expr[order_start + 8..];

        // Split by comma and parse each order expression
        for order_expr in order_part.split(',') {
            let order_expr = order_expr.trim();

            let (variable, descending) = if order_expr.starts_with("DESC(") {
                // DESC(?variable)
                if let Some(start) = order_expr.find('(') {
                    if let Some(end) = order_expr.find(')') {
                        let var_part = &order_expr[start + 1..end];
                        (var_part.trim_start_matches('?').to_string(), true)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else if order_expr.starts_with("ASC(") {
                // ASC(?variable)
                if let Some(start) = order_expr.find('(') {
                    if let Some(end) = order_expr.find(')') {
                        let var_part = &order_expr[start + 1..end];
                        (var_part.trim_start_matches('?').to_string(), false)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else if order_expr.starts_with('?') {
                // Simple variable (defaults to ASC)
                (order_expr.trim_start_matches('?').to_string(), false)
            } else {
                // Try to extract variable from complex expressions
                if let Some(var_match) = order_expr.split_whitespace().find(|s| s.starts_with('?'))
                {
                    let descending = order_expr.to_lowercase().contains("desc");
                    (var_match.trim_start_matches('?').to_string(), descending)
                } else {
                    continue;
                }
            };

            clauses.push(OrderClause {
                variable,
                descending,
            });
        }
    }

    clauses
}

/// Compare two bindings for a specific variable
pub fn compare_bindings(
    a: &SparqlBinding,
    b: &SparqlBinding,
    variable: &str,
) -> std::cmp::Ordering {
    let a_value = a.get(variable);
    let b_value = b.get(variable);

    match (a_value, b_value) {
        (Some(a_val), Some(b_val)) => compare_sparql_values(a_val, b_val),
        (Some(_), None) => std::cmp::Ordering::Greater, // Non-null values come after null
        (None, Some(_)) => std::cmp::Ordering::Less,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// Compare two SPARQL values with type-aware comparison
pub fn compare_sparql_values(a: &SparqlValue, b: &SparqlValue) -> std::cmp::Ordering {
    // Compare by type first (URIs < literals < blank nodes)
    let type_order = |value_type: &str| match value_type {
        "uri" => 0,
        "literal" => 1,
        "bnode" => 2,
        _ => 3,
    };

    let a_type_order = type_order(&a.value_type);
    let b_type_order = type_order(&b.value_type);

    match a_type_order.cmp(&b_type_order) {
        std::cmp::Ordering::Equal => {
            // Same type, compare values
            match a.value_type.as_str() {
                "literal" => compare_literal_values(a, b),
                _ => a.value.cmp(&b.value), // String comparison for URIs and blank nodes
            }
        }
        other => other,
    }
}

/// Compare literal values with datatype-aware comparison
pub fn compare_literal_values(a: &SparqlValue, b: &SparqlValue) -> std::cmp::Ordering {
    // If both have the same datatype, try type-specific comparison
    if let (Some(a_dt), Some(b_dt)) = (&a.datatype, &b.datatype) {
        if a_dt == b_dt {
            return compare_typed_literals(a, b, a_dt);
        }
    }

    // Fall back to string comparison
    a.value.cmp(&b.value)
}

/// Compare typed literal values
pub fn compare_typed_literals(
    a: &SparqlValue,
    b: &SparqlValue,
    datatype: &str,
) -> std::cmp::Ordering {
    match datatype {
        "http://www.w3.org/2001/XMLSchema#integer"
        | "http://www.w3.org/2001/XMLSchema#int"
        | "http://www.w3.org/2001/XMLSchema#long" => {
            match (a.value.parse::<i64>(), b.value.parse::<i64>()) {
                (Ok(a_num), Ok(b_num)) => a_num.cmp(&b_num),
                _ => a.value.cmp(&b.value),
            }
        }
        "http://www.w3.org/2001/XMLSchema#decimal"
        | "http://www.w3.org/2001/XMLSchema#double"
        | "http://www.w3.org/2001/XMLSchema#float" => {
            match (a.value.parse::<f64>(), b.value.parse::<f64>()) {
                (Ok(a_num), Ok(b_num)) => a_num
                    .partial_cmp(&b_num)
                    .unwrap_or(std::cmp::Ordering::Equal),
                _ => a.value.cmp(&b.value),
            }
        }
        "http://www.w3.org/2001/XMLSchema#dateTime" | "http://www.w3.org/2001/XMLSchema#date" => {
            // For dates, ISO format string comparison usually works
            a.value.cmp(&b.value)
        }
        "http://www.w3.org/2001/XMLSchema#boolean" => {
            match (a.value.parse::<bool>(), b.value.parse::<bool>()) {
                (Ok(a_bool), Ok(b_bool)) => a_bool.cmp(&b_bool),
                _ => a.value.cmp(&b.value),
            }
        }
        _ => a.value.cmp(&b.value), // Default to string comparison
    }
}

// Placeholder implementations for aggregation functions
// These would need to be properly implemented based on the original file
