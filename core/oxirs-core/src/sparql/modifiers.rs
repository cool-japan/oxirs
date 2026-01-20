//! SPARQL query modifiers: ORDER BY, DISTINCT, LIMIT, OFFSET

use crate::model::Term;
use crate::rdf_store::VariableBinding;
use crate::Result;
use std::collections::HashSet;

/// ORDER BY specification
#[derive(Debug, Clone)]
pub struct OrderBy {
    pub variable: String,
    pub descending: bool,
}

/// Extract LIMIT value from SPARQL query
pub fn extract_limit(sparql: &str) -> Result<Option<usize>> {
    let sparql_upper = sparql.to_uppercase();

    if let Some(limit_start) = sparql_upper.find("LIMIT") {
        let after_limit = &sparql[limit_start + 5..];

        // Find the first token after LIMIT
        for token in after_limit.split_whitespace() {
            if let Ok(limit_value) = token.parse::<usize>() {
                return Ok(Some(limit_value));
            }
        }
    }

    Ok(None)
}

/// Extract OFFSET value from SPARQL query
pub fn extract_offset(sparql: &str) -> Result<usize> {
    let sparql_upper = sparql.to_uppercase();

    if let Some(offset_start) = sparql_upper.find("OFFSET") {
        let after_offset = &sparql[offset_start + 6..];

        // Find the first token after OFFSET
        for token in after_offset.split_whitespace() {
            if let Ok(offset_value) = token.parse::<usize>() {
                return Ok(offset_value);
            }
        }
    }

    Ok(0)
}

/// Extract ORDER BY clause from SPARQL query
pub fn extract_order_by(sparql: &str) -> Result<Option<OrderBy>> {
    let sparql_upper = sparql.to_uppercase();

    if let Some(order_start) = sparql_upper.find("ORDER BY") {
        let after_order = &sparql[order_start + 8..];

        // Check for DESC or ASC
        let mut descending = false;
        let tokens: Vec<&str> = after_order.split_whitespace().collect();

        if tokens.is_empty() {
            return Ok(None);
        }

        let mut var_token = tokens[0];

        // Check for DESC/ASC modifier
        if tokens.len() > 1 {
            let modifier = tokens[1].to_uppercase();
            if modifier == "DESC" {
                descending = true;
            } else if modifier == "ASC" {
                descending = false;
            }
        } else if var_token.to_uppercase().ends_with("DESC") {
            // Handle DESC attached to variable
            var_token = var_token.trim_end_matches("DESC").trim_end_matches("desc");
            descending = true;
        } else if var_token.to_uppercase().ends_with("ASC") {
            // Handle ASC attached to variable
            var_token = var_token.trim_end_matches("ASC").trim_end_matches("asc");
        }

        // Check if DESC() or ASC() function
        if var_token.to_uppercase().starts_with("DESC(") {
            descending = true;
            var_token = var_token[5..].trim_end_matches(')');
        } else if var_token.to_uppercase().starts_with("ASC(") {
            var_token = var_token[4..].trim_end_matches(')');
        }

        // Extract variable name
        let variable = var_token.trim_start_matches('?').trim().to_string();

        if !variable.is_empty() {
            return Ok(Some(OrderBy {
                variable,
                descending,
            }));
        }
    }

    Ok(None)
}

/// Remove duplicate bindings for DISTINCT
pub fn remove_duplicate_bindings(
    results: Vec<VariableBinding>,
    variables: &[String],
) -> Vec<VariableBinding> {
    let mut seen = HashSet::new();
    let mut unique_results = Vec::new();

    for binding in results {
        // Create a signature for this binding based on the selected variables
        let mut signature = String::new();
        for var in variables {
            if let Some(term) = binding.get(var) {
                signature.push_str(&format!("{:?}|", term));
            } else {
                signature.push_str("UNBOUND|");
            }
        }

        if seen.insert(signature) {
            unique_results.push(binding);
        }
    }

    unique_results
}

/// Sort results according to ORDER BY specification
pub fn sort_results(results: &mut [VariableBinding], order_by: &OrderBy) {
    results.sort_by(|a, b| {
        let a_val = a.get(&order_by.variable);
        let b_val = b.get(&order_by.variable);

        let cmp = match (a_val, b_val) {
            (Some(a_term), Some(b_term)) => compare_terms(a_term, b_term),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };

        if order_by.descending {
            cmp.reverse()
        } else {
            cmp
        }
    });
}

/// Compare two terms for ordering
pub fn compare_terms(a: &Term, b: &Term) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (a, b) {
        (Term::Literal(a_lit), Term::Literal(b_lit)) => {
            let a_val = a_lit.value();
            let b_val = b_lit.value();

            // Try numeric comparison first
            if let (Ok(a_num), Ok(b_num)) = (a_val.parse::<f64>(), b_val.parse::<f64>()) {
                if a_num < b_num {
                    Ordering::Less
                } else if a_num > b_num {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            } else {
                // Lexicographic comparison
                a_val.cmp(b_val)
            }
        }
        (Term::NamedNode(a_node), Term::NamedNode(b_node)) => a_node.as_str().cmp(b_node.as_str()),
        (Term::BlankNode(a_bnode), Term::BlankNode(b_bnode)) => {
            a_bnode.as_str().cmp(b_bnode.as_str())
        }
        // Mixed types: Literals < URIs < BlankNodes
        (Term::Literal(_), _) => Ordering::Less,
        (_, Term::Literal(_)) => Ordering::Greater,
        (Term::NamedNode(_), Term::BlankNode(_)) => Ordering::Less,
        (Term::BlankNode(_), Term::NamedNode(_)) => Ordering::Greater,
        _ => Ordering::Equal,
    }
}
