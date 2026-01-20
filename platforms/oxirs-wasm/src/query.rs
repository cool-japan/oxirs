//! SPARQL query execution for WASM

use crate::error::{WasmError, WasmResult};
use crate::store::OxiRSStore;
use crate::Triple;
use std::collections::HashMap;

/// Execute a SPARQL SELECT query
pub fn execute_select(
    sparql: &str,
    store: &OxiRSStore,
) -> WasmResult<Vec<HashMap<String, String>>> {
    let query = parse_select_query(sparql)?;
    evaluate_select(&query, store)
}

/// Execute a SPARQL ASK query
pub fn execute_ask(sparql: &str, store: &OxiRSStore) -> WasmResult<bool> {
    let query = parse_ask_query(sparql)?;
    evaluate_ask(&query, store)
}

/// Execute a SPARQL CONSTRUCT query
pub fn execute_construct(sparql: &str, store: &OxiRSStore) -> WasmResult<Vec<Triple>> {
    // Simplified CONSTRUCT - just return matching triples
    let query = parse_select_query(sparql)?;
    let bindings = evaluate_select(&query, store)?;

    let mut triples = Vec::new();
    for binding in bindings {
        if let (Some(s), Some(p), Some(o)) = (
            binding.get("s").or(binding.get("subject")),
            binding.get("p").or(binding.get("predicate")),
            binding.get("o").or(binding.get("object")),
        ) {
            triples.push(Triple::new(s, p, o));
        }
    }

    Ok(triples)
}

/// Parsed SELECT query
struct SelectQuery {
    variables: Vec<String>,
    patterns: Vec<TriplePattern>,
    filters: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

/// Triple pattern
#[derive(Debug, Clone)]
struct TriplePattern {
    subject: PatternTerm,
    predicate: PatternTerm,
    object: PatternTerm,
}

/// Pattern term (variable or value)
#[derive(Debug, Clone)]
enum PatternTerm {
    Variable(String),
    Value(String),
}

impl PatternTerm {
    fn is_variable(&self) -> bool {
        matches!(self, PatternTerm::Variable(_))
    }

    fn variable_name(&self) -> Option<&str> {
        match self {
            PatternTerm::Variable(name) => Some(name),
            PatternTerm::Value(_) => None,
        }
    }

    fn matches(&self, value: &str, bindings: &HashMap<String, String>) -> bool {
        match self {
            PatternTerm::Variable(name) => {
                if let Some(bound_value) = bindings.get(name) {
                    bound_value == value
                } else {
                    true // Unbound variable matches anything
                }
            }
            PatternTerm::Value(v) => v == value,
        }
    }
}

/// Parse a SELECT query (simplified)
fn parse_select_query(sparql: &str) -> WasmResult<SelectQuery> {
    let sparql = sparql.trim();

    // Extract variables from SELECT clause
    let select_start = sparql
        .to_uppercase()
        .find("SELECT")
        .ok_or_else(|| WasmError::QueryError("No SELECT clause".to_string()))?;

    let where_start = sparql
        .to_uppercase()
        .find("WHERE")
        .ok_or_else(|| WasmError::QueryError("No WHERE clause".to_string()))?;

    let select_clause = &sparql[select_start + 6..where_start].trim();
    let variables = if *select_clause == "*" {
        vec![] // Will collect all variables from patterns
    } else {
        select_clause
            .split_whitespace()
            .filter(|s| s.starts_with('?') || s.starts_with('$'))
            .map(|s| s.trim_start_matches(['?', '$']).to_string())
            .collect()
    };

    // Find WHERE clause body
    let where_body_start = sparql[where_start..]
        .find('{')
        .ok_or_else(|| WasmError::QueryError("No WHERE body".to_string()))?
        + where_start
        + 1;

    let where_body_end = sparql
        .rfind('}')
        .ok_or_else(|| WasmError::QueryError("No closing brace".to_string()))?;

    let where_body = &sparql[where_body_start..where_body_end];

    // Parse triple patterns
    let patterns = parse_where_body(where_body)?;

    // Parse LIMIT and OFFSET
    let limit = parse_modifier(sparql, "LIMIT");
    let offset = parse_modifier(sparql, "OFFSET");

    Ok(SelectQuery {
        variables,
        patterns,
        filters: vec![],
        limit,
        offset,
    })
}

/// Parse WHERE body into triple patterns
fn parse_where_body(body: &str) -> WasmResult<Vec<TriplePattern>> {
    let mut patterns = Vec::new();

    // Split by '.' to get individual patterns
    for statement in body.split('.') {
        let statement = statement.trim();
        if statement.is_empty() {
            continue;
        }

        // Skip FILTER clauses for now
        if statement.to_uppercase().starts_with("FILTER") {
            continue;
        }

        // Split into subject, predicate, object
        let tokens: Vec<&str> = statement.split_whitespace().collect();

        if tokens.len() >= 3 {
            let subject = parse_pattern_term(tokens[0]);
            let predicate = if tokens[1] == "a" {
                PatternTerm::Value("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
            } else {
                parse_pattern_term(tokens[1])
            };
            let object = parse_pattern_term(tokens[2]);

            patterns.push(TriplePattern {
                subject,
                predicate,
                object,
            });
        }
    }

    Ok(patterns)
}

/// Parse a pattern term
fn parse_pattern_term(term: &str) -> PatternTerm {
    if term.starts_with('?') || term.starts_with('$') {
        PatternTerm::Variable(term.trim_start_matches(['?', '$']).to_string())
    } else if term.starts_with('<') && term.ends_with('>') {
        PatternTerm::Value(term[1..term.len() - 1].to_string())
    } else if term.starts_with('"') {
        // Literal - keep as is
        PatternTerm::Value(term.to_string())
    } else {
        PatternTerm::Value(term.to_string())
    }
}

/// Parse modifier like LIMIT or OFFSET
fn parse_modifier(sparql: &str, modifier: &str) -> Option<usize> {
    let upper = sparql.to_uppercase();
    let idx = upper.find(modifier)?;
    let rest = &sparql[idx + modifier.len()..];
    let num_str: String = rest
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}

/// Parse ASK query
fn parse_ask_query(sparql: &str) -> WasmResult<SelectQuery> {
    // Convert ASK { ... } to SELECT * WHERE { ... }
    let modified = sparql.replace("ASK", "SELECT * WHERE");
    parse_select_query(&modified)
}

/// Evaluate SELECT query
fn evaluate_select(
    query: &SelectQuery,
    store: &OxiRSStore,
) -> WasmResult<Vec<HashMap<String, String>>> {
    // Start with empty bindings
    let mut results: Vec<HashMap<String, String>> = vec![HashMap::new()];

    // Apply each pattern as a join
    for pattern in &query.patterns {
        let mut new_results = Vec::new();

        for binding in &results {
            // Find matching triples
            for triple in store.all_triples() {
                if pattern.subject.matches(&triple.subject, binding)
                    && pattern.predicate.matches(&triple.predicate, binding)
                    && pattern.object.matches(&triple.object, binding)
                {
                    let mut new_binding = binding.clone();

                    // Bind variables
                    if let PatternTerm::Variable(name) = &pattern.subject {
                        new_binding.insert(name.clone(), triple.subject.clone());
                    }
                    if let PatternTerm::Variable(name) = &pattern.predicate {
                        new_binding.insert(name.clone(), triple.predicate.clone());
                    }
                    if let PatternTerm::Variable(name) = &pattern.object {
                        new_binding.insert(name.clone(), triple.object.clone());
                    }

                    new_results.push(new_binding);
                }
            }
        }

        results = new_results;
    }

    // Apply OFFSET and LIMIT
    let mut final_results = results;

    if let Some(offset) = query.offset {
        if offset < final_results.len() {
            final_results = final_results.into_iter().skip(offset).collect();
        } else {
            final_results.clear();
        }
    }

    if let Some(limit) = query.limit {
        final_results.truncate(limit);
    }

    // Project to requested variables
    if !query.variables.is_empty() {
        final_results = final_results
            .into_iter()
            .map(|binding| {
                let mut projected = HashMap::new();
                for var in &query.variables {
                    if let Some(value) = binding.get(var) {
                        projected.insert(var.clone(), value.clone());
                    }
                }
                projected
            })
            .collect();
    }

    Ok(final_results)
}

/// Evaluate ASK query
fn evaluate_ask(query: &SelectQuery, store: &OxiRSStore) -> WasmResult<bool> {
    let results = evaluate_select(query, store)?;
    Ok(!results.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select() {
        let sparql = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let query = parse_select_query(sparql).unwrap();

        assert_eq!(query.variables.len(), 3);
        assert_eq!(query.patterns.len(), 1);
    }

    #[test]
    fn test_evaluate_select() {
        let mut store = OxiRSStore::new();
        store.insert("http://a", "http://b", "http://c");

        let sparql = "SELECT ?s ?o WHERE { ?s <http://b> ?o }";
        let results = execute_select(sparql, &store).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("s").unwrap(), "http://a");
        assert_eq!(results[0].get("o").unwrap(), "http://c");
    }

    #[test]
    fn test_evaluate_ask() {
        let mut store = OxiRSStore::new();
        store.insert("http://a", "http://b", "http://c");

        assert!(execute_ask("ASK { <http://a> <http://b> <http://c> }", &store).unwrap());
        assert!(!execute_ask("ASK { <http://x> <http://y> <http://z> }", &store).unwrap());
    }
}
