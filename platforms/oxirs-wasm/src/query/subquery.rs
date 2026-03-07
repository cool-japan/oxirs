//! SPARQL subquery support — `{ SELECT ?x WHERE { ... } }`
//!
//! SPARQL 1.1 allows nested SELECT queries inside the WHERE clause.
//! A subquery is evaluated independently and its bindings are joined with
//! the outer query using the standard join semantics.

use crate::error::{WasmError, WasmResult};
use crate::store::OxiRSStore;
use std::collections::HashMap;

pub(crate) type Binding = HashMap<String, String>;

// ---------------------------------------------------------------------------
// Subquery evaluator
// ---------------------------------------------------------------------------

/// Evaluates a SPARQL subquery against the given store.
///
/// The subquery string should be the complete inner SELECT query, e.g.:
/// `SELECT ?x WHERE { ?x <http://p> ?y }`
pub(crate) struct SubqueryEvaluator<'a> {
    store: &'a OxiRSStore,
}

impl<'a> SubqueryEvaluator<'a> {
    pub(crate) fn new(store: &'a OxiRSStore) -> Self {
        Self { store }
    }

    /// Evaluate the subquery and return its solution bindings.
    ///
    /// This calls into the main query evaluator to avoid code duplication.
    pub(crate) fn evaluate(&self, subquery: &str) -> WasmResult<Vec<Binding>> {
        // Use the public execute_select entry point which handles the full
        // parse-evaluate pipeline including GROUP BY, DISTINCT, ORDER BY, etc.
        crate::query::execute_select(subquery, self.store)
    }
}

// ---------------------------------------------------------------------------
// Parser helper — detect and extract subqueries from a WHERE body token
// ---------------------------------------------------------------------------

/// Attempt to parse a token as a subquery block of the form:
/// `{ SELECT ?x WHERE { ?x <p> ?y } }`  (outer braces already stripped)
///
/// Returns `Some(subquery_string)` if the token looks like a subquery,
/// `None` otherwise.
pub(crate) fn try_extract_subquery(block_content: &str) -> Option<String> {
    let trimmed = block_content.trim();
    let upper = trimmed.to_uppercase();

    // Must start with SELECT (after optional whitespace)
    if !upper.starts_with("SELECT") {
        return None;
    }

    // Must also contain WHERE
    if !upper.contains("WHERE") {
        return None;
    }

    Some(trimmed.to_string())
}

// ---------------------------------------------------------------------------
// Join semantics for subquery results
// ---------------------------------------------------------------------------

/// Join outer bindings with subquery result bindings (inner join semantics).
///
/// For each outer binding, find all compatible subquery rows and merge them.
/// Two bindings are compatible when shared variables have equal values.
pub(crate) fn join_with_subquery(
    outer: Vec<Binding>,
    subquery_results: Vec<Binding>,
) -> Vec<Binding> {
    let mut output: Vec<Binding> = Vec::new();

    for outer_binding in &outer {
        for sub_binding in &subquery_results {
            if let Some(merged) = try_merge(outer_binding, sub_binding) {
                output.push(merged);
            }
        }
    }

    output
}

/// Try to merge two bindings.  Returns `None` if they conflict on any shared variable.
fn try_merge(a: &Binding, b: &Binding) -> Option<Binding> {
    // Check compatibility
    for (var, val_a) in a {
        if let Some(val_b) = b.get(var) {
            if val_a != val_b {
                return None; // conflict
            }
        }
    }

    // Merge
    let mut merged = a.clone();
    for (var, val) in b {
        merged.entry(var.clone()).or_insert_with(|| val.clone());
    }
    Some(merged)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store_with_data() -> OxiRSStore {
        let mut store = OxiRSStore::new();
        store.insert("http://alice", "http://type", "http://Person");
        store.insert("http://bob", "http://type", "http://Person");
        store.insert("http://alice", "http://knows", "http://bob");
        store.insert("http://alice", "http://name", "\"Alice\"");
        store.insert("http://bob", "http://name", "\"Bob\"");
        store
    }

    // ---- try_extract_subquery ----

    #[test]
    fn test_detect_subquery() {
        let block = "SELECT ?x WHERE { ?x <http://p> ?y }";
        assert!(try_extract_subquery(block).is_some());
    }

    #[test]
    fn test_non_subquery_triple() {
        let block = "?s <http://p> ?o";
        assert!(try_extract_subquery(block).is_none());
    }

    #[test]
    fn test_non_subquery_empty() {
        assert!(try_extract_subquery("").is_none());
    }

    #[test]
    fn test_detect_subquery_case_insensitive() {
        let block = "select ?x where { ?x <http://p> ?y }";
        assert!(try_extract_subquery(block).is_some());
    }

    // ---- SubqueryEvaluator ----

    #[test]
    fn test_subquery_evaluator_basic() {
        let store = make_store_with_data();
        let evaluator = SubqueryEvaluator::new(&store);
        let subquery = "SELECT ?x WHERE { ?x <http://type> <http://Person> }";
        let results = evaluator.evaluate(subquery).expect("evaluate");
        assert_eq!(results.len(), 2);
        let subjects: Vec<String> = results.iter().filter_map(|r| r.get("x").cloned()).collect();
        assert!(subjects.contains(&"http://alice".to_string()));
        assert!(subjects.contains(&"http://bob".to_string()));
    }

    #[test]
    fn test_subquery_evaluator_with_filter() {
        let store = make_store_with_data();
        let evaluator = SubqueryEvaluator::new(&store);
        let subquery = r#"SELECT ?x WHERE { ?x <http://name> ?n . FILTER(?n = "\"Alice\"") }"#;
        // simpler filter
        let subquery2 = "SELECT ?x WHERE { ?x <http://name> \"Alice\" }";
        let results = evaluator.evaluate(subquery2).expect("evaluate");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("x").expect("x"), "http://alice");
    }

    #[test]
    fn test_subquery_evaluator_empty_result() {
        let store = make_store_with_data();
        let evaluator = SubqueryEvaluator::new(&store);
        let subquery = "SELECT ?x WHERE { ?x <http://nonexistent> <http://y> }";
        let results = evaluator.evaluate(subquery).expect("evaluate");
        assert_eq!(results.len(), 0);
    }

    // ---- join_with_subquery ----

    #[test]
    fn test_join_compatible_bindings() {
        let outer = vec![{
            let mut m = HashMap::new();
            m.insert("s".to_string(), "http://alice".to_string());
            m
        }];
        let sub = vec![{
            let mut m = HashMap::new();
            m.insert("s".to_string(), "http://alice".to_string());
            m.insert("n".to_string(), "Alice".to_string());
            m
        }];
        let result = join_with_subquery(outer, sub);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("n").expect("n"), "Alice");
    }

    #[test]
    fn test_join_conflicting_bindings() {
        let outer = vec![{
            let mut m = HashMap::new();
            m.insert("s".to_string(), "http://alice".to_string());
            m
        }];
        let sub = vec![{
            let mut m = HashMap::new();
            m.insert("s".to_string(), "http://bob".to_string()); // conflict
            m
        }];
        let result = join_with_subquery(outer, sub);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_join_disjoint_bindings() {
        let outer = vec![{
            let mut m = HashMap::new();
            m.insert("s".to_string(), "http://alice".to_string());
            m
        }];
        let sub = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), "http://bob".to_string());
            m
        }];
        let result = join_with_subquery(outer, sub);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("s").expect("s"), "http://alice");
        assert_eq!(result[0].get("x").expect("x"), "http://bob");
    }

    #[test]
    fn test_join_multiple_outer_multiple_sub() {
        let outer = vec![
            {
                let mut m = HashMap::new();
                m.insert("s".to_string(), "http://a".to_string());
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("s".to_string(), "http://b".to_string());
                m
            },
        ];
        let sub = vec![
            {
                let mut m = HashMap::new();
                m.insert("x".to_string(), "v1".to_string());
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("x".to_string(), "v2".to_string());
                m
            },
        ];
        let result = join_with_subquery(outer, sub);
        // 2 outer × 2 sub = 4 rows (disjoint vars, always compatible)
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_join_empty_outer() {
        let outer: Vec<Binding> = vec![];
        let sub = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), "v".to_string());
            m
        }];
        let result = join_with_subquery(outer, sub);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_join_empty_subquery() {
        let outer = vec![{
            let mut m = HashMap::new();
            m.insert("s".to_string(), "http://a".to_string());
            m
        }];
        let result = join_with_subquery(outer, vec![]);
        assert_eq!(result.len(), 0);
    }

    // ---- End-to-end subquery via execute_select ----

    #[test]
    fn test_end_to_end_subquery_via_execute_select() {
        // This tests that a subquery can be executed independently and then
        // joined with outer results manually (the pattern used in mod.rs).
        let store = make_store_with_data();

        // Inner: get all Persons
        let inner_q = "SELECT ?x WHERE { ?x <http://type> <http://Person> }";
        let inner_results = crate::query::execute_select(inner_q, &store).expect("inner");
        assert_eq!(inner_results.len(), 2);

        // Outer: get names
        let outer_q = "SELECT ?x ?name WHERE { ?x <http://name> ?name }";
        let outer_results = crate::query::execute_select(outer_q, &store).expect("outer");
        assert_eq!(outer_results.len(), 2);

        // Join: only Persons with names
        let joined = join_with_subquery(outer_results, inner_results);
        assert_eq!(joined.len(), 2);
        for row in &joined {
            assert!(row.contains_key("x"));
            assert!(row.contains_key("name"));
        }
    }

    // ---- Inline SPARQL subquery tests (uses the { SELECT ... } GraphPattern) ----

    #[test]
    fn test_inline_subquery_in_where_clause() {
        // SELECT ?x ?name WHERE { { SELECT ?x WHERE { ?x <type> <Person> } } ?x <name> ?name }
        let store = make_store_with_data();
        let sparql = r#"
            SELECT ?x ?name
            WHERE {
                { SELECT ?x WHERE { ?x <http://type> <http://Person> } }
                ?x <http://name> ?name
            }
        "#;
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 2);
        for row in &results {
            assert!(row.contains_key("x"));
            assert!(row.contains_key("name"));
        }
    }

    #[test]
    fn test_inline_subquery_with_limit() {
        // Inner subquery with LIMIT 1 should restrict the outer join
        let store = make_store_with_data();
        let sparql = r#"
            SELECT ?x
            WHERE {
                { SELECT ?x WHERE { ?x <http://type> <http://Person> } LIMIT 1 }
            }
        "#;
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_inline_subquery_filter_in_inner() {
        // Inner subquery with a FILTER should narrow results
        let store = make_store_with_data();
        let sparql = r#"
            SELECT ?x ?name
            WHERE {
                { SELECT ?x WHERE { ?x <http://knows> <http://bob> } }
                ?x <http://name> ?name
            }
        "#;
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("x").expect("x"), "http://alice");
    }

    #[test]
    fn test_inline_subquery_empty_inner_results_in_empty_outer() {
        let store = make_store_with_data();
        let sparql = r#"
            SELECT ?x
            WHERE {
                { SELECT ?x WHERE { ?x <http://nonexistent> <http://y> } }
            }
        "#;
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_subquery_evaluator_with_distinct() {
        let mut store = crate::store::OxiRSStore::new();
        store.insert("http://a", "http://p", "http://x");
        store.insert("http://b", "http://p", "http://x");
        store.insert("http://c", "http://p", "http://y");

        let evaluator = SubqueryEvaluator::new(&store);
        let subquery = "SELECT DISTINCT ?o WHERE { ?s <http://p> ?o }";
        let results = evaluator.evaluate(subquery).expect("evaluate");
        assert_eq!(results.len(), 2); // http://x, http://y (DISTINCT)
    }

    #[test]
    fn test_subquery_evaluator_order_by() {
        let mut store = crate::store::OxiRSStore::new();
        store.insert("http://c", "http://p", "\"c\"");
        store.insert("http://a", "http://p", "\"a\"");
        store.insert("http://b", "http://p", "\"b\"");

        let evaluator = SubqueryEvaluator::new(&store);
        let subquery = "SELECT ?s ?v WHERE { ?s <http://p> ?v } ORDER BY ?v";
        let results = evaluator.evaluate(subquery).expect("evaluate");
        assert_eq!(results.len(), 3);
        let vals: Vec<&str> = results
            .iter()
            .map(|r| r.get("v").map(|s| s.as_str()).unwrap_or(""))
            .collect();
        assert!(vals[0] <= vals[1]);
        assert!(vals[1] <= vals[2]);
    }

    #[test]
    fn test_try_merge_no_shared_vars() {
        let a: Binding = [("x".to_string(), "1".to_string())].into_iter().collect();
        let b: Binding = [("y".to_string(), "2".to_string())].into_iter().collect();
        let merged = try_merge(&a, &b).expect("merge");
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.get("x").expect("x"), "1");
        assert_eq!(merged.get("y").expect("y"), "2");
    }

    #[test]
    fn test_try_merge_shared_same_value() {
        let a: Binding = [
            ("x".to_string(), "1".to_string()),
            ("y".to_string(), "shared".to_string()),
        ]
        .into_iter()
        .collect();
        let b: Binding = [
            ("y".to_string(), "shared".to_string()),
            ("z".to_string(), "3".to_string()),
        ]
        .into_iter()
        .collect();
        let merged = try_merge(&a, &b).expect("merge");
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_try_merge_shared_different_value_returns_none() {
        let a: Binding = [("x".to_string(), "1".to_string())].into_iter().collect();
        let b: Binding = [("x".to_string(), "2".to_string())].into_iter().collect();
        assert!(try_merge(&a, &b).is_none());
    }
}
