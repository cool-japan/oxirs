//! Tests for [`QueryExecutor`].
//!
//! Extracted from `executor.rs` to keep individual source files under 2000 lines.

use super::executor::QueryExecutor;
use crate::model::*;
use crate::rdf_store::storage::{MemoryStorage, StorageBackend};
use std::sync::{Arc, RwLock};

/// Helper: create a Memory-backed StorageBackend with no data
fn make_empty_backend() -> StorageBackend {
    StorageBackend::Memory(Arc::new(RwLock::new(MemoryStorage::new())))
}

/// Helper: create a Memory-backed StorageBackend pre-populated with sample RDF triples
fn make_populated_backend() -> StorageBackend {
    let mut storage = MemoryStorage::new();

    let alice = NamedNode::new_unchecked("http://example.org/alice");
    let name_pred = NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name");
    let bob = NamedNode::new_unchecked("http://example.org/bob");
    let knows_pred = NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/knows");
    let age_pred = NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age");

    let alice_name_quad = Quad::new(
        Subject::NamedNode(alice.clone()),
        Predicate::NamedNode(name_pred.clone()),
        Object::Literal(Literal::new("Alice")),
        GraphName::DefaultGraph,
    );
    let bob_name_quad = Quad::new(
        Subject::NamedNode(bob.clone()),
        Predicate::NamedNode(name_pred),
        Object::Literal(Literal::new("Bob")),
        GraphName::DefaultGraph,
    );
    let alice_knows_bob = Quad::new(
        Subject::NamedNode(alice.clone()),
        Predicate::NamedNode(knows_pred.clone()),
        Object::NamedNode(bob.clone()),
        GraphName::DefaultGraph,
    );
    let bob_knows_alice = Quad::new(
        Subject::NamedNode(bob.clone()),
        Predicate::NamedNode(knows_pred),
        Object::NamedNode(alice.clone()),
        GraphName::DefaultGraph,
    );
    let alice_age_quad = Quad::new(
        Subject::NamedNode(alice),
        Predicate::NamedNode(age_pred),
        Object::Literal(Literal::new("30")),
        GraphName::DefaultGraph,
    );

    storage.insert_quad(alice_name_quad);
    storage.insert_quad(bob_name_quad);
    storage.insert_quad(alice_knows_bob);
    storage.insert_quad(bob_knows_alice);
    storage.insert_quad(alice_age_quad);

    StorageBackend::Memory(Arc::new(RwLock::new(storage)))
}

// --- Executor construction and stats ---

#[test]
fn test_executor_new_initial_stats_are_zero() {
    let backend = make_empty_backend();
    let executor = QueryExecutor::new(&backend);
    let stats = executor.get_stats();
    assert_eq!(
        stats.total_queries, 0,
        "Initially total_queries should be 0"
    );
    assert_eq!(
        stats.select_queries, 0,
        "Initially select_queries should be 0"
    );
    assert_eq!(stats.ask_queries, 0, "Initially ask_queries should be 0");
    assert_eq!(stats.construct_queries, 0);
    assert_eq!(stats.describe_queries, 0);
}

// --- SELECT query tests ---

#[test]
fn test_select_star_on_empty_store_returns_empty() {
    let backend = make_empty_backend();
    let executor = QueryExecutor::new(&backend);
    let result = executor
        .execute("SELECT * WHERE { ?s ?p ?o }")
        .expect("SELECT * on empty store should not error");
    assert!(
        result.is_empty(),
        "Result should be empty for an empty store"
    );
}

#[test]
fn test_select_star_returns_all_triples() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let result = executor
        .execute("SELECT * WHERE { ?s ?p ?o }")
        .expect("SELECT * should succeed");
    assert_eq!(result.len(), 5, "Expected 5 results, got {}", result.len());
}

#[test]
fn test_select_with_specific_subject() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let query = "SELECT ?p ?o WHERE { <http://example.org/alice> ?p ?o }";
    let result = executor
        .execute(query)
        .expect("SELECT with specific subject should succeed");
    assert_eq!(
        result.len(),
        3,
        "Alice should have 3 properties, got {}",
        result.len()
    );
}

#[test]
fn test_select_with_specific_predicate() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let query = "SELECT ?s ?o WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?o }";
    let result = executor
        .execute(query)
        .expect("SELECT with specific predicate should succeed");
    assert_eq!(
        result.len(),
        2,
        "Expected 2 name bindings, got {}",
        result.len()
    );
}

#[test]
fn test_select_with_specific_object_iri() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let query = "SELECT ?s WHERE { ?s <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> }";
    let result = executor
        .execute(query)
        .expect("SELECT with specific object should succeed");
    assert_eq!(
        result.len(),
        1,
        "Expected 1 result (alice knows bob), got {}",
        result.len()
    );
}

#[test]
fn test_select_with_limit() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 2";
    let result = executor
        .execute(query)
        .expect("SELECT with LIMIT should succeed");
    assert!(
        result.len() <= 2,
        "LIMIT 2 should return at most 2 results, got {}",
        result.len()
    );
}

#[test]
fn test_select_with_offset_and_limit() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);

    let all_results = executor
        .execute("SELECT * WHERE { ?s ?p ?o }")
        .expect("Unbounded SELECT should succeed");
    let total = all_results.len();

    let limited_results = executor
        .execute("SELECT * WHERE { ?s ?p ?o } LIMIT 3")
        .expect("SELECT with LIMIT 3 should succeed");

    if total >= 3 {
        assert_eq!(
            limited_results.len(),
            3,
            "Expected exactly 3 results with LIMIT 3"
        );
    }
}

// --- ASK query tests ---

#[test]
fn test_ask_returns_true_when_matching() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let result = executor
        .execute("ASK WHERE { ?s ?p ?o }")
        .expect("ASK WHERE should succeed");
    match result.results() {
        crate::rdf_store::types::QueryResults::Boolean(val) => {
            assert!(*val, "ASK WHERE should return true when store is non-empty");
        }
        other => panic!("Expected Boolean result from ASK WHERE, got: {:?}", other),
    }
}

#[test]
fn test_ask_returns_false_on_empty_store() {
    let backend = make_empty_backend();
    let executor = QueryExecutor::new(&backend);
    let result = executor
        .execute("ASK WHERE { ?s ?p ?o }")
        .expect("ASK WHERE on empty store should succeed");
    match result.results() {
        crate::rdf_store::types::QueryResults::Boolean(val) => {
            assert!(!val, "ASK WHERE on empty store should return false");
        }
        other => panic!("Expected Boolean result from ASK WHERE, got: {:?}", other),
    }
}

// --- CONSTRUCT query tests ---

#[test]
fn test_construct_query_basic() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let query = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }";
    let result = executor.execute(query).expect("CONSTRUCT should succeed");
    assert!(
        !result.is_empty(),
        "CONSTRUCT should return non-empty graph"
    );
}

// --- DESCRIBE query tests ---

#[test]
fn test_describe_query_specific_resource() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let query = "DESCRIBE <http://example.org/alice>";
    let result = executor.execute(query).expect("DESCRIBE should succeed");
    let _ = result.len();
}

// --- Error handling ---

#[test]
fn test_unsupported_query_type_returns_error() {
    let backend = make_empty_backend();
    let executor = QueryExecutor::new(&backend);
    let result = executor.execute("INSERT DATA { <s> <p> <o> }");
    assert!(
        result.is_err(),
        "Unsupported query type should return an error"
    );
}

#[test]
fn test_execute_and_query_are_equivalent() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let sparql = "SELECT * WHERE { ?s ?p ?o }";
    let via_execute = executor.execute(sparql).expect("execute() should succeed");
    let via_query = executor.query(sparql).expect("query() should succeed");
    assert_eq!(
        via_execute.len(),
        via_query.len(),
        "execute() and query() should return the same number of results"
    );
}

// --- Stats tracking after queries ---

#[test]
fn test_stats_track_select_queries() {
    let backend = make_empty_backend();
    let executor = QueryExecutor::new(&backend);
    executor.execute("SELECT * WHERE { ?s ?p ?o }").ok();
    executor.execute("SELECT ?s WHERE { ?s ?p ?o }").ok();
    let stats = executor.get_stats();
    assert_eq!(
        stats.select_queries, 2,
        "Should have tracked 2 SELECT queries"
    );
    assert_eq!(stats.total_queries, 2);
}

#[test]
fn test_stats_track_ask_queries() {
    let backend = make_empty_backend();
    let executor = QueryExecutor::new(&backend);
    executor.execute("ASK WHERE { ?s ?p ?o }").ok();
    let stats = executor.get_stats();
    assert_eq!(stats.ask_queries, 1, "Should have tracked 1 ASK query");
}

#[test]
fn test_stats_track_mixed_query_types() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    executor.execute("SELECT * WHERE { ?s ?p ?o }").ok();
    executor.execute("ASK WHERE { ?s ?p ?o }").ok();
    executor
        .execute("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
        .ok();
    executor.execute("DESCRIBE <http://example.org/alice>").ok();
    let stats = executor.get_stats();
    assert_eq!(stats.select_queries, 1);
    assert_eq!(stats.ask_queries, 1);
    assert_eq!(stats.construct_queries, 1);
    assert_eq!(stats.describe_queries, 1);
    assert_eq!(stats.total_queries, 4);
}

// --- PREFIX expansion ---

#[test]
fn test_prefix_expansion_in_select() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let query = "PREFIX foaf: <http://xmlns.com/foaf/0.1/> \
                 SELECT ?s ?o WHERE { ?s foaf:name ?o }";
    let result = executor
        .execute(query)
        .expect("PREFIX expansion SELECT should succeed");
    assert_eq!(
        result.len(),
        2,
        "Expected 2 results with prefix expansion, got {}",
        result.len()
    );
}

// --- SELECT DISTINCT ---

#[test]
fn test_select_distinct_reduces_duplicates() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let all_subjects = executor
        .execute("SELECT ?s WHERE { ?s ?p ?o }")
        .expect("SELECT should succeed");
    let distinct_subjects = executor
        .execute("SELECT DISTINCT ?s WHERE { ?s ?p ?o }")
        .expect("SELECT DISTINCT should succeed");
    assert!(
        distinct_subjects.len() <= all_subjects.len(),
        "DISTINCT should not exceed total results: {} vs {}",
        distinct_subjects.len(),
        all_subjects.len()
    );
}

// --- Optional WHERE keyword (SPARQL 1.1 `WhereClause ::= 'WHERE'? GroupGraphPattern`) ---

#[test]
fn test_ask_without_where_keyword_true_when_matching() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let result = executor
        .execute("ASK { ?s ?p ?o }")
        .expect("ASK without WHERE should succeed");
    match result.results() {
        crate::rdf_store::types::QueryResults::Boolean(val) => {
            assert!(
                *val,
                "ASK without WHERE keyword should return true against a non-empty store"
            );
        }
        other => panic!("Expected Boolean result from ASK, got: {:?}", other),
    }
}

#[test]
fn test_ask_without_where_keyword_false_on_empty_store() {
    let backend = make_empty_backend();
    let executor = QueryExecutor::new(&backend);
    let result = executor
        .execute("ASK { ?s ?p ?o }")
        .expect("ASK without WHERE on empty store should succeed");
    match result.results() {
        crate::rdf_store::types::QueryResults::Boolean(val) => {
            assert!(
                !val,
                "ASK without WHERE keyword should return false against an empty store"
            );
        }
        other => panic!("Expected Boolean result from ASK, got: {:?}", other),
    }
}

#[test]
fn test_select_star_without_where_keyword_returns_rows() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let with_where = executor
        .execute("SELECT * WHERE { ?s ?p ?o }")
        .expect("SELECT * WHERE should succeed");
    let without_where = executor
        .execute("SELECT * { ?s ?p ?o }")
        .expect("SELECT * without WHERE should succeed");
    assert_eq!(
        without_where.len(),
        5,
        "SELECT * without WHERE should return all 5 triples, got {}",
        without_where.len()
    );
    assert_eq!(
        without_where.len(),
        with_where.len(),
        "SELECT * with and without WHERE must return the same number of rows"
    );
}

#[test]
fn test_select_projection_without_where_keyword() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    let query = "SELECT ?o { ?s <http://xmlns.com/foaf/0.1/name> ?o }";
    let result = executor
        .execute(query)
        .expect("SELECT ?o without WHERE should succeed");
    assert_eq!(
        result.len(),
        2,
        "Expected 2 name bindings without WHERE keyword, got {}",
        result.len()
    );
}

#[test]
fn test_construct_without_where_keyword() {
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);
    // CONSTRUCT with an omitted WHERE: first `{...}` is the template, the
    // second is the graph pattern.
    let query = "CONSTRUCT { ?s ?p ?o } { ?s ?p ?o }";
    let result = executor
        .execute(query)
        .expect("CONSTRUCT without WHERE should succeed");
    assert!(
        !result.is_empty(),
        "CONSTRUCT without WHERE keyword should return a non-empty graph"
    );
}

#[test]
fn test_where_keyword_forms_unchanged() {
    // Regression guard: the WHERE-spelled forms must keep working identically.
    let backend = make_populated_backend();
    let executor = QueryExecutor::new(&backend);

    let select = executor
        .execute("SELECT * WHERE { ?s ?p ?o }")
        .expect("SELECT * WHERE should succeed");
    assert_eq!(select.len(), 5);

    let ask = executor
        .execute("ASK WHERE { ?s ?p ?o }")
        .expect("ASK WHERE should succeed");
    match ask.results() {
        crate::rdf_store::types::QueryResults::Boolean(val) => assert!(*val),
        other => panic!("Expected Boolean from ASK WHERE, got: {:?}", other),
    }

    let construct = executor
        .execute("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
        .expect("CONSTRUCT WHERE should succeed");
    assert!(!construct.is_empty());
}

#[test]
fn test_where_substring_in_iri_not_treated_as_keyword() {
    // An IRI containing the substring "where" must not confuse WHERE-locating.
    let mut storage = MemoryStorage::new();
    let s = NamedNode::new_unchecked("http://example.org/nowhere");
    let p = NamedNode::new_unchecked("http://example.org/somewhere");
    let o = NamedNode::new_unchecked("http://example.org/elsewhere");
    storage.insert_quad(Quad::new(
        Subject::NamedNode(s),
        Predicate::NamedNode(p),
        Object::NamedNode(o),
        GraphName::DefaultGraph,
    ));
    let backend = StorageBackend::Memory(Arc::new(RwLock::new(storage)));
    let executor = QueryExecutor::new(&backend);

    let result = executor
        .execute("SELECT * { ?s <http://example.org/somewhere> ?o }")
        .expect("SELECT with 'where' substring inside IRI should succeed");
    assert_eq!(
        result.len(),
        1,
        "Expected exactly 1 row despite 'where' substring inside the IRI, got {}",
        result.len()
    );
}
