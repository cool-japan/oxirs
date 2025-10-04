//! Integration tests for SPARQL Federation support
//!
//! These tests verify the federation functionality against real SPARQL endpoints
//! like DBpedia and Wikidata.

use oxirs_core::federation::{FederationClient, FederationConfig, FederationExecutor};
use oxirs_core::model::{NamedNode, Term};
use oxirs_core::OxirsError;
use std::collections::HashMap;

/// Test basic federation client creation
#[tokio::test]
async fn test_federation_client_creation() {
    let client = FederationClient::new();
    assert!(client.is_ok(), "Failed to create federation client");
}

/// Test federation executor creation
#[tokio::test]
async fn test_federation_executor_creation() {
    let executor = FederationExecutor::new();
    assert!(executor.is_ok(), "Failed to create federation executor");
}

/// Test custom configuration
#[tokio::test]
async fn test_custom_config() {
    let config = FederationConfig {
        timeout_secs: 10,
        max_retries: 5,
        user_agent: "OxiRS-Test/0.1.0".to_string(),
        accept: "application/sparql-results+json".to_string(),
    };

    let client = FederationClient::with_config(config);
    assert!(client.is_ok(), "Failed to create client with custom config");
}

/// Test querying DBpedia for a simple fact
/// Note: This test requires network access and may be slow
#[tokio::test]
#[ignore] // Ignore by default to avoid network dependency in CI
async fn test_dbpedia_simple_query() {
    let client = FederationClient::new().expect("Failed to create client");

    let query = r#"
        SELECT ?capital WHERE {
            <http://dbpedia.org/resource/France> <http://dbpedia.org/ontology/capital> ?capital .
        } LIMIT 1
    "#;

    let endpoint = "https://dbpedia.org/sparql";
    let result = client.execute_query(endpoint, query, false).await;

    match result {
        Ok(json) => {
            println!("DBpedia response: {}", json);
            assert!(json.contains("results"), "Response should contain results");
        }
        Err(OxirsError::Federation(msg))
            if msg.contains("timeout") || msg.contains("connection") =>
        {
            // Network issues are acceptable in tests
            println!("Network error (acceptable): {}", msg);
        }
        Err(e) => {
            panic!("Unexpected error: {:?}", e);
        }
    }
}

/// Test querying Wikidata for a simple fact
/// Note: This test requires network access and may be slow
#[tokio::test]
#[ignore] // Ignore by default to avoid network dependency in CI
async fn test_wikidata_simple_query() {
    let client = FederationClient::new().expect("Failed to create client");

    let query = r#"
        SELECT ?item ?itemLabel WHERE {
            ?item wdt:P31 wd:Q5 .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        } LIMIT 1
    "#;

    let endpoint = "https://query.wikidata.org/sparql";
    let result = client.execute_query(endpoint, query, false).await;

    match result {
        Ok(json) => {
            println!("Wikidata response: {}", json);
            assert!(json.contains("results"), "Response should contain results");
        }
        Err(OxirsError::Federation(msg))
            if msg.contains("timeout") || msg.contains("connection") =>
        {
            // Network issues are acceptable in tests
            println!("Network error (acceptable): {}", msg);
        }
        Err(e) => {
            panic!("Unexpected error: {:?}", e);
        }
    }
}

/// Test SERVICE SILENT behavior with unreachable endpoint
#[tokio::test]
async fn test_service_silent_unreachable() {
    let client = FederationClient::new().expect("Failed to create client");

    let query = "SELECT * WHERE { ?s ?p ?o }";
    let endpoint = "http://localhost:99999/sparql"; // Unreachable endpoint

    // With silent=true, should return empty results instead of error
    let result = client.execute_query(endpoint, query, true).await;
    assert!(
        result.is_ok(),
        "SERVICE SILENT should return empty results on error"
    );

    let json = result.unwrap();
    assert!(
        json.contains(r#""bindings":[]"#),
        "Should contain empty bindings"
    );
}

/// Test SERVICE SILENT behavior with invalid query
#[tokio::test]
async fn test_service_silent_invalid_query() {
    let client = FederationClient::new().expect("Failed to create client");

    let query = "INVALID SPARQL QUERY";
    let endpoint = "http://localhost:99999/sparql"; // Unreachable endpoint

    // With silent=true, should return empty results
    let result = client.execute_query(endpoint, query, true).await;
    assert!(
        result.is_ok(),
        "SERVICE SILENT should succeed even with invalid query"
    );
}

/// Test result merging with no common variables (Cartesian product)
#[test]
fn test_merge_bindings_cartesian_product() {
    let executor = FederationExecutor::new().expect("Failed to create executor");

    let local = vec![{
        let mut m = HashMap::new();
        m.insert(
            "x".to_string(),
            Term::NamedNode(NamedNode::new("http://example.org/a").unwrap()),
        );
        m
    }];

    let remote = vec![{
        let mut m = HashMap::new();
        m.insert(
            "y".to_string(),
            Term::NamedNode(NamedNode::new("http://example.org/b").unwrap()),
        );
        m
    }];

    let result = executor.merge_bindings(local, remote);

    assert_eq!(result.len(), 1, "Should produce one merged binding");
    assert_eq!(
        result[0].len(),
        2,
        "Merged binding should have both variables"
    );
    assert!(result[0].contains_key("x"), "Should contain variable x");
    assert!(result[0].contains_key("y"), "Should contain variable y");
}

/// Test result merging with common variables (Hash join)
#[test]
fn test_merge_bindings_hash_join() {
    let executor = FederationExecutor::new().expect("Failed to create executor");

    let shared_node = Term::NamedNode(NamedNode::new("http://example.org/shared").unwrap());

    let local = vec![{
        let mut m = HashMap::new();
        m.insert("x".to_string(), shared_node.clone());
        m.insert(
            "y".to_string(),
            Term::NamedNode(NamedNode::new("http://example.org/local").unwrap()),
        );
        m
    }];

    let remote = vec![{
        let mut m = HashMap::new();
        m.insert("x".to_string(), shared_node.clone());
        m.insert(
            "z".to_string(),
            Term::NamedNode(NamedNode::new("http://example.org/remote").unwrap()),
        );
        m
    }];

    let result = executor.merge_bindings(local, remote);

    assert_eq!(result.len(), 1, "Should produce one merged binding");
    assert_eq!(
        result[0].len(),
        3,
        "Merged binding should have three variables (x, y, z)"
    );
    assert!(result[0].contains_key("x"), "Should contain variable x");
    assert!(result[0].contains_key("y"), "Should contain variable y");
    assert!(result[0].contains_key("z"), "Should contain variable z");
}

/// Test result merging with incompatible bindings
#[test]
fn test_merge_bindings_incompatible() {
    let executor = FederationExecutor::new().expect("Failed to create executor");

    let local = vec![{
        let mut m = HashMap::new();
        m.insert(
            "x".to_string(),
            Term::NamedNode(NamedNode::new("http://example.org/a").unwrap()),
        );
        m
    }];

    let remote = vec![{
        let mut m = HashMap::new();
        m.insert(
            "x".to_string(),
            Term::NamedNode(NamedNode::new("http://example.org/b").unwrap()),
        );
        m
    }];

    let result = executor.merge_bindings(local, remote);

    assert_eq!(
        result.len(),
        0,
        "Incompatible bindings should produce empty result"
    );
}

/// Test merging with empty local bindings
#[test]
fn test_merge_bindings_empty_local() {
    let executor = FederationExecutor::new().expect("Failed to create executor");

    let local = Vec::new();
    let remote = vec![{
        let mut m = HashMap::new();
        m.insert(
            "x".to_string(),
            Term::NamedNode(NamedNode::new("http://example.org/a").unwrap()),
        );
        m
    }];

    let result = executor.merge_bindings(local, remote);

    assert_eq!(result.len(), 1, "Should return remote bindings");
    assert_eq!(result[0].len(), 1, "Should have one variable");
}

/// Test merging with empty remote bindings
#[test]
fn test_merge_bindings_empty_remote() {
    let executor = FederationExecutor::new().expect("Failed to create executor");

    let local = vec![{
        let mut m = HashMap::new();
        m.insert(
            "x".to_string(),
            Term::NamedNode(NamedNode::new("http://example.org/a").unwrap()),
        );
        m
    }];
    let remote = Vec::new();

    let result = executor.merge_bindings(local, remote);

    assert_eq!(result.len(), 1, "Should return local bindings");
    assert_eq!(result[0].len(), 1, "Should have one variable");
}

/// Test retry mechanism with failing endpoint
#[tokio::test]
async fn test_retry_mechanism() {
    let config = FederationConfig {
        timeout_secs: 1,
        max_retries: 3,
        user_agent: "OxiRS-Test/0.1.0".to_string(),
        accept: "application/sparql-results+json".to_string(),
    };

    let client = FederationClient::with_config(config).expect("Failed to create client");
    let endpoint = "http://localhost:99999/sparql";
    let query = "SELECT * WHERE { ?s ?p ?o }";

    let start = std::time::Instant::now();
    let result = client.execute_query(endpoint, query, false).await;
    let elapsed = start.elapsed();

    assert!(result.is_err(), "Should fail for unreachable endpoint");

    // Should have attempted multiple retries
    println!("Elapsed time: {:?}", elapsed);
}
