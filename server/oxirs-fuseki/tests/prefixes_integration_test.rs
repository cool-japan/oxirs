//! Prefix Management Integration Tests
//!
//! Tests for namespace prefix CRUD operations

use std::sync::Arc;

/// Test listing all prefixes (includes well-known prefixes)
#[tokio::test]
async fn test_list_prefixes() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();
    let state = Arc::new(store);

    // TODO: Build router and test GET /$/prefixes
    // Verify response includes rdf, rdfs, owl, xsd, foaf, dc, dcterms, skos
}

/// Test getting a specific well-known prefix
#[tokio::test]
async fn test_get_well_known_prefix() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    let result = store.get("rdf").unwrap();
    assert_eq!(
        result,
        Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string())
    );

    let result = store.get("rdfs").unwrap();
    assert_eq!(
        result,
        Some("http://www.w3.org/2000/01/rdf-schema#".to_string())
    );

    let result = store.get("owl").unwrap();
    assert_eq!(result, Some("http://www.w3.org/2002/07/owl#".to_string()));

    let result = store.get("xsd").unwrap();
    assert_eq!(
        result,
        Some("http://www.w3.org/2001/XMLSchema#".to_string())
    );
}

/// Test getting non-existent prefix returns None
#[tokio::test]
async fn test_get_nonexistent_prefix() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    let result = store.get("nonexistent").unwrap();
    assert_eq!(result, None);
}

/// Test adding new prefix
#[tokio::test]
async fn test_add_prefix() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    // Add new prefix
    store
        .set("ex".to_string(), "http://example.org/".to_string())
        .unwrap();

    // Verify it exists
    assert!(store.exists("ex").unwrap());

    // Verify we can retrieve it
    let result = store.get("ex").unwrap();
    assert_eq!(result, Some("http://example.org/".to_string()));
}

/// Test updating existing prefix
#[tokio::test]
async fn test_update_prefix() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    // Add initial prefix
    store
        .set("ex".to_string(), "http://example.org/".to_string())
        .unwrap();

    // Update to new URI
    store
        .set("ex".to_string(), "http://example.com/".to_string())
        .unwrap();

    // Verify updated value
    let result = store.get("ex").unwrap();
    assert_eq!(result, Some("http://example.com/".to_string()));
}

/// Test deleting prefix
#[tokio::test]
async fn test_delete_prefix() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    // Add prefix
    store
        .set("ex".to_string(), "http://example.org/".to_string())
        .unwrap();
    assert!(store.exists("ex").unwrap());

    // Delete it
    let deleted = store.delete("ex").unwrap();
    assert!(deleted);

    // Verify it no longer exists
    assert!(!store.exists("ex").unwrap());
}

/// Test deleting non-existent prefix returns false
#[tokio::test]
async fn test_delete_nonexistent_prefix() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    let deleted = store.delete("nonexistent").unwrap();
    assert!(!deleted);
}

/// Test expanding prefixed name
#[tokio::test]
async fn test_expand_prefixed_name() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    // Expand well-known prefix
    let expanded = store.expand("rdf:type").unwrap();
    assert_eq!(
        expanded,
        Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
    );

    let expanded = store.expand("rdfs:label").unwrap();
    assert_eq!(
        expanded,
        Some("http://www.w3.org/2000/01/rdf-schema#label".to_string())
    );
}

/// Test expanding with custom prefix
#[tokio::test]
async fn test_expand_custom_prefix() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    store
        .set("ex".to_string(), "http://example.org/".to_string())
        .unwrap();

    let expanded = store.expand("ex:Person").unwrap();
    assert_eq!(expanded, Some("http://example.org/Person".to_string()));

    let expanded = store.expand("ex:name").unwrap();
    assert_eq!(expanded, Some("http://example.org/name".to_string()));
}

/// Test expanding with unknown prefix returns None
#[tokio::test]
async fn test_expand_unknown_prefix() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    let expanded = store.expand("unknown:foo").unwrap();
    assert_eq!(expanded, None);
}

/// Test expanding invalid format returns None
#[tokio::test]
async fn test_expand_invalid_format() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    // No colon
    let expanded = store.expand("nocolon").unwrap();
    assert_eq!(expanded, None);
}

/// Test prefix validation - valid prefixes
#[tokio::test]
async fn test_valid_prefix_names() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    // Valid prefixes
    assert!(store
        .set("ex".to_string(), "http://example.org/".to_string())
        .is_ok());
    assert!(store
        .set("foo123".to_string(), "http://example.org/".to_string())
        .is_ok());
    assert!(store
        .set("foo_bar".to_string(), "http://example.org/".to_string())
        .is_ok());
    assert!(store
        .set("foo-bar".to_string(), "http://example.org/".to_string())
        .is_ok());
}

/// Test prefix validation - invalid prefixes
#[tokio::test]
async fn test_invalid_prefix_names() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    // Empty prefix
    assert!(store
        .set("".to_string(), "http://example.org/".to_string())
        .is_err());

    // Starts with number
    assert!(store
        .set("123foo".to_string(), "http://example.org/".to_string())
        .is_err());

    // Contains colon
    assert!(store
        .set("foo:bar".to_string(), "http://example.org/".to_string())
        .is_err());

    // Contains space
    assert!(store
        .set("foo bar".to_string(), "http://example.org/".to_string())
        .is_err());

    // Contains special characters
    assert!(store
        .set("foo@bar".to_string(), "http://example.org/".to_string())
        .is_err());
}

/// Test URI validation - valid URIs
#[tokio::test]
async fn test_valid_uris() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    assert!(store
        .set("ex1".to_string(), "http://example.org/".to_string())
        .is_ok());
    assert!(store
        .set("ex2".to_string(), "https://example.org/".to_string())
        .is_ok());
    assert!(store
        .set("ex3".to_string(), "urn:example:123".to_string())
        .is_ok());
}

/// Test URI validation - invalid URIs
#[tokio::test]
async fn test_invalid_uris() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    // Empty URI
    assert!(store.set("ex".to_string(), "".to_string()).is_err());

    // Relative URI
    assert!(store
        .set("ex".to_string(), "example.org".to_string())
        .is_err());

    // Protocol-relative URI
    assert!(store
        .set("ex".to_string(), "//example.org".to_string())
        .is_err());
}

/// Test concurrent access to prefix store
#[tokio::test]
async fn test_concurrent_access() {
    use tokio::task;

    let store = Arc::new(oxirs_fuseki::handlers::PrefixStore::new());
    let mut handles = vec![];

    // Spawn 10 tasks that add prefixes concurrently
    for i in 0..10 {
        let store_clone = store.clone();
        let handle = task::spawn(async move {
            store_clone
                .set(format!("ex{}", i), format!("http://example.org/{}/", i))
                .unwrap();
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all prefixes were added
    for i in 0..10 {
        assert!(store.exists(&format!("ex{}", i)).unwrap());
    }
}

/// Test prefix store thread safety with reads and writes
#[tokio::test]
async fn test_concurrent_reads_writes() {
    use tokio::task;

    let store = Arc::new(oxirs_fuseki::handlers::PrefixStore::new());
    let mut handles = vec![];

    // Add initial prefix
    store
        .set("ex".to_string(), "http://example.org/".to_string())
        .unwrap();

    // Spawn 5 readers
    for _ in 0..5 {
        let store_clone = store.clone();
        let handle = task::spawn(async move {
            let result = store_clone.get("ex").unwrap();
            assert!(result.is_some());
        });
        handles.push(handle);
    }

    // Spawn 5 writers
    for i in 0..5 {
        let store_clone = store.clone();
        let handle = task::spawn(async move {
            store_clone
                .set(format!("test{}", i), format!("http://test{}.org/", i))
                .unwrap();
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test prefix store with special characters in local names
#[tokio::test]
async fn test_expand_with_special_chars() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    store
        .set("ex".to_string(), "http://example.org/".to_string())
        .unwrap();

    // Local names can contain various characters
    let expanded = store.expand("ex:Person#1").unwrap();
    assert_eq!(expanded, Some("http://example.org/Person#1".to_string()));

    let expanded = store.expand("ex:path/to/resource").unwrap();
    assert_eq!(
        expanded,
        Some("http://example.org/path/to/resource".to_string())
    );
}

/// Test listing returns all prefixes
#[tokio::test]
async fn test_list_contains_all_prefixes() {
    let store = oxirs_fuseki::handlers::PrefixStore::new();

    // Add custom prefixes
    store
        .set("ex1".to_string(), "http://example1.org/".to_string())
        .unwrap();
    store
        .set("ex2".to_string(), "http://example2.org/".to_string())
        .unwrap();

    let all_prefixes = store.list().unwrap();

    // Should contain well-known prefixes
    assert!(all_prefixes.contains_key("rdf"));
    assert!(all_prefixes.contains_key("rdfs"));
    assert!(all_prefixes.contains_key("owl"));

    // Should contain custom prefixes
    assert!(all_prefixes.contains_key("ex1"));
    assert!(all_prefixes.contains_key("ex2"));

    // Check we have at least 10 prefixes (8 well-known + 2 custom)
    assert!(all_prefixes.len() >= 10);
}
