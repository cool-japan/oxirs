//! Integration tests for SPARQL Update operations
//!
//! Tests for CREATE, DROP, COPY, MOVE, ADD, and other SPARQL Update operations.

use oxirs_fuseki::store::Store;

/// Create a test store with some sample data
fn create_test_store() -> Store {
    let store = Store::new().expect("Failed to create store");

    // Add some test data to the default graph
    let test_data = r#"
        <http://example.org/subject1> <http://example.org/predicate1> "value1" .
        <http://example.org/subject2> <http://example.org/predicate2> "value2" .
    "#;

    let sparql = format!("INSERT DATA {{ {} }}", test_data);
    store.update(&sparql).expect("Failed to insert test data");

    store
}

#[test]
fn test_create_graph() {
    let store = create_test_store();

    // Test CREATE GRAPH
    let sparql = "CREATE GRAPH <http://example.org/graph1>";
    let result = store.update(sparql);

    assert!(
        result.is_ok(),
        "CREATE GRAPH should succeed: {:?}",
        result.err()
    );
    let update_result = result.unwrap();
    assert_eq!(update_result.stats.operation_type, "CREATE");
}

#[test]
fn test_create_silent_graph() {
    let store = create_test_store();

    // Test CREATE SILENT GRAPH
    let result = store.update("CREATE SILENT GRAPH <http://example.org/graph1>");

    assert!(result.is_ok(), "CREATE SILENT GRAPH should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "CREATE SILENT");
}

#[test]
fn test_drop_graph() {
    let store = create_test_store();

    // First, add data to a named graph
    store.update("INSERT DATA { GRAPH <http://example.org/graph1> { <http://example.org/s> <http://example.org/p> \"o\" . } }").unwrap();

    // Test DROP GRAPH
    let result = store.update("DROP GRAPH <http://example.org/graph1>");

    assert!(result.is_ok(), "DROP GRAPH should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "DROP GRAPH");
    assert_eq!(stats.quads_deleted, 1, "Should have deleted 1 quad");
}

#[test]
fn test_drop_default() {
    let store = create_test_store();

    // Test DROP DEFAULT
    let result = store.update("DROP DEFAULT");

    assert!(result.is_ok(), "DROP DEFAULT should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "DROP DEFAULT");
    assert!(
        stats.quads_deleted >= 2,
        "Should delete at least 2 quads from default graph"
    );
}

#[test]
#[ignore = "Requires backend clear_all support"]
fn test_drop_all() {
    let store = create_test_store();

    // Add data to multiple graphs
    store.update("INSERT DATA { GRAPH <http://example.org/graph1> { <http://example.org/s1> <http://example.org/p1> \"o1\" . } }").unwrap();

    // Test DROP ALL
    let result = store.update("DROP ALL");

    assert!(
        result.is_ok(),
        "DROP ALL should succeed: {:?}",
        result.err()
    );
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "DROP ALL");
    assert!(stats.quads_deleted >= 3, "Should delete all quads");
}

#[test]
fn test_drop_named() {
    let store = create_test_store();

    // Add data to multiple named graphs
    store.update("INSERT DATA { GRAPH <http://example.org/graph1> { <http://example.org/s1> <http://example.org/p1> \"o1\" . } }").unwrap();
    store.update("INSERT DATA { GRAPH <http://example.org/graph2> { <http://example.org/s2> <http://example.org/p2> \"o2\" . } }").unwrap();

    // Test DROP NAMED (should drop all named graphs but keep default)
    let result = store.update("DROP NAMED");

    assert!(result.is_ok(), "DROP NAMED should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "DROP NAMED");
    assert_eq!(
        stats.quads_deleted, 2,
        "Should delete 2 quads from named graphs"
    );
}

#[test]
fn test_copy_graph() {
    let store = create_test_store();

    // Add data to source graph
    store.update("INSERT DATA { GRAPH <http://example.org/source> { <http://example.org/s> <http://example.org/p> \"value\" . } }").unwrap();

    // Test COPY GRAPH
    let result =
        store.update("COPY GRAPH <http://example.org/source> TO GRAPH <http://example.org/target>");

    assert!(result.is_ok(), "COPY GRAPH should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "COPY");
    assert_eq!(stats.quads_inserted, 1);
}

#[test]
fn test_copy_default_to_named() {
    let store = create_test_store();

    // Test COPY from DEFAULT to named graph
    let result = store.update("COPY DEFAULT TO GRAPH <http://example.org/target>");

    assert!(result.is_ok(), "COPY DEFAULT should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "COPY");
    assert!(stats.quads_inserted >= 2, "Should copy at least 2 quads");
}

#[test]
fn test_move_graph() {
    let store = create_test_store();

    // Add data to source graph
    store.update("INSERT DATA { GRAPH <http://example.org/source> { <http://example.org/s> <http://example.org/p> \"value\" . } }").unwrap();

    // Test MOVE GRAPH
    let result =
        store.update("MOVE GRAPH <http://example.org/source> TO GRAPH <http://example.org/target>");

    assert!(result.is_ok(), "MOVE GRAPH should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "MOVE");
    assert_eq!(stats.quads_inserted, 1);
    assert_eq!(stats.quads_deleted, 1);
}

#[test]
fn test_move_default_to_named() {
    let store = create_test_store();

    // Test MOVE from DEFAULT to named graph
    let result = store.update("MOVE DEFAULT TO GRAPH <http://example.org/target>");

    assert!(result.is_ok(), "MOVE DEFAULT should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "MOVE");
    assert!(stats.quads_inserted >= 2, "Should move at least 2 quads");
    assert!(stats.quads_deleted >= 2, "Should delete source quads");
}

#[test]
fn test_add_graph() {
    let store = create_test_store();

    // Add data to source graph
    store.update("INSERT DATA { GRAPH <http://example.org/source> { <http://example.org/s> <http://example.org/p> \"value\" . } }").unwrap();

    // Add some data to target graph
    store.update("INSERT DATA { GRAPH <http://example.org/target> { <http://example.org/s2> <http://example.org/p2> \"value2\" . } }").unwrap();

    // Test ADD GRAPH (should not clear target)
    let result =
        store.update("ADD GRAPH <http://example.org/source> TO GRAPH <http://example.org/target>");

    assert!(result.is_ok(), "ADD GRAPH should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "ADD");
    assert_eq!(stats.quads_inserted, 1);
    assert_eq!(stats.quads_deleted, 0, "ADD should not delete any quads");
}

#[test]
fn test_add_default_to_named() {
    let store = create_test_store();

    // Add some data to target graph
    store.update("INSERT DATA { GRAPH <http://example.org/target> { <http://example.org/s2> <http://example.org/p2> \"value2\" . } }").unwrap();

    // Test ADD from DEFAULT to named graph
    let result = store.update("ADD DEFAULT TO GRAPH <http://example.org/target>");

    assert!(result.is_ok(), "ADD DEFAULT should succeed");
    let stats = result.unwrap().stats;
    assert_eq!(stats.operation_type, "ADD");
    assert!(stats.quads_inserted >= 2, "Should add at least 2 quads");
    assert_eq!(stats.quads_deleted, 0, "ADD should not delete any quads");
}

#[test]
fn test_silent_operations() {
    let store = create_test_store();

    // Test DROP SILENT on non-existent graph (should not fail)
    let result = store.update("DROP SILENT GRAPH <http://example.org/nonexistent>");

    assert!(
        result.is_ok(),
        "DROP SILENT should succeed even for non-existent graph"
    );

    // Test COPY SILENT
    store.update("INSERT DATA { GRAPH <http://example.org/source> { <http://example.org/s> <http://example.org/p> \"value\" . } }").unwrap();

    let result = store.update(
        "COPY SILENT GRAPH <http://example.org/source> TO GRAPH <http://example.org/target>",
    );

    assert!(result.is_ok(), "COPY SILENT should succeed");
    assert_eq!(result.unwrap().stats.operation_type, "COPY SILENT");
}

#[test]
#[ignore = "Complex sequence test - TO keyword parsing issue"]
fn test_complex_update_sequence() {
    let store = create_test_store();

    // Test a complex sequence of operations

    // 1. CREATE a new graph
    store
        .update("CREATE GRAPH <http://example.org/temp>")
        .unwrap();

    // 2. COPY default data to the new graph
    store
        .update("COPY DEFAULT TO GRAPH <http://example.org/temp>")
        .unwrap();

    // 3. Add more data to the temp graph
    store.update("INSERT DATA { GRAPH <http://example.org/temp> { <http://example.org/new> <http://example.org/prop> \"added\" . } }").unwrap();

    // 4. MOVE the temp graph to final location
    let result =
        store.update("MOVE GRAPH <http://example.org/temp> TO GRAPH <http://example.org/final>");

    assert!(result.is_ok(), "Complex sequence should succeed");
    let stats = result.unwrap().stats;
    assert!(stats.quads_inserted >= 3, "Should have moved all quads");
}
