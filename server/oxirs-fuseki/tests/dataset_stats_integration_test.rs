//! Dataset Statistics Integration Tests
//!
//! Tests for RDF dataset statistics collection

use oxirs_core::model::{GraphName, NamedNode, Quad, Subject};
use oxirs_core::Store as CoreStore;
use oxirs_fuseki::handlers::dataset_stats::StatisticsCollector;
use std::collections::HashMap;
use std::sync::Arc;

/// Helper to create test store with data
fn create_test_store_with_data() -> Arc<oxirs_fuseki::store::Store> {
    let store = oxirs_fuseki::store::Store::new().unwrap();
    Arc::new(store)
}

/// Test collecting statistics for empty store
#[tokio::test]
async fn test_empty_store_statistics() {
    let store = create_test_store_with_data();

    let stats = StatisticsCollector::collect_dataset_stats("test", &store).unwrap();

    assert_eq!(stats.dataset_name, "test");
    assert_eq!(stats.triples_in_default_graph, 0);
    assert_eq!(stats.triples_in_named_graphs, 0);
    assert_eq!(stats.total_quads, 0);
    assert_eq!(stats.named_graph_count, 0);
    assert_eq!(stats.total_triples(), 0);
}

/// Test collecting statistics with default graph triples
#[tokio::test]
async fn test_default_graph_statistics() {
    let store = create_test_store_with_data();

    // Add triples to default graph
    let s = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let p = NamedNode::new("http://example.org/name").unwrap();
    let o =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("Alice"));

    let quad = Quad::new(s, p, o, GraphName::DefaultGraph);

    store.insert_quad(quad).unwrap();

    let stats = StatisticsCollector::collect_dataset_stats("test", &store).unwrap();

    assert_eq!(stats.triples_in_default_graph, 1);
    assert_eq!(stats.triples_in_named_graphs, 0);
    assert_eq!(stats.total_quads, 1);
    assert_eq!(stats.named_graph_count, 0);
}

/// Test collecting statistics with named graph triples
#[tokio::test]
async fn test_named_graph_statistics() {
    let store = create_test_store_with_data();

    // Add triples to named graph
    let graph = GraphName::NamedNode(NamedNode::new("http://example.org/graph1").unwrap());

    let s = Subject::NamedNode(NamedNode::new("http://example.org/bob").unwrap());
    let p = NamedNode::new("http://example.org/age").unwrap();
    let o =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("30"));

    let quad = Quad::new(s, p, o, graph);
    store.insert_quad(quad).unwrap();

    let stats = StatisticsCollector::collect_dataset_stats("test", &store).unwrap();

    assert_eq!(stats.triples_in_default_graph, 0);
    assert_eq!(stats.triples_in_named_graphs, 1);
    assert_eq!(stats.total_quads, 1);
    assert_eq!(stats.named_graph_count, 1);
    assert_eq!(stats.named_graphs.len(), 1);
    assert!(stats
        .named_graphs
        .contains(&"http://example.org/graph1".to_string()));
}

/// Test collecting statistics with multiple named graphs
#[tokio::test]
async fn test_multiple_named_graphs() {
    let store = create_test_store_with_data();

    // Add to graph1
    let graph1 = GraphName::NamedNode(NamedNode::new("http://example.org/graph1").unwrap());
    let s1 = Subject::NamedNode(NamedNode::new("http://example.org/s1").unwrap());
    let p1 = NamedNode::new("http://example.org/p1").unwrap();
    let o1 =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("v1"));
    store.insert_quad(Quad::new(s1, p1, o1, graph1)).unwrap();

    // Add to graph2
    let graph2 = GraphName::NamedNode(NamedNode::new("http://example.org/graph2").unwrap());
    let s2 = Subject::NamedNode(NamedNode::new("http://example.org/s2").unwrap());
    let p2 = NamedNode::new("http://example.org/p2").unwrap();
    let o2 =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("v2"));
    store
        .insert_quad(Quad::new(
            s2.clone(),
            p2.clone(),
            o2.clone(),
            graph2.clone(),
        ))
        .unwrap();

    // Add another triple to graph2
    let s3 = Subject::NamedNode(NamedNode::new("http://example.org/s3").unwrap());
    let p3 = NamedNode::new("http://example.org/p3").unwrap();
    let o3 =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("v3"));
    store.insert_quad(Quad::new(s3, p3, o3, graph2)).unwrap();

    let stats = StatisticsCollector::collect_dataset_stats("test", &store).unwrap();

    assert_eq!(stats.triples_in_default_graph, 0);
    assert_eq!(stats.triples_in_named_graphs, 3);
    assert_eq!(stats.total_quads, 3);
    assert_eq!(stats.named_graph_count, 2);
    assert_eq!(stats.named_graphs.len(), 2);
}

/// Test statistics with both default and named graphs
#[tokio::test]
async fn test_mixed_graphs_statistics() {
    let store = create_test_store_with_data();

    // Add to default graph
    let s1 = Subject::NamedNode(NamedNode::new("http://example.org/default1").unwrap());
    let p1 = NamedNode::new("http://example.org/p1").unwrap();
    let o1 =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("v1"));
    store
        .insert_quad(Quad::new(s1, p1, o1, GraphName::DefaultGraph))
        .unwrap();

    // Add to named graph
    let graph = GraphName::NamedNode(NamedNode::new("http://example.org/graph1").unwrap());
    let s2 = Subject::NamedNode(NamedNode::new("http://example.org/named1").unwrap());
    let p2 = NamedNode::new("http://example.org/p2").unwrap();
    let o2 =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("v2"));
    store.insert_quad(Quad::new(s2, p2, o2, graph)).unwrap();

    let stats = StatisticsCollector::collect_dataset_stats("test", &store).unwrap();

    assert_eq!(stats.triples_in_default_graph, 1);
    assert_eq!(stats.triples_in_named_graphs, 1);
    assert_eq!(stats.total_quads, 2);
    assert_eq!(stats.named_graph_count, 1);
    assert_eq!(stats.total_triples(), 2);
}

/// Test storage size estimation
#[tokio::test]
async fn test_storage_size_estimation() {
    let store = create_test_store_with_data();

    // Add some triples
    for i in 0..10 {
        let s = Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", i)).unwrap());
        let p = NamedNode::new("http://example.org/p").unwrap();
        let o = oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal(
            &format!("value{}", i),
        ));
        store
            .insert_quad(Quad::new(s, p, o, GraphName::DefaultGraph))
            .unwrap();
    }

    let stats = StatisticsCollector::collect_dataset_stats("test", &store).unwrap();

    assert!(stats.storage_size_bytes.is_some());
    let size = stats.storage_size_bytes.unwrap();
    assert!(size > 0);
    // Rough estimate: 200 bytes per quad * 10 quads = 2000 bytes
    assert_eq!(size, 2000);
}

/// Test server statistics collection
#[tokio::test]
async fn test_server_statistics() {
    let store1 = create_test_store_with_data();
    let store2 = create_test_store_with_data();

    // Add data to store1
    let s1 = Subject::NamedNode(NamedNode::new("http://example.org/s1").unwrap());
    let p1 = NamedNode::new("http://example.org/p1").unwrap();
    let o1 =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("v1"));
    store1
        .insert_quad(Quad::new(s1, p1, o1, GraphName::DefaultGraph))
        .unwrap();

    // Add data to store2
    let s2 = Subject::NamedNode(NamedNode::new("http://example.org/s2").unwrap());
    let p2 = NamedNode::new("http://example.org/p2").unwrap();
    let o2 =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("v2"));
    store2
        .insert_quad(Quad::new(s2, p2, o2, GraphName::DefaultGraph))
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("dataset1".to_string(), store1);
    datasets.insert("dataset2".to_string(), store2);

    let server_stats = StatisticsCollector::collect_server_stats(&datasets, None).unwrap();

    assert_eq!(server_stats.dataset_count, 2);
    assert_eq!(server_stats.datasets.len(), 2);
    assert_eq!(server_stats.total_triples(), 2);
}

/// Test metadata storage
#[tokio::test]
async fn test_statistics_metadata() {
    let store = create_test_store_with_data();

    let stats = StatisticsCollector::collect_dataset_stats("test", &store).unwrap();

    assert!(stats.metadata.contains_key("store_type"));
    assert_eq!(stats.metadata.get("store_type"), Some(&"OxiRS".to_string()));
}

/// Test statistics timestamp
#[tokio::test]
async fn test_statistics_timestamp() {
    let store = create_test_store_with_data();

    let before = std::time::SystemTime::now();
    let stats = StatisticsCollector::collect_dataset_stats("test", &store).unwrap();
    let after = std::time::SystemTime::now();

    assert!(stats.collected_at >= before);
    assert!(stats.collected_at <= after);
}

/// Test human-readable storage size formatting
#[tokio::test]
async fn test_storage_size_human_readable() {
    use oxirs_fuseki::handlers::dataset_stats::DatasetStatistics;

    let mut stats = DatasetStatistics::new("test".to_string());

    stats.storage_size_bytes = None;
    assert_eq!(stats.storage_size_human(), "Unknown");

    stats.storage_size_bytes = Some(512);
    assert_eq!(stats.storage_size_human(), "512.00 B");

    stats.storage_size_bytes = Some(1024);
    assert_eq!(stats.storage_size_human(), "1.00 KB");

    stats.storage_size_bytes = Some(1048576);
    assert_eq!(stats.storage_size_human(), "1.00 MB");

    stats.storage_size_bytes = Some(1073741824);
    assert_eq!(stats.storage_size_human(), "1.00 GB");
}

/// Test total triples calculation
#[tokio::test]
async fn test_total_triples_calculation() {
    use oxirs_fuseki::handlers::dataset_stats::DatasetStatistics;

    let mut stats = DatasetStatistics::new("test".to_string());
    stats.triples_in_default_graph = 100;
    stats.triples_in_named_graphs = 200;

    assert_eq!(stats.total_triples(), 300);
}

/// Test server statistics aggregation
#[tokio::test]
async fn test_server_statistics_aggregation() {
    use oxirs_fuseki::handlers::dataset_stats::{DatasetStatistics, ServerStatistics};

    let mut server_stats = ServerStatistics::new();

    let mut ds1 = DatasetStatistics::new("ds1".to_string());
    ds1.triples_in_default_graph = 50;
    ds1.triples_in_named_graphs = 50;
    ds1.named_graph_count = 2;

    let mut ds2 = DatasetStatistics::new("ds2".to_string());
    ds2.triples_in_default_graph = 100;
    ds2.triples_in_named_graphs = 100;
    ds2.named_graph_count = 3;

    server_stats.datasets.push(ds1);
    server_stats.datasets.push(ds2);
    server_stats.dataset_count = 2;

    assert_eq!(server_stats.total_triples(), 300);
    assert_eq!(server_stats.total_named_graphs(), 5);
}

/// Test large dataset statistics
#[tokio::test]
async fn test_large_dataset_statistics() {
    let store = create_test_store_with_data();

    // Add 100 triples
    for i in 0..100 {
        let s = Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", i)).unwrap());
        let p = NamedNode::new("http://example.org/p").unwrap();
        let o = oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal(
            &format!("value{}", i),
        ));
        store
            .insert_quad(Quad::new(s, p, o, GraphName::DefaultGraph))
            .unwrap();
    }

    let stats = StatisticsCollector::collect_dataset_stats("test", &store).unwrap();

    assert_eq!(stats.triples_in_default_graph, 100);
    assert_eq!(stats.total_triples(), 100);
    assert_eq!(stats.storage_size_bytes, Some(20000)); // 100 * 200 bytes
}

/// Test concurrent statistics collection
#[tokio::test]
async fn test_concurrent_statistics_collection() {
    use tokio::task;

    let store = Arc::new(create_test_store_with_data());

    // Add some data
    let s = Subject::NamedNode(NamedNode::new("http://example.org/test").unwrap());
    let p = NamedNode::new("http://example.org/p").unwrap();
    let o =
        oxirs_core::model::Object::Literal(oxirs_core::model::Literal::new_simple_literal("value"));
    store
        .insert_quad(Quad::new(s, p, o, GraphName::DefaultGraph))
        .unwrap();

    let mut handles = vec![];

    // Collect statistics from multiple threads
    for _ in 0..10 {
        let store_clone = store.clone();
        let handle = task::spawn(async move {
            StatisticsCollector::collect_dataset_stats("test", &store_clone).unwrap()
        });
        handles.push(handle);
    }

    // All should succeed
    for handle in handles {
        let stats = handle.await.unwrap();
        assert_eq!(stats.triples_in_default_graph, 1);
    }
}
