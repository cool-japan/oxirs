//! Integration tests for adaptive indexing

use oxirs_core::model::*;
use oxirs_core::store::{AdaptiveConfig, AdaptiveIndexManager, IndexedGraph, QueryPattern};
use std::time::Duration;

#[test]
fn test_adaptive_index_basic() {
    // Create manager with low thresholds for testing
    let config = AdaptiveConfig {
        min_queries_for_index: 2,
        min_frequency_for_index: 0.01,
        maintenance_interval: Duration::from_millis(10),
        ..Default::default()
    };

    let graph = IndexedGraph::new();
    let manager = AdaptiveIndexManager::new(graph, config);

    // Insert test data
    for i in 0..10 {
        let triple = Triple::new(
            NamedNode::new(format!("http://example.org/s{i}")).unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new(format!("value{i}")),
        );
        manager.insert(triple).unwrap();
    }

    // Query multiple times to trigger index creation
    let pred = Predicate::NamedNode(NamedNode::new("http://example.org/p").unwrap());
    for _ in 0..3 {
        let results = manager.query(None, Some(&pred), None).unwrap();
        assert_eq!(results.len(), 10);
    }

    // Check statistics
    let stats = manager.get_stats();
    assert!(stats.total_queries >= 3);
}

#[test]
fn test_query_pattern_detection() {
    let config = AdaptiveConfig::default();
    let graph = IndexedGraph::new();
    let manager = AdaptiveIndexManager::new(graph, config);

    // Insert diverse data
    let predicates = vec!["name", "age", "email", "city"];
    for i in 0..20 {
        for pred in &predicates {
            let triple = Triple::new(
                NamedNode::new(&format!("http://example.org/person{}", i)).unwrap(),
                NamedNode::new(&format!("http://example.org/{}", pred)).unwrap(),
                Literal::new(&format!("{} value {}", pred, i)),
            );
            manager.insert(triple).unwrap();
        }
    }

    // Test different query patterns
    let patterns_tested = vec![
        // Full scan
        (None, None, None, QueryPattern::FullScan),
        // Subject query
        (
            Some(Subject::NamedNode(
                NamedNode::new("http://example.org/person1").unwrap(),
            )),
            None,
            None,
            QueryPattern::SubjectQuery,
        ),
        // Predicate query
        (
            None,
            Some(Predicate::NamedNode(
                NamedNode::new("http://example.org/name").unwrap(),
            )),
            None,
            QueryPattern::PredicateQuery,
        ),
        // Subject-Predicate query
        (
            Some(Subject::NamedNode(
                NamedNode::new("http://example.org/person1").unwrap(),
            )),
            Some(Predicate::NamedNode(
                NamedNode::new("http://example.org/name").unwrap(),
            )),
            None,
            QueryPattern::SubjectPredicate,
        ),
    ];

    for (subject, predicate, object, expected_pattern) in patterns_tested {
        let results = manager
            .query(subject.as_ref(), predicate.as_ref(), object.as_ref())
            .unwrap();
        assert!(!results.is_empty() || matches!(expected_pattern, QueryPattern::FullScan));

        // Verify pattern detection
        let detected =
            QueryPattern::from_components(subject.as_ref(), predicate.as_ref(), object.as_ref());
        assert_eq!(detected, expected_pattern);
    }
}

#[test]
fn test_adaptive_index_updates() {
    let config = AdaptiveConfig {
        min_queries_for_index: 2,
        min_frequency_for_index: 0.01,
        ..Default::default()
    };

    let graph = IndexedGraph::new();
    let manager = AdaptiveIndexManager::new(graph, config);

    // Insert initial data
    let pred = NamedNode::new("http://example.org/type").unwrap();
    for i in 0..5 {
        let triple = Triple::new(
            NamedNode::new(&format!("http://example.org/resource{}", i)).unwrap(),
            pred.clone(),
            NamedNode::new("http://example.org/Document").unwrap(),
        );
        manager.insert(triple).unwrap();
    }

    // Query to potentially create index
    let pred_term = Predicate::NamedNode(pred.clone());
    for _ in 0..3 {
        manager.query(None, Some(&pred_term), None).unwrap();
    }

    // Insert more data
    let new_triple = Triple::new(
        NamedNode::new("http://example.org/resource99").unwrap(),
        pred.clone(),
        NamedNode::new("http://example.org/Document").unwrap(),
    );
    assert!(manager.insert(new_triple.clone()).unwrap());

    // Verify new data is found
    let results = manager.query(None, Some(&pred_term), None).unwrap();
    assert_eq!(results.len(), 6);

    // Remove data
    assert!(manager.remove(&new_triple).unwrap());
    let results = manager.query(None, Some(&pred_term), None).unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn test_concurrent_adaptive_queries() {
    use std::sync::Arc;
    use std::thread;

    let config = AdaptiveConfig::default();
    let graph = IndexedGraph::new();
    let manager = Arc::new(AdaptiveIndexManager::new(graph, config));

    // Insert data
    for i in 0..100 {
        let triple = Triple::new(
            NamedNode::new(&format!("http://example.org/s{}", i)).unwrap(),
            NamedNode::new(&format!("http://example.org/p{}", i % 10)).unwrap(),
            Literal::new(&format!("value{}", i)),
        );
        manager.insert(triple).unwrap();
    }

    // Concurrent queries
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let manager = manager.clone();
            thread::spawn(move || {
                let pred = Predicate::NamedNode(
                    NamedNode::new(&format!("http://example.org/p{}", thread_id)).unwrap(),
                );

                for _ in 0..10 {
                    let results = manager.query(None, Some(&pred), None).unwrap();
                    assert_eq!(results.len(), 10);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify statistics
    let stats = manager.get_stats();
    assert_eq!(stats.total_queries, 40);
}

#[test]
fn test_index_benefit_estimation() {
    let config = AdaptiveConfig {
        min_queries_for_index: 5,
        index_cost_threshold: 0.5, // Create index if it reduces cost by 50%
        ..Default::default()
    };

    let graph = IndexedGraph::new();
    let manager = AdaptiveIndexManager::new(graph, config);

    // Insert 1000 triples with skewed distribution
    // 90% have predicate p1, 10% have predicate p2
    for i in 0..1000 {
        let pred = if i % 10 == 0 { "p2" } else { "p1" };
        let triple = Triple::new(
            NamedNode::new(&format!("http://example.org/s{}", i)).unwrap(),
            NamedNode::new(&format!("http://example.org/{}", pred)).unwrap(),
            Literal::new(&format!("value{}", i)),
        );
        manager.insert(triple).unwrap();
    }

    // Query p1 frequently (should benefit from index)
    let p1 = Predicate::NamedNode(NamedNode::new("http://example.org/p1").unwrap());
    for _ in 0..10 {
        let results = manager.query(None, Some(&p1), None).unwrap();
        assert_eq!(results.len(), 900);
    }

    // Query p2 less frequently (less benefit from index)
    let p2 = Predicate::NamedNode(NamedNode::new("http://example.org/p2").unwrap());
    for _ in 0..10 {
        let results = manager.query(None, Some(&p2), None).unwrap();
        assert_eq!(results.len(), 100);
    }

    // Force maintenance
    manager.run_maintenance();

    // Check that appropriate indexes were created
    let stats = manager.get_stats();
    println!("Created indexes: {:?}", stats.active_indexes);

    // Both patterns should be tracked
    assert!(!stats.pattern_stats.is_empty());
}
