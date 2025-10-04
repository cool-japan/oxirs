//! Comprehensive tests for the IndexedGraph implementation

use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple};
use oxirs_core::store::{IndexedGraph, TermInterner};
use std::sync::Arc;
use std::time::Instant;

/// Helper to create a test triple with named nodes
fn create_triple(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(
        NamedNode::new(s).unwrap(),
        NamedNode::new(p).unwrap(),
        Literal::new(o),
    )
}

/// Helper to create a test triple with a named node object
#[allow(dead_code)]
fn create_triple_nn(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(
        NamedNode::new(s).unwrap(),
        NamedNode::new(p).unwrap(),
        NamedNode::new(o).unwrap(),
    )
}

#[test]
fn test_empty_graph() {
    let graph = IndexedGraph::new();
    assert!(graph.is_empty());
    assert_eq!(graph.len(), 0);
    assert_eq!(graph.query(None, None, None).len(), 0);
}

#[test]
fn test_single_triple_operations() {
    let graph = IndexedGraph::new();
    let triple = create_triple("http://example.org/s", "http://example.org/p", "object");

    // Insert
    assert!(graph.insert(&triple));
    assert_eq!(graph.len(), 1);
    assert!(!graph.is_empty());

    // Duplicate insert should return false
    assert!(!graph.insert(&triple));
    assert_eq!(graph.len(), 1);

    // Query
    let results = graph.query(None, None, None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], triple);

    // Remove
    assert!(graph.remove(&triple));
    assert_eq!(graph.len(), 0);
    assert!(graph.is_empty());

    // Remove non-existent should return false
    assert!(!graph.remove(&triple));
}

#[test]
fn test_multiple_triples() {
    let graph = IndexedGraph::new();

    let triples = vec![
        create_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "Bob",
        ),
        create_triple("http://example.org/alice", "http://example.org/age", "30"),
        create_triple(
            "http://example.org/bob",
            "http://example.org/knows",
            "Charlie",
        ),
        create_triple("http://example.org/bob", "http://example.org/age", "25"),
    ];

    for triple in &triples {
        assert!(graph.insert(triple));
    }

    assert_eq!(graph.len(), 4);

    // Query all
    let all = graph.query(None, None, None);
    assert_eq!(all.len(), 4);
}

#[test]
fn test_subject_queries() {
    let graph = IndexedGraph::new();

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let bob = Subject::NamedNode(NamedNode::new("http://example.org/bob").unwrap());

    graph.insert(&create_triple(
        "http://example.org/alice",
        "http://example.org/p1",
        "o1",
    ));
    graph.insert(&create_triple(
        "http://example.org/alice",
        "http://example.org/p2",
        "o2",
    ));
    graph.insert(&create_triple(
        "http://example.org/bob",
        "http://example.org/p1",
        "o3",
    ));

    // Query by subject
    let alice_triples = graph.query(Some(&alice), None, None);
    assert_eq!(alice_triples.len(), 2);

    let bob_triples = graph.query(Some(&bob), None, None);
    assert_eq!(bob_triples.len(), 1);
}

#[test]
fn test_predicate_queries() {
    let graph = IndexedGraph::new();

    let p1 = Predicate::NamedNode(NamedNode::new("http://example.org/p1").unwrap());
    let p2 = Predicate::NamedNode(NamedNode::new("http://example.org/p2").unwrap());

    graph.insert(&create_triple(
        "http://example.org/s1",
        "http://example.org/p1",
        "o1",
    ));
    graph.insert(&create_triple(
        "http://example.org/s2",
        "http://example.org/p1",
        "o2",
    ));
    graph.insert(&create_triple(
        "http://example.org/s3",
        "http://example.org/p2",
        "o3",
    ));

    let p1_triples = graph.query(None, Some(&p1), None);
    assert_eq!(p1_triples.len(), 2);

    let p2_triples = graph.query(None, Some(&p2), None);
    assert_eq!(p2_triples.len(), 1);
}

#[test]
fn test_object_queries() {
    let graph = IndexedGraph::new();

    let o1 = Object::Literal(Literal::new("object1"));
    let o2 = Object::Literal(Literal::new("object2"));

    graph.insert(&create_triple(
        "http://example.org/s1",
        "http://example.org/p1",
        "object1",
    ));
    graph.insert(&create_triple(
        "http://example.org/s2",
        "http://example.org/p2",
        "object1",
    ));
    graph.insert(&create_triple(
        "http://example.org/s3",
        "http://example.org/p3",
        "object2",
    ));

    let o1_triples = graph.query(None, None, Some(&o1));
    assert_eq!(o1_triples.len(), 2);

    let o2_triples = graph.query(None, None, Some(&o2));
    assert_eq!(o2_triples.len(), 1);
}

#[test]
fn test_compound_queries() {
    let graph = IndexedGraph::new();

    // Create a small knowledge graph
    graph.insert(&create_triple(
        "http://example.org/alice",
        "http://example.org/knows",
        "Bob",
    ));
    graph.insert(&create_triple(
        "http://example.org/alice",
        "http://example.org/age",
        "30",
    ));
    graph.insert(&create_triple(
        "http://example.org/bob",
        "http://example.org/knows",
        "Charlie",
    ));
    graph.insert(&create_triple(
        "http://example.org/bob",
        "http://example.org/age",
        "25",
    ));

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let knows = Predicate::NamedNode(NamedNode::new("http://example.org/knows").unwrap());
    let age = Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap());

    // SP query
    let alice_knows = graph.query(Some(&alice), Some(&knows), None);
    assert_eq!(alice_knows.len(), 1);
    assert_eq!(alice_knows[0].object().to_string(), "\"Bob\"");

    // PO query
    let age_30 = Object::Literal(Literal::new("30"));
    let who_is_30 = graph.query(None, Some(&age), Some(&age_30));
    assert_eq!(who_is_30.len(), 1);
    assert_eq!(
        who_is_30[0].subject().to_string(),
        "<http://example.org/alice>"
    );
}

#[test]
fn test_blank_nodes() {
    let graph = IndexedGraph::new();

    let blank1 = Subject::BlankNode(BlankNode::new("b1").unwrap());
    let blank2 = Object::BlankNode(BlankNode::new("b2").unwrap());
    let pred = Predicate::NamedNode(NamedNode::new("http://example.org/p").unwrap());

    let triple = Triple::new(blank1.clone(), pred.clone(), blank2.clone());

    assert!(graph.insert(&triple));

    // Query by blank node subject
    let results = graph.query(Some(&blank1), None, None);
    assert_eq!(results.len(), 1);

    // Query by blank node object
    let results = graph.query(None, None, Some(&blank2));
    assert_eq!(results.len(), 1);
}

#[test]
fn test_typed_literals() {
    let graph = IndexedGraph::new();

    let integer_type = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap();
    let typed_literal = Object::Literal(Literal::new_typed("42", integer_type));

    let triple = Triple::new(
        NamedNode::new("http://example.org/s").unwrap(),
        NamedNode::new("http://example.org/value").unwrap(),
        typed_literal.clone(),
    );

    assert!(graph.insert(&triple));

    let results = graph.query(None, None, Some(&typed_literal));
    assert_eq!(results.len(), 1);
}

#[test]
fn test_language_tagged_literals() {
    let graph = IndexedGraph::new();

    let lang_literal =
        Object::Literal(Literal::new_language_tagged_literal("Hello", "en").unwrap());

    let triple = Triple::new(
        NamedNode::new("http://example.org/s").unwrap(),
        NamedNode::new("http://example.org/label").unwrap(),
        lang_literal.clone(),
    );

    assert!(graph.insert(&triple));

    let results = graph.query(None, None, Some(&lang_literal));
    assert_eq!(results.len(), 1);
}

#[test]
fn test_batch_insert_performance() {
    let graph = IndexedGraph::new();

    // Create 1000 triples
    let triples: Vec<Triple> = (0..1000)
        .map(|i| {
            create_triple(
                &format!("http://example.org/s{}", i),
                &format!("http://example.org/p{}", i % 10),
                &format!("object{}", i),
            )
        })
        .collect();

    let start = Instant::now();
    let results = graph.batch_insert(&triples);
    let duration = start.elapsed();

    assert_eq!(results.len(), 1000);
    assert!(results.iter().all(|&r| r));
    assert_eq!(graph.len(), 1000);

    println!("Batch insert 1000 triples: {:?}", duration);
}

#[test]
fn test_query_performance() {
    let graph = IndexedGraph::new();

    // Insert test data
    for i in 0..100 {
        for j in 0..10 {
            let triple = create_triple(
                &format!("http://example.org/s{}", i),
                &format!("http://example.org/p{}", j),
                &format!("object{}_{}", i, j),
            );
            graph.insert(&triple);
        }
    }

    assert_eq!(graph.len(), 1000);

    // Test different query patterns
    let s = Subject::NamedNode(NamedNode::new("http://example.org/s50").unwrap());
    let p = Predicate::NamedNode(NamedNode::new("http://example.org/p5").unwrap());

    let start = Instant::now();
    let results = graph.query(Some(&s), None, None);
    let duration = start.elapsed();
    assert_eq!(results.len(), 10);
    println!("Query by subject (1000 triples): {:?}", duration);

    let start = Instant::now();
    let results = graph.query(None, Some(&p), None);
    let duration = start.elapsed();
    assert_eq!(results.len(), 100);
    println!("Query by predicate (1000 triples): {:?}", duration);
}

#[test]
fn test_memory_usage() {
    let graph = IndexedGraph::new();

    // Insert some triples
    for i in 0..100 {
        let triple = create_triple(
            &format!("http://example.org/subject{}", i),
            "http://example.org/predicate",
            &format!("This is a longer object value for triple {}", i),
        );
        graph.insert(&triple);
    }

    let usage = graph.memory_usage();
    println!("Memory usage for 100 triples:");
    println!("  Term interner: {} bytes", usage.term_interner_bytes);
    println!("  SPO index: {} bytes", usage.spo_index_bytes);
    println!("  POS index: {} bytes", usage.pos_index_bytes);
    println!("  OSP index: {} bytes", usage.osp_index_bytes);
    println!("  Total: {} bytes", usage.total_bytes());
    println!("  Bytes per triple: {:.2}", usage.bytes_per_triple());

    assert!(usage.total_bytes() > 0);
    assert!(usage.bytes_per_triple() > 0.0);
}

#[test]
fn test_index_stats() {
    let graph = IndexedGraph::new();

    // Insert test data
    for i in 0..10 {
        let triple = create_triple(
            &format!("http://example.org/s{}", i),
            "http://example.org/p",
            &format!("o{}", i),
        );
        graph.insert(&triple);
    }

    // Perform various queries to update stats
    let s = Subject::NamedNode(NamedNode::new("http://example.org/s5").unwrap());
    let p = Predicate::NamedNode(NamedNode::new("http://example.org/p").unwrap());
    let o = Object::Literal(Literal::new("o5"));

    graph.query(Some(&s), None, None); // SPO index
    graph.query(None, Some(&p), None); // POS index
    graph.query(None, None, Some(&o)); // OSP index

    let stats = graph.index_stats();
    assert_eq!(stats.spo_lookups, 1);
    assert_eq!(stats.pos_lookups, 1);
    assert_eq!(stats.osp_lookups, 1);
    assert_eq!(stats.total_insertions, 10);
}

#[test]
fn test_clear_and_reuse() {
    let graph = IndexedGraph::new();

    // Insert and clear
    graph.insert(&create_triple(
        "http://example.org/s",
        "http://example.org/p",
        "o",
    ));
    assert_eq!(graph.len(), 1);

    graph.clear();
    assert_eq!(graph.len(), 0);
    assert!(graph.is_empty());

    // Reuse after clear
    graph.insert(&create_triple(
        "http://example.org/s2",
        "http://example.org/p2",
        "o2",
    ));
    assert_eq!(graph.len(), 1);
}

#[test]
fn test_shared_interner() {
    let interner = Arc::new(TermInterner::new());
    let graph1 = IndexedGraph::with_interner(Arc::clone(&interner));
    let graph2 = IndexedGraph::with_interner(Arc::clone(&interner));

    // Insert in graph1
    graph1.insert(&create_triple(
        "http://example.org/s",
        "http://example.org/p",
        "o",
    ));

    // Insert in graph2
    graph2.insert(&create_triple(
        "http://example.org/s",
        "http://example.org/p2",
        "o2",
    ));

    // Both graphs should share the same interner
    let stats = interner.stats();
    assert_eq!(stats.subject_count, 1); // Same subject in both
    assert_eq!(stats.predicate_count, 2); // Different predicates
    assert_eq!(stats.object_count, 2); // Different objects
}

#[test]
fn test_concurrent_operations() {
    use std::thread;

    let graph = Arc::new(IndexedGraph::new());
    let mut handles = vec![];

    // Concurrent insertions
    for i in 0..10 {
        let graph_clone = Arc::clone(&graph);
        let handle = thread::spawn(move || {
            for j in 0..10 {
                let triple = create_triple(
                    &format!("http://example.org/thread{}/s{}", i, j),
                    &format!("http://example.org/p{}", j),
                    &format!("object_{}_{}", i, j),
                );
                graph_clone.insert(&triple);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(graph.len(), 100);

    // Concurrent queries
    let mut handles = vec![];
    for i in 0..10 {
        let graph_clone = Arc::clone(&graph);
        let handle = thread::spawn(move || {
            let p = Predicate::NamedNode(
                NamedNode::new(&format!("http://example.org/p{}", i)).unwrap(),
            );
            graph_clone.query(None, Some(&p), None).len()
        });
        handles.push(handle);
    }

    let results: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(results.iter().all(|&count| count == 10));
}
