//! Demo of the high-performance indexed graph implementation

use oxirs_core::model::{Triple, NamedNode, Literal};
use oxirs_core::store::IndexedGraph;
use std::time::Instant;

fn main() {
    println!("OxiRS IndexedGraph Demo");
    println!("======================\n");

    // Create a new indexed graph
    let graph = IndexedGraph::new();

    // Insert some triples
    println!("Inserting triples...");
    let triples = vec![
        Triple::new(
            NamedNode::new("http://example.org/alice").unwrap(),
            NamedNode::new("http://example.org/knows").unwrap(),
            NamedNode::new("http://example.org/bob").unwrap(),
        ),
        Triple::new(
            NamedNode::new("http://example.org/alice").unwrap(),
            NamedNode::new("http://example.org/age").unwrap(),
            Literal::new("30"),
        ),
        Triple::new(
            NamedNode::new("http://example.org/bob").unwrap(),
            NamedNode::new("http://example.org/knows").unwrap(),
            NamedNode::new("http://example.org/charlie").unwrap(),
        ),
        Triple::new(
            NamedNode::new("http://example.org/bob").unwrap(),
            NamedNode::new("http://example.org/age").unwrap(),
            Literal::new("25"),
        ),
    ];

    for triple in &triples {
        graph.insert(triple);
    }
    println!("Inserted {} triples\n", graph.len());

    // Query by subject
    println!("Query: Who does Alice know?");
    let alice = oxirs_core::model::Subject::NamedNode(
        NamedNode::new("http://example.org/alice").unwrap()
    );
    let knows = oxirs_core::model::Predicate::NamedNode(
        NamedNode::new("http://example.org/knows").unwrap()
    );
    
    let results = graph.query(Some(&alice), Some(&knows), None);
    for triple in results {
        println!("  {} knows {}", 
            triple.subject().to_string().trim_matches('<').trim_matches('>'),
            triple.object().to_string().trim_matches('"').trim_matches('<').trim_matches('>')
        );
    }

    // Performance test
    println!("\nPerformance Test:");
    println!("-----------------");
    
    // Create 10,000 triples
    let mut test_triples = Vec::new();
    for i in 0..100 {
        for j in 0..10 {
            for k in 0..10 {
                test_triples.push(Triple::new(
                    NamedNode::new(&format!("http://example.org/subject{}", i)).unwrap(),
                    NamedNode::new(&format!("http://example.org/predicate{}", j)).unwrap(),
                    Literal::new(&format!("object{}_{}", i, k)),
                ));
            }
        }
    }

    // Measure batch insert performance
    let start = Instant::now();
    graph.batch_insert(&test_triples);
    let duration = start.elapsed();
    
    println!("Batch inserted {} triples in {:?}", test_triples.len(), duration);
    println!("Rate: {:.2} triples/second", test_triples.len() as f64 / duration.as_secs_f64());

    // Measure query performance
    let subject = oxirs_core::model::Subject::NamedNode(
        NamedNode::new("http://example.org/subject50").unwrap()
    );
    
    let start = Instant::now();
    let results = graph.query(Some(&subject), None, None);
    let duration = start.elapsed();
    
    println!("\nQueried {} results in {:?}", results.len(), duration);

    // Show memory usage
    let usage = graph.memory_usage();
    println!("\nMemory Usage:");
    println!("  Term interner: {} KB", usage.term_interner_bytes / 1024);
    println!("  SPO index: {} KB", usage.spo_index_bytes / 1024);
    println!("  POS index: {} KB", usage.pos_index_bytes / 1024);
    println!("  OSP index: {} KB", usage.osp_index_bytes / 1024);
    println!("  Total: {} KB", usage.total_bytes() / 1024);
    println!("  Bytes per triple: {:.2}", usage.bytes_per_triple());

    // Show index statistics
    let stats = graph.index_stats();
    println!("\nIndex Statistics:");
    println!("  SPO lookups: {}", stats.spo_lookups);
    println!("  POS lookups: {}", stats.pos_lookups);
    println!("  OSP lookups: {}", stats.osp_lookups);
    println!("  Most used index: {:?}", stats.most_used_index());
}