//! Example demonstrating lock-free concurrent graph operations

use oxirs_core::concurrent::ConcurrentGraph;
use oxirs_core::model::{NamedNode, Object, Predicate, Subject, Triple};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn main() {
    println!("=== OxiRS Concurrent Graph Demo ===\n");

    // Create a concurrent graph
    let graph = Arc::new(ConcurrentGraph::new());

    demo_basic_operations(&graph);
    demo_concurrent_writes(&graph);
    demo_pattern_matching(&graph);
    demo_performance(&graph);
}

fn demo_basic_operations(graph: &Arc<ConcurrentGraph>) {
    println!("1. Basic Operations Demo");
    println!("------------------------");

    // Insert a triple
    let triple = Triple::new(
        Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap()),
        Predicate::NamedNode(NamedNode::new("http://example.org/knows").unwrap()),
        Object::NamedNode(NamedNode::new("http://example.org/bob").unwrap()),
    );

    match graph.insert(triple.clone()) {
        Ok(inserted) => println!("Triple inserted: {inserted}"),
        Err(e) => println!("Error inserting triple: {e}"),
    }

    // Check if triple exists
    let exists = graph.contains(&triple);
    println!("Triple exists: {exists}");
    let size = graph.len();
    println!("Graph size: {size}");

    // Remove the triple
    match graph.remove(&triple) {
        Ok(removed) => println!("Triple removed: {removed}"),
        Err(e) => println!("Error removing triple: {e}"),
    }

    let size = graph.len();
    println!("Graph size after removal: {size}\n");
}

fn demo_concurrent_writes(graph: &Arc<ConcurrentGraph>) {
    println!("2. Concurrent Writes Demo");
    println!("-------------------------");

    // Clear the graph first
    graph.clear().unwrap();

    let num_threads = 4;
    let triples_per_thread = 1000;

    let start = Instant::now();

    // Spawn multiple writer threads
    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let graph = graph.clone();
            thread::spawn(move || {
                for i in 0..triples_per_thread {
                    let triple = Triple::new(
                        Subject::NamedNode(
                            NamedNode::new(format!("http://thread{thread_id}/entity{i}")).unwrap(),
                        ),
                        Predicate::NamedNode(
                            NamedNode::new("http://example.org/property").unwrap(),
                        ),
                        Object::NamedNode(NamedNode::new(format!("http://value{i}")).unwrap()),
                    );
                    graph.insert(triple).unwrap();
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start.elapsed();

    let graph_size = graph.len();
    println!("Inserted {graph_size} triples from {num_threads} threads in {duration:?}");
    let throughput = (num_threads * triples_per_thread) as f64 / duration.as_secs_f64();
    println!("Throughput: {throughput:.2} triples/sec\n");
}

fn demo_pattern_matching(graph: &Arc<ConcurrentGraph>) {
    println!("3. Pattern Matching Demo");
    println!("------------------------");

    // Insert some test data
    graph.clear().unwrap();

    let subjects = ["alice", "bob", "charlie"];
    let predicates = ["knows", "likes", "follows"];

    for subj in &subjects {
        for pred in &predicates {
            for obj in &subjects {
                if subj != obj {
                    let triple = Triple::new(
                        Subject::NamedNode(
                            NamedNode::new(format!("http://example.org/{subj}")).unwrap(),
                        ),
                        Predicate::NamedNode(
                            NamedNode::new(format!("http://example.org/{pred}")).unwrap(),
                        ),
                        Object::NamedNode(
                            NamedNode::new(format!("http://example.org/{obj}")).unwrap(),
                        ),
                    );
                    graph.insert(triple).unwrap();
                }
            }
        }
    }

    println!("Total triples: {}", graph.len());

    // Query: Who does Alice know?
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let knows = Predicate::NamedNode(NamedNode::new("http://example.org/knows").unwrap());

    let matches = graph.match_pattern(Some(&alice), Some(&knows), None);
    println!("\nAlice knows:");
    for triple in matches {
        println!("  - {}", triple.object());
    }

    // Query: Who knows Bob?
    let bob = Object::NamedNode(NamedNode::new("http://example.org/bob").unwrap());
    let matches = graph.match_pattern(None, Some(&knows), Some(&bob));
    println!("\nWho knows Bob:");
    for triple in matches {
        println!("  - {}", triple.subject());
    }

    println!();
}

fn demo_performance(graph: &Arc<ConcurrentGraph>) {
    println!("4. Performance Comparison Demo");
    println!("------------------------------");

    graph.clear().unwrap();

    // Measure concurrent performance
    let num_operations = 10000;
    let num_threads = 4;

    // Concurrent writes
    let start = Instant::now();
    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let graph = graph.clone();
            let ops_per_thread = num_operations / num_threads;
            thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let triple = Triple::new(
                        Subject::NamedNode(
                            NamedNode::new(format!("http://s{}", thread_id * ops_per_thread + i))
                                .unwrap(),
                        ),
                        Predicate::NamedNode(NamedNode::new("http://p").unwrap()),
                        Object::NamedNode(NamedNode::new(format!("http://o{i}")).unwrap()),
                    );
                    graph.insert(triple).unwrap();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
    let concurrent_duration = start.elapsed();

    // Concurrent reads
    let start = Instant::now();
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let graph = graph.clone();
            thread::spawn(move || {
                let mut count = 0;
                for _ in 0..1000 {
                    let matches = graph.match_pattern(None, None, None);
                    count += matches.len();
                }
                count
            })
        })
        .collect();

    let total_reads: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let read_duration = start.elapsed();

    println!("Concurrent write performance:");
    println!(
        "  {} operations in {:?}",
        num_operations, concurrent_duration
    );
    println!(
        "  {:.2} operations/sec",
        num_operations as f64 / concurrent_duration.as_secs_f64()
    );

    println!("\nConcurrent read performance:");
    println!("  {} reads in {:?}", total_reads, read_duration);
    println!(
        "  {:.2} reads/sec",
        total_reads as f64 / read_duration.as_secs_f64()
    );

    // Show final statistics
    let stats = graph.stats();
    println!("\nGraph Statistics:");
    println!("  Triple count: {}", stats.triple_count);
    println!("  Total operations: {}", stats.operation_count);
    println!("  Current epoch: {}", stats.current_epoch);
}
