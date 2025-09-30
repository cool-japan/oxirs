//! Integration tests for concurrent graph operations

use oxirs_core::concurrent::ConcurrentGraph;
use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

fn create_triple_with_literal(s: &str, p: &str, value: &str) -> Triple {
    Triple::new(
        Subject::NamedNode(NamedNode::new(s).unwrap()),
        Predicate::NamedNode(NamedNode::new(p).unwrap()),
        Object::Literal(Literal::new_simple_literal(value)),
    )
}

fn create_triple_with_blank(id: usize, p: &str, o: &str) -> Triple {
    Triple::new(
        Subject::BlankNode(BlankNode::new(format!("b{id}")).unwrap()),
        Predicate::NamedNode(NamedNode::new(p).unwrap()),
        Object::NamedNode(NamedNode::new(o).unwrap()),
    )
}

#[test]
fn test_concurrent_readers_writers() {
    let graph = Arc::new(ConcurrentGraph::new());
    let num_writers = 3;
    let num_readers = 5;
    let ops_per_thread = 200;
    let barrier = Arc::new(Barrier::new(num_writers + num_readers));

    // Spawn writer threads
    let writer_handles: Vec<_> = (0..num_writers)
        .map(|writer_id| {
            let graph = graph.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for i in 0..ops_per_thread {
                    let triple = create_triple_with_literal(
                        &format!("http://writer{writer_id}/subject{i}"),
                        "http://predicate",
                        &format!("value_{i}"),
                    );
                    graph.insert(triple).unwrap();

                    // Occasionally remove some triples
                    if i % 10 == 0 && i > 0 {
                        let old_triple = create_triple_with_literal(
                            &format!("http://writer{writer_id}/subject{}", i - 10),
                            "http://predicate",
                            &format!("value_{}", i - 10),
                        );
                        graph.remove(&old_triple).ok();
                    }
                }
            })
        })
        .collect();

    // Spawn reader threads
    let reader_handles: Vec<_> = (0..num_readers)
        .map(|_| {
            let graph = graph.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                let mut total_reads = 0;
                let mut total_found = 0;

                for _ in 0..ops_per_thread * 2 {
                    // Various read operations
                    let size = graph.len();
                    total_reads += 1;

                    if size > 0 {
                        total_found += 1;
                    }

                    // Pattern matching
                    let pred = Predicate::NamedNode(NamedNode::new("http://predicate").unwrap());
                    let matches = graph.match_pattern(None, Some(&pred), None);
                    total_found += matches.len();

                    // Check specific triple
                    let check_triple = create_triple_with_literal(
                        "http://writer0/subject50",
                        "http://predicate",
                        "value_50",
                    );
                    if graph.contains(&check_triple) {
                        total_found += 1;
                    }

                    thread::yield_now();
                }

                (total_reads, total_found)
            })
        })
        .collect();

    // Wait for all threads
    for handle in writer_handles {
        handle.join().unwrap();
    }

    let reader_results: Vec<_> = reader_handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();

    // Verify results
    let total_reads: usize = reader_results.iter().map(|(r, _)| r).sum();
    let total_found: usize = reader_results.iter().map(|(_, f)| f).sum();

    println!("Concurrent test results:");
    println!("  Final graph size: {}", graph.len());
    println!("  Total reads: {total_reads}");
    println!("  Total items found: {total_found}");
    println!("  Graph stats: {:?}", graph.stats());

    // The graph should have some triples
    assert!(!graph.is_empty());
    assert!(total_found > 0);
}

#[test]
fn test_stress_with_mixed_types() {
    let graph = Arc::new(ConcurrentGraph::new());
    let num_threads = 6;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let graph = graph.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();

                for i in 0..100 {
                    // Mix of different triple types
                    match i % 3 {
                        0 => {
                            // Named nodes
                            let triple = Triple::new(
                                Subject::NamedNode(
                                    NamedNode::new(format!("http://thread{thread_id}/s{i}"))
                                        .unwrap(),
                                ),
                                Predicate::NamedNode(NamedNode::new("http://pred").unwrap()),
                                Object::NamedNode(
                                    NamedNode::new(format!("http://obj{i}")).unwrap(),
                                ),
                            );
                            graph.insert(triple).unwrap();
                        }
                        1 => {
                            // Blank nodes
                            let triple = create_triple_with_blank(
                                thread_id * 100 + i,
                                "http://blank-pred",
                                &format!("http://blank-obj{i}"),
                            );
                            graph.insert(triple).unwrap();
                        }
                        2 => {
                            // Literals
                            let triple = create_triple_with_literal(
                                &format!("http://lit-subject{}", thread_id * 100 + i),
                                "http://has-value",
                                &format!("String value {i}"),
                            );
                            graph.insert(triple).unwrap();
                        }
                        _ => unreachable!(),
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(graph.len(), num_threads * 100);

    // Test pattern matching with different types
    let literal_matches = graph
        .iter()
        .filter(|t| matches!(t.object(), Object::Literal(_)))
        .count();
    let blank_matches = graph
        .iter()
        .filter(|t| matches!(t.subject(), Subject::BlankNode(_)))
        .count();

    println!("Mixed types test:");
    println!("  Total triples: {}", graph.len());
    println!("  Literal objects: {literal_matches}");
    println!("  Blank subjects: {blank_matches}");

    assert!(literal_matches > 0);
    assert!(blank_matches > 0);
}

#[test]
fn test_memory_safety_under_pressure() {
    let graph = Arc::new(ConcurrentGraph::new());
    let duration = Duration::from_secs(2);
    let start = std::time::Instant::now();

    // Create a high-pressure scenario
    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            let graph = graph.clone();
            thread::spawn(move || {
                let mut local_ops = 0;
                while start.elapsed() < duration {
                    for i in 0..10 {
                        let id = thread_id * 1000000 + local_ops * 10 + i;
                        let triple = Triple::new(
                            Subject::NamedNode(NamedNode::new(format!("http://s{id}")).unwrap()),
                            Predicate::NamedNode(NamedNode::new("http://p").unwrap()),
                            Object::Literal(Literal::new_simple_literal(format!("{id}"))),
                        );

                        // Rapid insert/remove cycles
                        graph.insert(triple.clone()).unwrap();
                        if local_ops % 2 == 0 {
                            graph.remove(&triple).ok();
                        }
                    }

                    // Force collection periodically
                    if local_ops % 100 == 0 {
                        graph.collect();
                    }

                    local_ops += 1;
                }
                local_ops
            })
        })
        .collect();

    let total_ops: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    println!("Memory pressure test:");
    println!("  Total operations: {total_ops}");
    println!("  Final size: {}", graph.len());
    println!("  Stats: {:?}", graph.stats());

    // Graph should still be functional
    let _ = graph.match_pattern(None, None, None);
    graph.clear().unwrap();
    assert_eq!(graph.len(), 0);
}

#[test]
fn test_epoch_based_cleanup() {
    let graph = Arc::new(ConcurrentGraph::new());

    // Phase 1: Insert data
    let triples: Vec<_> = (0..1000)
        .map(|i| {
            Triple::new(
                Subject::NamedNode(NamedNode::new(format!("http://s{i}")).unwrap()),
                Predicate::NamedNode(NamedNode::new("http://ephemeral").unwrap()),
                Object::Literal(Literal::new_simple_literal(format!("temp_{i}"))),
            )
        })
        .collect();

    graph.insert_batch(triples.clone()).unwrap();
    assert_eq!(graph.len(), 1000);

    // Phase 2: Remove all data
    graph.remove_batch(&triples).unwrap();
    assert_eq!(graph.len(), 0);

    // Phase 3: Force multiple collection cycles
    for _ in 0..10 {
        graph.collect();
        thread::sleep(Duration::from_millis(10));
    }

    // Phase 4: Verify graph is still usable
    let new_triple = Triple::new(
        Subject::NamedNode(NamedNode::new("http://new").unwrap()),
        Predicate::NamedNode(NamedNode::new("http://pred").unwrap()),
        Object::Literal(Literal::new_simple_literal("value")),
    );

    assert!(graph.insert(new_triple.clone()).unwrap());
    assert!(graph.contains(&new_triple));
    assert_eq!(graph.len(), 1);
}

#[test]
fn test_concurrent_batch_operations() {
    let graph = Arc::new(ConcurrentGraph::new());
    let num_threads = 4;
    let batches_per_thread = 10;
    let batch_size = 50;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let graph = graph.clone();
            thread::spawn(move || {
                let mut inserted = 0;
                let mut removed = 0;

                for batch_id in 0..batches_per_thread {
                    let base_id = thread_id * 10000 + batch_id * batch_size;

                    // Create batch
                    let batch: Vec<_> = (0..batch_size)
                        .map(|i| {
                            Triple::new(
                                Subject::NamedNode(
                                    NamedNode::new(format!("http://batch/s{}", base_id + i))
                                        .unwrap(),
                                ),
                                Predicate::NamedNode(NamedNode::new("http://batch/pred").unwrap()),
                                Object::Literal(Literal::new_simple_literal(format!(
                                    "batch_{}",
                                    base_id + i
                                ))),
                            )
                        })
                        .collect();

                    // Insert batch
                    inserted += graph.insert_batch(batch.clone()).unwrap();

                    // Remove half of the batch
                    if batch_id % 2 == 0 {
                        let to_remove = &batch[0..batch_size / 2];
                        removed += graph.remove_batch(to_remove).unwrap();
                    }
                }

                (inserted, removed)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let total_inserted: usize = results.iter().map(|(i, _)| i).sum();
    let total_removed: usize = results.iter().map(|(_, r)| r).sum();

    println!("Batch operations test:");
    println!("  Total inserted: {total_inserted}");
    println!("  Total removed: {total_removed}");
    println!("  Final size: {}", graph.len());

    assert_eq!(graph.len(), total_inserted - total_removed);
}
