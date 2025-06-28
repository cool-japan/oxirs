//! Concurrent data structures for high-performance graph operations
//!
//! This module provides lock-free and wait-free data structures for
//! concurrent access to RDF graphs, using epoch-based memory reclamation
//! and atomic operations.

pub mod batch_builder;
pub mod epoch;
pub mod lock_free_graph;
pub mod parallel_batch;

pub use batch_builder::{BatchBuilder, BatchBuilderConfig, BatchBuilderStats, CoalescingStrategy};
pub use epoch::{EpochManager, HazardPointer, VersionedPointer};
pub use lock_free_graph::{ConcurrentGraph, GraphStats};
pub use parallel_batch::{
    BatchConfig, BatchOperation, BatchStats, BatchStatsSummary, ParallelBatchProcessor,
    ProgressCallback,
};

/// Re-export crossbeam epoch types for convenience
pub use crossbeam_epoch::{pin, Guard};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Object, Predicate, Subject, Triple};
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    fn create_test_triple(id: usize) -> Triple {
        Triple::new(
            Subject::NamedNode(NamedNode::new(&format!("http://subject/{}", id)).unwrap()),
            Predicate::NamedNode(NamedNode::new(&format!("http://predicate/{}", id)).unwrap()),
            Object::NamedNode(NamedNode::new(&format!("http://object/{}", id)).unwrap()),
        )
    }

    #[test]
    fn test_concurrent_stress() {
        let graph = Arc::new(ConcurrentGraph::new());
        let num_threads = 8;
        let ops_per_thread = 1000;

        let start = Instant::now();

        // Spawn writer threads
        let writer_handles: Vec<_> = (0..num_threads / 2)
            .map(|thread_id| {
                let graph = graph.clone();
                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        let id = thread_id * ops_per_thread + i;
                        let triple = create_test_triple(id);
                        graph.insert(triple).unwrap();
                    }
                })
            })
            .collect();

        // Spawn reader threads
        let reader_handles: Vec<_> = (0..num_threads / 2)
            .map(|_| {
                let graph = graph.clone();
                thread::spawn(move || {
                    let mut read_count = 0;
                    for _ in 0..ops_per_thread {
                        let count = graph.len();
                        if count > 0 {
                            read_count += 1;
                        }
                        // Perform some pattern matching
                        let _ = graph.match_pattern(None, None, None);
                    }
                    read_count
                })
            })
            .collect();

        // Wait for writers
        for handle in writer_handles {
            handle.join().unwrap();
        }

        // Wait for readers
        let total_reads: usize = reader_handles.into_iter().map(|h| h.join().unwrap()).sum();

        let duration = start.elapsed();

        // Verify final state
        assert_eq!(graph.len(), (num_threads / 2) * ops_per_thread);

        println!("Concurrent stress test completed:");
        println!("  Duration: {:?}", duration);
        println!("  Total writes: {}", (num_threads / 2) * ops_per_thread);
        println!("  Total reads: {}", total_reads);
        println!("  Final graph size: {}", graph.len());
        println!("  Stats: {:?}", graph.stats());
    }

    #[test]
    fn test_memory_reclamation() {
        let graph = Arc::new(ConcurrentGraph::new());
        let num_cycles = 10;
        let triples_per_cycle = 1000;

        for cycle in 0..num_cycles {
            // Insert triples
            let triples: Vec<_> = (0..triples_per_cycle)
                .map(|i| create_test_triple(cycle * triples_per_cycle + i))
                .collect();

            graph.insert_batch(triples.clone()).unwrap();
            assert_eq!(graph.len(), triples_per_cycle);

            // Remove all triples
            graph.remove_batch(&triples).unwrap();
            assert_eq!(graph.len(), 0);

            // Force memory reclamation
            graph.collect();
        }

        // Final verification
        assert!(graph.is_empty());
    }

    #[test]
    fn test_concurrent_mixed_operations() {
        let graph = Arc::new(ConcurrentGraph::new());
        let num_threads = 6;
        let ops_per_thread = 500;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let graph = graph.clone();
                thread::spawn(move || {
                    let mut local_count = 0;
                    for i in 0..ops_per_thread {
                        let id = thread_id * ops_per_thread + i;
                        let triple = create_test_triple(id);

                        match i % 3 {
                            0 => {
                                // Insert
                                if graph.insert(triple).unwrap() {
                                    local_count += 1;
                                }
                            }
                            1 => {
                                // Query
                                let _ = graph.contains(&triple);
                                let _ = graph.match_pattern(Some(triple.subject()), None, None);
                            }
                            2 => {
                                // Remove (might fail if not inserted)
                                if graph.remove(&triple).unwrap() {
                                    local_count -= 1;
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                    local_count
                })
            })
            .collect();

        let total_net_insertions: i32 = handles.into_iter().map(|h| h.join().unwrap()).sum();

        println!("Mixed operations test:");
        println!("  Net insertions: {}", total_net_insertions);
        println!("  Final graph size: {}", graph.len());
        println!("  Stats: {:?}", graph.stats());

        // The graph size should be consistent with net insertions
        assert!(total_net_insertions >= 0);
    }

    #[test]
    fn test_epoch_progression() {
        let graph = Arc::new(ConcurrentGraph::new());
        let initial_stats = graph.stats();

        // Perform operations
        for i in 0..100 {
            let triple = create_test_triple(i);
            graph.insert(triple.clone()).unwrap();
            graph.remove(&triple).unwrap();
        }

        // Force collection multiple times
        for _ in 0..5 {
            graph.collect();
        }

        let final_stats = graph.stats();

        assert!(final_stats.operation_count > initial_stats.operation_count);
        assert!(final_stats.current_epoch > initial_stats.current_epoch);
        assert_eq!(final_stats.triple_count, 0);
    }
}
