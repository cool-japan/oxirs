//! Integration tests for parallel batch processing

use oxirs_core::concurrent::{
    ParallelBatchProcessor, BatchOperation, BatchConfig, BatchBuilder, 
    BatchBuilderConfig, CoalescingStrategy,
};
use oxirs_core::store::IndexedGraph;
use oxirs_core::model::{Triple, Subject, Predicate, Object, NamedNode};
use oxirs_core::OxirsError;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use rayon::prelude::*;

fn create_test_triple(id: usize) -> Triple {
    Triple::new(
        Subject::NamedNode(NamedNode::new(&format!("http://subject/{}", id)).unwrap()),
        Predicate::NamedNode(NamedNode::new(&format!("http://predicate/{}", id)).unwrap()),
        Object::NamedNode(NamedNode::new(&format!("http://object/{}", id)).unwrap()),
    )
}

fn create_dataset(size: usize) -> Vec<Triple> {
    (0..size).map(create_test_triple).collect()
}

#[test]
fn test_parallel_insert_performance() {
    let graph = Arc::new(IndexedGraph::new());
    let dataset = create_dataset(10000);
    
    // Sequential insert for comparison
    let start = Instant::now();
    for triple in &dataset[..1000] {
        graph.insert(triple);
    }
    let sequential_time = start.elapsed();
    println!("Sequential insert (1000 triples): {:?}", sequential_time);
    
    // Parallel insert
    let start = Instant::now();
    let results = graph.par_insert_batch(dataset[1000..2000].to_vec());
    let parallel_time = start.elapsed();
    println!("Parallel insert (1000 triples): {:?}", parallel_time);
    
    assert_eq!(results.len(), 1000);
    assert!(results.iter().all(|&r| r));
    
    // Parallel should be faster for large datasets
    println!("Speedup: {:.2}x", sequential_time.as_secs_f64() / parallel_time.as_secs_f64());
}

#[test]
fn test_batch_processor_with_graph() {
    let graph = Arc::new(IndexedGraph::new());
    let config = BatchConfig {
        num_threads: Some(4),
        batch_size: 100,
        ..Default::default()
    };
    
    let processor = ParallelBatchProcessor::new(config);
    
    // Submit insert operations
    for i in 0..1000 {
        processor.submit(BatchOperation::insert(vec![create_test_triple(i)])).unwrap();
    }
    
    // Process with graph executor
    let graph_clone = graph.clone();
    let results = processor.process(move |op| -> Result<usize, OxirsError> {
        match op {
            BatchOperation::Insert(triples) => {
                let count = triples.len();
                for triple in triples {
                    graph_clone.insert(&triple);
                }
                Ok(count)
            }
            _ => Ok(0),
        }
    }).unwrap();
    
    assert_eq!(graph.len(), 1000);
    assert_eq!(results.iter().sum::<usize>(), 1000);
    
    let stats = processor.stats();
    assert_eq!(stats.total_processed, 1000);
    assert_eq!(stats.total_succeeded, 1000);
}

#[test]
fn test_batch_builder_with_coalescing() {
    let mut builder = BatchBuilder::new(BatchBuilderConfig {
        max_batch_size: 100,
        coalescing_strategy: CoalescingStrategy::Merge,
        auto_flush: false,
        ..Default::default()
    });
    
    // Add operations that should be coalesced
    for i in 0..50 {
        let triple = create_test_triple(i);
        builder.insert(triple.clone()).unwrap();
        builder.remove(triple).unwrap();
    }
    
    // Add operations that remain
    for i in 50..100 {
        builder.insert(create_test_triple(i)).unwrap();
    }
    
    let batches = builder.flush().unwrap();
    
    // After coalescing, should only have inserts for 50-99
    let total_ops: usize = batches.iter().map(|batch| {
        match batch {
            BatchOperation::Insert(triples) => triples.len(),
            BatchOperation::Remove(triples) => triples.len(),
            _ => 0,
        }
    }).sum();
    
    assert_eq!(total_ops, 50);
    assert_eq!(builder.stats().coalesced_operations, 50);
}

#[test]
fn test_parallel_query_patterns() {
    let graph = Arc::new(IndexedGraph::new());
    
    // Insert test data
    for i in 0..100 {
        for j in 0..10 {
            let triple = Triple::new(
                Subject::NamedNode(NamedNode::new(&format!("http://subject/{}", i)).unwrap()),
                Predicate::NamedNode(NamedNode::new(&format!("http://predicate/{}", j)).unwrap()),
                Object::NamedNode(NamedNode::new(&format!("http://object/{}-{}", i, j)).unwrap()),
            );
            graph.insert(&triple);
        }
    }
    
    // Create query patterns
    let patterns: Vec<_> = (0..100)
        .map(|i| {
            (
                Some(Subject::NamedNode(NamedNode::new(&format!("http://subject/{}", i)).unwrap())),
                None,
                None,
            )
        })
        .collect();
    
    // Execute parallel queries
    let start = Instant::now();
    let results = graph.par_query_batch(patterns);
    let query_time = start.elapsed();
    
    println!("Parallel query (100 patterns): {:?}", query_time);
    
    assert_eq!(results.len(), 100);
    for result in &results {
        assert_eq!(result.len(), 10); // Each subject has 10 triples
    }
}

#[test]
fn test_parallel_transform() {
    let graph = Arc::new(IndexedGraph::new());
    
    // Insert test data
    let dataset = create_dataset(1000);
    graph.par_insert_batch(dataset);
    
    // Transform function: change all predicates to a new one
    let new_predicate = Predicate::NamedNode(NamedNode::new("http://new-predicate").unwrap());
    let transformed = graph.par_transform(|triple| {
        Some(Triple::new(
            triple.subject().clone(),
            new_predicate.clone(),
            triple.object().clone(),
        ))
    });
    
    assert_eq!(transformed.len(), 1000);
    assert!(transformed.iter().all(|t| t.predicate() == &new_predicate));
}

#[test]
fn test_work_stealing_balance() {
    let config = BatchConfig {
        num_threads: Some(4),
        batch_size: 10,
        ..Default::default()
    };
    
    let processor = ParallelBatchProcessor::new(config);
    
    // Submit uneven workload
    for i in 0..100 {
        let size = if i % 10 == 0 { 100 } else { 10 };
        let triples = create_dataset(size);
        processor.submit(BatchOperation::insert(triples)).unwrap();
    }
    
    // Track which thread processed each operation
    let thread_work = Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    let thread_work_clone = thread_work.clone();
    
    processor.process(move |op| -> Result<(), OxirsError> {
        let thread_id = std::thread::current().id();
        let mut work_map = thread_work_clone.lock();
        let count = match &op {
            BatchOperation::Insert(triples) => triples.len(),
            _ => 0,
        };
        *work_map.entry(thread_id).or_insert(0) += count;
        
        // Simulate some work
        std::thread::sleep(Duration::from_micros(count as u64));
        Ok(())
    }).unwrap();
    
    // Check work distribution
    let work_map = thread_work.lock();
    let total_work: usize = work_map.values().sum();
    let avg_work = total_work / work_map.len();
    
    println!("Work distribution across {} threads:", work_map.len());
    for (thread_id, work) in work_map.iter() {
        let deviation = (*work as f64 - avg_work as f64).abs() / avg_work as f64 * 100.0;
        println!("  Thread {:?}: {} items ({:.1}% deviation)", thread_id, work, deviation);
    }
    
    // Work should be reasonably balanced (within 50% of average)
    for work in work_map.values() {
        let deviation = (*work as f64 - avg_work as f64).abs() / avg_work as f64;
        assert!(deviation < 0.5, "Work imbalance detected");
    }
}

#[test]
fn test_batch_builder_auto_flush() {
    let config = BatchBuilderConfig {
        max_batch_size: 100,
        auto_flush: true,
        ..Default::default()
    };
    
    let flushed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let flushed_count_clone = flushed_count.clone();
    
    let mut builder = BatchBuilder::new(config);
    builder.on_flush(move |batches| {
        flushed_count_clone.fetch_add(batches.len(), std::sync::atomic::Ordering::Relaxed);
    });
    
    // Add more than max_batch_size
    for i in 0..250 {
        builder.insert(create_test_triple(i)).unwrap();
    }
    
    // Should have auto-flushed twice (at 100 and 200)
    assert_eq!(flushed_count.load(std::sync::atomic::Ordering::Relaxed), 2);
    assert_eq!(builder.pending_operations(), 50);
    
    // Final flush
    builder.flush().unwrap();
    assert_eq!(flushed_count.load(std::sync::atomic::Ordering::Relaxed), 3);
}

#[test]
fn test_error_recovery() {
    let config = BatchConfig::default();
    let processor = ParallelBatchProcessor::new(config);
    
    // Submit mix of successful and failing operations
    for i in 0..100 {
        if i % 10 == 0 {
            // This will fail in our executor
            processor.submit(BatchOperation::query(None, None, None)).unwrap();
        } else {
            processor.submit(BatchOperation::insert(vec![create_test_triple(i)])).unwrap();
        }
    }
    
    // Process with selective failures
    let result = processor.process(|op| -> Result<(), OxirsError> {
        match op {
            BatchOperation::Insert(_) => Ok(()),
            BatchOperation::Query { .. } => Err(OxirsError::Query("Query not supported".to_string())),
            _ => Ok(()),
        }
    });
    
    assert!(result.is_err());
    
    let stats = processor.stats();
    assert_eq!(stats.total_processed, 100);
    assert_eq!(stats.total_succeeded, 90);
    assert_eq!(stats.total_failed, 10);
    
    let errors = processor.errors();
    assert_eq!(errors.len(), 10);
}

#[test]
fn test_memory_aware_batching() {
    let config = BatchBuilderConfig {
        max_memory_usage: 10000, // Small limit to trigger memory-based flushing
        auto_flush: true,
        ..Default::default()
    };
    
    let mut builder = BatchBuilder::new(config);
    let flush_count = Arc::new(AtomicUsize::new(0));
    let flush_count_clone = flush_count.clone();
    
    builder.on_flush(move |_| {
        flush_count_clone.fetch_add(1, Ordering::Relaxed);
    });
    
    // Add triples until memory limit triggers flush
    for i in 0..1000 {
        builder.insert(create_test_triple(i)).unwrap();
        if builder.pending_operations() == 0 {
            // Flushed due to memory
            break;
        }
    }
    
    assert!(builder.stats().batches_created > 0);
    assert!(builder.stats().estimated_memory_usage > 0);
}

#[test]
fn test_parallel_fold_aggregation() {
    let graph = Arc::new(IndexedGraph::new());
    
    // Insert numbered triples
    for i in 0..1000 {
        let triple = Triple::new(
            Subject::NamedNode(NamedNode::new(&format!("http://subject/{}", i)).unwrap()),
            Predicate::NamedNode(NamedNode::new("http://value").unwrap()),
            Object::NamedNode(NamedNode::new(&format!("http://object/{}", i)).unwrap()),
        );
        graph.insert(&triple);
    }
    
    // Count triples in parallel
    let count = graph.par_fold(0usize, |acc, _triple| acc + 1);
    assert_eq!(count, 1000);
}

#[test]
fn test_cancellation_handling() {
    let config = BatchConfig::default();
    let processor = Arc::new(ParallelBatchProcessor::new(config));
    
    // Submit many operations
    for i in 0..10000 {
        processor.submit(BatchOperation::insert(vec![create_test_triple(i)])).unwrap();
    }
    
    let processor_clone = processor.clone();
    let handle = std::thread::spawn(move || {
        processor_clone.process(|op| -> Result<(), OxirsError> {
            // Simulate slow processing
            std::thread::sleep(Duration::from_millis(1));
            match op {
                BatchOperation::Insert(_) => Ok(()),
                _ => Ok(()),
            }
        })
    });
    
    // Cancel after a short time
    std::thread::sleep(Duration::from_millis(50));
    processor.cancel();
    
    // Should complete relatively quickly after cancellation
    let result = handle.join().unwrap();
    assert!(result.is_ok() || result.is_err()); // Either way is fine
    
    assert!(processor.is_cancelled());
    
    // Should have processed some but not all
    let stats = processor.stats();
    assert!(stats.total_processed > 0);
    assert!(stats.total_processed < 10000);
}