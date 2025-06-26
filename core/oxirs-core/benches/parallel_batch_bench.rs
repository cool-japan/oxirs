//! Benchmarks for parallel batch processing

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use oxirs_core::concurrent::{
    ParallelBatchProcessor, BatchOperation, BatchConfig, BatchBuilder,
    BatchBuilderConfig, CoalescingStrategy,
};
use oxirs_core::store::IndexedGraph;
use oxirs_core::graph::Graph;
use oxirs_core::model::{Triple, Subject, Predicate, Object, NamedNode};
use std::sync::Arc;
use rayon::prelude::*;

fn create_test_triple(id: usize) -> Triple {
    Triple::new(
        Subject::NamedNode(NamedNode::new(&format!("http://s/{}", id)).unwrap()),
        Predicate::NamedNode(NamedNode::new(&format!("http://p/{}", id % 100)).unwrap()),
        Object::NamedNode(NamedNode::new(&format!("http://o/{}", id)).unwrap()),
    )
}

fn create_dataset(size: usize) -> Vec<Triple> {
    (0..size).map(create_test_triple).collect()
}

fn bench_parallel_vs_sequential_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_comparison");
    
    for size in [1000, 10000, 100000] {
        let dataset = create_dataset(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let graph = IndexedGraph::new();
                    for triple in dataset {
                        graph.insert(black_box(triple));
                    }
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let graph = IndexedGraph::new();
                    graph.par_insert_batch(black_box(dataset.clone()));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("batch_processor", size),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let graph = Arc::new(IndexedGraph::new());
                    let processor = ParallelBatchProcessor::new(BatchConfig::default());
                    
                    // Submit in batches
                    for chunk in dataset.chunks(1000) {
                        processor.submit(BatchOperation::insert(chunk.to_vec())).unwrap();
                    }
                    
                    let graph_clone = graph.clone();
                    processor.process(move |op| match op {
                        BatchOperation::Insert(triples) => {
                            for triple in triples {
                                graph_clone.insert(&triple);
                            }
                            Ok(())
                        }
                        _ => Ok(()),
                    }).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_builder_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_builder_strategies");
    let dataset = create_dataset(10000);
    
    for strategy in [
        CoalescingStrategy::None,
        CoalescingStrategy::Deduplicate,
        CoalescingStrategy::Merge,
        CoalescingStrategy::OptimizeOrder,
    ] {
        group.bench_with_input(
            BenchmarkId::new("strategy", format!("{:?}", strategy)),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let mut builder = BatchBuilder::new(BatchBuilderConfig {
                        coalescing_strategy: strategy,
                        auto_flush: false,
                        ..Default::default()
                    });
                    
                    for triple in dataset {
                        builder.insert(black_box(triple.clone())).unwrap();
                    }
                    
                    builder.flush().unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_parallel_query_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_query");
    
    // Setup graph with data
    let graph = Arc::new(IndexedGraph::new());
    for i in 0..10000 {
        for j in 0..10 {
            let triple = Triple::new(
                Subject::NamedNode(NamedNode::new(&format!("http://s/{}", i)).unwrap()),
                Predicate::NamedNode(NamedNode::new(&format!("http://p/{}", j)).unwrap()),
                Object::NamedNode(NamedNode::new(&format!("http://o/{}-{}", i, j)).unwrap()),
            );
            graph.insert(&triple);
        }
    }
    
    for num_patterns in [10, 100, 1000] {
        let patterns: Vec<_> = (0..num_patterns)
            .map(|i| {
                (
                    Some(Subject::NamedNode(NamedNode::new(&format!("http://s/{}", i)).unwrap())),
                    None,
                    None,
                )
            })
            .collect();
        
        group.throughput(Throughput::Elements(num_patterns as u64));
        
        group.bench_with_input(
            BenchmarkId::new("sequential", num_patterns),
            &patterns,
            |b, patterns| {
                b.iter(|| {
                    let mut results = Vec::new();
                    for (s, p, o) in patterns {
                        results.push(graph.query(s.as_ref(), p.as_ref(), o.as_ref()));
                    }
                    black_box(results)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel", num_patterns),
            &patterns,
            |b, patterns| {
                b.iter(|| {
                    black_box(graph.par_query_batch(patterns.clone()))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_work_stealing_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing");
    
    // Create uneven workload
    let mut operations = Vec::new();
    for i in 0..1000 {
        let size = if i % 10 == 0 { 100 } else { 10 };
        operations.push(BatchOperation::insert(create_dataset(size)));
    }
    
    for num_threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &operations,
            |b, operations| {
                b.iter(|| {
                    let processor = ParallelBatchProcessor::new(BatchConfig {
                        num_threads: Some(num_threads),
                        ..Default::default()
                    });
                    
                    for op in operations {
                        processor.submit(op.clone()).unwrap();
                    }
                    
                    processor.process(|op| match op {
                        BatchOperation::Insert(triples) => {
                            // Simulate work proportional to batch size
                            std::thread::sleep(std::time::Duration::from_micros(triples.len() as u64));
                            Ok(())
                        }
                        _ => Ok(()),
                    }).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_parallel_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_transform");
    
    for size in [1000, 10000, 50000] {
        let graph = Arc::new(IndexedGraph::new());
        let dataset = create_dataset(size);
        graph.par_insert_batch(dataset);
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("transform", size),
            &graph,
            |b, graph| {
                b.iter(|| {
                    black_box(graph.par_transform(|triple| {
                        Some(Triple::new(
                            triple.subject().clone(),
                            Predicate::NamedNode(NamedNode::new("http://new-predicate").unwrap()),
                            triple.object().clone(),
                        ))
                    }))
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("filter", size),
            &graph,
            |b, graph| {
                b.iter(|| {
                    black_box(graph.par_filter(|triple| {
                        match triple.subject() {
                            Subject::NamedNode(node) => node.as_str().ends_with('0'),
                            _ => false,
                        }
                    }))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_impact");
    let dataset = create_dataset(100000);
    
    for batch_size in [100, 1000, 5000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let processor = ParallelBatchProcessor::new(BatchConfig {
                        batch_size,
                        ..Default::default()
                    });
                    
                    // Submit all at once
                    for chunk in dataset.chunks(batch_size) {
                        processor.submit(BatchOperation::insert(chunk.to_vec())).unwrap();
                    }
                    
                    processor.process_rayon(|op| match op {
                        BatchOperation::Insert(triples) => Ok(triples.len()),
                        _ => Ok(0),
                    }).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    let dataset = create_dataset(50000);
    
    group.bench_function("with_deduplication", |b| {
        b.iter(|| {
            let mut builder = BatchBuilder::new(BatchBuilderConfig {
                coalescing_strategy: CoalescingStrategy::Deduplicate,
                auto_flush: false,
                ..Default::default()
            });
            
            // Add duplicates
            for triple in &dataset[..10000] {
                builder.insert(triple.clone()).unwrap();
                builder.insert(triple.clone()).unwrap(); // Duplicate
            }
            
            let batches = builder.flush().unwrap();
            black_box(batches)
        });
    });
    
    group.bench_function("without_deduplication", |b| {
        b.iter(|| {
            let mut builder = BatchBuilder::new(BatchBuilderConfig {
                coalescing_strategy: CoalescingStrategy::None,
                auto_flush: false,
                ..Default::default()
            });
            
            // Add duplicates
            for triple in &dataset[..10000] {
                builder.insert(triple.clone()).unwrap();
                builder.insert(triple.clone()).unwrap(); // Duplicate
            }
            
            let batches = builder.flush().unwrap();
            black_box(batches)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_vs_sequential_insert,
    bench_batch_builder_strategies,
    bench_parallel_query_patterns,
    bench_work_stealing_efficiency,
    bench_parallel_transform,
    bench_batch_size_impact,
    bench_memory_efficiency
);
criterion_main!(benches);