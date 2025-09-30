//! Benchmarks for concurrent graph operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_core::concurrent::ConcurrentGraph;
use oxirs_core::model::{NamedNode, Object, Predicate, Subject, Triple};
use std::sync::Arc;
use std::thread;

fn create_test_triple(id: usize) -> Triple {
    Triple::new(
        Subject::NamedNode(NamedNode::new(format!("http://subject/{id}")).unwrap()),
        Predicate::NamedNode(NamedNode::new(format!("http://predicate/{id}")).unwrap()),
        Object::NamedNode(NamedNode::new(format!("http://object/{id}")).unwrap()),
    )
}

fn bench_single_thread_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_thread_insert");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let graph = ConcurrentGraph::new();
                for i in 0..size {
                    let triple = create_test_triple(i);
                    graph.insert(triple).unwrap();
                }
                black_box(graph.len())
            });
        });
    }
    group.finish();
}

fn bench_concurrent_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_insert");

    for num_threads in [2, 4, 8].iter() {
        group.throughput(Throughput::Elements(10000));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let graph = Arc::new(ConcurrentGraph::new());
                    let triples_per_thread = 10000 / num_threads;

                    let handles: Vec<_> = (0..num_threads)
                        .map(|thread_id| {
                            let graph = graph.clone();
                            thread::spawn(move || {
                                for i in 0..triples_per_thread {
                                    let id = thread_id * triples_per_thread + i;
                                    let triple = create_test_triple(id);
                                    graph.insert(triple).unwrap();
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    black_box(graph.len())
                });
            },
        );
    }
    group.finish();
}

fn bench_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");

    // Setup graph with test data
    let graph = Arc::new(ConcurrentGraph::new());
    for i in 0..10000 {
        let triple = create_test_triple(i);
        graph.insert(triple).unwrap();
    }

    group.bench_function("match_all", |b| {
        b.iter(|| {
            let results = graph.match_pattern(None, None, None);
            black_box(results.len())
        });
    });

    group.bench_function("match_subject", |b| {
        let subject = Subject::NamedNode(NamedNode::new("http://subject/500").unwrap());
        b.iter(|| {
            let results = graph.match_pattern(Some(&subject), None, None);
            black_box(results.len())
        });
    });

    group.bench_function("match_subject_predicate", |b| {
        let subject = Subject::NamedNode(NamedNode::new("http://subject/500").unwrap());
        let predicate = Predicate::NamedNode(NamedNode::new("http://predicate/500").unwrap());
        b.iter(|| {
            let results = graph.match_pattern(Some(&subject), Some(&predicate), None);
            black_box(results.len())
        });
    });

    group.finish();
}

fn bench_mixed_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_operations");

    group.bench_function("read_heavy_workload", |b| {
        let graph = Arc::new(ConcurrentGraph::new());

        // Pre-populate
        for i in 0..1000 {
            graph.insert(create_test_triple(i)).unwrap();
        }

        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|thread_id| {
                    let graph = graph.clone();
                    thread::spawn(move || {
                        for i in 0..100 {
                            match thread_id {
                                0 => {
                                    // Writer thread
                                    let triple = create_test_triple(1000 + i);
                                    graph.insert(triple).unwrap();
                                }
                                _ => {
                                    // Reader threads
                                    let _ = graph.len();
                                    let _ = graph.match_pattern(None, None, None);
                                }
                            }
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for batch_size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("insert_batch", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let graph = ConcurrentGraph::new();
                    let triples: Vec<_> = (0..batch_size).map(create_test_triple).collect();

                    graph.insert_batch(triples).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_reclamation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_reclamation");

    group.bench_function("collect_cycles", |b| {
        b.iter(|| {
            let graph = ConcurrentGraph::new();

            // Insert and remove in cycles
            for cycle in 0..10 {
                for i in 0..100 {
                    let triple = create_test_triple(cycle * 100 + i);
                    graph.insert(triple.clone()).unwrap();
                    graph.remove(&triple).unwrap();
                }
                graph.collect();
            }

            black_box(graph.stats())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_thread_insert,
    bench_concurrent_insert,
    bench_pattern_matching,
    bench_mixed_operations,
    bench_batch_operations,
    bench_memory_reclamation
);
criterion_main!(benches);
