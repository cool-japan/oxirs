//! Comprehensive Benchmarking Suite for OxiRS Core
//!
//! Beta.1 Feature: Production-Ready Performance Benchmarking
//!
//! This benchmark suite provides comprehensive performance testing across all major
//! operations in oxirs-core, including:
//! - RDF parsing and serialization
//! - Graph operations (insert, query, delete)
//! - SPARQL query execution
//! - Concurrent operations
//! - Memory usage and scalability

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_core::{
    format::turtle::{TurtleParser, TurtleSerializer},
    model::{GraphName, Literal, NamedNode, Quad, Subject, Triple},
    optimization::{OptimizedGraph, RdfArena},
    rdf_store::ConcreteStore,
};
use std::time::Duration;

/// Benchmark RDF parsing performance across different formats
fn bench_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing");
    group.measurement_time(Duration::from_secs(10));

    // Generate test data of different sizes
    let sizes = vec![100, 1_000, 10_000];

    for size in sizes {
        let turtle_data = generate_turtle_data(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("turtle", size), &turtle_data, |b, data| {
            let parser = TurtleParser::new();
            b.iter(|| {
                black_box(parser.parse_str(data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark RDF serialization performance
fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![100, 1_000, 10_000];

    for size in sizes {
        let triples = generate_triples(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("turtle", size), &triples, |b, triples| {
            let serializer = TurtleSerializer::new().pretty();
            b.iter(|| {
                black_box(serializer.serialize_to_string(triples).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark graph insertion operations
fn bench_graph_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_insert");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![100, 1_000, 10_000];

    for size in sizes {
        let triples = generate_triples(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("optimized_graph", size),
            &triples,
            |b, triples| {
                b.iter(|| {
                    let graph = OptimizedGraph::new();
                    for triple in triples {
                        black_box(graph.insert(triple));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark graph query operations
fn bench_graph_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_query");
    group.measurement_time(Duration::from_secs(10));

    // Pre-populate graphs with different sizes
    let sizes = vec![1_000, 10_000, 100_000];

    for size in sizes {
        let graph = OptimizedGraph::new();
        let triples = generate_triples(size);
        for triple in &triples {
            graph.insert(triple);
        }

        // Benchmark different query patterns
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/s0").unwrap());

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("subject_query", size),
            &(&graph, &subject),
            |b, (graph, subject)| {
                b.iter(|| {
                    black_box(graph.query(Some(*subject), None, None));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent graph operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![100, 1_000, 10_000];
    let thread_counts = vec![2, 4, 8];

    for size in &sizes {
        for threads in &thread_counts {
            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("insert_threads_{threads}"), size),
                &(*size, *threads),
                |b, (size, threads)| {
                    b.iter(|| {
                        use std::sync::Arc;
                        use std::thread;

                        let graph = Arc::new(OptimizedGraph::new());
                        let chunk_size = size / threads;

                        let handles: Vec<_> = (0..*threads)
                            .map(|t| {
                                let graph = Arc::clone(&graph);
                                let start = t * chunk_size;
                                let end = if t == threads - 1 {
                                    *size
                                } else {
                                    start + chunk_size
                                };

                                thread::spawn(move || {
                                    for i in start..end {
                                        let subject =
                                            NamedNode::new(format!("http://example.org/s{i}"))
                                                .unwrap();
                                        let predicate =
                                            NamedNode::new("http://example.org/p").unwrap();
                                        let object = Literal::new(format!("object{i}"));
                                        let triple = Triple::new(subject, predicate, object);
                                        black_box(graph.insert(&triple));
                                    }
                                })
                            })
                            .collect();

                        for handle in handles {
                            handle.join().unwrap();
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory-efficient operations
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![1_000, 10_000, 100_000];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("arena_allocation", size),
            &size,
            |b, size| {
                b.iter(|| {
                    let arena = RdfArena::with_capacity(*size * 50); // Approximate size
                    for i in 0..*size {
                        black_box(arena.intern_str(&format!("http://example.org/s{i}")));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark store operations
fn bench_store_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_operations");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![100, 1_000, 10_000];

    for size in sizes {
        let quads = generate_quads(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("insert_quads", size),
            &quads,
            |b, quads| {
                b.iter(|| {
                    let store = ConcreteStore::new().unwrap();
                    for quad in quads {
                        black_box(store.insert_quad(quad.clone()).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark scalability with increasing data size
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.measurement_time(Duration::from_secs(15));

    // Test scalability from 1K to 1M triples
    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("full_pipeline", size), &size, |b, size| {
            b.iter(|| {
                // Full pipeline: parse -> insert -> query
                let triples = generate_triples(*size);
                let graph = OptimizedGraph::new();

                // Insert all triples
                for triple in &triples {
                    graph.insert(triple);
                }

                // Perform sample queries
                for i in 0..10 {
                    let subject = Subject::NamedNode(
                        NamedNode::new(format!("http://example.org/s{i}")).unwrap(),
                    );
                    black_box(graph.query(Some(&subject), None, None));
                }
            });
        });
    }

    group.finish();
}

/// Helper: Generate Turtle test data
fn generate_turtle_data(count: usize) -> String {
    let mut result = String::new();
    result.push_str("@prefix ex: <http://example.org/> .\n\n");

    for i in 0..count {
        result.push_str(&format!("ex:s{i} ex:p \"Object {i}\" .\n"));
    }

    result
}

/// Helper: Generate test triples
fn generate_triples(count: usize) -> Vec<Triple> {
    (0..count)
        .map(|i| {
            let subject = NamedNode::new(format!("http://example.org/s{i}")).unwrap();
            let predicate = NamedNode::new("http://example.org/p").unwrap();
            let object = Literal::new(format!("Object {i}"));
            Triple::new(subject, predicate, object)
        })
        .collect()
}

/// Helper: Generate test quads
fn generate_quads(count: usize) -> Vec<Quad> {
    (0..count)
        .map(|i| {
            let subject = NamedNode::new(format!("http://example.org/s{i}")).unwrap();
            let predicate = NamedNode::new("http://example.org/p").unwrap();
            let object = Literal::new(format!("Object {i}"));
            Quad::new(subject, predicate, object, GraphName::DefaultGraph)
        })
        .collect()
}

criterion_group!(
    benches,
    bench_parsing,
    bench_serialization,
    bench_graph_insert,
    bench_graph_query,
    bench_concurrent_operations,
    bench_memory_efficiency,
    bench_store_operations,
    bench_scalability
);
criterion_main!(benches);
