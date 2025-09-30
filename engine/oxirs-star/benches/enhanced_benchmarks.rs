//! Enhanced benchmarks for OxiRS-Star performance testing
//!
//! This module provides detailed benchmarks for parsing, serialization,
//! storage, and memory performance characteristics.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_star::{
    model::{StarGraph, StarQuad, StarTerm, StarTriple},
    parser::{StarFormat, StarParser},
    serializer::StarSerializer,
    store::StarStore,
};
use std::time::Duration;

/// Benchmark parsing performance with different complexity levels
fn benchmark_parsing_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing_complexity");

    // Test different nesting depths
    let nesting_depths = [0, 1, 2, 3];
    let triple_count = 1000;

    for depth in nesting_depths.iter() {
        let data = generate_nested_turtle_star(*depth, triple_count);

        group.throughput(Throughput::Bytes(data.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("nesting_depth", depth),
            &data,
            |b, data| {
                let parser = StarParser::new();
                b.iter(|| black_box(parser.parse_str(data, StarFormat::TurtleStar).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark serialization with different graph structures
fn benchmark_serialization_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization_structures");

    // Different graph structures
    let structures = [
        ("flat", generate_flat_graph(1000)),
        ("quoted_heavy", generate_quoted_heavy_graph(1000)),
        ("mixed", generate_mixed_graph(1000)),
        ("deep_nested", generate_deep_nested_graph(100)),
    ];

    for (name, graph) in structures.iter() {
        let triple_count = graph.total_len();
        group.throughput(Throughput::Elements(triple_count as u64));

        for format in [StarFormat::NTriplesStar, StarFormat::TurtleStar] {
            group.bench_with_input(
                BenchmarkId::new(format!("{name}/{format:?}"), triple_count),
                graph,
                |b, graph| {
                    let serializer = StarSerializer::new();
                    b.iter(|| black_box(serializer.serialize_to_string(graph, format).unwrap()))
                },
            );
        }
    }

    group.finish();
}

/// Benchmark store indexing performance
fn benchmark_store_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_indexing");

    // Different query patterns
    let patterns = [
        ("subject_only", 1, 0, 0),
        ("predicate_only", 0, 1, 0),
        ("object_only", 0, 0, 1),
        ("subject_predicate", 1, 1, 0),
        ("all_bound", 1, 1, 1),
    ];

    let store_size = 10000;
    let store = setup_indexed_store(store_size);

    for (name, s, p, o) in patterns.iter() {
        group.bench_with_input(
            BenchmarkId::new("query_pattern", name),
            &store,
            |b, store| {
                let subject = if *s == 1 {
                    Some(StarTerm::iri("http://example.org/subject500").unwrap())
                } else {
                    None
                };
                let predicate = if *p == 1 {
                    Some(StarTerm::iri("http://example.org/predicate").unwrap())
                } else {
                    None
                };
                let object = if *o == 1 {
                    Some(StarTerm::literal("500").unwrap())
                } else {
                    None
                };

                b.iter(|| {
                    black_box(store.query_triples(
                        subject.as_ref(),
                        predicate.as_ref(),
                        object.as_ref(),
                    ))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark quoted triple operations
fn benchmark_quoted_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("quoted_operations");

    let sizes = [100, 1000, 5000];

    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Benchmark finding triples containing specific quoted patterns
        let store = setup_quoted_store(*size);

        group.bench_with_input(
            BenchmarkId::new("find_containing_quoted", size),
            &store,
            |b, store| {
                let quoted = StarTriple::new(
                    StarTerm::iri("http://example.org/alice").unwrap(),
                    StarTerm::iri("http://example.org/age").unwrap(),
                    StarTerm::literal("25").unwrap(),
                );

                b.iter(|| black_box(store.find_triples_containing_quoted(&quoted)))
            },
        );

        // Benchmark nesting depth queries
        group.bench_with_input(
            BenchmarkId::new("find_by_nesting_depth", size),
            &store,
            |b, store| b.iter(|| black_box(store.find_triples_by_nesting_depth(1, Some(2)))),
        );
    }

    group.finish();
}

/// Benchmark memory efficiency for large graphs
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.measurement_time(Duration::from_secs(15));

    // Benchmark graph-to-store conversion
    group.bench_function("graph_to_store_conversion", |b| {
        let graph = generate_large_mixed_graph(5000);
        b.iter(|| {
            let store = StarStore::new();
            store.from_graph(&graph).unwrap();
            black_box(());
            store
        })
    });

    // Benchmark store optimization
    group.bench_function("store_optimization", |b| {
        let store = setup_fragmented_store(5000);
        b.iter(|| {
            store.optimize().unwrap();
            black_box(())
        })
    });

    group.finish();
}

/// Benchmark concurrent operations
fn benchmark_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");

    let thread_counts = [1, 2, 4];
    let operations_per_thread = 1000;

    for threads in thread_counts.iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_reads", threads),
            threads,
            |b, &thread_count| {
                let store = std::sync::Arc::new(setup_indexed_store(10000));

                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|_| {
                            let store_clone = std::sync::Arc::clone(&store);
                            std::thread::spawn(move || {
                                for i in 0..operations_per_thread {
                                    let idx = i % 1000;
                                    let subject =
                                        StarTerm::iri(&format!("http://example.org/subject{idx}"))
                                            .unwrap();
                                    let _ = black_box(store_clone.query_triples(
                                        Some(&subject),
                                        None,
                                        None,
                                    ));
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

// Helper functions for generating test data

fn generate_nested_turtle_star(depth: usize, count: usize) -> String {
    let mut data = String::new();
    data.push_str("@prefix ex: <http://example.org/> .\n\n");

    for i in 0..count {
        let triple = generate_nested_triple_string(depth, i);
        data.push_str(&format!("{triple} .\n"));
    }

    data
}

fn generate_nested_triple_string(depth: usize, id: usize) -> String {
    if depth == 0 {
        format!("ex:s{id} ex:p{id} ex:o{id}")
    } else {
        let inner = generate_nested_triple_string(depth - 1, id);
        format!("<<{inner}>> ex:meta{depth} \"{depth}\"")
    }
}

fn generate_flat_graph(size: usize) -> StarGraph {
    let mut graph = StarGraph::new();

    for i in 0..size {
        let triple = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/subject{i}")).unwrap(),
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::literal(&format!("{i}")).unwrap(),
        );
        graph.insert(triple).unwrap();
    }

    graph
}

fn generate_quoted_heavy_graph(size: usize) -> StarGraph {
    let mut graph = StarGraph::new();

    for i in 0..size {
        let inner = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal(&format!("{i}")).unwrap(),
        );

        let quoted = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/confidence").unwrap(),
            StarTerm::literal(&format!("0.{}", i % 100)).unwrap(),
        );

        graph.insert(quoted).unwrap();
    }

    graph
}

fn generate_mixed_graph(size: usize) -> StarGraph {
    let mut graph = StarGraph::new();

    for i in 0..size {
        if i % 3 == 0 {
            // Quoted triple
            let inner = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&format!("{i}")).unwrap(),
            );

            let quoted = StarTriple::new(
                StarTerm::quoted_triple(inner),
                StarTerm::iri("http://example.org/meta").unwrap(),
                StarTerm::literal("metadata").unwrap(),
            );

            graph.insert(quoted).unwrap();
        } else {
            // Regular triple
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&format!("{i}")).unwrap(),
            );

            graph.insert(triple).unwrap();
        }
    }

    graph
}

fn generate_deep_nested_graph(size: usize) -> StarGraph {
    let mut graph = StarGraph::new();

    for i in 0..size {
        let mut current = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/base{i}")).unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal(&format!("{i}")).unwrap(),
        );

        // Create nested structure
        for depth in 0..3 {
            current = StarTriple::new(
                StarTerm::quoted_triple(current),
                StarTerm::iri(&format!("http://example.org/level{depth}")).unwrap(),
                StarTerm::literal(&format!("depth{depth}")).unwrap(),
            );
        }

        graph.insert(current).unwrap();
    }

    graph
}

fn generate_large_mixed_graph(size: usize) -> StarGraph {
    let mut graph = StarGraph::new();

    // Mix of different triple types
    for i in 0..size {
        match i % 5 {
            0 => {
                // Simple triple
                let triple = StarTriple::new(
                    StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
                    StarTerm::iri("http://example.org/p").unwrap(),
                    StarTerm::literal(&format!("{i}")).unwrap(),
                );
                graph.insert(triple).unwrap();
            }
            1 => {
                // Quoted triple
                let inner = StarTriple::new(
                    StarTerm::iri(&format!("http://example.org/alice{i}")).unwrap(),
                    StarTerm::iri("http://example.org/age").unwrap(),
                    StarTerm::literal("25").unwrap(),
                );

                let quoted = StarTriple::new(
                    StarTerm::quoted_triple(inner),
                    StarTerm::iri("http://example.org/certainty").unwrap(),
                    StarTerm::literal("0.9").unwrap(),
                );

                graph.insert(quoted).unwrap();
            }
            2 => {
                // Blank node triple
                let triple = StarTriple::new(
                    StarTerm::blank_node(&format!("b{i}")).unwrap(),
                    StarTerm::iri("http://example.org/type").unwrap(),
                    StarTerm::iri("http://example.org/Thing").unwrap(),
                );
                graph.insert(triple).unwrap();
            }
            3 => {
                // Literal with language
                let triple = StarTriple::new(
                    StarTerm::iri(&format!("http://example.org/doc{i}")).unwrap(),
                    StarTerm::iri("http://example.org/title").unwrap(),
                    StarTerm::literal_with_language(&format!("Title {i}"), "en").unwrap(),
                );
                graph.insert(triple).unwrap();
            }
            _ => {
                // Quad with named graph
                let quad = StarQuad::new(
                    StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
                    StarTerm::iri("http://example.org/p").unwrap(),
                    StarTerm::literal(&format!("{i}")).unwrap(),
                    Some(StarTerm::iri(&format!("http://example.org/graph{}", i % 10)).unwrap()),
                );
                graph.insert_quad(quad).unwrap();
            }
        }
    }

    graph
}

fn setup_indexed_store(size: usize) -> StarStore {
    let store = StarStore::new();

    for i in 0..size {
        let triple = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/subject{i}")).unwrap(),
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::literal(&format!("{i}")).unwrap(),
        );
        store.insert(&triple).unwrap();
    }

    store
}

fn setup_quoted_store(size: usize) -> StarStore {
    let store = StarStore::new();

    for i in 0..size {
        let depth = (i % 3) + 1;
        let mut current = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        for d in 0..depth {
            current = StarTriple::new(
                StarTerm::quoted_triple(current),
                StarTerm::iri(&format!("http://example.org/meta{d}")).unwrap(),
                StarTerm::literal(&format!("level{d}")).unwrap(),
            );
        }

        store.insert(&current).unwrap();
    }

    store
}

fn setup_fragmented_store(size: usize) -> StarStore {
    let store = StarStore::new();

    // Insert and remove to create fragmentation
    for i in 0..size * 2 {
        let triple = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/s{i}")).unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal(&format!("{i}")).unwrap(),
        );
        store.insert(&triple).unwrap();

        // Remove every other triple
        if i % 2 == 0 && i > 0 {
            let to_remove = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{}", i - 1)).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&format!("{}", i - 1)).unwrap(),
            );
            store.remove(&to_remove).unwrap();
        }
    }

    store
}

criterion_group!(
    benches,
    benchmark_parsing_complexity,
    benchmark_serialization_structures,
    benchmark_store_indexing,
    benchmark_quoted_operations,
    benchmark_memory_efficiency,
    benchmark_concurrent_operations
);

criterion_main!(benches);
