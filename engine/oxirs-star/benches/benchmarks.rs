//! Comprehensive benchmarks for OxiRS-Star
//!
//! This module provides benchmarks for all major components of the RDF-star implementation
//! including parsing, serialization, querying, and storage operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_star::{
    parser::{StarParser, StarFormat},
    serializer::StarSerializer,
    model::{StarGraph, StarTriple, StarTerm},
    store::StarStore,
};
use std::time::Duration;

/// Benchmark parsing performance for different RDF-star formats
fn benchmark_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing");
    
    // Test data for different sizes
    let test_sizes = [100, 1000, 10000];
    
    for size in test_sizes.iter() {
        // Generate test data
        let turtle_star_data = generate_turtle_star_data(*size);
        let ntriples_star_data = generate_ntriples_star_data(*size);
        let nquads_star_data = generate_nquads_star_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Benchmark Turtle-star parsing
        group.bench_with_input(
            BenchmarkId::new("turtle_star", size),
            &turtle_star_data,
            |b, data| {
                let parser = StarParser::new();
                b.iter(|| {
                    black_box(parser.parse_str(data, StarFormat::TurtleStar).unwrap())
                })
            },
        );
        
        // Benchmark N-Triples-star parsing
        group.bench_with_input(
            BenchmarkId::new("ntriples_star", size),
            &ntriples_star_data,
            |b, data| {
                let parser = StarParser::new();
                b.iter(|| {
                    black_box(parser.parse_str(data, StarFormat::NTriplesStar).unwrap())
                })
            },
        );
        
        // Benchmark N-Quads-star parsing
        group.bench_with_input(
            BenchmarkId::new("nquads_star", size),
            &nquads_star_data,
            |b, data| {
                let parser = StarParser::new();
                b.iter(|| {
                    black_box(parser.parse_str(data, StarFormat::NQuadsStar).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark serialization performance for different formats
fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    
    let test_sizes = [100, 1000, 10000];
    
    for size in test_sizes.iter() {
        let graph = generate_star_graph(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Benchmark different serialization formats
        for format in [StarFormat::TurtleStar, StarFormat::NTriplesStar, StarFormat::TrigStar, StarFormat::NQuadsStar] {
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", format), size),
                &graph,
                |b, graph| {
                    let serializer = StarSerializer::new();
                    b.iter(|| {
                        black_box(serializer.serialize_to_string(graph, format).unwrap())
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark store operations including indexing and querying
fn benchmark_store_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_operations");
    
    let test_sizes = [100, 1000, 10000];
    
    for size in test_sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Benchmark triple insertion
        group.bench_with_input(
            BenchmarkId::new("insert_triples", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let store = StarStore::new();
                        let triples = generate_test_triples(size);
                        (store, triples)
                    },
                    |(store, triples)| {
                        for triple in triples {
                            black_box(store.insert(&triple).unwrap());
                        }
                        black_box(store)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
        
        // Benchmark quoted triple indexing
        group.bench_with_input(
            BenchmarkId::new("quoted_triple_indexing", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let store = StarStore::new();
                        let quoted_triples = generate_quoted_triples(size);
                        (store, quoted_triples)
                    },
                    |(store, triples)| {
                        for triple in triples {
                            black_box(store.insert(&triple).unwrap());
                        }
                        black_box(store)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
        
        // Benchmark pattern-based queries
        let store = setup_populated_store(*size);
        group.bench_with_input(
            BenchmarkId::new("pattern_queries", size),
            &store,
            |b, store| {
                let subject_pattern = StarTerm::iri("http://example.org/subject1").unwrap();
                b.iter(|| {
                    black_box(store.query_triples(
                        Some(&subject_pattern),
                        None,
                        None
                    ))
                })
            },
        );
        
        // Benchmark quoted triple pattern queries
        group.bench_with_input(
            BenchmarkId::new("quoted_pattern_queries", size),
            &store,
            |b, store| {
                let pattern = StarTerm::iri("http://example.org/alice").unwrap();
                b.iter(|| {
                    black_box(store.find_triples_by_quoted_pattern(
                        Some(&pattern),
                        None,
                        None
                    ))
                })
            },
        );
    }
    
    group.finish();
}

/*
/// Benchmark SPARQL-star query execution
fn benchmark_sparql_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql_queries");
    
    let store = setup_populated_store(5000);
    let executor = QueryExecutor::new();
    
    // Simple triple pattern query
    let simple_query = r#"
        SELECT ?s ?p ?o WHERE {
            ?s ?p ?o .
        }
    "#;
    
    // Quoted triple pattern query
    let quoted_query = r#"
        SELECT ?confidence WHERE {
            <<?s <http://example.org/predicate> ?o>> <http://example.org/confidence> ?confidence .
        }
    "#;
    
    // Complex nested query
    let complex_query = r#"
        SELECT ?s ?certainty WHERE {
            ?s <http://example.org/knows> ?o .
            <<?s <http://example.org/knows> ?o>> <http://example.org/certainty> ?certainty .
            FILTER(?certainty > 0.8)
        }
    "#;
    
    for (name, query) in [
        ("simple_pattern", simple_query),
        ("quoted_pattern", quoted_query),
        ("complex_nested", complex_query),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(executor.execute_query(
                    black_box(query),
                    black_box(&store)
                ).unwrap())
            })
        });
    }
    
    group.finish();
}
*/

/*
/// Benchmark reification operations
fn benchmark_reification(c: &mut Criterion) {
    let mut group = c.benchmark_group("reification");
    
    let test_sizes = [100, 1000, 5000];
    
    for size in test_sizes.iter() {
        let graph = generate_star_graph(*size);
        let reificator = Reificator::new();
        let dereificator = Dereificator::new();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Benchmark reification
        group.bench_with_input(
            BenchmarkId::new("reify_graph", size),
            &graph,
            |b, graph| {
                b.iter(|| {
                    black_box(reificator.reify_graph(black_box(graph)).unwrap())
                })
            },
        );
        
        // Benchmark dereification
        let reified_graph = reificator.reify_graph(&graph).unwrap();
        group.bench_with_input(
            BenchmarkId::new("dereify_graph", size),
            &reified_graph,
            |b, reified_graph| {
                b.iter(|| {
                    black_box(dereificator.dereify_graph(black_box(reified_graph)).unwrap())
                })
            },
        );
    }
    
    group.finish();
}
*/

/// Benchmark memory usage and allocation patterns
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark memory-efficient parsing
    group.bench_function("memory_efficient_parsing", |b| {
        let large_data = generate_turtle_star_data(50000);
        b.iter(|| {
            let parser = StarParser::new();
            let graph = black_box(parser.parse_str(&large_data, StarFormat::TurtleStar).unwrap());
            drop(graph); // Explicit drop to measure deallocation
        })
    });
    
    // Benchmark memory usage during indexing
    group.bench_function("indexing_memory_usage", |b| {
        b.iter(|| {
            let store = StarStore::new();
            let triples = generate_quoted_triples(10000);
            for triple in triples {
                black_box(store.insert(&triple).unwrap());
            }
            black_box(store)
        })
    });
    
    group.finish();
}

// Helper functions for generating test data

fn generate_turtle_star_data(size: usize) -> String {
    let mut data = String::new();
    data.push_str("@prefix ex: <http://example.org/> .\n\n");
    
    for i in 0..size {
        if i % 3 == 0 {
            // Generate quoted triple
            data.push_str(&format!(
                "<<ex:subject{} ex:predicate{} ex:object{}>> ex:confidence \"0.{}\" .\n",
                i, i, i, (i % 100)
            ));
        } else {
            // Generate regular triple
            data.push_str(&format!(
                "ex:subject{} ex:predicate{} ex:object{} .\n",
                i, i, i
            ));
        }
    }
    
    data
}

fn generate_ntriples_star_data(size: usize) -> String {
    let mut data = String::new();
    
    for i in 0..size {
        if i % 3 == 0 {
            data.push_str(&format!(
                "<<<http://example.org/subject{}> <http://example.org/predicate{}> <http://example.org/object{}>>> <http://example.org/confidence> \"0.{}\" .\n",
                i, i, i, (i % 100)
            ));
        } else {
            data.push_str(&format!(
                "<http://example.org/subject{}> <http://example.org/predicate{}> <http://example.org/object{}> .\n",
                i, i, i
            ));
        }
    }
    
    data
}

fn generate_nquads_star_data(size: usize) -> String {
    let mut data = String::new();
    
    for i in 0..size {
        if i % 3 == 0 {
            data.push_str(&format!(
                "<<<http://example.org/subject{}> <http://example.org/predicate{}> <http://example.org/object{}>>> <http://example.org/confidence> \"0.{}\" <http://example.org/graph{}> .\n",
                i, i, i, (i % 100), (i % 5)
            ));
        } else {
            data.push_str(&format!(
                "<http://example.org/subject{}> <http://example.org/predicate{}> <http://example.org/object{}> <http://example.org/graph{}> .\n",
                i, i, i, (i % 5)
            ));
        }
    }
    
    data
}

fn generate_star_graph(size: usize) -> StarGraph {
    let mut graph = StarGraph::new();
    
    for i in 0..size {
        if i % 3 == 0 {
            // Add quoted triple
            let base_triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/subject{}", i)).unwrap(),
                StarTerm::iri(&format!("http://example.org/predicate{}", i)).unwrap(),
                StarTerm::iri(&format!("http://example.org/object{}", i)).unwrap(),
            );
            
            let quoted_triple = StarTriple::new(
                StarTerm::quoted_triple(base_triple),
                StarTerm::iri("http://example.org/confidence").unwrap(),
                StarTerm::literal(&format!("0.{}", i % 100)).unwrap(),
            );
            
            graph.insert(quoted_triple).unwrap();
        } else {
            // Add regular triple
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/subject{}", i)).unwrap(),
                StarTerm::iri(&format!("http://example.org/predicate{}", i)).unwrap(),
                StarTerm::iri(&format!("http://example.org/object{}", i)).unwrap(),
            );
            
            graph.insert(triple).unwrap();
        }
    }
    
    graph
}

fn generate_test_triples(size: usize) -> Vec<StarTriple> {
    (0..size)
        .map(|i| {
            StarTriple::new(
                StarTerm::iri(&format!("http://example.org/subject{}", i)).unwrap(),
                StarTerm::iri(&format!("http://example.org/predicate{}", i)).unwrap(),
                StarTerm::iri(&format!("http://example.org/object{}", i)).unwrap(),
            )
        })
        .collect()
}

fn generate_quoted_triples(size: usize) -> Vec<StarTriple> {
    (0..size)
        .map(|i| {
            let base_triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/alice{}", i)).unwrap(),
                StarTerm::iri(&format!("http://example.org/age")).unwrap(),
                StarTerm::literal(&format!("{}", 20 + (i % 50))).unwrap(),
            );
            
            StarTriple::new(
                StarTerm::quoted_triple(base_triple),
                StarTerm::iri("http://example.org/certainty").unwrap(),
                StarTerm::literal(&format!("0.{}", 50 + (i % 50))).unwrap(),
            )
        })
        .collect()
}

fn setup_populated_store(size: usize) -> StarStore {
    let store = StarStore::new();
    let triples = generate_test_triples(size / 2);
    let quoted_triples = generate_quoted_triples(size / 2);
    
    for triple in triples {
        store.insert(&triple).unwrap();
    }
    
    for triple in quoted_triples {
        store.insert(&triple).unwrap();
    }
    
    store
}

criterion_group!(
    benches,
    benchmark_parsing,
    benchmark_serialization,
    benchmark_store_operations,
    benchmark_memory_usage
);

criterion_main!(benches);