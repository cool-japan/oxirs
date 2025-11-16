//! SPARQL-star Performance Benchmarking Suite
//!
//! Comprehensive benchmarks for SPARQL 1.2 / SPARQL-star features:
//! - Quoted triple creation and insertion
//! - Quoted triple pattern matching
//! - Nested quoted triple performance
//! - SPARQL-star built-in functions
//! - Nesting depth queries
//! - Term conversion performance
//! - Statistics tracking overhead
//! - Scalability with different nesting levels

#[cfg(feature = "star")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(feature = "star")]
use oxirs_arq::star_integration::{
    pattern_matching, sparql_star_functions, star_statistics::SparqlStarStatistics,
    SparqlStarExecutor,
};
#[cfg(feature = "star")]
use oxirs_star::{StarTerm, StarTriple};
#[cfg(feature = "star")]
use std::time::Duration;

#[cfg(feature = "star")]
/// Benchmark quoted triple creation
fn bench_quoted_triple_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_triple_creation");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("simple_quoted_triple", |b| {
        b.iter(|| {
            let subject = StarTerm::iri("http://example.org/Alice").unwrap();
            let predicate = StarTerm::iri("http://xmlns.com/foaf/0.1/knows").unwrap();
            let object = StarTerm::iri("http://example.org/Bob").unwrap();

            let triple = StarTriple::new(subject, predicate, object);
            black_box(triple)
        });
    });

    group.bench_function("quoted_triple_with_metadata", |b| {
        b.iter(|| {
            let inner = StarTriple::new(
                StarTerm::iri("http://example.org/Alice").unwrap(),
                StarTerm::iri("http://xmlns.com/foaf/0.1/age").unwrap(),
                StarTerm::literal("25").unwrap(),
            );

            let metadata = StarTriple::new(
                StarTerm::quoted_triple(inner),
                StarTerm::iri("http://example.org/certainty").unwrap(),
                StarTerm::literal("0.95").unwrap(),
            );
            black_box(metadata)
        });
    });

    group.bench_function("nested_quoted_triple_depth_2", |b| {
        b.iter(|| {
            let inner = StarTriple::new(
                StarTerm::iri("http://example.org/Alice").unwrap(),
                StarTerm::iri("http://xmlns.com/foaf/0.1/knows").unwrap(),
                StarTerm::iri("http://example.org/Bob").unwrap(),
            );

            let middle = StarTriple::new(
                StarTerm::quoted_triple(inner),
                StarTerm::iri("http://example.org/certainty").unwrap(),
                StarTerm::literal("0.95").unwrap(),
            );

            let outer = StarTriple::new(
                StarTerm::iri("http://example.org/Charlie").unwrap(),
                StarTerm::iri("http://example.org/believes").unwrap(),
                StarTerm::quoted_triple(middle),
            );
            black_box(outer)
        });
    });

    group.finish();
}

#[cfg(feature = "star")]
/// Benchmark store insertion performance
fn bench_store_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_store_insertion");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![100, 1_000, 10_000];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("quoted_triples", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut executor = SparqlStarExecutor::new();
                    let store = executor.store_mut();

                    for i in 0..size {
                        let inner = StarTriple::new(
                            StarTerm::iri(&format!("http://example.org/person{}", i)).unwrap(),
                            StarTerm::iri("http://xmlns.com/foaf/0.1/age").unwrap(),
                            StarTerm::literal(&format!("{}", 20 + (i % 50))).unwrap(),
                        );

                        let metadata = StarTriple::new(
                            StarTerm::quoted_triple(inner),
                            StarTerm::iri("http://example.org/reportedBy").unwrap(),
                            StarTerm::iri("http://example.org/Alice").unwrap(),
                        );

                        store.insert(&metadata).unwrap();
                    }
                    black_box(store.len())
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "star")]
/// Benchmark pattern matching performance
fn bench_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_pattern_matching");
    group.measurement_time(Duration::from_secs(15));

    let sizes = vec![100, 500, 1_000];

    for size in sizes {
        let store = setup_star_store(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("find_by_subject", size),
            &store,
            |b, store| {
                let alice = StarTerm::iri("http://example.org/Alice").unwrap();
                b.iter(|| {
                    let results = store.find_triples_by_quoted_pattern(Some(&alice), None, None);
                    black_box(results.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("find_by_predicate", size),
            &store,
            |b, store| {
                let age_pred = StarTerm::iri("http://xmlns.com/foaf/0.1/age").unwrap();
                b.iter(|| {
                    let results = store.find_triples_by_quoted_pattern(None, Some(&age_pred), None);
                    black_box(results.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("find_by_object", size),
            &store,
            |b, store| {
                let literal_25 = StarTerm::literal("25").unwrap();
                b.iter(|| {
                    let results =
                        store.find_triples_by_quoted_pattern(None, None, Some(&literal_25));
                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "star")]
/// Benchmark nesting depth queries
fn bench_nesting_depth_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_nesting_depth");
    group.measurement_time(Duration::from_secs(10));

    let store = setup_multi_depth_store(1000);

    group.bench_function("find_depth_0", |b| {
        b.iter(|| {
            let results = store.find_triples_by_nesting_depth(0, Some(0));
            black_box(results.len())
        });
    });

    group.bench_function("find_depth_1", |b| {
        b.iter(|| {
            let results = store.find_triples_by_nesting_depth(1, Some(1));
            black_box(results.len())
        });
    });

    group.bench_function("find_depth_2", |b| {
        b.iter(|| {
            let results = store.find_triples_by_nesting_depth(2, Some(2));
            black_box(results.len())
        });
    });

    group.bench_function("find_depth_range_0_to_2", |b| {
        b.iter(|| {
            let results = store.find_triples_by_nesting_depth(0, Some(2));
            black_box(results.len())
        });
    });

    group.finish();
}

#[cfg(feature = "star")]
/// Benchmark SPARQL-star utility functions
fn bench_utility_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_utility_functions");
    group.measurement_time(Duration::from_secs(10));

    // Setup test data
    let quoted_term =
        oxirs_arq::algebra::Term::QuotedTriple(Box::new(oxirs_arq::algebra::TriplePattern::new(
            oxirs_arq::algebra::Term::Iri(
                oxirs_core::model::NamedNode::new("http://example.org/Alice").unwrap(),
            ),
            oxirs_arq::algebra::Term::Iri(
                oxirs_core::model::NamedNode::new("http://xmlns.com/foaf/0.1/age").unwrap(),
            ),
            oxirs_arq::algebra::Term::Literal(oxirs_arq::algebra::Literal::new(
                "25".to_string(),
                None,
                None,
            )),
        )));

    group.bench_function("is_quoted_triple", |b| {
        b.iter(|| {
            let result = sparql_star_functions::is_quoted_triple(&quoted_term);
            black_box(result)
        });
    });

    group.bench_function("get_subject", |b| {
        b.iter(|| {
            let result = sparql_star_functions::get_subject(&quoted_term);
            black_box(result)
        });
    });

    group.bench_function("get_predicate", |b| {
        b.iter(|| {
            let result = sparql_star_functions::get_predicate(&quoted_term);
            black_box(result)
        });
    });

    group.bench_function("get_object", |b| {
        b.iter(|| {
            let result = sparql_star_functions::get_object(&quoted_term);
            black_box(result)
        });
    });

    group.bench_function("nesting_depth", |b| {
        b.iter(|| {
            let depth = pattern_matching::nesting_depth(&quoted_term);
            black_box(depth)
        });
    });

    group.bench_function("extract_quoted_triples", |b| {
        b.iter(|| {
            let triples = pattern_matching::extract_quoted_triples(&quoted_term);
            black_box(triples.len())
        });
    });

    group.finish();
}

#[cfg(feature = "star")]
/// Benchmark term conversion performance
fn bench_term_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_term_conversion");
    group.measurement_time(Duration::from_secs(10));

    let arq_iri = oxirs_arq::algebra::Term::Iri(
        oxirs_core::model::NamedNode::new("http://example.org/test").unwrap(),
    );

    let star_iri = StarTerm::iri("http://example.org/test").unwrap();

    group.bench_function("arq_to_star_iri", |b| {
        b.iter(|| {
            let result = SparqlStarExecutor::term_to_star_term(&arq_iri).unwrap();
            black_box(result)
        });
    });

    group.bench_function("star_to_arq_iri", |b| {
        b.iter(|| {
            let result = SparqlStarExecutor::star_term_to_term(&star_iri).unwrap();
            black_box(result)
        });
    });

    // Benchmark literal conversion
    let arq_literal = oxirs_arq::algebra::Term::Literal(
        oxirs_arq::algebra::Literal::with_language("Hello".to_string(), "en".to_string()),
    );

    let star_literal = StarTerm::literal_with_language("Hello", "en").unwrap();

    group.bench_function("arq_to_star_literal", |b| {
        b.iter(|| {
            let result = SparqlStarExecutor::term_to_star_term(&arq_literal).unwrap();
            black_box(result)
        });
    });

    group.bench_function("star_to_arq_literal", |b| {
        b.iter(|| {
            let result = SparqlStarExecutor::star_term_to_term(&star_literal).unwrap();
            black_box(result)
        });
    });

    // Benchmark quoted triple conversion
    let inner_pattern = oxirs_arq::algebra::TriplePattern::new(
        oxirs_arq::algebra::Term::Iri(
            oxirs_core::model::NamedNode::new("http://example.org/s").unwrap(),
        ),
        oxirs_arq::algebra::Term::Iri(
            oxirs_core::model::NamedNode::new("http://example.org/p").unwrap(),
        ),
        oxirs_arq::algebra::Term::Iri(
            oxirs_core::model::NamedNode::new("http://example.org/o").unwrap(),
        ),
    );

    group.bench_function("triple_pattern_to_star_triple", |b| {
        b.iter(|| {
            let result = SparqlStarExecutor::triple_pattern_to_star_triple(&inner_pattern).unwrap();
            black_box(result)
        });
    });

    group.finish();
}

#[cfg(feature = "star")]
/// Benchmark statistics tracking overhead
fn bench_statistics_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_statistics");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("record_operations", |b| {
        b.iter(|| {
            let mut stats = SparqlStarStatistics::new();

            for i in 0..1000 {
                stats.record_quoted_pattern(i % 3);
                stats.record_star_function();
            }

            stats.record_execution_time(5000);
            stats.record_results(1000);

            black_box(stats.avg_time_per_result())
        });
    });

    group.bench_function("stats_creation", |b| {
        b.iter(|| {
            let stats = SparqlStarStatistics::new();
            black_box(stats)
        });
    });

    group.finish();
}

#[cfg(feature = "star")]
/// Benchmark scalability with increasing nesting depth
fn bench_scalability_by_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_scalability_depth");
    group.measurement_time(Duration::from_secs(15));

    for depth in vec![1, 2, 3, 4] {
        group.bench_with_input(
            BenchmarkId::new("create_nested_triple", depth),
            &depth,
            |b, &depth| {
                b.iter(|| {
                    let triple = create_nested_triple(depth);
                    black_box(triple)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("calculate_nesting_depth", depth),
            &depth,
            |b, &depth| {
                let triple = create_nested_triple_arq(depth);
                b.iter(|| {
                    let calculated_depth = pattern_matching::nesting_depth(&triple);
                    black_box(calculated_depth)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "star")]
/// Helper: Setup star store with test data
fn setup_star_store(size: usize) -> oxirs_star::StarStore {
    let store = oxirs_star::StarStore::new();

    for i in 0..size {
        // Create base statement
        let inner = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/person{}", i)).unwrap(),
            StarTerm::iri("http://xmlns.com/foaf/0.1/age").unwrap(),
            StarTerm::literal(&format!("{}", 20 + (i % 50))).unwrap(),
        );

        // Add metadata about certainty (50% of triples)
        if i % 2 == 0 {
            let metadata = StarTriple::new(
                StarTerm::quoted_triple(inner.clone()),
                StarTerm::iri("http://example.org/certainty").unwrap(),
                StarTerm::literal(&format!("{:.2}", 0.7 + (i % 30) as f64 / 100.0)).unwrap(),
            );
            store.insert(&metadata).unwrap();
        }

        // Add provenance information (70% of triples)
        if i % 10 < 7 {
            let provenance = StarTriple::new(
                StarTerm::quoted_triple(inner.clone()),
                StarTerm::iri("http://example.org/reportedBy").unwrap(),
                StarTerm::iri("http://example.org/Alice").unwrap(),
            );
            store.insert(&provenance).unwrap();
        }
    }

    store
}

#[cfg(feature = "star")]
/// Helper: Setup store with multiple nesting depths
fn setup_multi_depth_store(base_size: usize) -> oxirs_star::StarStore {
    let store = oxirs_star::StarStore::new();

    for i in 0..base_size {
        let depth = i % 3; // Depth 1, 2, or 3

        if depth >= 1 {
            // Depth 1: Simple quoted triple
            let inner = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/person{}", i)).unwrap(),
                StarTerm::iri("http://xmlns.com/foaf/0.1/age").unwrap(),
                StarTerm::literal(&format!("{}", 20 + (i % 50))).unwrap(),
            );

            let depth1 = StarTriple::new(
                StarTerm::quoted_triple(inner),
                StarTerm::iri("http://example.org/certainty").unwrap(),
                StarTerm::literal("0.9").unwrap(),
            );

            if depth == 1 {
                store.insert(&depth1).unwrap();
            } else if depth >= 2 {
                // Depth 2: Nested quoted triple
                let depth2 = StarTriple::new(
                    StarTerm::iri(&format!("http://example.org/person{}", i + 1000)).unwrap(),
                    StarTerm::iri("http://example.org/believes").unwrap(),
                    StarTerm::quoted_triple(depth1),
                );
                store.insert(&depth2).unwrap();
            }
        }
    }

    store
}

#[cfg(feature = "star")]
/// Helper: Create nested triple with specific depth
fn create_nested_triple(depth: usize) -> StarTriple {
    let mut current = StarTriple::new(
        StarTerm::iri("http://example.org/Alice").unwrap(),
        StarTerm::iri("http://xmlns.com/foaf/0.1/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );

    for i in 0..depth {
        current = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/person{}", i)).unwrap(),
            StarTerm::iri("http://example.org/believes").unwrap(),
            StarTerm::quoted_triple(current),
        );
    }

    current
}

#[cfg(feature = "star")]
/// Helper: Create nested ARQ triple pattern with specific depth
fn create_nested_triple_arq(depth: usize) -> oxirs_arq::algebra::Term {
    let mut current =
        oxirs_arq::algebra::Term::QuotedTriple(Box::new(oxirs_arq::algebra::TriplePattern::new(
            oxirs_arq::algebra::Term::Iri(
                oxirs_core::model::NamedNode::new("http://example.org/Alice").unwrap(),
            ),
            oxirs_arq::algebra::Term::Iri(
                oxirs_core::model::NamedNode::new("http://xmlns.com/foaf/0.1/age").unwrap(),
            ),
            oxirs_arq::algebra::Term::Literal(oxirs_arq::algebra::Literal::new(
                "25".to_string(),
                None,
                None,
            )),
        )));

    for i in 1..depth {
        current = oxirs_arq::algebra::Term::QuotedTriple(Box::new(
            oxirs_arq::algebra::TriplePattern::new(
                oxirs_arq::algebra::Term::Iri(
                    oxirs_core::model::NamedNode::new(&format!("http://example.org/person{}", i))
                        .unwrap(),
                ),
                oxirs_arq::algebra::Term::Iri(
                    oxirs_core::model::NamedNode::new("http://example.org/believes").unwrap(),
                ),
                current,
            ),
        ));
    }

    current
}

#[cfg(feature = "star")]
criterion_group!(
    star_benches,
    bench_quoted_triple_creation,
    bench_store_insertion,
    bench_pattern_matching,
    bench_nesting_depth_queries,
    bench_utility_functions,
    bench_term_conversion,
    bench_statistics_tracking,
    bench_scalability_by_depth
);

#[cfg(feature = "star")]
criterion_main!(star_benches);

#[cfg(not(feature = "star"))]
fn main() {
    eprintln!("SPARQL-star benchmarks require the 'star' feature.");
    eprintln!("Run with: cargo bench --bench sparql_star_bench --features star");
}
