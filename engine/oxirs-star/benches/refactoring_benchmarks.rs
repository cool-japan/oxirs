//! Benchmarks for v0.2.0 refactored modules.
//!
//! These benchmarks verify that the refactoring did not introduce performance regressions.
//! They measure the performance of core operations after extracting modules.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_star::model::{StarTerm, StarTriple};
use oxirs_star::parser::StarParser;
use oxirs_star::store::StarStore;
use std::io::Cursor;
use std::time::Duration;

// ============================================================================
// Parser Module Benchmarks
// ============================================================================

fn benchmark_parser_context_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser_refactored/context");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("parser_creation", |b| {
        b.iter(|| {
            let parser = black_box(StarParser::new());
            black_box(parser)
        })
    });

    group.finish();
}

fn benchmark_turtle_star_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser_refactored/turtle_star");
    group.measurement_time(Duration::from_secs(10));

    let simple_turtle = r#"
        @prefix ex: <http://example.org/> .
        << ex:Alice ex:knows ex:Bob >> ex:certainty 0.9 .
    "#;

    let nested_turtle = r#"
        @prefix ex: <http://example.org/> .
        << << ex:Alice ex:knows ex:Bob >> ex:source ex:Survey1 >> ex:confidence 0.95 .
    "#;

    group.bench_function("simple_quoted_triple", |b| {
        b.iter(|| {
            let parser = StarParser::new();
            let cursor = Cursor::new(simple_turtle.as_bytes());
            let result = black_box(parser.parse_turtle_star(cursor));
            black_box(result)
        })
    });

    group.bench_function("nested_quoted_triple", |b| {
        b.iter(|| {
            let parser = StarParser::new();
            let cursor = Cursor::new(nested_turtle.as_bytes());
            let result = black_box(parser.parse_turtle_star(cursor));
            black_box(result)
        })
    });

    group.finish();
}

fn benchmark_format_parsing_refactored(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser_refactored/formats");
    group.measurement_time(Duration::from_secs(10));

    let ntriples =
        r#"<<<http://ex.org/a> <http://ex.org/b> <http://ex.org/c>>> <http://ex.org/p> "value" ."#;
    let turtle = r#"@prefix ex: <http://ex.org/> . << ex:a ex:b ex:c >> ex:p "value" ."#;

    group.bench_function("ntriples_star", |b| {
        b.iter(|| {
            let parser = StarParser::new();
            let cursor = Cursor::new(ntriples.as_bytes());
            let result = black_box(parser.parse_ntriples_star(cursor));
            black_box(result)
        })
    });

    group.bench_function("turtle_star", |b| {
        b.iter(|| {
            let parser = StarParser::new();
            let cursor = Cursor::new(turtle.as_bytes());
            let result = black_box(parser.parse_turtle_star(cursor));
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Store Module Benchmarks
// ============================================================================

fn benchmark_store_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_refactored/creation");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("store_new", |b| {
        b.iter(|| {
            let store = black_box(StarStore::new());
            black_box(store)
        })
    });

    group.finish();
}

fn benchmark_store_insert_refactored(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_refactored/insert");
    group.measurement_time(Duration::from_secs(10));

    // Create test triples
    fn create_simple_triple(index: usize) -> StarTriple {
        StarTriple {
            subject: StarTerm::NamedNode(oxirs_star::model::NamedNode {
                iri: format!("http://example.org/subject{}", index),
            }),
            predicate: StarTerm::NamedNode(oxirs_star::model::NamedNode {
                iri: "http://example.org/predicate".to_string(),
            }),
            object: StarTerm::Literal(oxirs_star::model::Literal {
                value: format!("value{}", index),
                datatype: None,
                language: None,
            }),
        }
    }

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let store = StarStore::new();
                for i in 0..size {
                    let triple = create_simple_triple(i);
                    black_box(store.insert(&triple)).ok();
                }
                black_box(store)
            })
        });
    }

    group.finish();
}

fn benchmark_store_query_refactored(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_refactored/query");
    group.measurement_time(Duration::from_secs(10));

    // Pre-populate store
    let store = StarStore::new();
    for i in 0..1000 {
        let triple = StarTriple {
            subject: StarTerm::NamedNode(oxirs_star::model::NamedNode {
                iri: format!("http://example.org/subject{}", i % 100),
            }),
            predicate: StarTerm::NamedNode(oxirs_star::model::NamedNode {
                iri: format!("http://example.org/predicate{}", i % 10),
            }),
            object: StarTerm::Literal(oxirs_star::model::Literal {
                value: format!("value{}", i),
                datatype: None,
                language: None,
            }),
        };
        store.insert(&triple).ok();
    }

    group.bench_function("query_all", |b| {
        b.iter(|| {
            let result = black_box(store.all_triples());
            black_box(result)
        })
    });

    group.bench_function("count_triples", |b| {
        b.iter(|| {
            let count = black_box(store.len());
            black_box(count)
        })
    });

    group.finish();
}

fn benchmark_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_refactored/cache");
    group.measurement_time(Duration::from_secs(10));

    use oxirs_star::store::{PublicCacheConfig, PublicStarCache};

    let cache = PublicStarCache::new(PublicCacheConfig::default());

    // Warm up cache
    for i in 0..100 {
        let key = format!("key{}", i);
        cache.put(key, vec![]);
    }

    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            let result = black_box(cache.get("key50"));
            black_box(result)
        })
    });

    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            let result = black_box(cache.get("nonexistent"));
            black_box(result)
        })
    });

    group.finish();
}

fn benchmark_store_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_refactored/statistics");
    group.measurement_time(Duration::from_secs(10));

    // Create store with data
    let store = StarStore::new();
    for i in 0..1000 {
        let triple = StarTriple {
            subject: StarTerm::NamedNode(oxirs_star::model::NamedNode {
                iri: format!("http://example.org/s{}", i),
            }),
            predicate: StarTerm::NamedNode(oxirs_star::model::NamedNode {
                iri: format!("http://example.org/p{}", i % 100),
            }),
            object: StarTerm::Literal(oxirs_star::model::Literal {
                value: format!("val{}", i),
                datatype: None,
                language: None,
            }),
        };
        store.insert(&triple).ok();
    }

    group.bench_function("get_statistics", |b| {
        b.iter(|| {
            let stats = black_box(store.statistics());
            black_box(stats)
        })
    });

    group.bench_function("is_empty", |b| {
        b.iter(|| {
            let empty = black_box(store.is_empty());
            black_box(empty)
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group!(
    parser_benches,
    benchmark_parser_context_creation,
    benchmark_turtle_star_parsing,
    benchmark_format_parsing_refactored,
);

criterion_group!(
    store_benches,
    benchmark_store_creation,
    benchmark_store_insert_refactored,
    benchmark_store_query_refactored,
    benchmark_cache_operations,
    benchmark_store_statistics,
);

criterion_main!(parser_benches, store_benches);
