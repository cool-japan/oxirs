//! Performance benchmarks for oxirs-tdb against 100M+ triple targets
//!
//! This benchmark suite tests the performance of oxirs-tdb with large-scale datasets
//! to ensure it meets the performance requirements specified in the TODO.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use oxirs_tdb::{TdbStore, TdbConfig, Term};
use rand::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

/// Generate random terms for benchmarking
fn generate_random_triple(rng: &mut impl Rng) -> (Term, Term, Term) {
    let subject = Term::iri(&format!("http://example.org/subject/{}", rng.gen::<u64>()));
    let predicate = Term::iri(&format!("http://example.org/predicate/{}", rng.gen::<u8>()));
    let object = if rng.gen_bool(0.5) {
        Term::literal(&format!("value_{}", rng.gen::<u32>()))
    } else {
        Term::iri(&format!("http://example.org/object/{}", rng.gen::<u64>()))
    };
    
    (subject, predicate, object)
}

/// Benchmark insertion of triples
fn bench_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);
    
    for size in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched_ref(
                || {
                    let temp_dir = TempDir::new().unwrap();
                    let config = TdbConfig {
                        location: temp_dir.path().to_string_lossy().to_string(),
                        ..Default::default()
                    };
                    let store = TdbStore::new(config).unwrap();
                    let mut rng = rand::thread_rng();
                    let triples: Vec<(Term, Term, Term)> = (0..size)
                        .map(|_| generate_random_triple(&mut rng))
                        .collect();
                    (temp_dir, store, triples)
                },
                |(_temp_dir, store, triples)| {
                    for (subject, predicate, object) in triples.iter() {
                        store.insert_triple(black_box(subject), black_box(predicate), black_box(object)).unwrap();
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Benchmark bulk loading of triples via transactions
fn bench_bulk_load_transactions(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_load_transactions");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);
    
    for size in [10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched_ref(
                || {
                    let temp_dir = TempDir::new().unwrap();
                    let config = TdbConfig {
                        location: temp_dir.path().to_string_lossy().to_string(),
                        ..Default::default()
                    };
                    let store = TdbStore::new(config).unwrap();
                    let mut rng = rand::thread_rng();
                    let triples: Vec<(Term, Term, Term)> = (0..size)
                        .map(|_| generate_random_triple(&mut rng))
                        .collect();
                    (temp_dir, store, triples)
                },
                |(_temp_dir, store, triples)| {
                    let tx = store.begin_transaction().unwrap();
                    for (subject, predicate, object) in triples.iter() {
                        store.triple_store().insert_triple_in_transaction(
                            tx.id(),
                            &store.triple_store().store_term(subject).unwrap(),
                            &store.triple_store().store_term(predicate).unwrap(),
                            &store.triple_store().store_term(object).unwrap()
                        ).unwrap();
                    }
                    store.commit_transaction(tx).unwrap();
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Benchmark query performance on pre-loaded datasets
fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    group.measurement_time(Duration::from_secs(10));
    
    // Prepare stores with different sizes
    let sizes = vec![10_000, 100_000];
    let stores: Vec<(usize, TempDir, Arc<TdbStore>)> = sizes
        .into_iter()
        .map(|size| {
            let temp_dir = TempDir::new().unwrap();
            let config = TdbConfig {
                location: temp_dir.path().to_string_lossy().to_string(),
                ..Default::default()
            };
            let store = TdbStore::new(config).unwrap();
            let mut rng = rand::thread_rng();
            
            // Load triples
            for _ in 0..size {
                let (subject, predicate, object) = generate_random_triple(&mut rng);
                store.insert_triple(&subject, &predicate, &object).unwrap();
            }
            
            (size, temp_dir, Arc::new(store))
        })
        .collect();
    
    // Benchmark different query patterns
    for (size, _temp_dir, store) in stores.iter() {
        // Query with specific subject
        group.bench_function(BenchmarkId::new("subject_query", size), |b| {
            let subject = Term::iri("http://example.org/subject/12345");
            b.iter(|| {
                let results = store.query_triples(
                    Some(&subject),
                    None,
                    None,
                ).unwrap();
                black_box(results);
            });
        });
        
        // Query with specific predicate (should be selective)
        group.bench_function(BenchmarkId::new("predicate_query", size), |b| {
            let predicate = Term::iri("http://example.org/predicate/42");
            b.iter(|| {
                let results = store.query_triples(
                    None,
                    Some(&predicate),
                    None,
                ).unwrap();
                black_box(results);
            });
        });
        
        // Complex pattern query
        group.bench_function(BenchmarkId::new("pattern_query", size), |b| {
            let predicate = Term::iri("http://example.org/predicate/10");
            let object = Term::literal("value_1000");
            b.iter(|| {
                let results = store.query_triples(
                    None,
                    Some(&predicate),
                    Some(&object),
                ).unwrap();
                black_box(results);
            });
        });
    }
    
    group.finish();
}

/// Benchmark concurrent access patterns
fn bench_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");
    group.measurement_time(Duration::from_secs(10));
    
    let temp_dir = TempDir::new().unwrap();
    let config = TdbConfig {
        location: temp_dir.path().to_string_lossy().to_string(),
        ..Default::default()
    };
    let store = Arc::new(TdbStore::new(config).unwrap());
    
    // Pre-load 100K triples
    let mut rng = rand::thread_rng();
    for _ in 0..100_000 {
        let (subject, predicate, object) = generate_random_triple(&mut rng);
        store.insert_triple(&subject, &predicate, &object).unwrap();
    }
    
    // Benchmark concurrent reads
    group.bench_function("concurrent_reads", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..8)
                .map(|i| {
                    let store = Arc::clone(&store);
                    std::thread::spawn(move || {
                        let subject = Term::iri(&format!("http://example.org/subject/{}", i * 1000));
                        let results = store.query_triples(
                            Some(&subject),
                            None,
                            None,
                        ).unwrap();
                        black_box(results);
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
    
    // Keep temp_dir alive
    let _temp_dir = temp_dir;
    
    group.finish();
}

/// Benchmark MVCC and transaction performance  
fn bench_mvcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("mvcc");
    group.measurement_time(Duration::from_secs(5));
    
    let temp_dir = TempDir::new().unwrap();
    let config = TdbConfig {
        location: temp_dir.path().to_string_lossy().to_string(),
        enable_mvcc: true,
        ..Default::default()
    };
    let store = TdbStore::new(config).unwrap();
    
    // Pre-load some data
    let mut rng = rand::thread_rng();
    for _ in 0..10_000 {
        let (subject, predicate, object) = generate_random_triple(&mut rng);
        store.insert_triple(&subject, &predicate, &object).unwrap();
    }
    
    // Benchmark transaction creation/commit
    group.bench_function("transaction_overhead", |b| {
        b.iter(|| {
            let tx = store.begin_transaction().unwrap();
            store.commit_transaction(tx).unwrap();
        });
    });
    
    // Benchmark read transaction performance
    group.bench_function("read_transaction", |b| {
        let subject = Term::iri("http://example.org/subject/5000");
        b.iter(|| {
            let _tx = store.begin_read_transaction().unwrap();
            let results = store.query_triples(
                Some(&subject),
                None,
                None,
            ).unwrap();
            black_box(results);
            // Transaction dropped automatically
        });
    });
    
    // Keep temp_dir alive
    let _temp_dir = temp_dir;
    
    group.finish();
}

/// Main benchmark suite for testing performance at scale
fn bench_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);
    
    // Test 1M triple insertion performance
    group.bench_function("insert_1M_triples", |b| {
        b.iter_batched_ref(
            || {
                let temp_dir = TempDir::new().unwrap();
                let config = TdbConfig {
                    location: temp_dir.path().to_string_lossy().to_string(),
                    cache_size: 1024 * 1024 * 500, // 500MB cache for large dataset
                    ..Default::default()
                };
                let store = TdbStore::new(config).unwrap();
                (temp_dir, store)
            },
            |(_temp_dir, store)| {
                let mut rng = rand::thread_rng();
                // Insert in batches to avoid overwhelming memory
                for batch in 0..100 {
                    let tx = store.begin_transaction().unwrap();
                    for _ in 0..10_000 {
                        let (subject, predicate, object) = generate_random_triple(&mut rng);
                        store.insert_triple(&subject, &predicate, &object).unwrap();
                    }
                    store.commit_transaction(tx).unwrap();
                    
                    // Report progress
                    if batch % 10 == 0 {
                        println!("Inserted {}K triples", (batch + 1) * 10);
                    }
                }
            },
            BatchSize::PerIteration,
        );
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_insertion,
    bench_bulk_load_transactions,
    bench_query,
    bench_concurrent,
    bench_mvcc,
    bench_large_scale
);
criterion_main!(benches);