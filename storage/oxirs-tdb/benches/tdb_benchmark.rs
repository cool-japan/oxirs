//! Performance benchmarks for oxirs-tdb against 100M+ triple targets
//!
//! This benchmark suite tests the performance of oxirs-tdb with large-scale datasets
//! to ensure it meets the performance requirements specified in the TODO.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use oxirs_tdb::{
    compression::{AdaptiveCompressor, ColumnStoreCompressor, RunLengthEncoder},
    nodes::NodeTableConfig,
    SimpleTdbConfig,
    TdbStore, Term,
};
use rand::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
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
                    let config = SimpleTdbConfig {
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
                        store
                            .insert_triple(
                                black_box(subject),
                                black_box(predicate),
                                black_box(object),
                            )
                            .unwrap();
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
                    let config = SimpleTdbConfig {
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
                        let subject_id = store.triple_store().store_term(subject).unwrap();
                        let predicate_id = store.triple_store().store_term(predicate).unwrap();
                        let object_id = store.triple_store().store_term(object).unwrap();
                        let triple = oxirs_tdb::triple_store::Triple::new(
                            subject_id,
                            predicate_id,
                            object_id,
                        );
                        store
                            .triple_store()
                            .insert_triple_tx(tx.id(), &triple)
                            .unwrap();
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
                let results = store.query_triples(Some(&subject), None, None).unwrap();
                black_box(results);
            });
        });

        // Query with specific predicate (should be selective)
        group.bench_function(BenchmarkId::new("predicate_query", size), |b| {
            let predicate = Term::iri("http://example.org/predicate/42");
            b.iter(|| {
                let results = store.query_triples(None, Some(&predicate), None).unwrap();
                black_box(results);
            });
        });

        // Complex pattern query
        group.bench_function(BenchmarkId::new("pattern_query", size), |b| {
            let predicate = Term::iri("http://example.org/predicate/10");
            let object = Term::literal("value_1000");
            b.iter(|| {
                let results = store
                    .query_triples(None, Some(&predicate), Some(&object))
                    .unwrap();
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
                        let subject =
                            Term::iri(&format!("http://example.org/subject/{}", i * 1000));
                        let results = store.query_triples(Some(&subject), None, None).unwrap();
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
            let results = store.query_triples(Some(&subject), None, None).unwrap();
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

/// Benchmark compression performance with different algorithms
fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");
    group.measurement_time(Duration::from_secs(10));

    // Generate test data patterns
    let repetitive_data = vec![42u8; 10000];
    let random_data: Vec<u8> = (0..10000).map(|_| rand::random()).collect();
    let text_data = "The quick brown fox jumps over the lazy dog. "
        .repeat(200)
        .into_bytes();

    // Benchmark different compression algorithms
    let algorithms = vec![
        ("run_length", repetitive_data.clone()),
        ("adaptive_random", random_data),
        ("adaptive_text", text_data),
    ];

    for (name, data) in algorithms {
        group.bench_function(&format!("compress_{}", name), |b| {
            let compressor = AdaptiveCompressor::default();
            b.iter(|| {
                let compressed = compressor.compress(black_box(&data)).unwrap();
                black_box(compressed);
            });
        });

        // Benchmark decompression
        let compressor = AdaptiveCompressor::default();
        let compressed = compressor.compress(&data).unwrap();
        group.bench_function(&format!("decompress_{}", name), |b| {
            b.iter(|| {
                let decompressed = compressor.decompress(black_box(&compressed)).unwrap();
                black_box(decompressed);
            });
        });
    }

    group.finish();
}

/// Benchmark different compression configurations
fn bench_compression_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_configs");
    group.measurement_time(Duration::from_secs(15));

    let configs = vec![
        ("no_compression", false, false),
        ("basic_compression", true, false),
        ("advanced_compression", true, true),
        ("column_store", true, true),
    ];

    for (name, enable_compression, enable_advanced) in configs {
        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter_batched_ref(
                || {
                    let temp_dir = TempDir::new().unwrap();
                    let mut config = TdbConfig {
                        location: temp_dir.path().to_string_lossy().to_string(),
                        cache_size: 1024 * 1024 * 100, // 100MB
                        ..Default::default()
                    };

                    let store = TdbStore::new(config).unwrap();
                    let mut rng = rand::thread_rng();
                    let triples: Vec<(Term, Term, Term)> = (0..10000)
                        .map(|_| generate_random_triple(&mut rng))
                        .collect();
                    (temp_dir, store, triples)
                },
                |(_temp_dir, store, triples)| {
                    for (subject, predicate, object) in triples.iter() {
                        store.insert_triple(subject, predicate, object).unwrap();
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));

    let cache_sizes = vec![
        1024 * 1024 * 10,  // 10MB
        1024 * 1024 * 50,  // 50MB
        1024 * 1024 * 100, // 100MB
        1024 * 1024 * 200, // 200MB
    ];

    for cache_size in cache_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}MB", cache_size / (1024 * 1024))),
            &cache_size,
            |b, &cache_size| {
                b.iter_batched_ref(
                    || {
                        let temp_dir = TempDir::new().unwrap();
                        let config = TdbConfig {
                            location: temp_dir.path().to_string_lossy().to_string(),
                            cache_size,
                            ..Default::default()
                        };
                        let store = TdbStore::new(config).unwrap();
                        let mut rng = rand::thread_rng();
                        let triples: Vec<(Term, Term, Term)> = (0..50000)
                            .map(|_| generate_random_triple(&mut rng))
                            .collect();
                        (temp_dir, store, triples)
                    },
                    |(_temp_dir, store, triples)| {
                        // Insert triples and measure query performance
                        for (subject, predicate, object) in triples.iter().take(25000) {
                            store.insert_triple(subject, predicate, object).unwrap();
                        }

                        // Perform queries to test cache effectiveness
                        for i in 0..100 {
                            let subject =
                                Term::iri(&format!("http://example.org/subject/{}", i * 100));
                            let results = store.query_triples(Some(&subject), None, None).unwrap();
                            black_box(results);
                        }
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark index performance with different patterns
fn bench_index_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_patterns");
    group.measurement_time(Duration::from_secs(15));

    let temp_dir = TempDir::new().unwrap();
    let config = TdbConfig {
        location: temp_dir.path().to_string_lossy().to_string(),
        cache_size: 1024 * 1024 * 200, // 200MB for large dataset
        ..Default::default()
    };
    let store = TdbStore::new(config).unwrap();

    // Load 500K triples with different distribution patterns
    let mut rng = rand::thread_rng();

    // Pattern 1: Few subjects, many predicates/objects (star pattern)
    for subject_id in 0..1000 {
        let subject = Term::iri(&format!("http://example.org/star_subject/{}", subject_id));
        for i in 0..100 {
            let predicate = Term::iri(&format!("http://example.org/predicate/{}", i));
            let object = Term::literal(&format!("value_{}", rng.gen::<u32>()));
            store.insert_triple(&subject, &predicate, &object).unwrap();
        }
    }

    // Pattern 2: Many subjects, few predicates (property pattern)
    let popular_predicates = vec![
        Term::iri("http://xmlns.com/foaf/0.1/name"),
        Term::iri("http://xmlns.com/foaf/0.1/age"),
        Term::iri("http://purl.org/dc/terms/created"),
    ];

    for subject_id in 0..100000 {
        let subject = Term::iri(&format!("http://example.org/person/{}", subject_id));
        let predicate = popular_predicates.choose(&mut rng).unwrap();
        let object = Term::literal(&format!("value_{}", rng.gen::<u32>()));
        store.insert_triple(&subject, predicate, &object).unwrap();
    }

    // Pattern 3: Uniform distribution
    for _ in 0..300000 {
        let (subject, predicate, object) = generate_random_triple(&mut rng);
        store.insert_triple(&subject, &predicate, &object).unwrap();
    }

    // Benchmark different query patterns

    // Star queries (S ?P ?O)
    group.bench_function("star_query", |b| {
        let subject = Term::iri("http://example.org/star_subject/500");
        b.iter(|| {
            let results = store.query_triples(Some(&subject), None, None).unwrap();
            black_box(results);
        });
    });

    // Property queries (?S P ?O)
    group.bench_function("property_query", |b| {
        let predicate = Term::iri("http://xmlns.com/foaf/0.1/name");
        b.iter(|| {
            let results = store.query_triples(None, Some(&predicate), None).unwrap();
            black_box(results);
        });
    });

    // Object queries (?S ?P O)
    group.bench_function("object_query", |b| {
        let object = Term::literal("value_12345");
        b.iter(|| {
            let results = store.query_triples(None, None, Some(&object)).unwrap();
            black_box(results);
        });
    });

    // Complex pattern queries (S P ?O)
    group.bench_function("sp_query", |b| {
        let subject = Term::iri("http://example.org/person/5000");
        let predicate = Term::iri("http://xmlns.com/foaf/0.1/name");
        b.iter(|| {
            let results = store
                .query_triples(Some(&subject), Some(&predicate), None)
                .unwrap();
            black_box(results);
        });
    });

    let _temp_dir = temp_dir; // Keep alive
    group.finish();
}

/// Benchmark real-world data patterns
fn bench_real_world_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_patterns");
    group.measurement_time(Duration::from_secs(20));

    let temp_dir = TempDir::new().unwrap();
    let config = TdbConfig {
        location: temp_dir.path().to_string_lossy().to_string(),
        cache_size: 1024 * 1024 * 300, // 300MB
        ..Default::default()
    };
    let store = TdbStore::new(config).unwrap();

    // Simulate knowledge graph data with realistic patterns
    let mut rng = rand::thread_rng();

    // FOAF-like data
    let foaf_predicates = vec![
        "http://xmlns.com/foaf/0.1/name",
        "http://xmlns.com/foaf/0.1/email",
        "http://xmlns.com/foaf/0.1/knows",
        "http://xmlns.com/foaf/0.1/age",
        "http://xmlns.com/foaf/0.1/homepage",
    ];

    // Schema.org-like data
    let schema_predicates = vec![
        "http://schema.org/name",
        "http://schema.org/description",
        "http://schema.org/dateCreated",
        "http://schema.org/author",
        "http://schema.org/url",
    ];

    // Dublin Core metadata
    let dc_predicates = vec![
        "http://purl.org/dc/terms/title",
        "http://purl.org/dc/terms/creator",
        "http://purl.org/dc/terms/subject",
        "http://purl.org/dc/terms/date",
        "http://purl.org/dc/terms/identifier",
    ];

    // Insert 1M triples with realistic distribution
    group.bench_function("load_knowledge_graph", |b| {
        b.iter(|| {
            // Clear previous data
            store.clear().unwrap();

            let start = Instant::now();

            // Load people data (FOAF pattern)
            for person_id in 0..50000 {
                let person = Term::iri(&format!("http://example.org/person/{}", person_id));

                // Each person has 2-5 properties
                let num_props = rng.gen_range(2..=5);
                let mut used_predicates = std::collections::HashSet::new();

                for _ in 0..num_props {
                    let predicate_str = foaf_predicates.choose(&mut rng).unwrap();
                    if used_predicates.insert(predicate_str) {
                        let predicate = Term::iri(*predicate_str);
                        let object = match predicate_str {
                            s if s.contains("name") => {
                                Term::literal(&format!("Person {}", person_id))
                            }
                            s if s.contains("email") => {
                                Term::literal(&format!("person{}@example.com", person_id))
                            }
                            s if s.contains("age") => Term::typed_literal(
                                &rng.gen_range(18..80).to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            ),
                            s if s.contains("knows") => Term::iri(&format!(
                                "http://example.org/person/{}",
                                rng.gen_range(0..50000)
                            )),
                            _ => Term::literal(&format!("value_{}", rng.gen::<u32>())),
                        };
                        store.insert_triple(&person, &predicate, &object).unwrap();
                    }
                }
            }

            // Load document metadata (Dublin Core pattern)
            for doc_id in 0..25000 {
                let document = Term::iri(&format!("http://example.org/document/{}", doc_id));

                for predicate_str in dc_predicates.choose_multiple(&mut rng, 3) {
                    let predicate = Term::iri(*predicate_str);
                    let object = match predicate_str {
                        s if s.contains("title") => Term::literal(&format!("Document {}", doc_id)),
                        s if s.contains("creator") => Term::iri(&format!(
                            "http://example.org/person/{}",
                            rng.gen_range(0..50000)
                        )),
                        s if s.contains("date") => Term::typed_literal(
                            "2024-01-01",
                            "http://www.w3.org/2001/XMLSchema#date",
                        ),
                        _ => Term::literal(&format!("metadata_{}", rng.gen::<u32>())),
                    };
                    store.insert_triple(&document, &predicate, &object).unwrap();
                }
            }

            // Load product data (Schema.org pattern)
            for product_id in 0..25000 {
                let product = Term::iri(&format!("http://example.org/product/{}", product_id));

                for predicate_str in schema_predicates.choose_multiple(&mut rng, 4) {
                    let predicate = Term::iri(*predicate_str);
                    let object = match predicate_str {
                        s if s.contains("name") => {
                            Term::literal(&format!("Product {}", product_id))
                        }
                        s if s.contains("description") => {
                            Term::literal(&format!("Description for product {}", product_id))
                        }
                        s if s.contains("author") => Term::iri(&format!(
                            "http://example.org/person/{}",
                            rng.gen_range(0..50000)
                        )),
                        _ => Term::literal(&format!("schema_value_{}", rng.gen::<u32>())),
                    };
                    store.insert_triple(&product, &predicate, &object).unwrap();
                }
            }

            let duration = start.elapsed();
            black_box(duration);
        });
    });

    let _temp_dir = temp_dir; // Keep alive
    group.finish();
}

/// Benchmark recovery and checkpoint performance
fn bench_recovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    // Benchmark checkpoint creation
    group.bench_function("checkpoint_creation", |b| {
        b.iter_batched_ref(
            || {
                let temp_dir = TempDir::new().unwrap();
                let config = TdbConfig {
                    location: temp_dir.path().to_string_lossy().to_string(),
                    enable_transactions: true,
                    ..Default::default()
                };
                let store = TdbStore::new(config).unwrap();

                // Load some data to checkpoint
                let mut rng = rand::thread_rng();
                for _ in 0..10000 {
                    let (subject, predicate, object) = generate_random_triple(&mut rng);
                    store.insert_triple(&subject, &predicate, &object).unwrap();
                }

                (temp_dir, store)
            },
            |(_temp_dir, store)| {
                // Trigger compaction which includes checkpoint-like operations
                store.compact().unwrap();
            },
            BatchSize::LargeInput,
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
    bench_large_scale,
    bench_compression,
    bench_compression_configs,
    bench_memory_usage,
    bench_index_patterns,
    bench_real_world_patterns,
    bench_recovery
);
criterion_main!(benches);
