//! Storage Engine Benchmarks for oxirs-tdb
//!
//! v1.0.0 LTS benchmark suite covering:
//! - Triple insert rate (10k triples)
//! - Point lookup by subject
//! - Range scan over predicate index
//! - Bulk load of 100k triples
//! - Compression (delta store)
//! - Full-text query over literals

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_tdb::{
    compression::{DeltaStoreConfig, TripleDeltaStore},
    index::EncodedTriple,
    store::{TdbConfig, TdbStore},
};
use std::hint::black_box;
use std::time::Duration;

// --- Helpers ---

fn tmp_store() -> (TdbStore, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let cfg = TdbConfig::new(dir.path())
        .with_compression(true)
        .with_bloom_filters(true);
    let store = TdbStore::open_with_config(cfg).expect("open TdbStore");
    (store, dir)
}

fn populate_store(store: &mut TdbStore, count: usize) {
    for i in 0..count {
        store
            .insert(
                &format!("http://example.org/s/{i}"),
                &format!("http://example.org/p/{}", i % 20),
                &format!("http://example.org/o/{}", i % 100),
            )
            .expect("insert triple");
    }
}

// --- Benchmarks ---

fn bench_triple_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/triple_insert");
    group.measurement_time(Duration::from_secs(15));

    for insert_count in [1_000usize, 5_000, 10_000] {
        group.throughput(Throughput::Elements(insert_count as u64));
        group.bench_with_input(
            BenchmarkId::new("triples", insert_count),
            &insert_count,
            |b, &n| {
                b.iter(|| {
                    let (mut store, _dir) = tmp_store();
                    for i in 0..n {
                        store
                            .insert(
                                black_box(&format!("http://example.org/s/{i}")),
                                black_box(&format!("http://example.org/p/{}", i % 20)),
                                black_box(&format!("http://example.org/o/{}", i % 100)),
                            )
                            .expect("insert");
                    }
                    store.count()
                });
            },
        );
    }

    group.finish();
}

fn bench_triple_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/triple_lookup");
    group.measurement_time(Duration::from_secs(10));

    let (mut store, _dir) = tmp_store();
    populate_store(&mut store, 10_000);

    for lookup_index in [0usize, 2500, 5000, 7500] {
        group.bench_with_input(
            BenchmarkId::new("by_subject_index", lookup_index),
            &lookup_index,
            |b, &idx| {
                let subject = format!("http://example.org/s/{idx}");
                b.iter(|| {
                    black_box(
                        store
                            .contains(
                                black_box(&subject),
                                black_box("http://example.org/p/0"),
                                black_box("http://example.org/o/0"),
                            )
                            .expect("contains check"),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_range_scan(c: &mut Criterion) {
    use oxirs_tdb::dictionary::Term;

    let mut group = c.benchmark_group("storage/range_scan");
    group.measurement_time(Duration::from_secs(12));

    let (mut store, _dir) = tmp_store();
    populate_store(&mut store, 10_000);

    for predicate_mod in [0usize, 5, 10, 15] {
        group.bench_with_input(
            BenchmarkId::new("predicate_scan", predicate_mod),
            &predicate_mod,
            |b, &pred_mod| {
                let predicate = Term::iri(format!("http://example.org/p/{pred_mod}"));
                b.iter(|| {
                    let results = store
                        .query_triples(None, Some(black_box(&predicate)), None)
                        .expect("query_triples");
                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

fn bench_bulk_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/bulk_load");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    for bulk_size in [10_000usize, 50_000, 100_000] {
        group.throughput(Throughput::Elements(bulk_size as u64));
        group.bench_with_input(
            BenchmarkId::new("triples", bulk_size),
            &bulk_size,
            |b, &n| {
                // Build data outside the timed loop
                let triples: Vec<(String, String, String)> = (0..n)
                    .map(|i| {
                        (
                            format!("http://example.org/bulk/s/{i}"),
                            format!("http://example.org/bulk/p/{}", i % 50),
                            format!("http://example.org/bulk/o/{}", i % 200),
                        )
                    })
                    .collect();

                b.iter(|| {
                    let (mut store, _dir) = tmp_store();
                    for (s, p, o) in &triples {
                        store
                            .insert(black_box(s), black_box(p), black_box(o))
                            .expect("insert");
                    }
                    store.count()
                });
            },
        );
    }

    group.finish();
}

fn bench_compression_delta_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/delta_store");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    // Benchmark delta store insert throughput (compressed incremental store)
    for entry_count in [1_000usize, 5_000, 10_000] {
        group.throughput(Throughput::Elements(entry_count as u64));
        group.bench_with_input(
            BenchmarkId::new("delta_inserts", entry_count),
            &entry_count,
            |b, &n| {
                // Pre-generate encoded triples outside the timed loop
                let encoded: Vec<EncodedTriple> = (0..n)
                    .map(|i| {
                        let s = (i % (n / 10).max(1)) as u64;
                        let p = 1u64; // shared predicate = good compression
                        let o = (i % 10) as u64;
                        EncodedTriple::new(s, p, o)
                    })
                    .collect();

                b.iter(|| {
                    let cfg = DeltaStoreConfig::default();
                    let mut delta_store = TripleDeltaStore::new(cfg);
                    for triple in &encoded {
                        delta_store
                            .insert(black_box(*triple))
                            .expect("delta insert");
                    }
                    black_box(delta_store.stats().expect("stats ok"))
                });
            },
        );
    }

    group.finish();
}

fn bench_full_text_search(c: &mut Criterion) {
    use oxirs_tdb::dictionary::Term;

    let mut group = c.benchmark_group("storage/full_text_search");
    group.measurement_time(Duration::from_secs(15));

    // Populate a store with text-like triples (using URIs as values)
    let (mut store, _dir) = tmp_store();
    let subject_prefix = "http://example.org/doc";
    let predicate = "http://schema.org/description";

    for i in 0..1_000usize {
        let obj_uri = format!("http://example.org/object/{}", i % 100);
        store
            .insert(&format!("{subject_prefix}/{i}"), predicate, &obj_uri)
            .expect("insert text triple");
    }

    // Scan triples by predicate (simulating full-text search over a predicate)
    let pred_term = Term::iri(predicate);
    group.bench_function("scan_by_predicate_1k_docs", |b| {
        b.iter(|| {
            let results = store
                .query_triples(None, Some(black_box(&pred_term)), None)
                .expect("query ok");
            black_box(results.len())
        });
    });

    // Lookup by specific object URI
    for obj_num in [0usize, 25, 50, 75] {
        let obj_uri = format!("http://example.org/object/{obj_num}");
        let obj_term = Term::iri(obj_uri.clone());
        group.bench_with_input(
            BenchmarkId::new("lookup_by_object", obj_num),
            &obj_num,
            |b, _| {
                b.iter(|| {
                    let results = store
                        .query_triples(None, None, Some(black_box(&obj_term)))
                        .expect("query ok");
                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_triple_insert,
    bench_triple_lookup,
    bench_range_scan,
    bench_bulk_load,
    bench_compression_delta_store,
    bench_full_text_search,
);
criterion_main!(benches);
