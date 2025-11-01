//! Benchmarks for v0.1.0 advanced features
//!
//! This benchmark suite focuses on the new features added in v0.1.0:
//! - Zero-copy RDF operations
//! - ACID transactions with WAL
//! - Enhanced concurrent index updates
//! - SIMD triple matching
//! - Query plan caching

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_core::concurrent::lock_free_graph::ConcurrentGraph;
use oxirs_core::model::{
    pattern::{PredicatePattern, SubjectPattern, TriplePattern},
    Literal, NamedNode, Object, Predicate, Subject, Triple,
};
use oxirs_core::simd_triple_matching::SimdTripleMatcher;
use oxirs_core::zero_copy_rdf::ZeroCopyTripleStore;
use std::sync::Arc;
use tempfile::tempdir;

/// Create test triples for benchmarking
fn create_test_triples(count: usize) -> Vec<Triple> {
    (0..count)
        .map(|i| {
            Triple::new(
                Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", i)).unwrap()),
                Predicate::NamedNode(
                    NamedNode::new(format!("http://example.org/p{}", i % 10)).unwrap(),
                ),
                Object::Literal(Literal::new(format!("value{}", i))),
            )
        })
        .collect()
}

/// Create N-Triples data for parsing benchmarks
fn create_ntriples_data(count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0..count {
        let line = format!(
            "<http://example.org/s{}> <http://example.org/p{}> \"value{}\" .\n",
            i,
            i % 10,
            i
        );
        data.extend_from_slice(line.as_bytes());
    }
    data
}

// Zero-Copy RDF Benchmarks

fn bench_zero_copy_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_insert");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut store = ZeroCopyTripleStore::new().unwrap();
                let triples = create_test_triples(size);
                for triple in triples {
                    store.insert_zero_copy(black_box(triple)).unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_zero_copy_bulk_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_bulk_insert");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut store = ZeroCopyTripleStore::new().unwrap();
                let triples = create_test_triples(size);
                store.bulk_insert_zero_copy(black_box(triples)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_zero_copy_load_file(c: &mut Criterion) {
    use std::fs;
    use std::io::Write;

    let mut group = c.benchmark_group("zero_copy_load_file");

    for size in [100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let data = create_ntriples_data(*size);

        // Create temporary file
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.nt");
        let mut file = fs::File::create(&file_path).unwrap();
        file.write_all(&data).unwrap();
        drop(file);

        group.bench_with_input(BenchmarkId::from_parameter(size), &file_path, |b, path| {
            b.iter(|| {
                let mut store = ZeroCopyTripleStore::new().unwrap();
                store.load_file_zero_copy(black_box(path)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_zero_copy_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_query");

    // Setup: Pre-populate store
    let mut store = ZeroCopyTripleStore::new().unwrap();
    let triples = create_test_triples(10000);
    store.bulk_insert_zero_copy(triples).unwrap();

    group.bench_function("match_all", |b| {
        b.iter(|| {
            let results = store.query_zero_copy(None, None, None).unwrap();
            black_box(results);
        });
    });

    let predicate =
        Predicate::NamedNode(NamedNode::new("http://example.org/p1".to_string()).unwrap());
    group.bench_function("match_predicate", |b| {
        b.iter(|| {
            let results = store.query_zero_copy(None, Some(&predicate), None).unwrap();
            black_box(results);
        });
    });

    group.bench_function("query_indices", |b| {
        b.iter(|| {
            let indices = store.query_indices(None, Some(&predicate), None).unwrap();
            black_box(indices);
        });
    });

    group.finish();
}

// Concurrent Index Updates Benchmarks

fn bench_concurrent_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_batch_insert");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let graph = ConcurrentGraph::new();
                let triples = create_test_triples(size);
                graph.insert_batch(black_box(triples)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_concurrent_rebuild_indices(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_rebuild_indices");

    for size in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let graph = ConcurrentGraph::new();
        let triples = create_test_triples(*size);
        graph.insert_batch(triples).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &graph, |b, graph| {
            b.iter(|| {
                graph.rebuild_indices().unwrap();
            });
        });
    }
    group.finish();
}

fn bench_concurrent_parallel_query(c: &mut Criterion) {
    use std::thread;

    let mut group = c.benchmark_group("concurrent_parallel_query");

    let graph = Arc::new(ConcurrentGraph::new());
    let triples = create_test_triples(10000);
    graph.insert_batch(triples).unwrap();

    group.bench_function("4_threads_concurrent_query", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let graph = graph.clone();
                    thread::spawn(move || {
                        let _results = graph.match_pattern(None, None, None);
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

// SIMD Triple Matching Benchmarks

fn bench_simd_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_pattern_matching");

    let matcher = SimdTripleMatcher::new();
    let triples = create_test_triples(10000);

    let subject = NamedNode::new("http://example.org/s100".to_string()).unwrap();
    let pattern = TriplePattern::new(Some(SubjectPattern::NamedNode(subject)), None, None);

    group.bench_function("simd_match_subject", |b| {
        b.iter(|| {
            let _results = matcher.match_batch(&pattern, &triples);
        });
    });

    let predicate = NamedNode::new("http://example.org/p1".to_string()).unwrap();
    let pattern = TriplePattern::new(None, Some(PredicatePattern::NamedNode(predicate)), None);
    group.bench_function("simd_match_predicate", |b| {
        b.iter(|| {
            let _results = matcher.match_batch(&pattern, &triples);
        });
    });

    group.finish();
}

fn bench_simd_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_sequential");

    let matcher = SimdTripleMatcher::new();
    let triples = create_test_triples(10000);

    let target_predicate =
        Predicate::NamedNode(NamedNode::new("http://example.org/p1".to_string()).unwrap());

    group.bench_function("sequential_match", |b| {
        b.iter(|| {
            let results: Vec<_> = triples
                .iter()
                .filter(|t| t.predicate() == &target_predicate)
                .cloned()
                .collect();
            black_box(results);
        });
    });

    let predicate = NamedNode::new("http://example.org/p1".to_string()).unwrap();
    let pattern = TriplePattern::new(None, Some(PredicatePattern::NamedNode(predicate)), None);
    group.bench_function("simd_match", |b| {
        b.iter(|| {
            let _results = matcher.match_batch(&pattern, &triples);
        });
    });

    group.finish();
}

// Transaction Benchmarks

fn bench_transaction_commit(c: &mut Criterion) {
    use oxirs_core::transaction::{IsolationLevel, TransactionManager};

    let mut group = c.benchmark_group("transaction_commit");

    let dir = tempdir().unwrap();

    group.bench_function("snapshot_isolation", |b| {
        b.iter(|| {
            let mut tx_mgr = TransactionManager::new(dir.path()).unwrap();
            let mut tx = tx_mgr.begin(IsolationLevel::Snapshot).unwrap();
            // Simulate some work
            black_box(&mut tx);
            tx.commit().unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    zero_copy_benches,
    bench_zero_copy_insert,
    bench_zero_copy_bulk_insert,
    bench_zero_copy_load_file,
    bench_zero_copy_query
);

criterion_group!(
    concurrent_benches,
    bench_concurrent_batch_insert,
    bench_concurrent_rebuild_indices,
    bench_concurrent_parallel_query
);

criterion_group!(
    simd_benches,
    bench_simd_pattern_matching,
    bench_simd_vs_sequential
);

criterion_group!(transaction_benches, bench_transaction_commit);

criterion_main!(
    zero_copy_benches,
    concurrent_benches,
    simd_benches,
    transaction_benches
);
