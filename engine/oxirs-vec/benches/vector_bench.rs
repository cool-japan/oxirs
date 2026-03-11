//! Vector Search Performance Benchmarks for oxirs-vec
//!
//! v1.0.0 LTS benchmark suite covering:
//! - HNSW-style index: insert 1000 vectors (128-dim)
//! - k-NN search k=10
//! - k-NN search k=100
//! - Batch search (100 queries)
//! - SIMD vs scalar dot product
//! - Full index build for 10k vectors

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_vec::{index::DistanceMetric, MemoryVectorIndex, Vector, VectorIndex};
use std::hint::black_box;
use std::time::Duration;

// --- Helpers ---

/// Generate a deterministic pseudo-random f32 vector (no rand dependency)
fn gen_vector(seed: usize, dim: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(dim);
    let mut state = seed as u64 * 6364136223846793005_u64 + 1442695040888963407;
    for _ in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let float_val = ((state >> 32) as f32) / (u32::MAX as f32);
        values.push(float_val * 2.0 - 1.0); // [-1, 1)
    }
    values
}

/// Normalise a vector to unit length for cosine similarity
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// --- HNSW benchmarks ---

fn bench_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec/hnsw_insert");
    group.measurement_time(Duration::from_secs(15));

    const DIM: usize = 128;

    for insert_count in [100usize, 500, 1_000] {
        let vectors: Vec<Vec<f32>> = (0..insert_count)
            .map(|i| {
                let mut v = gen_vector(i, DIM);
                normalize(&mut v);
                v
            })
            .collect();

        group.throughput(Throughput::Elements(insert_count as u64));
        group.bench_with_input(
            BenchmarkId::new("vectors_128d", insert_count),
            &insert_count,
            |b, &n| {
                b.iter(|| {
                    let mut idx = MemoryVectorIndex::new();
                    for (i, vector) in vectors.iter().enumerate().take(n) {
                        let uri = format!("http://example.org/vec/{i}");
                        let vec = Vector::new(vector.clone());
                        idx.insert(black_box(uri), black_box(vec)).expect("insert");
                    }
                    // Return the vector count via the stored vectors field
                    black_box(n)
                });
            },
        );
    }

    group.finish();
}

fn bench_hnsw_search_k10(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec/hnsw_search_k10");
    group.measurement_time(Duration::from_secs(12));

    const DIM: usize = 128;
    const INDEX_SIZE: usize = 1_000;

    let mut idx = MemoryVectorIndex::new();
    for i in 0..INDEX_SIZE {
        let mut vals = gen_vector(i, DIM);
        normalize(&mut vals);
        let uri = format!("http://example.org/vec/{i}");
        idx.insert(uri, Vector::new(vals)).expect("insert");
    }

    let mut query_vals = gen_vector(99999, DIM);
    normalize(&mut query_vals);
    let query = Vector::new(query_vals);

    group.bench_function("k=10 over 1k vectors", |b| {
        b.iter(|| black_box(idx.search_knn(black_box(&query), 10).expect("search ok")));
    });

    group.finish();
}

fn bench_hnsw_search_k100(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec/hnsw_search_k100");
    group.measurement_time(Duration::from_secs(12));

    const DIM: usize = 128;
    const INDEX_SIZE: usize = 2_000;

    let mut idx = MemoryVectorIndex::new();
    for i in 0..INDEX_SIZE {
        let mut vals = gen_vector(i, DIM);
        normalize(&mut vals);
        let uri = format!("http://example.org/vec/{i}");
        idx.insert(uri, Vector::new(vals)).expect("insert");
    }

    let mut query_vals = gen_vector(88888, DIM);
    normalize(&mut query_vals);
    let query = Vector::new(query_vals);

    group.bench_function("k=100 over 2k vectors", |b| {
        b.iter(|| black_box(idx.search_knn(black_box(&query), 100).expect("search ok")));
    });

    group.finish();
}

fn bench_batch_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec/batch_search");
    group.measurement_time(Duration::from_secs(15));

    const DIM: usize = 128;
    const INDEX_SIZE: usize = 1_000;
    const BATCH_SIZE: usize = 100;

    let mut idx = MemoryVectorIndex::new();
    for i in 0..INDEX_SIZE {
        let mut vals = gen_vector(i, DIM);
        normalize(&mut vals);
        idx.insert(format!("http://example.org/vec/{i}"), Vector::new(vals))
            .expect("insert");
    }

    let queries: Vec<Vector> = (0..BATCH_SIZE)
        .map(|i| {
            let mut vals = gen_vector(i + 1_000_000, DIM);
            normalize(&mut vals);
            Vector::new(vals)
        })
        .collect();

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("100_queries_k10", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                let results = idx.search_knn(black_box(q), 10).expect("search ok");
                total += results.len();
            }
            black_box(total)
        });
    });

    group.finish();
}

fn bench_simd_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec/dot_product");
    group.measurement_time(Duration::from_secs(10));

    // Compare Cosine vs DotProduct vs Euclidean distance via the DistanceMetric API
    for dim in [64usize, 128, 256, 512] {
        let a: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 / dim as f32).collect();

        group.throughput(Throughput::Elements(dim as u64));

        // Scalar dot product
        group.bench_with_input(BenchmarkId::new("scalar_dot", dim), &dim, |bench, _| {
            bench.iter(|| {
                let result: f32 = black_box(&a)
                    .iter()
                    .zip(black_box(&b).iter())
                    .map(|(x, y)| x * y)
                    .sum();
                black_box(result)
            });
        });

        // Cosine distance via DistanceMetric
        let metric = DistanceMetric::Cosine;
        group.bench_with_input(
            BenchmarkId::new("cosine_distance", dim),
            &dim,
            |bench, _| {
                bench.iter(|| black_box(metric.distance(black_box(&a), black_box(&b))));
            },
        );

        // Euclidean distance via DistanceMetric
        let metric = DistanceMetric::Euclidean;
        group.bench_with_input(
            BenchmarkId::new("euclidean_distance", dim),
            &dim,
            |bench, _| {
                bench.iter(|| black_box(metric.distance(black_box(&a), black_box(&b))));
            },
        );
    }

    group.finish();
}

fn bench_index_build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec/index_build");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    const DIM: usize = 128;

    for vector_count in [1_000usize, 5_000, 10_000] {
        let vectors: Vec<Vec<f32>> = (0..vector_count)
            .map(|i| {
                let mut v = gen_vector(i, DIM);
                normalize(&mut v);
                v
            })
            .collect();

        group.throughput(Throughput::Elements(vector_count as u64));
        group.bench_with_input(
            BenchmarkId::new("vectors_128d", vector_count),
            &vector_count,
            |b, &n| {
                b.iter(|| {
                    let mut idx = MemoryVectorIndex::new();
                    for (i, vector) in vectors.iter().enumerate().take(n) {
                        idx.insert(
                            format!("http://example.org/v/{i}"),
                            Vector::new(vector.clone()),
                        )
                        .expect("insert");
                    }
                    black_box(n)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_insert,
    bench_hnsw_search_k10,
    bench_hnsw_search_k100,
    bench_batch_search,
    bench_simd_dot_product,
    bench_index_build_time,
);
criterion_main!(benches);
