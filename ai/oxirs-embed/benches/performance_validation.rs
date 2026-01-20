//! Performance benchmarks for scirs2 integration optimizations
//!
//! This benchmark suite validates that our scirs2 integration provides
//! significant performance improvements over baseline implementations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_embed::models::common::*;
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::Random;

/// Benchmark vectorized vs non-vectorized distance computations
fn bench_distance_computations(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_computations");

    // Setup test data
    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        let vectors: Vec<Array1<f64>> = (0..size)
            .map(|i| Array1::from_vec(vec![i as f64; 128]))
            .collect();

        // Benchmark optimized pairwise distances
        group.bench_with_input(
            BenchmarkId::new("optimized_pairwise_distances", size),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    black_box(pairwise_distances(vectors));
                });
            },
        );

        // Benchmark optimized L2 distance
        group.bench_with_input(
            BenchmarkId::new("optimized_l2_distance", size),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    for i in 0..vectors.len().min(100) {
                        for j in (i + 1)..vectors.len().min(100) {
                            black_box(l2_distance(&vectors[i], &vectors[j]));
                        }
                    }
                });
            },
        );

        // Benchmark optimized cosine similarity
        group.bench_with_input(
            BenchmarkId::new("optimized_cosine_similarity", size),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    for i in 0..vectors.len().min(100) {
                        for j in (i + 1)..vectors.len().min(100) {
                            black_box(cosine_similarity(&vectors[i], &vectors[j]));
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark gradient update optimizations
fn bench_gradient_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_updates");

    let sizes = vec![100, 500, 1000];

    for size in sizes {
        let _embeddings: Vec<Array2<f64>> = (0..10).map(|_| Array2::zeros((size, 128))).collect();

        let _gradients: Vec<Array2<f64>> = (0..10).map(|_| Array2::ones((size, 128))).collect();

        // Benchmark batch gradient update
        group.bench_with_input(
            BenchmarkId::new("batch_gradient_update", size),
            &size,
            |b, &size| {
                let mut embeddings: Vec<Array2<f64>> = (0..size)
                    .map(|_| Array2::from_elem((128, 64), 0.1))
                    .collect();
                let gradients: Vec<Array2<f64>> = (0..size)
                    .map(|_| Array2::from_elem((128, 64), 0.01))
                    .collect();

                b.iter(|| {
                    batch_gradient_update(&mut embeddings, &gradients, 0.01, 0.001);
                    black_box(());
                });
            },
        );

        // Benchmark individual gradient updates
        group.bench_with_input(
            BenchmarkId::new("individual_gradient_updates", size),
            &size,
            |b, &size| {
                let mut embeddings: Vec<Array2<f64>> = (0..size)
                    .map(|_| Array2::from_elem((128, 64), 0.1))
                    .collect();
                let gradients: Vec<Array2<f64>> = (0..size)
                    .map(|_| Array2::from_elem((128, 64), 0.01))
                    .collect();

                b.iter(|| {
                    for (embedding, gradient) in embeddings.iter_mut().zip(gradients.iter()) {
                        gradient_update(embedding, gradient, 0.01, 0.001);
                        black_box(());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark random sampling optimizations
fn bench_sampling_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_optimizations");

    let sizes = vec![1000, 5000, 10000];

    for size in sizes {
        let data: Vec<u32> = (0..size).collect();
        let mut rng = Random::default();

        // Benchmark optimized sampling without replacement
        group.bench_with_input(
            BenchmarkId::new("optimized_sampling_without_replacement", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(sample_without_replacement(
                        data,
                        (size / 10) as usize,
                        &mut rng,
                    ));
                });
            },
        );

        // Benchmark optimized batch shuffling
        group.bench_with_input(
            BenchmarkId::new("optimized_batch_shuffle", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut batch = data.clone();
                    shuffle_batch(&mut batch, &mut rng);
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark embedding initialization optimizations
fn bench_embedding_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_initialization");

    let sizes = vec![(100, 128), (500, 256), (1000, 512)];

    for (rows, cols) in sizes {
        let mut rng = Random::default();

        // Benchmark batch Xavier initialization
        group.bench_with_input(
            BenchmarkId::new("batch_xavier_init", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                b.iter(|| {
                    let shapes = vec![(rows, cols); 10];
                    black_box(batch_xavier_init(&shapes, rows, cols, &mut rng));
                });
            },
        );

        // Benchmark individual Xavier initialization
        group.bench_with_input(
            BenchmarkId::new("individual_xavier_init", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                b.iter(|| {
                    for _ in 0..10 {
                        black_box(xavier_init((rows, cols), rows, cols, &mut rng));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch processing optimizations
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    let sizes = vec![1000, 5000, 10000];
    let batch_size = 32;

    for size in sizes {
        let data: Vec<u32> = (0..size).collect();

        // Benchmark zero-copy batch references
        group.bench_with_input(
            BenchmarkId::new("zero_copy_batch_refs", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let batches = create_batch_refs(data, batch_size);
                    let count = batches.count();
                    black_box(count);
                });
            },
        );

        // Benchmark optimized batch creation with pre-allocation
        group.bench_with_input(
            BenchmarkId::new("optimized_batch_creation", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(create_batches(data, batch_size));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_computations,
    bench_gradient_updates,
    bench_sampling_optimizations,
    bench_embedding_initialization,
    bench_batch_processing
);

criterion_main!(benches);
