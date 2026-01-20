//! GPU-Accelerated SPARQL Operations Benchmarking Suite
//!
//! Beta.2+ Feature: GPU-Accelerated Vector Operations Performance Testing
//!
//! This benchmark suite measures the performance of GPU-accelerated operations:
//! - Vector similarity search with varying dataset sizes
//! - Cache effectiveness and hit rates
//! - Batch processing throughput
//! - Configuration impact on performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_arq::gpu_accelerated_ops::{GpuConfig, GpuQueryEngine};
use scirs2_core::ndarray_ext::{Array1, Array2};
use std::time::Duration;

/// Generate synthetic embedding vectors for benchmarking
fn generate_embeddings(n_vectors: usize, dim: usize) -> Array2<f32> {
    let mut data = Vec::with_capacity(n_vectors * dim);
    for i in 0..n_vectors {
        for j in 0..dim {
            // Generate deterministic but varied values
            let val = ((i * 31 + j * 17) % 100) as f32 / 100.0;
            data.push(val);
        }
    }
    Array2::from_shape_vec((n_vectors, dim), data).unwrap()
}

/// Generate query vector
fn generate_query(dim: usize) -> Array1<f32> {
    let data: Vec<f32> = (0..dim).map(|i| (i % 10) as f32 / 10.0).collect();
    Array1::from_vec(data)
}

/// Benchmark vector similarity search with varying dataset sizes
fn bench_vector_similarity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_similarity_scaling");
    group.measurement_time(Duration::from_secs(15));

    let embedding_dim = 128;
    let sizes = vec![100, 500, 1_000, 5_000];

    for size in sizes {
        let embeddings = generate_embeddings(size, embedding_dim);
        let query = generate_query(embedding_dim);

        let config = GpuConfig::auto_detect();
        let engine = GpuQueryEngine::new(config).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("top_10", size),
            &(embeddings, query),
            |b, (embeddings, query)| {
                b.iter(|| {
                    black_box(
                        engine
                            .vector_similarity_search(embeddings, query, 10)
                            .unwrap(),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cache effectiveness
fn bench_cache_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_cache_effectiveness");
    group.measurement_time(Duration::from_secs(10));

    let embedding_dim = 64;
    let n_vectors = 1_000;

    let embeddings = generate_embeddings(n_vectors, embedding_dim);
    let query = generate_query(embedding_dim);

    // Test with cache enabled
    let config_cached = GpuConfig::auto_detect();
    let engine_cached = GpuQueryEngine::new(config_cached).unwrap();

    group.bench_function("cache_enabled_first_hit", |b| {
        b.iter(|| {
            engine_cached.clear_cache(); // Clear before each iteration
            black_box(
                engine_cached
                    .vector_similarity_search(&embeddings, &query, 10)
                    .unwrap(),
            );
        });
    });

    group.bench_function("cache_enabled_repeat", |b| {
        // Prime the cache
        let _ = engine_cached.vector_similarity_search(&embeddings, &query, 10);

        b.iter(|| {
            black_box(
                engine_cached
                    .vector_similarity_search(&embeddings, &query, 10)
                    .unwrap(),
            );
        });
    });

    // Test with cache disabled
    let mut config_no_cache = GpuConfig::auto_detect();
    config_no_cache.enable_caching = false;
    let engine_no_cache = GpuQueryEngine::new(config_no_cache).unwrap();

    group.bench_function("cache_disabled", |b| {
        b.iter(|| {
            black_box(
                engine_no_cache
                    .vector_similarity_search(&embeddings, &query, 10)
                    .unwrap(),
            );
        });
    });

    group.finish();
}

/// Benchmark batch processing throughput
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_batch_processing");
    group.measurement_time(Duration::from_secs(15));

    let embedding_dim = 64;
    let n_vectors = 1_000;
    let batch_sizes = vec![1, 10, 50, 100];

    let embeddings = generate_embeddings(n_vectors, embedding_dim);

    for batch_size in batch_sizes {
        let queries: Vec<Array1<f32>> = (0..batch_size)
            .map(|i| {
                let data: Vec<f32> = (0..embedding_dim)
                    .map(|j| ((i + j) % 10) as f32 / 10.0)
                    .collect();
                Array1::from_vec(data)
            })
            .collect();

        let config = GpuConfig::auto_detect();
        let engine = GpuQueryEngine::new(config).unwrap();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &queries,
            |b, queries| {
                b.iter(|| {
                    for query in queries {
                        black_box(
                            engine
                                .vector_similarity_search(&embeddings, query, 10)
                                .unwrap(),
                        );
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different top-k values
fn bench_topk_variation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_topk_variation");
    group.measurement_time(Duration::from_secs(10));

    let embedding_dim = 128;
    let n_vectors = 2_000;

    let embeddings = generate_embeddings(n_vectors, embedding_dim);
    let query = generate_query(embedding_dim);

    let config = GpuConfig::auto_detect();
    let engine = GpuQueryEngine::new(config).unwrap();

    let topk_values = vec![1, 10, 50, 100, 500];

    for k in topk_values {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                black_box(
                    engine
                        .vector_similarity_search(&embeddings, &query, k)
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark embedding dimension impact
fn bench_embedding_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_embedding_dimensions");
    group.measurement_time(Duration::from_secs(15));

    let n_vectors = 1_000;
    let dimensions = vec![32, 64, 128, 256, 512];

    for dim in dimensions {
        let embeddings = generate_embeddings(n_vectors, dim);
        let query = generate_query(dim);

        let config = GpuConfig::auto_detect();
        let engine = GpuQueryEngine::new(config).unwrap();

        group.throughput(Throughput::Elements((n_vectors * dim) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(embeddings, query),
            |b, (embeddings, query)| {
                b.iter(|| {
                    black_box(
                        engine
                            .vector_similarity_search(embeddings, query, 10)
                            .unwrap(),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark configuration impact
fn bench_configuration_profiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_config_profiles");
    group.measurement_time(Duration::from_secs(15));

    let embedding_dim = 128;
    let n_vectors = 2_000;

    let embeddings = generate_embeddings(n_vectors, embedding_dim);
    let query = generate_query(embedding_dim);

    let configs = vec![
        ("low_memory", GpuConfig::low_memory()),
        ("auto_detect", GpuConfig::auto_detect()),
        ("high_performance", GpuConfig::high_performance()),
    ];

    for (name, config) in configs {
        let engine = GpuQueryEngine::new(config).unwrap();

        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(
                    engine
                        .vector_similarity_search(&embeddings, &query, 10)
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark parallel query execution
fn bench_parallel_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_parallel_queries");
    group.measurement_time(Duration::from_secs(15));

    let embedding_dim = 64;
    let n_vectors = 1_000;
    let n_concurrent_queries = vec![1, 4, 8, 16];

    let embeddings = generate_embeddings(n_vectors, embedding_dim);

    for n_queries in n_concurrent_queries {
        let queries: Vec<Array1<f32>> = (0..n_queries)
            .map(|i| {
                let data: Vec<f32> = (0..embedding_dim)
                    .map(|j| ((i * 7 + j) % 10) as f32 / 10.0)
                    .collect();
                Array1::from_vec(data)
            })
            .collect();

        let config = GpuConfig::high_performance();
        let engine = GpuQueryEngine::new(config).unwrap();

        group.throughput(Throughput::Elements(n_queries as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_queries),
            &queries,
            |b, queries| {
                b.iter(|| {
                    for query in queries {
                        black_box(
                            engine
                                .vector_similarity_search(&embeddings, query, 10)
                                .unwrap(),
                        );
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_memory_efficiency");
    group.measurement_time(Duration::from_secs(10));

    let embedding_dim = 128;
    let sizes = vec![500, 1_000, 2_000, 5_000];

    for size in sizes {
        let embeddings = generate_embeddings(size, embedding_dim);
        let query = generate_query(embedding_dim);

        let config = GpuConfig::low_memory();
        let engine = GpuQueryEngine::new(config).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(embeddings, query),
            |b, (embeddings, query)| {
                b.iter(|| {
                    black_box(
                        engine
                            .vector_similarity_search(embeddings, query, 10)
                            .unwrap(),
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_similarity_scaling,
    bench_cache_effectiveness,
    bench_batch_processing,
    bench_topk_variation,
    bench_embedding_dimensions,
    bench_configuration_profiles,
    bench_parallel_queries,
    bench_memory_efficiency,
);

criterion_main!(benches);
