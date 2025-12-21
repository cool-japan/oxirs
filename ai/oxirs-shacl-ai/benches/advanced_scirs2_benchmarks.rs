//! # Advanced SciRS2 Performance Benchmarks
//!
//! Comprehensive benchmark suite for evaluating GPU, SIMD, and parallel processing
//! performance gains in SHACL validation operations.
//!
//! ## Benchmark Categories
//! 1. GPU Acceleration (embeddings, matrix operations)
//! 2. SIMD Optimization (vector operations, triple processing)
//! 3. Parallel Processing (multi-threaded execution)
//! 4. Memory Efficiency (large dataset handling)
//! 5. End-to-End Workflows (realistic validation scenarios)
//!
//! ## Running Benchmarks
//! ```bash
//! cargo bench --bench advanced_scirs2_benchmarks
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_shacl_ai::advanced_scirs2_integration::{AdvancedSciRS2Config, AdvancedSciRS2Engine};
use scirs2_core::gpu::GpuBackend;
use scirs2_core::ndarray_ext::Array2;
use tokio::runtime::Runtime;

/// Benchmark GPU-accelerated embeddings computation
fn bench_gpu_embeddings(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_embeddings");

    let rt = Runtime::new().unwrap();

    // Test different matrix sizes
    let sizes = vec![
        (100, 64, 128),
        (500, 128, 256),
        (1000, 256, 512),
        (5000, 512, 1024),
    ];

    for (nodes, input_dim, output_dim) in sizes {
        // GPU configuration
        let gpu_config = AdvancedSciRS2Config {
            enable_gpu: true,
            gpu_backend: GpuBackend::Wgpu,
            ..Default::default()
        };

        if let Ok(gpu_engine) = AdvancedSciRS2Engine::with_config(gpu_config) {
            group.bench_with_input(
                BenchmarkId::new("gpu", format!("{}x{}→{}", nodes, input_dim, output_dim)),
                &(nodes, input_dim, output_dim),
                |b, &(n, i, o)| {
                    let node_data = Array2::from_elem((n, i), 1.0f32);
                    let edge_data = Array2::from_elem((i, o), 1.0f32);

                    b.to_async(&rt).iter(|| async {
                        let result = gpu_engine
                            .compute_embeddings_gpu(&node_data, &edge_data)
                            .await;
                        black_box(result)
                    });
                },
            );
        }

        // SIMD configuration for comparison
        let simd_config = AdvancedSciRS2Config {
            enable_gpu: false,
            enable_simd: true,
            ..Default::default()
        };

        let simd_engine = AdvancedSciRS2Engine::with_config(simd_config).unwrap();

        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}→{}", nodes, input_dim, output_dim)),
            &(nodes, input_dim, output_dim),
            |b, &(n, i, o)| {
                let node_data = Array2::from_elem((n, i), 1.0f32);
                let edge_data = Array2::from_elem((i, o), 1.0f32);

                b.to_async(&rt).iter(|| async {
                    let result = simd_engine
                        .compute_embeddings_simd(&node_data, &edge_data)
                        .await;
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SIMD-optimized triple processing
fn bench_simd_triple_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_triple_processing");

    let rt = Runtime::new().unwrap();

    let triple_counts = vec![1000, 5000, 10000, 50000];

    for count in triple_counts {
        // SIMD enabled
        let simd_config = AdvancedSciRS2Config {
            enable_simd: true,
            ..Default::default()
        };
        let simd_engine = AdvancedSciRS2Engine::with_config(simd_config).unwrap();

        group.bench_with_input(
            BenchmarkId::new("simd_enabled", count),
            &count,
            |b, &cnt| {
                let triples = Array2::from_elem((cnt, 64), 1.0f32);

                b.to_async(&rt).iter(|| async {
                    let result = simd_engine.process_triples_parallel(triples.view()).await;
                    black_box(result)
                });
            },
        );

        // SIMD disabled for comparison
        let no_simd_config = AdvancedSciRS2Config {
            enable_simd: false,
            parallel_workers: 1,
            ..Default::default()
        };
        let no_simd_engine = AdvancedSciRS2Engine::with_config(no_simd_config).unwrap();

        group.bench_with_input(
            BenchmarkId::new("simd_disabled", count),
            &count,
            |b, &cnt| {
                let triples = Array2::from_elem((cnt, 64), 1.0f32);

                b.to_async(&rt).iter(|| async {
                    let result = no_simd_engine
                        .process_triples_parallel(triples.view())
                        .await;
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel processing scalability
fn bench_parallel_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scalability");

    let rt = Runtime::new().unwrap();

    let worker_counts = vec![1, 2, 4, 8, num_cpus::get()];
    let triple_count = 50000;

    for workers in worker_counts {
        let config = AdvancedSciRS2Config {
            parallel_workers: workers,
            enable_simd: true,
            ..Default::default()
        };

        let engine = AdvancedSciRS2Engine::with_config(config).unwrap();

        group.bench_with_input(BenchmarkId::new("workers", workers), &workers, |b, _| {
            let triples = Array2::from_elem((triple_count, 64), 1.0f32);

            b.to_async(&rt).iter(|| async {
                let result = engine.process_triples_parallel(triples.view()).await;
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark memory-efficient operations
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    let rt = Runtime::new().unwrap();

    // Benchmark different memory limits
    let memory_limits = vec![512, 1024, 2048, 4096]; // MB

    for limit_mb in memory_limits {
        let config = AdvancedSciRS2Config {
            enable_mmap: true,
            memory_limit_mb: limit_mb,
            ..Default::default()
        };

        let engine = AdvancedSciRS2Engine::with_config(config).unwrap();

        group.bench_with_input(
            BenchmarkId::new("memory_limit_mb", limit_mb),
            &limit_mb,
            |b, _| {
                use std::fs::File;
                use std::io::Write;

                // Create test file
                let test_file = std::env::temp_dir().join(format!("bench_mmap_{}.bin", limit_mb));

                {
                    let mut file = File::create(&test_file).unwrap();
                    let data: Vec<f32> = (0..50000).map(|i| i as f32).collect();
                    let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
                    file.write_all(&bytes).unwrap();
                }

                b.to_async(&rt).iter(|| async {
                    let mmap = engine.load_large_dataset(test_file.to_str().unwrap()).await;
                    if let Ok(mmap) = mmap {
                        let result = engine.process_with_adaptive_chunking(&mmap).await;
                        black_box(result)
                    } else {
                        black_box(Ok(vec![]))
                    }
                });

                // Clean up
                std::fs::remove_file(&test_file).ok();
            },
        );
    }

    group.finish();
}

/// Benchmark end-to-end SHACL validation workflow
fn bench_end_to_end_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_workflow");

    let rt = Runtime::new().unwrap();

    // Realistic workflow sizes
    let scenarios = vec![
        ("small", 1000, 32),
        ("medium", 10000, 64),
        ("large", 50000, 128),
    ];

    for (name, node_count, feature_dim) in scenarios {
        // Full-featured configuration
        let full_config = AdvancedSciRS2Config {
            enable_gpu: true,
            enable_simd: true,
            parallel_workers: num_cpus::get(),
            enable_profiling: true,
            enable_metrics: true,
            ..Default::default()
        };

        if let Ok(full_engine) = AdvancedSciRS2Engine::with_config(full_config) {
            group.bench_with_input(
                BenchmarkId::new("full_featured", name),
                &(node_count, feature_dim),
                |b, &(nodes, dims)| {
                    let graph_data = Array2::from_elem((nodes, dims), 1.0f32);
                    let edge_data = Array2::from_elem((dims, dims * 2), 1.0f32);

                    b.to_async(&rt).iter(|| async {
                        // Simulate full validation workflow
                        let embeddings = full_engine
                            .compute_embeddings_gpu(&graph_data, &edge_data)
                            .await;
                        let processed = full_engine
                            .process_triples_parallel(graph_data.view())
                            .await;
                        black_box((embeddings, processed))
                    });
                },
            );
        }

        // Minimal configuration for comparison
        let minimal_config = AdvancedSciRS2Config {
            enable_gpu: false,
            enable_simd: false,
            parallel_workers: 1,
            enable_profiling: false,
            enable_metrics: false,
            ..Default::default()
        };

        let minimal_engine = AdvancedSciRS2Engine::with_config(minimal_config).unwrap();

        group.bench_with_input(
            BenchmarkId::new("minimal", name),
            &(node_count, feature_dim),
            |b, &(nodes, dims)| {
                let graph_data = Array2::from_elem((nodes, dims), 1.0f32);
                let edge_data = Array2::from_elem((dims, dims * 2), 1.0f32);

                b.to_async(&rt).iter(|| async {
                    let embeddings = minimal_engine
                        .compute_embeddings_simd(&graph_data, &edge_data)
                        .await;
                    let processed = minimal_engine
                        .process_triples_parallel(graph_data.view())
                        .await;
                    black_box((embeddings, processed))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark profiling overhead
fn bench_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_overhead");

    let rt = Runtime::new().unwrap();

    let test_data = Array2::from_elem((10000, 64), 1.0f32);

    // With profiling
    let with_profiling = AdvancedSciRS2Config {
        enable_profiling: true,
        ..Default::default()
    };
    let prof_engine = AdvancedSciRS2Engine::with_config(with_profiling).unwrap();

    group.bench_function("with_profiling", |b| {
        b.to_async(&rt).iter(|| async {
            let result = prof_engine.process_triples_parallel(test_data.view()).await;
            black_box(result)
        });
    });

    // Without profiling
    let without_profiling = AdvancedSciRS2Config {
        enable_profiling: false,
        ..Default::default()
    };
    let no_prof_engine = AdvancedSciRS2Engine::with_config(without_profiling).unwrap();

    group.bench_function("without_profiling", |b| {
        b.to_async(&rt).iter(|| async {
            let result = no_prof_engine
                .process_triples_parallel(test_data.view())
                .await;
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gpu_embeddings,
    bench_simd_triple_processing,
    bench_parallel_scalability,
    bench_memory_efficiency,
    bench_end_to_end_workflow,
    bench_profiling_overhead,
);

criterion_main!(benches);
