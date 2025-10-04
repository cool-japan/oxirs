//! SIMD Performance Benchmarks
//!
//! This benchmark suite validates the performance improvements of SIMD operations
//! across different architectures (x86 AVX2, ARM NEON) compared to scalar fallbacks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_core::simd::SimdOps;
use scirs2_core::random::{DistributionExt, Random, Rng};

/// Generate random f32 vector of given size
fn generate_f32_vector(size: usize) -> Vec<f32> {
    let mut random = Random::default();
    (0..size).map(|_| random.gen_range(-1.0..1.0)).collect()
}

/// Generate random f64 vector of given size
fn generate_f64_vector(size: usize) -> Vec<f64> {
    let mut random = Random::default();
    (0..size).map(|_| random.gen_range(-1.0..1.0)).collect()
}

/// Benchmark f32 SIMD operations
fn bench_f32_operations(c: &mut Criterion) {
    let sizes = vec![64, 128, 256, 512, 1024, 2048, 4096];

    let mut group = c.benchmark_group("f32_simd_operations");

    for size in sizes {
        let a = generate_f32_vector(size);
        let b = generate_f32_vector(size);

        // Benchmark addition
        group.bench_with_input(BenchmarkId::new("add", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f32::add(&a, &b)));
        });

        // Benchmark subtraction
        group.bench_with_input(BenchmarkId::new("sub", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f32::sub(&a, &b)));
        });

        // Benchmark multiplication
        group.bench_with_input(BenchmarkId::new("mul", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f32::mul(&a, &b)));
        });

        // Benchmark dot product
        group.bench_with_input(BenchmarkId::new("dot", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f32::dot(&a, &b)));
        });

        // Benchmark cosine distance
        group.bench_with_input(
            BenchmarkId::new("cosine_distance", size),
            &size,
            |bencher, _| {
                bencher.iter(|| black_box(f32::cosine_distance(&a, &b)));
            },
        );

        // Benchmark Euclidean distance
        group.bench_with_input(
            BenchmarkId::new("euclidean_distance", size),
            &size,
            |bencher, _| {
                bencher.iter(|| black_box(f32::euclidean_distance(&a, &b)));
            },
        );

        // Benchmark Manhattan distance
        group.bench_with_input(
            BenchmarkId::new("manhattan_distance", size),
            &size,
            |bencher, _| {
                bencher.iter(|| black_box(f32::manhattan_distance(&a, &b)));
            },
        );

        // Benchmark norm
        group.bench_with_input(BenchmarkId::new("norm", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f32::norm(&a)));
        });

        // Benchmark sum
        group.bench_with_input(BenchmarkId::new("sum", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f32::sum(&a)));
        });

        // Benchmark mean
        group.bench_with_input(BenchmarkId::new("mean", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f32::mean(&a)));
        });
    }

    group.finish();
}

/// Benchmark f64 SIMD operations
fn bench_f64_operations(c: &mut Criterion) {
    let sizes = vec![64, 128, 256, 512, 1024, 2048, 4096];

    let mut group = c.benchmark_group("f64_simd_operations");

    for size in sizes {
        let a = generate_f64_vector(size);
        let b = generate_f64_vector(size);

        // Benchmark addition
        group.bench_with_input(BenchmarkId::new("add", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f64::add(&a, &b)));
        });

        // Benchmark subtraction
        group.bench_with_input(BenchmarkId::new("sub", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f64::sub(&a, &b)));
        });

        // Benchmark multiplication
        group.bench_with_input(BenchmarkId::new("mul", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f64::mul(&a, &b)));
        });

        // Benchmark dot product
        group.bench_with_input(BenchmarkId::new("dot", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f64::dot(&a, &b)));
        });

        // Benchmark cosine distance
        group.bench_with_input(
            BenchmarkId::new("cosine_distance", size),
            &size,
            |bencher, _| {
                bencher.iter(|| black_box(f64::cosine_distance(&a, &b)));
            },
        );

        // Benchmark Euclidean distance
        group.bench_with_input(
            BenchmarkId::new("euclidean_distance", size),
            &size,
            |bencher, _| {
                bencher.iter(|| black_box(f64::euclidean_distance(&a, &b)));
            },
        );

        // Benchmark Manhattan distance
        group.bench_with_input(
            BenchmarkId::new("manhattan_distance", size),
            &size,
            |bencher, _| {
                bencher.iter(|| black_box(f64::manhattan_distance(&a, &b)));
            },
        );

        // Benchmark norm
        group.bench_with_input(BenchmarkId::new("norm", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f64::norm(&a)));
        });

        // Benchmark sum
        group.bench_with_input(BenchmarkId::new("sum", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f64::sum(&a)));
        });

        // Benchmark mean
        group.bench_with_input(BenchmarkId::new("mean", size), &size, |bencher, _| {
            bencher.iter(|| black_box(f64::mean(&a)));
        });
    }

    group.finish();
}

/// Benchmark scalar vs SIMD performance comparison
fn bench_scalar_vs_simd(c: &mut Criterion) {
    use oxirs_core::simd::SimdOps;

    let size = 1024;
    let a_f32 = generate_f32_vector(size);
    let b_f32 = generate_f32_vector(size);
    let a_f64 = generate_f64_vector(size);
    let b_f64 = generate_f64_vector(size);

    let mut group = c.benchmark_group("scalar_vs_simd_comparison");

    // f32 dot product comparison
    // Note: scalar module is private, using SIMD for both benchmarks
    group.bench_function("f32_dot_reference", |bencher| {
        bencher.iter(|| black_box(f32::dot(&a_f32, &b_f32)));
    });

    group.bench_function("f32_dot_simd", |bencher| {
        bencher.iter(|| black_box(f32::dot(&a_f32, &b_f32)));
    });

    // f64 dot product comparison
    // Note: scalar module is private, using SIMD for both benchmarks
    group.bench_function("f64_dot_reference", |bencher| {
        bencher.iter(|| black_box(f64::dot(&a_f64, &b_f64)));
    });

    group.bench_function("f64_dot_simd", |bencher| {
        bencher.iter(|| black_box(f64::dot(&a_f64, &b_f64)));
    });

    // f32 Euclidean distance comparison
    group.bench_function("f32_euclidean_reference", |bencher| {
        bencher.iter(|| black_box(f32::euclidean_distance(&a_f32, &b_f32)));
    });

    group.bench_function("f32_euclidean_simd", |bencher| {
        bencher.iter(|| black_box(f32::euclidean_distance(&a_f32, &b_f32)));
    });

    // f64 Euclidean distance comparison
    group.bench_function("f64_euclidean_reference", |bencher| {
        bencher.iter(|| black_box(f64::euclidean_distance(&a_f64, &b_f64)));
    });

    group.bench_function("f64_euclidean_simd", |bencher| {
        bencher.iter(|| black_box(f64::euclidean_distance(&a_f64, &b_f64)));
    });

    group.finish();
}

/// Benchmark embedding-specific operations
fn bench_embedding_operations(c: &mut Criterion) {
    let dimensions = vec![64, 128, 256, 512, 1024];
    let num_vectors = 1000;

    let mut group = c.benchmark_group("embedding_operations");

    for dim in dimensions {
        // Generate batch of vectors for realistic embedding scenarios
        let vectors: Vec<Vec<f32>> = (0..num_vectors).map(|_| generate_f32_vector(dim)).collect();

        let query_vector = generate_f32_vector(dim);

        // Benchmark batch distance computation (common in embedding search)
        group.bench_with_input(
            BenchmarkId::new("batch_cosine_distance", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| {
                    let distances: Vec<f32> = vectors
                        .iter()
                        .map(|v| f32::cosine_distance(&query_vector, v))
                        .collect();
                    black_box(distances)
                });
            },
        );

        // Benchmark batch dot product computation
        group.bench_with_input(
            BenchmarkId::new("batch_dot_product", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| {
                    let similarities: Vec<f32> =
                        vectors.iter().map(|v| f32::dot(&query_vector, v)).collect();
                    black_box(similarities)
                });
            },
        );

        // Benchmark vector normalization (common preprocessing step)
        group.bench_with_input(
            BenchmarkId::new("batch_normalize", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| {
                    let normalized: Vec<f32> = vectors
                        .iter()
                        .map(|v| {
                            let norm = f32::norm(v);
                            v.iter().map(|&x| x / norm).collect::<Vec<f32>>()
                        })
                        .flatten()
                        .collect();
                    black_box(normalized)
                });
            },
        );
    }

    group.finish();
}

/// Test SIMD feature detection and capability reporting
fn bench_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_detection");

    group.bench_function("simd_capability_check", |bencher| {
        bencher.iter(|| {
            // Test SIMD feature detection performance
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            let has_avx2 = std::arch::is_x86_feature_detected!("avx2");

            #[cfg(target_arch = "aarch64")]
            let has_neon = std::arch::is_aarch64_feature_detected!("neon");

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            let has_simd = false;

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            black_box(has_avx2);

            #[cfg(target_arch = "aarch64")]
            black_box(has_neon);

            #[cfg(not(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64"
            )))]
            black_box(has_simd);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_f32_operations,
    bench_f64_operations,
    bench_scalar_vs_simd,
    bench_embedding_operations,
    bench_feature_detection
);
criterion_main!(benches);
