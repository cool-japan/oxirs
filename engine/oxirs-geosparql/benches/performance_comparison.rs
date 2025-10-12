//! Performance comparison benchmarks
//!
//! Compares SIMD vs standard implementations, parallel vs sequential, etc.
//!
//! Run with: cargo bench --bench performance_comparison --features "parallel"

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use geo_types::{Geometry as GeoGeometry, Point};
use oxirs_geosparql::functions::geometric_operations::distance as standard_distance;
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::performance::{parallel, simd, BatchProcessor};

/// Generate test geometries
fn generate_points(n: usize) -> Vec<Geometry> {
    (0..n)
        .map(|i| {
            let x = (i as f64 * 0.1) % 100.0;
            let y = (i as f64 * 0.2) % 100.0;
            Geometry::new(GeoGeometry::Point(Point::new(x, y)))
        })
        .collect()
}

/// Benchmark: SIMD vs Standard Distance Calculation
fn bench_distance_simd_vs_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_calculation");

    let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
    let p2 = Geometry::new(GeoGeometry::Point(Point::new(100.0, 100.0)));

    group.bench_function("standard_distance", |b| {
        b.iter(|| black_box(standard_distance(&p1, &p2).unwrap()))
    });

    group.bench_function("simd_distance", |b| {
        b.iter(|| black_box(simd::euclidean_distance(&p1, &p2).unwrap()))
    });

    group.bench_function("simd_distance_squared", |b| {
        b.iter(|| black_box(simd::euclidean_distance_squared(&p1, &p2).unwrap()))
    });

    group.finish();
}

/// Benchmark: Batch Distance Calculations (various sizes)
fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");

    let query = Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0)));

    for size in [10, 100, 1000, 10000].iter() {
        let targets = generate_points(*size);

        group.throughput(Throughput::Elements(*size as u64));

        // Sequential standard distance
        group.bench_with_input(
            BenchmarkId::new("sequential_standard", size),
            size,
            |b, _| {
                b.iter(|| {
                    let distances: Vec<_> = targets
                        .iter()
                        .map(|t| standard_distance(&query, t).unwrap())
                        .collect();
                    black_box(distances)
                })
            },
        );

        // SIMD batch distance
        group.bench_with_input(BenchmarkId::new("simd_batch", size), size, |b, _| {
            b.iter(|| black_box(simd::batch_euclidean_distance(&query, &targets).unwrap()))
        });

        // Parallel distance (requires parallel feature)
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |b, _| {
            b.iter(|| black_box(parallel::parallel_distances(&query, &targets).unwrap()))
        });

        // BatchProcessor (auto-optimization)
        group.bench_with_input(BenchmarkId::new("batch_processor", size), size, |b, _| {
            let processor = BatchProcessor::new();
            b.iter(|| black_box(processor.distances(&query, &targets).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark: Distance Matrix (parallel vs sequential)
fn bench_distance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_matrix");
    group.sample_size(10); // Smaller sample for expensive operations

    for size in [10, 50, 100].iter() {
        let geometries = generate_points(*size);

        group.throughput(Throughput::Elements((size * size) as u64));

        // Sequential distance matrix
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |b, _| {
            b.iter(|| {
                let matrix: Vec<Vec<_>> = geometries
                    .iter()
                    .map(|g1| {
                        geometries
                            .iter()
                            .map(|g2| standard_distance(g1, g2).unwrap())
                            .collect()
                    })
                    .collect();
                black_box(matrix)
            })
        });

        // Parallel distance matrix
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |b, _| {
            b.iter(|| black_box(parallel::parallel_distance_matrix(&geometries).unwrap()))
        });

        // BatchProcessor distance matrix
        group.bench_with_input(BenchmarkId::new("batch_processor", size), size, |b, _| {
            let processor = BatchProcessor::new();
            b.iter(|| black_box(processor.distance_matrix(&geometries).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark: k-Nearest Neighbors
#[cfg(feature = "parallel")]
fn bench_nearest_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_neighbors");
    group.sample_size(10);

    for size in [100, 500, 1000].iter() {
        let geometries = generate_points(*size);
        let k = 5;

        group.throughput(Throughput::Elements((size * k) as u64));

        group.bench_with_input(BenchmarkId::new("parallel_knn", size), size, |b, _| {
            b.iter(|| black_box(parallel::parallel_nearest_neighbors(&geometries, k).unwrap()))
        });

        group.bench_with_input(
            BenchmarkId::new("batch_processor_knn", size),
            size,
            |b, _| {
                let processor = BatchProcessor::new();
                b.iter(|| black_box(processor.nearest_neighbors(&geometries, k).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark: Pairwise Distance
fn bench_pairwise_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_distance");

    for size in [100, 1000, 10000].iter() {
        let set1 = generate_points(*size);
        let set2 = generate_points(*size);

        group.throughput(Throughput::Elements(*size as u64));

        // Sequential pairwise
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |b, _| {
            b.iter(|| {
                let distances: Vec<_> = set1
                    .iter()
                    .zip(set2.iter())
                    .map(|(g1, g2)| standard_distance(g1, g2).unwrap())
                    .collect();
                black_box(distances)
            })
        });

        // SIMD pairwise
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| black_box(simd::pairwise_euclidean_distance(&set1, &set2).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark: Memory-Efficient Streaming
fn bench_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming");

    let query = Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0)));
    let targets = generate_points(10000);

    group.bench_function("batch_load_all", |b| {
        let processor = BatchProcessor::new();
        b.iter(|| black_box(processor.distances(&query, &targets).unwrap()))
    });

    group.bench_function("streaming_chunks", |b| {
        let processor = BatchProcessor::new();
        b.iter(|| {
            let mut count = 0;
            processor
                .stream_distances(&query, &targets, |chunk| {
                    count += chunk.len();
                    Ok(())
                })
                .unwrap();
            black_box(count)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_simd_vs_standard,
    bench_batch_distances,
    bench_distance_matrix,
    bench_pairwise_distance,
    bench_streaming,
);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, bench_nearest_neighbors,);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);
