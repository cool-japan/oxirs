//! Query performance benchmarks for oxirs-tsdb
//!
//! Benchmarks time-series query operations including range queries,
//! aggregations, window functions, and resampling.

use chrono::{Duration, TimeZone, Utc};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_tsdb::DataPoint;

/// Generate test data points with realistic patterns
fn generate_data_points(count: usize) -> Vec<DataPoint> {
    let base_time = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let mut points = Vec::with_capacity(count);
    let mut current_value = 22.5;

    for i in 0..count {
        let timestamp = base_time + Duration::seconds(i as i64);
        // Simulate temperature with drift and noise
        current_value += (i as f64 * 0.001).sin() * 0.1;
        current_value += (i as f64 * 0.1).sin() * 0.01;

        points.push(DataPoint {
            timestamp,
            value: current_value,
        });
    }

    points
}

/// Benchmark aggregation operations
fn bench_aggregations(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregations");

    for size in [1000, 10000, 100000] {
        let points = generate_data_points(size);
        let values: Vec<f64> = points.iter().map(|p| p.value).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mean", size), &values, |b, data| {
            b.iter(|| {
                let sum: f64 = data.iter().sum();
                let mean = sum / data.len() as f64;
                black_box(mean)
            });
        });

        group.bench_with_input(BenchmarkId::new("min_max", size), &values, |b, data| {
            b.iter(|| {
                let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                black_box((min, max))
            });
        });

        group.bench_with_input(BenchmarkId::new("stddev", size), &values, |b, data| {
            b.iter(|| {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let variance =
                    data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64;
                let stddev = variance.sqrt();
                black_box(stddev)
            });
        });

        group.bench_with_input(BenchmarkId::new("median", size), &values, |b, data| {
            b.iter(|| {
                let mut sorted = data.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = sorted.len() / 2;
                let median = if sorted.len() % 2 == 0 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                };
                black_box(median)
            });
        });
    }

    group.finish();
}

/// Benchmark window functions
fn bench_window_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_functions");

    for size in [1000, 10000] {
        let points = generate_data_points(size);
        let values: Vec<f64> = points.iter().map(|p| p.value).collect();

        group.throughput(Throughput::Elements(size as u64));

        // Moving average with various window sizes
        for window_size in [10, 50, 100] {
            group.bench_with_input(
                BenchmarkId::new(format!("moving_avg_w{}", window_size), size),
                &values,
                |b, data| {
                    b.iter(|| {
                        let mut result = Vec::with_capacity(data.len());
                        for i in 0..data.len() {
                            let start = if i >= window_size {
                                i - window_size + 1
                            } else {
                                0
                            };
                            let sum: f64 = data[start..=i].iter().sum();
                            let count = (i - start + 1) as f64;
                            result.push(sum / count);
                        }
                        black_box(result)
                    });
                },
            );
        }

        // Exponential moving average
        group.bench_with_input(
            BenchmarkId::new("ema_alpha0.1", size),
            &values,
            |b, data| {
                b.iter(|| {
                    let alpha = 0.1;
                    let mut result = Vec::with_capacity(data.len());
                    let mut ema = data[0];
                    result.push(ema);

                    for &value in &data[1..] {
                        ema = alpha * value + (1.0 - alpha) * ema;
                        result.push(ema);
                    }
                    black_box(result)
                });
            },
        );

        // Rate of change
        group.bench_with_input(
            BenchmarkId::new("rate_of_change", size),
            &values,
            |b, data| {
                b.iter(|| {
                    let mut result = Vec::with_capacity(data.len());
                    result.push(0.0); // First point has no rate

                    for i in 1..data.len() {
                        let rate = data[i] - data[i - 1];
                        result.push(rate);
                    }
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark resampling operations
fn bench_resampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampling");

    for size in [1000, 10000, 100000] {
        let points = generate_data_points(size);

        group.throughput(Throughput::Elements(size as u64));

        // Downsample: 1-second data to 1-minute buckets
        group.bench_with_input(
            BenchmarkId::new("downsample_1s_to_1m", size),
            &points,
            |b, data| {
                b.iter(|| {
                    let bucket_size = 60; // 60 seconds = 1 minute
                    let mut buckets: Vec<Vec<f64>> = Vec::new();
                    let mut current_bucket = Vec::new();
                    let mut bucket_start = data[0].timestamp.timestamp();

                    for point in data {
                        let ts = point.timestamp.timestamp();
                        if ts >= bucket_start + bucket_size {
                            if !current_bucket.is_empty() {
                                buckets.push(current_bucket);
                            }
                            current_bucket = Vec::new();
                            bucket_start = (ts / bucket_size) * bucket_size;
                        }
                        current_bucket.push(point.value);
                    }
                    if !current_bucket.is_empty() {
                        buckets.push(current_bucket);
                    }

                    // Calculate averages
                    let averages: Vec<f64> = buckets
                        .iter()
                        .map(|b| b.iter().sum::<f64>() / b.len() as f64)
                        .collect();

                    black_box(averages)
                });
            },
        );

        // Downsample: 1-second data to 1-hour buckets
        group.bench_with_input(
            BenchmarkId::new("downsample_1s_to_1h", size),
            &points,
            |b, data| {
                b.iter(|| {
                    let bucket_size = 3600; // 3600 seconds = 1 hour
                    let mut buckets: Vec<Vec<f64>> = Vec::new();
                    let mut current_bucket = Vec::new();
                    let mut bucket_start = data[0].timestamp.timestamp();

                    for point in data {
                        let ts = point.timestamp.timestamp();
                        if ts >= bucket_start + bucket_size {
                            if !current_bucket.is_empty() {
                                buckets.push(current_bucket);
                            }
                            current_bucket = Vec::new();
                            bucket_start = (ts / bucket_size) * bucket_size;
                        }
                        current_bucket.push(point.value);
                    }
                    if !current_bucket.is_empty() {
                        buckets.push(current_bucket);
                    }

                    let averages: Vec<f64> = buckets
                        .iter()
                        .map(|b| b.iter().sum::<f64>() / b.len() as f64)
                        .collect();

                    black_box(averages)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark interpolation operations
fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation");

    for size in [1000, 10000] {
        // Create sparse data (every 10th point)
        let full_points = generate_data_points(size);
        let sparse_points: Vec<DataPoint> = full_points
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 10 == 0)
            .map(|(_, p)| *p)
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // Linear interpolation to fill gaps
        group.bench_with_input(
            BenchmarkId::new("linear_fill", size),
            &sparse_points,
            |b, data| {
                b.iter(|| {
                    let mut result = Vec::with_capacity(size);

                    for i in 0..data.len() - 1 {
                        let start = &data[i];
                        let end = &data[i + 1];
                        let steps = 10; // Fill 10 points between each pair

                        for j in 0..steps {
                            let t = j as f64 / steps as f64;
                            let value = start.value + (end.value - start.value) * t;
                            result.push(value);
                        }
                    }
                    result.push(data.last().unwrap().value);

                    black_box(result)
                });
            },
        );

        // Forward fill (LOCF - Last Observation Carried Forward)
        group.bench_with_input(
            BenchmarkId::new("forward_fill", size),
            &sparse_points,
            |b, data| {
                b.iter(|| {
                    let mut result = Vec::with_capacity(size);

                    for current in data.iter().take(data.len() - 1) {
                        let steps = 10;

                        for _ in 0..steps {
                            result.push(current.value);
                        }
                    }
                    result.push(data.last().unwrap().value);

                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark range query operations
fn bench_range_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_queries");

    for size in [10000, 100000, 1000000] {
        let points = generate_data_points(size);

        group.throughput(Throughput::Elements(size as u64));

        // Query 10% of data
        let query_size = size / 10;
        group.bench_with_input(
            BenchmarkId::new("query_10_percent", size),
            &points,
            |b, data| {
                b.iter(|| {
                    let start_idx = size / 4;
                    let end_idx = start_idx + query_size;
                    let result: Vec<&DataPoint> = data[start_idx..end_idx].iter().collect();
                    black_box(result)
                });
            },
        );

        // Query 1% of data
        let query_size_small = size / 100;
        group.bench_with_input(
            BenchmarkId::new("query_1_percent", size),
            &points,
            |b, data| {
                b.iter(|| {
                    let start_idx = size / 2;
                    let end_idx = start_idx + query_size_small;
                    let result: Vec<&DataPoint> = data[start_idx..end_idx].iter().collect();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_aggregations,
    bench_window_functions,
    bench_resampling,
    bench_interpolation,
    bench_range_queries
);
criterion_main!(benches);
