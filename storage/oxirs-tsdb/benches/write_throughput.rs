//! Write throughput benchmarks for oxirs-tsdb
//!
//! Benchmarks data ingestion performance including batch writes,
//! compression overhead, and buffer management.

use chrono::{Duration, TimeZone, Utc};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_tsdb::{DataPoint, GorillaCompressor};

/// Generate test data points
fn generate_data_points(count: usize) -> Vec<DataPoint> {
    let base_time = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let mut points = Vec::with_capacity(count);
    let mut current_value = 22.5;

    for i in 0..count {
        let timestamp = base_time + Duration::seconds(i as i64);
        current_value += (i as f64 * 0.001).sin() * 0.1;
        current_value += (i as f64 * 0.1).sin() * 0.01;

        points.push(DataPoint {
            timestamp,
            value: current_value,
        });
    }

    points
}

/// Benchmark single point writes
fn bench_single_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_writes");

    for size in [1000, 10000, 100000] {
        let points = generate_data_points(size);

        group.throughput(Throughput::Elements(size as u64));

        // Write to in-memory buffer
        group.bench_with_input(BenchmarkId::new("to_buffer", size), &points, |b, data| {
            b.iter(|| {
                let mut buffer: Vec<DataPoint> = Vec::with_capacity(size);
                for point in data {
                    buffer.push(*point);
                }
                black_box(buffer)
            });
        });

        // Write with immediate compression
        group.bench_with_input(
            BenchmarkId::new("to_compressed", size),
            &points,
            |b, data| {
                b.iter(|| {
                    let mut compressor = GorillaCompressor::new(data[0].value);
                    for point in &data[1..] {
                        compressor.compress(point.value);
                    }
                    black_box(compressor.finish())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch writes
fn bench_batch_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_writes");

    for total_size in [10000, 100000, 1000000] {
        let points = generate_data_points(total_size);

        group.throughput(Throughput::Elements(total_size as u64));

        // Small batch size (100 points)
        group.bench_with_input(
            BenchmarkId::new("batch_100", total_size),
            &points,
            |b, data| {
                b.iter(|| {
                    let batch_size = 100;
                    let mut buffers: Vec<Vec<DataPoint>> = Vec::new();

                    for chunk in data.chunks(batch_size) {
                        let batch: Vec<DataPoint> = chunk.to_vec();
                        buffers.push(batch);
                    }

                    black_box(buffers)
                });
            },
        );

        // Medium batch size (1000 points)
        group.bench_with_input(
            BenchmarkId::new("batch_1000", total_size),
            &points,
            |b, data| {
                b.iter(|| {
                    let batch_size = 1000;
                    let mut buffers: Vec<Vec<DataPoint>> = Vec::new();

                    for chunk in data.chunks(batch_size) {
                        let batch: Vec<DataPoint> = chunk.to_vec();
                        buffers.push(batch);
                    }

                    black_box(buffers)
                });
            },
        );

        // Large batch size (10000 points)
        group.bench_with_input(
            BenchmarkId::new("batch_10000", total_size),
            &points,
            |b, data| {
                b.iter(|| {
                    let batch_size = 10000;
                    let mut buffers: Vec<Vec<DataPoint>> = Vec::new();

                    for chunk in data.chunks(batch_size) {
                        let batch: Vec<DataPoint> = chunk.to_vec();
                        buffers.push(batch);
                    }

                    black_box(buffers)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multi-series writes
fn bench_multi_series_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_series_writes");

    // Simulate writing to multiple series
    for num_series in [10, 100, 1000] {
        let points_per_series = 1000;
        let total_points = num_series * points_per_series;

        // Pre-generate data for each series
        let series_data: Vec<Vec<DataPoint>> = (0..num_series)
            .map(|_| generate_data_points(points_per_series))
            .collect();

        group.throughput(Throughput::Elements(total_points as u64));

        // Round-robin writes (interleaved)
        group.bench_with_input(
            BenchmarkId::new("round_robin", num_series),
            &series_data,
            |b, data| {
                b.iter(|| {
                    let mut buffers: Vec<Vec<DataPoint>> = vec![Vec::new(); data.len()];

                    for point_idx in 0..points_per_series {
                        for (series_idx, series) in data.iter().enumerate() {
                            buffers[series_idx].push(series[point_idx]);
                        }
                    }

                    black_box(buffers)
                });
            },
        );

        // Sequential writes (one series at a time)
        group.bench_with_input(
            BenchmarkId::new("sequential", num_series),
            &series_data,
            |b, data| {
                b.iter(|| {
                    let mut buffers: Vec<Vec<DataPoint>> = Vec::with_capacity(data.len());

                    for series in data {
                        let buffer: Vec<DataPoint> = series.clone();
                        buffers.push(buffer);
                    }

                    black_box(buffers)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark buffer flush operations
fn bench_buffer_flush(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_flush");

    for buffer_size in [1000, 10000, 100000] {
        let points = generate_data_points(buffer_size);

        group.throughput(Throughput::Elements(buffer_size as u64));

        // Flush to compressed format
        group.bench_with_input(
            BenchmarkId::new("to_compressed", buffer_size),
            &points,
            |b, data| {
                b.iter(|| {
                    let mut compressor = GorillaCompressor::new(data[0].value);
                    for point in &data[1..] {
                        compressor.compress(point.value);
                    }
                    let compressed = compressor.finish();

                    // Simulate serialization
                    let serialized = compressed.clone();
                    black_box(serialized)
                });
            },
        );

        // Flush with sort (simulating out-of-order writes)
        let mut unsorted_points = points.clone();
        // Shuffle the points
        for i in (1..unsorted_points.len()).rev() {
            let j = i % (i + 1);
            unsorted_points.swap(i, j);
        }

        group.bench_with_input(
            BenchmarkId::new("sort_then_compress", buffer_size),
            &unsorted_points,
            |b, data| {
                b.iter(|| {
                    // Sort by timestamp
                    let mut sorted = data.clone();
                    sorted.sort_by_key(|p| p.timestamp);

                    // Then compress
                    let mut compressor = GorillaCompressor::new(sorted[0].value);
                    for point in &sorted[1..] {
                        compressor.compress(point.value);
                    }
                    black_box(compressor.finish())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark data point creation
fn bench_datapoint_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("datapoint_creation");

    for size in [1000, 10000, 100000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::new("create", size), |b| {
            let base_time = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();

            b.iter(|| {
                let mut points = Vec::with_capacity(size);
                for i in 0..size {
                    let timestamp = base_time + Duration::seconds(i as i64);
                    points.push(DataPoint {
                        timestamp,
                        value: i as f64 * 0.1,
                    });
                }
                black_box(points)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_writes,
    bench_batch_writes,
    bench_multi_series_writes,
    bench_buffer_flush,
    bench_datapoint_creation
);
criterion_main!(benches);
