//! Compression ratio benchmarks for oxirs-tsdb
//!
//! Benchmarks Gorilla and delta-of-delta compression performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_tsdb::{DeltaOfDeltaCompressor, GorillaCompressor, GorillaDecompressor};

/// Generate test data simulating temperature sensor readings
fn generate_temperature_data(count: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(count);
    let mut current = 22.5;
    for i in 0..count {
        // Simulate slow temperature drift with small variations
        current += (i as f64 * 0.001).sin() * 0.1;
        current += (i as f64 * 0.1).sin() * 0.01;
        values.push(current);
    }
    values
}

/// Generate test data simulating high-variance vibration readings
fn generate_vibration_data(count: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let base = (i as f64 * 0.5).sin() * 100.0;
        let noise = (i as f64 * 7.3).sin() * 50.0;
        values.push(base + noise);
    }
    values
}

/// Generate regular timestamps (1 second intervals)
fn generate_regular_timestamps(count: usize) -> Vec<i64> {
    let base = 1700000000i64; // Unix timestamp
    (0..count).map(|i| base + i as i64).collect()
}

/// Generate irregular timestamps (varying intervals)
fn generate_irregular_timestamps(count: usize) -> Vec<i64> {
    let base = 1700000000i64;
    let mut timestamps = Vec::with_capacity(count);
    let mut current = base;
    for i in 0..count {
        // Intervals vary between 800ms and 1200ms
        let interval = 1000 + ((i as f64 * 0.1).sin() * 200.0) as i64;
        current += interval;
        timestamps.push(current);
    }
    timestamps
}

fn bench_gorilla_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("gorilla_compression");

    for size in [1000, 10000, 100000] {
        let temp_data = generate_temperature_data(size);
        let vib_data = generate_vibration_data(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("temperature", size),
            &temp_data,
            |b, data| {
                b.iter(|| {
                    let mut compressor = GorillaCompressor::new(data[0]);
                    for &value in &data[1..] {
                        compressor.compress(black_box(value));
                    }
                    black_box(compressor.finish())
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("vibration", size), &vib_data, |b, data| {
            b.iter(|| {
                let mut compressor = GorillaCompressor::new(data[0]);
                for &value in &data[1..] {
                    compressor.compress(black_box(value));
                }
                black_box(compressor.finish())
            });
        });
    }

    group.finish();
}

fn bench_gorilla_decompression(c: &mut Criterion) {
    let mut group = c.benchmark_group("gorilla_decompression");

    for size in [1000, 10000, 100000] {
        let temp_data = generate_temperature_data(size);

        // Pre-compress data
        let mut compressor = GorillaCompressor::new(temp_data[0]);
        for &value in &temp_data[1..] {
            compressor.compress(value);
        }
        let compressed = compressor.finish();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("temperature", size),
            &compressed,
            |b, data| {
                b.iter(|| {
                    let mut decompressor = GorillaDecompressor::new(data).unwrap();
                    let mut values = Vec::with_capacity(size);
                    while let Some(value) = decompressor.next_value() {
                        values.push(value);
                    }
                    black_box(values)
                });
            },
        );
    }

    group.finish();
}

fn bench_delta_of_delta_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_of_delta_compression");

    for size in [1000, 10000, 100000] {
        let regular_ts = generate_regular_timestamps(size);
        let irregular_ts = generate_irregular_timestamps(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("regular_1sec", size),
            &regular_ts,
            |b, data| {
                b.iter(|| {
                    let mut compressor = DeltaOfDeltaCompressor::new(data[0]);
                    for &ts in &data[1..] {
                        compressor.compress(black_box(ts));
                    }
                    black_box(compressor.finish())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("irregular", size),
            &irregular_ts,
            |b, data| {
                b.iter(|| {
                    let mut compressor = DeltaOfDeltaCompressor::new(data[0]);
                    for &ts in &data[1..] {
                        compressor.compress(black_box(ts));
                    }
                    black_box(compressor.finish())
                });
            },
        );
    }

    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio_analysis");

    // Test with 10000 points
    let size = 10000;
    let temp_data = generate_temperature_data(size);
    let vib_data = generate_vibration_data(size);

    group.bench_function("temperature_ratio", |b| {
        b.iter(|| {
            let mut compressor = GorillaCompressor::new(temp_data[0]);
            for &value in &temp_data[1..] {
                compressor.compress(value);
            }
            let compressed = compressor.finish();
            let raw_size = size * 8; // 8 bytes per f64
            let compressed_size = compressed.len();
            let ratio = raw_size as f64 / compressed_size as f64;
            black_box(ratio)
        });
    });

    group.bench_function("vibration_ratio", |b| {
        b.iter(|| {
            let mut compressor = GorillaCompressor::new(vib_data[0]);
            for &value in &vib_data[1..] {
                compressor.compress(value);
            }
            let compressed = compressor.finish();
            let raw_size = size * 8;
            let compressed_size = compressed.len();
            let ratio = raw_size as f64 / compressed_size as f64;
            black_box(ratio)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gorilla_compression,
    bench_gorilla_decompression,
    bench_delta_of_delta_compression,
    bench_compression_ratio
);
criterion_main!(benches);
