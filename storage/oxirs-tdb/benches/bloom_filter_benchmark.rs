//! Bloom Filter Performance Benchmarks
//!
//! Comprehensive benchmarks for bloom filter implementation to validate:
//! - Insert performance
//! - Lookup performance
//! - False positive rate under load
//! - Memory efficiency
//! - Counting bloom filter operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_tdb::index::bloom_filter::{BloomFilter, BloomFilterConfig, CountingBloomFilter};
use std::collections::HashSet;

/// Benchmark basic bloom filter insertion
fn bench_bloom_filter_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter_insert");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let config = BloomFilterConfig {
                    expected_elements: size,
                    false_positive_rate: 0.01,
                    enable_metrics: false,
                    ..Default::default()
                };
                let mut filter = BloomFilter::new(config).unwrap();

                for i in 0..size {
                    filter.insert(&black_box(i));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark bloom filter lookups
fn bench_bloom_filter_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter_lookup");

    for size in [1_000, 10_000, 100_000].iter() {
        let config = BloomFilterConfig {
            expected_elements: *size,
            false_positive_rate: 0.01,
            enable_metrics: false,
            ..Default::default()
        };
        let mut filter = BloomFilter::new(config).unwrap();

        // Pre-populate filter
        for i in 0..*size {
            filter.insert(&i);
        }

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(filter.contains(&i));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark bloom filter vs HashSet for membership testing
fn bench_bloom_vs_hashset(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_vs_hashset");
    let size = 100_000;

    // Bloom filter benchmark
    group.bench_function("bloom_filter", |b| {
        let config = BloomFilterConfig {
            expected_elements: size,
            false_positive_rate: 0.01,
            enable_metrics: false,
            ..Default::default()
        };
        let mut filter = BloomFilter::new(config).unwrap();

        for i in 0..size {
            filter.insert(&i);
        }

        b.iter(|| {
            for i in 0..1000 {
                black_box(filter.contains(&i));
            }
        });
    });

    // HashSet benchmark
    group.bench_function("hashset", |b| {
        let mut set = HashSet::new();

        for i in 0..size {
            set.insert(i);
        }

        b.iter(|| {
            for i in 0..1000 {
                black_box(set.contains(&i));
            }
        });
    });

    group.finish();
}

/// Benchmark false positive rate measurement
fn bench_false_positive_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("false_positive_rate");

    for fpr in [0.001, 0.01, 0.05].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(fpr), fpr, |b, &fpr| {
            b.iter(|| {
                let config = BloomFilterConfig {
                    expected_elements: 10_000,
                    false_positive_rate: fpr,
                    enable_metrics: false,
                    ..Default::default()
                };
                let mut filter = BloomFilter::new(config).unwrap();

                // Insert 10K elements
                for i in 0..10_000 {
                    filter.insert(&i);
                }

                // Test 10K non-inserted elements
                let mut false_positives = 0;
                for i in 10_000..20_000 {
                    if filter.contains(&i) {
                        false_positives += 1;
                    }
                }

                black_box(false_positives)
            });
        });
    }

    group.finish();
}

/// Benchmark counting bloom filter operations
fn bench_counting_bloom_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("counting_bloom_filter");

    group.bench_function("insert_delete_cycle", |b| {
        b.iter(|| {
            let config = BloomFilterConfig {
                expected_elements: 10_000,
                false_positive_rate: 0.01,
                enable_counting: true,
                enable_metrics: false,
                ..Default::default()
            };
            let mut filter = CountingBloomFilter::new(config).unwrap();

            // Insert elements
            for i in 0..1_000 {
                filter.insert(&black_box(i));
            }

            // Delete elements
            for i in 0..1_000 {
                filter.delete(&black_box(i));
            }
        });
    });

    group.finish();
}

/// Benchmark bloom filter with different configurations
fn bench_bloom_filter_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter_configs");

    let configs = vec![("low_fpr", 0.001), ("medium_fpr", 0.01), ("high_fpr", 0.05)];

    for (name, fpr) in configs {
        group.bench_function(name, |b| {
            b.iter(|| {
                let config = BloomFilterConfig {
                    expected_elements: 10_000,
                    false_positive_rate: fpr,
                    enable_metrics: false,
                    ..Default::default()
                };
                let mut filter = BloomFilter::new(config).unwrap();

                for i in 0..10_000 {
                    filter.insert(&black_box(i));
                }

                for i in 0..10_000 {
                    black_box(filter.contains(&i));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark memory efficiency
fn bench_bloom_filter_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter_memory");

    group.bench_function("fill_rate_tracking", |b| {
        let config = BloomFilterConfig {
            expected_elements: 100_000,
            false_positive_rate: 0.01,
            enable_metrics: false,
            ..Default::default()
        };
        let mut filter = BloomFilter::new(config).unwrap();

        b.iter(|| {
            // Insert elements in batches and track fill rate
            for i in 0..1_000 {
                filter.insert(&i);
            }
            black_box(filter.fill_rate())
        });
    });

    group.finish();
}

/// Benchmark bloom filter statistics collection
fn bench_bloom_filter_stats(c: &mut Criterion) {
    let config = BloomFilterConfig {
        expected_elements: 100_000,
        false_positive_rate: 0.01,
        enable_metrics: false,
        ..Default::default()
    };
    let mut filter = BloomFilter::new(config).unwrap();

    // Pre-populate
    for i in 0..100_000 {
        filter.insert(&i);
    }

    c.bench_function("bloom_filter_stats", |b| {
        b.iter(|| black_box(filter.stats()));
    });
}

criterion_group!(
    benches,
    bench_bloom_filter_insert,
    bench_bloom_filter_lookup,
    bench_bloom_vs_hashset,
    bench_false_positive_rate,
    bench_counting_bloom_filter,
    bench_bloom_filter_configs,
    bench_bloom_filter_memory,
    bench_bloom_filter_stats,
);

criterion_main!(benches);
