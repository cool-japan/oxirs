//! # Advanced Sampling Benchmarks
//!
//! Performance benchmarks for probabilistic data structures
//! measuring throughput, memory usage, and accuracy.
//!
//! Run with: cargo bench --bench sampling_benchmarks --all-features

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_stream::{
    AdvancedSamplingManager, BloomFilter, CountMinSketch, EventMetadata, HyperLogLog,
    ReservoirSampler, SamplingConfig, StreamEvent, TDigest,
};
use std::collections::HashMap;

fn create_test_event(id: usize) -> StreamEvent {
    StreamEvent::TripleAdded {
        subject: format!("http://example.org/entity-{}", id),
        predicate: "http://example.org/prop".to_string(),
        object: format!("value-{}", id),
        graph: None,
        metadata: EventMetadata {
            event_id: format!("event-{}", id),
            timestamp: chrono::Utc::now(),
            source: "benchmark".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        },
    }
}

fn bench_reservoir_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("reservoir_sampling");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut sampler = ReservoirSampler::new(1000);
                for i in 0..size {
                    sampler.add(black_box(create_test_event(i)));
                }
                black_box(sampler.stats())
            });
        });
    }

    group.finish();
}

fn bench_hyperloglog_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperloglog_insertion");

    for precision in [10, 12, 14].iter() {
        group.throughput(Throughput::Elements(10000));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("p{}", precision)),
            precision,
            |b, &precision| {
                b.iter(|| {
                    let mut hll = HyperLogLog::new(precision);
                    for i in 0..10000 {
                        hll.add(&black_box(format!("element-{}", i)));
                    }
                    black_box(hll.cardinality())
                });
            },
        );
    }

    group.finish();
}

fn bench_hyperloglog_cardinality(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperloglog_cardinality");

    for precision in [10, 12, 14].iter() {
        let mut hll = HyperLogLog::new(*precision);
        for i in 0..10000 {
            hll.add(&format!("element-{}", i));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("p{}", precision)),
            precision,
            |b, _| {
                b.iter(|| black_box(hll.cardinality()));
            },
        );
    }

    group.finish();
}

fn bench_count_min_sketch(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_min_sketch");

    for size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut cms = CountMinSketch::new(4, 10000);
                for i in 0..size {
                    cms.add(&black_box(format!("element-{}", i % 1000)), 1);
                }
                black_box(cms.estimate(&"element-500"))
            });
        });
    }

    group.finish();
}

fn bench_tdigest_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdigest_insertion");

    for size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut digest = TDigest::new(0.01);
                for i in 0..size {
                    digest.add(black_box(i as f64), 1.0);
                }
                black_box(digest.stats())
            });
        });
    }

    group.finish();
}

fn bench_tdigest_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdigest_quantile");

    let mut digest = TDigest::new(0.01);
    for i in 0..100000 {
        digest.add(i as f64, 1.0);
    }

    group.bench_function("p50", |b| {
        b.iter(|| black_box(digest.clone().quantile(0.5)));
    });

    group.bench_function("p99", |b| {
        b.iter(|| black_box(digest.clone().quantile(0.99)));
    });

    group.finish();
}

fn bench_bloom_filter_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter_insertion");

    for size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut bloom = BloomFilter::optimal(size, 0.01);
                for i in 0..size {
                    bloom.add(&black_box(format!("element-{}", i)));
                }
                black_box(bloom.stats())
            });
        });
    }

    group.finish();
}

fn bench_bloom_filter_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter_lookup");

    let mut bloom = BloomFilter::optimal(100000, 0.01);
    for i in 0..50000 {
        bloom.add(&format!("element-{}", i));
    }

    group.bench_function("contains_present", |b| {
        b.iter(|| black_box(bloom.contains(&"element-25000")));
    });

    group.bench_function("contains_absent", |b| {
        b.iter(|| black_box(bloom.contains(&"element-99999")));
    });

    group.finish();
}

fn bench_sampling_manager(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_manager");

    for size in [1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let config = SamplingConfig {
                    reservoir_size: 500,
                    cms_hash_count: 4,
                    cms_width: 5000,
                    hll_precision: 12,
                    tdigest_delta: 0.01,
                    bloom_filter_bits: 50000,
                    bloom_filter_hashes: 7,
                    ..Default::default()
                };
                let mut manager = AdvancedSamplingManager::new(config);

                for i in 0..size {
                    let event = black_box(create_test_event(i));
                    manager.process_event(event).unwrap();
                }

                black_box(manager.stats())
            });
        });
    }

    group.finish();
}

fn bench_hyperloglog_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperloglog_merge");

    for precision in [10, 12, 14].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("p{}", precision)),
            precision,
            |b, &precision| {
                b.iter(|| {
                    let mut hll1 = HyperLogLog::new(precision);
                    let mut hll2 = HyperLogLog::new(precision);

                    for i in 0..5000 {
                        hll1.add(&format!("element-{}", i));
                    }
                    for i in 5000..10000 {
                        hll2.add(&format!("element-{}", i));
                    }

                    hll1.merge(&hll2);
                    black_box(hll1.cardinality())
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    group.bench_function("hyperloglog_p14", |b| {
        b.iter(|| {
            let hll = HyperLogLog::new(14);
            black_box(std::mem::size_of_val(&hll))
        });
    });

    group.bench_function("count_min_sketch_4x10k", |b| {
        b.iter(|| {
            let cms = CountMinSketch::new(4, 10000);
            black_box(std::mem::size_of_val(&cms))
        });
    });

    group.bench_function("bloom_filter_100k", |b| {
        b.iter(|| {
            let bloom = BloomFilter::optimal(100000, 0.01);
            black_box(std::mem::size_of_val(&bloom))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_reservoir_sampling,
    bench_hyperloglog_insertion,
    bench_hyperloglog_cardinality,
    bench_hyperloglog_merge,
    bench_count_min_sketch,
    bench_tdigest_insertion,
    bench_tdigest_quantile,
    bench_bloom_filter_insertion,
    bench_bloom_filter_lookup,
    bench_sampling_manager,
    bench_memory_efficiency,
);

criterion_main!(benches);
