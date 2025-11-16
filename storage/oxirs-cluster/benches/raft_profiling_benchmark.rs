//! Raft Profiling Performance Benchmarks
//!
//! Comprehensive benchmarks to measure profiling overhead and validate that
//! SciRS2-Core integration doesn't negatively impact Raft consensus performance.
//!
//! ## Benchmarks
//! - Profiling overhead for operation tracking
//! - Histogram recording performance
//! - Counter increment performance
//! - Bottleneck analysis performance
//! - Prometheus export performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_cluster::raft_profiling::{RaftOperation, RaftProfiler};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark profiling overhead - operation start/stop
fn bench_profiling_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("profiling_overhead");

    // Benchmark with profiling enabled
    group.bench_function("profiling_enabled", |b| {
        let profiler = RaftProfiler::new(1);

        b.to_async(&rt).iter(|| async {
            let op = profiler.start_operation(RaftOperation::AppendEntries).await;
            // Simulate some work
            tokio::time::sleep(Duration::from_micros(10)).await;
            op.complete().await;
        });
    });

    // Benchmark with profiling disabled
    group.bench_function("profiling_disabled", |b| {
        let profiler = RaftProfiler::new(1);
        rt.block_on(profiler.disable());

        b.to_async(&rt).iter(|| async {
            let op = profiler.start_operation(RaftOperation::AppendEntries).await;
            tokio::time::sleep(Duration::from_micros(10)).await;
            op.complete().await;
        });
    });

    // Benchmark without profiling at all
    group.bench_function("no_profiling", |b| {
        b.to_async(&rt).iter(|| async {
            tokio::time::sleep(Duration::from_micros(10)).await;
        });
    });

    group.finish();
}

/// Benchmark multiple concurrent operations
fn bench_concurrent_profiling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_profiling");

    for concurrent_ops in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*concurrent_ops as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(concurrent_ops),
            concurrent_ops,
            |b, &ops| {
                let profiler = RaftProfiler::new(1);

                b.to_async(&rt).iter(|| async {
                    let mut handles = Vec::new();

                    for i in 0..ops {
                        let profiler = profiler.clone();
                        let handle = tokio::spawn(async move {
                            let operation = match i % 3 {
                                0 => RaftOperation::AppendEntries,
                                1 => RaftOperation::RequestVote,
                                _ => RaftOperation::CreateSnapshot,
                            };

                            let op = profiler.start_operation(operation).await;
                            tokio::time::sleep(Duration::from_micros(10)).await;
                            op.complete().await;
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark network roundtrip recording
fn bench_network_roundtrip_recording(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    c.bench_function("network_roundtrip_recording", |b| {
        b.to_async(&rt).iter(|| async {
            profiler
                .record_network_roundtrip(2, black_box(Duration::from_millis(5)))
                .await;
        });
    });
}

/// Benchmark query execution recording
fn bench_query_execution_recording(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    c.bench_function("query_execution_recording", |b| {
        b.to_async(&rt).iter(|| async {
            profiler
                .record_query_execution("test_query", black_box(Duration::from_millis(20)))
                .await;
        });
    });
}

/// Benchmark memory tracking
fn bench_memory_tracking(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    c.bench_function("memory_tracking", |b| {
        b.to_async(&rt).iter(|| async {
            profiler
                .record_memory_usage("snapshot", black_box(1024 * 1024))
                .await;
        });
    });
}

/// Benchmark metrics retrieval
fn bench_metrics_retrieval(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    // Pre-populate with some data
    rt.block_on(async {
        for _ in 0..100 {
            let op = profiler.start_operation(RaftOperation::AppendEntries).await;
            tokio::time::sleep(Duration::from_micros(10)).await;
            op.complete().await;
        }
    });

    c.bench_function("get_metrics", |b| {
        b.to_async(&rt)
            .iter(|| async { black_box(profiler.get_metrics(RaftOperation::AppendEntries).await) });
    });
}

/// Benchmark getting all metrics
fn bench_get_all_metrics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    // Pre-populate with diverse operations
    rt.block_on(async {
        for operation in [
            RaftOperation::AppendEntries,
            RaftOperation::RequestVote,
            RaftOperation::CreateSnapshot,
        ] {
            for _ in 0..50 {
                let op = profiler.start_operation(operation).await;
                tokio::time::sleep(Duration::from_micros(10)).await;
                op.complete().await;
            }
        }
    });

    c.bench_function("get_all_metrics", |b| {
        b.to_async(&rt)
            .iter(|| async { black_box(profiler.get_all_metrics().await) });
    });
}

/// Benchmark bottleneck analysis
fn bench_bottleneck_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    // Pre-populate with operations of varying durations
    rt.block_on(async {
        for (operation, duration_ms) in [
            (RaftOperation::AppendEntries, 5),
            (RaftOperation::CreateSnapshot, 50),
            (RaftOperation::QueryExecution, 100),
        ] {
            for _ in 0..20 {
                let op = profiler.start_operation(operation).await;
                tokio::time::sleep(Duration::from_millis(duration_ms)).await;
                op.complete().await;
            }
        }
    });

    c.bench_function("analyze_bottlenecks", |b| {
        b.to_async(&rt)
            .iter(|| async { black_box(profiler.analyze_bottlenecks().await) });
    });
}

/// Benchmark profiler report generation
fn bench_report_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    // Pre-populate
    rt.block_on(async {
        for _ in 0..100 {
            let op = profiler.start_operation(RaftOperation::AppendEntries).await;
            tokio::time::sleep(Duration::from_micros(10)).await;
            op.complete().await;
        }
    });

    c.bench_function("generate_report", |b| {
        b.to_async(&rt)
            .iter(|| async { black_box(profiler.generate_report().await) });
    });
}

/// Benchmark Prometheus export
fn bench_prometheus_export(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    // Pre-populate
    rt.block_on(async {
        for operation in [
            RaftOperation::AppendEntries,
            RaftOperation::RequestVote,
            RaftOperation::CreateSnapshot,
        ] {
            for _ in 0..50 {
                let op = profiler.start_operation(operation).await;
                tokio::time::sleep(Duration::from_micros(10)).await;
                op.complete().await;
            }
        }
    });

    c.bench_function("export_prometheus", |b| {
        b.to_async(&rt)
            .iter(|| async { black_box(profiler.export_prometheus().await) });
    });
}

/// Benchmark histogram statistics retrieval
fn bench_histogram_stats(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    // Pre-populate
    rt.block_on(async {
        for _ in 0..100 {
            let op = profiler.start_operation(RaftOperation::AppendEntries).await;
            tokio::time::sleep(Duration::from_micros(10)).await;
            op.complete().await;
        }
    });

    c.bench_function("get_histogram_stats", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(
                profiler
                    .get_histogram_stats(RaftOperation::AppendEntries)
                    .await,
            )
        });
    });
}

/// Benchmark operation counter retrieval
fn bench_operation_count(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = RaftProfiler::new(1);

    // Pre-populate
    rt.block_on(async {
        for _ in 0..100 {
            let op = profiler.start_operation(RaftOperation::AppendEntries).await;
            tokio::time::sleep(Duration::from_micros(10)).await;
            op.complete().await;
        }
    });

    c.bench_function("get_operation_count", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(
                profiler
                    .get_operation_count(RaftOperation::AppendEntries)
                    .await,
            )
        });
    });
}

criterion_group!(
    benches,
    bench_profiling_overhead,
    bench_concurrent_profiling,
    bench_network_roundtrip_recording,
    bench_query_execution_recording,
    bench_memory_tracking,
    bench_metrics_retrieval,
    bench_get_all_metrics,
    bench_bottleneck_analysis,
    bench_report_generation,
    bench_prometheus_export,
    bench_histogram_stats,
    bench_operation_count,
);

criterion_main!(benches);
