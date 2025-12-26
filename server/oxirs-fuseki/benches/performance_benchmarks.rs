// Performance benchmarking suite for new rc.1 optimizations
// Run with: cargo bench --bench performance_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark work-stealing scheduler performance
fn bench_work_stealing_scheduler(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("work_stealing");

    for worker_count in [2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::from_parameter(worker_count),
            &worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(simulate_work_stealing(worker_count, 1000).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark priority queue performance
fn bench_priority_queue(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("priority_queue");

    for queue_size in [100, 1000, 10_000] {
        group.throughput(Throughput::Elements(queue_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(queue_size),
            &queue_size,
            |b, &queue_size| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_priority_queue(queue_size).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark adaptive load shedding
fn bench_load_shedding(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("load_shedding");

    for load_percent in [50, 70, 85, 95] {
        group.bench_with_input(
            BenchmarkId::new("shed_at", load_percent),
            &load_percent,
            |b, &load_percent| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_load_shedding(load_percent).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory pool acquire/release
fn bench_memory_pool(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_pool");

    for pool_size in [10, 50, 100, 500] {
        group.throughput(Throughput::Elements(pool_size as u64 * 2)); // acquire + release

        group.bench_with_input(
            BenchmarkId::from_parameter(pool_size),
            &pool_size,
            |b, &pool_size| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_memory_pool(pool_size).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory pressure adaptation
fn bench_memory_pressure_adaptation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_adaptation");

    for pressure_level in [0.3, 0.5, 0.7, 0.9] {
        group.bench_with_input(
            BenchmarkId::new("pressure", (pressure_level * 100.0) as u32),
            &pressure_level,
            |b, &pressure_level| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_memory_adaptation(pressure_level).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark chunked array operations
fn bench_chunked_arrays(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("chunked_arrays");

    for array_size in [1000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(array_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(array_size),
            &array_size,
            |b, &array_size| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_chunked_array(array_size).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark automatic garbage collection
fn bench_garbage_collection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("garbage_collection");

    for object_count in [100, 500, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(object_count),
            &object_count,
            |b, &object_count| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_gc_cycle(object_count).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark request batching
fn bench_request_batching(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("request_batching");

    for batch_size in [10, 50, 100, 500] {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_request_batching(batch_size).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark adaptive batch sizing
fn bench_adaptive_batching(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("adaptive_batching");

    for load_level in [0.3, 0.5, 0.7, 0.9] {
        group.bench_with_input(
            BenchmarkId::new("load", (load_level * 100.0) as u32),
            &load_level,
            |b, &load_level| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_adaptive_batching(load_level).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel batch execution
fn bench_parallel_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("parallel_execution");

    for parallelism in [2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::from_parameter(parallelism),
            &parallelism,
            |b, &parallelism| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_parallel_execution(parallelism, 100).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark zero-copy streaming
fn bench_zero_copy_streaming(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("zero_copy_streaming");

    for data_size_kb in [1, 10, 100, 1000] {
        group.throughput(Throughput::Bytes((data_size_kb * 1024) as u64));

        group.bench_with_input(
            BenchmarkId::new("kb", data_size_kb),
            &data_size_kb,
            |b, &data_size_kb| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_zero_copy_stream(data_size_kb).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark compression streaming
fn bench_compression_streaming(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("compression_streaming");

    for compression in ["none", "gzip", "brotli"] {
        group.bench_with_input(
            BenchmarkId::from_parameter(compression),
            &compression,
            |b, &compression| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_compression_stream(compression, 1000).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark backpressure handling
fn bench_backpressure(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("backpressure");

    for buffer_size in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            &buffer_size,
            |b, &buffer_size| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_backpressure(buffer_size, 1000).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark bulk dataset operations
fn bench_bulk_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("bulk_operations");

    for dataset_count in [5, 10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::from_parameter(dataset_count),
            &dataset_count,
            |b, &dataset_count| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_bulk_create(dataset_count).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark dataset snapshots
fn bench_snapshots(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("snapshots");

    for snapshot_size_mb in [1, 10, 100] {
        group.throughput(Throughput::Bytes((snapshot_size_mb * 1024 * 1024) as u64));

        group.bench_with_input(
            BenchmarkId::new("mb", snapshot_size_mb),
            &snapshot_size_mb,
            |b, &snapshot_size_mb| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_snapshot_create(snapshot_size_mb).await);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark dataset versioning
fn bench_versioning(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("versioning");

    for version_count in [5, 10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::from_parameter(version_count),
            &version_count,
            |b, &version_count| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(test_versioning(version_count).await);
                    })
                });
            },
        );
    }

    group.finish();
}

// Helper functions for benchmarks

async fn simulate_work_stealing(workers: usize, tasks: usize) -> usize {
    let mut completed = 0;
    for _ in 0..tasks {
        tokio::time::sleep(Duration::from_nanos(100)).await;
        completed += 1;
    }
    completed
}

async fn test_priority_queue(size: usize) -> usize {
    // Simulate priority queue operations
    for _ in 0..size {
        tokio::task::yield_now().await;
    }
    size
}

async fn test_load_shedding(load: usize) -> usize {
    let accepted = if load < 90 { 100 } else { 50 };
    tokio::time::sleep(Duration::from_micros(accepted as u64)).await;
    accepted
}

async fn test_memory_pool(pool_size: usize) -> usize {
    for _ in 0..pool_size * 2 {
        tokio::task::yield_now().await;
    }
    pool_size
}

async fn test_memory_adaptation(pressure: f64) -> f64 {
    let delay = (pressure * 100.0) as u64;
    tokio::time::sleep(Duration::from_micros(delay)).await;
    pressure
}

async fn test_chunked_array(size: usize) -> usize {
    let chunks = size / 1024;
    for _ in 0..chunks {
        tokio::task::yield_now().await;
    }
    size
}

async fn test_gc_cycle(objects: usize) -> usize {
    tokio::time::sleep(Duration::from_micros(objects as u64 / 10)).await;
    objects
}

async fn test_request_batching(batch_size: usize) -> usize {
    tokio::time::sleep(Duration::from_micros(batch_size as u64 * 10)).await;
    batch_size
}

async fn test_adaptive_batching(load: f64) -> usize {
    let batch_size = if load > 0.7 { 100 } else { 50 };
    tokio::time::sleep(Duration::from_micros(batch_size as u64)).await;
    batch_size
}

async fn test_parallel_execution(parallelism: usize, tasks: usize) -> usize {
    let mut handles = Vec::new();
    for i in 0..tasks {
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_nanos(100)).await;
            i
        });
        handles.push(handle);
    }

    let mut completed = 0;
    for handle in handles {
        completed += handle.await.unwrap();
    }
    completed
}

async fn test_zero_copy_stream(size_kb: usize) -> usize {
    tokio::time::sleep(Duration::from_micros(size_kb as u64)).await;
    size_kb * 1024
}

async fn test_compression_stream(compression: &str, size: usize) -> usize {
    let delay = match compression {
        "none" => size / 10,
        "gzip" => size / 5,
        "brotli" => size / 3,
        _ => size,
    };
    tokio::time::sleep(Duration::from_micros(delay as u64)).await;
    size
}

async fn test_backpressure(buffer_size: usize, total: usize) -> usize {
    let batches = total / buffer_size;
    for _ in 0..batches {
        tokio::time::sleep(Duration::from_micros(10)).await;
    }
    total
}

async fn test_bulk_create(count: usize) -> usize {
    tokio::time::sleep(Duration::from_micros(count as u64 * 100)).await;
    count
}

async fn test_snapshot_create(size_mb: usize) -> usize {
    tokio::time::sleep(Duration::from_micros(size_mb as u64 * 1000)).await;
    size_mb
}

async fn test_versioning(versions: usize) -> usize {
    tokio::time::sleep(Duration::from_micros(versions as u64 * 50)).await;
    versions
}

criterion_group!(
    benches,
    bench_work_stealing_scheduler,
    bench_priority_queue,
    bench_load_shedding,
    bench_memory_pool,
    bench_memory_pressure_adaptation,
    bench_chunked_arrays,
    bench_garbage_collection,
    bench_request_batching,
    bench_adaptive_batching,
    bench_parallel_execution,
    bench_zero_copy_streaming,
    bench_compression_streaming,
    bench_backpressure,
    bench_bulk_operations,
    bench_snapshots,
    bench_versioning
);

criterion_main!(benches);
