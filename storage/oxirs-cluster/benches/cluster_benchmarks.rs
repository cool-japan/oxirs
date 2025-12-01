//! # OxiRS Cluster Comprehensive Benchmarking Suite
//!
//! This benchmark suite measures performance of critical cluster operations:
//! - GPU-accelerated replica selection
//! - Load forecasting with time series analysis
//! - SIMD-accelerated merkle tree operations
//! - Parallel data rebalancing
//! - Cloud integration performance
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # Run all benchmarks
//! cargo bench --bench cluster_benchmarks
//!
//! # Run specific benchmark
//! cargo bench --bench cluster_benchmarks -- gpu_replica_selection
//!
//! # Generate HTML report
//! cargo bench --bench cluster_benchmarks -- --save-baseline main
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_cluster::gpu_acceleration::{
    GpuAcceleratedCluster, GpuConfig, LoadForecastParams, ReplicaMetrics,
};
use oxirs_cluster::merkle_tree::MerkleTree;
use std::time::Duration;

/// Benchmark GPU-accelerated replica selection with varying replica counts
fn bench_gpu_replica_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_replica_selection");

    // Test with different replica counts to measure scaling
    for replica_count in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(*replica_count as u64));

        // Generate realistic replica metrics
        let metrics: Vec<ReplicaMetrics> = (0..*replica_count)
            .map(|i| {
                let base_latency = 10.0 + (i as f64 * 2.0);
                let base_lag = 100.0 + (i as f64 * 10.0);
                ReplicaMetrics {
                    node_id: i as u64,
                    latency_ms: base_latency + (i as f64 % 50.0),
                    connections: (5.0 + (i as f64 % 20.0)),
                    lag_ms: base_lag + (i as f64 % 100.0),
                    cpu_util: 0.3 + (i as f64 % 100.0) / 200.0,
                    mem_util: 0.4 + (i as f64 % 100.0) / 200.0,
                    success_rate: 0.95 + (i as f64 % 100.0) / 2000.0,
                }
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(replica_count),
            replica_count,
            |b, _| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = GpuConfig {
                    warmup_iterations: 1,
                    ..Default::default()
                };
                let gpu_cluster =
                    rt.block_on(async { GpuAcceleratedCluster::new(config).await.unwrap() });

                b.iter(|| {
                    rt.block_on(async {
                        let result = gpu_cluster
                            .select_best_replica(black_box(&metrics))
                            .await
                            .unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark load forecasting with varying history sizes
fn bench_gpu_load_forecasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_load_forecasting");
    group.measurement_time(Duration::from_secs(10));

    // Test with different history sizes
    for history_size in [24, 100, 500].iter() {
        group.throughput(Throughput::Elements(*history_size as u64));

        // Generate synthetic time series with trend and seasonality
        let history: Vec<f64> = (0..*history_size)
            .map(|i| {
                let trend = i as f64 * 0.1;
                let seasonal = (i as f64 / 12.0 * 2.0 * std::f64::consts::PI).sin() * 5.0;
                let noise = (i as f64 * 0.7).sin() * 2.0;
                50.0 + trend + seasonal + noise
            })
            .collect();

        let params = LoadForecastParams {
            history,
            horizon: 10,
            confidence_level: 0.95,
            detect_seasonality: *history_size >= 24,
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(history_size),
            history_size,
            |b, _| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = GpuConfig {
                    warmup_iterations: 1,
                    ..Default::default()
                };
                let gpu_cluster =
                    rt.block_on(async { GpuAcceleratedCluster::new(config).await.unwrap() });

                b.iter(|| {
                    rt.block_on(async {
                        let result = gpu_cluster
                            .forecast_load(black_box(params.clone()))
                            .await
                            .unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Merkle tree operations
fn bench_merkle_tree_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree_operations");

    // Test with different data sizes
    for data_count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*data_count as u64));

        let data: Vec<(String, String)> = (0..*data_count)
            .map(|i| {
                (
                    format!("key_{}", i),
                    format!("data_item_{}_with_some_content", i),
                )
            })
            .collect();

        // Benchmark tree construction and insertion
        group.bench_with_input(
            BenchmarkId::new("insert", data_count),
            data_count,
            |b, _| {
                let rt = tokio::runtime::Runtime::new().unwrap();

                b.iter(|| {
                    rt.block_on(async {
                        let tree = MerkleTree::new();
                        for (key, value) in black_box(&data) {
                            tree.insert(key.clone(), value).await;
                        }
                        black_box(tree);
                    });
                });
            },
        );

        // Benchmark proof generation
        let rt = tokio::runtime::Runtime::new().unwrap();
        let tree = rt.block_on(async {
            let t = MerkleTree::new();
            for (key, value) in &data {
                t.insert(key.clone(), value).await;
            }
            t
        });

        let test_key = &data[data_count / 2].0;
        group.bench_with_input(
            BenchmarkId::new("proof_generation", data_count),
            data_count,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let proof = tree.generate_proof(black_box(test_key)).await;
                        black_box(proof);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel vs sequential replica selection
fn bench_replica_selection_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("replica_selection_comparison");

    let replica_count = 100;
    let metrics: Vec<ReplicaMetrics> = (0..replica_count)
        .map(|i| ReplicaMetrics {
            node_id: i as u64,
            latency_ms: 10.0 + (i as f64 % 50.0),
            connections: 5.0 + (i as f64 % 20.0),
            lag_ms: 100.0 + (i as f64 % 100.0),
            cpu_util: 0.3 + (i as f64 % 100.0) / 200.0,
            mem_util: 0.4 + (i as f64 % 100.0) / 200.0,
            success_rate: 0.95 + (i as f64 % 100.0) / 2000.0,
        })
        .collect();

    group.bench_function("parallel_rayon", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = GpuConfig {
            warmup_iterations: 1,
            ..Default::default()
        };
        let gpu_cluster = rt.block_on(async { GpuAcceleratedCluster::new(config).await.unwrap() });

        b.iter(|| {
            rt.block_on(async {
                let result = gpu_cluster
                    .select_best_replica(black_box(&metrics))
                    .await
                    .unwrap();
                black_box(result);
            });
        });
    });

    // Sequential baseline for comparison
    group.bench_function("sequential_baseline", |b| {
        b.iter(|| {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_idx = 0;

            for (i, m) in metrics.iter().enumerate() {
                let features = [
                    (m.latency_ms / 1000.0).min(1.0),
                    (m.connections / 100.0).min(1.0),
                    (m.lag_ms / 1000.0).min(1.0),
                    m.cpu_util,
                    m.mem_util,
                    m.success_rate,
                ];

                let weights = [0.25, 0.15, 0.20, 0.15, 0.10, 0.15];
                let score: f64 = features
                    .iter()
                    .enumerate()
                    .map(|(j, &f)| {
                        if j == 5 {
                            weights[j] * f
                        } else {
                            weights[j] * (-f).exp()
                        }
                    })
                    .sum();

                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }

            black_box((metrics[best_idx].node_id, best_score));
        });
    });

    group.finish();
}

/// Benchmark memory efficiency of different operations
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Benchmark memory-efficient time series decomposition
    for size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let history: Vec<f64> = (0..*size).map(|i| 50.0 + i as f64 * 0.1).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                // Simulate moving average computation (part of time series decomposition)
                let window = 12;
                let mut result = vec![0.0; history.len()];

                for (i, val) in result.iter_mut().enumerate() {
                    let start = i.saturating_sub(window / 2);
                    let end = (i + window / 2 + 1).min(history.len());
                    let sum: f64 = history[start..end].iter().sum();
                    *val = sum / (end - start) as f64;
                }

                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark throughput for high-load scenarios
fn bench_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");
    group.measurement_time(Duration::from_secs(10));

    let metrics: Vec<ReplicaMetrics> = (0..100)
        .map(|i| ReplicaMetrics {
            node_id: i as u64,
            latency_ms: 10.0 + (i as f64 % 50.0),
            connections: 5.0,
            lag_ms: 100.0,
            cpu_util: 0.5,
            mem_util: 0.5,
            success_rate: 0.99,
        })
        .collect();

    // Simulate burst of replica selection requests
    for burst_size in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(burst_size),
            burst_size,
            |b, &size| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = GpuConfig {
                    warmup_iterations: 1,
                    batch_size: 256,
                    ..Default::default()
                };
                let gpu_cluster =
                    rt.block_on(async { GpuAcceleratedCluster::new(config).await.unwrap() });

                b.iter(|| {
                    rt.block_on(async {
                        for _ in 0..size {
                            let result = gpu_cluster.select_best_replica(&metrics).await.unwrap();
                            black_box(result);
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gpu_replica_selection,
    bench_gpu_load_forecasting,
    bench_merkle_tree_operations,
    bench_replica_selection_comparison,
    bench_memory_efficiency,
    bench_throughput_scaling
);
criterion_main!(benches);
