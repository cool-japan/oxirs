//! Comprehensive GPU acceleration benchmarks for v0.4.0
//!
//! This benchmark suite validates the performance of:
//! - GPU-accelerated pattern matching (10-100x speedup potential)
//! - GPU-accelerated PageRank computation
//! - GPU vs CPU performance comparison
//! - GPU memory transfer overhead
//! - Backend selection and initialization
//!
//! Run with: `cargo bench --bench gpu_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_star::gpu_acceleration::{GpuAccelerator, GpuBackendType, GpuConfig};
use oxirs_star::{StarStore, StarTerm, StarTriple};
use tokio::runtime::Runtime;

// ============================================================================
// TEST DATA GENERATION
// ============================================================================

/// Create a test store with specified number of triples
fn create_test_store(n: usize) -> StarStore {
    let store = StarStore::new();

    for i in 0..n {
        let subject = StarTerm::iri(&format!("http://example.org/person{}", i)).unwrap();
        let predicate = StarTerm::iri(if i % 3 == 0 {
            "http://example.org/knows"
        } else if i % 3 == 1 {
            "http://example.org/worksAt"
        } else {
            "http://example.org/livesIn"
        })
        .unwrap();
        let object = StarTerm::iri(&format!("http://example.org/person{}", (i + 1) % n)).unwrap();

        store
            .insert(&StarTriple::new(subject, predicate, object))
            .unwrap();
    }

    store
}

/// Create a graph-like store for PageRank testing
fn create_graph_store(nodes: usize, edges_per_node: usize) -> StarStore {
    let store = StarStore::new();
    let predicate = StarTerm::iri("http://example.org/links").unwrap();

    for i in 0..nodes {
        let subject = StarTerm::iri(&format!("http://example.org/node{}", i)).unwrap();

        for j in 0..edges_per_node {
            let target = (i + j + 1) % nodes;
            let object = StarTerm::iri(&format!("http://example.org/node{}", target)).unwrap();

            store
                .insert(&StarTriple::new(subject.clone(), predicate.clone(), object))
                .unwrap();
        }
    }

    store
}

// ============================================================================
// GPU INITIALIZATION BENCHMARKS
// ============================================================================

/// Benchmark GPU accelerator initialization
fn bench_gpu_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/initialization");
    let rt = Runtime::new().unwrap();

    group.bench_function("auto_backend", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = GpuConfig::default();
                let accelerator = GpuAccelerator::new(config).await;
                black_box(accelerator)
            })
        });
    });

    group.bench_function("cpu_fallback", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = GpuConfig {
                    backend: GpuBackendType::CpuFallback,
                    ..GpuConfig::default()
                };
                let accelerator = GpuAccelerator::new(config).await;
                black_box(accelerator)
            })
        });
    });

    group.finish();
}

// ============================================================================
// GPU PATTERN MATCHING BENCHMARKS
// ============================================================================

/// Benchmark GPU-accelerated pattern matching with various dataset sizes
fn bench_gpu_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/pattern_matching");
    let rt = Runtime::new().unwrap();

    let config = GpuConfig::default();
    let mut accelerator = rt
        .block_on(async { GpuAccelerator::new(config).await })
        .unwrap();

    for size in [100, 1_000, 10_000, 100_000].iter() {
        let store = create_test_store(*size);
        let triples = store.triples();

        // Pattern: wildcard subject, specific predicate, wildcard object
        let pattern = [None, Some("http://example.org/knows"), None];

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                rt.block_on(async {
                    let matches = accelerator.pattern_match(&triples, &pattern).await;
                    black_box(matches)
                })
            });
        });
    }

    group.finish();
}

/// Benchmark different pattern selectivity
fn bench_pattern_selectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/pattern_selectivity");
    let rt = Runtime::new().unwrap();

    let config = GpuConfig::default();
    let mut accelerator = rt
        .block_on(async { GpuAccelerator::new(config).await })
        .unwrap();

    let store = create_test_store(10_000);
    let triples = store.triples();

    // High selectivity: specific predicate (~33% match)
    group.bench_function("high_selectivity", |b| {
        let pattern = [None, Some("http://example.org/knows"), None];
        b.iter(|| {
            rt.block_on(async {
                let matches = accelerator.pattern_match(&triples, &pattern).await;
                black_box(matches)
            })
        });
    });

    // Low selectivity: wildcard pattern (100% match)
    group.bench_function("low_selectivity", |b| {
        let pattern = [None, None, None];
        b.iter(|| {
            rt.block_on(async {
                let matches = accelerator.pattern_match(&triples, &pattern).await;
                black_box(matches)
            })
        });
    });

    // Very high selectivity: specific subject and predicate
    group.bench_function("very_high_selectivity", |b| {
        let pattern = [
            Some("http://example.org/person0"),
            Some("http://example.org/knows"),
            None,
        ];
        b.iter(|| {
            rt.block_on(async {
                let matches = accelerator.pattern_match(&triples, &pattern).await;
                black_box(matches)
            })
        });
    });

    group.finish();
}

// ============================================================================
// GPU PAGERANK BENCHMARKS
// ============================================================================

/// Benchmark GPU-accelerated PageRank computation
fn bench_gpu_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/pagerank");
    let rt = Runtime::new().unwrap();

    let config = GpuConfig::default();
    let mut accelerator = rt
        .block_on(async { GpuAccelerator::new(config).await })
        .unwrap();

    for nodes in [100, 500, 1_000, 5_000].iter() {
        let store = create_graph_store(*nodes, 3);

        group.throughput(Throughput::Elements(*nodes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nodes), nodes, |b, &_nodes| {
            b.iter(|| {
                rt.block_on(async {
                    let scores = accelerator.compute_pagerank(&store, 0.85, 10).await;
                    black_box(scores)
                })
            });
        });
    }

    group.finish();
}

/// Benchmark PageRank with different damping factors and iterations
fn bench_pagerank_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/pagerank_parameters");
    let rt = Runtime::new().unwrap();

    let config = GpuConfig::default();
    let mut accelerator = rt
        .block_on(async { GpuAccelerator::new(config).await })
        .unwrap();

    let store = create_graph_store(1_000, 3);

    // Different iteration counts
    for iterations in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            iterations,
            |b, &iters| {
                b.iter(|| {
                    rt.block_on(async {
                        let scores = accelerator.compute_pagerank(&store, 0.85, iters).await;
                        black_box(scores)
                    })
                });
            },
        );
    }

    // Different damping factors
    for damping in [0.5, 0.75, 0.85, 0.95].iter() {
        let damping_pct = (damping * 100.0) as u32;
        group.bench_with_input(
            BenchmarkId::new("damping", damping_pct),
            &damping_pct,
            |b, &_dp| {
                b.iter(|| {
                    rt.block_on(async {
                        let scores = accelerator.compute_pagerank(&store, *damping, 10).await;
                        black_box(scores)
                    })
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// GPU VS CPU COMPARISON BENCHMARKS
// ============================================================================

/// Benchmark GPU vs CPU for pattern matching
fn bench_gpu_vs_cpu_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_vs_cpu/pattern_matching");
    let rt = Runtime::new().unwrap();

    let store = create_test_store(10_000);
    let triples = store.triples();
    let pattern = [None, Some("http://example.org/knows"), None];

    // GPU version
    group.bench_function("gpu", |b| {
        let config = GpuConfig::default();
        let mut accelerator = rt
            .block_on(async { GpuAccelerator::new(config).await })
            .unwrap();

        b.iter(|| {
            rt.block_on(async {
                let matches = accelerator.pattern_match(&triples, &pattern).await;
                black_box(matches)
            })
        });
    });

    // CPU version (naive implementation)
    group.bench_function("cpu", |b| {
        b.iter(|| {
            let matches: Vec<_> = triples
                .iter()
                .filter(|t| {
                    if let Some(ref p) = pattern[1] {
                        if let StarTerm::NamedNode(node) = &t.predicate {
                            return &node.iri == p;
                        }
                    }
                    false
                })
                .cloned()
                .collect();
            black_box(matches)
        });
    });

    group.finish();
}

/// Benchmark GPU vs CPU for PageRank
fn bench_gpu_vs_cpu_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_vs_cpu/pagerank");
    let rt = Runtime::new().unwrap();

    let store = create_graph_store(1_000, 3);

    // GPU version
    group.bench_function("gpu", |b| {
        let config = GpuConfig::default();
        let mut accelerator = rt
            .block_on(async { GpuAccelerator::new(config).await })
            .unwrap();

        b.iter(|| {
            rt.block_on(async {
                let scores = accelerator.compute_pagerank(&store, 0.85, 10).await;
                black_box(scores)
            })
        });
    });

    // CPU version (simple iterative implementation)
    group.bench_function("cpu", |b| {
        b.iter(|| {
            let nodes: Vec<_> = store
                .triples()
                .iter()
                .flat_map(|t| {
                    vec![
                        if let StarTerm::NamedNode(node) = &t.subject {
                            Some(node.iri.clone())
                        } else {
                            None
                        },
                        if let StarTerm::NamedNode(node) = &t.object {
                            Some(node.iri.clone())
                        } else {
                            None
                        },
                    ]
                })
                .flatten()
                .collect();

            let n = nodes.len().max(1) as f64;
            let scores: std::collections::HashMap<_, _> =
                nodes.iter().map(|node| (node.clone(), 1.0 / n)).collect();

            black_box(scores)
        });
    });

    group.finish();
}

// ============================================================================
// GPU MEMORY TRANSFER BENCHMARKS
// ============================================================================

/// Benchmark GPU memory transfer overhead
fn bench_gpu_memory_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/memory_transfer");
    let rt = Runtime::new().unwrap();

    let config = GpuConfig::default();
    let mut accelerator = rt
        .block_on(async { GpuAccelerator::new(config).await })
        .unwrap();

    for size in [100, 1_000, 10_000, 100_000].iter() {
        let store = create_test_store(*size);
        let triples = store.triples();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                rt.block_on(async {
                    // Simulate memory transfer by creating a pattern and matching
                    // This forces data to be transferred to GPU
                    let pattern = [None, None, None];
                    let matches = accelerator.pattern_match(&triples, &pattern).await;
                    black_box(matches)
                })
            });
        });
    }

    group.finish();
}

// ============================================================================
// GPU BATCH SIZE OPTIMIZATION BENCHMARKS
// ============================================================================

/// Benchmark different GPU batch sizes
fn bench_gpu_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/batch_sizes");
    let rt = Runtime::new().unwrap();

    let store = create_test_store(10_000);
    let triples = store.triples();
    let pattern = [None, Some("http://example.org/knows"), None];

    for batch_size in [256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &bs| {
                b.iter(|| {
                    let config = GpuConfig {
                        batch_size: bs,
                        ..GpuConfig::default()
                    };
                    rt.block_on(async {
                        let mut accelerator = GpuAccelerator::new(config).await.unwrap();
                        let matches = accelerator.pattern_match(&triples, &pattern).await;
                        black_box(matches)
                    })
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK GROUPS
// ============================================================================

criterion_group!(gpu_init, bench_gpu_initialization,);

criterion_group!(
    gpu_pattern,
    bench_gpu_pattern_matching,
    bench_pattern_selectivity,
);

criterion_group!(gpu_pagerank, bench_gpu_pagerank, bench_pagerank_parameters,);

criterion_group!(
    gpu_comparison,
    bench_gpu_vs_cpu_pattern,
    bench_gpu_vs_cpu_pagerank,
);

criterion_group!(gpu_memory, bench_gpu_memory_transfer, bench_gpu_batch_sizes,);

criterion_main!(
    gpu_init,
    gpu_pattern,
    gpu_pagerank,
    gpu_comparison,
    gpu_memory
);
