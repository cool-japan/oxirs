//! Adaptive Query Optimizer and ChunkedIterator Benchmarks
//!
//! Comprehensive performance benchmarks for v0.1.3 features including:
//! - ChunkedIterator performance across different data sizes and chunk sizes
//! - Adaptive Query Optimizer strategy selection and execution
//! - Regression detection overhead
//! - Workload profiling performance
//! - Multi-objective optimization

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use oxirs_star::{
    adaptive_query_optimizer::{AdaptiveQueryOptimizer, OptimizationObjective},
    model::{StarTerm, StarTriple},
    serializer::star_serializer::ChunkedIterator,
};
use std::time::Duration;

/// Benchmark ChunkedIterator with various data sizes and chunk sizes
fn benchmark_chunked_iterator(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_iterator");
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    // Test different data sizes
    let data_sizes = [100, 1_000, 10_000, 100_000];
    let chunk_sizes = [10, 100, 1_000];

    for data_size in data_sizes.iter() {
        for chunk_size in chunk_sizes.iter() {
            if chunk_size > data_size {
                continue;
            }

            group.throughput(Throughput::Elements(*data_size as u64));

            group.bench_with_input(
                BenchmarkId::new(
                    format!("data_{}/chunk_{}", data_size, chunk_size),
                    data_size,
                ),
                &(*data_size, *chunk_size),
                |b, &(data_size, chunk_size)| {
                    let data: Vec<i32> = (0..data_size as i32).collect();
                    b.iter(|| {
                        let chunks: Vec<Vec<i32>> =
                            ChunkedIterator::new(data.clone().into_iter(), chunk_size).collect();
                        black_box(chunks)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark ChunkedIterator vs standard chunking methods
fn benchmark_chunking_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking_comparison");

    let data_size = 10_000;
    let chunk_size = 100;
    let data: Vec<i32> = (0..data_size).collect();

    group.throughput(Throughput::Elements(data_size as u64));

    // Benchmark ChunkedIterator
    group.bench_function("ChunkedIterator", |b| {
        b.iter(|| {
            let chunks: Vec<Vec<i32>> =
                ChunkedIterator::new(data.clone().into_iter(), chunk_size).collect();
            black_box(chunks)
        })
    });

    // Benchmark manual chunking
    group.bench_function("manual_chunking", |b| {
        b.iter(|| {
            let chunks: Vec<Vec<i32>> = data
                .chunks(chunk_size)
                .map(|chunk| chunk.to_vec())
                .collect();
            black_box(chunks)
        })
    });

    // Benchmark collect + chunks (for comparison)
    group.bench_function("collect_chunks", |b| {
        b.iter(|| {
            let collected: Vec<_> = data.clone();
            let chunks: Vec<Vec<i32>> = collected
                .chunks(chunk_size)
                .map(|chunk| chunk.to_vec())
                .collect();
            black_box(chunks)
        })
    });

    group.finish();
}

/// Benchmark RDF triple batch processing with ChunkedIterator
fn benchmark_triple_batching(c: &mut Criterion) {
    let mut group = c.benchmark_group("triple_batching");

    let triple_count = 10_000;
    let batch_sizes = [10, 50, 100, 500, 1000];

    // Generate test triples
    let triples: Vec<StarTriple> = (0..triple_count)
        .map(|i| {
            StarTriple::new(
                StarTerm::iri(&format!("http://example.org/subject{}", i)).unwrap(),
                StarTerm::iri("http://example.org/predicate").unwrap(),
                StarTerm::literal(&format!("value{}", i)).unwrap(),
            )
        })
        .collect();

    for batch_size in batch_sizes.iter() {
        group.throughput(Throughput::Elements(triple_count as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let batches: Vec<Vec<StarTriple>> =
                        ChunkedIterator::new(triples.clone().into_iter(), batch_size).collect();
                    black_box(batches)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark AdaptiveQueryOptimizer with different query types
/// (replacing direct complexity estimation which is private)
fn benchmark_strategy_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_strategy_selection");

    // Different query complexity levels
    let queries = [
        ("simple", "SELECT * WHERE { ?s ?p ?o }"),
        (
            "medium",
            "SELECT * WHERE { ?s ?p ?o . OPTIONAL { ?s ?p2 ?o2 } FILTER(?x > 10) }",
        ),
        (
            "complex",
            "SELECT * WHERE { << ?s ?p ?o >> ?meta ?value . OPTIONAL { ?s ?p2 ?o2 } FILTER(?x > 10) } GROUP BY ?s",
        ),
        (
            "very_complex",
            "SELECT * WHERE { << << ?s ?p ?o >> ?m1 ?v1 >> ?m2 ?v2 . OPTIONAL { ?s ?p2 ?o2 } UNION { ?s ?p3 ?o3 } FILTER(?x > 10) } GROUP BY ?s ORDER BY ?o LIMIT 100",
        ),
    ];

    for (name, query) in queries.iter() {
        group.bench_with_input(BenchmarkId::new("query_type", name), query, |b, &query| {
            let mut optimizer = AdaptiveQueryOptimizer::new();
            b.iter(|| {
                // Benchmark the full optimization which includes strategy selection
                let result = optimizer.optimize_query(query).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark AdaptiveQueryOptimizer complete optimization workflow
fn benchmark_adaptive_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_optimization");
    group.measurement_time(Duration::from_secs(10));

    let queries = [
        ("simple", "SELECT * WHERE { ?s ?p ?o }"),
        (
            "complex",
            "SELECT * WHERE { << ?s ?p ?o >> ?meta ?value . FILTER(?x > 10) }",
        ),
    ];

    for (name, query) in queries.iter() {
        group.bench_with_input(BenchmarkId::new("optimize", name), query, |b, &query| {
            let mut optimizer = AdaptiveQueryOptimizer::new();
            b.iter(|| black_box(optimizer.optimize_query(query).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark regression detection overhead
fn benchmark_regression_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_detection");

    use oxirs_star::adaptive_query_optimizer::RegressionDetector;

    // Create detector and establish baseline
    let mut detector = RegressionDetector::new(1.3, 100);
    for _ in 0..40 {
        detector.update(100.0);
    }

    group.bench_function("update_only", |b| {
        let mut detector_clone = detector.clone();
        b.iter(|| {
            detector_clone.update(black_box(105.0));
        })
    });

    group.bench_function("update_and_detect", |b| {
        let mut detector_clone = detector.clone();
        b.iter(|| {
            detector_clone.update(black_box(105.0));
            let result = detector_clone.detect_regression();
            black_box(result)
        })
    });

    group.bench_function("detect_only", |b| {
        b.iter(|| {
            let result = detector.detect_regression();
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark multi-objective optimization configuration
fn benchmark_multi_objective_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_objective");

    let objectives = vec![
        OptimizationObjective::MinimizeLatency { weight: 0.5 },
        OptimizationObjective::MinimizeMemory { weight: 0.3 },
        OptimizationObjective::MaximizeAccuracy { weight: 0.2 },
    ];

    group.bench_function("set_objectives", |b| {
        b.iter(|| {
            let mut optimizer = AdaptiveQueryOptimizer::new();
            optimizer.set_objectives(black_box(objectives.clone()));
            black_box(optimizer)
        })
    });

    group.bench_function("optimize_with_objectives", |b| {
        let mut optimizer = AdaptiveQueryOptimizer::new();
        optimizer.set_objectives(objectives.clone());
        let query = "SELECT * WHERE { << ?s ?p ?o >> ?meta ?value }";

        b.iter(|| black_box(optimizer.optimize_query(query).unwrap()))
    });

    group.finish();
}

/// Benchmark workload profiling overhead
fn benchmark_workload_profiling(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload_profiling");

    let query_sets = [
        ("homogeneous", vec!["SELECT * WHERE { ?s ?p ?o }"; 100]),
        (
            "heterogeneous",
            (0..100)
                .map(|i| {
                    if i % 3 == 0 {
                        "SELECT * WHERE { ?s ?p ?o }"
                    } else if i % 3 == 1 {
                        "SELECT * WHERE { << ?s ?p ?o >> ?m ?v }"
                    } else {
                        "SELECT * WHERE { ?s ?p ?o . OPTIONAL { ?s ?p2 ?o2 } }"
                    }
                })
                .collect::<Vec<_>>(),
        ),
    ];

    for (name, queries) in query_sets.iter() {
        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("workload_type", name),
            queries,
            |b, queries| {
                b.iter(|| {
                    let mut optimizer = AdaptiveQueryOptimizer::new();
                    for query in queries {
                        let _ = optimizer.optimize_query(query);
                    }
                    let stats = optimizer.statistics();
                    black_box(stats)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark auto-tuning warmup phase
fn benchmark_auto_tuning_warmup(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_tuning_warmup");
    group.measurement_time(Duration::from_secs(15));

    let warmup_sizes = [10, 25, 50, 100];

    for warmup_size in warmup_sizes.iter() {
        group.throughput(Throughput::Elements(*warmup_size as u64));

        group.bench_with_input(
            BenchmarkId::new("warmup_queries", warmup_size),
            warmup_size,
            |b, &warmup_size| {
                b.iter(|| {
                    let mut optimizer = AdaptiveQueryOptimizer::new();
                    optimizer.set_auto_tuning(true);

                    for i in 0..warmup_size {
                        let query = if i % 2 == 0 {
                            "SELECT * WHERE { ?s ?p ?o }"
                        } else {
                            "SELECT * WHERE { << ?s ?p ?o >> ?m ?v }"
                        };
                        let _ = optimizer.optimize_query(query);
                    }

                    let stats = optimizer.statistics();
                    black_box(stats)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency of ChunkedIterator
fn benchmark_chunked_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_memory_efficiency");

    let large_dataset_size = 1_000_000;
    let chunk_sizes = [100, 1_000, 10_000, 100_000];

    for chunk_size in chunk_sizes.iter() {
        group.throughput(Throughput::Elements(large_dataset_size as u64));

        group.bench_with_input(
            BenchmarkId::new("chunk_size", chunk_size),
            chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    // Simulate processing large dataset in chunks
                    let mut processed = 0;
                    let data_iter = 0..large_dataset_size;
                    let chunked = ChunkedIterator::new(data_iter, chunk_size);

                    for chunk in chunked {
                        processed += chunk.len();
                        black_box(&chunk);
                    }

                    black_box(processed)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    chunked_iterator_benches,
    benchmark_chunked_iterator,
    benchmark_chunking_comparison,
    benchmark_triple_batching,
    benchmark_chunked_memory_efficiency,
);

criterion_group!(
    adaptive_optimizer_benches,
    benchmark_strategy_selection,
    benchmark_adaptive_optimization,
    benchmark_regression_detection,
    benchmark_multi_objective_setup,
    benchmark_workload_profiling,
    benchmark_auto_tuning_warmup,
);

criterion_main!(chunked_iterator_benches, adaptive_optimizer_benches);
