//! Benchmarks for Query Visualization and Profiling System
//!
//! This benchmark suite measures the performance of:
//! - Cost-based optimizer visualization export
//! - Query plan visualization rendering
//! - Profiled plan building
//! - Performance analysis and comparison
//! - Optimization hint generation

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_core::model::{NamedNode, Variable};
use oxirs_core::query::algebra::{AlgebraTriplePattern, GraphPattern, TermPattern};
use oxirs_core::query::cost_based_optimizer::CostBasedOptimizer;
use oxirs_core::query::profiled_plan_builder::ProfiledPlanBuilder;
use oxirs_core::query::query_plan_visualizer::QueryPlanVisualizer;
use oxirs_core::query::query_profiler::{ProfilerConfig, QueryProfiler, QueryStatistics};
use std::collections::HashMap;

// Helper functions for creating test patterns

fn create_simple_bgp(num_patterns: usize) -> GraphPattern {
    let person_var = Variable::new("person").unwrap();
    let name_var = Variable::new("name").unwrap();
    let foaf_name = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();

    let mut patterns = Vec::new();
    for i in 0..num_patterns {
        let var = Variable::new(format!("var{}", i)).unwrap();
        patterns.push(AlgebraTriplePattern {
            subject: TermPattern::Variable(person_var.clone()),
            predicate: TermPattern::NamedNode(foaf_name.clone()),
            object: TermPattern::Variable(if i % 2 == 0 { name_var.clone() } else { var }),
        });
    }

    GraphPattern::Bgp(patterns)
}

fn create_nested_joins(depth: usize) -> GraphPattern {
    let person_var = Variable::new("person").unwrap();
    let foaf_name = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();

    let base_pattern = GraphPattern::Bgp(vec![AlgebraTriplePattern {
        subject: TermPattern::Variable(person_var.clone()),
        predicate: TermPattern::NamedNode(foaf_name.clone()),
        object: TermPattern::Variable(Variable::new("name").unwrap()),
    }]);

    if depth == 0 {
        return base_pattern;
    }

    let mut result = base_pattern;
    for _ in 0..depth {
        let right = GraphPattern::Bgp(vec![AlgebraTriplePattern {
            subject: TermPattern::Variable(person_var.clone()),
            predicate: TermPattern::NamedNode(foaf_name.clone()),
            object: TermPattern::Variable(Variable::new("other").unwrap()),
        }]);
        result = GraphPattern::Join(Box::new(result), Box::new(right));
    }

    result
}

fn create_sample_statistics(pattern_count: usize) -> QueryStatistics {
    let mut pattern_matches = HashMap::new();
    for i in 0..pattern_count {
        pattern_matches.insert(format!("?s ?p ?o_{}", i), 1000 + i as u64 * 100);
    }

    let mut index_accesses = HashMap::new();
    index_accesses.insert("SPO".to_string(), pattern_count as u64);

    QueryStatistics {
        total_time_ms: 150,
        parse_time_ms: 10,
        planning_time_ms: 20,
        execution_time_ms: 120,
        triples_matched: pattern_count as u64 * 1000,
        results_count: 500,
        peak_memory_bytes: 2 * 1024 * 1024,
        join_operations: pattern_count.saturating_sub(1) as u64,
        cache_hit_rate: 0.75,
        pattern_matches,
        index_accesses,
        ..Default::default()
    }
}

// Benchmarks for Cost-Based Optimizer Visualization Export

fn bench_optimizer_visualization_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_visualization_export");

    for size in [1, 3, 5, 10, 20].iter() {
        let pattern = create_simple_bgp(*size);
        let optimizer = CostBasedOptimizer::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_patterns", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let plan = optimizer.optimize_pattern(black_box(&pattern)).unwrap();
                    let visual_plan =
                        optimizer.to_visual_plan(black_box(&pattern), black_box(&plan));
                    black_box(visual_plan);
                });
            },
        );
    }

    group.finish();
}

fn bench_nested_join_visualization(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_join_visualization");

    for depth in [1, 2, 3, 5].iter() {
        let pattern = create_nested_joins(*depth);
        let optimizer = CostBasedOptimizer::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("depth_{}", depth)),
            depth,
            |b, _| {
                b.iter(|| {
                    let plan = optimizer.optimize_pattern(black_box(&pattern)).unwrap();
                    let visual_plan =
                        optimizer.to_visual_plan(black_box(&pattern), black_box(&plan));
                    black_box(visual_plan);
                });
            },
        );
    }

    group.finish();
}

// Benchmarks for Query Plan Visualizer

fn bench_plan_visualization_rendering(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_visualization_rendering");

    for size in [1, 3, 5, 10, 20].iter() {
        let pattern = create_simple_bgp(*size);
        let optimizer = CostBasedOptimizer::new();
        let plan = optimizer.optimize_pattern(&pattern).unwrap();
        let visual_plan = optimizer.to_visual_plan(&pattern, &plan);
        let visualizer = QueryPlanVisualizer::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_nodes", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let tree = visualizer.visualize_as_tree(black_box(&visual_plan));
                    black_box(tree);
                });
            },
        );
    }

    group.finish();
}

fn bench_optimization_hint_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_hint_generation");

    for size in [1, 3, 5, 10, 20].iter() {
        let pattern = create_simple_bgp(*size);
        let optimizer = CostBasedOptimizer::new();
        let plan = optimizer.optimize_pattern(&pattern).unwrap();
        let visual_plan = optimizer.to_visual_plan(&pattern, &plan);
        let visualizer = QueryPlanVisualizer::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_patterns", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let hints = visualizer.suggest_optimizations(black_box(&visual_plan));
                    black_box(hints);
                });
            },
        );
    }

    group.finish();
}

fn bench_plan_summary_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_summary_generation");

    for size in [1, 3, 5, 10, 20].iter() {
        let pattern = create_simple_bgp(*size);
        let optimizer = CostBasedOptimizer::new();
        let plan = optimizer.optimize_pattern(&pattern).unwrap();
        let visual_plan = optimizer.to_visual_plan(&pattern, &plan);
        let visualizer = QueryPlanVisualizer::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_patterns", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let summary = visualizer.generate_summary(black_box(&visual_plan));
                    black_box(summary);
                });
            },
        );
    }

    group.finish();
}

// Benchmarks for Profiled Plan Builder

fn bench_profiled_plan_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiled_plan_building");

    for size in [1, 5, 10, 20, 50].iter() {
        let stats = create_sample_statistics(*size);
        let builder = ProfiledPlanBuilder::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_patterns", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let plan =
                        builder.build_from_stats(black_box(&stats), "SELECT * WHERE { ... }");
                    black_box(plan);
                });
            },
        );
    }

    group.finish();
}

fn bench_performance_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_analysis");

    for size in [1, 5, 10, 20, 50].iter() {
        let stats = create_sample_statistics(*size);
        let builder = ProfiledPlanBuilder::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_patterns", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let analysis = builder.analyze_performance(black_box(&stats));
                    black_box(analysis);
                });
            },
        );
    }

    group.finish();
}

fn bench_execution_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_comparison");

    let baseline = create_sample_statistics(10);
    let current = create_sample_statistics(10);
    let builder = ProfiledPlanBuilder::new();

    group.bench_function("compare_executions", |b| {
        b.iter(|| {
            let comparison = builder.compare_executions(black_box(&baseline), black_box(&current));
            black_box(comparison);
        });
    });

    group.finish();
}

fn bench_profiling_report_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_report_generation");

    for size in [1, 5, 10, 20].iter() {
        let stats = create_sample_statistics(*size);
        let builder = ProfiledPlanBuilder::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_patterns", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let report =
                        builder.generate_report(black_box(&stats), "SELECT * WHERE { ... }");
                    black_box(report);
                });
            },
        );
    }

    group.finish();
}

// Benchmarks for Query Profiler

fn bench_profiler_session_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiler_session_overhead");

    let profiler = QueryProfiler::new(ProfilerConfig::default());

    group.bench_function("session_start_finish", |b| {
        b.iter(|| {
            let session = profiler.start_session(black_box("SELECT * WHERE { ?s ?p ?o }"));
            black_box(session);
            // Session automatically finishes when dropped
        });
    });

    group.finish();
}

// Complete Integration Benchmark

fn bench_complete_integration_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_integration_pipeline");

    for size in [1, 3, 5, 10].iter() {
        let pattern = create_simple_bgp(*size);
        let stats = create_sample_statistics(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_patterns", size)),
            size,
            |b, _| {
                b.iter(|| {
                    // Step 1: Optimize
                    let optimizer = CostBasedOptimizer::new();
                    let plan = optimizer.optimize_pattern(black_box(&pattern)).unwrap();

                    // Step 2: Generate visual plan
                    let visual_plan =
                        optimizer.to_visual_plan(black_box(&pattern), black_box(&plan));

                    // Step 3: Build profiled plan
                    let builder = ProfiledPlanBuilder::new();
                    let profiled_plan =
                        builder.build_from_stats(black_box(&stats), "SELECT * WHERE { ... }");

                    // Step 4: Visualize
                    let visualizer = QueryPlanVisualizer::new();
                    let tree = visualizer.visualize_as_tree(black_box(&visual_plan));
                    let summary = visualizer.generate_summary(black_box(&profiled_plan));

                    // Step 5: Analyze
                    let analysis = builder.analyze_performance(black_box(&stats));

                    black_box((tree, summary, analysis));
                });
            },
        );
    }

    group.finish();
}

// JSON Export Benchmarks

fn bench_json_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_export");

    for size in [1, 5, 10, 20].iter() {
        let pattern = create_simple_bgp(*size);
        let optimizer = CostBasedOptimizer::new();
        let plan = optimizer.optimize_pattern(&pattern).unwrap();
        let visual_plan = optimizer.to_visual_plan(&pattern, &plan);
        let visualizer = QueryPlanVisualizer::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_nodes", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let json = visualizer.export_as_json(black_box(&visual_plan)).unwrap();
                    black_box(json);
                });
            },
        );
    }

    group.finish();
}

// Memory Efficiency Benchmarks

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test that visualization doesn't cause excessive allocations
    for size in [10, 50, 100].iter() {
        let pattern = create_simple_bgp(*size);
        let optimizer = CostBasedOptimizer::new();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_patterns_full_pipeline", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let plan = optimizer.optimize_pattern(black_box(&pattern)).unwrap();
                    let visual_plan =
                        optimizer.to_visual_plan(black_box(&pattern), black_box(&plan));
                    let visualizer = QueryPlanVisualizer::new();
                    let _tree = visualizer.visualize_as_tree(&visual_plan);
                    let _summary = visualizer.generate_summary(&visual_plan);
                    let _hints = visualizer.suggest_optimizations(&visual_plan);
                    // All data structures should be efficiently deallocated
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    visualization_benches,
    bench_optimizer_visualization_export,
    bench_nested_join_visualization,
    bench_plan_visualization_rendering,
    bench_optimization_hint_generation,
    bench_plan_summary_generation,
);

criterion_group!(
    profiling_benches,
    bench_profiled_plan_building,
    bench_performance_analysis,
    bench_execution_comparison,
    bench_profiling_report_generation,
    bench_profiler_session_overhead,
);

criterion_group!(
    integration_benches,
    bench_complete_integration_pipeline,
    bench_json_export,
    bench_memory_efficiency,
);

criterion_main!(
    visualization_benches,
    profiling_benches,
    integration_benches
);
