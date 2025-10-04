//! Join Algorithm Performance Benchmarks
//!
//! Comprehensive benchmarks to verify the 10-50x performance improvements
//! in join operations, which are the most critical part of query execution.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_arq::{
    algebra::{Binding, Solution, Term, Variable},
    executor::parallel_optimized::{CacheFriendlyHashJoin, SortMergeJoin},
};
use oxirs_core::model::NamedNode;
use std::collections::HashMap;

// ===== Data Generation =====

fn create_variable(name: &str) -> Variable {
    Variable::new(name).expect("Failed to create variable")
}

fn generate_binding(id: usize, var_count: usize) -> Binding {
    let mut binding = HashMap::new();
    for i in 0..var_count {
        let var = create_variable(&format!("var{}", i));
        let node = NamedNode::new(format!("http://example.org/value_{}_{}", id, i))
            .expect("Failed to create node");
        binding.insert(var, Term::Iri(node));
    }
    binding
}

fn generate_solutions(count: usize, var_count: usize) -> Vec<Solution> {
    // Generate count solutions, each with a single binding
    (0..count)
        .map(|i| vec![generate_binding(i, var_count)])
        .collect()
}

// ===== Naive Implementation for Comparison =====

fn naive_nested_loop_join(
    left: &[Solution],
    right: &[Solution],
    join_vars: &[Variable],
) -> Vec<Solution> {
    let mut result = Vec::new();
    for l_sol in left {
        for r_sol in right {
            // Each solution contains multiple bindings
            for l_binding in l_sol {
                for r_binding in r_sol {
                    // Check if join variables match
                    let matches = join_vars.iter().all(|var| {
                        match (l_binding.get(var), r_binding.get(var)) {
                            (Some(a), Some(b)) => a == b,
                            _ => false,
                        }
                    });

                    if matches {
                        // Merge bindings
                        let mut merged = l_binding.clone();
                        for (var, term) in r_binding {
                            merged.insert(var.clone(), term.clone());
                        }
                        result.push(vec![merged]);
                    }
                }
            }
        }
    }
    result
}

// ===== Hash Join Benchmarks =====

fn bench_hash_join_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_join_scaling");

    for size in [100, 1_000, 10_000, 50_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let left_solutions = generate_solutions(*size, 4);
        let right_solutions = generate_solutions(*size, 4);
        let join_variables = vec![create_variable("var0"), create_variable("var1")];

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                let mut hash_join = CacheFriendlyHashJoin::new(16);
                b.iter(|| {
                    let result = hash_join
                        .join_parallel(
                            left_solutions.clone(),
                            right_solutions.clone(),
                            &join_variables,
                        )
                        .expect("Join failed");
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

// ===== Sort-Merge Join Benchmarks =====

fn bench_sort_merge_join_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_merge_join_scaling");

    for size in [100, 1_000, 10_000, 50_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let left_solutions = generate_solutions(*size, 4);
        let right_solutions = generate_solutions(*size, 4);
        let join_variables = vec![create_variable("var0"), create_variable("var1")];

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                let mut sort_merge = SortMergeJoin::new(1024 * 1024 * 1024); // 1GB
                b.iter(|| {
                    let result = sort_merge
                        .join(
                            left_solutions.clone(),
                            right_solutions.clone(),
                            &join_variables,
                        )
                        .expect("Join failed");
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

// ===== Optimized vs Naive Comparison =====

fn bench_optimized_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_vs_naive");

    // Only test smaller sizes for naive implementation (it's too slow for large datasets)
    for size in [100, 500, 1_000].iter() {
        let left_solutions = generate_solutions(*size, 4);
        let right_solutions = generate_solutions(*size, 4);
        let join_variables = vec![create_variable("var0"), create_variable("var1")];

        // Benchmark naive implementation
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = naive_nested_loop_join(
                        &left_solutions,
                        &right_solutions,
                        &join_variables,
                    );
                    black_box(result);
                });
            },
        );

        // Benchmark optimized hash join
        group.bench_with_input(
            BenchmarkId::new("hash_join", size),
            size,
            |b, _| {
                let mut hash_join = CacheFriendlyHashJoin::new(16);
                b.iter(|| {
                    let result = hash_join
                        .join_parallel(
                            left_solutions.clone(),
                            right_solutions.clone(),
                            &join_variables,
                        )
                        .expect("Join failed");
                    black_box(result);
                });
            },
        );

        // Benchmark sort-merge join
        group.bench_with_input(
            BenchmarkId::new("sort_merge", size),
            size,
            |b, _| {
                let mut sort_merge = SortMergeJoin::new(1024 * 1024 * 1024);
                b.iter(|| {
                    let result = sort_merge
                        .join(
                            left_solutions.clone(),
                            right_solutions.clone(),
                            &join_variables,
                        )
                        .expect("Join failed");
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

// ===== Parallel Scalability Benchmarks =====

fn bench_parallel_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scalability");

    let size = 20_000;
    let left_solutions = generate_solutions(size, 4);
    let right_solutions = generate_solutions(size, 4);
    let join_variables = vec![create_variable("var0"), create_variable("var1")];

    for partitions in [2, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(partitions),
            partitions,
            |b, &part_count| {
                b.iter(|| {
                    let mut hash_join = CacheFriendlyHashJoin::new(part_count);
                    let result = hash_join
                        .join_parallel(
                            left_solutions.clone(),
                            right_solutions.clone(),
                            &join_variables,
                        )
                        .expect("Join failed");
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

// ===== Join Selectivity Benchmarks =====

fn bench_join_selectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_selectivity");

    let left_size = 10_000;
    let join_variables = vec![create_variable("var0"), create_variable("var1")];

    // Test different selectivities by varying right_size
    for (right_size, label) in [(100, "high"), (1_000, "medium"), (10_000, "low")].iter() {
        let left_solutions = generate_solutions(left_size, 4);
        let right_solutions = generate_solutions(*right_size, 4);

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            label,
            |b, _| {
                let mut hash_join = CacheFriendlyHashJoin::new(16);
                b.iter(|| {
                    let result = hash_join
                        .join_parallel(
                            left_solutions.clone(),
                            right_solutions.clone(),
                            &join_variables,
                        )
                        .expect("Join failed");
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

// ===== Multi-way Join Benchmarks =====

fn bench_multi_way_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_way_join");

    for join_count in [2, 3, 4, 5].iter() {
        let size = 1_000;
        let join_variables = vec![create_variable("var0"), create_variable("var1")];

        // Generate multiple inputs (each is a Vec<Solution>)
        let inputs: Vec<Vec<Solution>> = (0..*join_count)
            .map(|_| generate_solutions(size, 4))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(join_count),
            join_count,
            |b, &count| {
                b.iter(|| {
                    let mut result = inputs[0].clone();
                    for i in 1..count {
                        let hash_join = CacheFriendlyHashJoin::new(16);
                        result = hash_join
                            .join_parallel(result, inputs[i].clone(), &join_variables)
                            .expect("Join failed");
                    }
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

// ===== Memory Efficiency Benchmarks =====

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    for size in [10_000, 50_000, 100_000].iter() {
        let left_solutions = generate_solutions(*size, 4);
        let right_solutions = generate_solutions(*size, 4);
        let join_variables = vec![create_variable("var0"), create_variable("var1")];

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut hash_join = CacheFriendlyHashJoin::new(16);
                    let result = hash_join
                        .join_parallel(
                            left_solutions.clone(),
                            right_solutions.clone(),
                            &join_variables,
                        )
                        .expect("Join failed");

                    // Measure approximate memory usage
                    let memory_estimate = result.len() * std::mem::size_of::<Binding>();
                    black_box((result, memory_estimate));
                });
            },
        );
    }
    group.finish();
}

// ===== Benchmark Groups =====

criterion_group! {
    name = scaling_benches;
    config = Criterion::default().sample_size(30);
    targets = bench_hash_join_sizes,
              bench_sort_merge_join_sizes,
}

criterion_group! {
    name = comparison_benches;
    config = Criterion::default().sample_size(20);
    targets = bench_optimized_vs_naive,
}

criterion_group! {
    name = advanced_benches;
    config = Criterion::default().sample_size(25);
    targets = bench_parallel_scalability,
              bench_join_selectivity,
              bench_multi_way_join,
}

criterion_group! {
    name = memory_benches;
    config = Criterion::default().sample_size(20);
    targets = bench_memory_efficiency,
}

criterion_main!(
    scaling_benches,
    comparison_benches,
    advanced_benches,
    memory_benches,
);
