//! Property Path Evaluation Benchmarks
//!
//! Comprehensive benchmarks for property path evaluation performance after SplitRS refactoring.
//! These benchmarks ensure that the module reorganization maintains optimal performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_core::model::NamedNode;
use oxirs_shacl::{paths::PropertyPathEvaluator, PropertyPath};

/// Create a simple predicate path
fn create_predicate_path() -> PropertyPath {
    let predicate = NamedNode::new("http://example.org/knows").unwrap();
    PropertyPath::predicate(predicate)
}

/// Create a complex sequence path
fn create_sequence_path(depth: usize) -> PropertyPath {
    let predicates: Vec<PropertyPath> = (0..depth)
        .map(|i| {
            let pred = NamedNode::new(format!("http://example.org/prop{i}")).unwrap();
            PropertyPath::predicate(pred)
        })
        .collect();
    PropertyPath::sequence(predicates)
}

/// Create an alternative path
fn create_alternative_path(alternatives: usize) -> PropertyPath {
    let paths: Vec<PropertyPath> = (0..alternatives)
        .map(|i| {
            let pred = NamedNode::new(format!("http://example.org/alt{i}")).unwrap();
            PropertyPath::predicate(pred)
        })
        .collect();
    PropertyPath::alternative(paths)
}

/// Create a recursive path (zero-or-more)
fn create_recursive_path() -> PropertyPath {
    let predicate = NamedNode::new("http://example.org/parent").unwrap();
    PropertyPath::zero_or_more(PropertyPath::predicate(predicate))
}

/// Benchmark simple predicate path creation
fn bench_path_creation_simple(c: &mut Criterion) {
    c.bench_function("path_creation_simple", |b| {
        b.iter(|| black_box(create_predicate_path()))
    });
}

/// Benchmark complex path creation
fn bench_path_creation_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("path_creation_complex");

    for depth in [2, 5, 10].iter() {
        group.throughput(Throughput::Elements(*depth as u64));
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            b.iter(|| black_box(create_sequence_path(depth)))
        });
    }

    group.finish();
}

/// Benchmark path evaluator creation
fn bench_evaluator_creation(c: &mut Criterion) {
    c.bench_function("evaluator_creation_default", |b| {
        b.iter(|| black_box(PropertyPathEvaluator::new()))
    });

    c.bench_function("evaluator_creation_custom_limits", |b| {
        b.iter(|| black_box(PropertyPathEvaluator::with_limits(100, 5000)))
    });
}

/// Benchmark path complexity calculation
fn bench_path_complexity(c: &mut Criterion) {
    let simple_path = create_predicate_path();
    let complex_path = create_sequence_path(10);
    let recursive_path = create_recursive_path();

    c.bench_function("complexity_simple", |b| {
        b.iter(|| black_box(simple_path.complexity()))
    });

    c.bench_function("complexity_sequence", |b| {
        b.iter(|| black_box(complex_path.complexity()))
    });

    c.bench_function("complexity_recursive", |b| {
        b.iter(|| black_box(recursive_path.complexity()))
    });
}

/// Benchmark SPARQL path generation
fn bench_sparql_path_generation(c: &mut Criterion) {
    let simple_path = create_predicate_path();
    let sequence_path = create_sequence_path(5);
    let alternative_path = create_alternative_path(3);

    c.bench_function("sparql_gen_simple", |b| {
        b.iter(|| black_box(simple_path.to_sparql_path().unwrap()))
    });

    c.bench_function("sparql_gen_sequence", |b| {
        b.iter(|| black_box(sequence_path.to_sparql_path().unwrap()))
    });

    c.bench_function("sparql_gen_alternative", |b| {
        b.iter(|| black_box(alternative_path.to_sparql_path().unwrap()))
    });
}

/// Benchmark cache operations
fn bench_cache_operations(c: &mut Criterion) {
    let mut evaluator = PropertyPathEvaluator::new();

    c.bench_function("cache_stats_retrieval", |b| {
        b.iter(|| black_box(evaluator.get_cache_stats()))
    });

    c.bench_function("cache_clear", |b| {
        b.iter(|| {
            evaluator.clear_cache();
            black_box(())
        })
    });
}

/// Benchmark path type checking
fn bench_path_type_checking(c: &mut Criterion) {
    let predicate_path = create_predicate_path();
    let complex_path = create_sequence_path(5);

    c.bench_function("is_predicate_check", |b| {
        b.iter(|| black_box(predicate_path.is_predicate()))
    });

    c.bench_function("is_complex_check", |b| {
        b.iter(|| black_box(complex_path.is_complex()))
    });

    c.bench_function("as_predicate_extraction", |b| {
        b.iter(|| black_box(predicate_path.as_predicate()))
    });
}

/// Benchmark inverse path operations
fn bench_inverse_paths(c: &mut Criterion) {
    let predicate = NamedNode::new("http://example.org/knows").unwrap();
    let simple_path = PropertyPath::predicate(predicate);

    c.bench_function("inverse_path_creation", |b| {
        b.iter(|| black_box(PropertyPath::inverse(simple_path.clone())))
    });

    let inverse_path = PropertyPath::inverse(simple_path);

    c.bench_function("inverse_sparql_generation", |b| {
        b.iter(|| black_box(inverse_path.to_sparql_path().unwrap()))
    });
}

/// Benchmark alternative paths with varying alternatives
fn bench_alternative_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("alternative_scaling");

    for count in [2, 5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter(|| black_box(create_alternative_path(count)))
        });
    }

    group.finish();
}

/// Benchmark memory usage with evaluator limits
fn bench_evaluator_limits(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluator_limits");

    for max_depth in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(max_depth),
            max_depth,
            |b, &max_depth| {
                b.iter(|| black_box(PropertyPathEvaluator::with_limits(max_depth, 10000)))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_path_creation_simple,
    bench_path_creation_complex,
    bench_evaluator_creation,
    bench_path_complexity,
    bench_sparql_path_generation,
    bench_cache_operations,
    bench_path_type_checking,
    bench_inverse_paths,
    bench_alternative_scaling,
    bench_evaluator_limits,
);

criterion_main!(benches);
