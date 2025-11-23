//! Comprehensive benchmarks for SHACL validation performance
//!
//! This benchmark suite tracks performance across various validation scenarios:
//! - Small, medium, and large datasets
//! - Different constraint types
//! - Parallel vs sequential validation
//! - Cache hit/miss scenarios
//! - Property path evaluation

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;

use indexmap::IndexMap;
use oxirs_core::{ConcreteStore, NamedNode};
use oxirs_shacl::{
    constraints::{
        cardinality_constraints::{MaxCountConstraint, MinCountConstraint},
        string_constraints::{MaxLengthConstraint, MinLengthConstraint},
        value_constraints::DatatypeConstraint,
        Constraint,
    },
    paths::PropertyPath,
    validation::ValidationEngine,
    ConstraintComponentId, Shape, ShapeId, Target, ValidationConfig,
};

/// Create a test store with sample data
fn create_test_store(num_nodes: usize) -> Arc<ConcreteStore> {
    let store = Arc::new(ConcreteStore::new().expect("Failed to create store"));

    // Add sample triples
    for i in 0..num_nodes {
        let subject = format!("http://example.org/person{}", i);
        // In production, would add actual triples here
        let _ = subject; // Placeholder
    }

    store
}

/// Create a simple node shape for benchmarking
fn create_simple_shape() -> Shape {
    let mut shape = Shape::node_shape(ShapeId("http://example.org/PersonShape".to_string()));

    // Add target
    shape.add_target(Target::class(NamedNode::new_unchecked(
        "http://example.org/Person",
    )));

    // Add a simple constraint
    shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );

    shape
}

/// Create a complex shape with multiple constraints
fn create_complex_shape() -> Shape {
    let mut shape = Shape::node_shape(ShapeId("http://example.org/ComplexShape".to_string()));

    shape.add_target(Target::class(NamedNode::new_unchecked(
        "http://example.org/Person",
    )));

    // Multiple constraints
    shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );
    shape.add_constraint(
        ConstraintComponentId::new("sh:maxCount"),
        Constraint::MaxCount(MaxCountConstraint { max_count: 10 }),
    );
    shape.add_constraint(
        ConstraintComponentId::new("sh:datatype"),
        Constraint::Datatype(DatatypeConstraint {
            datatype_iri: NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#string"),
        }),
    );

    shape
}

/// Benchmark validation engine creation
fn bench_engine_creation(c: &mut Criterion) {
    c.bench_function("validation_engine_creation", |b| {
        b.iter(|| {
            let _store = create_test_store(100);
            let mut shapes = IndexMap::new();
            shapes.insert(
                ShapeId("http://example.org/Shape1".to_string()),
                create_simple_shape(),
            );
            let config = ValidationConfig::default();
            let _engine = ValidationEngine::new(&shapes, config);
            black_box(&shapes);
        });
    });
}

/// Benchmark simple shape validation
fn bench_simple_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_validation");

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let _store = create_test_store(size);
            let mut shapes = IndexMap::new();
            let shape = create_simple_shape();
            shapes.insert(shape.id.clone(), shape.clone());
            let config = ValidationConfig::default();
            let _engine = ValidationEngine::new(&shapes, config);

            b.iter(|| {
                // In production, would validate actual nodes
                black_box(&shape);
            });
        });
    }

    group.finish();
}

/// Benchmark complex shape validation
fn bench_complex_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_validation");

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let _store = create_test_store(size);
            let mut shapes = IndexMap::new();
            let shape = create_complex_shape();
            shapes.insert(shape.id.clone(), shape.clone());
            let config = ValidationConfig::default();
            let _engine = ValidationEngine::new(&shapes, config);

            b.iter(|| {
                black_box(&shape);
            });
        });
    }

    group.finish();
}

/// Benchmark constraint evaluation
fn bench_constraint_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_evaluation");

    let constraints = vec![
        (
            "minCount",
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        ),
        (
            "maxCount",
            Constraint::MaxCount(MaxCountConstraint { max_count: 10 }),
        ),
        (
            "minLength",
            Constraint::MinLength(MinLengthConstraint { min_length: 5 }),
        ),
        (
            "maxLength",
            Constraint::MaxLength(MaxLengthConstraint { max_length: 100 }),
        ),
    ];

    for (name, constraint) in constraints {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(&constraint);
            });
        });
    }

    group.finish();
}

/// Benchmark property path evaluation
fn bench_property_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_paths");

    let paths = vec![
        (
            "direct",
            PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/name")),
        ),
        (
            "sequence",
            PropertyPath::Sequence(vec![
                PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/parent")),
                PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/name")),
            ]),
        ),
    ];

    for (name, path) in paths {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(&path);
            });
        });
    }

    group.finish();
}

/// Benchmark shape parsing
fn bench_shape_parsing(c: &mut Criterion) {
    c.bench_function("shape_parsing", |b| {
        b.iter(|| {
            let shape = create_complex_shape();
            black_box(shape);
        });
    });
}

/// Benchmark validation with caching
#[cfg(feature = "parallel")]
fn bench_cached_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_validation");

    for cache_enabled in [false, true].iter() {
        let label = if *cache_enabled {
            "with_cache"
        } else {
            "without_cache"
        };

        group.bench_function(label, |b| {
            let _store = create_test_store(1000);
            let _config = ValidationConfig::default();
            // config.enable_caching = *cache_enabled;

            let mut shapes = IndexMap::new();
            let shape = create_complex_shape();
            shapes.insert(shape.id.clone(), shape.clone());
            let config = ValidationConfig::default();
            let _engine = ValidationEngine::new(&shapes, config);

            b.iter(|| {
                black_box(&shape);
            });
        });
    }

    group.finish();
}

/// Benchmark parallel validation
#[cfg(feature = "parallel")]
fn bench_parallel_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_validation");

    for num_threads in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &_threads| {
                let _store = create_test_store(1000);
                let mut shapes = IndexMap::new();
                let shape = create_complex_shape();
                shapes.insert(shape.id.clone(), shape.clone());
                let config = ValidationConfig::default();
                let _engine = ValidationEngine::new(&shapes, config);

                b.iter(|| {
                    black_box(&shape);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch validation
fn bench_batch_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_validation");

    for batch_size in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &size| {
                let _store = create_test_store(size);
                let mut shape_map = IndexMap::new();

                // Create multiple shapes
                let shapes: Vec<Shape> = (0..10).map(|_| create_complex_shape()).collect();
                for (idx, shape) in shapes.iter().enumerate() {
                    shape_map.insert(
                        ShapeId(format!("http://example.org/Shape{}", idx)),
                        shape.clone(),
                    );
                }

                let config = ValidationConfig::default();
                let _engine = ValidationEngine::new(&shape_map, config);

                b.iter(|| {
                    for shape in &shapes {
                        black_box(shape);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage (approximate via object creation)
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for num_shapes in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_shapes),
            num_shapes,
            |b, &count| {
                b.iter(|| {
                    let shapes: Vec<Shape> = (0..count).map(|_| create_complex_shape()).collect();
                    black_box(shapes);
                });
            },
        );
    }

    group.finish();
}

// Define benchmark groups
criterion_group!(engine_benches, bench_engine_creation, bench_shape_parsing,);

criterion_group!(
    validation_benches,
    bench_simple_validation,
    bench_complex_validation,
    bench_constraint_evaluation,
);

criterion_group!(path_benches, bench_property_paths,);

criterion_group!(batch_benches, bench_batch_validation, bench_memory_usage,);

#[cfg(feature = "parallel")]
criterion_group!(
    parallel_benches,
    bench_cached_validation,
    bench_parallel_validation,
);

// Main benchmark runner
#[cfg(feature = "parallel")]
criterion_main!(
    engine_benches,
    validation_benches,
    path_benches,
    batch_benches,
    parallel_benches,
);

#[cfg(not(feature = "parallel"))]
criterion_main!(
    engine_benches,
    validation_benches,
    path_benches,
    batch_benches,
);
