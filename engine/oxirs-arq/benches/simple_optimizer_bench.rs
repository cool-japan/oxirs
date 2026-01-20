//! Simple optimizer benchmark using current API
//!
//! This benchmark validates optimizer performance with the current codebase API.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_arq::algebra::{Algebra, BinaryOperator, Expression, Term, TriplePattern, Variable};
use oxirs_arq::optimizer::{Optimizer, OptimizerConfig};
use oxirs_core::NamedNode;
use std::time::Duration;

/// Create a simple BGP with n patterns
fn create_simple_bgp(num_patterns: usize) -> Algebra {
    let mut patterns = Vec::new();
    for i in 0..num_patterns {
        patterns.push(TriplePattern {
            subject: Term::Variable(Variable::new(format!("s{}", i)).expect("valid var")),
            predicate: Term::Variable(Variable::new(format!("p{}", i)).expect("valid var")),
            object: Term::Variable(Variable::new(format!("o{}", i)).expect("valid var")),
        });
    }
    Algebra::Bgp(patterns)
}

/// Create a complex query with join and filter
fn create_complex_query() -> Algebra {
    let bgp1 = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(Variable::new("person").expect("valid var")),
        predicate: Term::Iri(NamedNode::new("http://xmlns.com/foaf/0.1/name").expect("valid iri")),
        object: Term::Variable(Variable::new("name").expect("valid var")),
    }]);

    let bgp2 = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(Variable::new("person").expect("valid var")),
        predicate: Term::Iri(NamedNode::new("http://xmlns.com/foaf/0.1/knows").expect("valid iri")),
        object: Term::Variable(Variable::new("friend").expect("valid var")),
    }]);

    let join = Algebra::Join {
        left: Box::new(bgp1),
        right: Box::new(bgp2),
    };

    // Add filter
    let filter = Expression::Binary {
        op: BinaryOperator::Greater,
        left: Box::new(Expression::Variable(
            Variable::new("age").expect("valid var"),
        )),
        right: Box::new(Expression::Literal(oxirs_arq::algebra::Literal::integer(
            18,
        ))),
    };

    Algebra::Filter {
        pattern: Box::new(join),
        condition: filter,
    }
}

/// Benchmark optimizer with minimal configuration
fn bench_optimizer_minimal(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_minimal");
    group.measurement_time(Duration::from_secs(10));

    let config = OptimizerConfig {
        max_passes: 1, // Minimal optimization
        ..OptimizerConfig::default()
    };

    for num_patterns in [2, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_patterns),
            num_patterns,
            |b, &num| {
                let query = create_simple_bgp(num);
                let mut optimizer = Optimizer::new(config.clone());
                b.iter(|| {
                    optimizer.optimize(black_box(query.clone())).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark optimizer with full configuration
fn bench_optimizer_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_full");
    group.measurement_time(Duration::from_secs(10));

    let config = OptimizerConfig::default(); // All optimizations enabled

    for num_patterns in [2, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_patterns),
            num_patterns,
            |b, &num| {
                let query = create_simple_bgp(num);
                let mut optimizer = Optimizer::new(config.clone());
                b.iter(|| {
                    optimizer.optimize(black_box(query.clone())).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark complex query optimization
fn bench_complex_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_query");
    group.measurement_time(Duration::from_secs(10));

    let query = create_complex_query();
    let config = OptimizerConfig::default();

    group.bench_function("optimize_complex", |b| {
        let mut optimizer = Optimizer::new(config.clone());
        b.iter(|| {
            optimizer.optimize(black_box(query.clone())).unwrap();
        });
    });

    group.finish();
}

/// Benchmark join reordering
fn bench_join_reordering(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_reordering");
    group.measurement_time(Duration::from_secs(10));

    for depth in [2, 4, 6, 8].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            // Build nested joins
            let mut algebra = create_simple_bgp(1);
            for _ in 1..depth {
                let right = create_simple_bgp(1);
                algebra = Algebra::Join {
                    left: Box::new(algebra),
                    right: Box::new(right),
                };
            }

            let config = OptimizerConfig::default();
            let mut optimizer = Optimizer::new(config);
            b.iter(|| {
                optimizer.optimize(black_box(algebra.clone())).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_optimizer_minimal,
    bench_optimizer_full,
    bench_complex_query,
    bench_join_reordering,
);

criterion_main!(benches);
