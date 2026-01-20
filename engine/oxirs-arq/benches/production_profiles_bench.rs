//! Production Workload Profile Benchmarks
//!
//! This benchmark suite validates the performance characteristics of all 6
//! production optimizer profiles and demonstrates their suitability for
//! different deployment scenarios.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_arq::algebra::{Algebra, BinaryOperator, Expression, Term, TriplePattern, Variable};
use oxirs_arq::optimizer::production_tuning::ProductionOptimizerConfig;
use oxirs_arq::optimizer::Optimizer;
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

/// Create a complex analytical query (large BGP + multiple joins)
fn create_analytical_query() -> Algebra {
    // Create multiple BGPs representing an analytical query
    let bgp1 = Algebra::Bgp(vec![
        TriplePattern {
            subject: Term::Variable(Variable::new("product").expect("valid var")),
            predicate: Term::Iri(NamedNode::new("http://schema.org/name").expect("valid iri")),
            object: Term::Variable(Variable::new("name").expect("valid var")),
        },
        TriplePattern {
            subject: Term::Variable(Variable::new("product").expect("valid var")),
            predicate: Term::Iri(NamedNode::new("http://schema.org/price").expect("valid iri")),
            object: Term::Variable(Variable::new("price").expect("valid var")),
        },
        TriplePattern {
            subject: Term::Variable(Variable::new("product").expect("valid var")),
            predicate: Term::Iri(NamedNode::new("http://schema.org/category").expect("valid iri")),
            object: Term::Variable(Variable::new("category").expect("valid var")),
        },
    ]);

    let bgp2 = Algebra::Bgp(vec![
        TriplePattern {
            subject: Term::Variable(Variable::new("product").expect("valid var")),
            predicate: Term::Iri(NamedNode::new("http://schema.org/seller").expect("valid iri")),
            object: Term::Variable(Variable::new("seller").expect("valid var")),
        },
        TriplePattern {
            subject: Term::Variable(Variable::new("seller").expect("valid var")),
            predicate: Term::Iri(NamedNode::new("http://schema.org/rating").expect("valid iri")),
            object: Term::Variable(Variable::new("rating").expect("valid var")),
        },
    ]);

    let join1 = Algebra::Join {
        left: Box::new(bgp1),
        right: Box::new(bgp2),
    };

    // Add filter for price > 100
    let filter = Expression::Binary {
        op: BinaryOperator::Greater,
        left: Box::new(Expression::Variable(
            Variable::new("price").expect("valid var"),
        )),
        right: Box::new(Expression::Literal(oxirs_arq::algebra::Literal::integer(
            100,
        ))),
    };

    Algebra::Filter {
        pattern: Box::new(join1),
        condition: filter,
    }
}

/// Benchmark all 6 production profiles with simple queries
fn bench_profiles_simple_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiles_simple");
    group.measurement_time(Duration::from_secs(10));

    let profiles = vec![
        (
            "HighThroughput",
            ProductionOptimizerConfig::high_throughput(),
        ),
        (
            "Analytical",
            ProductionOptimizerConfig::analytical_queries(),
        ),
        ("Mixed", ProductionOptimizerConfig::mixed()),
        ("LowMemory", ProductionOptimizerConfig::low_memory()),
        ("LowCpu", ProductionOptimizerConfig::low_cpu()),
        (
            "MaxPerformance",
            ProductionOptimizerConfig::max_performance(),
        ),
    ];

    let query = create_simple_bgp(5);

    for (name, config) in profiles {
        group.bench_with_input(BenchmarkId::from_parameter(name), &config, |b, config| {
            b.iter(|| {
                // Create fresh optimizer each iteration to avoid plan cache
                let mut optimizer = Optimizer::new(config.base_config.clone());
                optimizer.optimize(black_box(query.clone())).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark profiles with analytical queries
fn bench_profiles_analytical(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiles_analytical");
    group.measurement_time(Duration::from_secs(15));

    let profiles = vec![
        (
            "HighThroughput",
            ProductionOptimizerConfig::high_throughput(),
        ),
        (
            "Analytical",
            ProductionOptimizerConfig::analytical_queries(),
        ),
        ("Mixed", ProductionOptimizerConfig::mixed()),
        (
            "MaxPerformance",
            ProductionOptimizerConfig::max_performance(),
        ),
    ];

    let query = create_analytical_query();

    for (name, config) in profiles {
        group.bench_with_input(BenchmarkId::from_parameter(name), &config, |b, config| {
            b.iter(|| {
                // Create fresh optimizer each iteration to avoid plan cache
                let mut optimizer = Optimizer::new(config.base_config.clone());
                optimizer.optimize(black_box(query.clone())).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark high-throughput profile with varying pattern counts
fn bench_high_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_throughput_scaling");
    group.measurement_time(Duration::from_secs(10));

    let config = ProductionOptimizerConfig::high_throughput();

    for num_patterns in [2, 5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_patterns),
            num_patterns,
            |b, &num| {
                let query = create_simple_bgp(num);
                b.iter(|| {
                    // Create fresh optimizer each iteration to avoid plan cache
                    let mut optimizer = Optimizer::new(config.base_config.clone());
                    optimizer.optimize(black_box(query.clone())).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark analytical profile with varying pattern counts
fn bench_analytical_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("analytical_scaling");
    group.measurement_time(Duration::from_secs(15));

    let config = ProductionOptimizerConfig::analytical_queries();

    for num_patterns in [2, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_patterns),
            num_patterns,
            |b, &num| {
                let query = create_simple_bgp(num);
                b.iter(|| {
                    // Create fresh optimizer each iteration to avoid plan cache
                    let mut optimizer = Optimizer::new(config.base_config.clone());
                    optimizer.optimize(black_box(query.clone())).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark low-memory profile
fn bench_low_memory_profile(c: &mut Criterion) {
    let mut group = c.benchmark_group("low_memory_profile");
    group.measurement_time(Duration::from_secs(10));

    let config = ProductionOptimizerConfig::low_memory();
    let queries = vec![
        ("simple", create_simple_bgp(5)),
        ("analytical", create_analytical_query()),
    ];

    for (query_type, query) in queries {
        group.bench_with_input(
            BenchmarkId::from_parameter(query_type),
            &query,
            |b, query| {
                b.iter(|| {
                    // Create fresh optimizer each iteration to avoid plan cache
                    let mut optimizer = Optimizer::new(config.base_config.clone());
                    optimizer.optimize(black_box(query.clone())).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark max-performance profile with complex queries
fn bench_max_performance_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_performance_complex");
    group.measurement_time(Duration::from_secs(15));

    let config = ProductionOptimizerConfig::max_performance();

    // Build increasingly complex join patterns
    for join_depth in [2, 4, 6, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(join_depth),
            join_depth,
            |b, &depth| {
                // Build nested joins
                let mut algebra = create_simple_bgp(1);
                for _ in 1..depth {
                    let right = create_simple_bgp(1);
                    algebra = Algebra::Join {
                        left: Box::new(algebra),
                        right: Box::new(right),
                    };
                }

                b.iter(|| {
                    // Create fresh optimizer each iteration to avoid plan cache
                    let mut optimizer = Optimizer::new(config.base_config.clone());
                    optimizer.optimize(black_box(algebra.clone())).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark resource estimation overhead
fn bench_resource_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("resource_estimation");
    group.measurement_time(Duration::from_secs(5));

    let profiles = vec![
        (
            "HighThroughput",
            ProductionOptimizerConfig::high_throughput(),
        ),
        (
            "Analytical",
            ProductionOptimizerConfig::analytical_queries(),
        ),
        ("Mixed", ProductionOptimizerConfig::mixed()),
        ("LowMemory", ProductionOptimizerConfig::low_memory()),
        ("LowCpu", ProductionOptimizerConfig::low_cpu()),
        (
            "MaxPerformance",
            ProductionOptimizerConfig::max_performance(),
        ),
    ];

    for (name, config) in profiles {
        group.bench_with_input(BenchmarkId::from_parameter(name), &config, |b, config| {
            b.iter(|| {
                let _resources = black_box(config.estimate_resource_requirements());
            });
        });
    }

    group.finish();
}

/// Benchmark configuration validation overhead
fn bench_config_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_validation");
    group.measurement_time(Duration::from_secs(5));

    let profiles = vec![
        (
            "HighThroughput",
            ProductionOptimizerConfig::high_throughput(),
        ),
        (
            "Analytical",
            ProductionOptimizerConfig::analytical_queries(),
        ),
        ("Mixed", ProductionOptimizerConfig::mixed()),
        ("LowMemory", ProductionOptimizerConfig::low_memory()),
        ("LowCpu", ProductionOptimizerConfig::low_cpu()),
        (
            "MaxPerformance",
            ProductionOptimizerConfig::max_performance(),
        ),
    ];

    for (name, config) in profiles {
        group.bench_with_input(BenchmarkId::from_parameter(name), &config, |b, config| {
            b.iter(|| {
                let _warnings = black_box(config.validate());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_profiles_simple_query,
    bench_profiles_analytical,
    bench_high_throughput_scaling,
    bench_analytical_scaling,
    bench_low_memory_profile,
    bench_max_performance_complex,
    bench_resource_estimation,
    bench_config_validation,
);

criterion_main!(benches);
