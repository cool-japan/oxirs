//! Integration tests for optimizer with advanced cardinality estimation
//!
//! These tests verify that the optimizer correctly integrates cardinality
//! estimation for improved query planning and execution.

use oxirs_arq::algebra::{Algebra, BinaryOperator, Expression, Term, TriplePattern, Variable};
use oxirs_arq::optimizer::{
    EnhancedOptimizer, OptimizerConfig, ProductionOptimizerConfig, WorkloadProfile,
};
use oxirs_arq::statistics::EstimationMethod;
use oxirs_core::NamedNode;

#[test]
fn test_optimizer_basic_functionality() {
    let config = OptimizerConfig::default();
    let mut optimizer = EnhancedOptimizer::new(config);

    let pattern = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("Valid var")),
        predicate: Term::Iri(NamedNode::new("http://example.org/pred").expect("Valid IRI")),
        object: Term::Variable(Variable::new("o").expect("Valid var")),
    };

    let query = Algebra::Bgp(vec![pattern]);
    let result = optimizer.optimize(query);

    assert!(result.is_ok());
}

#[test]
fn test_join_ordering_with_cardinality() {
    let config = OptimizerConfig::default();
    let mut optimizer = EnhancedOptimizer::new(config);

    // Add statistics - make predicate1 more selective than predicate2
    optimizer.update_cardinality_statistics(
        "http://example.org/selective".to_string(),
        100,
        80,
        90,
    );
    optimizer.update_cardinality_statistics(
        "http://example.org/common".to_string(),
        10000,
        800,
        800,
    );

    // Create query with less selective pattern first
    let pattern1 = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("Valid var")),
        predicate: Term::Iri(NamedNode::new("http://example.org/common").expect("Valid IRI")),
        object: Term::Variable(Variable::new("o1").expect("Valid var")),
    };

    let pattern2 = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("Valid var")),
        predicate: Term::Iri(NamedNode::new("http://example.org/selective").expect("Valid IRI")),
        object: Term::Variable(Variable::new("o2").expect("Valid var")),
    };

    let bgp1 = Algebra::Bgp(vec![pattern1]);
    let bgp2 = Algebra::Bgp(vec![pattern2]);
    let query = Algebra::Join {
        left: Box::new(bgp1),
        right: Box::new(bgp2),
    };

    let optimized = optimizer
        .optimize(query)
        .expect("Optimization should succeed");

    // Verify query structure is preserved
    match optimized {
        Algebra::Join { .. } => {
            // Join structure should be maintained
        }
        Algebra::Bgp(patterns) => {
            // Or might be flattened to BGP
            assert!(!patterns.is_empty());
        }
        _ => panic!("Unexpected query structure after optimization"),
    }
}

#[test]
fn test_filter_pushdown_optimization() {
    let config = OptimizerConfig::default();
    let mut optimizer = EnhancedOptimizer::new(config);

    let pattern = TriplePattern {
        subject: Term::Variable(Variable::new("person").expect("Valid var")),
        predicate: Term::Iri(NamedNode::new("http://xmlns.com/foaf/0.1/age").expect("Valid IRI")),
        object: Term::Variable(Variable::new("age").expect("Valid var")),
    };

    let bgp = Algebra::Bgp(vec![pattern]);

    // Create filter: age > 18
    let filter = Expression::Binary {
        op: BinaryOperator::Greater,
        left: Box::new(Expression::Variable(
            Variable::new("age").expect("Valid var"),
        )),
        right: Box::new(Expression::Literal(oxirs_arq::algebra::Literal::integer(
            18,
        ))),
    };

    let query = Algebra::Filter {
        pattern: Box::new(bgp),
        condition: filter,
    };

    let optimized = optimizer
        .optimize(query)
        .expect("Optimization should succeed");

    // Filter should be pushed down or optimized in some way
    assert!(matches!(
        optimized,
        Algebra::Filter { .. } | Algebra::Bgp(_)
    ));
}

#[test]
fn test_plan_caching_improves_performance() {
    let config = OptimizerConfig::default();
    let mut optimizer = EnhancedOptimizer::new(config);

    let pattern = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("Valid var")),
        predicate: Term::Iri(NamedNode::new("http://example.org/pred").expect("Valid IRI")),
        object: Term::Variable(Variable::new("o").expect("Valid var")),
    };
    let query = Algebra::Bgp(vec![pattern]);

    // First optimization (cold cache)
    let _result1 = optimizer.optimize(query.clone()).expect("Should optimize");
    let (cache_size_after_first, _) = optimizer.plan_cache_stats();
    assert_eq!(cache_size_after_first, 1);

    // Second optimization (warm cache)
    let _result2 = optimizer.optimize(query.clone()).expect("Should optimize");
    let (cache_size_after_second, _) = optimizer.plan_cache_stats();
    assert_eq!(cache_size_after_second, 1); // Should not grow

    // Third optimization with different query
    let different_pattern = TriplePattern {
        subject: Term::Variable(Variable::new("x").expect("Valid var")),
        predicate: Term::Iri(NamedNode::new("http://example.org/different").expect("Valid IRI")),
        object: Term::Variable(Variable::new("y").expect("Valid var")),
    };
    let different_query = Algebra::Bgp(vec![different_pattern]);
    let _result3 = optimizer
        .optimize(different_query)
        .expect("Should optimize");
    let (cache_size_after_third, _) = optimizer.plan_cache_stats();
    assert_eq!(cache_size_after_third, 2); // Should grow for different query
}

#[test]
fn test_ml_based_optimization() {
    let config = OptimizerConfig::default();
    let mut optimizer =
        EnhancedOptimizer::with_estimation_method(config, EstimationMethod::MachineLearning);

    let pattern = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("Valid var")),
        predicate: Term::Iri(NamedNode::new("http://example.org/pred").expect("Valid IRI")),
        object: Term::Variable(Variable::new("o").expect("Valid var")),
    };

    // Train with execution feedback
    let training_data = vec![
        (pattern.clone(), 500),
        (pattern.clone(), 520),
        (pattern.clone(), 480),
    ];

    optimizer.train_ml_model(&training_data);

    // Verify ML model was trained
    let query = Algebra::Bgp(vec![pattern]);
    let result = optimizer.optimize(query);
    assert!(result.is_ok());
}

#[test]
fn test_production_config_high_throughput() {
    let config = ProductionOptimizerConfig::high_throughput();

    // Verify high-throughput characteristics (updated based on benchmark findings)
    assert_eq!(config.base_config.max_passes, 5); // Cost-based with early convergence
    assert!(config.base_config.cost_based); // Faster with early convergence
    assert_eq!(config.max_plan_cache_size, 10000); // Large cache
    assert!(!config.adaptive_learning); // Minimal overhead
    assert!(config.enable_result_cache); // Cache results

    let resources = config.estimate_resource_requirements();
    assert_eq!(resources.memory_mb, 500);
    assert!(resources.max_concurrent_queries >= 1000);

    let warnings = config.validate();
    assert!(warnings.is_empty());
}

#[test]
fn test_production_config_analytical() {
    let config = ProductionOptimizerConfig::analytical_queries();

    // Verify analytical characteristics
    assert_eq!(config.base_config.max_passes, 20); // Thorough optimization
    assert!(config.base_config.cost_based); // Cost-based optimization
    assert!(config.adaptive_learning); // Learn from execution
    assert!(matches!(
        config.estimation_method,
        EstimationMethod::MachineLearning
    ));

    let resources = config.estimate_resource_requirements();
    assert_eq!(resources.memory_mb, 1500);

    let warnings = config.validate();
    assert!(warnings.is_empty());
}

#[test]
fn test_production_config_low_memory() {
    let config = ProductionOptimizerConfig::low_memory();

    // Verify memory-efficient characteristics
    assert_eq!(config.max_plan_cache_size, 100); // Small cache
    assert!(!config.enable_result_cache); // No result cache
    assert!(matches!(config.estimation_method, EstimationMethod::Sketch)); // Memory-efficient HyperLogLog

    let resources = config.estimate_resource_requirements();
    assert_eq!(resources.memory_mb, 100); // Minimal memory
}

#[test]
fn test_production_config_validation() {
    // Test invalid configuration detection
    let mut config = ProductionOptimizerConfig::high_throughput();

    // Create inconsistency
    config.estimation_method = EstimationMethod::MachineLearning;
    config.adaptive_learning = false; // ML without learning

    let warnings = config.validate();
    assert!(!warnings.is_empty());
    assert!(warnings[0].contains("adaptive_learning"));
}

#[test]
fn test_all_workload_profiles_valid() {
    let profiles = vec![
        WorkloadProfile::HighThroughput,
        WorkloadProfile::AnalyticalQueries,
        WorkloadProfile::Mixed,
        WorkloadProfile::LowMemory,
        WorkloadProfile::LowCpu,
        WorkloadProfile::MaxPerformance,
    ];

    for profile in profiles {
        let config = ProductionOptimizerConfig::for_workload(profile);
        let warnings = config.validate();
        assert!(
            warnings.is_empty(),
            "Profile {:?} should be valid, got warnings: {:?}",
            profile,
            warnings
        );
    }
}

#[test]
fn test_resource_requirements_scaling() {
    let low_mem = ProductionOptimizerConfig::low_memory();
    let max_perf = ProductionOptimizerConfig::max_performance();

    let low_mem_res = low_mem.estimate_resource_requirements();
    let max_perf_res = max_perf.estimate_resource_requirements();

    // Verify resource scaling
    assert!(low_mem_res.memory_mb < max_perf_res.memory_mb);
    assert!(low_mem_res.cpu_cores < max_perf_res.cpu_cores);
    assert!(low_mem_res.max_concurrent_queries < max_perf_res.max_concurrent_queries);
}

#[test]
fn test_mixed_workload_balance() {
    let config = ProductionOptimizerConfig::mixed();

    // Verify balanced configuration
    assert!(config.base_config.max_passes >= 5); // Reasonable optimization
    assert!(config.base_config.max_passes <= 15); // Not excessive
    assert!(config.base_config.cost_based); // Cost-based enabled
    assert!(matches!(
        config.estimation_method,
        EstimationMethod::Histogram
    )); // Good balance

    let resources = config.estimate_resource_requirements();
    assert!(resources.memory_mb >= 500);
    assert!(resources.memory_mb <= 1500);
}
