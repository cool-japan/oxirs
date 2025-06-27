//! Tests for optimization module

use super::*;

#[test]
fn test_optimization_engine_creation() {
    let engine = OptimizationEngine::new();
    assert!(engine.get_config().enable_shape_optimization);
    assert!(engine.get_config().enable_strategy_optimization);
}

#[test]
fn test_optimization_config_default() {
    let config = OptimizationConfig::default();
    assert!(config.enable_shape_optimization);
    assert!(config.enable_strategy_optimization);
    assert!(config.enable_performance_optimization);
    assert!(config.enable_parallel_optimization);
}

#[test]
fn test_optimized_validation_strategy_creation() {
    let strategy = OptimizedValidationStrategy::new();
    assert!(strategy.graph_analysis.is_none());
    assert!(strategy.shape_execution_order.is_empty());
    assert!(strategy.parallel_execution.is_none());
}

#[test]
fn test_performance_improvements_creation() {
    let improvements = PerformanceImprovements::new();
    assert_eq!(improvements.execution_time_improvement, 0.0);
    assert_eq!(improvements.memory_usage_reduction, 0.0);
    assert_eq!(improvements.throughput_increase, 0.0);
    assert_eq!(improvements.latency_reduction, 0.0);
}

#[test]
fn test_recommendation_priority_ordering() {
    use RecommendationPriority::*;

    assert!(Critical > High);
    assert!(High > Medium);
    assert!(Medium > Low);
}

#[test]
fn test_bottleneck_severity_levels() {
    use BottleneckSeverity::*;

    assert_eq!(Low, Low);
    assert_ne!(Low, Critical);
    assert_ne!(Medium, High);
}
