//! Tests for sophisticated validation optimization.

#![cfg(test)]

use crate::sophisticated_validation_optimization::*;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

#[test]
fn test_sophisticated_optimization_config_default() {
    let config = SophisticatedOptimizationConfig::default();
    assert!(config.enable_quantum_optimization);
    assert!(config.enable_neural_optimization);
    assert!(config.enable_multi_objective_optimization);
    assert_eq!(config.learning_rate, 0.001);
    assert_eq!(config.population_size, 100);
}

#[test]
fn test_sophisticated_validation_optimizer_creation() {
    let config = SophisticatedOptimizationConfig::default();
    let _optimizer = SophisticatedValidationOptimizer::new(config);
}

#[test]
fn test_optimization_objectives() {
    let objectives = [
        OptimizationObjective::MinimizeExecutionTime,
        OptimizationObjective::MaximizeAccuracy,
        OptimizationObjective::MinimizeMemoryUsage,
    ];

    assert_eq!(objectives.len(), 3);
    assert!(objectives.contains(&OptimizationObjective::MinimizeExecutionTime));
}

#[test]
fn test_optimization_solution_default() {
    let solution = OptimizationSolution::default();
    assert_eq!(solution.accuracy, 0.9);
    assert_eq!(solution.precision, 0.9);
    assert_eq!(solution.overall_score, 0.8);
}

#[test]
fn test_optimization_results() {
    let mut results = OptimizationResults::new();
    assert!(results.solutions.is_empty());
    assert_eq!(results.convergence_metric, 0.0);

    let other_results = OptimizationResults::new();
    results.merge(other_results);
    assert_eq!(results.convergence_metric, 0.0);
}

#[test]
fn test_cache_entry() {
    let result = OptimizationResult {
        optimization_id: Uuid::new_v4(),
        optimization_strategy: "test".to_string(),
        optimization_objectives: vec![],
        achieved_metrics: OptimizationMetrics {
            execution_time_ms: 100.0,
            memory_usage_mb: 50.0,
            cpu_usage_percent: 30.0,
            io_operations_count: 1000,
            accuracy: 0.9,
            precision: 0.85,
            recall: 0.8,
            f1_score: 0.82,
            throughput_ops_per_sec: 500.0,
            false_positive_rate: 0.05,
            false_negative_rate: 0.1,
            parallel_efficiency: 0.75,
            energy_consumption_joules: 25.0,
            overall_efficiency_score: 0.8,
        },
        execution_time: Duration::from_secs(1),
        convergence_achieved: true,
        pareto_solutions: vec![],
        optimization_path: vec![],
        recommendations: vec![],
        confidence_score: 0.85,
    };

    let cache_entry = CacheEntry::new(result, SystemTime::now());
    assert!(cache_entry.is_valid());
    assert_eq!(cache_entry.access_count, 0);
}

#[tokio::test]
async fn test_optimization_strategies() {
    let quantum_strategy = QuantumOptimizationStrategy::new();
    assert_eq!(quantum_strategy.name(), "quantum_optimization");

    let neural_strategy = NeuralOptimizationStrategy::new();
    assert_eq!(neural_strategy.name(), "neural_optimization");

    let hybrid_strategy = HybridOptimizationStrategy::new();
    assert_eq!(hybrid_strategy.name(), "hybrid_optimization");
}
