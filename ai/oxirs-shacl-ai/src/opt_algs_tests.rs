//! Tests for optimization algorithm implementations.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::opt_algs_evolutionary::{OptimizationProblem, OptimizationState};
    use crate::opt_algs_swarm::{
        OptimizationObjectiveFunction, OptimizationPoint, OptimizationResult,
        OptimizationSearchSpace,
    };
    use crate::optimization_engine::PerformanceMetrics;
    use crate::optimization_engine::{
        AdaptiveOptimizer, AdvancedOptimizationEngine, AntColonyOptimizer,
        DifferentialEvolutionOptimizer, ReinforcementLearningOptimizer,
    };
    use crate::shape::{PropertyConstraint, Shape};
    use crate::{Result, ShaclAiError};

    #[tokio::test]
    async fn test_optimization_engine_creation() {
        let engine = AdvancedOptimizationEngine::new();
        assert!(engine.config.enable_parallel_validation);
        assert!(engine.config.enable_constraint_caching);
        assert!(engine.config.enable_constraint_ordering);
    }

    #[tokio::test]
    async fn test_shape_optimization() {
        let mut engine = AdvancedOptimizationEngine::new();

        let mut shape = Shape::new("http://example.org/TestShape".to_string());
        shape.add_property_constraint(
            PropertyConstraint::new("http://example.org/prop1".to_string())
                .with_pattern(".*test.*".to_string()),
        );
        shape.add_property_constraint(
            PropertyConstraint::new("http://example.org/prop2".to_string())
                .with_datatype("xsd:string".to_string()),
        );

        let result = engine.optimize_shape(&shape).await;
        assert!(result.is_ok());

        let optimized = result.expect("should succeed");
        assert!(optimized.improvement_percentage >= 0.0);
        assert!(!optimized.applied_optimizations.is_empty());
    }

    #[tokio::test]
    async fn test_parallel_validation_config() {
        let mut engine = AdvancedOptimizationEngine::new();

        let mut shape = Shape::new("http://example.org/TestShape".to_string());
        for i in 0..5 {
            shape.add_property_constraint(
                PropertyConstraint::new(format!("http://example.org/prop{}", i))
                    .with_datatype("xsd:string".to_string()),
            );
        }

        let config = engine.enable_parallel_validation(&shape).await;
        assert!(config.is_ok());

        let parallel_config = config.expect("should succeed");
        assert!(parallel_config.enabled);
        assert!(parallel_config.max_parallel_constraints > 0);
        assert!(!parallel_config.constraint_groups.is_empty());
    }

    #[tokio::test]
    async fn test_cache_configuration() {
        let mut engine = AdvancedOptimizationEngine::new();

        let mut shape = Shape::new("http://example.org/TestShape".to_string());
        shape.add_property_constraint(
            PropertyConstraint::new("http://example.org/prop1".to_string())
                .with_pattern(".*expensive.*".to_string()),
        );
        shape.add_property_constraint(
            PropertyConstraint::new("http://example.org/prop2".to_string())
                .with_class("http://example.org/ExpensiveClass".to_string()),
        );

        let config = engine.configure_caching(&shape).await;
        assert!(config.is_ok());

        let cache_config = config.expect("should succeed");
        assert!(cache_config.enabled);
        assert!(!cache_config.cacheable_constraints.is_empty());
        assert!(cache_config.estimated_hit_rate > 0.0);
    }

    #[tokio::test]
    async fn test_constraint_complexity_calculation() {
        let engine = AdvancedOptimizationEngine::new();

        let pattern_constraint =
            PropertyConstraint::new("test_pattern".to_string()).with_pattern(".*".to_string());
        assert_eq!(
            engine.calculate_constraint_complexity(&pattern_constraint),
            3.5
        );

        let datatype_constraint = PropertyConstraint::new("test_datatype".to_string())
            .with_datatype("xsd:string".to_string());
        assert_eq!(
            engine.calculate_constraint_complexity(&datatype_constraint),
            0.8
        );
    }

    #[tokio::test]
    async fn test_parallelization_potential() {
        let engine = AdvancedOptimizationEngine::new();

        let single_constraint = vec![
            PropertyConstraint::new("test".to_string()).with_datatype("xsd:string".to_string())
        ];
        assert_eq!(
            engine.calculate_parallelization_potential(&single_constraint),
            0.0
        );

        let multiple_constraints = vec![
            PropertyConstraint::new("test1".to_string()).with_datatype("xsd:string".to_string()),
            PropertyConstraint::new("test2".to_string()).with_datatype("xsd:int".to_string()),
            PropertyConstraint::new("test3".to_string()).with_pattern(".*".to_string()),
        ];
        let potential = engine.calculate_parallelization_potential(&multiple_constraints);
        assert!(potential >= 0.0);
        assert!(potential <= 1.0);

        let many_constraints = vec![
            PropertyConstraint::new("test1".to_string()).with_datatype("xsd:string".to_string()),
            PropertyConstraint::new("test2".to_string()).with_datatype("xsd:int".to_string()),
            PropertyConstraint::new("test3".to_string()).with_pattern(".*".to_string()),
            PropertyConstraint::new("test4".to_string()).with_class("rdfs:Resource".to_string()),
            PropertyConstraint::new("test5".to_string()).with_node_kind("IRI".to_string()),
        ];
        let large_potential = engine.calculate_parallelization_potential(&many_constraints);
        assert!(large_potential >= 0.0);
        assert!(large_potential <= 1.0);
    }

    #[tokio::test]
    async fn test_ant_colony_optimizer() {
        let constraints = vec![
            PropertyConstraint::new("test1".to_string()).with_datatype("xsd:string".to_string()),
            PropertyConstraint::new("test2".to_string()).with_pattern(".*".to_string()),
            PropertyConstraint::new("test3".to_string()).with_class("rdfs:Resource".to_string()),
        ];

        let mut aco = AntColonyOptimizer::new(constraints.len());
        let result = aco.optimize_constraint_order(&constraints).await;
        assert!(result.is_ok());

        let optimized_order = result.expect("should succeed");
        assert_eq!(optimized_order.len(), constraints.len());
    }

    #[tokio::test]
    async fn test_differential_evolution() {
        let search_space = OptimizationSearchSpace {
            execution_time_range: (0.0, 100.0),
            memory_usage_range: (0.0, 1000.0),
            cache_efficiency_range: (0.0, 1.0),
        };

        let mut de = DifferentialEvolutionOptimizer::new();

        struct SimpleObjective;

        #[async_trait::async_trait]
        impl OptimizationObjectiveFunction for SimpleObjective {
            async fn evaluate(&self, point: &OptimizationPoint) -> Result<f64> {
                Ok(
                    100.0 - point.execution_time_weight - point.memory_usage_weight
                        + point.cache_efficiency_weight,
                )
            }
        }

        let objective = SimpleObjective;
        let result = de.optimize(&objective, &search_space).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_reinforcement_learning_optimizer() {
        let initial_state = OptimizationState {
            parallel_threads: 4,
            cache_size_mb: 100,
            constraint_order_entropy: 50,
        };

        let mut rl = ReinforcementLearningOptimizer::new();
        let result = rl.optimize(initial_state, 50).await;
        assert!(result.is_ok());

        let policy = result.expect("should succeed");
        assert!(policy.confidence >= 0.0);
        assert!(policy.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_adaptive_optimizer() {
        let problem = OptimizationProblem {
            constraints: vec![
                PropertyConstraint::new("test1".to_string())
                    .with_datatype("xsd:string".to_string()),
                PropertyConstraint::new("test2".to_string()).with_pattern(".*".to_string()),
            ],
            baseline_metrics: PerformanceMetrics {
                validation_time_ms: 100.0,
                memory_usage_mb: 50.0,
                cpu_usage_percent: 0.0,
                cache_hit_rate: 0.5,
                parallelization_factor: 1.0,
                constraint_execution_times: HashMap::new(),
            },
        };

        let mut adaptive = AdaptiveOptimizer::new();
        let result = adaptive.optimize(&problem).await;
        assert!(result.is_ok());

        let optimization_result = result.expect("should succeed");
        assert!(optimization_result.performance_improvement >= 0.0);
    }
}
