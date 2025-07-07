//! Advanced Validation Strategies
//!
//! This module provides sophisticated validation strategies that go beyond traditional
//! SHACL validation, including context-aware validation, adaptive constraint selection,
//! multi-objective optimization, and dynamic strategy selection.

pub mod config;
pub mod core;
pub mod manager;
pub mod strategies;
pub mod advanced_strategies;
pub mod types;

// Re-export main types and traits
pub use config::*;
pub use core::*;
pub use manager::*;
pub use strategies::*;
pub use advanced_strategies::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_validation_config_default() {
        let config = AdvancedValidationConfig::default();
        assert_eq!(
            config.strategy_selection,
            StrategySelectionApproach::AdaptiveMLBased
        );
        assert_eq!(config.context_awareness_level, ContextAwarenessLevel::High);
        assert!(config.enable_multi_objective_optimization);
        assert_eq!(config.max_concurrent_strategies, 4);
    }

    #[test]
    fn test_strategy_manager_creation() {
        let config = AdvancedValidationConfig::default();
        let manager = AdvancedValidationStrategyManager::new(config);
        assert_eq!(manager.strategies.len(), 8); // Should have 8 default strategies including new ones
    }

    #[test]
    fn test_strategy_capabilities() {
        let strategy = OptimizedSequentialStrategy::new();
        let capabilities = strategy.capabilities();
        assert!(!capabilities.supports_temporal_validation);
        assert!(capabilities.supports_semantic_enrichment);
        assert_eq!(
            capabilities.computational_complexity,
            ComputationalComplexity::Linear
        );
    }

    #[test]
    fn test_quality_metrics_creation() {
        let metrics = QualityMetrics {
            precision: 0.92,
            recall: 0.88,
            f1_score: 0.90,
            accuracy: 0.90,
            specificity: 0.85,
            false_positive_rate: 0.08,
            false_negative_rate: 0.12,
            Matthews_correlation_coefficient: 0.75,
            area_under_roc_curve: 0.89,
        };

        assert_eq!(metrics.precision, 0.92);
        assert_eq!(metrics.f1_score, 0.90);
    }

    #[test]
    fn test_uncertainty_metrics() {
        let uncertainty = UncertaintyMetrics {
            epistemic_uncertainty: 0.1,
            aleatoric_uncertainty: 0.05,
            total_uncertainty: 0.15,
            confidence_interval: ConfidenceInterval {
                lower_bound: 0.8,
                upper_bound: 0.95,
                confidence_level: 0.95,
            },
            uncertainty_sources: vec![],
        };

        assert_eq!(uncertainty.total_uncertainty, 0.15);
        assert_eq!(uncertainty.confidence_interval.confidence_level, 0.95);
    }

    #[tokio::test]
    async fn test_strategy_validation() {
        let strategy = OptimizedSequentialStrategy::new();
        // Test basic strategy properties
        assert_eq!(strategy.name(), "OptimizedSequential");
        assert!(!strategy.description().is_empty());
        // TODO: Implement proper test Store for full validation testing
    }

    #[test]
    fn test_quantum_enhanced_strategy() {
        let strategy = QuantumEnhancedStrategy::new();
        assert_eq!(strategy.name(), "QuantumEnhanced");
        assert!(!strategy.description().is_empty());
        assert!(strategy.parameters().contains_key("quantum_coherence_threshold"));
        assert!(strategy.parameters().contains_key("entanglement_strength"));
        
        let capabilities = strategy.capabilities();
        assert!(capabilities.supports_temporal_validation);
        assert!(capabilities.supports_parallel_processing);
        assert_eq!(capabilities.computational_complexity, ComputationalComplexity::Logarithmic);
    }

    #[test]
    fn test_neuromorphic_strategy() {
        let strategy = NeuromorphicValidationStrategy::new();
        assert_eq!(strategy.name(), "NeuromorphicValidation");
        assert!(!strategy.description().is_empty());
        assert!(strategy.parameters().contains_key("spike_threshold"));
        assert!(strategy.parameters().contains_key("plasticity_strength"));
        
        let capabilities = strategy.capabilities();
        assert!(capabilities.supports_temporal_validation);
        assert!(capabilities.supports_uncertainty_quantification);
    }

    #[test]
    fn test_bayesian_uncertainty_strategy() {
        let strategy = BayesianUncertaintyStrategy::new();
        assert_eq!(strategy.name(), "BayesianUncertainty");
        assert!(!strategy.description().is_empty());
        assert!(strategy.parameters().contains_key("prior_strength"));
        assert!(strategy.parameters().contains_key("mcmc_iterations"));
        
        let capabilities = strategy.capabilities();
        assert!(capabilities.supports_uncertainty_quantification);
        assert_eq!(capabilities.computational_complexity, ComputationalComplexity::LogLinear);
    }

    #[test]
    fn test_real_time_adaptive_strategy() {
        let strategy = RealTimeAdaptiveStrategy::new();
        assert_eq!(strategy.name(), "RealTimeAdaptive");
        assert!(!strategy.description().is_empty());
        assert!(strategy.parameters().contains_key("adaptation_rate"));
        assert!(strategy.parameters().contains_key("forgetting_factor"));
        
        let capabilities = strategy.capabilities();
        assert!(capabilities.supports_temporal_validation);
        assert!(capabilities.supports_incremental_validation);
    }

    #[test]
    fn test_new_strategy_performance_metrics() {
        // Test that new strategies have reasonable performance metrics
        let quantum = QuantumEnhancedStrategy::new();
        let neuromorphic = NeuromorphicValidationStrategy::new();
        let bayesian = BayesianUncertaintyStrategy::new();
        let adaptive = RealTimeAdaptiveStrategy::new();

        // All strategies should have high confidence scores for their optimal contexts
        assert!(quantum.confidence_for_context(&create_test_context(15000)) > 0.9);
        assert!(neuromorphic.confidence_for_context(&create_test_temporal_context()) > 0.9);
        assert!(bayesian.confidence_for_context(&create_test_explainable_context()) > 0.9);
        assert!(adaptive.confidence_for_context(&create_test_temporal_context()) > 0.9);
    }

    // Helper functions for test contexts
    fn create_test_context(triple_count: usize) -> ValidationContext {
        use std::collections::HashMap;
        use std::time::{Duration, SystemTime};

        ValidationContext {
            data_characteristics: DataCharacteristics {
                total_triples: triple_count,
                unique_subjects: triple_count / 2,
                unique_predicates: 100,
                unique_objects: triple_count * 3 / 4,
                average_degree: 2.5,
                graph_density: 0.001,
                has_temporal_data: false,
                has_spatial_data: false,
                data_quality_score: 0.85,
                schema_complexity: 0.6,
            },
            shape_characteristics: ShapeCharacteristics {
                total_shapes: 10,
                average_constraints_per_shape: 3.5,
                max_constraint_depth: 5,
                has_recursive_shapes: false,
                complexity_distribution: HashMap::new(),
                dependency_graph_complexity: 0.4,
            },
            domain_context: DomainContext {
                domain_type: DomainType::Generic,
                domain_specific_rules: Vec::new(),
                semantic_relationships: HashMap::new(),
                business_rules: Vec::new(),
            },
            performance_requirements: PerformanceRequirements {
                max_validation_time: Duration::from_secs(30),
                max_memory_usage_mb: 1024.0,
                min_throughput_per_second: 100.0,
                priority_level: PriorityLevel::Normal,
            },
            quality_requirements: QualityRequirements {
                min_precision: 0.85,
                min_recall: 0.80,
                min_f1_score: 0.82,
                max_false_positive_rate: 0.10,
                max_false_negative_rate: 0.15,
                require_explainability: false,
            },
            temporal_context: TemporalContext {
                validation_timestamp: SystemTime::now(),
                data_freshness: Duration::from_secs(3600),
                temporal_validation_window: None,
                historical_performance: Vec::new(),
            },
        }
    }

    fn create_test_temporal_context() -> ValidationContext {
        let mut context = create_test_context(5000);
        context.data_characteristics.has_temporal_data = true;
        context
    }

    fn create_test_explainable_context() -> ValidationContext {
        let mut context = create_test_context(1000);
        context.quality_requirements.require_explainability = true;
        context
    }
}