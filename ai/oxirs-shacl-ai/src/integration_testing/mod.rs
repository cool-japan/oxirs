//! Enhanced Integration Testing Framework
//!
//! This module provides comprehensive integration testing capabilities for the SHACL-AI
//! system, including end-to-end validation testing, performance validation, cross-module
//! integration testing, and automated test scenario generation.

pub mod config;
pub mod dependencies;
pub mod framework;
pub mod monitoring;
pub mod results;
pub mod runner;
pub mod scenario;
pub mod types;

// Re-export main types and functionality
pub use config::{IntegrationTestConfig, TestComplexityLevel};
pub use dependencies::DependencyAnalyzer;
pub use framework::IntegrationTestFramework;
pub use monitoring::{MemoryMonitor, PerformanceProfiler};
pub use results::TestResultCollector;
pub use runner::TestRunner;
pub use scenario::TestScenarioGenerator;
pub use types::{
    DataConfiguration, DependencyAnalysisResult, ErrorDetails, ExecutionMetadata,
    ImplementationEffort, IntegrationTestReport, LatencyPercentiles, PerformanceTestMetrics,
    QualityMetrics, QualityThresholds, RecommendationPriority, RecommendationType,
    ResourceUtilization, ScalabilityMetrics, StrategyPerformanceMetrics, TestCoverageMetrics,
    TestRecommendation, TestResult, TestScenario, TestStatus, TestSummary, TestType,
    ValidationTestResults,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_test_framework_creation() {
        let config = IntegrationTestConfig::default();
        let framework = IntegrationTestFramework::new(config);
        assert!(!framework.config.test_data_sizes.is_empty());
    }

    #[test]
    fn test_test_runner_creation() {
        let runner = TestRunner::new(4);
        assert_eq!(runner.worker_pool.len(), 4);
        assert!(runner.test_queue.is_empty());
        assert!(runner.active_tests.is_empty());
    }

    #[test]
    fn test_scenario_generator_creation() {
        let generator = TestScenarioGenerator::new();
        assert!(generator.scenario_templates.is_empty());
    }

    #[test]
    fn test_performance_profiler_creation() {
        let profiler = PerformanceProfiler::new();
        assert!(!profiler.is_profiling);
        assert!(profiler.profiling_sessions.is_empty());
    }

    #[test]
    fn test_memory_monitor_creation() {
        let monitor = MemoryMonitor::new(true);
        assert!(monitor.enabled);
        assert!(!monitor.is_monitoring);
    }

    #[test]
    fn test_dependency_analyzer_creation() {
        let analyzer = DependencyAnalyzer::new();
        assert!(analyzer.circular_dependencies.is_empty());
    }

    #[test]
    fn test_result_collector_creation() {
        let collector = TestResultCollector::new();
        assert!(collector.collected_results.is_empty());
        assert!(collector.test_suites.is_empty());
    }

    #[test]
    fn test_integration_test_config_default() {
        let config = IntegrationTestConfig::default();
        assert_eq!(config.test_timeout_seconds, 30);
        assert_eq!(config.parallel_workers, 4);
        assert!(config.enable_performance_profiling);
        assert!(config.enable_memory_monitoring);
        assert!(config.enable_dependency_testing);
        assert!(config.enable_scenario_generation);
        assert_eq!(config.test_data_sizes, vec![100, 1000, 10000]);
        assert_eq!(config.min_success_rate_threshold, 0.95);
        assert_eq!(config.max_memory_usage_mb, 1024.0);
        assert_eq!(config.max_execution_time_ms, 5000);
        assert!(config.enable_regression_testing);
        assert!(config.persist_test_results);
        assert!(config.generate_detailed_reports);
    }

    #[tokio::test]
    async fn test_scenario_generation() {
        let generator = TestScenarioGenerator::new();
        let result = generator
            .generate_scenario(
                TestType::EndToEndValidation,
                TestComplexityLevel::Simple,
                1000,
            )
            .await;
        assert!(result.is_ok());
        let scenario = result.unwrap();
        assert_eq!(scenario.test_type, TestType::EndToEndValidation);
        assert_eq!(scenario.complexity_level, TestComplexityLevel::Simple);
    }

    #[test]
    fn test_quality_metrics_creation() {
        let metrics = QualityMetrics::default();
        assert_eq!(metrics.precision, 0.0);
        assert_eq!(metrics.recall, 0.0);
        assert_eq!(metrics.f1_score, 0.0);
        assert_eq!(metrics.accuracy, 0.0);
    }

    #[test]
    fn test_new_strategy_performance_metrics() {
        let metrics = StrategyPerformanceMetrics::default();
        assert_eq!(metrics.strategy_name, "default");
        assert_eq!(metrics.memory_efficiency, 0.0);
        assert_eq!(metrics.cache_hit_rate, 0.0);
        assert_eq!(metrics.scalability_factor, 1.0);
    }

    #[test]
    fn test_test_recommendation() {
        let recommendation = TestRecommendation {
            recommendation_type: RecommendationType::Performance,
            priority: RecommendationPriority::High,
            title: "Test Recommendation".to_string(),
            description: "This is a test recommendation".to_string(),
            implementation_effort: ImplementationEffort::Medium,
            expected_impact: 0.8,
        };
        assert_eq!(
            recommendation.recommendation_type,
            RecommendationType::Performance
        );
        assert_eq!(recommendation.priority, RecommendationPriority::High);
        assert_eq!(recommendation.expected_impact, 0.8);
    }

    #[test]
    fn test_micro_benchmark_suite_creation() {
        // Test that we can create basic testing structures
        let config = IntegrationTestConfig::default();
        assert!(config
            .test_complexity_levels
            .contains(&TestComplexityLevel::Simple));
        assert!(config
            .test_complexity_levels
            .contains(&TestComplexityLevel::Medium));
        assert!(config
            .test_complexity_levels
            .contains(&TestComplexityLevel::Complex));
    }

    #[test]
    fn test_test_scenario_generation() {
        use std::collections::HashSet;
        use std::time::Duration;
        use uuid::Uuid;

        let scenario = TestScenario {
            scenario_id: Uuid::new_v4(),
            name: "test_scenario".to_string(),
            description: "Test scenario description".to_string(),
            test_type: TestType::EndToEndValidation,
            complexity_level: TestComplexityLevel::Simple,
            data_configuration: config::DataConfiguration {
                entity_count: 100,
                distribution: config::DataDistribution::Uniform,
                random_seed: Some(42),
                include_edge_cases: false,
            },
            validation_configuration: config::ValidationConfiguration {
                strict_mode: false,
                parallel_validation: false,
                max_validation_depth: 5,
                enable_shape_inference: false,
            },
            expected_outcomes: types::ExpectedOutcomes {
                should_succeed: true,
                expected_violation_count: Some(0),
                expected_execution_time_range: None,
                expected_memory_usage_range: None,
                expected_quality_metrics: None,
                expected_error_types: vec![],
            },
            timeout: Duration::from_secs(30),
            tags: HashSet::new(),
        };

        assert_eq!(scenario.test_type, TestType::EndToEndValidation);
        assert_eq!(scenario.complexity_level, TestComplexityLevel::Simple);
        assert_eq!(scenario.data_configuration.entity_count, 100);
    }

    #[test]
    fn test_quality_metrics() {
        let mut metrics = QualityMetrics::default();
        metrics.precision = 0.95;
        metrics.recall = 0.90;
        metrics.f1_score = 0.925;
        metrics.accuracy = 0.93;

        assert_eq!(metrics.precision, 0.95);
        assert_eq!(metrics.recall, 0.90);
        assert!(metrics.f1_score > 0.92);
        assert!(metrics.accuracy > 0.92);
    }
}
