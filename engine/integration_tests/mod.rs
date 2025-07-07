//! OxiRS Engine Integration Testing Framework
//!
//! This module provides comprehensive integration testing across all OxiRS engine modules,
//! including cross-module interactions, performance benchmarking, and system health validation.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

// Re-export main types
pub use self::{
    config::*,
    suite::*,
    report::*,
    tests::*,
};

mod config;
mod suite;
mod report;
mod tests;

/// Integration test configuration
#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    /// Enable parallel test execution
    pub parallel_execution: bool,
    /// Timeout for individual tests
    pub test_timeout: Duration,
    /// Memory usage threshold in MB
    pub memory_threshold_mb: u64,
    /// Performance thresholds for different operations
    pub performance_thresholds: PerformanceThresholds,
    /// Enable verbose logging
    pub verbose_logging: bool,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            parallel_execution: true,
            test_timeout: Duration::from_secs(30),
            memory_threshold_mb: 512,
            performance_thresholds: PerformanceThresholds::default(),
            verbose_logging: false,
        }
    }
}

/// Performance thresholds for different engine operations
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum SPARQL query execution time in milliseconds
    pub sparql_query_max_ms: u64,
    /// Maximum SHACL validation time in milliseconds
    pub shacl_validation_max_ms: u64,
    /// Maximum vector search time in milliseconds
    pub vector_search_max_ms: u64,
    /// Maximum rule inference time in milliseconds
    pub rule_inference_max_ms: u64,
    /// Maximum RDF-star operation time in milliseconds
    pub rdf_star_max_ms: u64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            sparql_query_max_ms: 500,
            shacl_validation_max_ms: 1000,
            vector_search_max_ms: 50,
            rule_inference_max_ms: 250,
            rdf_star_max_ms: 100,
        }
    }
}

/// Main integration test suite
#[derive(Debug)]
pub struct OxirsIntegrationTestSuite {
    config: IntegrationTestConfig,
    test_modules: Vec<Box<dyn TestModule>>,
    performance_monitor: PerformanceMonitor,
}

impl OxirsIntegrationTestSuite {
    /// Create a new test suite with the given configuration
    pub fn with_config(config: IntegrationTestConfig) -> Self {
        let mut suite = Self {
            config: config.clone(),
            test_modules: Vec::new(),
            performance_monitor: PerformanceMonitor::new(config.clone()),
        };

        // Register all test modules
        suite.register_test_modules();
        suite
    }

    /// Register all available test modules
    fn register_test_modules(&mut self) {
        // Core engine modules
        self.test_modules.push(Box::new(SparqlIntegrationTests::new()));
        self.test_modules.push(Box::new(ShaclIntegrationTests::new()));
        self.test_modules.push(Box::new(VectorIntegrationTests::new()));
        self.test_modules.push(Box::new(RuleIntegrationTests::new()));
        self.test_modules.push(Box::new(StarIntegrationTests::new()));

        // Cross-module integration tests
        self.test_modules.push(Box::new(NeuralSymbolicIntegrationTests::new()));
        self.test_modules.push(Box::new(PerformanceIntegrationTests::new()));
        self.test_modules.push(Box::new(SystemHealthTests::new()));
    }

    /// Run all integration tests
    pub fn run_all_tests(&mut self) -> Result<TestReport> {
        let start_time = Instant::now();
        let mut module_results = HashMap::new();
        let mut total_tests = 0;
        let mut passed_tests = 0;
        let mut failed_tests = 0;
        let mut all_recommendations = Vec::new();

        println!("🔧 Starting comprehensive integration tests...");

        // Run tests for each module
        for module in &mut self.test_modules {
            let module_name = module.name().to_string();
            if self.config.verbose_logging {
                println!("  📦 Testing module: {}", module_name);
            }

            let module_result = if self.config.parallel_execution {
                module.run_tests_parallel(&self.config)?
            } else {
                module.run_tests_sequential(&self.config)?
            };

            total_tests += module_result.total_tests;
            passed_tests += module_result.passed_tests;
            failed_tests += module_result.failed_tests;
            all_recommendations.extend(module_result.recommendations.clone());

            module_results.insert(module_name, module_result);
        }

        let total_execution_time = start_time.elapsed();
        let success_rate = if total_tests > 0 {
            (passed_tests as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        };

        // Analyze integration coverage
        let integration_analysis = self.analyze_integration_coverage(&module_results)?;

        // Calculate overall performance score
        let performance_score = self.calculate_performance_score(&module_results);

        let summary = TestSummary {
            total_tests,
            passed_tests,
            failed_tests,
            success_rate,
            total_execution_time,
            performance_score,
        };

        // Generate system recommendations
        let mut recommendations = all_recommendations;
        recommendations.extend(self.generate_system_recommendations(&summary, &integration_analysis));

        Ok(TestReport {
            summary,
            module_results,
            integration_analysis,
            recommendations,
        })
    }

    /// Analyze integration coverage across modules
    fn analyze_integration_coverage(&self, results: &HashMap<String, ModuleTestResult>) -> Result<IntegrationAnalysis> {
        let tested_points = vec![
            "SPARQL-Vector Integration".to_string(),
            "SHACL-Rule Integration".to_string(),
            "Vector-Neural Integration".to_string(),
            "RDF-Star Support".to_string(),
            "Cross-Module Performance".to_string(),
        ];

        let missing_points = vec![
            // Add any missing integration points discovered during testing
        ];

        let coverage_percentage = if !tested_points.is_empty() {
            ((tested_points.len() - missing_points.len()) as f64 / tested_points.len() as f64) * 100.0
        } else {
            0.0
        };

        Ok(IntegrationAnalysis {
            coverage_percentage,
            integration_points_tested: tested_points,
            missing_integration_points: missing_points,
        })
    }

    /// Calculate overall performance score
    fn calculate_performance_score(&self, results: &HashMap<String, ModuleTestResult>) -> f64 {
        let scores: Vec<f64> = results.values()
            .map(|r| r.average_performance_score)
            .collect();

        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        }
    }

    /// Generate system-level recommendations
    fn generate_system_recommendations(&self, summary: &TestSummary, analysis: &IntegrationAnalysis) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if summary.performance_score < 80.0 {
            recommendations.push(Recommendation {
                priority: RecommendationPriority::High,
                category: "Performance".to_string(),
                description: "Overall system performance below target. Consider optimization.".to_string(),
            });
        }

        // Coverage recommendations
        if analysis.coverage_percentage < 90.0 {
            recommendations.push(Recommendation {
                priority: RecommendationPriority::Medium,
                category: "Coverage".to_string(),
                description: "Integration test coverage could be improved.".to_string(),
            });
        }

        // Success rate recommendations
        if summary.success_rate < 95.0 {
            recommendations.push(Recommendation {
                priority: RecommendationPriority::Critical,
                category: "Reliability".to_string(),
                description: "Test success rate below target. Critical issues need attention.".to_string(),
            });
        }

        recommendations
    }
}

/// Performance monitoring for integration tests
#[derive(Debug)]
pub struct PerformanceMonitor {
    config: IntegrationTestConfig,
    metrics: HashMap<String, PerformanceMetric>,
}

impl PerformanceMonitor {
    fn new(config: IntegrationTestConfig) -> Self {
        Self {
            config,
            metrics: HashMap::new(),
        }
    }

    pub fn record_operation(&mut self, operation: &str, duration: Duration, success: bool) {
        let metric = self.metrics.entry(operation.to_string())
            .or_insert_with(|| PerformanceMetric::new(operation));

        metric.record_execution(duration, success);
    }

    pub fn get_metrics(&self) -> &HashMap<String, PerformanceMetric> {
        &self.metrics
    }
}

/// Performance metric tracking
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub operation_name: String,
    pub total_executions: u64,
    pub successful_executions: u64,
    pub total_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub avg_duration: Duration,
}

impl PerformanceMetric {
    fn new(operation_name: &str) -> Self {
        Self {
            operation_name: operation_name.to_string(),
            total_executions: 0,
            successful_executions: 0,
            total_duration: Duration::from_nanos(0),
            min_duration: Duration::from_secs(u64::MAX),
            max_duration: Duration::from_nanos(0),
            avg_duration: Duration::from_nanos(0),
        }
    }

    fn record_execution(&mut self, duration: Duration, success: bool) {
        self.total_executions += 1;
        if success {
            self.successful_executions += 1;
        }

        self.total_duration += duration;
        if duration < self.min_duration {
            self.min_duration = duration;
        }
        if duration > self.max_duration {
            self.max_duration = duration;
        }

        self.avg_duration = self.total_duration / self.total_executions as u32;
    }
}

/// Trait for test modules
pub trait TestModule: Send + Sync {
    fn name(&self) -> &str;
    fn run_tests_sequential(&mut self, config: &IntegrationTestConfig) -> Result<ModuleTestResult>;
    fn run_tests_parallel(&mut self, config: &IntegrationTestConfig) -> Result<ModuleTestResult>;
}