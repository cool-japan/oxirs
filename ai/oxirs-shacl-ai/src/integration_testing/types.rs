//! Core types for integration testing
//!
//! This module contains all data structures and enums used throughout the integration testing framework.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

pub use super::config::{
    DataConfiguration, DataDistribution, QualityRequirements, QualityThresholds,
    TestComplexityLevel, ValidationConfiguration, ValidationParameters,
};

/// Test scenario definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    pub scenario_id: Uuid,
    pub name: String,
    pub description: String,
    pub test_type: TestType,
    pub complexity_level: TestComplexityLevel,
    pub data_configuration: DataConfiguration,
    pub validation_configuration: ValidationConfiguration,
    pub expected_outcomes: ExpectedOutcomes,
    pub timeout: Duration,
    pub tags: HashSet<String>,
}

/// Types of integration tests
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TestType {
    EndToEndValidation,
    PerformanceValidation,
    MemoryValidation,
    ConcurrencyValidation,
    ScalabilityValidation,
    RegressionValidation,
    CrossModuleIntegration,
    DataQualityValidation,
    StrategyComparison,
    ErrorHandlingValidation,
}

/// Expected outcomes for test scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcomes {
    pub should_succeed: bool,
    pub expected_violation_count: Option<usize>,
    pub expected_execution_time_range: Option<(Duration, Duration)>,
    pub expected_memory_usage_range: Option<(f64, f64)>,
    pub expected_quality_metrics: Option<QualityThresholds>,
    pub expected_error_types: Vec<String>,
}

/// Running test instance
#[derive(Debug)]
pub struct RunningTest {
    pub test_id: Uuid,
    pub scenario: TestScenario,
    pub start_time: Instant,
    pub worker_id: usize,
    pub status: TestStatus,
    pub progress: f64,
}

/// Test execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Timeout,
    Cancelled,
}

/// Test worker for parallel execution
#[derive(Debug)]
pub struct TestWorker {
    pub worker_id: usize,
    pub is_busy: bool,
    pub current_test: Option<Uuid>,
    pub completed_tests: usize,
    pub total_execution_time: Duration,
}

/// Test execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionStats {
    pub total_tests_run: usize,
    pub successful_tests: usize,
    pub failed_tests: usize,
    pub timed_out_tests: usize,
    pub average_execution_time: Duration,
    pub total_execution_time: Duration,
    pub memory_usage_stats: MemoryUsageStats,
    pub performance_stats: PerformanceStats,
}

/// Test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_id: Uuid,
    pub scenario_name: String,
    pub test_type: TestType,
    pub status: TestStatus,
    pub execution_time: Duration,
    pub memory_usage_mb: f64,
    pub validation_results: ValidationTestResults,
    pub performance_metrics: PerformanceTestMetrics,
    pub error_details: Option<ErrorDetails>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<TestRecommendation>,
    pub timestamp: SystemTime,
}

/// Validation test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationTestResults {
    pub validation_successful: bool,
    pub violation_count: usize,
    pub constraint_results: HashMap<String, ConstraintTestResult>,
    pub quality_metrics: QualityMetrics,
    pub strategy_performance: StrategyPerformanceMetrics,
}

/// Quality metrics for validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub accuracy: f64,
    pub specificity: f64,
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            accuracy: 0.0,
            specificity: 0.0,
            false_positive_rate: 0.0,
            false_negative_rate: 0.0,
        }
    }
}

/// Constraint test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintTestResult {
    pub constraint_id: String,
    pub validation_successful: bool,
    pub execution_time: Duration,
    pub memory_usage_mb: f64,
    pub violation_count: usize,
    pub confidence_score: f64,
}

/// Strategy performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPerformanceMetrics {
    pub strategy_name: String,
    pub total_execution_time: Duration,
    pub average_constraint_time: Duration,
    pub memory_efficiency: f64,
    pub cache_hit_rate: f64,
    pub parallelization_efficiency: f64,
    pub quality_impact: f64,
    pub optimization_level: f64,
    pub resource_utilization: f64,
    pub scalability_factor: f64,
    pub error_recovery_rate: f64,
    pub adaptability_score: f64,
}

impl Default for StrategyPerformanceMetrics {
    fn default() -> Self {
        Self {
            strategy_name: String::from("default"),
            total_execution_time: Duration::from_millis(0),
            average_constraint_time: Duration::from_millis(0),
            memory_efficiency: 0.0,
            cache_hit_rate: 0.0,
            parallelization_efficiency: 0.0,
            quality_impact: 0.0,
            optimization_level: 0.0,
            resource_utilization: 0.0,
            scalability_factor: 1.0,
            error_recovery_rate: 0.0,
            adaptability_score: 0.0,
        }
    }
}

/// Performance test metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTestMetrics {
    pub latency_percentiles: LatencyPercentiles,
    pub resource_utilization: ResourceUtilization,
    pub scalability_metrics: ScalabilityMetrics,
}

/// Latency percentiles for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub disk_io_mb_per_sec: f64,
    pub network_io_mb_per_sec: f64,
}

/// Scalability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    pub throughput_ops_per_sec: f64,
    pub concurrent_users: usize,
    pub response_time_degradation: f64,
}

/// Error details for failed tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub error_type: String,
    pub error_message: String,
    pub stack_trace: Option<String>,
    pub context: HashMap<String, String>,
}

/// Test recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRecommendation {
    pub recommendation_type: RecommendationType,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub implementation_effort: ImplementationEffort,
    pub expected_impact: f64,
}

/// Types of recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    Performance,
    Memory,
    Quality,
    Configuration,
    Architecture,
    Testing,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation effort estimation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

/// Integration test report
#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrationTestReport {
    pub report_id: Uuid,
    pub test_summary: TestSummary,
    pub test_results: Vec<TestResult>,
    pub coverage_metrics: TestCoverageMetrics,
    pub execution_metadata: ExecutionMetadata,
}

/// Test summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub skipped_tests: usize,
    pub success_rate: f64,
    pub total_execution_time: Duration,
    pub average_test_time: Duration,
    pub memory_peak_usage_mb: f64,
    pub performance_regression_count: usize,
    pub quality_improvement_count: usize,
    pub recommendations_count: usize,
}

/// Test coverage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCoverageMetrics {
    pub constraint_coverage: f64,
    pub strategy_coverage: f64,
    pub data_type_coverage: f64,
    pub edge_case_coverage: f64,
}

/// Execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub environment_info: EnvironmentInfo,
    pub version_info: VersionInfo,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub os: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub rust_version: String,
}

/// Version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub shacl_ai_version: String,
    pub oxirs_version: String,
    pub test_framework_version: String,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    pub peak_memory_usage_mb: f64,
    pub average_memory_usage_mb: f64,
    pub memory_allocation_count: usize,
    pub memory_deallocation_count: usize,
    pub garbage_collection_count: usize,
}

impl Default for MemoryUsageStats {
    fn default() -> Self {
        Self {
            peak_memory_usage_mb: 0.0,
            average_memory_usage_mb: 0.0,
            memory_allocation_count: 0,
            memory_deallocation_count: 0,
            garbage_collection_count: 0,
        }
    }
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub average_response_time_ms: f64,
    pub median_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub cpu_utilization_percent: f64,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            average_response_time_ms: 0.0,
            median_response_time_ms: 0.0,
            p95_response_time_ms: 0.0,
            p99_response_time_ms: 0.0,
            throughput_ops_per_sec: 0.0,
            cpu_utilization_percent: 0.0,
        }
    }
}

impl Default for TestExecutionStats {
    fn default() -> Self {
        Self {
            total_tests_run: 0,
            successful_tests: 0,
            failed_tests: 0,
            timed_out_tests: 0,
            average_execution_time: Duration::from_millis(0),
            total_execution_time: Duration::from_millis(0),
            memory_usage_stats: MemoryUsageStats::default(),
            performance_stats: PerformanceStats::default(),
        }
    }
}

/// Validation test context
#[derive(Debug, Clone)]
pub struct ValidationTestContext {
    pub test_id: Uuid,
    pub scenario: TestScenario,
    pub validation_params: ValidationParameters,
    pub quality_requirements: QualityRequirements,
}

/// Dependency analysis result
#[derive(Debug, Serialize, Deserialize)]
pub struct DependencyAnalysisResult {
    pub has_circular_dependencies: bool,
    pub dependency_depth: usize,
    pub missing_dependencies: Vec<String>,
    pub compatibility_issues: Vec<String>,
}
