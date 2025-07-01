//! Test reporting and result types

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Complete test report containing all results and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    /// Overall test summary
    pub summary: TestSummary,
    /// Results by module
    pub module_results: HashMap<String, ModuleTestResult>,
    /// Integration analysis
    pub integration_analysis: IntegrationAnalysis,
    /// System recommendations
    pub recommendations: Vec<Recommendation>,
}

/// Test summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    /// Total number of tests executed
    pub total_tests: usize,
    /// Number of tests that passed
    pub passed_tests: usize,
    /// Number of tests that failed
    pub failed_tests: usize,
    /// Success rate as percentage
    pub success_rate: f64,
    /// Total time spent executing tests
    pub total_execution_time: Duration,
    /// Overall performance score (0-100)
    pub performance_score: f64,
}

/// Test results for a specific module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleTestResult {
    /// Module name
    pub module_name: String,
    /// Total tests in this module
    pub total_tests: usize,
    /// Passed tests in this module
    pub passed_tests: usize,
    /// Failed tests in this module
    pub failed_tests: usize,
    /// Individual test results
    pub test_results: Vec<IndividualTestResult>,
    /// Average performance score for this module
    pub average_performance_score: f64,
    /// Module-specific recommendations
    pub recommendations: Vec<Recommendation>,
    /// Execution time for this module
    pub execution_time: Duration,
}

impl ModuleTestResult {
    pub fn new(module_name: String) -> Self {
        Self {
            module_name,
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            test_results: Vec::new(),
            average_performance_score: 0.0,
            recommendations: Vec::new(),
            execution_time: Duration::from_nanos(0),
        }
    }

    pub fn add_test_result(&mut self, result: IndividualTestResult) {
        self.total_tests += 1;
        if result.passed {
            self.passed_tests += 1;
        } else {
            self.failed_tests += 1;
        }
        self.execution_time += result.execution_time;
        self.test_results.push(result);
        
        // Recalculate average performance score
        self.average_performance_score = self.test_results.iter()
            .map(|r| r.performance_score)
            .sum::<f64>() / self.test_results.len() as f64;
    }
}

/// Result of an individual test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndividualTestResult {
    /// Test name
    pub test_name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Test execution time
    pub execution_time: Duration,
    /// Performance score for this test (0-100)
    pub performance_score: f64,
    /// Memory usage during test (MB)
    pub memory_usage_mb: f64,
    /// Error message if test failed
    pub error_message: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl IndividualTestResult {
    pub fn new_passed(test_name: String, execution_time: Duration, performance_score: f64) -> Self {
        Self {
            test_name,
            passed: true,
            execution_time,
            performance_score,
            memory_usage_mb: 0.0,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    pub fn new_failed(test_name: String, execution_time: Duration, error: String) -> Self {
        Self {
            test_name,
            passed: false,
            execution_time,
            performance_score: 0.0,
            memory_usage_mb: 0.0,
            error_message: Some(error),
            metadata: HashMap::new(),
        }
    }

    pub fn with_memory_usage(mut self, memory_mb: f64) -> Self {
        self.memory_usage_mb = memory_mb;
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Analysis of integration coverage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationAnalysis {
    /// Percentage of integration points covered
    pub coverage_percentage: f64,
    /// List of integration points that were tested
    pub integration_points_tested: Vec<String>,
    /// List of integration points that are missing tests
    pub missing_integration_points: Vec<String>,
}

/// System recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority level of the recommendation
    pub priority: RecommendationPriority,
    /// Category of the recommendation
    pub category: String,
    /// Description of the recommendation
    pub description: String,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Operations per second
    pub ops_per_sec: f64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// 95th percentile latency in microseconds
    pub p95_latency_us: f64,
    /// 99th percentile latency in microseconds
    pub p99_latency_us: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageMetrics {
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Average memory usage in MB
    pub avg_memory_mb: f64,
    /// Peak CPU usage percentage
    pub peak_cpu_percent: f64,
    /// Average CPU usage percentage
    pub avg_cpu_percent: f64,
    /// Total disk I/O operations
    pub disk_io_operations: u64,
    /// Total network I/O bytes
    pub network_io_bytes: u64,
}

impl TestReport {
    /// Generate a human-readable summary
    pub fn generate_summary(&self) -> String {
        format!(
            "Integration Test Report\n\
             ======================\n\
             Total Tests: {}\n\
             Passed: {} ({:.1}%)\n\
             Failed: {}\n\
             Execution Time: {:?}\n\
             Performance Score: {:.1}/100\n\
             Integration Coverage: {:.1}%\n\
             Recommendations: {} ({} critical)",
            self.summary.total_tests,
            self.summary.passed_tests,
            self.summary.success_rate,
            self.summary.failed_tests,
            self.summary.total_execution_time,
            self.summary.performance_score,
            self.integration_analysis.coverage_percentage,
            self.recommendations.len(),
            self.recommendations.iter()
                .filter(|r| r.priority == RecommendationPriority::Critical)
                .count()
        )
    }

    /// Check if all tests passed
    pub fn all_tests_passed(&self) -> bool {
        self.summary.failed_tests == 0
    }

    /// Get critical recommendations
    pub fn get_critical_recommendations(&self) -> Vec<&Recommendation> {
        self.recommendations.iter()
            .filter(|r| r.priority == RecommendationPriority::Critical)
            .collect()
    }

    /// Get performance score by module
    pub fn get_module_performance_scores(&self) -> HashMap<String, f64> {
        self.module_results.iter()
            .map(|(name, result)| (name.clone(), result.average_performance_score))
            .collect()
    }
}