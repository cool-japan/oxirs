//! Configuration types for integration testing
//!
//! This module contains all configuration-related structures for the integration testing framework.

use serde::{Deserialize, Serialize};

/// Integration testing framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationTestConfig {
    /// Test execution timeout per test case
    pub test_timeout_seconds: u64,

    /// Number of parallel test workers
    pub parallel_workers: usize,

    /// Enable performance profiling during tests
    pub enable_performance_profiling: bool,

    /// Enable memory usage monitoring
    pub enable_memory_monitoring: bool,

    /// Enable cross-module dependency testing
    pub enable_dependency_testing: bool,

    /// Generate random test scenarios
    pub enable_scenario_generation: bool,

    /// Test data size variations
    pub test_data_sizes: Vec<usize>,

    /// Test complexity levels
    pub test_complexity_levels: Vec<TestComplexityLevel>,

    /// Minimum success rate threshold
    pub min_success_rate_threshold: f64,

    /// Maximum allowed memory usage (MB)
    pub max_memory_usage_mb: f64,

    /// Maximum allowed execution time per test
    pub max_execution_time_ms: u64,

    /// Enable regression testing
    pub enable_regression_testing: bool,

    /// Test result persistence
    pub persist_test_results: bool,

    /// Test report generation
    pub generate_detailed_reports: bool,
}

/// Test complexity levels for comprehensive testing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestComplexityLevel {
    /// Simple validation rules with minimal constraints
    Simple,
    /// Moderate complexity with multiple constraints
    Medium,
    /// Complex validation with nested constraints
    Complex,
    /// Ultra-complex validation with advanced features
    UltraComplex,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            test_timeout_seconds: 30,
            parallel_workers: 4,
            enable_performance_profiling: true,
            enable_memory_monitoring: true,
            enable_dependency_testing: true,
            enable_scenario_generation: true,
            test_data_sizes: vec![100, 1000, 10000],
            test_complexity_levels: vec![
                TestComplexityLevel::Simple,
                TestComplexityLevel::Medium,
                TestComplexityLevel::Complex,
            ],
            min_success_rate_threshold: 0.95,
            max_memory_usage_mb: 1024.0,
            max_execution_time_ms: 5000,
            enable_regression_testing: true,
            persist_test_results: true,
            generate_detailed_reports: true,
        }
    }
}

/// Data configuration for test scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfiguration {
    /// Number of test entities to generate
    pub entity_count: usize,
    /// Data distribution pattern
    pub distribution: DataDistribution,
    /// Random seed for reproducible tests
    pub random_seed: Option<u64>,
    /// Include edge cases in test data
    pub include_edge_cases: bool,
}

/// Data distribution patterns for test generation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataDistribution {
    /// Uniform distribution across all possible values
    Uniform,
    /// Normal distribution with mean and std dev
    Normal,
    /// Skewed distribution with bias towards specific values
    Skewed,
}

/// Validation configuration for test scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfiguration {
    /// Use strict validation mode
    pub strict_mode: bool,
    /// Enable parallel validation
    pub parallel_validation: bool,
    /// Maximum validation depth
    pub max_validation_depth: u32,
    /// Enable shape inference
    pub enable_shape_inference: bool,
}

/// Quality thresholds for test validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum accuracy threshold (0.0 - 1.0)
    pub min_accuracy: f64,
    /// Minimum completeness threshold (0.0 - 1.0)
    pub min_completeness: f64,
    /// Minimum consistency threshold (0.0 - 1.0)
    pub min_consistency: f64,
}

/// Validation parameters for test contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationParameters {
    /// Enable closed world assumption
    pub closed_world: bool,
    /// Maximum depth for recursive validation
    pub max_depth: u32,
    /// Enable optimization
    pub optimize: bool,
}

/// Quality requirements for tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    /// Required accuracy level
    pub accuracy: f64,
    /// Required completeness level
    pub completeness: f64,
    /// Required performance level
    pub performance: f64,
}
