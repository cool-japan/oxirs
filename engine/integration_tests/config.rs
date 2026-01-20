//! Configuration types for integration testing

use std::time::Duration;

/// Test execution strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStrategy {
    Sequential,
    Parallel,
    Adaptive,
}

/// Test environment configuration
#[derive(Debug, Clone)]
pub struct TestEnvironment {
    /// Maximum memory usage allowed during tests (MB)
    pub max_memory_mb: u64,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f64,
    /// Temporary directory for test data
    pub temp_dir: String,
    /// Enable resource monitoring
    pub monitor_resources: bool,
}

impl Default for TestEnvironment {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,
            max_cpu_percent: 90.0,
            temp_dir: "/tmp/oxirs_test".to_string(),
            monitor_resources: true,
        }
    }
}

/// Test data configuration
#[derive(Debug, Clone)]
pub struct TestDataConfig {
    /// Path to test datasets
    pub dataset_path: String,
    /// Maximum dataset size to load (MB)
    pub max_dataset_size_mb: u64,
    /// Enable synthetic data generation
    pub generate_synthetic_data: bool,
    /// Seed for reproducible data generation
    pub random_seed: u64,
}

impl Default for TestDataConfig {
    fn default() -> Self {
        Self {
            dataset_path: "data/test".to_string(),
            max_dataset_size_mb: 100,
            generate_synthetic_data: true,
            random_seed: 42,
        }
    }
}

/// Logging configuration for tests
#[derive(Debug, Clone)]
pub struct TestLoggingConfig {
    /// Log level for test execution
    pub log_level: String,
    /// Enable performance logging
    pub log_performance: bool,
    /// Enable memory usage logging
    pub log_memory: bool,
    /// Log file path
    pub log_file: Option<String>,
}

impl Default for TestLoggingConfig {
    fn default() -> Self {
        Self {
            log_level: "INFO".to_string(),
            log_performance: true,
            log_memory: true,
            log_file: None,
        }
    }
}

/// Extended integration test configuration
#[derive(Debug, Clone)]
pub struct ExtendedIntegrationTestConfig {
    /// Basic configuration
    pub basic: super::IntegrationTestConfig,
    /// Test environment settings
    pub environment: TestEnvironment,
    /// Test data configuration
    pub data: TestDataConfig,
    /// Logging configuration
    pub logging: TestLoggingConfig,
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
}

impl Default for ExtendedIntegrationTestConfig {
    fn default() -> Self {
        Self {
            basic: super::IntegrationTestConfig::default(),
            environment: TestEnvironment::default(),
            data: TestDataConfig::default(),
            logging: TestLoggingConfig::default(),
            execution_strategy: ExecutionStrategy::Adaptive,
        }
    }
}