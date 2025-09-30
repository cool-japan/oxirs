//! Configuration types for performance benchmarking
//!
//! This module contains configuration structures for the performance
//! benchmarking framework and its various components.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::types::{MeasurementConfig, SuccessCriteria, WorkloadConfig};

/// Performance benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Benchmark execution timeout per benchmark
    pub benchmark_timeout_seconds: u64,

    /// Number of benchmark iterations
    pub benchmark_iterations: usize,

    /// Warmup iterations before measurement
    pub warmup_iterations: usize,

    /// Enable statistical analysis
    pub enable_statistical_analysis: bool,

    /// Enable memory profiling
    pub enable_memory_profiling: bool,

    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,

    /// Enable I/O profiling
    pub enable_io_profiling: bool,

    /// Data sizes for scalability testing
    pub scalability_data_sizes: Vec<usize>,

    /// Concurrency levels for parallel testing
    pub concurrency_levels: Vec<usize>,

    /// Performance regression threshold (percentage)
    pub regression_threshold_percent: f64,

    /// Enable automated performance tuning
    pub enable_auto_tuning: bool,

    /// Benchmark result persistence
    pub persist_results: bool,

    /// Generate detailed reports
    pub generate_detailed_reports: bool,

    /// Enable comparative analysis
    pub enable_comparative_analysis: bool,

    /// Baseline comparison enabled
    pub enable_baseline_comparison: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            benchmark_timeout_seconds: 600,
            benchmark_iterations: 10,
            warmup_iterations: 3,
            enable_statistical_analysis: true,
            enable_memory_profiling: true,
            enable_cpu_profiling: true,
            enable_io_profiling: true,
            scalability_data_sizes: vec![100, 1000, 10000, 100000, 1000000],
            concurrency_levels: vec![1, 2, 4, 8, 16],
            regression_threshold_percent: 5.0,
            enable_auto_tuning: true,
            persist_results: true,
            generate_detailed_reports: true,
            enable_comparative_analysis: true,
            enable_baseline_comparison: true,
        }
    }
}

/// Benchmark suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteConfig {
    /// Suite name
    pub name: String,
    /// Suite description
    pub description: String,
    /// Global timeout for the entire suite
    pub suite_timeout: Duration,
    /// Default workload configuration
    pub default_workload: WorkloadConfig,
    /// Default measurement configuration
    pub default_measurement: MeasurementConfig,
    /// Default success criteria
    pub default_success_criteria: SuccessCriteria,
    /// Enable parallel execution of benchmarks
    pub enable_parallel_execution: bool,
    /// Maximum concurrent benchmarks
    pub max_concurrent_benchmarks: usize,
    /// Fail fast on first benchmark failure
    pub fail_fast: bool,
    /// Generate summary report
    pub generate_summary: bool,
}

impl Default for BenchmarkSuiteConfig {
    fn default() -> Self {
        Self {
            name: "Default Benchmark Suite".to_string(),
            description: "Default benchmark suite configuration".to_string(),
            suite_timeout: Duration::from_secs(3600), // 1 hour
            default_workload: WorkloadConfig::default(),
            default_measurement: MeasurementConfig::default(),
            default_success_criteria: SuccessCriteria::default(),
            enable_parallel_execution: true,
            max_concurrent_benchmarks: 4,
            fail_fast: false,
            generate_summary: true,
        }
    }
}

/// Profiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Memory sampling frequency (Hz)
    pub memory_sample_frequency_hz: f64,
    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,
    /// CPU sampling frequency (Hz)
    pub cpu_sample_frequency_hz: f64,
    /// Enable I/O profiling
    pub enable_io_profiling: bool,
    /// I/O sampling frequency (Hz)
    pub io_sample_frequency_hz: f64,
    /// Enable call stack profiling
    pub enable_call_stack_profiling: bool,
    /// Maximum call stack depth
    pub max_call_stack_depth: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_memory_profiling: true,
            memory_sample_frequency_hz: 10.0,
            enable_cpu_profiling: true,
            cpu_sample_frequency_hz: 100.0,
            enable_io_profiling: true,
            io_sample_frequency_hz: 10.0,
            enable_call_stack_profiling: false,
            max_call_stack_depth: 50,
        }
    }
}

/// Scalability testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityConfig {
    /// Data sizes to test
    pub data_sizes: Vec<usize>,
    /// Concurrency levels to test
    pub concurrency_levels: Vec<usize>,
    /// Enable throughput scaling analysis
    pub enable_throughput_analysis: bool,
    /// Enable latency scaling analysis
    pub enable_latency_analysis: bool,
    /// Enable resource usage scaling analysis
    pub enable_resource_analysis: bool,
    /// Maximum acceptable degradation percentage
    pub max_degradation_percent: f64,
}

impl Default for ScalabilityConfig {
    fn default() -> Self {
        Self {
            data_sizes: vec![100, 1000, 10000, 100000],
            concurrency_levels: vec![1, 2, 4, 8, 16],
            enable_throughput_analysis: true,
            enable_latency_analysis: true,
            enable_resource_analysis: true,
            max_degradation_percent: 20.0,
        }
    }
}

/// Regression detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Regression threshold percentage
    pub threshold_percent: f64,
    /// Number of historical runs to compare against
    pub historical_runs_count: usize,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    /// Enable statistical significance testing
    pub enable_statistical_testing: bool,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Enable automated alerts
    pub enable_automated_alerts: bool,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            threshold_percent: 5.0,
            historical_runs_count: 10,
            enable_trend_analysis: true,
            enable_statistical_testing: true,
            confidence_level: 0.95,
            enable_automated_alerts: true,
        }
    }
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Enable statistical analysis
    pub enable_statistical_analysis: bool,
    /// Enable performance trend analysis
    pub enable_trend_analysis: bool,
    /// Enable bottleneck detection
    pub enable_bottleneck_detection: bool,
    /// Enable optimization recommendations
    pub enable_optimization_recommendations: bool,
    /// Enable comparative analysis
    pub enable_comparative_analysis: bool,
    /// Analysis depth level (1-5)
    pub analysis_depth: u8,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_statistical_analysis: true,
            enable_trend_analysis: true,
            enable_bottleneck_detection: true,
            enable_optimization_recommendations: true,
            enable_comparative_analysis: true,
            analysis_depth: 3,
        }
    }
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Generate HTML reports
    pub generate_html_reports: bool,
    /// Generate JSON reports
    pub generate_json_reports: bool,
    /// Generate CSV reports
    pub generate_csv_reports: bool,
    /// Generate charts and graphs
    pub generate_charts: bool,
    /// Include detailed metrics
    pub include_detailed_metrics: bool,
    /// Include recommendations
    pub include_recommendations: bool,
    /// Output directory for reports
    pub output_directory: String,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            generate_html_reports: true,
            generate_json_reports: true,
            generate_csv_reports: true,
            generate_charts: true,
            include_detailed_metrics: true,
            include_recommendations: true,
            output_directory: "./benchmark_reports".to_string(),
        }
    }
}
