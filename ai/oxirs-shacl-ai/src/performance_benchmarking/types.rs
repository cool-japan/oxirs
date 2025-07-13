//! Core types for performance benchmarking
//!
//! This module contains the fundamental data structures and enums used
//! throughout the performance benchmarking system.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Benchmark type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BenchmarkType {
    /// Micro benchmarks for individual components
    Micro,
    /// Macro benchmarks for end-to-end workflows
    Macro,
    /// Load testing benchmarks
    Load,
    /// Stress testing benchmarks
    Stress,
    /// Memory usage benchmarks
    Memory,
    /// CPU utilization benchmarks
    Cpu,
    /// I/O performance benchmarks
    Io,
    /// Scalability benchmarks
    Scalability,
    /// Regression testing benchmarks
    Regression,
}

/// Target component for benchmarking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetComponent {
    /// SHACL validation engine
    ValidationEngine,
    /// Pattern recognition system
    PatternRecognition,
    /// AI orchestrator
    AiOrchestrator,
    /// Constraint optimizer
    ConstraintOptimizer,
    /// Cache subsystem
    CacheSubsystem,
    /// Query processor
    QueryProcessor,
    /// Rule engine
    RuleEngine,
    /// Full system integration
    FullSystem,
}

/// Data access patterns for benchmarking
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Sequential data access
    Sequential,
    /// Random data access
    Random,
    /// Hot-spot concentrated access
    HotSpot,
    /// Uniform distribution access
    Uniform,
    /// Zipfian distribution access
    Zipfian,
}

/// Cache behavior configuration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheBehavior {
    /// Cold cache (no pre-warming)
    Cold,
    /// Warm cache (pre-warmed)
    Warm,
    /// Hot cache (fully loaded)
    Hot,
    /// Mixed behavior
    Mixed,
}

/// Data distribution patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataDistribution {
    /// Uniform data distribution
    Uniform,
    /// Normal (Gaussian) distribution
    Normal,
    /// Exponential distribution
    Exponential,
    /// Power-law distribution
    PowerLaw,
    /// Custom distribution
    Custom,
}

/// Measurement precision levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionLevel {
    /// Low precision, faster execution
    Low,
    /// Medium precision, balanced
    Medium,
    /// High precision, slower execution
    High,
    /// Ultra precision, very slow execution
    Ultra,
}

/// Benchmark execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkStatus {
    /// Benchmark is pending execution
    Pending,
    /// Benchmark is currently running
    Running,
    /// Benchmark completed successfully
    Completed,
    /// Benchmark failed with error
    Failed,
    /// Benchmark was cancelled
    Cancelled,
    /// Benchmark timed out
    TimedOut,
}

/// Workload configuration for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    /// Data size in number of records
    pub data_size: usize,
    /// Concurrency level (number of parallel workers)
    pub concurrency_level: usize,
    /// Data access pattern
    pub access_pattern: AccessPattern,
    /// Cache behavior
    pub cache_behavior: CacheBehavior,
    /// Data distribution
    pub data_distribution: DataDistribution,
    /// Request rate (requests per second)
    pub request_rate: f64,
    /// Duration of the workload
    pub duration: Duration,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            data_size: 10000,
            concurrency_level: 4,
            access_pattern: AccessPattern::Random,
            cache_behavior: CacheBehavior::Cold,
            data_distribution: DataDistribution::Uniform,
            request_rate: 100.0,
            duration: Duration::from_secs(60),
        }
    }
}

/// Measurement configuration for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementConfig {
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Measurement precision level
    pub precision_level: PrecisionLevel,
    /// Sample collection frequency
    pub sample_frequency_hz: f64,
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Enable memory tracking
    pub enable_memory_tracking: bool,
    /// Enable CPU monitoring
    pub enable_cpu_monitoring: bool,
    /// Enable I/O monitoring
    pub enable_io_monitoring: bool,
}

impl Default for MeasurementConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            warmup_iterations: 3,
            precision_level: PrecisionLevel::Medium,
            sample_frequency_hz: 10.0,
            enable_profiling: true,
            enable_memory_tracking: true,
            enable_cpu_monitoring: true,
            enable_io_monitoring: true,
        }
    }
}

/// Success criteria for benchmark validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    /// Maximum acceptable execution time
    pub max_execution_time: Duration,
    /// Maximum acceptable memory usage (MB)
    pub max_memory_usage_mb: f64,
    /// Maximum acceptable CPU usage (percentage)
    pub max_cpu_usage_percent: f64,
    /// Minimum acceptable throughput (operations/second)
    pub min_throughput_ops_per_sec: f64,
    /// Maximum acceptable error rate (percentage)
    pub max_error_rate_percent: f64,
    /// Minimum acceptable success rate (percentage)
    pub min_success_rate_percent: f64,
}

impl Default for SuccessCriteria {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(300),
            max_memory_usage_mb: 1024.0,
            max_cpu_usage_percent: 80.0,
            min_throughput_ops_per_sec: 10.0,
            max_error_rate_percent: 1.0,
            min_success_rate_percent: 99.0,
        }
    }
}

/// Running benchmark state information
#[derive(Debug, Clone)]
pub struct RunningBenchmark {
    pub benchmark_id: Uuid,
    pub name: String,
    pub status: BenchmarkStatus,
    pub start_time: Instant,
    pub current_iteration: usize,
    pub total_iterations: usize,
    pub progress_percentage: f64,
    pub estimated_time_remaining: Option<Duration>,
}

/// Benchmark execution context
#[derive(Debug, Clone)]
pub struct BenchmarkExecutionContext {
    pub benchmark_id: Uuid,
    pub iteration_number: usize,
    pub start_time: Instant,
    pub workload_config: WorkloadConfig,
    pub measurement_config: MeasurementConfig,
    pub success_criteria: SuccessCriteria,
}

/// Performance counters for detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCounters {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Memory allocation rate (MB/s)
    pub memory_allocation_rate_mb_per_sec: f64,
    /// Garbage collection time (ms)
    pub gc_time_ms: f64,
    /// I/O read rate (MB/s)
    pub io_read_rate_mb_per_sec: f64,
    /// I/O write rate (MB/s)
    pub io_write_rate_mb_per_sec: f64,
    /// Network input rate (MB/s)
    pub network_input_rate_mb_per_sec: f64,
    /// Network output rate (MB/s)
    pub network_output_rate_mb_per_sec: f64,
    /// Thread count
    pub thread_count: usize,
    /// Context switches per second
    pub context_switches_per_sec: f64,
    /// Cache hit rate percentage
    pub cache_hit_rate_percent: f64,
}

impl Default for PerformanceCounters {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            memory_allocation_rate_mb_per_sec: 0.0,
            gc_time_ms: 0.0,
            io_read_rate_mb_per_sec: 0.0,
            io_write_rate_mb_per_sec: 0.0,
            network_input_rate_mb_per_sec: 0.0,
            network_output_rate_mb_per_sec: 0.0,
            thread_count: 1,
            context_switches_per_sec: 0.0,
            cache_hit_rate_percent: 0.0,
        }
    }
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_id: Uuid,
    pub name: String,
    pub benchmark_type: BenchmarkType,
    pub target_component: TargetComponent,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub execution_time: Duration,
    pub iterations_completed: usize,
    pub status: BenchmarkStatus,
    pub success: bool,
    pub throughput_ops_per_sec: f64,
    pub latency_percentiles: BTreeMap<String, Duration>,
    pub performance_counters: PerformanceCounters,
    pub memory_usage_stats: MemoryUsageStats,
    pub cpu_usage_stats: CpuUsageStats,
    pub io_usage_stats: IoUsageStats,
    pub error_count: usize,
    pub error_messages: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub total_allocations_mb: f64,
    pub allocation_rate_mb_per_sec: f64,
    pub gc_count: usize,
    pub gc_total_time_ms: f64,
}

impl Default for MemoryUsageStats {
    fn default() -> Self {
        Self {
            peak_memory_mb: 0.0,
            average_memory_mb: 0.0,
            total_allocations_mb: 0.0,
            allocation_rate_mb_per_sec: 0.0,
            gc_count: 0,
            gc_total_time_ms: 0.0,
        }
    }
}

/// CPU usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUsageStats {
    pub peak_cpu_percent: f64,
    pub average_cpu_percent: f64,
    pub user_time_percent: f64,
    pub system_time_percent: f64,
    pub context_switches: usize,
    pub cpu_cycles: u64,
}

impl Default for CpuUsageStats {
    fn default() -> Self {
        Self {
            peak_cpu_percent: 0.0,
            average_cpu_percent: 0.0,
            user_time_percent: 0.0,
            system_time_percent: 0.0,
            context_switches: 0,
            cpu_cycles: 0,
        }
    }
}

/// I/O usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoUsageStats {
    pub total_bytes_read: u64,
    pub total_bytes_written: u64,
    pub read_operations: usize,
    pub write_operations: usize,
    pub read_rate_mb_per_sec: f64,
    pub write_rate_mb_per_sec: f64,
    pub average_io_latency: Duration,
}

impl Default for IoUsageStats {
    fn default() -> Self {
        Self {
            total_bytes_read: 0,
            total_bytes_written: 0,
            read_operations: 0,
            write_operations: 0,
            read_rate_mb_per_sec: 0.0,
            write_rate_mb_per_sec: 0.0,
            average_io_latency: Duration::from_millis(0),
        }
    }
}

/// Execution summary for a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSummary {
    pub total_benchmarks: usize,
    pub successful_benchmarks: usize,
    pub failed_benchmarks: usize,
    pub total_execution_time: Duration,
    pub average_execution_time: Duration,
    pub throughput_summary: ThroughputSummary,
    pub resource_usage_summary: ResourceUsageSummary,
    pub success_rate_percent: f64,
}

/// Throughput summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputSummary {
    pub min_throughput_ops_per_sec: f64,
    pub max_throughput_ops_per_sec: f64,
    pub average_throughput_ops_per_sec: f64,
    pub median_throughput_ops_per_sec: f64,
    pub throughput_std_dev: f64,
}

/// Resource usage summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageSummary {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub peak_cpu_percent: f64,
    pub average_cpu_percent: f64,
    pub total_io_mb: f64,
    pub average_io_latency: Duration,
}
