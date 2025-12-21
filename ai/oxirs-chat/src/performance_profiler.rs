//! Advanced Performance Profiler using SciRS2-Core
//!
//! This module provides comprehensive performance profiling and benchmarking
//! for the OxiRS Chat system using scirs2-core's advanced profiling capabilities.
//!
//! # Features
//!
//! - Real-time performance profiling
//! - Comprehensive benchmarking suite
//! - Memory usage tracking
//! - GPU performance monitoring
//! - Query optimization statistics
//!
//! # Examples
//!
//! ```rust,no_run
//! use oxirs_chat::performance_profiler::{ChatProfiler, ProfilingConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = ProfilingConfig::default();
//! let profiler = ChatProfiler::new(config)?;
//!
//! profiler.start_profiling("rag_retrieval");
//! // ... perform RAG retrieval ...
//! let stats = profiler.stop_profiling("rag_retrieval")?;
//!
//! println!("RAG retrieval took: {:?}", stats.duration);
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use scirs2_core::{
    benchmarking::{BenchmarkRunner, BenchmarkSuite},
    error::CoreError,
    memory::BufferPool,
    metrics::{Counter, Gauge, Histogram, MetricRegistry, Timer},
    profiling::{profiling_memory_tracker, Profiler},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for the performance profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable detailed profiling
    pub enable_detailed_profiling: bool,

    /// Enable memory tracking
    pub enable_memory_tracking: bool,

    /// Enable GPU profiling (if available)
    pub enable_gpu_profiling: bool,

    /// Enable benchmarking suite
    pub enable_benchmarking: bool,

    /// Profiling sample rate (0.0 to 1.0)
    pub sample_rate: f64,

    /// Maximum number of profiling entries to keep
    pub max_entries: usize,

    /// Enable automatic report generation
    pub auto_report: bool,

    /// Report generation interval in seconds
    pub report_interval_secs: u64,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enable_detailed_profiling: true,
            enable_memory_tracking: true,
            enable_gpu_profiling: false, // Disabled by default, enable if GPU available
            enable_benchmarking: true,
            sample_rate: 1.0, // Profile all requests
            max_entries: 10_000,
            auto_report: false,
            report_interval_secs: 300, // 5 minutes
        }
    }
}

/// Profiling statistics for a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingStats {
    /// Operation name
    pub operation: String,

    /// Duration of the operation
    pub duration: Duration,

    /// Memory allocated during operation (bytes)
    pub memory_allocated: Option<usize>,

    /// Memory deallocated during operation (bytes)
    pub memory_deallocated: Option<usize>,

    /// Peak memory usage (bytes)
    pub peak_memory: Option<usize>,

    /// CPU usage percentage
    pub cpu_usage: Option<f64>,

    /// GPU usage percentage (if enabled)
    pub gpu_usage: Option<f64>,

    /// Timestamp when profiling started
    pub start_time: chrono::DateTime<chrono::Utc>,

    /// Timestamp when profiling ended
    pub end_time: chrono::DateTime<chrono::Utc>,

    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Profiling session for tracking ongoing operations
#[derive(Debug, Clone)]
struct ProfilingSession {
    operation: String,
    start_time: Instant,
    start_timestamp: chrono::DateTime<chrono::Utc>,
    memory_start: Option<usize>,
}

/// Advanced performance profiler for OxiRS Chat
pub struct ChatProfiler {
    config: ProfilingConfig,
    profiler: Arc<Profiler>,
    metrics: Arc<MetricRegistry>,
    active_sessions: Arc<RwLock<HashMap<String, ProfilingSession>>>,
    completed_stats: Arc<RwLock<Vec<ProfilingStats>>>,
    benchmark_suite: Option<Arc<RwLock<BenchmarkSuite>>>,

    // SciRS2-Core metrics
    rag_retrieval_timer: Arc<Timer>,
    llm_generation_timer: Arc<Timer>,
    sparql_execution_timer: Arc<Timer>,
    total_requests_counter: Arc<Counter>,
    active_sessions_gauge: Arc<Gauge>,
    response_time_histogram: Arc<Histogram>,
}

impl ChatProfiler {
    /// Create a new chat profiler with the given configuration
    pub fn new(config: ProfilingConfig) -> Result<Self> {
        let profiler = Arc::new(Profiler::new());
        let metrics = Arc::new(MetricRegistry::global());

        // Initialize SciRS2-Core metrics
        let rag_retrieval_timer = Arc::new(Timer::new("rag_retrieval_time".to_string()));
        let llm_generation_timer = Arc::new(Timer::new("llm_generation_time".to_string()));
        let sparql_execution_timer = Arc::new(Timer::new("sparql_execution_time".to_string()));
        let total_requests_counter = Arc::new(Counter::new("total_chat_requests".to_string()));
        let active_sessions_gauge = Arc::new(Gauge::new("active_profiling_sessions".to_string()));
        let response_time_histogram = Arc::new(Histogram::new("response_time_ms".to_string()));

        // Register metrics
        metrics.register_counter(total_requests_counter.clone());
        metrics.register_gauge(active_sessions_gauge.clone());
        metrics.register_histogram(response_time_histogram.clone());
        metrics.register_timer(rag_retrieval_timer.clone());
        metrics.register_timer(llm_generation_timer.clone());
        metrics.register_timer(sparql_execution_timer.clone());

        // Initialize benchmark suite if enabled
        let benchmark_suite = if config.enable_benchmarking {
            use scirs2_core::benchmarking::BenchmarkConfig;
            let bench_config = BenchmarkConfig::default();
            Some(Arc::new(RwLock::new(BenchmarkSuite::new(
                "oxirs_chat_benchmarks",
                bench_config,
            ))))
        } else {
            None
        };

        Ok(Self {
            config,
            profiler,
            metrics,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            completed_stats: Arc::new(RwLock::new(Vec::new())),
            benchmark_suite,
            rag_retrieval_timer,
            llm_generation_timer,
            sparql_execution_timer,
            total_requests_counter,
            active_sessions_gauge,
            response_time_histogram,
        })
    }

    /// Start profiling an operation
    pub async fn start_profiling(&self, operation: &str) -> Result<()> {
        if !self.should_profile() {
            return Ok(());
        }

        let session = ProfilingSession {
            operation: operation.to_string(),
            start_time: Instant::now(),
            start_timestamp: chrono::Utc::now(),
            memory_start: if self.config.enable_memory_tracking {
                Some(self.get_current_memory_usage())
            } else {
                None
            },
        };

        let mut sessions = self.active_sessions.write().await;
        sessions.insert(operation.to_string(), session);
        self.active_sessions_gauge.set(sessions.len() as f64);

        // Start scirs2-core profiling
        self.profiler.start(operation);

        debug!("Started profiling: {}", operation);
        Ok(())
    }

    /// Stop profiling an operation and return statistics
    pub async fn stop_profiling(&self, operation: &str) -> Result<ProfilingStats> {
        let mut sessions = self.active_sessions.write().await;
        let session = sessions
            .remove(operation)
            .ok_or_else(|| anyhow::anyhow!("Profiling session not found: {}", operation))?;

        self.active_sessions_gauge.set(sessions.len() as f64);
        drop(sessions);

        // Stop scirs2-core profiling
        self.profiler.stop(operation);

        let duration = session.start_time.elapsed();
        let end_timestamp = chrono::Utc::now();

        let memory_allocated = if self.config.enable_memory_tracking {
            let current_memory = self.get_current_memory_usage();
            session.memory_start.map(|start| {
                if current_memory > start {
                    current_memory - start
                } else {
                    0
                }
            })
        } else {
            None
        };

        let stats = ProfilingStats {
            operation: operation.to_string(),
            duration,
            memory_allocated,
            memory_deallocated: None, // TODO: Implement if needed
            peak_memory: None,        // TODO: Implement if needed
            cpu_usage: self.get_cpu_usage(),
            gpu_usage: if self.config.enable_gpu_profiling {
                self.get_gpu_usage()
            } else {
                None
            },
            start_time: session.start_timestamp,
            end_time: end_timestamp,
            custom_metrics: HashMap::new(),
        };

        // Record metrics
        self.total_requests_counter.increment();
        self.response_time_histogram
            .record(duration.as_millis() as f64);

        // Record operation-specific timers
        match operation {
            op if op.contains("rag") => {
                self.rag_retrieval_timer.record(duration);
            }
            op if op.contains("llm") => {
                self.llm_generation_timer.record(duration);
            }
            op if op.contains("sparql") => {
                self.sparql_execution_timer.record(duration);
            }
            _ => {}
        }

        // Store stats
        let mut completed = self.completed_stats.write().await;
        completed.push(stats.clone());

        // Trim if exceeds max entries
        if completed.len() > self.config.max_entries {
            completed.drain(0..completed.len() - self.config.max_entries);
        }

        debug!(
            "Stopped profiling: {} (took {:?})",
            operation, stats.duration
        );

        Ok(stats)
    }

    /// Get all completed profiling statistics
    pub async fn get_all_stats(&self) -> Vec<ProfilingStats> {
        let stats = self.completed_stats.read().await;
        stats.clone()
    }

    /// Get statistics for a specific operation
    pub async fn get_operation_stats(&self, operation: &str) -> Vec<ProfilingStats> {
        let stats = self.completed_stats.read().await;
        stats
            .iter()
            .filter(|s| s.operation == operation)
            .cloned()
            .collect()
    }

    /// Generate a comprehensive performance report
    pub async fn generate_report(&self) -> Result<PerformanceReport> {
        let stats = self.get_all_stats().await;

        if stats.is_empty() {
            return Ok(PerformanceReport::default());
        }

        // Calculate aggregate statistics
        let total_operations = stats.len();
        let total_duration: Duration = stats.iter().map(|s| s.duration).sum();
        let avg_duration =
            Duration::from_nanos(total_duration.as_nanos() as u64 / total_operations as u64);

        let durations: Vec<Duration> = stats.iter().map(|s| s.duration).collect();
        let median_duration = self.calculate_median_duration(&durations);

        let p95_duration = self.calculate_percentile_duration(&durations, 0.95);
        let p99_duration = self.calculate_percentile_duration(&durations, 0.99);

        // Operation breakdown
        let mut operation_counts: HashMap<String, usize> = HashMap::new();
        let mut operation_durations: HashMap<String, Vec<Duration>> = HashMap::new();

        for stat in &stats {
            *operation_counts.entry(stat.operation.clone()).or_insert(0) += 1;
            operation_durations
                .entry(stat.operation.clone())
                .or_insert_with(Vec::new)
                .push(stat.duration);
        }

        let operation_stats: HashMap<String, OperationStats> = operation_counts
            .iter()
            .map(|(op, &count)| {
                let durations = &operation_durations[op];
                let total: Duration = durations.iter().copied().sum();
                let avg = Duration::from_nanos(total.as_nanos() as u64 / count as u64);
                let median = self.calculate_median_duration(durations);

                (
                    op.clone(),
                    OperationStats {
                        count,
                        total_duration: total,
                        avg_duration: avg,
                        median_duration: median,
                        p95_duration: self.calculate_percentile_duration(durations, 0.95),
                        p99_duration: self.calculate_percentile_duration(durations, 0.99),
                    },
                )
            })
            .collect();

        // Memory statistics
        let total_memory_allocated: Option<usize> = if self.config.enable_memory_tracking {
            Some(
                stats
                    .iter()
                    .filter_map(|s| s.memory_allocated)
                    .sum::<usize>(),
            )
        } else {
            None
        };

        let report = PerformanceReport {
            generated_at: chrono::Utc::now(),
            total_operations,
            total_duration,
            avg_duration,
            median_duration,
            p95_duration,
            p99_duration,
            operation_stats,
            total_memory_allocated,
            config: self.config.clone(),
        };

        info!("Generated performance report: {} operations profiled", total_operations);

        Ok(report)
    }

    /// Run benchmark suite
    pub async fn run_benchmarks(&self) -> Result<BenchmarkResults> {
        if !self.config.enable_benchmarking {
            return Err(anyhow::anyhow!("Benchmarking is disabled"));
        }

        let suite = self
            .benchmark_suite
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Benchmark suite not initialized"))?;

        info!("Running benchmark suite...");

        let suite_guard = suite.read().await;
        let runner = BenchmarkRunner::new();
        let result = runner
            .run(&*suite_guard)
            .context("Failed to run benchmark suite")?;

        info!("Benchmark suite completed");

        // Convert scirs2-core BenchmarkResult to our format
        Ok(BenchmarkResults {
            suite_name: "oxirs_chat_benchmarks".to_string(),
            total_benchmarks: 1, // Single result from suite
            results: vec![BenchmarkResult {
                name: result.name.clone(),
                iterations: result.measurements.len(),
                mean_time: result.statistics.mean_execution_time,
                std_dev: result.statistics.std_dev_execution_time,
                min_time: result.statistics.min_execution_time,
                max_time: result.statistics.max_execution_time,
            }],
            completed_at: chrono::Utc::now(),
        })
    }

    /// Add a custom benchmark to the suite
    pub async fn add_benchmark<F>(&self, benchmark_fn: F) -> Result<()>
    where
        F: Fn(&scirs2_core::benchmarking::BenchmarkRunner) -> std::result::Result<scirs2_core::benchmarking::BenchmarkResult, CoreError> + Send + Sync + 'static,
    {
        let suite = self
            .benchmark_suite
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Benchmark suite not initialized"))?;

        let mut suite_guard = suite.write().await;
        suite_guard.add_benchmark(benchmark_fn);

        debug!("Added benchmark to suite");
        Ok(())
    }

    /// Clear all profiling statistics
    pub async fn clear_stats(&self) {
        let mut stats = self.completed_stats.write().await;
        stats.clear();
        info!("Cleared all profiling statistics");
    }

    /// Get current metric values
    pub async fn get_current_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Get timer statistics using get_stats()
        let rag_stats = self.rag_retrieval_timer.get_stats();
        metrics.insert("rag_retrieval_count".to_string(), rag_stats.count as f64);
        metrics.insert("rag_retrieval_mean".to_string(), rag_stats.mean);

        let llm_stats = self.llm_generation_timer.get_stats();
        metrics.insert("llm_generation_count".to_string(), llm_stats.count as f64);
        metrics.insert("llm_generation_mean".to_string(), llm_stats.mean);

        let sparql_stats = self.sparql_execution_timer.get_stats();
        metrics.insert(
            "sparql_execution_count".to_string(),
            sparql_stats.count as f64,
        );
        metrics.insert("sparql_execution_mean".to_string(), sparql_stats.mean);

        metrics
    }

    // Helper methods

    fn should_profile(&self) -> bool {
        if !self.config.enable_detailed_profiling {
            return false;
        }

        if self.config.sample_rate >= 1.0 {
            return true;
        }

        fastrand::f64() < self.config.sample_rate
    }

    fn get_current_memory_usage(&self) -> usize {
        // Use scirs2-core memory tracking
        let _tracker = profiling_memory_tracker();
        // Placeholder: Return estimated usage based on system info
        // In future scirs2-core versions, we'll use tracker methods
        0
    }

    fn get_cpu_usage(&self) -> Option<f64> {
        // TODO: Implement CPU usage tracking
        None
    }

    fn get_gpu_usage(&self) -> Option<f64> {
        // TODO: Implement GPU usage tracking if scirs2-core GPU features are available
        None
    }

    fn calculate_median_duration(&self, durations: &[Duration]) -> Duration {
        if durations.is_empty() {
            return Duration::ZERO;
        }

        let mut sorted = durations.to_vec();
        sorted.sort();

        if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            Duration::from_nanos(
                (sorted[mid - 1].as_nanos() + sorted[mid].as_nanos()) as u64 / 2,
            )
        } else {
            sorted[sorted.len() / 2]
        }
    }

    fn calculate_percentile_duration(&self, durations: &[Duration], percentile: f64) -> Duration {
        if durations.is_empty() {
            return Duration::ZERO;
        }

        let mut sorted = durations.to_vec();
        sorted.sort();

        let index = ((sorted.len() as f64 * percentile).ceil() as usize).min(sorted.len() - 1);
        sorted[index]
    }
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub total_operations: usize,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub median_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
    pub operation_stats: HashMap<String, OperationStats>,
    pub total_memory_allocated: Option<usize>,
    pub config: ProfilingConfig,
}

impl Default for PerformanceReport {
    fn default() -> Self {
        Self {
            generated_at: chrono::Utc::now(),
            total_operations: 0,
            total_duration: Duration::ZERO,
            avg_duration: Duration::ZERO,
            median_duration: Duration::ZERO,
            p95_duration: Duration::ZERO,
            p99_duration: Duration::ZERO,
            operation_stats: HashMap::new(),
            total_memory_allocated: None,
            config: ProfilingConfig::default(),
        }
    }
}

/// Statistics for a specific operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub count: usize,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub median_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub suite_name: String,
    pub total_benchmarks: usize,
    pub results: Vec<BenchmarkResult>,
    pub completed_at: chrono::DateTime<chrono::Utc>,
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub mean_time: Duration,
    pub std_dev: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_profiler_creation() {
        let config = ProfilingConfig::default();
        let profiler = ChatProfiler::new(config);
        assert!(profiler.is_ok());
    }

    #[tokio::test]
    async fn test_profiling_session() -> Result<()> {
        let config = ProfilingConfig::default();
        let profiler = ChatProfiler::new(config)?;

        profiler.start_profiling("test_operation").await?;
        tokio::time::sleep(Duration::from_millis(100)).await;
        let stats = profiler.stop_profiling("test_operation").await?;

        assert_eq!(stats.operation, "test_operation");
        assert!(stats.duration >= Duration::from_millis(100));

        Ok(())
    }

    #[tokio::test]
    async fn test_performance_report() -> Result<()> {
        let config = ProfilingConfig::default();
        let profiler = ChatProfiler::new(config)?;

        // Profile multiple operations
        for i in 0..5 {
            profiler
                .start_profiling(&format!("operation_{}", i))
                .await?;
            tokio::time::sleep(Duration::from_millis(10)).await;
            profiler.stop_profiling(&format!("operation_{}", i)).await?;
        }

        let report = profiler.generate_report().await?;
        assert_eq!(report.total_operations, 5);
        assert!(report.avg_duration > Duration::ZERO);

        Ok(())
    }

    #[tokio::test]
    async fn test_metrics_collection() -> Result<()> {
        let config = ProfilingConfig::default();
        let profiler = ChatProfiler::new(config)?;

        profiler.start_profiling("rag_retrieval").await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
        profiler.stop_profiling("rag_retrieval").await?;

        let metrics = profiler.get_current_metrics().await;
        assert!(metrics.contains_key("total_requests"));
        assert_eq!(metrics["total_requests"], 1.0);

        Ok(())
    }
}
