//! Performance Monitoring and Profiling Tools for Oxirs CLI
//!
//! This module provides comprehensive performance monitoring, profiling, and benchmarking
//! capabilities for the Oxirs CLI toolkit, enabling real-time performance analysis,
//! resource utilization tracking, and benchmark comparisons.

use crate::cli::{error::CliError, output::OutputFormatter};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use sysinfo::System;
use tokio::time::interval;
use tracing::{debug, info, warn};

/// Performance monitoring and profiling engine
pub struct PerformanceMonitor {
    system: Arc<Mutex<System>>,
    metrics: Arc<Mutex<PerformanceMetrics>>,
    active_sessions: Arc<Mutex<HashMap<String, ProfilingSession>>>,
    counters: Arc<Mutex<HashMap<String, AtomicU64>>>,
    config: MonitoringConfig,
}

/// Performance metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub memory_total: u64,
    pub disk_io_read: u64,
    pub disk_io_write: u64,
    pub network_rx: u64,
    pub network_tx: u64,
    pub load_average: Vec<f64>,
    pub uptime: Duration,
    pub timestamp: SystemTime,
}

/// Profiling session for tracking operation performance
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    pub session_id: String,
    pub operation_name: String,
    pub start_time: Instant,
    pub start_metrics: PerformanceMetrics,
    pub checkpoints: Vec<ProfileCheckpoint>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Performance checkpoint during operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCheckpoint {
    pub name: String,
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub duration_from_start: Duration,
    pub metrics: PerformanceMetrics,
    pub memory_delta: i64,
    pub custom_data: HashMap<String, String>,
}

impl Default for ProfileCheckpoint {
    fn default() -> Self {
        Self {
            name: String::new(),
            timestamp: Instant::now(),
            duration_from_start: Duration::from_secs(0),
            metrics: PerformanceMetrics::default(),
            memory_delta: 0,
            custom_data: HashMap::new(),
        }
    }
}

/// Benchmark comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub baseline_name: String,
    pub current_name: String,
    pub performance_ratio: f64,
    pub memory_ratio: f64,
    pub time_ratio: f64,
    pub improvement_summary: String,
    pub detailed_metrics: HashMap<String, MetricComparison>,
}

/// Individual metric comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub baseline_value: f64,
    pub current_value: f64,
    pub ratio: f64,
    pub improvement_percentage: f64,
    pub significance: ComparisonSignificance,
}

/// Significance of performance difference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonSignificance {
    Negligible,  // < 5% difference
    Minor,       // 5-15% difference
    Moderate,    // 15-30% difference
    Significant, // 30-50% difference
    Major,       // > 50% difference
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_continuous_monitoring: bool,
    pub sampling_interval_ms: u64,
    pub memory_tracking: bool,
    pub cpu_tracking: bool,
    pub io_tracking: bool,
    pub network_tracking: bool,
    pub auto_profiling: bool,
    pub profile_threshold_ms: u64,
    pub max_sessions: usize,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_continuous_monitoring: true,
            sampling_interval_ms: 1000,
            memory_tracking: true,
            cpu_tracking: true,
            io_tracking: true,
            network_tracking: true,
            auto_profiling: false,
            profile_threshold_ms: 100,
            max_sessions: 100,
        }
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor with configuration
    pub fn new(config: MonitoringConfig) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self {
            system: Arc::new(Mutex::new(system)),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            active_sessions: Arc::new(Mutex::new(HashMap::new())),
            counters: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Start continuous performance monitoring
    pub async fn start_monitoring(&self) -> Result<(), CliError> {
        if !self.config.enable_continuous_monitoring {
            return Ok(());
        }

        let system: Arc<Mutex<System>> = Arc::clone(&self.system);
        let metrics = Arc::clone(&self.metrics);
        let interval_duration = Duration::from_millis(self.config.sampling_interval_ms);

        tokio::spawn(async move {
            let mut interval = interval(interval_duration);

            loop {
                interval.tick().await;

                // Update system information
                if let Ok(mut sys) = system.lock() {
                    sys.refresh_all();

                    // Collect current metrics
                    let current_metrics = PerformanceMetrics {
                        cpu_usage: sys.global_cpu_info().cpu_usage(),
                        memory_usage: sys.used_memory(),
                        memory_total: sys.total_memory(),
                        disk_io_read: Self::calculate_total_disk_read(&sys),
                        disk_io_write: Self::calculate_total_disk_write(&sys),
                        network_rx: Self::calculate_total_network_rx(&sys),
                        network_tx: Self::calculate_total_network_tx(&sys),
                        load_average: vec![0.0, 0.0, 0.0], // Load average not available in newer sysinfo
                        uptime: Duration::from_secs(System::uptime()),
                        timestamp: SystemTime::now(),
                    };

                    // Update metrics
                    if let Ok(mut metrics_lock) = metrics.lock() {
                        *metrics_lock = current_metrics;
                    }
                }
            }
        });

        info!(
            "Performance monitoring started with {}ms interval",
            self.config.sampling_interval_ms
        );
        Ok(())
    }

    /// Start a new profiling session
    pub fn start_profiling(&self, operation_name: &str) -> Result<String, CliError> {
        let session_id = uuid::Uuid::new_v4().to_string();
        let current_metrics = self.get_current_metrics()?;

        let session = ProfilingSession {
            session_id: session_id.clone(),
            operation_name: operation_name.to_string(),
            start_time: Instant::now(),
            start_metrics: current_metrics,
            checkpoints: Vec::new(),
            custom_metrics: HashMap::new(),
        };

        if let Ok(mut sessions) = self.active_sessions.lock() {
            // Cleanup old sessions if we exceed max limit
            if sessions.len() >= self.config.max_sessions {
                let oldest_key = sessions.keys().next().cloned();
                if let Some(key) = oldest_key {
                    sessions.remove(&key);
                    warn!("Removed oldest profiling session due to limit");
                }
            }

            sessions.insert(session_id.clone(), session);
        }

        debug!(
            "Started profiling session '{}' for operation '{}'",
            session_id, operation_name
        );
        Ok(session_id)
    }

    /// Add a checkpoint to an active profiling session
    pub fn add_checkpoint(&self, session_id: &str, checkpoint_name: &str) -> Result<(), CliError> {
        if let Ok(mut sessions) = self.active_sessions.lock() {
            if let Some(session) = sessions.get_mut(session_id) {
                let current_metrics = self.get_current_metrics()?;
                let duration_from_start = session.start_time.elapsed();

                let memory_delta =
                    current_metrics.memory_usage as i64 - session.start_metrics.memory_usage as i64;

                let checkpoint = ProfileCheckpoint {
                    name: checkpoint_name.to_string(),
                    timestamp: Instant::now(),
                    duration_from_start,
                    metrics: current_metrics,
                    memory_delta,
                    custom_data: HashMap::new(),
                };

                session.checkpoints.push(checkpoint);
                debug!(
                    "Added checkpoint '{}' to session '{}'",
                    checkpoint_name, session_id
                );
                return Ok(());
            }
        }

        Err(CliError::profile_error(format!(
            "Profiling session '{session_id}' not found"
        )))
    }

    /// Finish profiling session and return results
    pub fn finish_profiling(&self, session_id: &str) -> Result<ProfilingResult, CliError> {
        if let Ok(mut sessions) = self.active_sessions.lock() {
            if let Some(session) = sessions.remove(session_id) {
                let _end_time = Instant::now();
                let total_duration = session.start_time.elapsed();
                let end_metrics = self.get_current_metrics()?;

                let performance_summary =
                    self.generate_performance_summary(&session, &end_metrics, total_duration)?;

                let result = ProfilingResult {
                    session_id: session.session_id.clone(),
                    operation_name: session.operation_name.clone(),
                    total_duration,
                    start_metrics: session.start_metrics.clone(),
                    end_metrics: end_metrics.clone(),
                    checkpoints: session.checkpoints.clone(),
                    custom_metrics: session.custom_metrics.clone(),
                    performance_summary,
                };

                info!(
                    "Completed profiling session '{}' - Duration: {:.2}s",
                    session_id,
                    total_duration.as_secs_f64()
                );
                return Ok(result);
            }
        }

        Err(CliError::profile_error(format!(
            "Profiling session '{session_id}' not found"
        )))
    }

    /// Get current system performance metrics
    pub fn get_current_metrics(&self) -> Result<PerformanceMetrics, CliError> {
        match self.metrics.lock() {
            Ok(metrics) => Ok(metrics.clone()),
            _ => Err(CliError::profile_error(
                "Failed to access performance metrics".to_string(),
            )),
        }
    }

    /// Compare performance between two benchmark results
    pub fn compare_benchmarks(
        &self,
        baseline: &ProfilingResult,
        current: &ProfilingResult,
    ) -> Result<BenchmarkComparison, CliError> {
        let time_ratio =
            current.total_duration.as_secs_f64() / baseline.total_duration.as_secs_f64();
        let memory_ratio =
            current.end_metrics.memory_usage as f64 / baseline.end_metrics.memory_usage as f64;

        // Calculate overall performance ratio (lower is better)
        let performance_ratio = (time_ratio + memory_ratio) / 2.0;

        let improvement_summary = if performance_ratio < 0.95 {
            format!(
                "🚀 Significant improvement: {:.1}% faster overall",
                (1.0 - performance_ratio) * 100.0
            )
        } else if performance_ratio < 1.05 {
            "⚖️ Performance is comparable".to_string()
        } else {
            format!(
                "📉 Performance regression: {:.1}% slower overall",
                (performance_ratio - 1.0) * 100.0
            )
        };

        let mut detailed_metrics = HashMap::new();

        // Time comparison
        detailed_metrics.insert(
            "execution_time".to_string(),
            MetricComparison {
                baseline_value: baseline.total_duration.as_secs_f64(),
                current_value: current.total_duration.as_secs_f64(),
                ratio: time_ratio,
                improvement_percentage: (1.0 - time_ratio) * 100.0,
                significance: Self::calculate_significance(time_ratio),
            },
        );

        // Memory comparison
        detailed_metrics.insert(
            "memory_usage".to_string(),
            MetricComparison {
                baseline_value: baseline.end_metrics.memory_usage as f64,
                current_value: current.end_metrics.memory_usage as f64,
                ratio: memory_ratio,
                improvement_percentage: (1.0 - memory_ratio) * 100.0,
                significance: Self::calculate_significance(memory_ratio),
            },
        );

        Ok(BenchmarkComparison {
            baseline_name: baseline.operation_name.clone(),
            current_name: current.operation_name.clone(),
            performance_ratio,
            memory_ratio,
            time_ratio,
            improvement_summary,
            detailed_metrics,
        })
    }

    /// Increment a performance counter
    pub fn increment_counter(&self, counter_name: &str, value: u64) -> Result<(), CliError> {
        match self.counters.lock() {
            Ok(mut counters) => {
                counters
                    .entry(counter_name.to_string())
                    .or_insert_with(|| AtomicU64::new(0))
                    .fetch_add(value, Ordering::Relaxed);
                Ok(())
            }
            _ => Err(CliError::profile_error(
                "Failed to access performance counters".to_string(),
            )),
        }
    }

    /// Get performance counter value
    pub fn get_counter(&self, counter_name: &str) -> Result<u64, CliError> {
        match self.counters.lock() {
            Ok(counters) => Ok(counters
                .get(counter_name)
                .map(|counter| counter.load(Ordering::Relaxed))
                .unwrap_or(0)),
            _ => Err(CliError::profile_error(
                "Failed to access performance counters".to_string(),
            )),
        }
    }

    /// Generate comprehensive performance report
    pub fn generate_performance_report(&self) -> Result<PerformanceReport, CliError> {
        let current_metrics = self.get_current_metrics()?;
        let active_sessions = match self.active_sessions.lock() {
            Ok(sessions) => sessions.len(),
            _ => 0,
        };

        let counters_snapshot = match self.counters.lock() {
            Ok(counters) => counters
                .iter()
                .map(|(name, counter)| (name.clone(), counter.load(Ordering::Relaxed)))
                .collect(),
            _ => HashMap::new(),
        };

        Ok(PerformanceReport {
            timestamp: SystemTime::now(),
            current_metrics,
            active_profiling_sessions: active_sessions,
            performance_counters: counters_snapshot,
            system_health: self.assess_system_health()?,
            recommendations: self.generate_performance_recommendations()?,
        })
    }

    // Private helper methods

    fn calculate_total_disk_read(_system: &System) -> u64 {
        // Disk I/O monitoring not available in this sysinfo version
        0
    }

    fn calculate_total_disk_write(_system: &System) -> u64 {
        // Disk I/O monitoring not available in this sysinfo version
        0
    }

    fn calculate_total_network_rx(_system: &System) -> u64 {
        // Network monitoring not available in this sysinfo version
        0
    }

    fn calculate_total_network_tx(_system: &System) -> u64 {
        // Network monitoring not available in this sysinfo version
        0
    }

    fn generate_performance_summary(
        &self,
        session: &ProfilingSession,
        end_metrics: &PerformanceMetrics,
        total_duration: Duration,
    ) -> Result<PerformanceSummary, CliError> {
        let memory_delta =
            end_metrics.memory_usage as i64 - session.start_metrics.memory_usage as i64;
        let cpu_avg = (session.start_metrics.cpu_usage + end_metrics.cpu_usage) / 2.0;

        Ok(PerformanceSummary {
            total_execution_time: total_duration,
            memory_delta_bytes: memory_delta,
            average_cpu_usage: cpu_avg,
            checkpoints_count: session.checkpoints.len(),
            peak_memory: session
                .checkpoints
                .iter()
                .map(|cp| cp.metrics.memory_usage)
                .max()
                .unwrap_or(end_metrics.memory_usage),
            efficiency_score: self.calculate_efficiency_score(
                total_duration,
                memory_delta,
                cpu_avg,
            ),
        })
    }

    fn calculate_efficiency_score(
        &self,
        duration: Duration,
        memory_delta: i64,
        cpu_avg: f32,
    ) -> f64 {
        // Simple efficiency scoring (0-100, higher is better)
        let time_score = (1.0 / duration.as_secs_f64()).min(1.0) * 40.0;
        let memory_score = if memory_delta <= 0 {
            30.0
        } else {
            (1.0 / (memory_delta as f64 / 1_000_000.0)).min(1.0) * 30.0
        };
        let cpu_score = ((100.0 - cpu_avg as f64) / 100.0) * 30.0;

        time_score + memory_score + cpu_score
    }

    fn calculate_significance(ratio: f64) -> ComparisonSignificance {
        let diff_percentage = (ratio - 1.0).abs() * 100.0;

        if diff_percentage < 5.0 {
            ComparisonSignificance::Negligible
        } else if diff_percentage < 15.0 {
            ComparisonSignificance::Minor
        } else if diff_percentage < 30.0 {
            ComparisonSignificance::Moderate
        } else if diff_percentage < 50.0 {
            ComparisonSignificance::Significant
        } else {
            ComparisonSignificance::Major
        }
    }

    fn assess_system_health(&self) -> Result<SystemHealth, CliError> {
        let metrics = self.get_current_metrics()?;

        let memory_usage_percentage =
            (metrics.memory_usage as f64 / metrics.memory_total as f64) * 100.0;
        let cpu_usage = metrics.cpu_usage as f64;

        let health_status = if cpu_usage > 90.0 || memory_usage_percentage > 95.0 {
            HealthStatus::Critical
        } else if cpu_usage > 70.0 || memory_usage_percentage > 80.0 {
            HealthStatus::Warning
        } else if cpu_usage > 50.0 || memory_usage_percentage > 60.0 {
            HealthStatus::Moderate
        } else {
            HealthStatus::Healthy
        };

        Ok(SystemHealth {
            status: health_status,
            cpu_usage_percentage: cpu_usage,
            memory_usage_percentage,
            disk_space_issues: false, // Simplified for now
            network_issues: false,    // Simplified for now
            recommendations: self
                .generate_health_recommendations(cpu_usage, memory_usage_percentage)?,
        })
    }

    fn generate_health_recommendations(
        &self,
        cpu_usage: f64,
        memory_usage: f64,
    ) -> Result<Vec<String>, CliError> {
        let mut recommendations = Vec::new();

        if cpu_usage > 80.0 {
            recommendations.push(
                "High CPU usage detected. Consider reducing concurrent operations.".to_string(),
            );
        }

        if memory_usage > 85.0 {
            recommendations.push("High memory usage detected. Consider increasing available memory or optimizing data structures.".to_string());
        }

        if cpu_usage < 20.0 && memory_usage < 30.0 {
            recommendations.push(
                "System resources are underutilized. Consider increasing parallelism.".to_string(),
            );
        }

        Ok(recommendations)
    }

    fn generate_performance_recommendations(&self) -> Result<Vec<String>, CliError> {
        let recommendations = vec![
            "Monitor performance regularly to identify optimization opportunities.".to_string(),
            "Use profiling sessions for performance-critical operations.".to_string(),
            "Compare benchmarks to track performance regressions.".to_string(),
        ];

        Ok(recommendations)
    }
}

/// Profiling result with comprehensive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingResult {
    pub session_id: String,
    pub operation_name: String,
    pub total_duration: Duration,
    pub start_metrics: PerformanceMetrics,
    pub end_metrics: PerformanceMetrics,
    pub checkpoints: Vec<ProfileCheckpoint>,
    pub custom_metrics: HashMap<String, f64>,
    pub performance_summary: PerformanceSummary,
}

/// Performance summary for a profiling session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_execution_time: Duration,
    pub memory_delta_bytes: i64,
    pub average_cpu_usage: f32,
    pub checkpoints_count: usize,
    pub peak_memory: u64,
    pub efficiency_score: f64,
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: SystemTime,
    pub current_metrics: PerformanceMetrics,
    pub active_profiling_sessions: usize,
    pub performance_counters: HashMap<String, u64>,
    pub system_health: SystemHealth,
    pub recommendations: Vec<String>,
}

/// System health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub status: HealthStatus,
    pub cpu_usage_percentage: f64,
    pub memory_usage_percentage: f64,
    pub disk_space_issues: bool,
    pub network_issues: bool,
    pub recommendations: Vec<String>,
}

/// Health status levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Healthy,
    Moderate,
    Warning,
    Critical,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            memory_total: 0,
            disk_io_read: 0,
            disk_io_write: 0,
            network_rx: 0,
            network_tx: 0,
            load_average: Vec::new(),
            uptime: Duration::from_secs(0),
            timestamp: UNIX_EPOCH,
        }
    }
}

/// CLI command for performance monitoring
pub async fn monitor_performance_command(
    duration_secs: Option<u64>,
    output_format: Option<String>,
    continuous: bool,
) -> Result<(), CliError> {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    monitor.start_monitoring().await?;

    if continuous {
        info!("Starting continuous performance monitoring (Ctrl+C to stop)");

        // Run until interrupted
        tokio::signal::ctrl_c().await.map_err(|e| {
            CliError::profile_error(format!("Failed to wait for interrupt signal: {e}"))
        })?;

        println!("Performance monitoring stopped.");
    } else {
        let duration = Duration::from_secs(duration_secs.unwrap_or(10));
        info!("Monitoring performance for {} seconds", duration.as_secs());

        tokio::time::sleep(duration).await;

        let report = monitor.generate_performance_report()?;
        let formatter = OutputFormatter::new(output_format.as_deref().unwrap_or("table"));
        formatter.print_performance_report(&report)?;
    }

    Ok(())
}

/// CLI command for profiling operations
pub async fn profile_operation_command(
    operation_name: String,
    _command_args: Vec<String>,
    output_format: Option<String>,
) -> Result<(), CliError> {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    monitor.start_monitoring().await?;

    let session_id = monitor.start_profiling(&operation_name)?;

    // Execute the operation (simplified - would integrate with actual command execution)
    info!("Profiling operation: {}", operation_name);
    monitor.add_checkpoint(&session_id, "operation_start")?;

    // Simulate operation execution
    tokio::time::sleep(Duration::from_millis(100)).await;

    monitor.add_checkpoint(&session_id, "operation_complete")?;

    let result = monitor.finish_profiling(&session_id)?;

    let formatter = OutputFormatter::new(output_format.as_deref().unwrap_or("table"));
    formatter.print_profiling_result(&result)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let metrics = monitor.get_current_metrics();
        assert!(metrics.is_ok());
    }

    #[tokio::test]
    async fn test_profiling_session() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let session_id = monitor.start_profiling("test_operation").unwrap();

        tokio::time::sleep(Duration::from_millis(10)).await;

        monitor.add_checkpoint(&session_id, "checkpoint1").unwrap();

        let result = monitor.finish_profiling(&session_id).unwrap();

        assert_eq!(result.operation_name, "test_operation");
        assert_eq!(result.checkpoints.len(), 1);
        assert!(result.total_duration > Duration::from_millis(5));
    }

    #[test]
    fn test_significance_calculation() {
        assert!(matches!(
            PerformanceMonitor::calculate_significance(1.02),
            ComparisonSignificance::Negligible
        ));

        assert!(matches!(
            PerformanceMonitor::calculate_significance(1.10),
            ComparisonSignificance::Minor
        ));

        assert!(matches!(
            PerformanceMonitor::calculate_significance(1.60),
            ComparisonSignificance::Major
        ));
    }

    #[tokio::test]
    async fn test_performance_counters() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        monitor.increment_counter("test_counter", 5).unwrap();
        monitor.increment_counter("test_counter", 3).unwrap();

        let value = monitor.get_counter("test_counter").unwrap();
        assert_eq!(value, 8);
    }
}
