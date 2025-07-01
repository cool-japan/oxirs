//! # Advanced Metrics Collection and Monitoring
//!
//! Provides comprehensive metrics collection, performance monitoring,
//! and operational insights for TDB storage operations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::query_optimizer::{IndexType, PatternType};

/// Comprehensive system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU utilization percentage (0.0-100.0)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Available memory in bytes
    pub memory_available: u64,
    /// Disk I/O read rate (bytes/second)
    pub disk_read_rate: u64,
    /// Disk I/O write rate (bytes/second)
    pub disk_write_rate: u64,
    /// Network I/O rate (bytes/second)
    pub network_io_rate: u64,
    /// Number of active threads
    pub active_threads: u32,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            memory_available: 0,
            disk_read_rate: 0,
            disk_write_rate: 0,
            network_io_rate: 0,
            active_threads: 0,
            timestamp: SystemTime::now(),
        }
    }
}

/// Query performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Total number of queries executed
    pub total_queries: u64,
    /// Average query execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// P50 query execution time in milliseconds
    pub p50_execution_time_ms: f64,
    /// P95 query execution time in milliseconds
    pub p95_execution_time_ms: f64,
    /// P99 query execution time in milliseconds
    pub p99_execution_time_ms: f64,
    /// Maximum query execution time in milliseconds
    pub max_execution_time_ms: f64,
    /// Average result set size
    pub avg_result_size: f64,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f64,
    /// Index usage distribution
    pub index_usage: HashMap<IndexType, u64>,
    /// Pattern type distribution
    pub pattern_type_usage: HashMap<PatternType, u64>,
    /// Queries per second over last minute
    pub queries_per_second: f64,
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            avg_execution_time_ms: 0.0,
            p50_execution_time_ms: 0.0,
            p95_execution_time_ms: 0.0,
            p99_execution_time_ms: 0.0,
            max_execution_time_ms: 0.0,
            avg_result_size: 0.0,
            cache_hit_rate: 0.0,
            index_usage: HashMap::new(),
            pattern_type_usage: HashMap::new(),
            queries_per_second: 0.0,
        }
    }
}

/// Storage-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    /// Total number of triples stored
    pub total_triples: u64,
    /// Database size on disk in bytes
    pub database_size_bytes: u64,
    /// Index size on disk in bytes
    pub index_size_bytes: u64,
    /// Compression ratio (0.0-1.0, lower is better compression)
    pub compression_ratio: f64,
    /// Number of active transactions
    pub active_transactions: u32,
    /// Number of completed transactions
    pub completed_transactions: u64,
    /// Number of failed transactions
    pub failed_transactions: u64,
    /// WAL (Write-Ahead Log) size in bytes
    pub wal_size_bytes: u64,
    /// Number of checkpoints performed
    pub checkpoint_count: u64,
    /// Average checkpoint duration in milliseconds
    pub avg_checkpoint_duration_ms: f64,
    /// Buffer pool hit rate (0.0-1.0)
    pub buffer_pool_hit_rate: f64,
    /// Number of dirty pages in buffer pool
    pub dirty_pages: u32,
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self {
            total_triples: 0,
            database_size_bytes: 0,
            index_size_bytes: 0,
            compression_ratio: 1.0,
            active_transactions: 0,
            completed_transactions: 0,
            failed_transactions: 0,
            wal_size_bytes: 0,
            checkpoint_count: 0,
            avg_checkpoint_duration_ms: 0.0,
            buffer_pool_hit_rate: 0.0,
            dirty_pages: 0,
        }
    }
}

/// Error and alert metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Total number of errors
    pub total_errors: u64,
    /// Error rate per minute
    pub error_rate_per_minute: f64,
    /// Error distribution by type
    pub error_types: HashMap<String, u64>,
    /// Number of warnings
    pub warning_count: u64,
    /// Number of critical alerts
    pub critical_alerts: u64,
    /// Last error timestamp
    pub last_error_time: Option<SystemTime>,
    /// Average time to recover from errors (milliseconds)
    pub avg_recovery_time_ms: f64,
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            error_rate_per_minute: 0.0,
            error_types: HashMap::new(),
            warning_count: 0,
            critical_alerts: 0,
            last_error_time: None,
            avg_recovery_time_ms: 0.0,
        }
    }
}

/// Comprehensive metrics aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveMetrics {
    pub system: SystemMetrics,
    pub query: QueryMetrics,
    pub storage: StorageMetrics,
    pub errors: ErrorMetrics,
    pub timestamp: SystemTime,
}

impl Default for ComprehensiveMetrics {
    fn default() -> Self {
        Self {
            system: SystemMetrics::default(),
            query: QueryMetrics::default(),
            storage: StorageMetrics::default(),
            errors: ErrorMetrics::default(),
            timestamp: SystemTime::now(),
        }
    }
}

/// Time-series data point for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsDataPoint {
    pub timestamp: SystemTime,
    pub value: f64,
    pub labels: HashMap<String, String>,
}

/// Time-series metrics storage
#[derive(Debug, Clone)]
pub struct TimeSeriesMetrics {
    data: Arc<Mutex<Vec<MetricsDataPoint>>>,
    max_points: usize,
}

impl TimeSeriesMetrics {
    /// Create new time-series metrics with maximum number of data points
    pub fn new(max_points: usize) -> Self {
        Self {
            data: Arc::new(Mutex::new(Vec::new())),
            max_points,
        }
    }

    /// Add a new data point
    pub fn add_point(&self, value: f64, labels: HashMap<String, String>) -> Result<()> {
        let point = MetricsDataPoint {
            timestamp: SystemTime::now(),
            value,
            labels,
        };

        let mut data = self
            .data
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire lock on time series data"))?;

        data.push(point);

        // Maintain max points limit
        if data.len() > self.max_points {
            let data_len = data.len();
            data.drain(0..data_len - self.max_points);
        }

        Ok(())
    }

    /// Get all data points
    pub fn get_data(&self) -> Result<Vec<MetricsDataPoint>> {
        let data = self
            .data
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire lock on time series data"))?;
        Ok(data.clone())
    }

    /// Get data points within a time range
    pub fn get_data_range(
        &self,
        start: SystemTime,
        end: SystemTime,
    ) -> Result<Vec<MetricsDataPoint>> {
        let data = self
            .data
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire lock on time series data"))?;

        let filtered: Vec<_> = data
            .iter()
            .filter(|point| point.timestamp >= start && point.timestamp <= end)
            .cloned()
            .collect();

        Ok(filtered)
    }

    /// Calculate statistics for a metric over time
    pub fn calculate_stats(&self, duration: Duration) -> Result<MetricsStatistics> {
        let now = SystemTime::now();
        let start = now.checked_sub(duration).unwrap_or(UNIX_EPOCH);
        let data = self.get_data_range(start, now)?;

        if data.is_empty() {
            return Ok(MetricsStatistics::default());
        }

        let values: Vec<f64> = data.iter().map(|p| p.value).collect();
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = values.iter().sum();
        let count = values.len();
        let mean = sum / count as f64;

        let min = sorted_values.first().cloned().unwrap_or(0.0);
        let max = sorted_values.last().cloned().unwrap_or(0.0);

        let p50_index = (count as f64 * 0.5) as usize;
        let p95_index = (count as f64 * 0.95) as usize;
        let p99_index = (count as f64 * 0.99) as usize;

        let p50 = sorted_values.get(p50_index).cloned().unwrap_or(0.0);
        let p95 = sorted_values.get(p95_index).cloned().unwrap_or(0.0);
        let p99 = sorted_values.get(p99_index).cloned().unwrap_or(0.0);

        // Calculate standard deviation
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        Ok(MetricsStatistics {
            count: count as u64,
            sum,
            mean,
            min,
            max,
            p50,
            p95,
            p99,
            std_dev,
            duration,
        })
    }
}

/// Statistical summary of metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsStatistics {
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub std_dev: f64,
    pub duration: Duration,
}

impl Default for MetricsStatistics {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            std_dev: 0.0,
            duration: Duration::from_secs(0),
        }
    }
}

/// Advanced metrics collector with time-series support
pub struct MetricsCollector {
    query_times: TimeSeriesMetrics,
    error_rates: TimeSeriesMetrics,
    memory_usage: TimeSeriesMetrics,
    cpu_usage: TimeSeriesMetrics,
    throughput: TimeSeriesMetrics,
    current_metrics: Arc<Mutex<ComprehensiveMetrics>>,
    collection_interval: Duration,
    last_collection: Arc<Mutex<Instant>>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(collection_interval: Duration) -> Self {
        Self {
            query_times: TimeSeriesMetrics::new(10_000),
            error_rates: TimeSeriesMetrics::new(10_000),
            memory_usage: TimeSeriesMetrics::new(10_000),
            cpu_usage: TimeSeriesMetrics::new(10_000),
            throughput: TimeSeriesMetrics::new(10_000),
            current_metrics: Arc::new(Mutex::new(ComprehensiveMetrics::default())),
            collection_interval,
            last_collection: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Record query execution time
    pub fn record_query_time(
        &self,
        duration: Duration,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        self.query_times
            .add_point(duration.as_secs_f64() * 1000.0, labels)
    }

    /// Record error occurrence
    pub fn record_error(&self, error_type: &str) -> Result<()> {
        let mut labels = HashMap::new();
        labels.insert("error_type".to_string(), error_type.to_string());
        self.error_rates.add_point(1.0, labels)
    }

    /// Record system metrics
    pub fn record_system_metrics(&self, metrics: SystemMetrics) -> Result<()> {
        let mut labels = HashMap::new();
        labels.insert("metric_type".to_string(), "system".to_string());

        self.memory_usage
            .add_point(metrics.memory_usage as f64, labels.clone())?;
        self.cpu_usage.add_point(metrics.cpu_usage, labels)?;

        // Update current metrics
        let mut current = self
            .current_metrics
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire lock on current metrics"))?;
        current.system = metrics;

        Ok(())
    }

    /// Record throughput metrics
    pub fn record_throughput(
        &self,
        operations_per_second: f64,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        self.throughput.add_point(operations_per_second, labels)
    }

    /// Get current metrics snapshot
    pub fn get_current_metrics(&self) -> Result<ComprehensiveMetrics> {
        let metrics = self
            .current_metrics
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire lock on current metrics"))?;
        Ok(metrics.clone())
    }

    /// Get query performance statistics
    pub fn get_query_stats(&self, duration: Duration) -> Result<MetricsStatistics> {
        self.query_times.calculate_stats(duration)
    }

    /// Get error rate statistics
    pub fn get_error_stats(&self, duration: Duration) -> Result<MetricsStatistics> {
        self.error_rates.calculate_stats(duration)
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self, duration: Duration) -> Result<MetricsStatistics> {
        self.memory_usage.calculate_stats(duration)
    }

    /// Get CPU usage statistics
    pub fn get_cpu_stats(&self, duration: Duration) -> Result<MetricsStatistics> {
        self.cpu_usage.calculate_stats(duration)
    }

    /// Get throughput statistics
    pub fn get_throughput_stats(&self, duration: Duration) -> Result<MetricsStatistics> {
        self.throughput.calculate_stats(duration)
    }

    /// Generate comprehensive metrics report
    pub fn generate_report(&self, duration: Duration) -> Result<MetricsReport> {
        let query_stats = self.get_query_stats(duration)?;
        let error_stats = self.get_error_stats(duration)?;
        let memory_stats = self.get_memory_stats(duration)?;
        let cpu_stats = self.get_cpu_stats(duration)?;
        let throughput_stats = self.get_throughput_stats(duration)?;
        let current_metrics = self.get_current_metrics()?;

        Ok(MetricsReport {
            query_performance: query_stats,
            error_rates: error_stats,
            memory_usage: memory_stats,
            cpu_usage: cpu_stats,
            throughput: throughput_stats,
            current_snapshot: current_metrics,
            report_duration: duration,
            generated_at: SystemTime::now(),
        })
    }

    /// Check if metrics need to be collected based on interval
    pub fn should_collect(&self) -> bool {
        if let Ok(last) = self.last_collection.lock() {
            last.elapsed() >= self.collection_interval
        } else {
            true
        }
    }

    /// Update last collection time
    pub fn mark_collected(&self) -> Result<()> {
        let mut last = self
            .last_collection
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire lock on last collection time"))?;
        *last = Instant::now();
        Ok(())
    }
}

/// Comprehensive metrics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub query_performance: MetricsStatistics,
    pub error_rates: MetricsStatistics,
    pub memory_usage: MetricsStatistics,
    pub cpu_usage: MetricsStatistics,
    pub throughput: MetricsStatistics,
    pub current_snapshot: ComprehensiveMetrics,
    pub report_duration: Duration,
    pub generated_at: SystemTime,
}

impl MetricsReport {
    /// Export report as JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize report: {}", e))
    }

    /// Export report as CSV-like format
    pub fn to_csv(&self) -> String {
        format!(
            "metric,count,mean,p50,p95,p99,min,max,std_dev\n\
             query_time_ms,{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n\
             error_rate,{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n\
             memory_mb,{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n\
             cpu_percent,{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n\
             throughput_ops,{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            self.query_performance.count,
            self.query_performance.mean,
            self.query_performance.p50,
            self.query_performance.p95,
            self.query_performance.p99,
            self.query_performance.min,
            self.query_performance.max,
            self.query_performance.std_dev,
            self.error_rates.count,
            self.error_rates.mean,
            self.error_rates.p50,
            self.error_rates.p95,
            self.error_rates.p99,
            self.error_rates.min,
            self.error_rates.max,
            self.error_rates.std_dev,
            self.memory_usage.count,
            self.memory_usage.mean / 1024.0 / 1024.0, // Convert to MB
            self.memory_usage.p50 / 1024.0 / 1024.0,
            self.memory_usage.p95 / 1024.0 / 1024.0,
            self.memory_usage.p99 / 1024.0 / 1024.0,
            self.memory_usage.min / 1024.0 / 1024.0,
            self.memory_usage.max / 1024.0 / 1024.0,
            self.memory_usage.std_dev / 1024.0 / 1024.0,
            self.cpu_usage.count,
            self.cpu_usage.mean,
            self.cpu_usage.p50,
            self.cpu_usage.p95,
            self.cpu_usage.p99,
            self.cpu_usage.min,
            self.cpu_usage.max,
            self.cpu_usage.std_dev,
            self.throughput.count,
            self.throughput.mean,
            self.throughput.p50,
            self.throughput.p95,
            self.throughput.p99,
            self.throughput.min,
            self.throughput.max,
            self.throughput.std_dev,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_time_series_metrics() {
        let metrics = TimeSeriesMetrics::new(100);
        let mut labels = HashMap::new();
        labels.insert("test".to_string(), "value".to_string());

        metrics.add_point(10.0, labels.clone()).unwrap();
        metrics.add_point(20.0, labels.clone()).unwrap();
        metrics.add_point(30.0, labels).unwrap();

        let data = metrics.get_data().unwrap();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0].value, 10.0);
        assert_eq!(data[1].value, 20.0);
        assert_eq!(data[2].value, 30.0);
    }

    #[test]
    fn test_metrics_statistics() {
        let metrics = TimeSeriesMetrics::new(100);
        let labels = HashMap::new();

        for i in 1..=100 {
            metrics.add_point(i as f64, labels.clone()).unwrap();
        }

        let stats = metrics.calculate_stats(Duration::from_secs(60)).unwrap();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 100.0);
        assert_eq!(stats.mean, 50.5);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new(Duration::from_secs(1));
        let mut labels = HashMap::new();
        labels.insert("operation".to_string(), "insert".to_string());

        collector
            .record_query_time(Duration::from_millis(50), labels)
            .unwrap();
        collector.record_error("timeout").unwrap();

        let query_stats = collector.get_query_stats(Duration::from_secs(60)).unwrap();
        assert_eq!(query_stats.count, 1);
        assert_eq!(query_stats.mean, 50.0);

        let error_stats = collector.get_error_stats(Duration::from_secs(60)).unwrap();
        assert_eq!(error_stats.count, 1);
    }

    #[test]
    fn test_metrics_report_generation() {
        let collector = MetricsCollector::new(Duration::from_secs(1));

        // Add some test data
        let mut labels = HashMap::new();
        labels.insert("operation".to_string(), "query".to_string());
        collector
            .record_query_time(Duration::from_millis(100), labels)
            .unwrap();
        collector.record_throughput(1000.0, HashMap::new()).unwrap();

        let report = collector.generate_report(Duration::from_secs(60)).unwrap();
        assert!(report.query_performance.count > 0);

        let json = report.to_json().unwrap();
        assert!(!json.is_empty());

        let csv = report.to_csv();
        assert!(csv.contains("query_time_ms"));
    }

    #[test]
    fn test_max_points_limit() {
        let metrics = TimeSeriesMetrics::new(5);
        let labels = HashMap::new();

        // Add more points than the limit
        for i in 1..=10 {
            metrics.add_point(i as f64, labels.clone()).unwrap();
        }

        let data = metrics.get_data().unwrap();
        assert_eq!(data.len(), 5); // Should be limited to max_points
        assert_eq!(data[0].value, 6.0); // Should keep the latest 5 points
        assert_eq!(data[4].value, 10.0);
    }
}
