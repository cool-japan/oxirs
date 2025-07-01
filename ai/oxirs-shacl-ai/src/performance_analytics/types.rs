//! Core types for performance analytics
//!
//! This module contains the fundamental data structures and enums
//! used throughout the performance analytics system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Performance metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    /// Metric name
    pub name: String,

    /// Metric value
    pub value: f64,

    /// Timestamp of measurement
    pub timestamp: SystemTime,

    /// Metric unit
    pub unit: String,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Performance statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,

    /// Median response time in milliseconds
    pub median_response_time_ms: f64,

    /// 95th percentile response time
    pub p95_response_time_ms: f64,

    /// 99th percentile response time
    pub p99_response_time_ms: f64,

    /// Maximum response time
    pub max_response_time_ms: f64,

    /// Minimum response time
    pub min_response_time_ms: f64,

    /// Throughput (requests per second)
    pub throughput_rps: f64,

    /// Error rate percentage
    pub error_rate_percent: f64,

    /// Memory usage in MB
    pub memory_usage_mb: f64,

    /// CPU usage percentage
    pub cpu_usage_percent: f64,

    /// Number of total requests
    pub total_requests: u64,

    /// Number of successful requests
    pub successful_requests: u64,

    /// Number of failed requests
    pub failed_requests: u64,
}

impl Default for PerformanceStatistics {
    fn default() -> Self {
        Self {
            avg_response_time_ms: 0.0,
            median_response_time_ms: 0.0,
            p95_response_time_ms: 0.0,
            p99_response_time_ms: 0.0,
            max_response_time_ms: 0.0,
            min_response_time_ms: f64::MAX,
            throughput_rps: 0.0,
            error_rate_percent: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
        }
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU usage percentage
    pub cpu_percent: f64,

    /// Memory usage in MB
    pub memory_mb: f64,

    /// Memory usage percentage
    pub memory_percent: f64,

    /// Disk usage in MB
    pub disk_mb: f64,

    /// Disk I/O rate in MB/s
    pub disk_io_mbs: f64,

    /// Network bandwidth usage in KB/s
    pub network_kbs: f64,

    /// Number of open file descriptors
    pub open_files: u32,

    /// Number of active threads
    pub active_threads: u32,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_mb: 0.0,
            memory_percent: 0.0,
            disk_mb: 0.0,
            disk_io_mbs: 0.0,
            network_kbs: 0.0,
            open_files: 0,
            active_threads: 0,
        }
    }
}

/// Performance alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Performance alert type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighResponseTime,
    HighMemoryUsage,
    HighCpuUsage,
    HighErrorRate,
    LowThroughput,
    ResourceExhaustion,
    AnomalyDetected,
    ThresholdViolation,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert ID
    pub id: String,

    /// Alert type
    pub alert_type: AlertType,

    /// Severity level
    pub severity: AlertSeverity,

    /// Alert message
    pub message: String,

    /// Timestamp when alert was triggered
    pub timestamp: SystemTime,

    /// Metric value that triggered the alert
    pub trigger_value: f64,

    /// Threshold that was violated
    pub threshold: f64,

    /// Additional context
    pub context: HashMap<String, String>,

    /// Whether the alert has been acknowledged
    pub acknowledged: bool,

    /// Resolution timestamp (if resolved)
    pub resolved_at: Option<SystemTime>,
}

/// Performance optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub id: String,

    /// Recommendation type
    pub recommendation_type: String,

    /// Description of the recommendation
    pub description: String,

    /// Expected performance improvement percentage
    pub expected_improvement_percent: f64,

    /// Implementation complexity (1-10 scale)
    pub implementation_complexity: u8,

    /// Priority level (1-10 scale)
    pub priority: u8,

    /// Estimated implementation time in hours
    pub estimated_hours: f64,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Performance trend direction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Metric name
    pub metric_name: String,

    /// Trend direction
    pub direction: TrendDirection,

    /// Trend strength (0.0 - 1.0)
    pub strength: f64,

    /// Rate of change per time unit
    pub rate_of_change: f64,

    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,

    /// Time period analyzed
    pub time_period: Duration,

    /// Analysis timestamp
    pub analyzed_at: SystemTime,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Metric name
    pub metric_name: String,

    /// Whether an anomaly was detected
    pub is_anomaly: bool,

    /// Anomaly score (0.0 - 1.0)
    pub anomaly_score: f64,

    /// Expected value
    pub expected_value: f64,

    /// Actual observed value
    pub observed_value: f64,

    /// Deviation from expected
    pub deviation: f64,

    /// Detection timestamp
    pub detected_at: SystemTime,

    /// Additional context
    pub context: String,
}
