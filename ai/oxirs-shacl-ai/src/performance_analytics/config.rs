//! Configuration types for performance analytics
//!
//! This module contains all configuration structures and settings
//! for the performance analytics system.

use serde::{Deserialize, Serialize};

/// Configuration for performance analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalyticsConfig {
    /// Enable real-time monitoring
    pub enable_real_time_monitoring: bool,

    /// Enable automatic optimization
    pub enable_auto_optimization: bool,

    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: u64,

    /// Performance threshold for alerts
    pub performance_threshold_ms: f64,

    /// Memory usage threshold in MB
    pub memory_threshold_mb: f64,

    /// CPU usage threshold percentage
    pub cpu_threshold_percent: f64,

    /// Enable adaptive thresholds
    pub enable_adaptive_thresholds: bool,

    /// Metrics retention period in hours
    pub metrics_retention_hours: u32,

    /// Enable performance profiling
    pub enable_profiling: bool,

    /// Profiling sample rate (0.0 - 1.0)
    pub profiling_sample_rate: f64,

    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,

    /// Alert cooldown period in minutes
    pub alert_cooldown_minutes: u32,

    /// Enable performance optimization suggestions
    pub enable_optimization_suggestions: bool,

    /// Optimization aggressiveness (0.0 - 1.0)
    pub optimization_aggressiveness: f64,
}

impl Default for PerformanceAnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_real_time_monitoring: true,
            enable_auto_optimization: false, // Conservative default
            monitoring_interval_ms: 1000,
            performance_threshold_ms: 5000.0,
            memory_threshold_mb: 1024.0,
            cpu_threshold_percent: 80.0,
            enable_adaptive_thresholds: true,
            metrics_retention_hours: 24,
            enable_profiling: true,
            profiling_sample_rate: 0.1,
            enable_anomaly_detection: true,
            alert_cooldown_minutes: 5,
            enable_optimization_suggestions: true,
            optimization_aggressiveness: 0.5,
        }
    }
}

/// Configuration for real-time monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Monitoring interval in milliseconds
    pub interval_ms: u64,

    /// Enable metric collection
    pub enable_metrics: bool,

    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,

    /// Enable performance profiling
    pub enable_profiling: bool,

    /// Buffer size for metrics
    pub metrics_buffer_size: usize,

    /// Maximum number of monitored sessions
    pub max_sessions: usize,

    /// Session timeout in seconds
    pub session_timeout_secs: u64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            interval_ms: 1000,
            enable_metrics: true,
            enable_resource_monitoring: true,
            enable_profiling: false,
            metrics_buffer_size: 10000,
            max_sessions: 100,
            session_timeout_secs: 3600,
        }
    }
}

/// Configuration for alert engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerting
    pub enable_alerts: bool,

    /// Response time threshold for alerts (ms)
    pub response_time_threshold_ms: f64,

    /// Memory usage threshold for alerts (MB)
    pub memory_threshold_mb: f64,

    /// CPU usage threshold for alerts (%)
    pub cpu_threshold_percent: f64,

    /// Error rate threshold for alerts (%)
    pub error_rate_threshold_percent: f64,

    /// Alert cooldown period in minutes
    pub cooldown_minutes: u32,

    /// Enable anomaly-based alerts
    pub enable_anomaly_alerts: bool,

    /// Anomaly score threshold for alerts
    pub anomaly_threshold: f64,

    /// Enable email notifications
    pub enable_email_notifications: bool,

    /// Email recipients
    pub email_recipients: Vec<String>,

    /// Enable webhook notifications
    pub enable_webhook_notifications: bool,

    /// Webhook URLs
    pub webhook_urls: Vec<String>,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enable_alerts: true,
            response_time_threshold_ms: 5000.0,
            memory_threshold_mb: 1024.0,
            cpu_threshold_percent: 80.0,
            error_rate_threshold_percent: 5.0,
            cooldown_minutes: 5,
            enable_anomaly_alerts: true,
            anomaly_threshold: 0.8,
            enable_email_notifications: false,
            email_recipients: Vec::new(),
            enable_webhook_notifications: false,
            webhook_urls: Vec::new(),
        }
    }
}

/// Configuration for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimizationConfig {
    /// Enable automatic optimization
    pub enable_auto_optimization: bool,

    /// Optimization aggressiveness (0.0 - 1.0)
    pub aggressiveness: f64,

    /// Minimum improvement threshold to apply optimization
    pub min_improvement_threshold: f64,

    /// Maximum optimization attempts per session
    pub max_optimization_attempts: u32,

    /// Optimization cooldown period in minutes
    pub optimization_cooldown_minutes: u32,

    /// Enable constraint reordering optimization
    pub enable_constraint_reordering: bool,

    /// Enable caching optimizations
    pub enable_caching_optimizations: bool,

    /// Enable parallel processing optimizations
    pub enable_parallel_optimizations: bool,

    /// Enable memory optimization
    pub enable_memory_optimization: bool,

    /// Target memory usage percentage
    pub target_memory_percent: f64,

    /// Target CPU usage percentage
    pub target_cpu_percent: f64,
}

impl Default for PerformanceOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_auto_optimization: false,
            aggressiveness: 0.5,
            min_improvement_threshold: 0.1,
            max_optimization_attempts: 3,
            optimization_cooldown_minutes: 10,
            enable_constraint_reordering: true,
            enable_caching_optimizations: true,
            enable_parallel_optimizations: true,
            enable_memory_optimization: true,
            target_memory_percent: 70.0,
            target_cpu_percent: 60.0,
        }
    }
}

/// Configuration for metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enable_collection: bool,

    /// Metrics retention period in hours
    pub retention_hours: u32,

    /// Sampling rate for metrics (0.0 - 1.0)
    pub sampling_rate: f64,

    /// Maximum number of metrics in memory
    pub max_metrics_in_memory: usize,

    /// Enable metric aggregation
    pub enable_aggregation: bool,

    /// Aggregation window size in seconds
    pub aggregation_window_secs: u64,

    /// Enable metric export
    pub enable_export: bool,

    /// Export format (json, csv, prometheus)
    pub export_format: String,

    /// Export interval in seconds
    pub export_interval_secs: u64,

    /// Export destination path
    pub export_path: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_collection: true,
            retention_hours: 24,
            sampling_rate: 1.0,
            max_metrics_in_memory: 100000,
            enable_aggregation: true,
            aggregation_window_secs: 60,
            enable_export: false,
            export_format: "json".to_string(),
            export_interval_secs: 300,
            export_path: "/tmp/metrics".to_string(),
        }
    }
}
