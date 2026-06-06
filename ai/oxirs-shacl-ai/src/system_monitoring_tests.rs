//! Tests for the system monitoring subsystem.

#![cfg(test)]

use std::collections::HashMap;

use chrono::Utc;

use crate::system_monitoring::*;

#[test]
fn test_system_monitor_creation() {
    let monitor = SystemMonitor::new();
    assert!(monitor.config.enable_real_time);
    assert_eq!(monitor.config.collection_interval_secs, 60);
}

#[test]
fn test_monitoring_config_default() {
    let config = MonitoringConfig::default();
    assert!(config.enable_alerting);
    assert!(config.enable_performance_tracking);
    assert!(config.enable_quality_tracking);
    assert!(config.enable_error_tracking);
}

#[test]
fn test_alert_thresholds_default() {
    let thresholds = AlertThresholds::default();
    assert_eq!(thresholds.max_response_time_ms, 5000.0);
    assert_eq!(thresholds.min_quality_score, 0.8);
    assert_eq!(thresholds.max_error_rate, 0.05);
}

#[test]
fn test_health_status_ordering() {
    assert!(HealthStatus::Critical > HealthStatus::Warning);
    assert!(HealthStatus::Warning > HealthStatus::Healthy);
}

#[test]
fn test_metrics_collector() {
    let mut collector = MetricsCollector::new();

    let metric = PerformanceMetric {
        timestamp: Utc::now(),
        response_time_ms: 150.0,
        throughput: 1000.0,
        concurrent_requests: 10,
        memory_usage_mb: 512.0,
        cpu_usage_percent: 45.0,
        disk_io_mb_s: 10.0,
        network_io_mb_s: 25.0,
        gc_time_ms: Some(5.0),
        cache_hit_rate: Some(0.85),
        tags: HashMap::new(),
    };

    collector
        .add_performance_metric(metric)
        .expect("should succeed");
    assert_eq!(collector.performance_metrics.len(), 1);
    assert!(collector.total_metrics_count() > 0);
}

#[test]
fn test_dashboard_creation() {
    let dashboard = MonitoringDashboard::new();
    assert_eq!(dashboard.overall_health, SystemHealth::Healthy);
    assert!(dashboard.performance_summary.performance_score > 0.0);
    assert!(dashboard.quality_summary.overall_quality_score > 0.0);
}
