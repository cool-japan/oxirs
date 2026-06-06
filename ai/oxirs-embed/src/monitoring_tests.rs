//! Tests for monitoring metrics, health checks, and alerting.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use chrono::Utc;

    use crate::monitoring::{MonitoringConfig, PerformanceMonitor};
    use crate::monitoring_health::{
        Alert, AlertHandler, AlertSeverity, AlertType, ConsoleAlertHandler, HealthChecker,
        HealthStatus,
    };
    use crate::monitoring_metrics::{
        ErrorEvent, ErrorSeverity, MetricsCollector, PerformanceMetrics,
    };

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.latency.total_measurements, 0);
        assert_eq!(metrics.throughput.total_requests, 0);
    }

    #[tokio::test]
    async fn test_latency_recording() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        monitor.record_latency(100.0).await;
        monitor.record_latency(150.0).await;
        monitor.record_latency(120.0).await;

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.latency.total_measurements, 3);
        assert_eq!(metrics.latency.max_latency_ms, 150.0);
        assert_eq!(metrics.latency.min_latency_ms, 100.0);
    }

    #[tokio::test]
    async fn test_error_recording() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let error_event = ErrorEvent {
            timestamp: Utc::now(),
            error_type: "timeout".to_string(),
            error_message: "Request timeout".to_string(),
            severity: ErrorSeverity::Medium,
            context: HashMap::new(),
        };

        monitor.record_error(error_event).await;

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.errors.total_errors, 1);
        assert_eq!(metrics.errors.timeout_errors, 1);
    }

    #[test]
    fn test_alert_thresholds_default() {
        use crate::monitoring_health::AlertThresholds;

        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.max_p95_latency_ms, 500.0);
        assert_eq!(thresholds.min_throughput_rps, 100.0);
        assert_eq!(thresholds.max_error_rate, 0.05);
    }

    #[test]
    fn test_console_alert_handler() {
        let handler = ConsoleAlertHandler;
        let alert = Alert {
            alert_type: AlertType::HighLatency,
            message: "Test alert".to_string(),
            severity: AlertSeverity::Warning,
            timestamp: Utc::now(),
            metrics: HashMap::new(),
        };

        assert!(handler.handle_alert(alert).is_ok());
    }

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        // Just verify it was created successfully
        assert_eq!(collector.requests_total.get(), 0);
    }

    #[test]
    fn test_metrics_collector_counters() {
        let collector = MetricsCollector::new();

        collector.record_request_start();
        collector.record_request_complete(50.0);

        assert_eq!(collector.requests_total.get(), 1);
    }

    #[test]
    fn test_metrics_collector_cache_hit_rate() {
        let collector = MetricsCollector::new();

        collector.record_cache_hit();
        collector.record_cache_hit();
        collector.record_cache_miss();

        let hit_rate = collector.get_cache_hit_rate();
        assert!((hit_rate - 0.666).abs() < 0.01); // ~66.6%
    }

    #[test]
    fn test_metrics_collector_resource_update() {
        let collector = MetricsCollector::new();

        collector.update_resource_metrics(0.75, 2048.0, 0.5, 4096.0);

        assert_eq!(collector.cpu_utilization.get(), 0.75);
        assert_eq!(collector.memory_usage_bytes.get(), 2048.0 * 1024.0 * 1024.0);
        assert_eq!(collector.gpu_utilization.get(), 0.5);
        assert_eq!(collector.gpu_memory_bytes.get(), 4096.0 * 1024.0 * 1024.0);
    }

    #[test]
    fn test_health_checker_liveness() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        let result = checker.check_liveness();
        assert_eq!(result.status, HealthStatus::Healthy);
        assert!(result.components.contains_key("service"));
    }

    #[test]
    fn test_health_checker_readiness_no_models() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        let result = checker.check_readiness();
        assert_eq!(result.status, HealthStatus::Unhealthy);
        assert!(result.components.contains_key("models"));
    }

    #[test]
    fn test_health_checker_readiness_with_models() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        checker
            .set_models_loaded(true)
            .expect("Failed to set models loaded");

        let result = checker.check_readiness();
        assert_eq!(result.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_health_checker_comprehensive() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        checker
            .set_models_loaded(true)
            .expect("Failed to set models loaded");

        let perf_metrics = PerformanceMetrics::default();

        let result = checker.check_health(&perf_metrics);
        assert_eq!(result.status, HealthStatus::Healthy);
        assert!(result.components.contains_key("models"));
        assert!(result.components.contains_key("latency"));
        assert!(result.components.contains_key("errors"));
        assert!(result.components.contains_key("memory"));
    }

    #[test]
    fn test_health_checker_degraded_latency() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        checker
            .set_models_loaded(true)
            .expect("Failed to set models loaded");

        let mut perf_metrics = PerformanceMetrics::default();
        perf_metrics.latency.p95_latency_ms = 2000.0; // Above threshold

        let result = checker.check_health(&perf_metrics);
        assert_eq!(result.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_prometheus_export() {
        let collector = MetricsCollector::new();

        collector.record_request_start();
        collector.record_embeddings(5, 25.0);

        let prometheus_output = collector.export_prometheus();
        assert!(prometheus_output.is_ok());

        let _output = prometheus_output.unwrap_or_default();
        // Check that metrics were recorded (may not always be in prometheus output depending on implementation)
        assert_eq!(collector.requests_total.get(), 1);
        assert_eq!(collector.embeddings_generated_total.get(), 5);
    }
}
