//! Tests for the real-time analytics module.

#[cfg(test)]
mod tests {
    use crate::rta_aggregators::ExportFormat;
    use crate::rta_engine::{AlertSeverity, AlertType, AnalyticsConfig, VectorAnalyticsEngine};
    use std::time::Duration;

    #[test]
    fn test_analytics_engine_creation() {
        let config = AnalyticsConfig::default();
        let engine = VectorAnalyticsEngine::new(config);

        assert!(engine.config.enable_real_time);
        assert_eq!(engine.config.collection_interval, 1);
    }

    #[test]
    fn test_query_recording() {
        let config = AnalyticsConfig::default();
        let engine = VectorAnalyticsEngine::new(config);

        let result = engine.record_query_execution(
            "test_query_1".to_string(),
            "similarity_search".to_string(),
            Duration::from_millis(50),
            10,
            true,
        );

        assert!(result.is_ok());

        let metrics = engine.metrics_collector.query_metrics.read();
        assert_eq!(metrics.total_queries, 1);
        assert_eq!(metrics.successful_queries, 1);
    }

    #[test]
    fn test_alert_creation() {
        let config = AnalyticsConfig::default();
        let engine = VectorAnalyticsEngine::new(config);

        let result = engine.create_alert(
            AlertType::HighLatency,
            AlertSeverity::Warning,
            "Test alert message".to_string(),
        );

        assert!(result.is_ok());

        let current_alerts = engine.performance_monitor.current_alerts.read();
        assert_eq!(current_alerts.len(), 1);
    }

    #[test]
    fn test_metrics_export() {
        let config = AnalyticsConfig::default();
        let engine = VectorAnalyticsEngine::new(config);

        // Record some metrics
        let _ = engine.record_query_execution(
            "test".to_string(),
            "search".to_string(),
            Duration::from_millis(25),
            5,
            true,
        );

        // Test JSON export
        let dir = tempfile::tempdir().expect("tempdir");
        let temp_file = dir.path().join("test_metrics.json");
        let temp_file = temp_file.display().to_string();
        let result = engine.export_metrics(ExportFormat::Json, &temp_file);
        assert!(result.is_ok());
        // `dir` (TempDir) cleans up automatically on drop.
    }
}
