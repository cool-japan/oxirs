//! Tests for performance monitoring.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::performance_monitoring::{MonitoringConfig, PerformanceMonitor};
    use crate::performance_monitoring_types::{
        AnomalyEvent, AnomalySeverity, GcActivity, ImpactAssessment, ImplementationEffort,
        OptimizationRecommendation, PerformanceAnomalyType, PerformanceMetric, ResolutionStatus,
        RiskLevel,
    };

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        assert!(monitor.config.enable_realtime_monitoring);
        assert!(monitor.config.enable_optimization_recommendations);
        assert!(monitor.config.enable_alerting);
    }

    #[test]
    fn test_monitoring_config() {
        let config = MonitoringConfig::default();
        assert_eq!(config.monitoring_interval_ms, 1000);
        assert_eq!(config.metrics_retention_seconds, 86400);
        assert_eq!(config.performance_alert_threshold, 0.8);
    }

    #[test]
    fn test_performance_metric() {
        let metric = PerformanceMetric {
            timestamp: chrono::Utc::now(),
            validation_latency_ms: 150.0,
            memory_usage_mb: 512.0,
            cpu_usage_percent: 45.0,
            throughput_validations_per_second: 100.0,
            cache_hit_rate: 0.85,
            error_rate: 0.025,
            concurrent_validations: 5,
            queue_depth: 2,
            gc_activity: GcActivity {
                gc_count: 10,
                gc_time_ms: 50.0,
                heap_usage_mb: 256.0,
                heap_growth_rate: 0.1,
            },
        };

        assert_eq!(metric.validation_latency_ms, 150.0);
        assert_eq!(metric.memory_usage_mb, 512.0);
        assert_eq!(metric.cache_hit_rate, 0.85);
    }

    #[test]
    fn test_optimization_recommendation() {
        let recommendation = OptimizationRecommendation {
            recommendation_id: "opt_001".to_string(),
            timestamp: chrono::Utc::now(),
            optimization_type: "Cache Optimization".to_string(),
            description: "Increase cache size".to_string(),
            current_performance: 80.0,
            expected_performance: 95.0,
            improvement_percentage: 15.0,
            implementation_effort: ImplementationEffort::Low,
            risk_level: RiskLevel::Low,
            prerequisites: vec![],
            implementation_steps: vec![],
            success_criteria: vec![],
            monitoring_recommendations: vec![],
        };

        assert_eq!(recommendation.improvement_percentage, 15.0);
        assert!(matches!(
            recommendation.implementation_effort,
            ImplementationEffort::Low
        ));
    }

    #[test]
    fn test_anomaly_event() {
        let event = AnomalyEvent {
            event_id: "anomaly_001".to_string(),
            timestamp: chrono::Utc::now(),
            anomaly_type: PerformanceAnomalyType::LatencyIncrease,
            severity: AnomalySeverity::Warning,
            affected_metrics: vec!["validation_latency".to_string()],
            description: "Validation latency increased by 50%".to_string(),
            impact_assessment: ImpactAssessment {
                user_impact_level: 0.3,
                system_impact_level: 0.5,
                business_impact_level: 0.2,
                affected_operations: vec!["validation".to_string()],
                estimated_resolution_time: Duration::from_secs(1800),
                potential_data_loss_risk: 0.0,
            },
            recommended_actions: vec!["Check system resources".to_string()],
            resolution_status: ResolutionStatus::Open,
        };

        assert!(matches!(
            event.anomaly_type,
            PerformanceAnomalyType::LatencyIncrease
        ));
        assert!(matches!(event.severity, AnomalySeverity::Warning));
    }
}
