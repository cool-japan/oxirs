//! Cross-Module Performance — Tests

#[cfg(test)]
mod tests {
    use crate::cross_module_performance_profiler::{
        calculate_anomaly_score, calculate_percentage_change, AnomalyDetector,
        ModulePerformanceMonitor, PredictionModel, ResourceAllocator,
    };
    use crate::cross_module_performance_reporter::{
        CrossModulePerformanceCoordinator, OptimizationCache,
    };
    use crate::cross_module_performance_types::{
        AnomalyType, CoordinatorConfig, ModuleMetrics, OptimizationRecommendation,
        OptimizationType, PerformanceImpact, Priority,
    };
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = CoordinatorConfig::default();
        let coordinator = CrossModulePerformanceCoordinator::new(config);
        let _ = coordinator.register_module("dummy_check".to_string()).await;
    }

    #[tokio::test]
    async fn test_module_registration() {
        let config = CoordinatorConfig::default();
        let coordinator = CrossModulePerformanceCoordinator::new(config);
        let result = coordinator.register_module("test_module".to_string()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_update() {
        let config = CoordinatorConfig::default();
        let coordinator = CrossModulePerformanceCoordinator::new(config);
        coordinator
            .register_module("test_module".to_string())
            .await
            .expect("should succeed");

        let metrics = ModuleMetrics {
            cpu_usage: 75.0,
            memory_usage: 4_000_000_000,
            gpu_memory_usage: Some(2_000_000_000),
            network_io_bps: 1_000_000,
            disk_io_bps: 500_000,
            request_rate: 100.0,
            avg_response_time: Duration::from_millis(150),
            error_rate: 2.0,
            cache_hit_rate: 85.0,
            active_connections: 50,
            queue_depth: 10,
        };
        let result = coordinator
            .update_module_metrics("test_module", metrics)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_anomaly_detection() {
        let detector = AnomalyDetector::new();
        let mut performance_data = HashMap::new();
        performance_data.insert(
            "test_module".to_string(),
            ModuleMetrics {
                cpu_usage: 95.0,
                memory_usage: 4_000_000_000,
                gpu_memory_usage: Some(2_000_000_000),
                network_io_bps: 1_000_000,
                disk_io_bps: 500_000,
                request_rate: 100.0,
                avg_response_time: Duration::from_millis(1500),
                error_rate: 8.0,
                cache_hit_rate: 85.0,
                active_connections: 50,
                queue_depth: 10,
            },
        );
        let anomalies = detector
            .detect(&performance_data)
            .await
            .expect("should succeed");
        assert!(!anomalies.is_empty());
        assert_eq!(
            anomalies[0].anomaly_type,
            AnomalyType::PerformanceDegradation
        );
    }

    #[tokio::test]
    async fn test_resource_allocation() {
        let allocator = ResourceAllocator::new();
        let recommendation = OptimizationRecommendation {
            module_name: "test_module".to_string(),
            optimization_type: OptimizationType::ResourceReallocation,
            priority: Priority::High,
            description: "Test optimization".to_string(),
            estimated_impact: PerformanceImpact {
                latency_change_pct: -20.0,
                throughput_change_pct: 30.0,
                efficiency_change_pct: 15.0,
                overall_score: 80.0,
            },
            implementation_steps: vec!["Increase CPU allocation".to_string()],
        };
        let result = allocator
            .reallocate_resources("test_module", &recommendation)
            .await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_percentage_change_calculation() {
        assert_eq!(calculate_percentage_change(100.0, 120.0), 20.0);
        assert_eq!(calculate_percentage_change(100.0, 80.0), -20.0);
        assert_eq!(calculate_percentage_change(0.0, 100.0), 0.0);
    }

    #[test]
    fn test_anomaly_score_calculation() {
        let metrics = ModuleMetrics {
            cpu_usage: 90.0,
            memory_usage: 4_000_000_000,
            gpu_memory_usage: Some(2_000_000_000),
            network_io_bps: 1_000_000,
            disk_io_bps: 500_000,
            request_rate: 100.0,
            avg_response_time: Duration::from_millis(800),
            error_rate: 5.0,
            cache_hit_rate: 85.0,
            active_connections: 50,
            queue_depth: 10,
        };
        let score = calculate_anomaly_score(&metrics);
        assert!(score > 0.0);
        assert!(score > 50.0);
    }

    #[tokio::test]
    async fn test_optimization_cache() {
        let cache = OptimizationCache::new();
        assert_eq!(cache.stats.size.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_module_monitor_creation() {
        let monitor = ModulePerformanceMonitor::new("test_module".to_string());
        assert_eq!(monitor.module_name, "test_module");
        let metrics = monitor.get_current_metrics().await.expect("should succeed");
        assert_eq!(metrics.cpu_usage, 0.0);
    }

    #[test]
    fn test_prediction_model() {
        let model = PredictionModel::new();
        assert!(model.parameters.is_empty());
    }
}
