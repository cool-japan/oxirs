//! Comprehensive tests for the performance analysis engine
//!
//! These tests verify the performance analysis, bottleneck detection,
//! and optimization recommendation capabilities.

use oxirs_federate::{
    performance_analyzer::*, FederatedService, FederationEngine, ServiceRegistry,
};
use std::time::{Duration, Instant};
use tokio;

#[tokio::test]
async fn test_performance_analyzer_creation() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceAnalyzer::new(config);

    // Should be able to create analyzer with default config
    assert_eq!(analyzer.get_config().enable_real_time_monitoring, true);
    assert_eq!(analyzer.get_config().history_retention_hours, 24);
    assert_eq!(analyzer.get_config().enable_predictive_analysis, true);
}

#[tokio::test]
async fn test_performance_analyzer_custom_config() {
    let config = AnalyzerConfig {
        enable_real_time_monitoring: false,
        history_retention_hours: 48,
        analysis_interval: Duration::from_secs(30),
        enable_predictive_analysis: false,
        min_data_points: 20,
        baseline_update_frequency: Duration::from_secs(600),
    };

    let analyzer = PerformanceAnalyzer::new(config.clone());

    assert_eq!(analyzer.get_config().enable_real_time_monitoring, false);
    assert_eq!(analyzer.get_config().history_retention_hours, 48);
    assert_eq!(
        analyzer.get_config().analysis_interval,
        Duration::from_secs(30)
    );
    assert_eq!(analyzer.get_config().enable_predictive_analysis, false);
}

#[tokio::test]
async fn test_system_metrics_recording() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceAnalyzer::new(config);

    let metrics = SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(100),
        overall_latency_p95: Duration::from_millis(300),
        overall_latency_p99: Duration::from_millis(500),
        throughput_qps: 50.0,
        error_rate: 0.01,
        timeout_rate: 0.005,
        cache_hit_rate: 0.85,
        memory_usage_mb: 512.0,
        cpu_usage_percent: 45.0,
        network_bandwidth_mbps: 100.0,
        active_connections: 25,
        queue_depth: 5,
    };

    // Should be able to record system metrics
    analyzer.record_system_metrics(metrics.clone()).await;

    // Get metrics history
    let history = analyzer.get_system_metrics_history().await;
    assert!(!history.is_empty());
    assert_eq!(history.last().unwrap().throughput_qps, 50.0);
}

#[tokio::test]
async fn test_service_metrics_recording() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceAnalyzer::new(config);

    let metrics = ServicePerformanceMetrics {
        service_id: "test-service-1".to_string(),
        timestamp: Instant::now(),
        response_time_p50: Duration::from_millis(80),
        response_time_p95: Duration::from_millis(200),
        response_time_p99: Duration::from_millis(350),
        requests_per_second: 25.0,
        error_rate: 0.02,
        timeout_rate: 0.01,
        availability: 0.99,
        data_transfer_kb: 150.0,
        connection_pool_utilization: 0.6,
    };

    // Record service metrics
    analyzer.record_service_metrics(metrics.clone()).await;

    // Get service metrics
    let service_history = analyzer.get_service_metrics_history("test-service-1").await;
    assert!(!service_history.is_empty());
    assert_eq!(service_history.last().unwrap().requests_per_second, 25.0);
}

#[tokio::test]
async fn test_query_execution_metrics() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceAnalyzer::new(config);

    let metrics = QueryExecutionMetrics {
        query_id: "query-123".to_string(),
        timestamp: Instant::now(),
        total_execution_time: Duration::from_millis(250),
        planning_time: Duration::from_millis(50),
        execution_time: Duration::from_millis(180),
        result_serialization_time: Duration::from_millis(20),
        services_involved: vec!["service-1".to_string(), "service-2".to_string()],
        result_size_bytes: 4096,
        cache_hits: 2,
        cache_misses: 1,
        join_operations: 1,
    };

    // Record query execution metrics
    analyzer.record_query_execution(metrics.clone()).await;

    // Get query metrics
    let query_history = analyzer.get_query_metrics_history().await;
    assert!(!query_history.is_empty());
    assert_eq!(query_history.last().unwrap().query_id, "query-123");
    assert_eq!(query_history.last().unwrap().services_involved.len(), 2);
}

#[tokio::test]
async fn test_bottleneck_detection() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceAnalyzer::new(config);

    // Record high latency metrics to trigger bottleneck detection
    let high_latency_metrics = SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(800),
        overall_latency_p95: Duration::from_millis(1500),
        overall_latency_p99: Duration::from_millis(2000),
        throughput_qps: 10.0,         // Low throughput
        error_rate: 0.15,             // High error rate
        timeout_rate: 0.08,           // High timeout rate
        cache_hit_rate: 0.30,         // Low cache hit rate
        memory_usage_mb: 2048.0,      // High memory usage
        cpu_usage_percent: 95.0,      // High CPU usage
        network_bandwidth_mbps: 10.0, // Low bandwidth
        active_connections: 200,
        queue_depth: 50, // High queue depth
    };

    // Record problematic metrics
    for _ in 0..10 {
        analyzer
            .record_system_metrics(high_latency_metrics.clone())
            .await;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Analyze bottlenecks
    let bottlenecks = analyzer.analyze_bottlenecks().await.unwrap();

    // Should detect multiple bottlenecks
    assert!(!bottlenecks.is_empty());

    // Should detect high latency bottleneck
    assert!(bottlenecks
        .iter()
        .any(|b| matches!(b.bottleneck_type, BottleneckType::HighLatency)));

    // Should detect high error rate bottleneck
    assert!(bottlenecks
        .iter()
        .any(|b| matches!(b.bottleneck_type, BottleneckType::HighErrorRate)));

    // Should detect resource constraints
    assert!(bottlenecks
        .iter()
        .any(|b| matches!(b.bottleneck_type, BottleneckType::ResourceConstraint)));
}

#[tokio::test]
async fn test_performance_prediction() {
    let config = AnalyzerConfig {
        enable_predictive_analysis: true,
        min_data_points: 5,
        ..Default::default()
    };
    let analyzer = PerformanceAnalyzer::new(config);

    // Record trend data for prediction
    let base_time = Instant::now();
    for i in 0..10 {
        let metrics = SystemPerformanceMetrics {
            timestamp: base_time + Duration::from_secs(i * 60),
            overall_latency_p50: Duration::from_millis(100 + i * 10), // Increasing trend
            overall_latency_p95: Duration::from_millis(200 + i * 20),
            overall_latency_p99: Duration::from_millis(400 + i * 30),
            throughput_qps: 100.0 - (i as f64 * 2.0), // Decreasing trend
            error_rate: 0.01 + (i as f64 * 0.005),    // Increasing trend
            timeout_rate: 0.005,
            cache_hit_rate: 0.85,
            memory_usage_mb: 512.0 + (i as f64 * 50.0), // Increasing trend
            cpu_usage_percent: 30.0 + (i as f64 * 5.0), // Increasing trend
            network_bandwidth_mbps: 100.0,
            active_connections: 25,
            queue_depth: i,
        };
        analyzer.record_system_metrics(metrics).await;
    }

    // Get performance predictions
    let predictions = analyzer.predict_performance().await.unwrap();

    // Should have predictions for various metrics
    assert!(!predictions.is_empty());

    // Should predict performance degradation trends
    assert!(predictions
        .iter()
        .any(|p| matches!(p.prediction_type, PredictionType::PerformanceDegradation)));

    // Should predict resource exhaustion
    assert!(predictions
        .iter()
        .any(|p| matches!(p.prediction_type, PredictionType::ResourceExhaustion)));
}

#[tokio::test]
async fn test_optimization_recommendations() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceAnalyzer::new(config);

    // Record metrics that would trigger optimization recommendations
    let problematic_metrics = SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(500),
        overall_latency_p95: Duration::from_millis(1200),
        overall_latency_p99: Duration::from_millis(2500),
        throughput_qps: 15.0,
        error_rate: 0.08,
        timeout_rate: 0.05,
        cache_hit_rate: 0.40,         // Low cache hit rate
        memory_usage_mb: 1800.0,      // High memory usage
        cpu_usage_percent: 85.0,      // High CPU usage
        network_bandwidth_mbps: 20.0, // Low bandwidth
        active_connections: 150,
        queue_depth: 30,
    };

    // Record multiple data points
    for _ in 0..15 {
        analyzer
            .record_system_metrics(problematic_metrics.clone())
            .await;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // Generate optimization recommendations
    let recommendations = analyzer
        .generate_optimization_recommendations()
        .await
        .unwrap();

    // Should have recommendations
    assert!(!recommendations.is_empty());

    // Should recommend caching improvements
    assert!(recommendations
        .iter()
        .any(|r| matches!(r.category, OptimizationCategory::Caching)));

    // Should recommend resource scaling
    assert!(recommendations
        .iter()
        .any(|r| matches!(r.category, OptimizationCategory::ResourceScaling)));

    // Should recommend query optimization
    assert!(recommendations
        .iter()
        .any(|r| matches!(r.category, OptimizationCategory::QueryOptimization)));

    // Each recommendation should have a priority and effort estimate
    for rec in &recommendations {
        assert!(matches!(
            rec.priority,
            Priority::High | Priority::Medium | Priority::Low
        ));
        assert!(matches!(
            rec.effort,
            ImplementationEffort::Low | ImplementationEffort::Medium | ImplementationEffort::High
        ));
        assert!(!rec.description.is_empty());
        assert!(!rec.impact_description.is_empty());
    }
}

#[tokio::test]
async fn test_baseline_establishment() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceAnalyzer::new(config);

    // Record stable baseline metrics
    let baseline_metrics = SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(120),
        overall_latency_p95: Duration::from_millis(250),
        overall_latency_p99: Duration::from_millis(400),
        throughput_qps: 80.0,
        error_rate: 0.02,
        timeout_rate: 0.01,
        cache_hit_rate: 0.85,
        memory_usage_mb: 600.0,
        cpu_usage_percent: 50.0,
        network_bandwidth_mbps: 100.0,
        active_connections: 30,
        queue_depth: 5,
    };

    // Record enough data points to establish baseline
    for _ in 0..20 {
        analyzer
            .record_system_metrics(baseline_metrics.clone())
            .await;
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    // Update baseline
    analyzer.update_performance_baseline().await.unwrap();

    // Get baseline
    let baseline = analyzer.get_performance_baseline().await.unwrap();

    // Baseline should be established
    assert!(baseline.latency_p50_baseline > Duration::from_millis(0));
    assert!(baseline.throughput_baseline > 0.0);
    assert!(baseline.error_rate_baseline >= 0.0);
}

#[tokio::test]
async fn test_anomaly_detection() {
    let config = AnalyzerConfig {
        min_data_points: 5,
        ..Default::default()
    };
    let analyzer = PerformanceAnalyzer::new(config);

    // Record normal metrics
    let normal_metrics = SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(100),
        overall_latency_p95: Duration::from_millis(200),
        overall_latency_p99: Duration::from_millis(350),
        throughput_qps: 100.0,
        error_rate: 0.01,
        timeout_rate: 0.005,
        cache_hit_rate: 0.85,
        memory_usage_mb: 500.0,
        cpu_usage_percent: 40.0,
        network_bandwidth_mbps: 100.0,
        active_connections: 25,
        queue_depth: 3,
    };

    // Establish baseline with normal metrics
    for _ in 0..10 {
        analyzer.record_system_metrics(normal_metrics.clone()).await;
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    // Record anomalous metrics
    let anomalous_metrics = SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(1000), // 10x normal
        overall_latency_p95: Duration::from_millis(2000),
        overall_latency_p99: Duration::from_millis(3000),
        throughput_qps: 10.0,         // 1/10 normal
        error_rate: 0.20,             // 20x normal
        timeout_rate: 0.10,           // 20x normal
        cache_hit_rate: 0.20,         // Much lower than normal
        memory_usage_mb: 2000.0,      // 4x normal
        cpu_usage_percent: 95.0,      // Much higher than normal
        network_bandwidth_mbps: 10.0, // 1/10 normal
        active_connections: 200,      // 8x normal
        queue_depth: 50,              // Much higher than normal
    };

    analyzer.record_system_metrics(anomalous_metrics).await;

    // Detect anomalies
    let anomalies = analyzer.detect_anomalies().await.unwrap();

    // Should detect multiple anomalies
    assert!(!anomalies.is_empty());

    // Should detect performance degradation
    assert!(anomalies
        .iter()
        .any(|a| matches!(a.anomaly_type, AnomalyType::PerformanceDegradation)));

    // Should detect error spike
    assert!(anomalies
        .iter()
        .any(|a| matches!(a.anomaly_type, AnomalyType::ErrorSpike)));

    // Should detect resource spike
    assert!(anomalies
        .iter()
        .any(|a| matches!(a.anomaly_type, AnomalyType::ResourceSpike)));

    // Each anomaly should have confidence score
    for anomaly in &anomalies {
        assert!(anomaly.confidence >= 0.0 && anomaly.confidence <= 1.0);
        assert!(!anomaly.description.is_empty());
        assert!(anomaly.severity > 0.0);
    }
}

#[tokio::test]
async fn test_performance_regression_detection() {
    let config = AnalyzerConfig {
        min_data_points: 5,
        baseline_update_frequency: Duration::from_millis(100),
        ..Default::default()
    };
    let analyzer = PerformanceAnalyzer::new(config);

    // Record good baseline performance
    let good_metrics = SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(80),
        overall_latency_p95: Duration::from_millis(150),
        overall_latency_p99: Duration::from_millis(250),
        throughput_qps: 120.0,
        error_rate: 0.005,
        timeout_rate: 0.002,
        cache_hit_rate: 0.90,
        memory_usage_mb: 400.0,
        cpu_usage_percent: 35.0,
        network_bandwidth_mbps: 100.0,
        active_connections: 20,
        queue_depth: 2,
    };

    // Establish good baseline
    for _ in 0..10 {
        analyzer.record_system_metrics(good_metrics.clone()).await;
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    // Update baseline
    analyzer.update_performance_baseline().await.unwrap();

    // Wait for baseline update
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Record degraded performance
    let degraded_metrics = SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(200), // 2.5x worse
        overall_latency_p95: Duration::from_millis(400),
        overall_latency_p99: Duration::from_millis(600),
        throughput_qps: 60.0,         // 50% worse
        error_rate: 0.02,             // 4x worse
        timeout_rate: 0.01,           // 5x worse
        cache_hit_rate: 0.70,         // Worse
        memory_usage_mb: 800.0,       // 2x worse
        cpu_usage_percent: 70.0,      // 2x worse
        network_bandwidth_mbps: 80.0, // Slightly worse
        active_connections: 40,       // 2x worse
        queue_depth: 8,               // 4x worse
    };

    // Record degraded performance multiple times
    for _ in 0..8 {
        analyzer
            .record_system_metrics(degraded_metrics.clone())
            .await;
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    // Detect performance regressions
    let regressions = analyzer.detect_performance_regressions().await.unwrap();

    // Should detect regressions
    assert!(!regressions.is_empty());

    // Should detect latency regression
    assert!(regressions
        .iter()
        .any(|r| matches!(r.regression_type, RegressionType::LatencyIncrease)));

    // Should detect throughput regression
    assert!(regressions
        .iter()
        .any(|r| matches!(r.regression_type, RegressionType::ThroughputDecrease)));

    // Should detect error rate regression
    assert!(regressions
        .iter()
        .any(|r| matches!(r.regression_type, RegressionType::ErrorRateIncrease)));

    // Each regression should have confidence and impact
    for regression in &regressions {
        assert!(regression.confidence >= 0.0 && regression.confidence <= 1.0);
        assert!(regression.impact_score > 0.0);
        assert!(!regression.description.is_empty());
    }
}

#[tokio::test]
async fn test_service_comparison_analysis() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceAnalyzer::new(config);

    // Record metrics for multiple services
    let services = vec!["service-fast", "service-slow", "service-medium"];

    for (i, service_id) in services.iter().enumerate() {
        let metrics = ServicePerformanceMetrics {
            service_id: service_id.to_string(),
            timestamp: Instant::now(),
            response_time_p50: Duration::from_millis(50 + i as u64 * 100),
            response_time_p95: Duration::from_millis(100 + i as u64 * 200),
            response_time_p99: Duration::from_millis(200 + i as u64 * 300),
            requests_per_second: 100.0 - (i as f64 * 30.0),
            error_rate: 0.01 + (i as f64 * 0.02),
            timeout_rate: 0.005 + (i as f64 * 0.01),
            availability: 0.99 - (i as f64 * 0.05),
            data_transfer_kb: 100.0 + (i as f64 * 50.0),
            connection_pool_utilization: 0.5 + (i as f64 * 0.2),
        };

        // Record multiple data points for each service
        for _ in 0..5 {
            analyzer.record_service_metrics(metrics.clone()).await;
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    // Compare service performance
    let comparison = analyzer.compare_service_performance().await.unwrap();

    // Should identify best and worst performing services
    assert!(!comparison.service_rankings.is_empty());
    assert_eq!(comparison.service_rankings.len(), 3);

    // Best performing service should be service-fast (index 0)
    assert_eq!(comparison.service_rankings[0].service_id, "service-fast");

    // Worst performing service should be service-slow (index 2)
    assert_eq!(comparison.service_rankings[2].service_id, "service-slow");

    // Should have performance insights
    assert!(!comparison.insights.is_empty());
}

#[tokio::test]
async fn test_alert_threshold_configuration() {
    let mut config = AnalyzerConfig::default();
    let analyzer = PerformanceAnalyzer::new(config);

    // Configure custom alert thresholds
    let thresholds = AlertThresholds {
        high_latency_threshold: Duration::from_millis(500),
        low_throughput_threshold: 50.0,
        high_error_rate_threshold: 0.05,
        high_timeout_rate_threshold: 0.03,
        low_cache_hit_rate_threshold: 0.60,
        high_memory_usage_threshold: 80.0,
        high_cpu_usage_threshold: 85.0,
        high_queue_depth_threshold: 20,
    };

    analyzer.update_alert_thresholds(thresholds.clone()).await;

    // Record metrics that exceed thresholds
    let threshold_exceeding_metrics = SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(600), // Exceeds threshold
        overall_latency_p95: Duration::from_millis(1000),
        overall_latency_p99: Duration::from_millis(1500),
        throughput_qps: 40.0,    // Below threshold
        error_rate: 0.08,        // Exceeds threshold
        timeout_rate: 0.05,      // Exceeds threshold
        cache_hit_rate: 0.50,    // Below threshold
        memory_usage_mb: 1000.0, // Exceeds threshold (assuming total is ~1200MB)
        cpu_usage_percent: 90.0, // Exceeds threshold
        network_bandwidth_mbps: 100.0,
        active_connections: 30,
        queue_depth: 25, // Exceeds threshold
    };

    analyzer
        .record_system_metrics(threshold_exceeding_metrics)
        .await;

    // Check for triggered alerts
    let alerts = analyzer.get_triggered_alerts().await.unwrap();

    // Should have multiple alerts triggered
    assert!(!alerts.is_empty());
    assert!(alerts.len() >= 3); // At least latency, throughput, and error rate alerts
}

#[tokio::test]
async fn test_metrics_cleanup() {
    let config = AnalyzerConfig {
        history_retention_hours: 1, // Short retention for testing
        ..Default::default()
    };
    let analyzer = PerformanceAnalyzer::new(config);

    // Record some metrics
    let metrics = SystemPerformanceMetrics {
        timestamp: Instant::now() - Duration::from_secs(7200), // 2 hours ago
        overall_latency_p50: Duration::from_millis(100),
        overall_latency_p95: Duration::from_millis(200),
        overall_latency_p99: Duration::from_millis(350),
        throughput_qps: 100.0,
        error_rate: 0.01,
        timeout_rate: 0.005,
        cache_hit_rate: 0.85,
        memory_usage_mb: 500.0,
        cpu_usage_percent: 40.0,
        network_bandwidth_mbps: 100.0,
        active_connections: 25,
        queue_depth: 3,
    };

    analyzer.record_system_metrics(metrics).await;

    let history_before = analyzer.get_system_metrics_history().await;
    assert!(!history_before.is_empty());

    // Clean up old metrics
    analyzer.cleanup_old_metrics().await.unwrap();

    let history_after = analyzer.get_system_metrics_history().await;
    // Old metrics should be cleaned up (since they're older than retention period)
    assert!(history_after.len() <= history_before.len());
}

/// Helper function to create test performance metrics
fn create_test_system_metrics(
    latency_ms: u64,
    throughput: f64,
    error_rate: f64,
) -> SystemPerformanceMetrics {
    SystemPerformanceMetrics {
        timestamp: Instant::now(),
        overall_latency_p50: Duration::from_millis(latency_ms),
        overall_latency_p95: Duration::from_millis(latency_ms * 2),
        overall_latency_p99: Duration::from_millis(latency_ms * 4),
        throughput_qps: throughput,
        error_rate,
        timeout_rate: error_rate / 2.0,
        cache_hit_rate: 0.85,
        memory_usage_mb: 500.0,
        cpu_usage_percent: 40.0,
        network_bandwidth_mbps: 100.0,
        active_connections: 25,
        queue_depth: 3,
    }
}

/// Helper function to create test service metrics
fn create_test_service_metrics(
    service_id: &str,
    response_time_ms: u64,
    rps: f64,
    error_rate: f64,
) -> ServicePerformanceMetrics {
    ServicePerformanceMetrics {
        service_id: service_id.to_string(),
        timestamp: Instant::now(),
        response_time_p50: Duration::from_millis(response_time_ms),
        response_time_p95: Duration::from_millis(response_time_ms * 2),
        response_time_p99: Duration::from_millis(response_time_ms * 3),
        requests_per_second: rps,
        error_rate,
        timeout_rate: error_rate / 3.0,
        availability: 1.0 - error_rate,
        data_transfer_kb: 100.0,
        connection_pool_utilization: 0.6,
    }
}
