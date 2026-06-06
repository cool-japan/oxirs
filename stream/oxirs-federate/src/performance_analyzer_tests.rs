//! Tests for the performance analyzer subsystem.

#![cfg(test)]

use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

use crate::performance_analyzer_collector::PerformanceAnalyzer;
use crate::performance_analyzer_reporter::QueryPlanOptimizer;
use crate::performance_analyzer_types::*;

#[tokio::test]
async fn test_performance_analyzer_creation() {
    let analyzer = PerformanceAnalyzer::new();
    assert!(analyzer.config.enable_real_time_monitoring);
}

#[tokio::test]
async fn test_metrics_recording() {
    let analyzer = PerformanceAnalyzer::new();

    let system_metrics = SystemPerformanceMetrics {
        timestamp: SystemTime::now(),
        overall_latency_p50: Duration::from_millis(100),
        overall_latency_p95: Duration::from_millis(200),
        overall_latency_p99: Duration::from_millis(500),
        throughput_qps: 100.0,
        error_rate: 0.01,
        timeout_rate: 0.001,
        cache_hit_rate: 0.85,
        memory_usage_mb: 512.0,
        cpu_usage_percent: 65.0,
        network_bandwidth_mbps: 100.0,
        active_connections: 50,
        queue_depth: 10,
    };

    let result = analyzer.record_system_metrics(system_metrics).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_trend_analysis() {
    let analyzer = PerformanceAnalyzer::new();

    // Record multiple metrics to enable trend analysis
    for i in 0..15 {
        let metrics = SystemPerformanceMetrics {
            timestamp: SystemTime::now(),
            overall_latency_p50: Duration::from_millis(100 + i * 10),
            overall_latency_p95: Duration::from_millis(200 + i * 20),
            overall_latency_p99: Duration::from_millis(500 + i * 50),
            throughput_qps: 100.0 - i as f64,
            error_rate: 0.01,
            timeout_rate: 0.001,
            cache_hit_rate: 0.85,
            memory_usage_mb: 512.0,
            cpu_usage_percent: 65.0,
            network_bandwidth_mbps: 100.0,
            active_connections: 50,
            queue_depth: 10,
        };

        analyzer
            .record_system_metrics(metrics)
            .await
            .expect("async operation should succeed");
    }

    let trends = analyzer
        .analyze_trends()
        .await
        .expect("async operation should succeed");
    assert!(!trends.is_empty());
}

#[tokio::test]
async fn test_query_optimizer_creation() {
    let config = QueryOptimizerConfig::default();
    let optimizer = QueryPlanOptimizer::new(config);

    let stats = optimizer.get_optimization_stats().await;
    assert_eq!(stats.total_optimizations.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn test_heuristic_plan_generation() {
    let config = QueryOptimizerConfig::default();
    let optimizer = QueryPlanOptimizer::new(config);

    let complexity = QueryComplexity {
        triple_patterns: 5,
        join_count: 3,
        optional_patterns: 1,
        filter_count: 2,
        union_count: 0,
        service_count: 2,
        complexity_score: 0.7,
    };

    let services = vec!["service1".to_string(), "service2".to_string()];
    let plan = optimizer
        .optimize_query_plan("test_query", &complexity, &services)
        .await
        .expect("operation should succeed");

    assert!(plan.confidence > 0.0);
    assert!(plan.predicted_execution_time > Duration::from_millis(0));
}

#[tokio::test]
async fn test_execution_recording() {
    let config = QueryOptimizerConfig::default();
    let optimizer = QueryPlanOptimizer::new(config);

    let execution = QueryPlanExecution {
        query_hash: "test_hash".to_string(),
        plan_type: PlanType::Sequential,
        services_involved: vec!["service1".to_string()],
        execution_time: Duration::from_millis(1000),
        result_count: 100,
        memory_usage_mb: 50.0,
        cpu_usage_percent: 25.0,
        network_io_mb: 10.0,
        cache_hit_rate: 0.8,
        timestamp: SystemTime::now(),
        query_complexity: QueryComplexity {
            triple_patterns: 3,
            join_count: 1,
            optional_patterns: 0,
            filter_count: 1,
            union_count: 0,
            service_count: 1,
            complexity_score: 0.3,
        },
        success: true,
    };

    let result = optimizer.record_execution(execution).await;
    assert!(result.is_ok());
}
