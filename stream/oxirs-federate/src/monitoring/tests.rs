//! Tests for federation monitoring functionality

use super::*;
use std::time::Duration;

#[tokio::test]
async fn test_monitor_creation() {
    let monitor = FederationMonitor::new();
    let stats = monitor.get_stats().await;

    assert_eq!(stats.total_queries, 0);
    assert_eq!(stats.success_rate, 0.0);
}

#[tokio::test]
async fn test_query_recording() {
    let monitor = FederationMonitor::new();

    monitor
        .record_query_execution("sparql", Duration::from_millis(100), true)
        .await;
    monitor
        .record_query_execution("graphql", Duration::from_millis(200), false)
        .await;

    let stats = monitor.get_stats().await;
    assert_eq!(stats.total_queries, 2);
    assert_eq!(stats.successful_queries, 1);
    assert_eq!(stats.failed_queries, 1);
    assert_eq!(stats.success_rate, 0.5);
}

#[tokio::test]
async fn test_service_metrics() {
    let monitor = FederationMonitor::new();

    monitor
        .record_service_interaction("service1", Duration::from_millis(150), true, Some(1024))
        .await;
    monitor
        .record_service_interaction("service1", Duration::from_millis(250), false, None)
        .await;

    let stats = monitor.get_stats().await;
    assert!(stats.service_metrics.contains_key("service1"));

    let service_metrics = &stats.service_metrics["service1"];
    assert_eq!(service_metrics.total_requests, 2);
    assert_eq!(service_metrics.successful_requests, 1);
    assert_eq!(service_metrics.failed_requests, 1);
}

#[tokio::test]
async fn test_cache_metrics() {
    let monitor = FederationMonitor::new();

    monitor.record_cache_hit("query_cache", true).await;
    monitor.record_cache_hit("query_cache", true).await;
    monitor.record_cache_hit("query_cache", false).await;

    let stats = monitor.get_stats().await;
    assert!(stats.cache_metrics.contains_key("query_cache"));

    let cache_metrics = &stats.cache_metrics["query_cache"];
    assert_eq!(cache_metrics.total_requests, 3);
    assert_eq!(cache_metrics.hits, 2);
    assert_eq!(cache_metrics.misses, 1);
    assert!((cache_metrics.hit_rate - 0.666).abs() < 0.01);
}

#[tokio::test]
async fn test_event_recording() {
    let monitor = FederationMonitor::new();

    monitor
        .record_federation_event(FederationEventType::ServiceRegistered, "New service added")
        .await;
    monitor
        .record_federation_event(FederationEventType::Error, "Query failed")
        .await;

    let stats = monitor.get_stats().await;
    assert_eq!(stats.recent_events_count, 2);
    assert_eq!(
        stats.event_type_counts[&FederationEventType::ServiceRegistered],
        1
    );
    assert_eq!(stats.event_type_counts[&FederationEventType::Error], 1);
}

#[tokio::test]
async fn test_prometheus_export() {
    let monitor = FederationMonitor::new();

    monitor
        .record_query_execution("sparql", Duration::from_millis(100), true)
        .await;

    let prometheus_output = monitor.export_prometheus_metrics().await;
    assert!(prometheus_output.contains("federation_queries_total"));
    assert!(prometheus_output.contains("federation_queries_successful"));
}

#[tokio::test]
async fn test_health_metrics() {
    let monitor = FederationMonitor::new();

    // Record some successful queries
    for _ in 0..9 {
        monitor
            .record_query_execution("sparql", Duration::from_millis(100), true)
            .await;
    }
    // Record one failed query
    monitor
        .record_query_execution("sparql", Duration::from_millis(100), false)
        .await;

    let health = monitor.get_health_metrics().await;
    assert_eq!(health.overall_health, HealthStatus::Healthy); // 10% error rate is still healthy
}