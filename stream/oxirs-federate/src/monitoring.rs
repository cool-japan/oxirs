//! Federation Monitoring and Metrics
//!
//! This module provides comprehensive monitoring and metrics collection for federated
//! query processing, including performance tracking, error monitoring, and observability.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Federation performance monitor
#[derive(Debug)]
pub struct FederationMonitor {
    metrics: Arc<RwLock<FederationMetrics>>,
    config: FederationMonitorConfig,
    start_time: Instant,
}

impl FederationMonitor {
    /// Create a new federation monitor
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(FederationMetrics::new())),
            config: FederationMonitorConfig::default(),
            start_time: Instant::now(),
        }
    }

    /// Create a new federation monitor with custom configuration
    pub fn with_config(config: FederationMonitorConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(FederationMetrics::new())),
            config,
            start_time: Instant::now(),
        }
    }

    /// Record a query execution
    pub async fn record_query_execution(
        &self,
        query_type: &str,
        duration: Duration,
        success: bool,
    ) {
        let mut metrics = self.metrics.write().await;

        metrics.total_queries += 1;
        if success {
            metrics.successful_queries += 1;
        } else {
            metrics.failed_queries += 1;
        }

        // Update type-specific metrics
        let type_metrics = metrics
            .query_type_metrics
            .entry(query_type.to_string())
            .or_insert_with(QueryTypeMetrics::new);
        type_metrics.total_count += 1;
        type_metrics.total_duration += duration;
        type_metrics.avg_duration = type_metrics.total_duration / type_metrics.total_count as u32;

        if success {
            type_metrics.success_count += 1;
        } else {
            type_metrics.error_count += 1;
        }

        // Update response time histogram
        self.update_response_time_histogram(&mut metrics, duration)
            .await;

        // Track recent queries for trend analysis
        let query_record = QueryRecord {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            query_type: query_type.to_string(),
            duration,
            success,
        };

        metrics.recent_queries.push(query_record);

        // Keep only recent queries (sliding window)
        if metrics.recent_queries.len() > self.config.max_recent_queries {
            metrics.recent_queries.remove(0);
        }

        debug!(
            "Recorded {} query execution: {}ms, success: {}",
            query_type,
            duration.as_millis(),
            success
        );
    }

    /// Record service interaction metrics
    pub async fn record_service_interaction(
        &self,
        service_id: &str,
        duration: Duration,
        success: bool,
        response_size: Option<usize>,
    ) {
        let mut metrics = self.metrics.write().await;

        let service_metrics = metrics
            .service_metrics
            .entry(service_id.to_string())
            .or_insert_with(ServiceMetrics::new);

        service_metrics.total_requests += 1;
        service_metrics.total_duration += duration;
        service_metrics.avg_duration =
            service_metrics.total_duration / service_metrics.total_requests as u32;

        if success {
            service_metrics.successful_requests += 1;
        } else {
            service_metrics.failed_requests += 1;
        }

        if let Some(size) = response_size {
            service_metrics.total_response_size += size as u64;
            service_metrics.avg_response_size = service_metrics.total_response_size
                / service_metrics.successful_requests.max(1) as u64;
        }

        // Update last seen timestamp
        service_metrics.last_seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        debug!(
            "Recorded service interaction for {}: {}ms, success: {}",
            service_id,
            duration.as_millis(),
            success
        );
    }

    /// Record federation-specific events
    pub async fn record_federation_event(&self, event_type: FederationEventType, details: &str) {
        let mut metrics = self.metrics.write().await;

        let event = FederationEvent {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            event_type,
            details: details.to_string(),
        };

        metrics.federation_events.push(event);

        // Keep only recent events
        if metrics.federation_events.len() > self.config.max_recent_events {
            metrics.federation_events.remove(0);
        }

        // Update event type counters
        *metrics.event_type_counts.entry(event_type).or_insert(0) += 1;

        info!("Federation event: {:?} - {}", event_type, details);
    }

    /// Record cache statistics
    pub async fn record_cache_hit(&self, cache_type: &str, hit: bool) {
        let mut metrics = self.metrics.write().await;

        let cache_metrics = metrics
            .cache_metrics
            .entry(cache_type.to_string())
            .or_insert_with(CacheMetrics::new);

        cache_metrics.total_requests += 1;
        if hit {
            cache_metrics.hits += 1;
        } else {
            cache_metrics.misses += 1;
        }

        cache_metrics.hit_rate = if cache_metrics.total_requests > 0 {
            cache_metrics.hits as f64 / cache_metrics.total_requests as f64
        } else {
            0.0
        };

        debug!(
            "Cache {} for {}: hit rate {:.2}%",
            if hit { "hit" } else { "miss" },
            cache_type,
            cache_metrics.hit_rate * 100.0
        );
    }

    /// Get comprehensive monitoring statistics
    pub async fn get_stats(&self) -> MonitorStats {
        let metrics = self.metrics.read().await;
        let uptime = self.start_time.elapsed();

        MonitorStats {
            uptime,
            total_queries: metrics.total_queries,
            successful_queries: metrics.successful_queries,
            failed_queries: metrics.failed_queries,
            success_rate: if metrics.total_queries > 0 {
                metrics.successful_queries as f64 / metrics.total_queries as f64
            } else {
                0.0
            },
            query_type_metrics: metrics.query_type_metrics.clone(),
            service_metrics: metrics.service_metrics.clone(),
            cache_metrics: metrics.cache_metrics.clone(),
            response_time_histogram: metrics.response_time_histogram.clone(),
            recent_events_count: metrics.federation_events.len(),
            event_type_counts: metrics.event_type_counts.clone(),
            avg_queries_per_second: if uptime.as_secs() > 0 {
                metrics.total_queries as f64 / uptime.as_secs() as f64
            } else {
                0.0
            },
        }
    }

    /// Get health metrics for monitoring systems
    pub async fn get_health_metrics(&self) -> HealthMetrics {
        let metrics = self.metrics.read().await;
        let stats = self.calculate_health_indicators(&metrics).await;

        HealthMetrics {
            overall_health: stats.overall_health,
            service_health: stats.service_health,
            error_rate: stats.error_rate,
            avg_response_time: stats.avg_response_time,
            active_services: metrics.service_metrics.len(),
            recent_error_count: stats.recent_error_count,
            cache_hit_rate: stats.cache_hit_rate,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus_metrics(&self) -> String {
        let metrics = self.metrics.read().await;
        let mut output = String::new();

        // Total queries metric
        output
            .push_str("# HELP oxirs_federation_queries_total Total number of federated queries\n");
        output.push_str("# TYPE oxirs_federation_queries_total counter\n");
        output.push_str(&format!(
            "oxirs_federation_queries_total {}\n",
            metrics.total_queries
        ));

        // Success rate metric
        let success_rate = if metrics.total_queries > 0 {
            metrics.successful_queries as f64 / metrics.total_queries as f64
        } else {
            0.0
        };
        output.push_str("# HELP oxirs_federation_success_rate Query success rate\n");
        output.push_str("# TYPE oxirs_federation_success_rate gauge\n");
        output.push_str(&format!(
            "oxirs_federation_success_rate {:.4}\n",
            success_rate
        ));

        // Query type metrics
        output.push_str("# HELP oxirs_federation_queries_by_type_total Queries by type\n");
        output.push_str("# TYPE oxirs_federation_queries_by_type_total counter\n");
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            output.push_str(&format!(
                "oxirs_federation_queries_by_type_total{{type=\"{}\"}} {}\n",
                query_type, type_metrics.total_count
            ));
        }

        // Average response time by type
        output.push_str("# HELP oxirs_federation_avg_response_time_seconds Average response time by query type\n");
        output.push_str("# TYPE oxirs_federation_avg_response_time_seconds gauge\n");
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            output.push_str(&format!(
                "oxirs_federation_avg_response_time_seconds{{type=\"{}\"}} {:.6}\n",
                query_type,
                type_metrics.avg_duration.as_secs_f64()
            ));
        }

        // Service metrics
        output.push_str("# HELP oxirs_federation_service_requests_total Requests per service\n");
        output.push_str("# TYPE oxirs_federation_service_requests_total counter\n");
        for (service_id, service_metrics) in &metrics.service_metrics {
            output.push_str(&format!(
                "oxirs_federation_service_requests_total{{service=\"{}\"}} {}\n",
                service_id, service_metrics.total_requests
            ));
        }

        // Cache hit rates
        output.push_str("# HELP oxirs_federation_cache_hit_rate Cache hit rate by type\n");
        output.push_str("# TYPE oxirs_federation_cache_hit_rate gauge\n");
        for (cache_type, cache_metrics) in &metrics.cache_metrics {
            output.push_str(&format!(
                "oxirs_federation_cache_hit_rate{{cache=\"{}\"}} {:.4}\n",
                cache_type, cache_metrics.hit_rate
            ));
        }

        output
    }

    /// Update response time histogram
    async fn update_response_time_histogram(
        &self,
        metrics: &mut FederationMetrics,
        duration: Duration,
    ) {
        let millis = duration.as_millis() as u64;

        let bucket = if millis < 10 {
            "0-10ms"
        } else if millis < 50 {
            "10-50ms"
        } else if millis < 100 {
            "50-100ms"
        } else if millis < 500 {
            "100-500ms"
        } else if millis < 1000 {
            "500ms-1s"
        } else if millis < 5000 {
            "1s-5s"
        } else {
            "5s+"
        };

        *metrics
            .response_time_histogram
            .entry(bucket.to_string())
            .or_insert(0) += 1;
    }

    /// Calculate health indicators
    async fn calculate_health_indicators(&self, metrics: &FederationMetrics) -> HealthIndicators {
        let error_rate = if metrics.total_queries > 0 {
            metrics.failed_queries as f64 / metrics.total_queries as f64
        } else {
            0.0
        };

        let avg_response_time = if !metrics.query_type_metrics.is_empty() {
            let total_duration: Duration = metrics
                .query_type_metrics
                .values()
                .map(|m| m.total_duration)
                .sum();
            let total_count: u64 = metrics
                .query_type_metrics
                .values()
                .map(|m| m.total_count)
                .sum();

            if total_count > 0 {
                total_duration / total_count as u32
            } else {
                Duration::from_secs(0)
            }
        } else {
            Duration::from_secs(0)
        };

        let recent_error_count = metrics.recent_queries.iter().filter(|q| !q.success).count();

        let cache_hit_rate = if !metrics.cache_metrics.is_empty() {
            let total_hit_rate: f64 = metrics.cache_metrics.values().map(|c| c.hit_rate).sum();
            total_hit_rate / metrics.cache_metrics.len() as f64
        } else {
            0.0
        };

        let overall_health = if error_rate > 0.1 {
            HealthStatus::Unhealthy
        } else if error_rate > 0.05 || avg_response_time > Duration::from_secs(5) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        // Calculate per-service health
        let mut service_health = HashMap::new();
        for (service_id, service_metrics) in &metrics.service_metrics {
            let service_error_rate = if service_metrics.total_requests > 0 {
                service_metrics.failed_requests as f64 / service_metrics.total_requests as f64
            } else {
                0.0
            };

            let health = if service_error_rate > 0.2 {
                HealthStatus::Unhealthy
            } else if service_error_rate > 0.1
                || service_metrics.avg_duration > Duration::from_secs(3)
            {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };

            service_health.insert(service_id.clone(), health);
        }

        HealthIndicators {
            overall_health,
            service_health,
            error_rate,
            avg_response_time,
            recent_error_count,
            cache_hit_rate,
        }
    }

    /// Generate performance report
    pub async fn generate_performance_report(&self) -> PerformanceReport {
        let metrics = self.metrics.read().await;
        let uptime = self.start_time.elapsed();

        // Calculate trends
        let mut query_trends = HashMap::new();
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            let trend = QueryTrend {
                query_type: query_type.clone(),
                total_queries: type_metrics.total_count,
                avg_response_time: type_metrics.avg_duration,
                error_rate: if type_metrics.total_count > 0 {
                    type_metrics.error_count as f64 / type_metrics.total_count as f64
                } else {
                    0.0
                },
                queries_per_second: if uptime.as_secs() > 0 {
                    type_metrics.total_count as f64 / uptime.as_secs() as f64
                } else {
                    0.0
                },
            };
            query_trends.insert(query_type.clone(), trend);
        }

        PerformanceReport {
            report_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            uptime,
            total_queries: metrics.total_queries,
            overall_success_rate: if metrics.total_queries > 0 {
                metrics.successful_queries as f64 / metrics.total_queries as f64
            } else {
                0.0
            },
            query_trends,
            top_errors: self.get_top_errors(&metrics).await,
            performance_summary: self.get_performance_summary(&metrics).await,
        }
    }

    async fn get_top_errors(&self, metrics: &FederationMetrics) -> Vec<ErrorSummary> {
        // Group recent errors by type/message
        let mut error_counts = HashMap::new();

        for event in &metrics.federation_events {
            if matches!(
                event.event_type,
                FederationEventType::Error | FederationEventType::ServiceFailure
            ) {
                *error_counts.entry(event.details.clone()).or_insert(0) += 1;
            }
        }

        let mut errors: Vec<ErrorSummary> = error_counts
            .into_iter()
            .map(|(message, count)| ErrorSummary { message, count })
            .collect();

        errors.sort_by(|a, b| b.count.cmp(&a.count));
        errors.truncate(10); // Top 10 errors

        errors
    }

    async fn get_performance_summary(&self, metrics: &FederationMetrics) -> PerformanceSummary {
        let total_services = metrics.service_metrics.len();
        let healthy_services = metrics
            .service_metrics
            .values()
            .filter(|s| {
                let error_rate = if s.total_requests > 0 {
                    s.failed_requests as f64 / s.total_requests as f64
                } else {
                    0.0
                };
                error_rate < 0.1
            })
            .count();

        PerformanceSummary {
            total_services,
            healthy_services,
            avg_query_time: if !metrics.query_type_metrics.is_empty() {
                let total_duration: Duration = metrics
                    .query_type_metrics
                    .values()
                    .map(|m| m.total_duration)
                    .sum();
                let total_count: u64 = metrics
                    .query_type_metrics
                    .values()
                    .map(|m| m.total_count)
                    .sum();

                if total_count > 0 {
                    total_duration / total_count as u32
                } else {
                    Duration::from_secs(0)
                }
            } else {
                Duration::from_secs(0)
            },
            cache_efficiency: if !metrics.cache_metrics.is_empty() {
                let avg_hit_rate: f64 = metrics
                    .cache_metrics
                    .values()
                    .map(|c| c.hit_rate)
                    .sum::<f64>()
                    / metrics.cache_metrics.len() as f64;
                avg_hit_rate
            } else {
                0.0
            },
        }
    }
}

impl Default for FederationMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for federation monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationMonitorConfig {
    pub enable_detailed_metrics: bool,
    pub metrics_retention_hours: u64,
    pub max_recent_queries: usize,
    pub max_recent_events: usize,
    pub enable_prometheus_export: bool,
    pub health_check_interval: Duration,
}

impl Default for FederationMonitorConfig {
    fn default() -> Self {
        Self {
            enable_detailed_metrics: true,
            metrics_retention_hours: 24,
            max_recent_queries: 1000,
            max_recent_events: 500,
            enable_prometheus_export: true,
            health_check_interval: Duration::from_secs(30),
        }
    }
}

/// Internal metrics storage
#[derive(Debug)]
struct FederationMetrics {
    total_queries: u64,
    successful_queries: u64,
    failed_queries: u64,
    query_type_metrics: HashMap<String, QueryTypeMetrics>,
    service_metrics: HashMap<String, ServiceMetrics>,
    cache_metrics: HashMap<String, CacheMetrics>,
    response_time_histogram: HashMap<String, u64>,
    federation_events: Vec<FederationEvent>,
    event_type_counts: HashMap<FederationEventType, u64>,
    recent_queries: Vec<QueryRecord>,
}

impl FederationMetrics {
    fn new() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            query_type_metrics: HashMap::new(),
            service_metrics: HashMap::new(),
            cache_metrics: HashMap::new(),
            response_time_histogram: HashMap::new(),
            federation_events: Vec::new(),
            event_type_counts: HashMap::new(),
            recent_queries: Vec::new(),
        }
    }
}

/// Metrics for specific query types
#[derive(Debug, Clone, Serialize)]
pub struct QueryTypeMetrics {
    pub total_count: u64,
    pub success_count: u64,
    pub error_count: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
}

impl QueryTypeMetrics {
    fn new() -> Self {
        Self {
            total_count: 0,
            success_count: 0,
            error_count: 0,
            total_duration: Duration::from_secs(0),
            avg_duration: Duration::from_secs(0),
        }
    }
}

/// Metrics for individual services
#[derive(Debug, Clone, Serialize)]
pub struct ServiceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub total_response_size: u64,
    pub avg_response_size: u64,
    pub last_seen: u64,
}

impl ServiceMetrics {
    fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_duration: Duration::from_secs(0),
            avg_duration: Duration::from_secs(0),
            total_response_size: 0,
            avg_response_size: 0,
            last_seen: 0,
        }
    }
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize)]
pub struct CacheMetrics {
    pub total_requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

impl CacheMetrics {
    fn new() -> Self {
        Self {
            total_requests: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
        }
    }
}

/// Types of federation events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum FederationEventType {
    QueryStart,
    QueryComplete,
    ServiceRegistered,
    ServiceUnregistered,
    ServiceFailure,
    SchemaUpdate,
    CacheInvalidation,
    Error,
    Warning,
}

/// Federation event record
#[derive(Debug, Clone)]
struct FederationEvent {
    timestamp: u64,
    event_type: FederationEventType,
    details: String,
}

/// Query execution record
#[derive(Debug, Clone)]
struct QueryRecord {
    timestamp: u64,
    query_type: String,
    duration: Duration,
    success: bool,
}

/// Health status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Monitoring statistics
#[derive(Debug, Clone, Serialize)]
pub struct MonitorStats {
    pub uptime: Duration,
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub success_rate: f64,
    pub query_type_metrics: HashMap<String, QueryTypeMetrics>,
    pub service_metrics: HashMap<String, ServiceMetrics>,
    pub cache_metrics: HashMap<String, CacheMetrics>,
    pub response_time_histogram: HashMap<String, u64>,
    pub recent_events_count: usize,
    pub event_type_counts: HashMap<FederationEventType, u64>,
    pub avg_queries_per_second: f64,
}

/// Health monitoring metrics
#[derive(Debug, Clone, Serialize)]
pub struct HealthMetrics {
    pub overall_health: HealthStatus,
    pub service_health: HashMap<String, HealthStatus>,
    pub error_rate: f64,
    pub avg_response_time: Duration,
    pub active_services: usize,
    pub recent_error_count: usize,
    pub cache_hit_rate: f64,
    pub timestamp: u64,
}

/// Internal health indicators
struct HealthIndicators {
    overall_health: HealthStatus,
    service_health: HashMap<String, HealthStatus>,
    error_rate: f64,
    avg_response_time: Duration,
    recent_error_count: usize,
    cache_hit_rate: f64,
}

/// Performance report
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceReport {
    pub report_timestamp: u64,
    pub uptime: Duration,
    pub total_queries: u64,
    pub overall_success_rate: f64,
    pub query_trends: HashMap<String, QueryTrend>,
    pub top_errors: Vec<ErrorSummary>,
    pub performance_summary: PerformanceSummary,
}

/// Query performance trend
#[derive(Debug, Clone, Serialize)]
pub struct QueryTrend {
    pub query_type: String,
    pub total_queries: u64,
    pub avg_response_time: Duration,
    pub error_rate: f64,
    pub queries_per_second: f64,
}

/// Error summary for reporting
#[derive(Debug, Clone, Serialize)]
pub struct ErrorSummary {
    pub message: String,
    pub count: u64,
}

/// Performance summary
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceSummary {
    pub total_services: usize,
    pub healthy_services: usize,
    pub avg_query_time: Duration,
    pub cache_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(prometheus_output.contains("oxirs_federation_queries_total"));
        assert!(prometheus_output.contains("oxirs_federation_success_rate"));
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
}
