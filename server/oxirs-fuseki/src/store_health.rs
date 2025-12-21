//! Store Health Monitoring
//!
//! Provides comprehensive health metrics for the RDF store, including:
//! - Store status and availability
//! - Performance metrics (query latency, throughput)
//! - Resource utilization (memory, connections)
//! - Error rates and recovery status

use crate::error::{FusekiError, FusekiResult};
use crate::store::{Store, StoreStats};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Comprehensive store health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreHealth {
    /// Overall health status
    pub status: HealthStatus,
    /// Component-level health checks
    pub components: Vec<ComponentHealth>,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Resource utilization
    pub resources: ResourceMetrics,
    /// Error metrics
    pub errors: ErrorMetrics,
    /// Last check timestamp
    pub checked_at: chrono::DateTime<chrono::Utc>,
    /// Health score (0-100)
    pub health_score: u8,
}

/// Overall health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// All systems operational
    Healthy,
    /// Some non-critical issues detected
    Degraded,
    /// Critical issues detected
    Unhealthy,
    /// System unavailable
    Down,
}

/// Component health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    /// Component status
    pub status: HealthStatus,
    /// Status message
    pub message: Option<String>,
    /// Last successful check
    pub last_success: Option<chrono::DateTime<chrono::Utc>>,
    /// Response time (milliseconds)
    pub response_time_ms: Option<u64>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average query latency (milliseconds)
    pub avg_query_latency_ms: f64,
    /// P95 query latency (milliseconds)
    pub p95_query_latency_ms: f64,
    /// P99 query latency (milliseconds)
    pub p99_query_latency_ms: f64,
    /// Queries per second
    pub queries_per_second: f64,
    /// Cache hit rate (0-1)
    pub cache_hit_rate: f64,
    /// Active query count
    pub active_queries: u32,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Memory usage (bytes)
    pub memory_used_bytes: u64,
    /// Memory usage percentage (0-100)
    pub memory_usage_percent: f64,
    /// Active connections
    pub active_connections: u32,
    /// Maximum connections
    pub max_connections: u32,
    /// Triple count
    pub triple_count: usize,
    /// Dataset count
    pub dataset_count: usize,
}

/// Error metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Total errors in last hour
    pub errors_last_hour: u64,
    /// Query failures in last hour
    pub query_failures_last_hour: u64,
    /// Update failures in last hour
    pub update_failures_last_hour: u64,
    /// Error rate (errors per second)
    pub error_rate: f64,
    /// Last error timestamp
    pub last_error: Option<chrono::DateTime<chrono::Utc>>,
    /// Last error message
    pub last_error_message: Option<String>,
}

/// Health monitor for tracking store health over time
pub struct StoreHealthMonitor {
    /// Store reference
    store: Arc<Store>,
    /// Health history
    health_history: Arc<RwLock<Vec<StoreHealth>>>,
    /// Performance tracking
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    /// Error tracking
    error_tracker: Arc<RwLock<ErrorTracker>>,
    /// Configuration
    config: HealthMonitorConfig,
}

/// Health monitor configuration
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// Maximum health history entries
    pub max_history: usize,
    /// Health check interval
    pub check_interval: Duration,
    /// Performance window for metrics
    pub performance_window: Duration,
    /// Error window for metrics
    pub error_window: Duration,
    /// Memory threshold for warnings (bytes)
    pub memory_warning_threshold: u64,
    /// Memory threshold for critical (bytes)
    pub memory_critical_threshold: u64,
    /// Maximum connections (from server config)
    pub max_connections: usize,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            max_history: 100,
            check_interval: Duration::from_secs(30),
            performance_window: Duration::from_secs(300),
            error_window: Duration::from_secs(3600),
            memory_warning_threshold: 2 * 1024 * 1024 * 1024, // 2GB
            memory_critical_threshold: 4 * 1024 * 1024 * 1024, // 4GB
            max_connections: 1000,                            // Default max connections
        }
    }
}

/// Performance tracking
#[derive(Debug, Clone, Default)]
struct PerformanceTracker {
    /// Query latencies (milliseconds)
    query_latencies: Vec<(Instant, f64)>,
    /// Query timestamps
    query_timestamps: Vec<Instant>,
    /// Active query count
    active_queries: u32,
    /// Active connection count (tracked from concurrent requests)
    active_connections: u32,
}

/// Error tracking
#[derive(Debug, Clone, Default)]
struct ErrorTracker {
    /// Error events
    errors: Vec<ErrorEvent>,
    /// Query failures
    query_failures: Vec<Instant>,
    /// Update failures
    update_failures: Vec<Instant>,
}

/// Error event
#[derive(Debug, Clone)]
struct ErrorEvent {
    timestamp: Instant,
    message: String,
    error_type: String,
}

impl StoreHealthMonitor {
    /// Create a new health monitor
    pub fn new(store: Arc<Store>) -> Self {
        Self::with_config(store, HealthMonitorConfig::default())
    }

    /// Create a new health monitor with custom configuration
    pub fn with_config(store: Arc<Store>, config: HealthMonitorConfig) -> Self {
        Self {
            store,
            health_history: Arc::new(RwLock::new(Vec::new())),
            performance_tracker: Arc::new(RwLock::new(PerformanceTracker::default())),
            error_tracker: Arc::new(RwLock::new(ErrorTracker::default())),
            config,
        }
    }

    /// Perform a comprehensive health check
    pub async fn check_health(&self) -> FusekiResult<StoreHealth> {
        let check_start = Instant::now();
        let checked_at = chrono::Utc::now();

        debug!("Performing comprehensive store health check");

        // Check store statistics
        let stats = self.store.get_stats(None).unwrap_or(StoreStats {
            triple_count: 0,
            dataset_count: 0,
            total_queries: 0,
            total_updates: 0,
            cache_hit_ratio: 0.0,
            uptime_seconds: 0,
            change_log_size: 0,
            latest_change_id: 0,
        });

        // Check components
        let components = self.check_components(&stats).await;

        // Get performance metrics
        let performance = self.get_performance_metrics(&stats).await?;

        // Get resource metrics
        let resources = self.get_resource_metrics(&stats).await?;

        // Get error metrics
        let errors = self.get_error_metrics().await?;

        // Calculate overall health status and score
        let (status, health_score) =
            self.calculate_health_status(&components, &performance, &resources, &errors);

        let health = StoreHealth {
            status,
            components,
            performance,
            resources,
            errors,
            checked_at,
            health_score,
        };

        // Store in history
        self.add_to_history(health.clone()).await;

        debug!("Health check completed in {:?}", check_start.elapsed());

        Ok(health)
    }

    /// Check individual components
    async fn check_components(&self, stats: &StoreStats) -> Vec<ComponentHealth> {
        let mut components = Vec::new();

        // Check default store
        let store_check_start = Instant::now();
        let store_status = if stats.triple_count == 0 && stats.total_queries == 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        components.push(ComponentHealth {
            name: "default_store".to_string(),
            status: store_status,
            message: Some(format!("{} triples", stats.triple_count)),
            last_success: Some(chrono::Utc::now()),
            response_time_ms: Some(store_check_start.elapsed().as_millis() as u64),
        });

        // Check query engine
        components.push(ComponentHealth {
            name: "query_engine".to_string(),
            status: HealthStatus::Healthy,
            message: Some(format!("{} queries executed", stats.total_queries)),
            last_success: Some(chrono::Utc::now()),
            response_time_ms: Some(1),
        });

        // Check datasets
        let dataset_status = if stats.dataset_count > 0 {
            HealthStatus::Healthy
        } else {
            HealthStatus::Degraded
        };

        components.push(ComponentHealth {
            name: "datasets".to_string(),
            status: dataset_status,
            message: Some(format!("{} datasets", stats.dataset_count)),
            last_success: Some(chrono::Utc::now()),
            response_time_ms: Some(1),
        });

        components
    }

    /// Get performance metrics
    async fn get_performance_metrics(
        &self,
        stats: &StoreStats,
    ) -> FusekiResult<PerformanceMetrics> {
        let tracker = self.performance_tracker.read().await;
        let now = Instant::now();
        let window_start = now - self.config.performance_window;

        // Filter recent latencies
        let recent_latencies: Vec<f64> = tracker
            .query_latencies
            .iter()
            .filter(|(ts, _)| *ts >= window_start)
            .map(|(_, latency)| *latency)
            .collect();

        // Calculate latency metrics
        let avg_query_latency_ms = if !recent_latencies.is_empty() {
            recent_latencies.iter().sum::<f64>() / recent_latencies.len() as f64
        } else {
            0.0
        };

        let mut sorted_latencies = recent_latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p95_query_latency_ms = if !sorted_latencies.is_empty() {
            let idx = (sorted_latencies.len() as f64 * 0.95) as usize;
            sorted_latencies.get(idx).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        let p99_query_latency_ms = if !sorted_latencies.is_empty() {
            let idx = (sorted_latencies.len() as f64 * 0.99) as usize;
            sorted_latencies.get(idx).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        // Calculate queries per second
        let recent_queries = tracker
            .query_timestamps
            .iter()
            .filter(|ts| **ts >= window_start)
            .count();

        let queries_per_second =
            recent_queries as f64 / self.config.performance_window.as_secs_f64();

        Ok(PerformanceMetrics {
            avg_query_latency_ms,
            p95_query_latency_ms,
            p99_query_latency_ms,
            queries_per_second,
            cache_hit_rate: stats.cache_hit_ratio,
            active_queries: tracker.active_queries,
        })
    }

    /// Get resource metrics
    async fn get_resource_metrics(&self, stats: &StoreStats) -> FusekiResult<ResourceMetrics> {
        // Get system memory info
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();

        let memory_used_bytes = sys.used_memory();
        let total_memory = sys.total_memory();
        let memory_usage_percent = if total_memory > 0 {
            (memory_used_bytes as f64 / total_memory as f64) * 100.0
        } else {
            0.0
        };

        // Get active connections from performance tracker
        // This is tracked based on concurrent query/update requests
        let active_connections = self.performance_tracker.read().await.active_connections;

        Ok(ResourceMetrics {
            memory_used_bytes,
            memory_usage_percent,
            active_connections,
            max_connections: self.config.max_connections as u32,
            triple_count: stats.triple_count,
            dataset_count: stats.dataset_count,
        })
    }

    /// Get error metrics
    async fn get_error_metrics(&self) -> FusekiResult<ErrorMetrics> {
        let tracker = self.error_tracker.read().await;
        let now = Instant::now();
        let window_start = now - self.config.error_window;

        // Count recent errors
        let recent_errors = tracker
            .errors
            .iter()
            .filter(|e| e.timestamp >= window_start)
            .count();

        let query_failures_last_hour = tracker
            .query_failures
            .iter()
            .filter(|ts| **ts >= window_start)
            .count();

        let update_failures_last_hour = tracker
            .update_failures
            .iter()
            .filter(|ts| **ts >= window_start)
            .count();

        let error_rate = recent_errors as f64 / self.config.error_window.as_secs_f64();

        let (last_error, last_error_message) = tracker
            .errors
            .last()
            .map(|e| {
                (
                    Some(
                        chrono::Utc::now()
                            - chrono::Duration::seconds((now - e.timestamp).as_secs() as i64),
                    ),
                    Some(e.message.clone()),
                )
            })
            .unwrap_or((None, None));

        Ok(ErrorMetrics {
            errors_last_hour: recent_errors as u64,
            query_failures_last_hour: query_failures_last_hour as u64,
            update_failures_last_hour: update_failures_last_hour as u64,
            error_rate,
            last_error,
            last_error_message,
        })
    }

    /// Calculate overall health status and score
    fn calculate_health_status(
        &self,
        components: &[ComponentHealth],
        performance: &PerformanceMetrics,
        resources: &ResourceMetrics,
        errors: &ErrorMetrics,
    ) -> (HealthStatus, u8) {
        let mut score = 100u8;

        // Check component health
        let unhealthy_components = components
            .iter()
            .filter(|c| c.status == HealthStatus::Unhealthy)
            .count();
        let degraded_components = components
            .iter()
            .filter(|c| c.status == HealthStatus::Degraded)
            .count();

        if unhealthy_components > 0 {
            score = score.saturating_sub(40);
        }
        score = score.saturating_sub((degraded_components * 10) as u8);

        // Check performance
        if performance.avg_query_latency_ms > 1000.0 {
            score = score.saturating_sub(15);
        } else if performance.avg_query_latency_ms > 500.0 {
            score = score.saturating_sub(10);
        }

        if performance.cache_hit_rate < 0.5 {
            score = score.saturating_sub(5);
        }

        // Check resources
        if resources.memory_usage_percent > 90.0 {
            score = score.saturating_sub(20);
        } else if resources.memory_usage_percent > 80.0 {
            score = score.saturating_sub(10);
        }

        // Check errors
        if errors.errors_last_hour > 100 {
            score = score.saturating_sub(20);
        } else if errors.errors_last_hour > 10 {
            score = score.saturating_sub(10);
        }

        // Determine overall status
        let status = if score >= 80 {
            HealthStatus::Healthy
        } else if score >= 60 {
            HealthStatus::Degraded
        } else if score >= 30 {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Down
        };

        (status, score)
    }

    /// Add health check to history
    async fn add_to_history(&self, health: StoreHealth) {
        let mut history = self.health_history.write().await;
        history.push(health);

        // Trim history if needed
        let history_len = history.len();
        if history_len > self.config.max_history {
            let drain_count = history_len - self.config.max_history;
            history.drain(0..drain_count);
        }
    }

    /// Get health history
    pub async fn get_health_history(&self) -> Vec<StoreHealth> {
        self.health_history.read().await.clone()
    }

    /// Record query execution
    pub async fn record_query(&self, latency_ms: f64) {
        let mut tracker = self.performance_tracker.write().await;
        let now = Instant::now();

        tracker.query_latencies.push((now, latency_ms));
        tracker.query_timestamps.push(now);

        // Clean old entries
        let window_start = now - self.config.performance_window;
        tracker
            .query_latencies
            .retain(|(ts, _)| *ts >= window_start);
        tracker.query_timestamps.retain(|ts| *ts >= window_start);
    }

    /// Increment active connection count
    pub async fn connection_started(&self) {
        let mut tracker = self.performance_tracker.write().await;
        tracker.active_connections = tracker.active_connections.saturating_add(1);
    }

    /// Decrement active connection count
    pub async fn connection_ended(&self) {
        let mut tracker = self.performance_tracker.write().await;
        tracker.active_connections = tracker.active_connections.saturating_sub(1);
    }

    /// Increment active query count
    pub async fn query_started(&self) {
        let mut tracker = self.performance_tracker.write().await;
        tracker.active_queries = tracker.active_queries.saturating_add(1);
    }

    /// Decrement active query count
    pub async fn query_ended(&self) {
        let mut tracker = self.performance_tracker.write().await;
        tracker.active_queries = tracker.active_queries.saturating_sub(1);
    }

    /// Record query error
    pub async fn record_query_error(&self, message: String) {
        let mut tracker = self.error_tracker.write().await;
        let now = Instant::now();

        tracker.errors.push(ErrorEvent {
            timestamp: now,
            message: message.clone(),
            error_type: "query".to_string(),
        });

        tracker.query_failures.push(now);

        // Clean old entries
        let window_start = now - self.config.error_window;
        tracker.errors.retain(|e| e.timestamp >= window_start);
        tracker.query_failures.retain(|ts| *ts >= window_start);

        warn!("Query error recorded: {}", message);
    }

    /// Record update error
    pub async fn record_update_error(&self, message: String) {
        let mut tracker = self.error_tracker.write().await;
        let now = Instant::now();

        tracker.errors.push(ErrorEvent {
            timestamp: now,
            message: message.clone(),
            error_type: "update".to_string(),
        });

        tracker.update_failures.push(now);

        // Clean old entries
        let window_start = now - self.config.error_window;
        tracker.errors.retain(|e| e.timestamp >= window_start);
        tracker.update_failures.retain(|ts| *ts >= window_start);

        warn!("Update error recorded: {}", message);
    }

    /// Start background health monitoring
    pub fn start_monitoring(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.config.check_interval);

            loop {
                interval.tick().await;

                if let Err(e) = self.check_health().await {
                    warn!("Health check failed: {}", e);
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor_creation() {
        let store = Arc::new(Store::new().unwrap());
        let monitor = StoreHealthMonitor::new(store);

        let health = monitor.check_health().await.unwrap();
        assert_eq!(health.status, HealthStatus::Degraded); // No data yet
        assert!(health.health_score <= 100);
    }

    #[tokio::test]
    async fn test_record_query() {
        let store = Arc::new(Store::new().unwrap());
        let monitor = StoreHealthMonitor::new(store);

        monitor.record_query(50.0).await;
        monitor.record_query(75.0).await;

        let health = monitor.check_health().await.unwrap();
        assert!(health.performance.avg_query_latency_ms > 0.0);
    }

    #[tokio::test]
    async fn test_record_error() {
        let store = Arc::new(Store::new().unwrap());
        let monitor = StoreHealthMonitor::new(store);

        monitor.record_query_error("Test error".to_string()).await;

        let health = monitor.check_health().await.unwrap();
        assert!(health.errors.errors_last_hour > 0);
    }

    #[tokio::test]
    async fn test_health_history() {
        let store = Arc::new(Store::new().unwrap());
        let monitor = StoreHealthMonitor::new(store);

        monitor.check_health().await.unwrap();
        monitor.check_health().await.unwrap();

        let history = monitor.get_health_history().await;
        assert_eq!(history.len(), 2);
    }
}
