//! GraphQL Performance Monitoring and Analytics
//!
//! This module provides comprehensive performance monitoring, metrics collection,
//! and analytics for GraphQL operations.

use crate::ast::{Document, OperationType};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Performance metrics for a single GraphQL operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    pub operation_name: Option<String>,
    pub operation_type: OperationType,
    pub query_hash: u64,
    pub execution_time: Duration,
    pub parsing_time: Duration,
    pub validation_time: Duration,
    pub planning_time: Duration,
    pub field_count: usize,
    pub depth: usize,
    pub complexity_score: usize,
    pub cache_hit: bool,
    pub error_count: usize,
    pub timestamp: SystemTime,
    pub client_info: ClientInfo,
}

/// Client information for request tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClientInfo {
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
}

/// Aggregated performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_requests: u64,
    pub total_errors: u64,
    pub avg_execution_time: Duration,
    pub p50_execution_time: Duration,
    pub p95_execution_time: Duration,
    pub p99_execution_time: Duration,
    pub cache_hit_ratio: f64,
    pub queries_per_second: f64,
    pub error_rate: f64,
    pub most_expensive_queries: Vec<ExpensiveQuery>,
    pub slowest_fields: Vec<SlowField>,
    pub client_stats: HashMap<String, ClientStats>,
}

/// Information about expensive queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpensiveQuery {
    pub query_hash: u64,
    pub operation_name: Option<String>,
    pub avg_execution_time: Duration,
    pub max_execution_time: Duration,
    pub execution_count: u64,
    pub complexity_score: usize,
}

/// Information about slow fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowField {
    pub field_name: String,
    pub avg_resolution_time: Duration,
    pub max_resolution_time: Duration,
    pub call_count: u64,
}

/// Per-client statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientStats {
    pub request_count: u64,
    pub error_count: u64,
    pub avg_execution_time: Duration,
    pub last_seen: SystemTime,
}

/// Real-time performance tracker
#[derive(Debug)]
pub struct PerformanceTracker {
    metrics: Arc<RwLock<VecDeque<OperationMetrics>>>,
    field_metrics: Arc<RwLock<HashMap<String, FieldMetrics>>>,
    config: PerformanceConfig,
    start_time: Instant,
}

/// Configuration for performance tracking
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub max_stored_metrics: usize,
    pub enable_detailed_tracking: bool,
    pub enable_client_tracking: bool,
    pub enable_field_tracking: bool,
    pub stats_window: Duration,
    pub expensive_query_threshold: Duration,
    pub slow_field_threshold: Duration,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_stored_metrics: 10000,
            enable_detailed_tracking: true,
            enable_client_tracking: true,
            enable_field_tracking: true,
            stats_window: Duration::from_secs(3600), // 1 hour
            expensive_query_threshold: Duration::from_millis(1000),
            slow_field_threshold: Duration::from_millis(100),
        }
    }
}

/// Field-level performance metrics
#[derive(Debug, Clone)]
struct FieldMetrics {
    call_count: u64,
    total_time: Duration,
    max_time: Duration,
    error_count: u64,
    last_called: Instant,
}

impl FieldMetrics {
    fn new() -> Self {
        Self {
            call_count: 0,
            total_time: Duration::from_secs(0),
            max_time: Duration::from_secs(0),
            error_count: 0,
            last_called: Instant::now(),
        }
    }

    fn record_call(&mut self, duration: Duration, had_error: bool) {
        self.call_count += 1;
        self.total_time += duration;
        self.max_time = self.max_time.max(duration);
        if had_error {
            self.error_count += 1;
        }
        self.last_called = Instant::now();
    }

    fn avg_time(&self) -> Duration {
        if self.call_count == 0 {
            Duration::from_secs(0)
        } else {
            self.total_time / self.call_count as u32
        }
    }
}

impl PerformanceTracker {
    /// Create a new performance tracker
    pub fn new() -> Self {
        Self::with_config(PerformanceConfig::default())
    }

    /// Create a new performance tracker with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(VecDeque::new())),
            field_metrics: Arc::new(RwLock::new(HashMap::new())),
            config,
            start_time: Instant::now(),
        }
    }

    /// Record operation metrics
    pub fn record_operation(&self, metrics: OperationMetrics) {
        if let Ok(mut storage) = self.metrics.write() {
            // Enforce size limit
            while storage.len() >= self.config.max_stored_metrics {
                storage.pop_front();
            }
            storage.push_back(metrics);
        }
    }

    /// Record field resolution metrics
    pub fn record_field_resolution(&self, field_name: &str, duration: Duration, had_error: bool) {
        if !self.config.enable_field_tracking {
            return;
        }

        if let Ok(mut field_metrics) = self.field_metrics.write() {
            let metrics = field_metrics
                .entry(field_name.to_string())
                .or_insert_with(FieldMetrics::new);
            metrics.record_call(duration, had_error);
        }
    }

    /// Get current performance statistics
    pub fn get_stats(&self) -> Result<PerformanceStats> {
        let metrics = self
            .metrics
            .read()
            .map_err(|_| anyhow::anyhow!("Lock poisoned"))?;
        let field_metrics = self
            .field_metrics
            .read()
            .map_err(|_| anyhow::anyhow!("Lock poisoned"))?;

        let cutoff_time = SystemTime::now() - self.config.stats_window;
        let recent_metrics: Vec<_> = metrics
            .iter()
            .filter(|m| m.timestamp > cutoff_time)
            .collect();

        if recent_metrics.is_empty() {
            return Ok(PerformanceStats {
                total_requests: 0,
                total_errors: 0,
                avg_execution_time: Duration::from_secs(0),
                p50_execution_time: Duration::from_secs(0),
                p95_execution_time: Duration::from_secs(0),
                p99_execution_time: Duration::from_secs(0),
                cache_hit_ratio: 0.0,
                queries_per_second: 0.0,
                error_rate: 0.0,
                most_expensive_queries: Vec::new(),
                slowest_fields: Vec::new(),
                client_stats: HashMap::new(),
            });
        }

        // Calculate basic stats
        let total_requests = recent_metrics.len() as u64;
        let total_errors = recent_metrics.iter().map(|m| m.error_count as u64).sum();
        let cache_hits = recent_metrics.iter().filter(|m| m.cache_hit).count() as u64;

        // Calculate execution time statistics
        let mut execution_times: Vec<Duration> =
            recent_metrics.iter().map(|m| m.execution_time).collect();
        execution_times.sort();

        let total_nanos = execution_times.iter().map(|d| d.as_nanos()).sum::<u128>()
            / execution_times.len() as u128;
        let avg_execution_time = Duration::from_nanos(total_nanos.min(u64::MAX as u128) as u64);

        let p50_execution_time = execution_times[execution_times.len() * 50 / 100];
        let p95_execution_time = execution_times[execution_times.len() * 95 / 100];
        let p99_execution_time = execution_times[execution_times.len() * 99 / 100];

        let cache_hit_ratio = if total_requests > 0 {
            cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let queries_per_second = total_requests as f64 / self.config.stats_window.as_secs() as f64;
        let error_rate = if total_requests > 0 {
            total_errors as f64 / total_requests as f64
        } else {
            0.0
        };

        // Find most expensive queries
        let most_expensive_queries = self.calculate_expensive_queries(&recent_metrics);

        // Find slowest fields
        let slowest_fields = self.calculate_slowest_fields(&field_metrics);

        // Calculate client stats
        let client_stats = if self.config.enable_client_tracking {
            self.calculate_client_stats(&recent_metrics)
        } else {
            HashMap::new()
        };

        Ok(PerformanceStats {
            total_requests,
            total_errors,
            avg_execution_time,
            p50_execution_time,
            p95_execution_time,
            p99_execution_time,
            cache_hit_ratio,
            queries_per_second,
            error_rate,
            most_expensive_queries,
            slowest_fields,
            client_stats,
        })
    }

    /// Clear all stored metrics
    pub fn clear_metrics(&self) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.clear();
        }
        if let Ok(mut field_metrics) = self.field_metrics.write() {
            field_metrics.clear();
        }
    }

    /// Get metrics for a specific time range
    pub fn get_metrics_in_range(
        &self,
        start: SystemTime,
        end: SystemTime,
    ) -> Result<Vec<OperationMetrics>> {
        let metrics = self
            .metrics
            .read()
            .map_err(|_| anyhow::anyhow!("Lock poisoned"))?;

        Ok(metrics
            .iter()
            .filter(|m| m.timestamp >= start && m.timestamp <= end)
            .cloned()
            .collect())
    }

    /// Export metrics to JSON
    pub fn export_metrics_json(&self) -> Result<String> {
        let stats = self.get_stats()?;
        serde_json::to_string_pretty(&stats)
            .map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    fn calculate_expensive_queries(&self, metrics: &[&OperationMetrics]) -> Vec<ExpensiveQuery> {
        let mut query_stats: HashMap<u64, (Duration, Duration, u64, usize, Option<String>)> =
            HashMap::new();

        for metric in metrics {
            if metric.execution_time >= self.config.expensive_query_threshold {
                let entry = query_stats.entry(metric.query_hash).or_insert((
                    Duration::from_secs(0),
                    Duration::from_secs(0),
                    0,
                    metric.complexity_score,
                    metric.operation_name.clone(),
                ));

                entry.0 += metric.execution_time; // total time
                entry.1 = entry.1.max(metric.execution_time); // max time
                entry.2 += 1; // count
            }
        }

        let mut expensive_queries: Vec<ExpensiveQuery> = query_stats
            .into_iter()
            .map(
                |(hash, (total_time, max_time, count, complexity, name))| ExpensiveQuery {
                    query_hash: hash,
                    operation_name: name,
                    avg_execution_time: total_time / count as u32,
                    max_execution_time: max_time,
                    execution_count: count,
                    complexity_score: complexity,
                },
            )
            .collect();

        expensive_queries.sort_by(|a, b| b.avg_execution_time.cmp(&a.avg_execution_time));
        expensive_queries.truncate(10); // Top 10
        expensive_queries
    }

    fn calculate_slowest_fields(
        &self,
        field_metrics: &HashMap<String, FieldMetrics>,
    ) -> Vec<SlowField> {
        let mut slow_fields: Vec<SlowField> = field_metrics
            .iter()
            .filter(|(_, metrics)| metrics.avg_time() >= self.config.slow_field_threshold)
            .map(|(name, metrics)| SlowField {
                field_name: name.clone(),
                avg_resolution_time: metrics.avg_time(),
                max_resolution_time: metrics.max_time,
                call_count: metrics.call_count,
            })
            .collect();

        slow_fields.sort_by(|a, b| b.avg_resolution_time.cmp(&a.avg_resolution_time));
        slow_fields.truncate(10); // Top 10
        slow_fields
    }

    fn calculate_client_stats(
        &self,
        metrics: &[&OperationMetrics],
    ) -> HashMap<String, ClientStats> {
        let mut client_stats: HashMap<String, ClientStats> = HashMap::new();

        for metric in metrics {
            if let Some(ref ip) = metric.client_info.ip_address {
                let stats = client_stats.entry(ip.clone()).or_insert(ClientStats {
                    request_count: 0,
                    error_count: 0,
                    avg_execution_time: Duration::from_secs(0),
                    last_seen: metric.timestamp,
                });

                stats.request_count += 1;
                stats.error_count += metric.error_count as u64;
                stats.last_seen = stats.last_seen.max(metric.timestamp);

                // Update running average
                let total_time = stats.avg_execution_time * (stats.request_count - 1) as u32
                    + metric.execution_time;
                stats.avg_execution_time = total_time / stats.request_count as u32;
            }
        }

        client_stats
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance monitoring middleware
pub struct PerformanceMonitor {
    tracker: Arc<PerformanceTracker>,
}

impl PerformanceMonitor {
    pub fn new(tracker: Arc<PerformanceTracker>) -> Self {
        Self { tracker }
    }

    /// Start timing an operation
    pub fn start_operation(&self) -> OperationTimer {
        OperationTimer::new(Arc::clone(&self.tracker))
    }

    /// Get current performance statistics
    pub fn get_stats(&self) -> Result<PerformanceStats> {
        self.tracker.get_stats()
    }

    /// Get the underlying tracker
    pub fn tracker(&self) -> &Arc<PerformanceTracker> {
        &self.tracker
    }
}

/// Timer for tracking operation execution
pub struct OperationTimer {
    tracker: Arc<PerformanceTracker>,
    start_time: Instant,
    parsing_time: Option<Duration>,
    validation_time: Option<Duration>,
    planning_time: Option<Duration>,
}

impl OperationTimer {
    fn new(tracker: Arc<PerformanceTracker>) -> Self {
        Self {
            tracker,
            start_time: Instant::now(),
            parsing_time: None,
            validation_time: None,
            planning_time: None,
        }
    }

    /// Mark parsing phase completion
    pub fn mark_parsing_complete(&mut self) {
        self.parsing_time = Some(self.start_time.elapsed());
    }

    /// Mark validation phase completion
    pub fn mark_validation_complete(&mut self) {
        self.validation_time =
            Some(self.start_time.elapsed() - self.parsing_time.unwrap_or_default());
    }

    /// Mark planning phase completion
    pub fn mark_planning_complete(&mut self) {
        let elapsed = self.start_time.elapsed();
        let previous =
            self.parsing_time.unwrap_or_default() + self.validation_time.unwrap_or_default();
        self.planning_time = Some(elapsed - previous);
    }

    /// Complete the operation timing
    pub fn complete(
        &self,
        document: &Document,
        complexity: crate::optimizer::QueryComplexity,
        cache_hit: bool,
        error_count: usize,
        client_info: ClientInfo,
    ) {
        let total_execution_time = self.start_time.elapsed();

        // Extract operation info
        let (operation_name, operation_type) =
            if let Some(crate::ast::Definition::Operation(op)) = document.definitions.first() {
                (op.name.clone(), op.operation_type.clone())
            } else {
                (None, OperationType::Query)
            };

        let metrics = OperationMetrics {
            operation_name,
            operation_type,
            query_hash: self.calculate_query_hash(document),
            execution_time: total_execution_time,
            parsing_time: self.parsing_time.unwrap_or_default(),
            validation_time: self.validation_time.unwrap_or_default(),
            planning_time: self.planning_time.unwrap_or_default(),
            field_count: complexity.field_count,
            depth: complexity.depth,
            complexity_score: complexity.complexity_score,
            cache_hit,
            error_count,
            timestamp: SystemTime::now(),
            client_info,
        };

        self.tracker.record_operation(metrics);
    }

    fn calculate_query_hash(&self, document: &Document) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{document:?}").hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_performance_tracker() {
        let tracker = PerformanceTracker::new();

        let metrics = OperationMetrics {
            operation_name: Some("TestQuery".to_string()),
            operation_type: OperationType::Query,
            query_hash: 12345,
            execution_time: Duration::from_millis(100),
            parsing_time: Duration::from_millis(10),
            validation_time: Duration::from_millis(5),
            planning_time: Duration::from_millis(15),
            field_count: 5,
            depth: 3,
            complexity_score: 50,
            cache_hit: false,
            error_count: 0,
            timestamp: SystemTime::now(),
            client_info: ClientInfo::default(),
        };

        tracker.record_operation(metrics);

        let stats = tracker.get_stats().unwrap();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.total_errors, 0);
    }

    #[test]
    fn test_field_metrics() {
        let tracker = PerformanceTracker::new();

        // Record an operation metric to ensure recent_metrics is not empty
        let operation_metrics = OperationMetrics {
            operation_name: Some("test_operation".to_string()),
            operation_type: OperationType::Query,
            query_hash: 12345,
            execution_time: Duration::from_millis(100),
            parsing_time: Duration::from_millis(10),
            validation_time: Duration::from_millis(5),
            planning_time: Duration::from_millis(5),
            field_count: 1,
            depth: 1,
            complexity_score: 10,
            cache_hit: false,
            error_count: 0,
            timestamp: SystemTime::now(),
            client_info: ClientInfo::default(),
        };
        tracker.record_operation(operation_metrics);

        tracker.record_field_resolution("test_field", Duration::from_millis(150), false);
        tracker.record_field_resolution("test_field", Duration::from_millis(200), true);

        let stats = tracker.get_stats().unwrap();
        assert_eq!(stats.slowest_fields.len(), 1);
        assert_eq!(stats.slowest_fields[0].field_name, "test_field");
        assert_eq!(stats.slowest_fields[0].call_count, 2);
    }

    #[test]
    fn test_operation_timer() {
        let tracker = Arc::new(PerformanceTracker::new());
        let monitor = PerformanceMonitor::new(tracker);

        let mut timer = monitor.start_operation();
        std::thread::sleep(Duration::from_millis(1));
        timer.mark_parsing_complete();

        assert!(timer.parsing_time.is_some());
        assert!(timer.parsing_time.unwrap() > Duration::from_millis(0));
    }
}
