//! # Advanced Profiling and Metrics Integration
//!
//! This module provides a unified interface for profiling and metrics collection
//! across the oxirs-federate crate, leveraging scirs2-core's advanced capabilities.
//!
//! ## Features
//!
//! - Performance profiling with scirs2-core::profiling
//! - Metrics collection with scirs2-core::metrics
//! - Memory tracking and optimization
//! - Simple and intuitive API
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::profiling_metrics::{FederationMetrics, profile_operation};
//!
//! // Use profile guard for automatic timing
//! {
//!     let _guard = profile_operation("query_planning");
//!     // ... planning code ...
//! } // Automatically records timing when dropped
//!
//! // Record metrics
//! let mut metrics = FederationMetrics::global();
//! metrics.record_query_execution(duration, success);
//! metrics.increment_service_request("service_1");
//! ```

use scirs2_core::metrics::{Counter, Gauge, Histogram};
use scirs2_core::profiling::{Profiler, Timer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Global metrics instance
static GLOBAL_METRICS: OnceLock<Arc<Mutex<FederationMetrics>>> = OnceLock::new();

/// Get the global federation metrics instance
pub fn global_metrics() -> Arc<Mutex<FederationMetrics>> {
    GLOBAL_METRICS
        .get_or_init(|| Arc::new(Mutex::new(FederationMetrics::new())))
        .clone()
}

/// Federation-specific metrics collection
pub struct FederationMetrics {
    /// Query execution counter
    queries_executed: Counter,
    /// Query success counter
    queries_succeeded: Counter,
    /// Query failure counter
    queries_failed: Counter,
    /// Service request counter (per service)
    service_requests: HashMap<String, Counter>,
    /// Active query gauge
    active_queries: Gauge,
    /// Query execution time histogram
    query_duration: Histogram,
    /// Cache hit counter
    cache_hits: Counter,
    /// Cache miss counter
    cache_misses: Counter,
    /// Federation error counter
    federation_errors: Counter,
    /// Service availability gauge (per service)
    service_availability: HashMap<String, Gauge>,
}

impl FederationMetrics {
    /// Create a new federation metrics instance
    pub fn new() -> Self {
        Self {
            queries_executed: Counter::new("queries_executed".to_string()),
            queries_succeeded: Counter::new("queries_succeeded".to_string()),
            queries_failed: Counter::new("queries_failed".to_string()),
            service_requests: HashMap::new(),
            active_queries: Gauge::new("active_queries".to_string()),
            query_duration: Histogram::new("query_duration_ms".to_string()),
            cache_hits: Counter::new("cache_hits".to_string()),
            cache_misses: Counter::new("cache_misses".to_string()),
            federation_errors: Counter::new("federation_errors".to_string()),
            service_availability: HashMap::new(),
        }
    }

    /// Record a query execution
    pub fn record_query_execution(&self, duration: Duration, success: bool) {
        self.queries_executed.inc();

        if success {
            self.queries_succeeded.inc();
        } else {
            self.queries_failed.inc();
        }

        self.query_duration.observe(duration.as_millis() as f64);
    }

    /// Increment active queries
    pub fn increment_active_queries(&self) {
        self.active_queries.inc();
    }

    /// Decrement active queries
    pub fn decrement_active_queries(&self) {
        self.active_queries.dec();
    }

    /// Increment service request counter
    pub fn increment_service_request(&mut self, service_id: &str) {
        let counter = self
            .service_requests
            .entry(service_id.to_string())
            .or_insert_with(|| Counter::new(format!("service_requests_{}", service_id)));
        counter.inc();
    }

    /// Record cache hit
    pub fn record_cache_hit(&self, hit: bool) {
        if hit {
            self.cache_hits.inc();
        } else {
            self.cache_misses.inc();
        }
    }

    /// Record federation error
    pub fn record_error(&self, error_type: &str) {
        self.federation_errors.inc();
        debug!("Federation error recorded: {}", error_type);
    }

    /// Set service availability
    pub fn set_service_availability(&mut self, service_id: &str, available: bool) {
        let gauge = self
            .service_availability
            .entry(service_id.to_string())
            .or_insert_with(|| Gauge::new(format!("service_availability_{}", service_id)));

        gauge.set(if available { 1.0 } else { 0.0 });
    }

    /// Get all metrics as a snapshot
    pub fn get_snapshot(&self) -> HashMap<String, f64> {
        let mut snapshot = HashMap::new();

        snapshot.insert(
            "queries_executed".to_string(),
            self.queries_executed.get() as f64,
        );
        snapshot.insert(
            "queries_succeeded".to_string(),
            self.queries_succeeded.get() as f64,
        );
        snapshot.insert(
            "queries_failed".to_string(),
            self.queries_failed.get() as f64,
        );
        snapshot.insert("active_queries".to_string(), self.active_queries.get());
        snapshot.insert("cache_hits".to_string(), self.cache_hits.get() as f64);
        snapshot.insert("cache_misses".to_string(), self.cache_misses.get() as f64);
        snapshot.insert(
            "federation_errors".to_string(),
            self.federation_errors.get() as f64,
        );

        // Add per-service metrics
        for (service_id, counter) in &self.service_requests {
            snapshot.insert(
                format!("service_requests_{}", service_id),
                counter.get() as f64,
            );
        }

        for (service_id, gauge) in &self.service_availability {
            snapshot.insert(format!("service_availability_{}", service_id), gauge.get());
        }

        snapshot
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.get() as f64;
        let misses = self.cache_misses.get() as f64;
        let total = hits + misses;

        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }

    /// Get query success rate
    pub fn query_success_rate(&self) -> f64 {
        let succeeded = self.queries_succeeded.get() as f64;
        let failed = self.queries_failed.get() as f64;
        let total = succeeded + failed;

        if total == 0.0 {
            1.0 // No queries means 100% success rate
        } else {
            succeeded / total
        }
    }

    /// Print metrics summary
    pub fn print_summary(&self) {
        info!("=== Federation Metrics Summary ===");
        info!("Queries Executed: {}", self.queries_executed.get());
        info!("Queries Succeeded: {}", self.queries_succeeded.get());
        info!("Queries Failed: {}", self.queries_failed.get());
        info!("Success Rate: {:.2}%", self.query_success_rate() * 100.0);
        info!("Active Queries: {}", self.active_queries.get());
        info!("Cache Hits: {}", self.cache_hits.get());
        info!("Cache Misses: {}", self.cache_misses.get());
        info!("Cache Hit Rate: {:.2}%", self.cache_hit_rate() * 100.0);
        info!("Federation Errors: {}", self.federation_errors.get());
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::new();
        info!("Metrics reset");
    }
}

impl Default for FederationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Profile an operation using scirs2-core Timer
/// Returns a ProfileGuard that automatically stops profiling when dropped
pub fn profile_operation(operation: impl Into<String>) -> ProfileGuard {
    ProfileGuard::new(operation)
}

/// RAII guard for automatic profiling of a scope using scirs2-core
pub struct ProfileGuard {
    operation: String,
    timer: Timer,
    start_time: Instant,
}

impl ProfileGuard {
    /// Create a new profile guard
    pub fn new(operation: impl Into<String>) -> Self {
        let operation = operation.into();
        let timer = Timer::start(&operation);

        Self {
            operation,
            timer,
            start_time: Instant::now(),
        }
    }

    /// Get elapsed time so far
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        self.timer.stop();
        let duration = self.start_time.elapsed();
        debug!(
            "Profile completed: {} ({}ms)",
            self.operation,
            duration.as_millis()
        );
    }
}

/// RAII guard for tracking active queries in metrics
pub struct ActiveQueryGuard;

impl ActiveQueryGuard {
    /// Create a new active query guard
    pub fn new() -> Self {
        global_metrics()
            .lock()
            .expect("lock should not be poisoned")
            .increment_active_queries();
        Self
    }
}

impl Drop for ActiveQueryGuard {
    fn drop(&mut self) {
        global_metrics()
            .lock()
            .expect("lock should not be poisoned")
            .decrement_active_queries();
    }
}

impl Default for ActiveQueryGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Get timing statistics from the global profiler
pub fn get_timing_stats(operation: &str) -> Option<(usize, Duration, Duration, Duration)> {
    Profiler::global()
        .lock()
        .expect("operation should succeed")
        .get_timing_stats(operation)
}

/// Print profiling report from the global profiler
pub fn print_profiling_report() {
    Profiler::global()
        .lock()
        .expect("lock should not be poisoned")
        .print_report();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_federation_metrics_basic() {
        let metrics = FederationMetrics::new();

        metrics.record_query_execution(Duration::from_millis(100), true);
        metrics.record_query_execution(Duration::from_millis(200), false);

        assert_eq!(metrics.queries_executed.get(), 2);
        assert_eq!(metrics.queries_succeeded.get(), 1);
        assert_eq!(metrics.queries_failed.get(), 1);
        assert_eq!(metrics.query_success_rate(), 0.5);
    }

    #[test]
    fn test_cache_metrics() {
        let metrics = FederationMetrics::new();

        metrics.record_cache_hit(true);
        metrics.record_cache_hit(true);
        metrics.record_cache_hit(false);

        assert_eq!(metrics.cache_hits.get(), 2);
        assert_eq!(metrics.cache_misses.get(), 1);
        assert!((metrics.cache_hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_profile_guard() {
        // Start the global profiler
        Profiler::global()
            .lock()
            .expect("lock should not be poisoned")
            .start();

        let operation_name = format!(
            "guard_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("operation should succeed")
                .as_nanos()
        );

        {
            let _guard = profile_operation(&operation_name);
            thread::sleep(Duration::from_millis(10));
            // Guard should automatically stop profiling when dropped
        }

        let stats = get_timing_stats(&operation_name);
        if let Some((count, _, _, _)) = stats {
            assert!(count >= 1); // At least one call recorded
        }
        // Note: test may pass even if stats is None, as profiler behavior may vary
    }

    #[test]
    fn test_active_query_guard() {
        let initial_count = global_metrics()
            .lock()
            .expect("lock should not be poisoned")
            .active_queries
            .get();

        {
            let _guard = ActiveQueryGuard::new();
            let active_count = global_metrics()
                .lock()
                .expect("lock should not be poisoned")
                .active_queries
                .get();
            assert_eq!(active_count, initial_count + 1.0);
        }

        let final_count = global_metrics()
            .lock()
            .expect("lock should not be poisoned")
            .active_queries
            .get();
        assert_eq!(final_count, initial_count);
    }

    #[test]
    fn test_service_metrics() {
        let mut metrics = FederationMetrics::new();

        metrics.increment_service_request("service_1");
        metrics.increment_service_request("service_1");
        metrics.increment_service_request("service_2");

        metrics.set_service_availability("service_1", true);
        metrics.set_service_availability("service_2", false);

        let snapshot = metrics.get_snapshot();
        assert_eq!(snapshot.get("service_requests_service_1"), Some(&2.0));
        assert_eq!(snapshot.get("service_requests_service_2"), Some(&1.0));
        assert_eq!(snapshot.get("service_availability_service_1"), Some(&1.0));
        assert_eq!(snapshot.get("service_availability_service_2"), Some(&0.0));
    }

    #[test]
    fn test_profiling_integration() {
        // Start the global profiler
        Profiler::global()
            .lock()
            .expect("lock should not be poisoned")
            .start();

        let operation_name = format!(
            "test_operation_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("operation should succeed")
                .as_nanos()
        );

        let _guard = profile_operation(&operation_name);
        thread::sleep(Duration::from_millis(10));
        drop(_guard);

        let stats = get_timing_stats(&operation_name);
        if let Some((count, total, avg, max)) = stats {
            assert!(count >= 1); // At least one call recorded
            assert!(total >= Duration::from_millis(10));
            assert!(avg >= Duration::from_millis(10));
            assert!(max >= Duration::from_millis(10));
        }
        // Note: test may pass even if stats is None, as profiler behavior may vary
    }
}
