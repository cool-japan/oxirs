//! # Query Timeout Enforcement
//!
//! Provides comprehensive query timeout enforcement to prevent runaway queries
//! from consuming excessive resources in production environments.

use scirs2_core::metrics::{Counter, Histogram};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for query timeout enforcement
#[derive(Debug, Clone)]
pub struct QueryTimeoutConfig {
    /// Default query timeout (None = no timeout)
    pub default_timeout: Option<Duration>,
    /// Maximum allowed timeout (prevents users from setting arbitrarily high timeouts)
    pub max_timeout: Duration,
    /// Grace period after timeout before forceful cancellation
    pub grace_period: Duration,
    /// Enable soft timeouts (warnings before hard timeout)
    pub enable_soft_timeouts: bool,
    /// Soft timeout threshold (% of total timeout)
    pub soft_timeout_threshold: f64,
}

impl Default for QueryTimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: Some(Duration::from_secs(30)), // 30 seconds default
            max_timeout: Duration::from_secs(300),          // 5 minutes max
            grace_period: Duration::from_secs(5),           // 5 seconds grace
            enable_soft_timeouts: true,
            soft_timeout_threshold: 0.8, // Warn at 80% of timeout
        }
    }
}

/// Query timeout status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutStatus {
    /// Query is within limits
    Ok,
    /// Approaching timeout (soft timeout reached)
    Warning,
    /// Timeout exceeded
    Exceeded,
    /// Forcefully cancelled after grace period
    Cancelled,
}

/// Query execution context with timeout tracking
pub struct QueryContext {
    /// Unique query ID
    query_id: u64,
    /// Start time
    start_time: Instant,
    /// Timeout deadline
    deadline: Option<Instant>,
    /// Cancellation flag
    cancelled: Arc<AtomicBool>,
    /// Soft timeout reached
    soft_timeout_reached: AtomicBool,
    /// Hard timeout reached
    hard_timeout_reached: AtomicBool,
    /// Configuration
    config: QueryTimeoutConfig,
}

impl QueryContext {
    /// Create a new query context
    pub fn new(
        query_id: u64,
        config: QueryTimeoutConfig,
        custom_timeout: Option<Duration>,
    ) -> Self {
        let timeout = custom_timeout
            .or(config.default_timeout)
            .map(|t| t.min(config.max_timeout));

        let deadline = timeout.map(|t| Instant::now() + t);

        Self {
            query_id,
            start_time: Instant::now(),
            deadline,
            cancelled: Arc::new(AtomicBool::new(false)),
            soft_timeout_reached: AtomicBool::new(false),
            hard_timeout_reached: AtomicBool::new(false),
            config,
        }
    }

    /// Check if query should continue execution
    pub fn check_timeout(&self) -> TimeoutStatus {
        // Check if cancelled
        if self.cancelled.load(Ordering::Relaxed) {
            return TimeoutStatus::Cancelled;
        }

        let Some(deadline) = self.deadline else {
            return TimeoutStatus::Ok;
        };

        let now = Instant::now();

        // Check hard timeout
        if now >= deadline {
            self.hard_timeout_reached.store(true, Ordering::Relaxed);
            return TimeoutStatus::Exceeded;
        }

        // Check soft timeout if enabled
        if self.config.enable_soft_timeouts && !self.soft_timeout_reached.load(Ordering::Relaxed) {
            let elapsed = now.duration_since(self.start_time);
            let total_timeout = deadline.duration_since(self.start_time);
            let threshold_duration = Duration::from_secs_f64(
                total_timeout.as_secs_f64() * self.config.soft_timeout_threshold,
            );

            if elapsed >= threshold_duration {
                self.soft_timeout_reached.store(true, Ordering::Relaxed);
                return TimeoutStatus::Warning;
            }
        }

        TimeoutStatus::Ok
    }

    /// Get elapsed time since query started
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get remaining time until timeout
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline.map(|deadline| {
            let now = Instant::now();
            if now < deadline {
                deadline - now
            } else {
                Duration::ZERO
            }
        })
    }

    /// Check if timeout has been exceeded
    pub fn is_timeout_exceeded(&self) -> bool {
        self.hard_timeout_reached.load(Ordering::Relaxed)
    }

    /// Check if query was cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Cancel the query
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Get cancellation handle for external cancellation
    pub fn cancellation_handle(&self) -> CancellationHandle {
        CancellationHandle {
            query_id: self.query_id,
            cancelled: Arc::clone(&self.cancelled),
        }
    }

    /// Get query ID
    pub fn query_id(&self) -> u64 {
        self.query_id
    }
}

/// Handle for cancelling a query from another thread
#[derive(Clone)]
pub struct CancellationHandle {
    query_id: u64,
    cancelled: Arc<AtomicBool>,
}

impl CancellationHandle {
    /// Cancel the query
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Check if query is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Get query ID
    pub fn query_id(&self) -> u64 {
        self.query_id
    }
}

/// Query timeout manager for tracking and enforcing timeouts
pub struct QueryTimeoutManager {
    /// Configuration
    config: QueryTimeoutConfig,
    /// Next query ID
    next_query_id: AtomicU64,
    /// Active query contexts
    active_queries: parking_lot::RwLock<std::collections::HashMap<u64, Arc<QueryContext>>>,
    /// Metrics
    timeout_counter: Counter,
    cancellation_counter: Counter,
    query_duration_histogram: Histogram,
}

impl QueryTimeoutManager {
    /// Create a new query timeout manager
    pub fn new(config: QueryTimeoutConfig) -> Self {
        Self {
            config,
            next_query_id: AtomicU64::new(1),
            active_queries: parking_lot::RwLock::new(std::collections::HashMap::new()),
            timeout_counter: Counter::new("query_timeouts_total".to_string()),
            cancellation_counter: Counter::new("query_cancellations_total".to_string()),
            query_duration_histogram: Histogram::new("query_duration_seconds".to_string()),
        }
    }

    /// Start a new query with timeout tracking
    pub fn start_query(&self, custom_timeout: Option<Duration>) -> Arc<QueryContext> {
        let query_id = self.next_query_id.fetch_add(1, Ordering::Relaxed);
        let context = Arc::new(QueryContext::new(
            query_id,
            self.config.clone(),
            custom_timeout,
        ));

        let mut active = self.active_queries.write();
        active.insert(query_id, Arc::clone(&context));

        context
    }

    /// Finish a query and record metrics
    pub fn finish_query(&self, query_id: u64) {
        let mut active = self.active_queries.write();
        if let Some(context) = active.remove(&query_id) {
            let duration = context.elapsed();
            self.query_duration_histogram
                .observe(duration.as_secs_f64());

            if context.is_timeout_exceeded() {
                self.timeout_counter.inc();
            }

            if context.is_cancelled() {
                self.cancellation_counter.inc();
            }
        }
    }

    /// Cancel a specific query
    pub fn cancel_query(&self, query_id: u64) -> bool {
        let active = self.active_queries.read();
        if let Some(context) = active.get(&query_id) {
            context.cancel();
            true
        } else {
            false
        }
    }

    /// Cancel all active queries
    pub fn cancel_all_queries(&self) {
        let active = self.active_queries.read();
        for context in active.values() {
            context.cancel();
        }
    }

    /// Get active query count
    pub fn active_query_count(&self) -> usize {
        self.active_queries.read().len()
    }

    /// Get active query IDs
    pub fn active_query_ids(&self) -> Vec<u64> {
        self.active_queries.read().keys().copied().collect()
    }

    /// Get query context for a specific query ID
    pub fn get_query_context(&self, query_id: u64) -> Option<Arc<QueryContext>> {
        self.active_queries.read().get(&query_id).cloned()
    }

    /// Get metrics
    pub fn get_metrics(&self) -> QueryTimeoutMetrics {
        let histogram_stats = self.query_duration_histogram.get_stats();

        QueryTimeoutMetrics {
            total_timeouts: self.timeout_counter.get(),
            total_cancellations: self.cancellation_counter.get(),
            active_queries: self.active_query_count(),
            avg_query_duration_secs: histogram_stats.mean,
        }
    }

    /// Cleanup finished queries (queries that are no longer in the active set)
    pub fn cleanup(&self) {
        let mut active = self.active_queries.write();
        active.retain(|_, context| {
            // Keep only queries that haven't been explicitly finished
            // In a real implementation, we'd have a way to mark queries as finished
            !context.is_cancelled()
        });
    }
}

/// Metrics for query timeout manager
#[derive(Debug, Clone)]
pub struct QueryTimeoutMetrics {
    /// Total queries that timed out
    pub total_timeouts: u64,
    /// Total queries that were cancelled
    pub total_cancellations: u64,
    /// Currently active queries
    pub active_queries: usize,
    /// Average query duration (seconds)
    pub avg_query_duration_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_context_creation() {
        let config = QueryTimeoutConfig::default();
        let context = QueryContext::new(1, config, None);

        assert_eq!(context.query_id(), 1);
        assert!(!context.is_timeout_exceeded());
        assert!(!context.is_cancelled());
    }

    #[test]
    fn test_no_timeout() {
        let config = QueryTimeoutConfig {
            default_timeout: None,
            ..Default::default()
        };
        let context = QueryContext::new(1, config, None);

        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(context.check_timeout(), TimeoutStatus::Ok);
        assert!(context.remaining().is_none());
    }

    #[test]
    fn test_timeout_detection() {
        let config = QueryTimeoutConfig {
            default_timeout: Some(Duration::from_millis(50)),
            enable_soft_timeouts: false,
            ..Default::default()
        };
        let context = QueryContext::new(1, config, None);

        // Should be ok initially
        assert_eq!(context.check_timeout(), TimeoutStatus::Ok);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(60));

        // Should detect timeout
        assert_eq!(context.check_timeout(), TimeoutStatus::Exceeded);
        assert!(context.is_timeout_exceeded());
    }

    #[test]
    fn test_soft_timeout() {
        let config = QueryTimeoutConfig {
            default_timeout: Some(Duration::from_millis(100)),
            enable_soft_timeouts: true,
            soft_timeout_threshold: 0.5, // 50ms
            ..Default::default()
        };
        let context = QueryContext::new(1, config, None);

        // Should be ok initially
        assert_eq!(context.check_timeout(), TimeoutStatus::Ok);

        // Wait for soft timeout (> 50ms but < 100ms)
        std::thread::sleep(Duration::from_millis(60));

        // Should detect soft timeout
        assert_eq!(context.check_timeout(), TimeoutStatus::Warning);
        assert!(!context.is_timeout_exceeded());

        // Wait for hard timeout
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(context.check_timeout(), TimeoutStatus::Exceeded);
    }

    #[test]
    fn test_custom_timeout() {
        let config = QueryTimeoutConfig {
            default_timeout: Some(Duration::from_secs(10)),
            ..Default::default()
        };
        let custom_timeout = Some(Duration::from_millis(50));
        let context = QueryContext::new(1, config, custom_timeout);

        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(context.check_timeout(), TimeoutStatus::Exceeded);
    }

    #[test]
    fn test_max_timeout_enforcement() {
        let config = QueryTimeoutConfig {
            max_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let custom_timeout = Some(Duration::from_secs(10)); // Try to set 10 seconds
        let context = QueryContext::new(1, config, custom_timeout);

        std::thread::sleep(Duration::from_millis(110));
        // Should timeout at max_timeout (100ms), not custom (10s)
        assert_eq!(context.check_timeout(), TimeoutStatus::Exceeded);
    }

    #[test]
    fn test_cancellation() {
        let config = QueryTimeoutConfig::default();
        let context = QueryContext::new(1, config, None);

        assert!(!context.is_cancelled());
        context.cancel();
        assert!(context.is_cancelled());
        assert_eq!(context.check_timeout(), TimeoutStatus::Cancelled);
    }

    #[test]
    fn test_cancellation_handle() {
        let config = QueryTimeoutConfig::default();
        let context = QueryContext::new(1, config, None);
        let handle = context.cancellation_handle();

        assert_eq!(handle.query_id(), 1);
        assert!(!handle.is_cancelled());

        handle.cancel();
        assert!(handle.is_cancelled());
        assert!(context.is_cancelled());
    }

    #[test]
    fn test_elapsed_and_remaining() {
        let config = QueryTimeoutConfig {
            default_timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        };
        let context = QueryContext::new(1, config, None);

        std::thread::sleep(Duration::from_millis(30));

        let elapsed = context.elapsed();
        assert!(elapsed >= Duration::from_millis(30));
        assert!(elapsed < Duration::from_millis(100));

        let remaining = context.remaining().unwrap();
        assert!(remaining > Duration::ZERO);
        assert!(remaining <= Duration::from_millis(70));
    }

    #[test]
    fn test_timeout_manager_creation() {
        let config = QueryTimeoutConfig::default();
        let manager = QueryTimeoutManager::new(config);

        assert_eq!(manager.active_query_count(), 0);
    }

    #[test]
    fn test_start_and_finish_query() {
        let manager = QueryTimeoutManager::new(QueryTimeoutConfig::default());

        let context = manager.start_query(None);
        let query_id = context.query_id();

        assert_eq!(manager.active_query_count(), 1);
        assert!(manager.active_query_ids().contains(&query_id));

        manager.finish_query(query_id);
        assert_eq!(manager.active_query_count(), 0);
    }

    #[test]
    fn test_cancel_query_by_id() {
        let manager = QueryTimeoutManager::new(QueryTimeoutConfig::default());

        let context = manager.start_query(None);
        let query_id = context.query_id();

        assert!(!context.is_cancelled());

        let cancelled = manager.cancel_query(query_id);
        assert!(cancelled);
        assert!(context.is_cancelled());

        // Try to cancel non-existent query
        let cancelled = manager.cancel_query(99999);
        assert!(!cancelled);
    }

    #[test]
    fn test_cancel_all_queries() {
        let manager = QueryTimeoutManager::new(QueryTimeoutConfig::default());

        let context1 = manager.start_query(None);
        let context2 = manager.start_query(None);
        let context3 = manager.start_query(None);

        assert!(!context1.is_cancelled());
        assert!(!context2.is_cancelled());
        assert!(!context3.is_cancelled());

        manager.cancel_all_queries();

        assert!(context1.is_cancelled());
        assert!(context2.is_cancelled());
        assert!(context3.is_cancelled());
    }

    #[test]
    fn test_get_query_context() {
        let manager = QueryTimeoutManager::new(QueryTimeoutConfig::default());

        let context = manager.start_query(None);
        let query_id = context.query_id();

        let retrieved = manager.get_query_context(query_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().query_id(), query_id);

        let not_found = manager.get_query_context(99999);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_metrics_tracking() {
        let config = QueryTimeoutConfig {
            default_timeout: Some(Duration::from_millis(50)),
            ..Default::default()
        };
        let manager = QueryTimeoutManager::new(config);

        // Start and finish a normal query
        let context1 = manager.start_query(None);
        let query_id1 = context1.query_id();
        std::thread::sleep(Duration::from_millis(10));
        manager.finish_query(query_id1);

        // Start and timeout a query
        let context2 = manager.start_query(None);
        let query_id2 = context2.query_id();
        std::thread::sleep(Duration::from_millis(60));
        context2.check_timeout(); // Trigger timeout detection
        manager.finish_query(query_id2);

        // Start and cancel a query
        let context3 = manager.start_query(None);
        let query_id3 = context3.query_id();
        context3.cancel();
        manager.finish_query(query_id3);

        let metrics = manager.get_metrics();
        assert_eq!(metrics.total_timeouts, 1);
        assert_eq!(metrics.total_cancellations, 1);
        assert!(metrics.avg_query_duration_secs > 0.0);
    }

    #[test]
    fn test_multiple_active_queries() {
        let manager = QueryTimeoutManager::new(QueryTimeoutConfig::default());

        let _context1 = manager.start_query(None);
        let _context2 = manager.start_query(None);
        let _context3 = manager.start_query(None);

        assert_eq!(manager.active_query_count(), 3);
        assert_eq!(manager.active_query_ids().len(), 3);
    }

    #[test]
    fn test_timeout_with_grace_period() {
        let config = QueryTimeoutConfig {
            default_timeout: Some(Duration::from_millis(50)),
            grace_period: Duration::from_millis(20),
            ..Default::default()
        };

        let context = QueryContext::new(1, config, None);

        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(context.check_timeout(), TimeoutStatus::Exceeded);

        // Grace period would be handled by the query executor
        // The context just reports the status
    }

    #[test]
    fn test_remaining_time_zero_after_timeout() {
        let config = QueryTimeoutConfig {
            default_timeout: Some(Duration::from_millis(50)),
            ..Default::default()
        };
        let context = QueryContext::new(1, config, None);

        std::thread::sleep(Duration::from_millis(60));

        let remaining = context.remaining().unwrap();
        assert_eq!(remaining, Duration::ZERO);
    }
}
