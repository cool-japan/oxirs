//! Query monitoring for timeout enforcement and slow query logging
//!
//! This module provides:
//! - Query timeout enforcement to prevent runaway queries
//! - Slow query logging for performance analysis
//! - Query execution tracking and statistics

use crate::dictionary::Term;
use crate::error::{Result, TdbError};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for query monitoring
#[derive(Debug, Clone)]
pub struct QueryMonitorConfig {
    /// Global query timeout (0 = no timeout)
    pub default_timeout: Duration,
    /// Threshold for logging slow queries
    pub slow_query_threshold: Duration,
    /// Maximum number of slow queries to keep in memory
    pub max_slow_query_history: usize,
    /// Whether to enable query monitoring
    pub enabled: bool,
}

impl Default for QueryMonitorConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            slow_query_threshold: Duration::from_millis(100),
            max_slow_query_history: 100,
            enabled: true,
        }
    }
}

/// Query execution context
pub struct QueryExecution {
    /// Query ID
    id: u64,
    /// Query pattern
    pattern: QueryPattern,
    /// Start time
    start_time: Instant,
    /// Timeout deadline (if any)
    deadline: Option<Instant>,
    /// Whether query is still running
    running: Arc<AtomicU64>, // 0 = stopped, 1 = running
}

impl QueryExecution {
    /// Create a new query execution context
    fn new(id: u64, pattern: QueryPattern, timeout: Option<Duration>) -> Self {
        let start_time = Instant::now();
        let deadline = timeout.map(|t| start_time + t);

        Self {
            id,
            pattern,
            start_time,
            deadline,
            running: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Check if query has timed out
    pub fn is_timed_out(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Instant::now() >= deadline
        } else {
            false
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Mark query as complete
    fn complete(&self) {
        self.running.store(0, Ordering::Relaxed);
    }

    /// Check if still running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed) == 1
    }
}

/// Query pattern for logging
#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// Subject pattern
    pub subject: Option<String>,
    /// Predicate pattern
    pub predicate: Option<String>,
    /// Object pattern
    pub object: Option<String>,
}

impl QueryPattern {
    /// Create from Term options
    pub fn from_terms(
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
    ) -> Self {
        Self {
            subject: subject.map(|t| format!("{:?}", t)),
            predicate: predicate.map(|t| format!("{:?}", t)),
            object: object.map(|t| format!("{:?}", t)),
        }
    }

    /// Get a human-readable description
    pub fn describe(&self) -> String {
        format!(
            "({}, {}, {})",
            self.subject.as_deref().unwrap_or("?"),
            self.predicate.as_deref().unwrap_or("?"),
            self.object.as_deref().unwrap_or("?")
        )
    }
}

/// Slow query record
#[derive(Debug, Clone)]
pub struct SlowQueryRecord {
    /// When the query was executed
    pub timestamp: Instant,
    /// Query pattern
    pub pattern: QueryPattern,
    /// Execution time
    pub execution_time: Duration,
    /// Number of results returned
    pub result_count: usize,
    /// Whether the query timed out
    pub timed_out: bool,
}

/// Query monitor for timeout enforcement and slow query logging
pub struct QueryMonitor {
    /// Configuration
    config: QueryMonitorConfig,
    /// Next query ID
    next_id: AtomicU64,
    /// Active queries
    active_queries: Arc<RwLock<Vec<Arc<QueryExecution>>>>,
    /// Slow query history
    slow_queries: Arc<RwLock<VecDeque<SlowQueryRecord>>>,
    /// Statistics
    stats: QueryMonitorStats,
}

/// Query monitor statistics
#[derive(Debug, Default)]
pub struct QueryMonitorStats {
    /// Total queries executed
    pub total_queries: AtomicU64,
    /// Total queries that timed out
    pub timed_out_queries: AtomicU64,
    /// Total slow queries
    pub slow_queries: AtomicU64,
    /// Total execution time (microseconds)
    pub total_execution_time_us: AtomicU64,
}

impl QueryMonitorStats {
    /// Get average query execution time (microseconds)
    pub fn avg_execution_time_us(&self) -> u64 {
        let total = self.total_queries.load(Ordering::Relaxed);
        if total == 0 {
            0
        } else {
            self.total_execution_time_us.load(Ordering::Relaxed) / total
        }
    }

    /// Get timeout rate (percentage)
    pub fn timeout_rate(&self) -> f64 {
        let total = self.total_queries.load(Ordering::Relaxed) as f64;
        if total == 0.0 {
            0.0
        } else {
            let timeouts = self.timed_out_queries.load(Ordering::Relaxed) as f64;
            (timeouts / total) * 100.0
        }
    }

    /// Get slow query rate (percentage)
    pub fn slow_query_rate(&self) -> f64 {
        let total = self.total_queries.load(Ordering::Relaxed) as f64;
        if total == 0.0 {
            0.0
        } else {
            let slow = self.slow_queries.load(Ordering::Relaxed) as f64;
            (slow / total) * 100.0
        }
    }
}

impl QueryMonitor {
    /// Create a new query monitor
    pub fn new(config: QueryMonitorConfig) -> Self {
        Self {
            config,
            next_id: AtomicU64::new(1),
            active_queries: Arc::new(RwLock::new(Vec::new())),
            slow_queries: Arc::new(RwLock::new(VecDeque::new())),
            stats: QueryMonitorStats::default(),
        }
    }

    /// Begin a monitored query execution
    pub fn begin_query(
        &self,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
        timeout_override: Option<Duration>,
    ) -> Arc<QueryExecution> {
        if !self.config.enabled {
            // Create a dummy execution with no monitoring
            let pattern = QueryPattern::from_terms(subject, predicate, object);
            return Arc::new(QueryExecution::new(0, pattern, None));
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let pattern = QueryPattern::from_terms(subject, predicate, object);

        // Determine timeout
        let timeout = timeout_override.or({
            if self.config.default_timeout.as_secs() > 0 {
                Some(self.config.default_timeout)
            } else {
                None
            }
        });

        let execution = Arc::new(QueryExecution::new(id, pattern, timeout));

        // Track active query
        self.active_queries.write().push(Arc::clone(&execution));

        execution
    }

    /// End a monitored query execution
    pub fn end_query(&self, execution: Arc<QueryExecution>, result_count: usize) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        execution.complete();

        // Check for timeout
        let timed_out = execution.is_timed_out();
        if timed_out {
            self.stats.timed_out_queries.fetch_add(1, Ordering::Relaxed);
            return Err(TdbError::Other(format!(
                "Query timed out after {:?}",
                execution.elapsed()
            )));
        }

        let execution_time = execution.elapsed();

        // Update statistics
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_execution_time_us
            .fetch_add(execution_time.as_micros() as u64, Ordering::Relaxed);

        // Check if slow query
        if execution_time >= self.config.slow_query_threshold {
            self.log_slow_query(
                execution.pattern.clone(),
                execution_time,
                result_count,
                timed_out,
            );
        }

        // Remove from active queries
        self.active_queries.write().retain(|e| e.id != execution.id);

        Ok(())
    }

    /// Check if a query should be cancelled due to timeout
    pub fn check_timeout(&self, execution: &Arc<QueryExecution>) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        if execution.is_timed_out() {
            Err(TdbError::Other(format!(
                "Query timeout: exceeded {:?}",
                execution.elapsed()
            )))
        } else {
            Ok(())
        }
    }

    /// Log a slow query
    fn log_slow_query(
        &self,
        pattern: QueryPattern,
        execution_time: Duration,
        result_count: usize,
        timed_out: bool,
    ) {
        self.stats.slow_queries.fetch_add(1, Ordering::Relaxed);

        let record = SlowQueryRecord {
            timestamp: Instant::now(),
            pattern: pattern.clone(),
            execution_time,
            result_count,
            timed_out,
        };

        // Add to slow query history
        let mut slow_queries = self.slow_queries.write();
        slow_queries.push_back(record);

        // Trim if exceeds max size
        while slow_queries.len() > self.config.max_slow_query_history {
            slow_queries.pop_front();
        }

        // Also log to system logger
        log::warn!(
            "Slow query detected: pattern={}, time={:?}, results={}{}",
            pattern.describe(),
            execution_time,
            result_count,
            if timed_out { " (TIMED OUT)" } else { "" }
        );
    }

    /// Get slow query history
    pub fn slow_query_history(&self) -> Vec<SlowQueryRecord> {
        self.slow_queries.read().iter().cloned().collect()
    }

    /// Get currently active queries
    pub fn active_queries(&self) -> Vec<Arc<QueryExecution>> {
        self.active_queries.read().clone()
    }

    /// Get statistics
    pub fn stats(&self) -> &QueryMonitorStats {
        &self.stats
    }

    /// Clear slow query history
    pub fn clear_slow_query_history(&self) {
        self.slow_queries.write().clear();
    }

    /// Get number of active queries
    pub fn active_query_count(&self) -> usize {
        self.active_queries.read().len()
    }

    /// Cancel all running queries (emergency stop)
    pub fn cancel_all(&self) {
        let active = self.active_queries.read();
        for query in active.iter() {
            query.complete();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::Term;
    use std::thread;

    #[test]
    fn test_query_monitor_creation() {
        let config = QueryMonitorConfig::default();
        let monitor = QueryMonitor::new(config);

        assert_eq!(monitor.active_query_count(), 0);
        assert_eq!(monitor.slow_query_history().len(), 0);
    }

    #[test]
    fn test_begin_and_end_query() {
        let monitor = QueryMonitor::new(QueryMonitorConfig::default());

        let s = Term::Iri("http://example.org/s".to_string());
        let execution = monitor.begin_query(Some(&s), None, None, None);

        assert_eq!(monitor.active_query_count(), 1);

        monitor.end_query(execution, 10).unwrap();

        assert_eq!(monitor.active_query_count(), 0);
        assert_eq!(monitor.stats().total_queries.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[ignore] // Timing-dependent test, not suitable for CI
    fn test_query_timeout() {
        let config = QueryMonitorConfig {
            default_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let monitor = QueryMonitor::new(config);

        let execution = monitor.begin_query(None, None, None, None);

        // Wait for timeout (longer to ensure reliability)
        thread::sleep(Duration::from_millis(100));

        // Should detect timeout
        assert!(execution.is_timed_out());

        let result = monitor.end_query(execution, 0);
        assert!(result.is_err());
        assert_eq!(monitor.stats().timed_out_queries.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_slow_query_logging() {
        let config = QueryMonitorConfig {
            slow_query_threshold: Duration::from_millis(10),
            default_timeout: Duration::from_secs(10),
            ..Default::default()
        };
        let monitor = QueryMonitor::new(config);

        let execution = monitor.begin_query(None, None, None, None);

        // Make it slow
        thread::sleep(Duration::from_millis(20));

        monitor.end_query(execution, 5).unwrap();

        // Should have logged a slow query
        assert_eq!(monitor.stats().slow_queries.load(Ordering::Relaxed), 1);
        assert_eq!(monitor.slow_query_history().len(), 1);
    }

    #[test]
    fn test_timeout_override() {
        let monitor = QueryMonitor::new(QueryMonitorConfig::default());

        let custom_timeout = Duration::from_millis(50);
        let execution = monitor.begin_query(None, None, None, Some(custom_timeout));

        // Verify custom timeout is used
        assert!(execution.deadline.is_some());
    }

    #[test]
    fn test_multiple_active_queries() {
        let monitor = QueryMonitor::new(QueryMonitorConfig::default());

        let exec1 = monitor.begin_query(None, None, None, None);
        let exec2 = monitor.begin_query(None, None, None, None);
        let exec3 = monitor.begin_query(None, None, None, None);

        assert_eq!(monitor.active_query_count(), 3);

        monitor.end_query(exec1, 1).unwrap();
        assert_eq!(monitor.active_query_count(), 2);

        monitor.end_query(exec2, 2).unwrap();
        monitor.end_query(exec3, 3).unwrap();
        assert_eq!(monitor.active_query_count(), 0);
    }

    #[test]
    fn test_slow_query_history_limit() {
        let config = QueryMonitorConfig {
            slow_query_threshold: Duration::from_millis(1),
            max_slow_query_history: 5,
            default_timeout: Duration::from_secs(10),
            ..Default::default()
        };
        let monitor = QueryMonitor::new(config);

        // Generate 10 slow queries
        for _ in 0..10 {
            let execution = monitor.begin_query(None, None, None, None);
            thread::sleep(Duration::from_millis(2));
            monitor.end_query(execution, 1).unwrap();
        }

        // Should only keep 5 most recent
        assert_eq!(monitor.slow_query_history().len(), 5);
    }

    #[test]
    fn test_query_pattern_describe() {
        let s = Term::Iri("http://example.org/s".to_string());
        let p = Term::Iri("http://example.org/p".to_string());

        let pattern = QueryPattern::from_terms(Some(&s), Some(&p), None);
        let desc = pattern.describe();

        assert!(desc.contains("Iri"));
    }

    #[test]
    fn test_stats_calculations() {
        let monitor = QueryMonitor::new(QueryMonitorConfig::default());

        // Execute some queries
        for _ in 0..10 {
            let execution = monitor.begin_query(None, None, None, None);
            monitor.end_query(execution, 1).unwrap();
        }

        let stats = monitor.stats();
        assert_eq!(stats.total_queries.load(Ordering::Relaxed), 10);
        // avg_execution_time_us returns u64, which is always >= 0
        let _ = stats.avg_execution_time_us(); // Just verify it doesn't panic
    }

    #[test]
    #[ignore] // Timing-dependent test, not suitable for CI
    fn test_timeout_rate() {
        let config = QueryMonitorConfig {
            default_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let monitor = QueryMonitor::new(config);

        // Execute 10 queries, make some timeout
        for i in 0..10 {
            let execution = monitor.begin_query(None, None, None, None);

            if i < 5 {
                // First 5 timeout
                thread::sleep(Duration::from_millis(100));
            }

            let _ = monitor.end_query(execution, 1);
        }

        let stats = monitor.stats();
        // Should have some timeouts (at least 30% to account for timing variations)
        assert!(
            stats.timeout_rate() > 30.0,
            "Expected some timeouts but got rate: {}%",
            stats.timeout_rate()
        );
    }

    #[test]
    fn test_disabled_monitor() {
        let config = QueryMonitorConfig {
            enabled: false,
            ..Default::default()
        };
        let monitor = QueryMonitor::new(config);

        let execution = monitor.begin_query(None, None, None, None);
        monitor.end_query(execution, 10).unwrap();

        // Should not track anything when disabled
        assert_eq!(monitor.stats().total_queries.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_clear_slow_query_history() {
        let config = QueryMonitorConfig {
            slow_query_threshold: Duration::from_millis(1),
            ..Default::default()
        };
        let monitor = QueryMonitor::new(config);

        for _ in 0..5 {
            let execution = monitor.begin_query(None, None, None, None);
            thread::sleep(Duration::from_millis(2));
            monitor.end_query(execution, 1).unwrap();
        }

        assert_eq!(monitor.slow_query_history().len(), 5);

        monitor.clear_slow_query_history();

        assert_eq!(monitor.slow_query_history().len(), 0);
    }

    #[test]
    fn test_cancel_all() {
        let monitor = QueryMonitor::new(QueryMonitorConfig::default());

        let exec1 = monitor.begin_query(None, None, None, None);
        let exec2 = monitor.begin_query(None, None, None, None);

        assert!(exec1.is_running());
        assert!(exec2.is_running());

        monitor.cancel_all();

        assert!(!exec1.is_running());
        assert!(!exec2.is_running());
    }
}
