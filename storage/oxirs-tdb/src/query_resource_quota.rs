//! Query Resource Quota Management
//!
//! This module provides per-query resource limiting to prevent resource exhaustion:
//! - Memory usage limits per query
//! - Execution time limits (wall clock and CPU time)
//! - Result set size limits
//! - Intermediate result size limits
//! - Concurrent query limits

use crate::error::{Result, TdbError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Query resource quota configuration
#[derive(Debug, Clone)]
pub struct QueryResourceQuotaConfig {
    /// Maximum memory per query in bytes (0 = unlimited)
    pub max_memory_per_query: usize,
    /// Maximum execution time per query
    pub max_execution_time: Duration,
    /// Maximum number of results per query (0 = unlimited)
    pub max_results: usize,
    /// Maximum intermediate result size in bytes (0 = unlimited)
    pub max_intermediate_size: usize,
    /// Maximum concurrent queries (0 = unlimited)
    pub max_concurrent_queries: usize,
    /// Enable CPU time tracking (may have overhead)
    pub track_cpu_time: bool,
    /// Kill query on quota violation (vs just warning)
    pub enforce_hard_limits: bool,
}

impl Default for QueryResourceQuotaConfig {
    fn default() -> Self {
        Self {
            max_memory_per_query: 256 * 1024 * 1024, // 256 MB
            max_execution_time: Duration::from_secs(30),
            max_results: 1_000_000,                   // 1 million results
            max_intermediate_size: 512 * 1024 * 1024, // 512 MB
            max_concurrent_queries: 100,
            track_cpu_time: false,
            enforce_hard_limits: true,
        }
    }
}

/// Query resource tracker for individual query execution
pub struct QueryResourceTracker {
    /// Query ID
    query_id: u64,
    /// Start time
    start_time: Instant,
    /// Current memory usage estimate (bytes)
    memory_used: AtomicUsize,
    /// Number of results produced
    result_count: AtomicUsize,
    /// Intermediate result size (bytes)
    intermediate_size: AtomicUsize,
    /// Configuration
    config: Arc<QueryResourceQuotaConfig>,
    /// Whether query has been cancelled
    cancelled: AtomicBool,
    /// Quota violations
    violations: RwLock<Vec<QuotaViolation>>,
}

impl QueryResourceTracker {
    /// Create a new query resource tracker
    pub fn new(query_id: u64, config: Arc<QueryResourceQuotaConfig>) -> Self {
        Self {
            query_id,
            start_time: Instant::now(),
            memory_used: AtomicUsize::new(0),
            result_count: AtomicUsize::new(0),
            intermediate_size: AtomicUsize::new(0),
            config,
            cancelled: AtomicBool::new(false),
            violations: RwLock::new(Vec::new()),
        }
    }

    /// Check if execution time limit has been exceeded
    pub fn check_execution_time(&self) -> Result<()> {
        let elapsed = self.start_time.elapsed();
        if elapsed > self.config.max_execution_time {
            self.record_violation(QuotaViolation {
                query_id: self.query_id,
                violation_type: ViolationType::ExecutionTime,
                limit: self.config.max_execution_time.as_secs() as usize,
                actual: elapsed.as_secs() as usize,
                timestamp: Instant::now(),
            });

            if self.config.enforce_hard_limits {
                self.cancelled.store(true, Ordering::Release);
                return Err(TdbError::Other(format!(
                    "Query {} execution time limit exceeded: {:?} > {:?}",
                    self.query_id, elapsed, self.config.max_execution_time
                )));
            }
        }
        Ok(())
    }

    /// Check if memory limit has been exceeded
    pub fn check_memory(&self) -> Result<()> {
        if self.config.max_memory_per_query == 0 {
            return Ok(());
        }

        let used = self.memory_used.load(Ordering::Relaxed);
        if used > self.config.max_memory_per_query {
            self.record_violation(QuotaViolation {
                query_id: self.query_id,
                violation_type: ViolationType::Memory,
                limit: self.config.max_memory_per_query,
                actual: used,
                timestamp: Instant::now(),
            });

            if self.config.enforce_hard_limits {
                self.cancelled.store(true, Ordering::Release);
                return Err(TdbError::Other(format!(
                    "Query {} memory limit exceeded: {} > {}",
                    self.query_id, used, self.config.max_memory_per_query
                )));
            }
        }
        Ok(())
    }

    /// Check if result count limit has been exceeded
    pub fn check_result_count(&self) -> Result<()> {
        if self.config.max_results == 0 {
            return Ok(());
        }

        let count = self.result_count.load(Ordering::Relaxed);
        if count > self.config.max_results {
            self.record_violation(QuotaViolation {
                query_id: self.query_id,
                violation_type: ViolationType::ResultCount,
                limit: self.config.max_results,
                actual: count,
                timestamp: Instant::now(),
            });

            if self.config.enforce_hard_limits {
                self.cancelled.store(true, Ordering::Release);
                return Err(TdbError::Other(format!(
                    "Query {} result count limit exceeded: {} > {}",
                    self.query_id, count, self.config.max_results
                )));
            }
        }
        Ok(())
    }

    /// Check if intermediate result size limit has been exceeded
    pub fn check_intermediate_size(&self) -> Result<()> {
        if self.config.max_intermediate_size == 0 {
            return Ok(());
        }

        let size = self.intermediate_size.load(Ordering::Relaxed);
        if size > self.config.max_intermediate_size {
            self.record_violation(QuotaViolation {
                query_id: self.query_id,
                violation_type: ViolationType::IntermediateSize,
                limit: self.config.max_intermediate_size,
                actual: size,
                timestamp: Instant::now(),
            });

            if self.config.enforce_hard_limits {
                self.cancelled.store(true, Ordering::Release);
                return Err(TdbError::Other(format!(
                    "Query {} intermediate result size limit exceeded: {} > {}",
                    self.query_id, size, self.config.max_intermediate_size
                )));
            }
        }
        Ok(())
    }

    /// Allocate memory for query operation
    pub fn allocate_memory(&self, bytes: usize) -> Result<()> {
        self.memory_used.fetch_add(bytes, Ordering::Relaxed);
        self.check_memory()
    }

    /// Free memory from query operation
    pub fn free_memory(&self, bytes: usize) {
        self.memory_used.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Record a result produced
    pub fn record_result(&self) -> Result<()> {
        self.result_count.fetch_add(1, Ordering::Relaxed);
        self.check_result_count()
    }

    /// Record intermediate result allocation
    pub fn allocate_intermediate(&self, bytes: usize) -> Result<()> {
        self.intermediate_size.fetch_add(bytes, Ordering::Relaxed);
        self.check_intermediate_size()
    }

    /// Free intermediate result storage
    pub fn free_intermediate(&self, bytes: usize) {
        self.intermediate_size.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Check if query has been cancelled due to quota violations
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Get current resource usage
    pub fn usage(&self) -> QueryResourceUsage {
        QueryResourceUsage {
            query_id: self.query_id,
            execution_time: self.start_time.elapsed(),
            memory_used: self.memory_used.load(Ordering::Relaxed),
            result_count: self.result_count.load(Ordering::Relaxed),
            intermediate_size: self.intermediate_size.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Acquire),
        }
    }

    /// Get all quota violations
    pub fn violations(&self) -> Vec<QuotaViolation> {
        self.violations.read().clone()
    }

    /// Record a quota violation
    fn record_violation(&self, violation: QuotaViolation) {
        self.violations.write().push(violation);
    }
}

/// Query resource quota manager (global)
pub struct QueryResourceQuotaManager {
    /// Configuration
    config: Arc<QueryResourceQuotaConfig>,
    /// Active query trackers
    active_queries: RwLock<HashMap<u64, Arc<QueryResourceTracker>>>,
    /// Next query ID
    next_query_id: AtomicU64,
    /// Total queries started
    total_queries: AtomicU64,
    /// Total queries cancelled
    cancelled_queries: AtomicU64,
    /// Total quota violations
    total_violations: AtomicU64,
}

impl QueryResourceQuotaManager {
    /// Create a new query resource quota manager
    pub fn new(config: QueryResourceQuotaConfig) -> Self {
        Self {
            config: Arc::new(config),
            active_queries: RwLock::new(HashMap::new()),
            next_query_id: AtomicU64::new(1),
            total_queries: AtomicU64::new(0),
            cancelled_queries: AtomicU64::new(0),
            total_violations: AtomicU64::new(0),
        }
    }

    /// Start tracking a new query
    pub fn start_query(&self) -> Result<Arc<QueryResourceTracker>> {
        // Check concurrent query limit
        let active_count = self.active_queries.read().len();
        if self.config.max_concurrent_queries > 0
            && active_count >= self.config.max_concurrent_queries
        {
            return Err(TdbError::Other(format!(
                "Concurrent query limit exceeded: {} >= {}",
                active_count, self.config.max_concurrent_queries
            )));
        }

        let query_id = self.next_query_id.fetch_add(1, Ordering::Relaxed);
        let tracker = Arc::new(QueryResourceTracker::new(
            query_id,
            Arc::clone(&self.config),
        ));

        self.active_queries
            .write()
            .insert(query_id, Arc::clone(&tracker));
        self.total_queries.fetch_add(1, Ordering::Relaxed);

        Ok(tracker)
    }

    /// Finish tracking a query
    pub fn finish_query(&self, query_id: u64) {
        if let Some(tracker) = self.active_queries.write().remove(&query_id) {
            if tracker.is_cancelled() {
                self.cancelled_queries.fetch_add(1, Ordering::Relaxed);
            }

            let violation_count = tracker.violations().len();
            if violation_count > 0 {
                self.total_violations
                    .fetch_add(violation_count as u64, Ordering::Relaxed);
            }
        }
    }

    /// Get active query count
    pub fn active_query_count(&self) -> usize {
        self.active_queries.read().len()
    }

    /// Get all active query usages
    pub fn active_query_usages(&self) -> Vec<QueryResourceUsage> {
        self.active_queries
            .read()
            .values()
            .map(|tracker| tracker.usage())
            .collect()
    }

    /// Get manager statistics
    pub fn stats(&self) -> QueryQuotaStats {
        QueryQuotaStats {
            total_queries: self.total_queries.load(Ordering::Relaxed),
            active_queries: self.active_queries.read().len() as u64,
            cancelled_queries: self.cancelled_queries.load(Ordering::Relaxed),
            total_violations: self.total_violations.load(Ordering::Relaxed),
            config: (*self.config).clone(),
        }
    }

    /// Cancel a specific query
    pub fn cancel_query(&self, query_id: u64) -> Result<()> {
        if let Some(tracker) = self.active_queries.read().get(&query_id) {
            tracker.cancelled.store(true, Ordering::Release);
            Ok(())
        } else {
            Err(TdbError::Other(format!("Query {} not found", query_id)))
        }
    }

    /// Cancel all queries
    pub fn cancel_all_queries(&self) {
        for tracker in self.active_queries.read().values() {
            tracker.cancelled.store(true, Ordering::Release);
        }
    }
}

/// Current resource usage for a query
#[derive(Debug, Clone)]
pub struct QueryResourceUsage {
    /// Query ID
    pub query_id: u64,
    /// Execution time so far
    pub execution_time: Duration,
    /// Memory used (bytes)
    pub memory_used: usize,
    /// Number of results produced
    pub result_count: usize,
    /// Intermediate result size (bytes)
    pub intermediate_size: usize,
    /// Whether query was cancelled
    pub cancelled: bool,
}

/// Quota violation record
#[derive(Debug, Clone)]
pub struct QuotaViolation {
    /// Query ID that violated quota
    pub query_id: u64,
    /// Type of violation
    pub violation_type: ViolationType,
    /// Configured limit
    pub limit: usize,
    /// Actual value that exceeded limit
    pub actual: usize,
    /// When violation occurred
    pub timestamp: Instant,
}

/// Type of quota violation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    /// Execution time exceeded
    ExecutionTime,
    /// Memory usage exceeded
    Memory,
    /// Result count exceeded
    ResultCount,
    /// Intermediate result size exceeded
    IntermediateSize,
}

/// Query quota manager statistics
#[derive(Debug, Clone)]
pub struct QueryQuotaStats {
    /// Total queries started
    pub total_queries: u64,
    /// Currently active queries
    pub active_queries: u64,
    /// Queries cancelled due to quota violations
    pub cancelled_queries: u64,
    /// Total quota violations across all queries
    pub total_violations: u64,
    /// Current configuration
    pub config: QueryResourceQuotaConfig,
}

impl QueryQuotaStats {
    /// Calculate cancellation rate
    pub fn cancellation_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.cancelled_queries as f64 / self.total_queries as f64
        }
    }

    /// Calculate average violations per query
    pub fn avg_violations_per_query(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.total_violations as f64 / self.total_queries as f64
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_query_resource_tracker_creation() {
        let config = Arc::new(QueryResourceQuotaConfig::default());
        let tracker = QueryResourceTracker::new(1, config);

        assert_eq!(tracker.query_id, 1);
        assert!(!tracker.is_cancelled());

        let usage = tracker.usage();
        assert_eq!(usage.memory_used, 0);
        assert_eq!(usage.result_count, 0);
    }

    #[test]
    fn test_memory_allocation_and_checking() {
        let mut config = QueryResourceQuotaConfig::default();
        config.max_memory_per_query = 1024;
        config.enforce_hard_limits = true;

        let tracker = QueryResourceTracker::new(1, Arc::new(config));

        // Allocate within limit
        assert!(tracker.allocate_memory(512).is_ok());
        assert_eq!(tracker.memory_used.load(Ordering::Relaxed), 512);

        // Allocate within limit again
        assert!(tracker.allocate_memory(256).is_ok());
        assert_eq!(tracker.memory_used.load(Ordering::Relaxed), 768);

        // Exceed limit
        let result = tracker.allocate_memory(512);
        assert!(result.is_err());
        assert!(tracker.is_cancelled());

        // Verify violation recorded
        let violations = tracker.violations();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_type, ViolationType::Memory);
    }

    #[test]
    fn test_result_count_limit() {
        let mut config = QueryResourceQuotaConfig::default();
        config.max_results = 5;
        config.enforce_hard_limits = true;

        let tracker = QueryResourceTracker::new(1, Arc::new(config));

        // Record results within limit
        for _ in 0..5 {
            assert!(tracker.record_result().is_ok());
        }

        // Exceed limit
        let result = tracker.record_result();
        assert!(result.is_err());
        assert!(tracker.is_cancelled());

        let violations = tracker.violations();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_type, ViolationType::ResultCount);
    }

    #[test]
    fn test_execution_time_limit() {
        let mut config = QueryResourceQuotaConfig::default();
        config.max_execution_time = Duration::from_millis(100);
        config.enforce_hard_limits = true;

        let tracker = QueryResourceTracker::new(1, Arc::new(config));

        // Immediate check should pass
        assert!(tracker.check_execution_time().is_ok());

        // Wait for limit to be exceeded
        std::thread::sleep(Duration::from_millis(150));

        let result = tracker.check_execution_time();
        assert!(result.is_err());
        assert!(tracker.is_cancelled());

        let violations = tracker.violations();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_type, ViolationType::ExecutionTime);
    }

    #[test]
    fn test_intermediate_size_limit() {
        let mut config = QueryResourceQuotaConfig::default();
        config.max_intermediate_size = 2048;
        config.enforce_hard_limits = true;

        let tracker = QueryResourceTracker::new(1, Arc::new(config));

        // Allocate within limit
        assert!(tracker.allocate_intermediate(1024).is_ok());
        assert_eq!(tracker.intermediate_size.load(Ordering::Relaxed), 1024);

        // Exceed limit
        let result = tracker.allocate_intermediate(2048);
        assert!(result.is_err());
        assert!(tracker.is_cancelled());

        let violations = tracker.violations();
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].violation_type,
            ViolationType::IntermediateSize
        );
    }

    #[test]
    fn test_soft_limits() {
        let mut config = QueryResourceQuotaConfig::default();
        config.max_memory_per_query = 1024;
        config.enforce_hard_limits = false; // Soft limits

        let tracker = QueryResourceTracker::new(1, Arc::new(config));

        // Exceed limit but don't cancel (soft limit)
        assert!(tracker.allocate_memory(2048).is_ok());
        assert!(!tracker.is_cancelled());

        // Violation still recorded
        let violations = tracker.violations();
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_query_quota_manager_creation() {
        let config = QueryResourceQuotaConfig::default();
        let manager = QueryResourceQuotaManager::new(config);

        assert_eq!(manager.active_query_count(), 0);

        let stats = manager.stats();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.active_queries, 0);
    }

    #[test]
    fn test_manager_start_finish_query() {
        let config = QueryResourceQuotaConfig::default();
        let manager = QueryResourceQuotaManager::new(config);

        // Start a query
        let tracker = manager.start_query().unwrap();
        assert_eq!(manager.active_query_count(), 1);
        assert_eq!(tracker.query_id, 1);

        // Finish the query
        manager.finish_query(tracker.query_id);
        assert_eq!(manager.active_query_count(), 0);

        let stats = manager.stats();
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.active_queries, 0);
    }

    #[test]
    fn test_concurrent_query_limit() {
        let mut config = QueryResourceQuotaConfig::default();
        config.max_concurrent_queries = 2;

        let manager = QueryResourceQuotaManager::new(config);

        // Start two queries (at limit)
        let _tracker1 = manager.start_query().unwrap();
        let _tracker2 = manager.start_query().unwrap();

        // Third query should fail
        let result = manager.start_query();
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_cancel_query() {
        let config = QueryResourceQuotaConfig::default();
        let manager = QueryResourceQuotaManager::new(config);

        let tracker = manager.start_query().unwrap();
        assert!(!tracker.is_cancelled());

        manager.cancel_query(tracker.query_id).unwrap();
        assert!(tracker.is_cancelled());
    }

    #[test]
    fn test_manager_cancel_all_queries() {
        let config = QueryResourceQuotaConfig::default();
        let manager = QueryResourceQuotaManager::new(config);

        let tracker1 = manager.start_query().unwrap();
        let tracker2 = manager.start_query().unwrap();

        manager.cancel_all_queries();

        assert!(tracker1.is_cancelled());
        assert!(tracker2.is_cancelled());
    }

    #[test]
    fn test_query_resource_usage() {
        let config = QueryResourceQuotaConfig::default();
        let tracker = QueryResourceTracker::new(42, Arc::new(config));

        tracker.allocate_memory(1024).unwrap();
        tracker.record_result().unwrap();
        tracker.allocate_intermediate(512).unwrap();

        let usage = tracker.usage();
        assert_eq!(usage.query_id, 42);
        assert_eq!(usage.memory_used, 1024);
        assert_eq!(usage.result_count, 1);
        assert_eq!(usage.intermediate_size, 512);
        assert!(!usage.cancelled);
    }

    #[test]
    fn test_quota_stats_calculations() {
        let mut config = QueryResourceQuotaConfig::default();
        config.max_concurrent_queries = 10;

        let manager = QueryResourceQuotaManager::new(config);

        // Simulate some queries
        for _ in 0..5 {
            let tracker = manager.start_query().unwrap();
            manager.finish_query(tracker.query_id);
        }

        // Simulate a cancelled query
        let tracker = manager.start_query().unwrap();
        manager.cancel_query(tracker.query_id).unwrap();
        manager.finish_query(tracker.query_id);

        let stats = manager.stats();
        assert_eq!(stats.total_queries, 6);
        assert_eq!(stats.cancelled_queries, 1);
        assert!(stats.cancellation_rate() > 0.0);
    }

    #[test]
    fn test_unlimited_quotas() {
        let mut config = QueryResourceQuotaConfig::default();
        config.max_memory_per_query = 0; // Unlimited
        config.max_results = 0; // Unlimited
        config.max_intermediate_size = 0; // Unlimited

        let tracker = QueryResourceTracker::new(1, Arc::new(config));

        // Should never fail with unlimited quotas
        assert!(tracker.allocate_memory(usize::MAX / 2).is_ok());
        for _ in 0..1000 {
            assert!(tracker.record_result().is_ok());
        }
        assert!(tracker.allocate_intermediate(usize::MAX / 2).is_ok());
    }

    #[test]
    fn test_memory_free() {
        let config = QueryResourceQuotaConfig::default();
        let tracker = QueryResourceTracker::new(1, Arc::new(config));

        tracker.allocate_memory(1024).unwrap();
        assert_eq!(tracker.memory_used.load(Ordering::Relaxed), 1024);

        tracker.free_memory(512);
        assert_eq!(tracker.memory_used.load(Ordering::Relaxed), 512);
    }

    #[test]
    fn test_intermediate_free() {
        let config = QueryResourceQuotaConfig::default();
        let tracker = QueryResourceTracker::new(1, Arc::new(config));

        tracker.allocate_intermediate(2048).unwrap();
        assert_eq!(tracker.intermediate_size.load(Ordering::Relaxed), 2048);

        tracker.free_intermediate(1024);
        assert_eq!(tracker.intermediate_size.load(Ordering::Relaxed), 1024);
    }

    #[test]
    fn test_active_query_usages() {
        let config = QueryResourceQuotaConfig::default();
        let manager = QueryResourceQuotaManager::new(config);

        let _tracker1 = manager.start_query().unwrap();
        let _tracker2 = manager.start_query().unwrap();

        let usages = manager.active_query_usages();
        assert_eq!(usages.len(), 2);
    }
}
