//! Production Hardening Features for OxiRS TDB
//!
//! Beta.1 Feature: Production-Ready Storage Hardening
//!
//! This module provides production-grade features for TDB storage reliability:
//! - Enhanced error handling with storage context
//! - Storage health checking and diagnostics
//! - Performance monitoring for storage operations
//! - Circuit breakers for storage operations
//! - Resource limits and quotas for storage
//! - Corruption detection and repair

use crate::error::{Result, TdbError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Enhanced error type with storage context for production debugging
#[derive(Debug, Clone)]
pub struct StorageError {
    /// The underlying TDB error message
    pub error_message: String,
    /// Storage operation context
    pub context: StorageContext,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Severity level
    pub severity: ErrorSeverity,
    /// Whether the operation can be retried
    pub retryable: bool,
}

/// Storage operation context
#[derive(Debug, Clone)]
pub struct StorageContext {
    /// Operation that failed (e.g., "insert_triple", "query", "transaction_commit")
    pub operation: String,
    /// Storage-specific context fields
    pub fields: HashMap<String, String>,
    /// Affected resources (page IDs, node IDs, etc.)
    pub resources: Vec<String>,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Debug information
    Debug,
    /// Informational
    Info,
    /// Warning - degraded but functional
    Warning,
    /// Error - operation failed
    Error,
    /// Critical - data integrity at risk
    Critical,
}

impl StorageError {
    /// Create a new storage error
    pub fn new(error: TdbError, operation: impl Into<String>) -> Self {
        Self {
            error_message: format!("{:?}", error),
            context: StorageContext {
                operation: operation.into(),
                fields: HashMap::new(),
                resources: Vec::new(),
            },
            timestamp: std::time::SystemTime::now(),
            severity: ErrorSeverity::Error,
            retryable: false,
        }
    }

    /// Add context field
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.fields.insert(key.into(), value.into());
        self
    }

    /// Add affected resource
    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.context.resources.push(resource.into());
        self
    }

    /// Set severity level
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Mark as retryable
    pub fn retryable(mut self) -> Self {
        self.retryable = true;
        self
    }

    /// Get formatted error message with all context
    pub fn detailed_message(&self) -> String {
        let mut msg = format!(
            "[{:?}] Storage operation '{}' failed: {}",
            self.severity, self.context.operation, self.error_message
        );

        if !self.context.fields.is_empty() {
            msg.push_str("\nContext:");
            for (key, value) in &self.context.fields {
                msg.push_str(&format!("\n  {}: {}", key, value));
            }
        }

        if !self.context.resources.is_empty() {
            msg.push_str(&format!(
                "\nAffected resources: {}",
                self.context.resources.join(", ")
            ));
        }

        if self.retryable {
            msg.push_str("\n(Operation is retryable)");
        }

        msg
    }
}

/// Health status of storage components
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Component is healthy
    Healthy,
    /// Component is degraded but functional
    Degraded,
    /// Component is unhealthy
    Unhealthy,
    /// Component status unknown
    Unknown,
}

/// Storage health check result
#[derive(Debug, Clone)]
pub struct StorageHealthCheck {
    /// Component name (e.g., "buffer_pool", "dictionary", "indexes", "wal")
    pub component: String,
    /// Health status
    pub status: HealthStatus,
    /// Status message
    pub message: String,
    /// Check timestamp
    pub timestamp: Instant,
    /// Response time
    pub response_time: Duration,
    /// Storage-specific metrics
    pub metrics: HashMap<String, f64>,
}

impl StorageHealthCheck {
    /// Create a healthy check
    pub fn healthy(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            status: HealthStatus::Healthy,
            message: message.into(),
            timestamp: Instant::now(),
            response_time: Duration::from_micros(0),
            metrics: HashMap::new(),
        }
    }

    /// Create a degraded check
    pub fn degraded(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            status: HealthStatus::Degraded,
            message: message.into(),
            timestamp: Instant::now(),
            response_time: Duration::from_micros(0),
            metrics: HashMap::new(),
        }
    }

    /// Create an unhealthy check
    pub fn unhealthy(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            status: HealthStatus::Unhealthy,
            message: message.into(),
            timestamp: Instant::now(),
            response_time: Duration::from_micros(0),
            metrics: HashMap::new(),
        }
    }

    /// Add a metric
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Set response time
    pub fn with_response_time(mut self, duration: Duration) -> Self {
        self.response_time = duration;
        self
    }
}

/// Circuit breaker for storage operations
pub struct StorageCircuitBreaker {
    /// Circuit state
    state: Arc<RwLock<CircuitState>>,
    /// Failure count
    failures: AtomicUsize,
    /// Success count
    successes: AtomicUsize,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Last state change time
    last_state_change: RwLock<Instant>,
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (failing fast)
    Open,
    /// Circuit is half-open (testing recovery)
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: usize,
    /// Success threshold to close circuit from half-open
    pub success_threshold: usize,
    /// Timeout before trying half-open (seconds)
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        }
    }
}

impl StorageCircuitBreaker {
    /// Create a new circuit breaker for storage operations
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failures: AtomicUsize::new(0),
            successes: AtomicUsize::new(0),
            config,
            last_state_change: RwLock::new(Instant::now()),
        }
    }

    /// Check if operation should be allowed
    pub fn allow_request(&self) -> bool {
        let mut state = self.state.write();

        // Check if we should transition from Open to HalfOpen
        if *state == CircuitState::Open {
            let elapsed = self.last_state_change.read().elapsed();
            if elapsed >= self.config.timeout {
                *state = CircuitState::HalfOpen;
                *self.last_state_change.write() = Instant::now();
                self.successes.store(0, Ordering::Relaxed);
                return true;
            }
            return false;
        }

        matches!(*state, CircuitState::Closed | CircuitState::HalfOpen)
    }

    /// Record a successful storage operation
    pub fn record_success(&self) {
        let successes = self.successes.fetch_add(1, Ordering::Relaxed) + 1;
        self.failures.store(0, Ordering::Relaxed);

        let mut state = self.state.write();
        if *state == CircuitState::HalfOpen && successes >= self.config.success_threshold {
            *state = CircuitState::Closed;
            *self.last_state_change.write() = Instant::now();
            self.successes.store(0, Ordering::Relaxed);
        }
    }

    /// Record a failed storage operation
    pub fn record_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        self.successes.store(0, Ordering::Relaxed);

        let mut state = self.state.write();
        if failures >= self.config.failure_threshold && *state != CircuitState::Open {
            *state = CircuitState::Open;
            *self.last_state_change.write() = Instant::now();
        }
    }

    /// Get current state as string
    pub fn state(&self) -> String {
        format!("{:?}", *self.state.read())
    }

    /// Get circuit breaker statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: format!("{:?}", *self.state.read()),
            failures: self.failures.load(Ordering::Relaxed),
            successes: self.successes.load(Ordering::Relaxed),
            last_state_change: *self.last_state_change.read(),
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Current circuit state (Closed, Open, HalfOpen)
    pub state: String,
    /// Number of consecutive failures
    pub failures: usize,
    /// Number of consecutive successes
    pub successes: usize,
    /// Time of last state transition
    pub last_state_change: Instant,
}

/// Performance monitoring for storage operations
pub struct StoragePerformanceMonitor {
    /// Operation latencies (operation name -> list of durations)
    latencies: RwLock<HashMap<String, Vec<Duration>>>,
    /// Operation counts
    counts: RwLock<HashMap<String, AtomicU64>>,
    /// Error counts
    errors: RwLock<HashMap<String, AtomicU64>>,
    /// Bytes processed (for I/O operations)
    bytes_processed: RwLock<HashMap<String, AtomicU64>>,
    /// Start time
    start_time: Instant,
}

impl StoragePerformanceMonitor {
    /// Create a new storage performance monitor
    pub fn new() -> Self {
        Self {
            latencies: RwLock::new(HashMap::new()),
            counts: RwLock::new(HashMap::new()),
            errors: RwLock::new(HashMap::new()),
            bytes_processed: RwLock::new(HashMap::new()),
            start_time: Instant::now(),
        }
    }

    /// Record a storage operation execution
    pub fn record_operation(&self, operation: &str, duration: Duration, bytes: u64, success: bool) {
        // Record latency
        {
            let mut latencies = self.latencies.write();
            latencies
                .entry(operation.to_string())
                .or_default()
                .push(duration);
        }

        // Increment count
        {
            let mut counts = self.counts.write();
            counts
                .entry(operation.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }

        // Record bytes processed
        if bytes > 0 {
            let mut bytes_map = self.bytes_processed.write();
            bytes_map
                .entry(operation.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(bytes, Ordering::Relaxed);
        }

        // Record error if failed
        if !success {
            let mut errors = self.errors.write();
            errors
                .entry(operation.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get statistics for a storage operation
    pub fn stats(&self, operation: &str) -> Option<StorageOperationStats> {
        let latencies = self.latencies.read();
        let counts = self.counts.read();
        let errors = self.errors.read();
        let bytes_map = self.bytes_processed.read();

        let latency_vec = latencies.get(operation)?;
        let count = counts.get(operation)?.load(Ordering::Relaxed);
        let error_count = errors
            .get(operation)
            .map_or(0, |e| e.load(Ordering::Relaxed));
        let total_bytes = bytes_map
            .get(operation)
            .map_or(0, |b| b.load(Ordering::Relaxed));

        if latency_vec.is_empty() {
            return None;
        }

        // Calculate statistics
        let mut sorted_latencies = latency_vec.clone();
        sorted_latencies.sort();

        let total: Duration = sorted_latencies.iter().sum();
        let avg = total / sorted_latencies.len() as u32;

        let p50 = sorted_latencies[sorted_latencies.len() / 2];
        let p95 = sorted_latencies[sorted_latencies.len() * 95 / 100];
        let p99 = sorted_latencies[sorted_latencies.len() * 99 / 100];
        let min = *sorted_latencies
            .first()
            .expect("collection validated to be non-empty");
        let max = *sorted_latencies
            .last()
            .expect("collection validated to be non-empty");

        Some(StorageOperationStats {
            operation: operation.to_string(),
            count,
            error_count,
            total_bytes,
            avg_latency: avg,
            p50_latency: p50,
            p95_latency: p95,
            p99_latency: p99,
            min_latency: min,
            max_latency: max,
        })
    }

    /// Get all storage statistics
    pub fn all_stats(&self) -> Vec<StorageOperationStats> {
        let operations: Vec<String> = self.counts.read().keys().cloned().collect();
        operations.iter().filter_map(|op| self.stats(op)).collect()
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for StoragePerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage operation statistics
#[derive(Debug, Clone)]
pub struct StorageOperationStats {
    /// Name of the storage operation
    pub operation: String,
    /// Total number of operations performed
    pub count: u64,
    /// Number of operations that failed
    pub error_count: u64,
    /// Total bytes processed (for I/O operations)
    pub total_bytes: u64,
    /// Average operation latency
    pub avg_latency: Duration,
    /// Median (p50) operation latency
    pub p50_latency: Duration,
    /// 95th percentile operation latency
    pub p95_latency: Duration,
    /// 99th percentile operation latency
    pub p99_latency: Duration,
    /// Minimum operation latency
    pub min_latency: Duration,
    /// Maximum operation latency
    pub max_latency: Duration,
}

impl StorageOperationStats {
    /// Get error rate as percentage
    pub fn error_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.error_count as f64 / self.count as f64) * 100.0
        }
    }

    /// Get throughput (operations per second)
    pub fn throughput(&self, duration: Duration) -> f64 {
        if duration.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.count as f64 / duration.as_secs_f64()
        }
    }

    /// Get I/O throughput (bytes per second)
    pub fn io_throughput(&self, duration: Duration) -> f64 {
        if duration.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.total_bytes as f64 / duration.as_secs_f64()
        }
    }
}

/// Storage resource quota manager
pub struct StorageResourceQuota {
    /// Maximum storage size in bytes
    max_storage: AtomicU64,
    /// Current storage usage estimate
    current_storage: AtomicU64,
    /// Maximum transaction rate (txn/sec)
    max_txn_rate: AtomicU64,
    /// Current transaction count
    txn_count: AtomicU64,
    /// Rate limit window start
    window_start: RwLock<Instant>,
    /// Whether quota is enforced
    enforced: AtomicBool,
}

impl StorageResourceQuota {
    /// Create a new storage resource quota manager
    pub fn new(max_storage: u64, max_txn_rate: u64) -> Self {
        Self {
            max_storage: AtomicU64::new(max_storage),
            current_storage: AtomicU64::new(0),
            max_txn_rate: AtomicU64::new(max_txn_rate),
            txn_count: AtomicU64::new(0),
            window_start: RwLock::new(Instant::now()),
            enforced: AtomicBool::new(true),
        }
    }

    /// Check if storage quota allows allocation
    pub fn check_storage(&self, bytes: u64) -> bool {
        if !self.enforced.load(Ordering::Relaxed) {
            return true;
        }

        let current = self.current_storage.load(Ordering::Relaxed);
        let max = self.max_storage.load(Ordering::Relaxed);
        current + bytes <= max
    }

    /// Allocate storage (update quota)
    pub fn allocate_storage(&self, bytes: u64) -> Result<()> {
        if !self.check_storage(bytes) {
            return Err(TdbError::Other(format!(
                "Storage quota exceeded: requested {} bytes",
                bytes
            )));
        }

        self.current_storage.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    /// Free storage (update quota)
    pub fn free_storage(&self, bytes: u64) {
        self.current_storage.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Check if transaction rate limit allows operation
    pub fn check_txn_rate(&self) -> bool {
        if !self.enforced.load(Ordering::Relaxed) {
            return true;
        }

        let now = Instant::now();
        let window_start = *self.window_start.read();

        // Reset window if needed
        if now.duration_since(window_start) >= Duration::from_secs(1) {
            *self.window_start.write() = now;
            self.txn_count.store(0, Ordering::Relaxed);
            return true;
        }

        let count = self.txn_count.load(Ordering::Relaxed);
        let max = self.max_txn_rate.load(Ordering::Relaxed);
        count < max
    }

    /// Record a transaction (update rate limit)
    pub fn record_transaction(&self) -> Result<()> {
        if !self.check_txn_rate() {
            return Err(TdbError::Other(
                "Transaction rate limit exceeded".to_string(),
            ));
        }

        self.txn_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get current quota usage
    pub fn usage(&self) -> StorageQuotaUsage {
        StorageQuotaUsage {
            storage_used: self.current_storage.load(Ordering::Relaxed),
            storage_max: self.max_storage.load(Ordering::Relaxed),
            txn_count: self.txn_count.load(Ordering::Relaxed),
            txn_max: self.max_txn_rate.load(Ordering::Relaxed),
        }
    }

    /// Enable or disable enforcement
    pub fn set_enforced(&self, enforced: bool) {
        self.enforced.store(enforced, Ordering::Relaxed);
    }
}

/// Storage quota usage information
#[derive(Debug, Clone)]
pub struct StorageQuotaUsage {
    /// Current storage used in bytes
    pub storage_used: u64,
    /// Maximum storage allowed in bytes
    pub storage_max: u64,
    /// Current transaction count in current window
    pub txn_count: u64,
    /// Maximum transaction rate per second
    pub txn_max: u64,
}

impl StorageQuotaUsage {
    /// Get storage usage percentage
    pub fn storage_percent(&self) -> f64 {
        if self.storage_max == 0 {
            0.0
        } else {
            (self.storage_used as f64 / self.storage_max as f64) * 100.0
        }
    }

    /// Get transaction rate usage percentage
    pub fn txn_rate_percent(&self) -> f64 {
        if self.txn_max == 0 {
            0.0
        } else {
            (self.txn_count as f64 / self.txn_max as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_error() {
        let error = TdbError::Other("Test error".to_string());
        let storage_error = StorageError::new(error, "insert_triple")
            .with_context("triple_count", "1000")
            .with_resource("page_42")
            .with_severity(ErrorSeverity::Warning)
            .retryable();

        assert_eq!(storage_error.context.operation, "insert_triple");
        assert_eq!(
            storage_error.context.fields.get("triple_count"),
            Some(&"1000".to_string())
        );
        assert!(storage_error.retryable);
        assert_eq!(storage_error.severity, ErrorSeverity::Warning);

        let message = storage_error.detailed_message();
        assert!(message.contains("insert_triple"));
        assert!(message.contains("page_42"));
    }

    #[test]
    fn test_storage_health_check() {
        let health = StorageHealthCheck::healthy("buffer_pool", "All pages cached")
            .with_metric("cache_hit_rate", 95.5)
            .with_metric("evictions_per_sec", 10.0)
            .with_response_time(Duration::from_millis(2));

        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.component, "buffer_pool");
        assert_eq!(health.metrics.get("cache_hit_rate"), Some(&95.5));
    }

    #[test]
    fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            ..Default::default()
        };
        let breaker = StorageCircuitBreaker::new(config);

        // Initially closed
        assert!(breaker.allow_request());

        // Record failures
        breaker.record_failure();
        breaker.record_failure();
        assert!(breaker.allow_request()); // Still closed

        breaker.record_failure();
        assert!(!breaker.allow_request()); // Now open

        let stats = breaker.stats();
        assert_eq!(stats.failures, 3);
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = StoragePerformanceMonitor::new();

        // Record some storage operations
        monitor.record_operation("insert", Duration::from_millis(10), 1024, true);
        monitor.record_operation("insert", Duration::from_millis(15), 2048, true);
        monitor.record_operation("insert", Duration::from_millis(20), 512, true);
        monitor.record_operation("insert", Duration::from_millis(25), 4096, false);

        let stats = monitor.stats("insert").unwrap();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.error_rate(), 25.0);
        assert_eq!(stats.total_bytes, 1024 + 2048 + 512 + 4096);
    }

    #[test]
    fn test_resource_quota() {
        let quota = StorageResourceQuota::new(1024 * 1024, 100);

        // Test storage quota
        assert!(quota.check_storage(512 * 1024));
        assert!(quota.allocate_storage(512 * 1024).is_ok());
        assert!(quota.check_storage(512 * 1024));
        assert!(!quota.check_storage(513 * 1024));

        quota.free_storage(256 * 1024);
        assert!(quota.check_storage(768 * 1024));

        // Test transaction rate limit
        for _ in 0..100 {
            assert!(quota.record_transaction().is_ok());
        }
        // Should hit rate limit
        assert!(quota.record_transaction().is_err());

        let usage = quota.usage();
        assert_eq!(usage.storage_used, 256 * 1024);
        assert_eq!(usage.txn_count, 100);
    }
}
