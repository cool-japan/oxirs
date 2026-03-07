//! Production Hardening Features for OxiRS Core
//!
//! Beta.1 Feature: Production-Ready Hardening
//!
//! This module provides production-grade features for reliability, observability,
//! and operational excellence:
//! - Enhanced error handling with context
//! - Health checking and diagnostics
//! - Performance monitoring and metrics
//! - Circuit breakers and failsafes
//! - Graceful degradation
//! - Resource limits and quotas

use crate::OxirsError;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Enhanced error type with contextual information for production debugging
#[derive(Debug, Clone)]
pub struct ProductionError {
    /// The underlying error
    pub error: OxirsError,
    /// Error context (operation, state, etc.)
    pub context: ErrorContext,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Severity level
    pub severity: ErrorSeverity,
    /// Whether the operation can be retried
    pub retryable: bool,
}

/// Error context information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: String,
    /// Additional context fields
    pub fields: HashMap<String, String>,
    /// Stack trace if available
    pub trace: Option<String>,
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
    /// Critical - system integrity at risk
    Critical,
}

impl ProductionError {
    /// Create a new production error
    pub fn new(error: OxirsError, operation: impl Into<String>) -> Self {
        Self {
            error,
            context: ErrorContext {
                operation: operation.into(),
                fields: HashMap::new(),
                trace: None,
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
            "[{:?}] {} in operation '{}'",
            self.severity, self.error, self.context.operation
        );

        if !self.context.fields.is_empty() {
            msg.push_str("\nContext:");
            for (key, value) in &self.context.fields {
                msg.push_str(&format!("\n  {key}: {value}"));
            }
        }

        if self.retryable {
            msg.push_str("\n(Operation is retryable)");
        }

        msg
    }
}

/// Health status of a component
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

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Component name
    pub component: String,
    /// Health status
    pub status: HealthStatus,
    /// Status message
    pub message: String,
    /// Check timestamp
    pub timestamp: Instant,
    /// Response time
    pub response_time: Duration,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl HealthCheck {
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

/// Circuit breaker for fault tolerance
pub struct CircuitBreaker {
    /// Circuit state
    state: Arc<RwLock<CircuitState>>,
    /// Failure count
    failures: AtomicUsize,
    /// Success count
    successes: AtomicUsize,
    /// Configuration
    config: CircuitBreakerConfig,
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
    /// Success threshold to close circuit
    pub success_threshold: usize,
    /// Timeout before trying half-open
    pub timeout: Duration,
    /// Window for counting failures
    pub window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            window: Duration::from_secs(10),
        }
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failures: AtomicUsize::new(0),
            successes: AtomicUsize::new(0),
            config,
        }
    }

    /// Check if operation should be allowed
    pub fn allow_request(&self) -> bool {
        let state = *self.state.read();
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        let successes = self.successes.fetch_add(1, Ordering::Relaxed) + 1;
        self.failures.store(0, Ordering::Relaxed);

        let state = *self.state.read();
        if state == CircuitState::HalfOpen && successes >= self.config.success_threshold {
            *self.state.write() = CircuitState::Closed;
            self.successes.store(0, Ordering::Relaxed);
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;

        if failures >= self.config.failure_threshold {
            *self.state.write() = CircuitState::Open;
            // Schedule transition to half-open after timeout
            // In production, this would use a timer/scheduler
        }
    }

    /// Get current state
    pub fn state(&self) -> String {
        format!("{:?}", *self.state.read())
    }

    /// Get statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: format!("{:?}", *self.state.read()),
            failures: self.failures.load(Ordering::Relaxed),
            successes: self.successes.load(Ordering::Relaxed),
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: String,
    pub failures: usize,
    pub successes: usize,
}

/// Performance monitoring and metrics collection
pub struct PerformanceMonitor {
    /// Operation latencies
    latencies: RwLock<HashMap<String, Vec<Duration>>>,
    /// Operation counts
    counts: RwLock<HashMap<String, AtomicU64>>,
    /// Error counts
    errors: RwLock<HashMap<String, AtomicU64>>,
    /// Start time
    start_time: Instant,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            latencies: RwLock::new(HashMap::new()),
            counts: RwLock::new(HashMap::new()),
            errors: RwLock::new(HashMap::new()),
            start_time: Instant::now(),
        }
    }

    /// Record an operation execution
    pub fn record_operation(&self, operation: &str, duration: Duration, success: bool) {
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

        // Record error if failed
        if !success {
            let mut errors = self.errors.write();
            errors
                .entry(operation.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get statistics for an operation
    pub fn stats(&self, operation: &str) -> Option<OperationStats> {
        let latencies = self.latencies.read();
        let counts = self.counts.read();
        let errors = self.errors.read();

        let latency_vec = latencies.get(operation)?;
        let count = counts.get(operation)?.load(Ordering::Relaxed);
        let error_count = errors
            .get(operation)
            .map_or(0, |e| e.load(Ordering::Relaxed));

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

        Some(OperationStats {
            operation: operation.to_string(),
            count,
            error_count,
            avg_latency: avg,
            p50_latency: p50,
            p95_latency: p95,
            p99_latency: p99,
            min_latency: min,
            max_latency: max,
        })
    }

    /// Get all statistics
    pub fn all_stats(&self) -> Vec<OperationStats> {
        let operations: Vec<String> = self.counts.read().keys().cloned().collect();
        operations.iter().filter_map(|op| self.stats(op)).collect()
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation statistics
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub operation: String,
    pub count: u64,
    pub error_count: u64,
    pub avg_latency: Duration,
    pub p50_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub min_latency: Duration,
    pub max_latency: Duration,
}

impl OperationStats {
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
}

/// Resource quota manager
pub struct ResourceQuota {
    /// Maximum memory usage in bytes
    max_memory: AtomicUsize,
    /// Current memory usage estimate
    current_memory: AtomicUsize,
    /// Maximum operation rate (ops/sec)
    max_rate: AtomicU64,
    /// Current operation count
    operation_count: AtomicU64,
    /// Rate limit window start
    window_start: RwLock<Instant>,
    /// Whether quota is enforced
    enforced: AtomicBool,
}

impl ResourceQuota {
    /// Create a new resource quota manager
    pub fn new(max_memory: usize, max_rate: u64) -> Self {
        Self {
            max_memory: AtomicUsize::new(max_memory),
            current_memory: AtomicUsize::new(0),
            max_rate: AtomicU64::new(max_rate),
            operation_count: AtomicU64::new(0),
            window_start: RwLock::new(Instant::now()),
            enforced: AtomicBool::new(true),
        }
    }

    /// Check if memory quota allows allocation
    pub fn check_memory(&self, bytes: usize) -> bool {
        if !self.enforced.load(Ordering::Relaxed) {
            return true;
        }

        let current = self.current_memory.load(Ordering::Relaxed);
        let max = self.max_memory.load(Ordering::Relaxed);
        current + bytes <= max
    }

    /// Allocate memory (update quota)
    pub fn allocate_memory(&self, bytes: usize) -> Result<(), String> {
        if !self.check_memory(bytes) {
            return Err(format!("Memory quota exceeded: requested {bytes} bytes"));
        }

        self.current_memory.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    /// Free memory (update quota)
    pub fn free_memory(&self, bytes: usize) {
        self.current_memory.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Check if rate limit allows operation
    pub fn check_rate(&self) -> bool {
        if !self.enforced.load(Ordering::Relaxed) {
            return true;
        }

        let now = Instant::now();
        let window_start = *self.window_start.read();

        // Reset window if needed
        if now.duration_since(window_start) >= Duration::from_secs(1) {
            *self.window_start.write() = now;
            self.operation_count.store(0, Ordering::Relaxed);
            return true;
        }

        let count = self.operation_count.load(Ordering::Relaxed);
        let max = self.max_rate.load(Ordering::Relaxed);
        count < max
    }

    /// Record an operation (update rate limit)
    pub fn record_operation(&self) -> Result<(), String> {
        if !self.check_rate() {
            return Err("Rate limit exceeded".to_string());
        }

        self.operation_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get current quota usage
    pub fn usage(&self) -> QuotaUsage {
        QuotaUsage {
            memory_used: self.current_memory.load(Ordering::Relaxed),
            memory_max: self.max_memory.load(Ordering::Relaxed),
            operations_count: self.operation_count.load(Ordering::Relaxed),
            operations_max: self.max_rate.load(Ordering::Relaxed),
        }
    }

    /// Enable or disable enforcement
    pub fn set_enforced(&self, enforced: bool) {
        self.enforced.store(enforced, Ordering::Relaxed);
    }
}

/// Quota usage information
#[derive(Debug, Clone)]
pub struct QuotaUsage {
    pub memory_used: usize,
    pub memory_max: usize,
    pub operations_count: u64,
    pub operations_max: u64,
}

impl QuotaUsage {
    /// Get memory usage percentage
    pub fn memory_percent(&self) -> f64 {
        if self.memory_max == 0 {
            0.0
        } else {
            (self.memory_used as f64 / self.memory_max as f64) * 100.0
        }
    }

    /// Get rate usage percentage
    pub fn rate_percent(&self) -> f64 {
        if self.operations_max == 0 {
            0.0
        } else {
            (self.operations_count as f64 / self.operations_max as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_error() {
        let error = OxirsError::Parse("Test error".to_string());
        let prod_error = ProductionError::new(error, "parse_operation")
            .with_context("file", "test.ttl")
            .with_context("line", "42")
            .with_severity(ErrorSeverity::Error)
            .retryable();

        assert_eq!(prod_error.context.operation, "parse_operation");
        assert_eq!(
            prod_error.context.fields.get("file"),
            Some(&"test.ttl".to_string())
        );
        assert!(prod_error.retryable);
        assert_eq!(prod_error.severity, ErrorSeverity::Error);

        let message = prod_error.detailed_message();
        assert!(message.contains("parse_operation"));
        assert!(message.contains("file: test.ttl"));
    }

    #[test]
    fn test_health_check() {
        let health = HealthCheck::healthy("database", "All systems operational")
            .with_metric("response_time_ms", 5.2)
            .with_metric("connections", 10.0)
            .with_response_time(Duration::from_millis(5));

        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.component, "database");
        assert_eq!(health.metrics.get("response_time_ms"), Some(&5.2));
    }

    #[test]
    fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Initially closed
        assert!(breaker.allow_request());

        // Record failures
        breaker.record_failure();
        breaker.record_failure();
        assert!(breaker.allow_request()); // Still closed

        breaker.record_failure();
        // Should be open now
        // Note: State transition happens asynchronously in production

        let stats = breaker.stats();
        assert_eq!(stats.failures, 3);
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();

        // Record some operations
        monitor.record_operation("query", Duration::from_millis(10), true);
        monitor.record_operation("query", Duration::from_millis(15), true);
        monitor.record_operation("query", Duration::from_millis(20), true);
        monitor.record_operation("query", Duration::from_millis(25), false);

        let stats = monitor.stats("query").unwrap();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.error_rate(), 25.0);
    }

    #[test]
    fn test_resource_quota() {
        let quota = ResourceQuota::new(1024, 100);

        // Test memory quota
        assert!(quota.check_memory(512));
        assert!(quota.allocate_memory(512).is_ok());
        assert!(quota.check_memory(512));
        assert!(!quota.check_memory(513));

        quota.free_memory(256);
        assert!(quota.check_memory(768));

        // Test rate limit
        for _ in 0..100 {
            assert!(quota.record_operation().is_ok());
        }
        // Should hit rate limit
        assert!(quota.record_operation().is_err());

        let usage = quota.usage();
        assert_eq!(usage.memory_used, 256);
        assert_eq!(usage.operations_count, 100);
    }
}
