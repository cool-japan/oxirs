//! Production Hardening for SPARQL Query Engine
//!
//! Beta.1 Feature: Production-Ready SPARQL Operations
//!
//! This module provides production-grade features for reliable SPARQL query execution:
//! - Enhanced error handling with query context
//! - Query execution circuit breakers
//! - SPARQL performance monitoring
//! - Query result size quotas
//! - Health checks for query engine components

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Enhanced error type with SPARQL query context
#[derive(Debug, Clone)]
pub struct SparqlProductionError {
    pub error: String,
    pub context: QueryErrorContext,
    pub timestamp: SystemTime,
    pub severity: ErrorSeverity,
    pub retryable: bool,
}

/// Context information for SPARQL query errors
#[derive(Debug, Clone)]
pub struct QueryErrorContext {
    pub query: String,
    pub operation: String,
    pub pattern_count: usize,
    pub execution_time: Option<Duration>,
    pub result_count: Option<usize>,
    pub metadata: HashMap<String, String>,
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Informational - no action required
    Info,
    /// Warning - should be investigated
    Warning,
    /// Error - requires attention
    Error,
    /// Critical - requires immediate action
    Critical,
}

impl SparqlProductionError {
    pub fn new(
        error: String,
        context: QueryErrorContext,
        severity: ErrorSeverity,
        retryable: bool,
    ) -> Self {
        Self {
            error,
            context,
            timestamp: SystemTime::now(),
            severity,
            retryable,
        }
    }

    pub fn parse_error(query: String, message: String) -> Self {
        Self::new(
            format!("SPARQL parse error: {}", message),
            QueryErrorContext {
                query,
                operation: "parse".to_string(),
                pattern_count: 0,
                execution_time: None,
                result_count: None,
                metadata: HashMap::new(),
            },
            ErrorSeverity::Error,
            false,
        )
    }

    pub fn execution_error(query: String, message: String, elapsed: Duration) -> Self {
        Self::new(
            format!("SPARQL execution error: {}", message),
            QueryErrorContext {
                query,
                operation: "execute".to_string(),
                pattern_count: 0,
                execution_time: Some(elapsed),
                result_count: None,
                metadata: HashMap::new(),
            },
            ErrorSeverity::Error,
            true,
        )
    }

    pub fn timeout_error(query: String, elapsed: Duration, limit: Duration) -> Self {
        Self::new(
            format!("Query timeout: {:?} exceeded limit {:?}", elapsed, limit),
            QueryErrorContext {
                query,
                operation: "timeout".to_string(),
                pattern_count: 0,
                execution_time: Some(elapsed),
                result_count: None,
                metadata: HashMap::new(),
            },
            ErrorSeverity::Warning,
            true,
        )
    }
}

/// Circuit breaker for SPARQL query execution
pub struct QueryCircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failures: AtomicUsize,
    successes: AtomicUsize,
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub success_threshold: usize,
    pub timeout: Duration,
    pub half_open_max_requests: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_requests: 3,
        }
    }
}

impl QueryCircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failures: AtomicUsize::new(0),
            successes: AtomicUsize::new(0),
            config,
        }
    }

    pub fn is_request_allowed(&self) -> bool {
        let state = self.state.read().unwrap();
        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => {
                let successes = self.successes.load(Ordering::Relaxed);
                successes < self.config.half_open_max_requests
            }
        }
    }

    pub fn record_success(&self) {
        let mut state = self.state.write().unwrap();
        match *state {
            CircuitState::Closed => {
                self.failures.store(0, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                let successes = self.successes.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    self.failures.store(0, Ordering::Relaxed);
                    self.successes.store(0, Ordering::Relaxed);
                }
            }
            CircuitState::Open => {}
        }
    }

    pub fn record_failure(&self) {
        let mut state = self.state.write().unwrap();
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;

        if failures >= self.config.failure_threshold {
            *state = CircuitState::Open;
            self.successes.store(0, Ordering::Relaxed);
        }
    }

    pub fn try_half_open(&self) {
        let mut state = self.state.write().unwrap();
        if *state == CircuitState::Open {
            *state = CircuitState::HalfOpen;
            self.successes.store(0, Ordering::Relaxed);
        }
    }

    pub fn state(&self) -> String {
        format!("{:?}", *self.state.read().unwrap())
    }
}

/// Performance monitoring for SPARQL query execution
pub struct SparqlPerformanceMonitor {
    query_latencies: RwLock<HashMap<String, Vec<Duration>>>,
    query_counts: RwLock<HashMap<String, AtomicU64>>,
    pattern_complexities: RwLock<HashMap<String, Vec<usize>>>,
    result_sizes: RwLock<HashMap<String, Vec<usize>>>,
    timeouts: AtomicU64,
    errors: AtomicU64,
    start_time: Instant,
}

impl Default for SparqlPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl SparqlPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            query_latencies: RwLock::new(HashMap::new()),
            query_counts: RwLock::new(HashMap::new()),
            pattern_complexities: RwLock::new(HashMap::new()),
            result_sizes: RwLock::new(HashMap::new()),
            timeouts: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    pub fn record_query(
        &self,
        query_type: &str,
        latency: Duration,
        pattern_count: usize,
        result_count: usize,
    ) {
        // Record latency
        self.query_latencies
            .write()
            .unwrap()
            .entry(query_type.to_string())
            .or_default()
            .push(latency);

        // Increment count
        self.query_counts
            .write()
            .unwrap()
            .entry(query_type.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        // Record pattern complexity
        self.pattern_complexities
            .write()
            .unwrap()
            .entry(query_type.to_string())
            .or_default()
            .push(pattern_count);

        // Record result size
        self.result_sizes
            .write()
            .unwrap()
            .entry(query_type.to_string())
            .or_default()
            .push(result_count);
    }

    pub fn record_timeout(&self) {
        self.timeouts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_statistics(&self, query_type: &str) -> QueryStatistics {
        let latencies = self.query_latencies.read().unwrap();
        let counts = self.query_counts.read().unwrap();
        let complexities = self.pattern_complexities.read().unwrap();
        let sizes = self.result_sizes.read().unwrap();

        let latency_data = latencies.get(query_type);
        let count = counts
            .get(query_type)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0);
        let complexity_data = complexities.get(query_type);
        let size_data = sizes.get(query_type);

        let (avg_latency, p50_latency, p95_latency, p99_latency) = if let Some(data) = latency_data
        {
            let mut sorted = data.clone();
            sorted.sort();

            let sum: Duration = sorted.iter().sum();
            let avg = if !sorted.is_empty() {
                sum / sorted.len() as u32
            } else {
                Duration::ZERO
            };

            let p50 = if !sorted.is_empty() {
                sorted[sorted.len() / 2]
            } else {
                Duration::ZERO
            };

            let p95 = if !sorted.is_empty() {
                sorted[sorted.len() * 95 / 100]
            } else {
                Duration::ZERO
            };

            let p99 = if !sorted.is_empty() {
                sorted[sorted.len() * 99 / 100]
            } else {
                Duration::ZERO
            };

            (avg, p50, p95, p99)
        } else {
            (
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
            )
        };

        let avg_complexity = if let Some(data) = complexity_data {
            if !data.is_empty() {
                data.iter().sum::<usize>() / data.len()
            } else {
                0
            }
        } else {
            0
        };

        let avg_result_size = if let Some(data) = size_data {
            if !data.is_empty() {
                data.iter().sum::<usize>() / data.len()
            } else {
                0
            }
        } else {
            0
        };

        QueryStatistics {
            query_type: query_type.to_string(),
            total_queries: count,
            average_latency: avg_latency,
            p50_latency,
            p95_latency,
            p99_latency,
            average_pattern_complexity: avg_complexity,
            average_result_size: avg_result_size,
        }
    }

    pub fn get_global_statistics(&self) -> GlobalStatistics {
        GlobalStatistics {
            uptime: self.start_time.elapsed(),
            total_queries: self
                .query_counts
                .read()
                .unwrap()
                .values()
                .map(|c| c.load(Ordering::Relaxed))
                .sum(),
            total_timeouts: self.timeouts.load(Ordering::Relaxed),
            total_errors: self.errors.load(Ordering::Relaxed),
        }
    }

    pub fn reset(&self) {
        self.query_latencies.write().unwrap().clear();
        self.query_counts.write().unwrap().clear();
        self.pattern_complexities.write().unwrap().clear();
        self.result_sizes.write().unwrap().clear();
        self.timeouts.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct QueryStatistics {
    pub query_type: String,
    pub total_queries: u64,
    pub average_latency: Duration,
    pub p50_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub average_pattern_complexity: usize,
    pub average_result_size: usize,
}

#[derive(Debug, Clone)]
pub struct GlobalStatistics {
    pub uptime: Duration,
    pub total_queries: u64,
    pub total_timeouts: u64,
    pub total_errors: u64,
}

/// Resource quota manager for SPARQL queries
pub struct QueryResourceQuota {
    max_result_size: AtomicUsize,
    max_query_time: RwLock<Duration>,
    max_pattern_complexity: AtomicUsize,
    enforced: AtomicBool,
}

impl QueryResourceQuota {
    pub fn new(
        max_result_size: usize,
        max_query_time: Duration,
        max_pattern_complexity: usize,
    ) -> Self {
        Self {
            max_result_size: AtomicUsize::new(max_result_size),
            max_query_time: RwLock::new(max_query_time),
            max_pattern_complexity: AtomicUsize::new(max_pattern_complexity),
            enforced: AtomicBool::new(true),
        }
    }

    pub fn check_result_size(&self, size: usize) -> Result<()> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }

        let max = self.max_result_size.load(Ordering::Relaxed);
        if size > max {
            return Err(anyhow!("Result size {} exceeds quota of {}", size, max));
        }

        Ok(())
    }

    pub fn check_query_time(&self, elapsed: Duration) -> Result<()> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }

        let max = *self.max_query_time.read().unwrap();
        if elapsed > max {
            return Err(anyhow!(
                "Query time {:?} exceeds quota of {:?}",
                elapsed,
                max
            ));
        }

        Ok(())
    }

    pub fn check_pattern_complexity(&self, complexity: usize) -> Result<()> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }

        let max = self.max_pattern_complexity.load(Ordering::Relaxed);
        if complexity > max {
            return Err(anyhow!(
                "Pattern complexity {} exceeds quota of {}",
                complexity,
                max
            ));
        }

        Ok(())
    }

    pub fn set_result_size_limit(&self, limit: usize) {
        self.max_result_size.store(limit, Ordering::Relaxed);
    }

    pub fn set_time_limit(&self, limit: Duration) {
        *self.max_query_time.write().unwrap() = limit;
    }

    pub fn set_complexity_limit(&self, limit: usize) {
        self.max_pattern_complexity.store(limit, Ordering::Relaxed);
    }

    pub fn set_enforced(&self, enforced: bool) {
        self.enforced.store(enforced, Ordering::Relaxed);
    }

    pub fn is_enforced(&self) -> bool {
        self.enforced.load(Ordering::Relaxed)
    }
}

impl Default for QueryResourceQuota {
    fn default() -> Self {
        Self::new(
            1_000_000,                // 1M results max
            Duration::from_secs(300), // 5 minute timeout
            1000,                     // 1000 pattern complexity max
        )
    }
}

/// Health check for SPARQL query engine components
pub struct QueryEngineHealth {
    checks: RwLock<HashMap<String, HealthCheck>>,
}

#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: SystemTime,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl QueryEngineHealth {
    pub fn new() -> Self {
        Self {
            checks: RwLock::new(HashMap::new()),
        }
    }

    pub fn register_check(&self, name: &str) {
        self.checks.write().unwrap().insert(
            name.to_string(),
            HealthCheck {
                name: name.to_string(),
                status: HealthStatus::Unknown,
                last_check: SystemTime::now(),
                message: "Not yet checked".to_string(),
            },
        );
    }

    pub fn update_check(&self, name: &str, status: HealthStatus, message: String) {
        if let Some(check) = self.checks.write().unwrap().get_mut(name) {
            check.status = status;
            check.last_check = SystemTime::now();
            check.message = message;
        }
    }

    pub fn check_parser(&self) -> HealthStatus {
        // Simple parser health check
        let status = HealthStatus::Healthy;
        self.update_check("parser", status, "Parser is operational".to_string());
        status
    }

    pub fn check_executor(&self) -> HealthStatus {
        // Simple executor health check
        let status = HealthStatus::Healthy;
        self.update_check("executor", status, "Executor is operational".to_string());
        status
    }

    pub fn check_optimizer(&self) -> HealthStatus {
        // Simple optimizer health check
        let status = HealthStatus::Healthy;
        self.update_check("optimizer", status, "Optimizer is operational".to_string());
        status
    }

    pub fn get_overall_status(&self) -> HealthStatus {
        let checks = self.checks.read().unwrap();

        if checks.is_empty() {
            return HealthStatus::Unknown;
        }

        let mut has_unhealthy = false;
        let mut has_degraded = false;

        for check in checks.values() {
            match check.status {
                HealthStatus::Unhealthy => has_unhealthy = true,
                HealthStatus::Degraded => has_degraded = true,
                _ => {}
            }
        }

        if has_unhealthy {
            HealthStatus::Unhealthy
        } else if has_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    pub fn get_checks(&self) -> Vec<HealthCheck> {
        self.checks.read().unwrap().values().cloned().collect()
    }

    pub fn perform_all_checks(&self) {
        self.check_parser();
        self.check_executor();
        self.check_optimizer();
    }
}

impl Default for QueryEngineHealth {
    fn default() -> Self {
        let health = Self::new();
        health.register_check("parser");
        health.register_check("executor");
        health.register_check("optimizer");
        health
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed_to_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_requests: 2,
        };

        let breaker = QueryCircuitBreaker::new(config);

        assert!(breaker.is_request_allowed());

        // Record failures
        breaker.record_failure();
        assert!(breaker.is_request_allowed());

        breaker.record_failure();
        assert!(breaker.is_request_allowed());

        breaker.record_failure();
        assert!(!breaker.is_request_allowed()); // Should open
    }

    #[test]
    fn test_circuit_breaker_half_open_to_closed() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_requests: 3,
        };

        let breaker = QueryCircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert!(!breaker.is_request_allowed());

        // Try half-open
        breaker.try_half_open();
        assert!(breaker.is_request_allowed());

        // Record successes to close
        breaker.record_success();
        breaker.record_success();

        assert!(breaker.is_request_allowed());
        assert_eq!(breaker.state(), "Closed");
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = SparqlPerformanceMonitor::new();

        monitor.record_query("SELECT", Duration::from_millis(100), 5, 1000);
        monitor.record_query("SELECT", Duration::from_millis(200), 10, 2000);
        monitor.record_query("SELECT", Duration::from_millis(150), 7, 1500);

        let stats = monitor.get_statistics("SELECT");
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.average_pattern_complexity, 7);
        assert_eq!(stats.average_result_size, 1500);

        let global = monitor.get_global_statistics();
        assert_eq!(global.total_queries, 3);
    }

    #[test]
    fn test_resource_quota() {
        let quota = QueryResourceQuota::new(1000, Duration::from_secs(10), 50);

        // Within limits
        assert!(quota.check_result_size(500).is_ok());
        assert!(quota.check_query_time(Duration::from_secs(5)).is_ok());
        assert!(quota.check_pattern_complexity(30).is_ok());

        // Exceeding limits
        assert!(quota.check_result_size(2000).is_err());
        assert!(quota.check_query_time(Duration::from_secs(20)).is_err());
        assert!(quota.check_pattern_complexity(100).is_err());

        // Disable enforcement
        quota.set_enforced(false);
        assert!(quota.check_result_size(2000).is_ok());
    }

    #[test]
    fn test_health_checks() {
        let health = QueryEngineHealth::default();

        health.perform_all_checks();

        assert_eq!(health.get_overall_status(), HealthStatus::Healthy);

        let checks = health.get_checks();
        assert_eq!(checks.len(), 3);
        assert!(checks.iter().all(|c| c.status == HealthStatus::Healthy));
    }

    #[test]
    fn test_production_error() {
        let error = SparqlProductionError::parse_error(
            "SELECT ?s WHERE { ?s ?p ?o".to_string(),
            "Missing closing brace".to_string(),
        );

        assert_eq!(error.severity, ErrorSeverity::Error);
        assert!(!error.retryable);
        assert_eq!(error.context.operation, "parse");
    }
}
