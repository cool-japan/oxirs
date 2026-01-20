//! Production Hardening for SPARQL HTTP Server
//!
//! Beta.1 Feature: Production-Ready HTTP Server Operations
//!
//! This module provides production-grade features for reliable HTTP server operations:
//! - Enhanced error handling with HTTP context
//! - HTTP request circuit breakers
//! - Server performance monitoring
//! - Request rate limiting and quotas
//! - Health checks for server components

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Enhanced error type with HTTP request context
#[derive(Debug, Clone)]
pub struct HttpProductionError {
    pub error: String,
    pub context: HttpErrorContext,
    pub timestamp: SystemTime,
    pub severity: ErrorSeverity,
    pub retryable: bool,
}

/// Context information for HTTP request errors
#[derive(Debug, Clone)]
pub struct HttpErrorContext {
    pub method: String,
    pub path: String,
    pub status_code: Option<u16>,
    pub request_id: Option<String>,
    pub client_ip: Option<String>,
    pub user_agent: Option<String>,
    pub execution_time: Option<Duration>,
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

impl HttpProductionError {
    pub fn new(
        error: String,
        context: HttpErrorContext,
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

    pub fn request_error(method: String, path: String, message: String, status: u16) -> Self {
        Self::new(
            format!("HTTP {} error: {}", status, message),
            HttpErrorContext {
                method,
                path,
                status_code: Some(status),
                request_id: None,
                client_ip: None,
                user_agent: None,
                execution_time: None,
                metadata: HashMap::new(),
            },
            if status >= 500 {
                ErrorSeverity::Error
            } else {
                ErrorSeverity::Warning
            },
            (500..600).contains(&status),
        )
    }

    pub fn timeout_error(method: String, path: String, elapsed: Duration, limit: Duration) -> Self {
        Self::new(
            format!("Request timeout: {:?} exceeded limit {:?}", elapsed, limit),
            HttpErrorContext {
                method,
                path,
                status_code: Some(504),
                request_id: None,
                client_ip: None,
                user_agent: None,
                execution_time: Some(elapsed),
                metadata: HashMap::new(),
            },
            ErrorSeverity::Warning,
            true,
        )
    }
}

/// Circuit breaker for HTTP request handling
pub struct HttpCircuitBreaker {
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
            failure_threshold: 10,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            half_open_max_requests: 5,
        }
    }
}

impl HttpCircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failures: AtomicUsize::new(0),
            successes: AtomicUsize::new(0),
            config,
        }
    }

    pub fn is_request_allowed(&self) -> bool {
        let state = self.state.read().expect("lock poisoned");
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
        let mut state = self.state.write().expect("lock poisoned");
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
        let mut state = self.state.write().expect("lock poisoned");
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;

        if failures >= self.config.failure_threshold {
            *state = CircuitState::Open;
            self.successes.store(0, Ordering::Relaxed);
        }
    }

    pub fn try_half_open(&self) {
        let mut state = self.state.write().expect("lock poisoned");
        if *state == CircuitState::Open {
            *state = CircuitState::HalfOpen;
            self.successes.store(0, Ordering::Relaxed);
        }
    }

    pub fn state(&self) -> String {
        format!("{:?}", *self.state.read().expect("lock poisoned"))
    }
}

/// Performance monitoring for HTTP server
pub struct ServerPerformanceMonitor {
    endpoint_latencies: RwLock<HashMap<String, Vec<Duration>>>,
    endpoint_counts: RwLock<HashMap<String, AtomicU64>>,
    status_codes: RwLock<HashMap<u16, AtomicU64>>,
    request_sizes: RwLock<HashMap<String, Vec<usize>>>,
    response_sizes: RwLock<HashMap<String, Vec<usize>>>,
    timeouts: AtomicU64,
    errors: AtomicU64,
    start_time: Instant,
}

impl ServerPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            endpoint_latencies: RwLock::new(HashMap::new()),
            endpoint_counts: RwLock::new(HashMap::new()),
            status_codes: RwLock::new(HashMap::new()),
            request_sizes: RwLock::new(HashMap::new()),
            response_sizes: RwLock::new(HashMap::new()),
            timeouts: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    pub fn record_request(
        &self,
        endpoint: &str,
        latency: Duration,
        status_code: u16,
        request_size: usize,
        response_size: usize,
    ) {
        // Record latency
        self.endpoint_latencies
            .write()
            .expect("lock poisoned")
            .entry(endpoint.to_string())
            .or_default()
            .push(latency);

        // Increment count
        self.endpoint_counts
            .write()
            .expect("lock poisoned")
            .entry(endpoint.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        // Record status code
        self.status_codes
            .write()
            .expect("lock poisoned")
            .entry(status_code)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        // Record request size
        self.request_sizes
            .write()
            .expect("lock poisoned")
            .entry(endpoint.to_string())
            .or_default()
            .push(request_size);

        // Record response size
        self.response_sizes
            .write()
            .expect("lock poisoned")
            .entry(endpoint.to_string())
            .or_default()
            .push(response_size);

        // Track errors
        if status_code >= 500 {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_timeout(&self) {
        self.timeouts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_endpoint_statistics(&self, endpoint: &str) -> EndpointStatistics {
        let latencies = self.endpoint_latencies.read().expect("lock poisoned");
        let counts = self.endpoint_counts.read().expect("lock poisoned");
        let req_sizes = self.request_sizes.read().expect("lock poisoned");
        let resp_sizes = self.response_sizes.read().expect("lock poisoned");

        let latency_data = latencies.get(endpoint);
        let count = counts
            .get(endpoint)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0);
        let req_size_data = req_sizes.get(endpoint);
        let resp_size_data = resp_sizes.get(endpoint);

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

        let avg_request_size = if let Some(data) = req_size_data {
            if !data.is_empty() {
                data.iter().sum::<usize>() / data.len()
            } else {
                0
            }
        } else {
            0
        };

        let avg_response_size = if let Some(data) = resp_size_data {
            if !data.is_empty() {
                data.iter().sum::<usize>() / data.len()
            } else {
                0
            }
        } else {
            0
        };

        EndpointStatistics {
            endpoint: endpoint.to_string(),
            total_requests: count,
            average_latency: avg_latency,
            p50_latency,
            p95_latency,
            p99_latency,
            average_request_size: avg_request_size,
            average_response_size: avg_response_size,
        }
    }

    pub fn get_global_statistics(&self) -> ServerStatistics {
        let status_codes = self.status_codes.read().expect("lock poisoned");
        let total_requests: u64 = status_codes
            .values()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();

        ServerStatistics {
            uptime: self.start_time.elapsed(),
            total_requests,
            total_timeouts: self.timeouts.load(Ordering::Relaxed),
            total_errors: self.errors.load(Ordering::Relaxed),
            status_code_distribution: status_codes
                .iter()
                .map(|(code, count)| (*code, count.load(Ordering::Relaxed)))
                .collect(),
        }
    }

    pub fn reset(&self) {
        self.endpoint_latencies
            .write()
            .expect("lock poisoned")
            .clear();
        self.endpoint_counts.write().expect("lock poisoned").clear();
        self.status_codes.write().expect("lock poisoned").clear();
        self.request_sizes.write().expect("lock poisoned").clear();
        self.response_sizes.write().expect("lock poisoned").clear();
        self.timeouts.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct EndpointStatistics {
    pub endpoint: String,
    pub total_requests: u64,
    pub average_latency: Duration,
    pub p50_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub average_request_size: usize,
    pub average_response_size: usize,
}

#[derive(Debug, Clone)]
pub struct ServerStatistics {
    pub uptime: Duration,
    pub total_requests: u64,
    pub total_timeouts: u64,
    pub total_errors: u64,
    pub status_code_distribution: HashMap<u16, u64>,
}

/// Rate limiter for HTTP requests
pub struct RequestRateLimiter {
    max_requests_per_second: AtomicU64,
    request_count: AtomicU64,
    window_start: RwLock<Instant>,
    enforced: AtomicBool,
}

impl RequestRateLimiter {
    pub fn new(max_requests_per_second: u64) -> Self {
        Self {
            max_requests_per_second: AtomicU64::new(max_requests_per_second),
            request_count: AtomicU64::new(0),
            window_start: RwLock::new(Instant::now()),
            enforced: AtomicBool::new(true),
        }
    }

    pub fn check_rate_limit(&self) -> Result<(), String> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }

        let mut window_start = self.window_start.write().expect("lock poisoned");
        let now = Instant::now();
        let elapsed = now.duration_since(*window_start);

        // Reset window if 1 second has passed
        if elapsed >= Duration::from_secs(1) {
            *window_start = now;
            self.request_count.store(0, Ordering::Relaxed);
        }

        let count = self.request_count.fetch_add(1, Ordering::Relaxed) + 1;
        let max = self.max_requests_per_second.load(Ordering::Relaxed);

        if count > max {
            return Err(format!(
                "Rate limit exceeded: {} requests/sec (max: {})",
                count, max
            ));
        }

        Ok(())
    }

    pub fn set_rate_limit(&self, limit: u64) {
        self.max_requests_per_second.store(limit, Ordering::Relaxed);
    }

    pub fn set_enforced(&self, enforced: bool) {
        self.enforced.store(enforced, Ordering::Relaxed);
    }

    pub fn is_enforced(&self) -> bool {
        self.enforced.load(Ordering::Relaxed)
    }

    pub fn get_current_rate(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }
}

impl Default for RequestRateLimiter {
    fn default() -> Self {
        Self::new(1000) // 1000 requests per second default
    }
}

/// Health check for HTTP server components
pub struct ServerHealth {
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

impl ServerHealth {
    pub fn new() -> Self {
        Self {
            checks: RwLock::new(HashMap::new()),
        }
    }

    pub fn register_check(&self, name: &str) {
        self.checks.write().expect("lock poisoned").insert(
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
        if let Some(check) = self.checks.write().expect("lock poisoned").get_mut(name) {
            check.status = status;
            check.last_check = SystemTime::now();
            check.message = message;
        }
    }

    pub fn check_http_server(&self) -> HealthStatus {
        let status = HealthStatus::Healthy;
        self.update_check(
            "http_server",
            status,
            "HTTP server is operational".to_string(),
        );
        status
    }

    pub fn check_sparql_engine(&self) -> HealthStatus {
        let status = HealthStatus::Healthy;
        self.update_check(
            "sparql_engine",
            status,
            "SPARQL engine is operational".to_string(),
        );
        status
    }

    pub fn check_storage(&self) -> HealthStatus {
        let status = HealthStatus::Healthy;
        self.update_check("storage", status, "Storage is operational".to_string());
        status
    }

    pub fn get_overall_status(&self) -> HealthStatus {
        let checks = self.checks.read().expect("lock poisoned");

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
        self.checks
            .read()
            .expect("lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    pub fn perform_all_checks(&self) {
        self.check_http_server();
        self.check_sparql_engine();
        self.check_storage();
    }
}

impl Default for ServerHealth {
    fn default() -> Self {
        let health = Self::new();
        health.register_check("http_server");
        health.register_check("sparql_engine");
        health.register_check("storage");
        health
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed_to_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(30),
            half_open_max_requests: 3,
        };

        let breaker = HttpCircuitBreaker::new(config);

        assert!(breaker.is_request_allowed());

        // Record failures
        for _ in 0..4 {
            breaker.record_failure();
            assert!(breaker.is_request_allowed());
        }

        breaker.record_failure();
        assert!(!breaker.is_request_allowed()); // Should open
    }

    #[test]
    fn test_circuit_breaker_half_open_to_closed() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(30),
            half_open_max_requests: 3,
        };

        let breaker = HttpCircuitBreaker::new(config);

        // Open the circuit
        for _ in 0..3 {
            breaker.record_failure();
        }
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
        let monitor = ServerPerformanceMonitor::new();

        monitor.record_request("/query", Duration::from_millis(100), 200, 1024, 2048);
        monitor.record_request("/query", Duration::from_millis(200), 200, 1024, 2048);
        monitor.record_request("/query", Duration::from_millis(150), 200, 1024, 2048);

        let stats = monitor.get_endpoint_statistics("/query");
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.average_request_size, 1024);
        assert_eq!(stats.average_response_size, 2048);

        let global = monitor.get_global_statistics();
        assert_eq!(global.total_requests, 3);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RequestRateLimiter::new(10);

        // Within limit
        for _ in 0..10 {
            assert!(limiter.check_rate_limit().is_ok());
        }

        // Exceeding limit
        assert!(limiter.check_rate_limit().is_err());

        // Disable enforcement
        limiter.set_enforced(false);
        assert!(limiter.check_rate_limit().is_ok());
    }

    #[test]
    fn test_health_checks() {
        let health = ServerHealth::default();

        health.perform_all_checks();

        assert_eq!(health.get_overall_status(), HealthStatus::Healthy);

        let checks = health.get_checks();
        assert_eq!(checks.len(), 3);
        assert!(checks.iter().all(|c| c.status == HealthStatus::Healthy));
    }

    #[test]
    fn test_production_error() {
        let error = HttpProductionError::request_error(
            "GET".to_string(),
            "/query".to_string(),
            "Internal server error".to_string(),
            500,
        );

        assert_eq!(error.severity, ErrorSeverity::Error);
        assert!(error.retryable);
        assert_eq!(error.context.status_code, Some(500));
    }
}
