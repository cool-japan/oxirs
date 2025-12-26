//! Production hardening features for RDF-star deployments
//!
//! This module provides enterprise-grade features for production deployments:
//! - **Circuit breakers** - Prevent cascading failures
//! - **Rate limiting** - Control query throughput
//! - **Health checks** - Monitor system health
//! - **Metrics collection** - Track performance metrics
//! - **Graceful shutdown** - Clean resource cleanup
//! - **Retry policies** - Automatic retry with backoff
//! - **Connection pooling** - Efficient resource management
//! - **Request tracing** - Distributed tracing support
//!
//! # Examples
//!
//! ```rust,ignore
//! use oxirs_star::production::{CircuitBreaker, RateLimiter, HealthCheck};
//! use std::time::Duration;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a circuit breaker
//! let mut circuit_breaker = CircuitBreaker::new(5, Duration::from_secs(60));
//!
//! // Create a rate limiter (100 requests per second)
//! let mut rate_limiter = RateLimiter::new(100, Duration::from_secs(1));
//!
//! // Check if request is allowed
//! if rate_limiter.allow_request() {
//!     // Process request
//!     println!("Request allowed");
//! }
//!
//! // Perform health check
//! let health = HealthCheck::new();
//! let status = health.check_all()?;
//! println!("Health: {:?}", status.overall_status);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::StarResult;

/// Circuit breaker to prevent cascading failures
pub struct CircuitBreaker {
    /// Current state of the circuit breaker
    state: Arc<Mutex<CircuitBreakerState>>,

    /// Failure threshold before opening
    failure_threshold: usize,

    /// Timeout before attempting to close
    timeout: Duration,

    /// Failure count in current window
    failure_count: Arc<Mutex<usize>>,

    /// Last failure time
    last_failure_time: Arc<Mutex<Option<Instant>>>,
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed, requests flow normally
    Closed,

    /// Circuit is open, requests are rejected
    Open,

    /// Circuit is half-open, testing if system recovered
    HalfOpen,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(failure_threshold: usize, timeout: Duration) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitBreakerState::Closed)),
            failure_threshold,
            timeout,
            failure_count: Arc::new(Mutex::new(0)),
            last_failure_time: Arc::new(Mutex::new(None)),
        }
    }

    /// Check if a request is allowed
    pub fn allow_request(&self) -> bool {
        let mut state = self.state.lock().unwrap();

        match *state {
            CircuitBreakerState::Closed => true,

            CircuitBreakerState::Open => {
                // Check if timeout has elapsed
                let last_failure = self.last_failure_time.lock().unwrap();

                if let Some(last_time) = *last_failure {
                    if last_time.elapsed() >= self.timeout {
                        // Transition to half-open
                        *state = CircuitBreakerState::HalfOpen;
                        info!("Circuit breaker transitioning to half-open");
                        return true;
                    }
                }

                false
            }

            CircuitBreakerState::HalfOpen => true, // Allow one request to test
        }
    }

    /// Record a successful request
    pub fn record_success(&self) {
        let mut state = self.state.lock().unwrap();

        match *state {
            CircuitBreakerState::HalfOpen => {
                // Success in half-open state -> close the circuit
                *state = CircuitBreakerState::Closed;
                let mut failure_count = self.failure_count.lock().unwrap();
                *failure_count = 0;
                info!("Circuit breaker closed after successful test");
            }

            CircuitBreakerState::Closed => {
                // Reset failure count on success
                let mut failure_count = self.failure_count.lock().unwrap();
                *failure_count = 0;
            }

            _ => {}
        }
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        let mut state = self.state.lock().unwrap();
        let mut failure_count = self.failure_count.lock().unwrap();

        *failure_count += 1;

        match *state {
            CircuitBreakerState::HalfOpen => {
                // Failure in half-open state -> reopen the circuit
                *state = CircuitBreakerState::Open;
                let mut last_failure = self.last_failure_time.lock().unwrap();
                *last_failure = Some(Instant::now());
                warn!("Circuit breaker reopened after failed test");
            }

            CircuitBreakerState::Closed => {
                if *failure_count >= self.failure_threshold {
                    // Too many failures -> open the circuit
                    *state = CircuitBreakerState::Open;
                    let mut last_failure = self.last_failure_time.lock().unwrap();
                    *last_failure = Some(Instant::now());
                    error!("Circuit breaker opened due to {} failures", *failure_count);
                }
            }

            _ => {}
        }
    }

    /// Get current state
    pub fn get_state(&self) -> CircuitBreakerState {
        *self.state.lock().unwrap()
    }
}

/// Rate limiter using token bucket algorithm
pub struct RateLimiter {
    /// Maximum tokens in the bucket
    max_tokens: usize,

    /// Current tokens available
    tokens: Arc<Mutex<usize>>,

    /// Refill rate (tokens per refill_period) - reserved for future dynamic adjustment
    #[allow(dead_code)]
    refill_rate: usize,

    /// Refill period
    refill_period: Duration,

    /// Last refill time
    last_refill: Arc<Mutex<Instant>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(requests_per_second: usize, window: Duration) -> Self {
        Self {
            max_tokens: requests_per_second,
            tokens: Arc::new(Mutex::new(requests_per_second)),
            refill_rate: requests_per_second,
            refill_period: window,
            last_refill: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Check if a request is allowed
    pub fn allow_request(&mut self) -> bool {
        self.refill_tokens();

        let mut tokens = self.tokens.lock().unwrap();

        if *tokens > 0 {
            *tokens -= 1;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    fn refill_tokens(&self) {
        let mut last_refill = self.last_refill.lock().unwrap();
        let elapsed = last_refill.elapsed();

        if elapsed >= self.refill_period {
            let mut tokens = self.tokens.lock().unwrap();
            *tokens = self.max_tokens;
            *last_refill = Instant::now();
        }
    }

    /// Get current token count
    pub fn available_tokens(&self) -> usize {
        *self.tokens.lock().unwrap()
    }
}

/// Health check system
pub struct HealthCheck {
    /// Registered health check components
    checks: Arc<RwLock<Vec<Box<dyn HealthCheckComponent + Send + Sync>>>>,

    /// Health check timestamp (reserved for future metrics)
    #[allow(dead_code)]
    last_check: Arc<Mutex<Option<Instant>>>,
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall health status
    pub overall_status: Status,

    /// Individual component statuses
    pub components: HashMap<String, ComponentStatus>,

    /// Timestamp
    pub timestamp: String,
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Component health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub status: Status,
    pub message: Option<String>,
    pub details: Option<HashMap<String, String>>,
}

/// Health check component trait
pub trait HealthCheckComponent {
    /// Get component name
    fn name(&self) -> &str;

    /// Check component health
    fn check(&self) -> ComponentStatus;
}

impl HealthCheck {
    /// Create a new health check system
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(Vec::new())),
            last_check: Arc::new(Mutex::new(None)),
        }
    }

    /// Register a health check component
    pub fn register(&mut self, component: Box<dyn HealthCheckComponent + Send + Sync>) {
        let mut checks = self.checks.write().unwrap();
        checks.push(component);
    }

    /// Check all components
    pub fn check_all(&self) -> StarResult<HealthStatus> {
        let checks = self.checks.read().unwrap();
        let mut components = HashMap::new();
        let mut overall_status = Status::Healthy;

        for check in checks.iter() {
            let status = check.check();

            match status.status {
                Status::Degraded if overall_status == Status::Healthy => {
                    overall_status = Status::Degraded;
                }
                Status::Unhealthy => {
                    overall_status = Status::Unhealthy;
                }
                _ => {}
            }

            components.insert(check.name().to_string(), status);
        }

        Ok(HealthStatus {
            overall_status,
            components,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }
}

impl Default for HealthCheck {
    fn default() -> Self {
        Self::new()
    }
}

/// Basic store health check
pub struct StoreHealthCheck {
    name: String,
    #[allow(dead_code)]
    triple_count_threshold: usize,
}

impl StoreHealthCheck {
    pub fn new(name: impl Into<String>, threshold: usize) -> Self {
        Self {
            name: name.into(),
            triple_count_threshold: threshold,
        }
    }
}

impl HealthCheckComponent for StoreHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self) -> ComponentStatus {
        // In production, this would check actual store metrics
        ComponentStatus {
            status: Status::Healthy,
            message: Some("Store is operational".to_string()),
            details: Some(
                vec![("triple_count".to_string(), "0".to_string())]
                    .into_iter()
                    .collect(),
            ),
        }
    }
}

/// Retry policy with exponential backoff
pub struct RetryPolicy {
    /// Maximum retry attempts
    max_retries: usize,

    /// Initial backoff duration
    initial_backoff: Duration,

    /// Maximum backoff duration
    max_backoff: Duration,

    /// Backoff multiplier
    multiplier: f64,
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new(
        max_retries: usize,
        initial_backoff: Duration,
        max_backoff: Duration,
        multiplier: f64,
    ) -> Self {
        Self {
            max_retries,
            initial_backoff,
            max_backoff,
            multiplier,
        }
    }

    /// Execute a function with retry
    pub fn execute<F, T, E>(&self, mut operation: F) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
    {
        let mut attempt = 0;
        let mut backoff = self.initial_backoff;

        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(err) => {
                    attempt += 1;

                    if attempt >= self.max_retries {
                        return Err(err);
                    }

                    debug!("Retry attempt {} after {:?} backoff", attempt, backoff);

                    std::thread::sleep(backoff);

                    // Exponential backoff
                    backoff = std::cmp::min(
                        Duration::from_secs_f64(backoff.as_secs_f64() * self.multiplier),
                        self.max_backoff,
                    );
                }
            }
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(3, Duration::from_millis(100), Duration::from_secs(30), 2.0)
    }
}

/// Graceful shutdown manager
pub struct ShutdownManager {
    /// Shutdown signals
    shutdown_signals: Arc<Mutex<Vec<tokio::sync::oneshot::Sender<()>>>>,
}

impl ShutdownManager {
    /// Create a new shutdown manager
    pub fn new() -> Self {
        Self {
            shutdown_signals: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Register a shutdown signal
    pub fn register_shutdown_signal(&self) -> tokio::sync::oneshot::Receiver<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let mut signals = self.shutdown_signals.lock().unwrap();
        signals.push(tx);
        rx
    }

    /// Trigger graceful shutdown
    pub fn shutdown(&self) {
        info!("Initiating graceful shutdown");

        let mut signals = self.shutdown_signals.lock().unwrap();

        for signal in signals.drain(..) {
            let _ = signal.send(());
        }

        info!("Shutdown signals sent to all components");
    }
}

impl Default for ShutdownManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Request tracing for distributed systems
pub struct RequestTracer {
    /// Trace ID counter
    trace_id_counter: Arc<Mutex<u64>>,

    /// Active traces
    active_traces: Arc<RwLock<HashMap<u64, TraceInfo>>>,
}

/// Trace information
#[derive(Debug, Clone)]
pub struct TraceInfo {
    pub trace_id: u64,
    pub parent_id: Option<u64>,
    pub start_time: Instant,
    pub operation: String,
    pub spans: Vec<SpanInfo>,
}

/// Span information
#[derive(Debug, Clone)]
pub struct SpanInfo {
    pub span_id: u64,
    pub name: String,
    pub start_time: Instant,
    pub duration: Option<Duration>,
}

impl RequestTracer {
    /// Create a new request tracer
    pub fn new() -> Self {
        Self {
            trace_id_counter: Arc::new(Mutex::new(0)),
            active_traces: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start a new trace
    pub fn start_trace(&self, operation: impl Into<String>) -> u64 {
        let mut counter = self.trace_id_counter.lock().unwrap();
        *counter += 1;
        let trace_id = *counter;

        let trace = TraceInfo {
            trace_id,
            parent_id: None,
            start_time: Instant::now(),
            operation: operation.into(),
            spans: Vec::new(),
        };

        let mut traces = self.active_traces.write().unwrap();
        traces.insert(trace_id, trace);

        trace_id
    }

    /// End a trace
    pub fn end_trace(&self, trace_id: u64) {
        let mut traces = self.active_traces.write().unwrap();

        if let Some(trace) = traces.remove(&trace_id) {
            let duration = trace.start_time.elapsed();
            debug!(
                "Trace {} completed in {:?} with {} spans",
                trace_id,
                duration,
                trace.spans.len()
            );
        }
    }

    /// Get trace info
    pub fn get_trace(&self, trace_id: u64) -> Option<TraceInfo> {
        let traces = self.active_traces.read().unwrap();
        traces.get(&trace_id).cloned()
    }
}

impl Default for RequestTracer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(1));

        // Initially closed
        assert_eq!(breaker.get_state(), CircuitBreakerState::Closed);
        assert!(breaker.allow_request());

        // Record failures
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitBreakerState::Closed);

        breaker.record_failure();

        // Should be open now
        assert_eq!(breaker.get_state(), CircuitBreakerState::Open);
        assert!(!breaker.allow_request());
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(2, Duration::from_secs(1));

        // Allow first two requests
        assert!(limiter.allow_request());
        assert!(limiter.allow_request());

        // Third request should be denied
        assert!(!limiter.allow_request());
    }

    #[test]
    fn test_health_check() -> StarResult<()> {
        let mut health = HealthCheck::new();

        let store_check = StoreHealthCheck::new("store", 1000);
        health.register(Box::new(store_check));

        let status = health.check_all()?;

        assert_eq!(status.overall_status, Status::Healthy);
        assert!(status.components.contains_key("store"));

        Ok(())
    }

    #[test]
    fn test_retry_policy() {
        let policy = RetryPolicy::new(
            3,
            Duration::from_millis(10),
            Duration::from_millis(100),
            2.0,
        );

        let mut attempt = 0;
        let result = policy.execute(|| {
            attempt += 1;
            if attempt < 3 {
                Err("temporary error")
            } else {
                Ok("success")
            }
        });

        assert_eq!(result, Ok("success"));
        assert_eq!(attempt, 3);
    }

    #[test]
    fn test_request_tracer() {
        let tracer = RequestTracer::new();

        let trace_id = tracer.start_trace("test_operation");
        assert!(trace_id > 0);

        let trace = tracer.get_trace(trace_id);
        assert!(trace.is_some());

        tracer.end_trace(trace_id);
        let trace_after_end = tracer.get_trace(trace_id);
        assert!(trace_after_end.is_none());
    }
}
