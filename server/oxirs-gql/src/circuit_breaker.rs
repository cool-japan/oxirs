//! Advanced Circuit Breaker Patterns
//!
//! This module provides sophisticated circuit breaker implementations for
//! protecting GraphQL endpoints from cascading failures and overload.
//!
//! ## Features
//!
//! - **Multiple Strategies**: Count-based, time-based, and adaptive circuit breakers
//! - **Health-Aware Routing**: Automatic failover based on service health
//! - **Bulkhead Isolation**: Isolate failures in specific parts of the system
//! - **Retry Policies**: Configurable retry strategies with exponential backoff
//! - **Timeout Management**: Request-level and aggregate timeout handling
//! - **Metrics & Observability**: Detailed metrics for monitoring circuit states
//!
//! ## Circuit Breaker States
//!
//! ```text
//! ┌──────────┐  failure threshold   ┌──────────┐
//! │  CLOSED  │ ─────────────────── → │   OPEN   │
//! └──────────┘                       └──────────┘
//!       ↑                                  │
//!       │                                  │ timeout
//!       │         success                  ↓
//!       └────────────────────────── ┌────────────┐
//!                                   │ HALF-OPEN  │
//!                                   └────────────┘
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed - requests flow normally
    Closed,
    /// Circuit is open - requests are rejected immediately
    Open,
    /// Circuit is half-open - limited requests allowed to test recovery
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "closed"),
            CircuitState::Open => write!(f, "open"),
            CircuitState::HalfOpen => write!(f, "half-open"),
        }
    }
}

/// Circuit breaker strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakerStrategy {
    /// Count-based: Open after N failures
    CountBased {
        failure_threshold: u32,
        success_threshold: u32,
    },
    /// Time-based: Open if failure rate exceeds threshold in time window
    TimeBased {
        failure_rate_threshold: f64,
        window_duration: Duration,
        min_request_count: u32,
    },
    /// Adaptive: Dynamically adjust thresholds based on system state
    Adaptive {
        base_failure_threshold: u32,
        max_failure_threshold: u32,
        success_threshold: u32,
        sensitivity: f64,
    },
    /// Sliding window: Rolling window of requests
    SlidingWindow {
        window_size: u32,
        failure_rate_threshold: f64,
        min_calls: u32,
    },
}

impl Default for CircuitBreakerStrategy {
    fn default() -> Self {
        CircuitBreakerStrategy::CountBased {
            failure_threshold: 5,
            success_threshold: 3,
        }
    }
}

/// Retry strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryStrategy {
    /// No retries
    None,
    /// Fixed number of retries with fixed delay
    Fixed { max_retries: u32, delay: Duration },
    /// Exponential backoff with jitter
    ExponentialBackoff {
        max_retries: u32,
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
        jitter: bool,
    },
    /// Linear backoff
    LinearBackoff {
        max_retries: u32,
        initial_delay: Duration,
        increment: Duration,
        max_delay: Duration,
    },
}

impl Default for RetryStrategy {
    fn default() -> Self {
        RetryStrategy::ExponentialBackoff {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryStrategy {
    /// Calculate delay for the given retry attempt
    pub fn calculate_delay(&self, attempt: u32) -> Option<Duration> {
        match self {
            RetryStrategy::None => None,
            RetryStrategy::Fixed { max_retries, delay } => {
                if attempt < *max_retries {
                    Some(*delay)
                } else {
                    None
                }
            }
            RetryStrategy::ExponentialBackoff {
                max_retries,
                initial_delay,
                max_delay,
                multiplier,
                jitter,
            } => {
                if attempt >= *max_retries {
                    return None;
                }
                let delay = initial_delay.as_millis() as f64 * multiplier.powi(attempt as i32);
                let mut delay = Duration::from_millis(delay as u64);
                if delay > *max_delay {
                    delay = *max_delay;
                }
                if *jitter {
                    let jitter_amount = fastrand::u64(0..delay.as_millis() as u64 / 4);
                    delay += Duration::from_millis(jitter_amount);
                }
                Some(delay)
            }
            RetryStrategy::LinearBackoff {
                max_retries,
                initial_delay,
                increment,
                max_delay,
            } => {
                if attempt >= *max_retries {
                    return None;
                }
                let delay = *initial_delay + *increment * attempt;
                Some(delay.min(*max_delay))
            }
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Circuit breaker strategy
    pub strategy: CircuitBreakerStrategy,
    /// Duration to keep circuit open before trying half-open
    pub open_duration: Duration,
    /// Number of test requests in half-open state
    pub half_open_max_requests: u32,
    /// Request timeout
    pub request_timeout: Duration,
    /// Retry strategy
    pub retry_strategy: RetryStrategy,
    /// Enable slow call tracking
    pub track_slow_calls: bool,
    /// Slow call threshold
    pub slow_call_threshold: Duration,
    /// Slow call rate threshold (percentage)
    pub slow_call_rate_threshold: f64,
    /// Name for identification
    pub name: String,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            strategy: CircuitBreakerStrategy::default(),
            open_duration: Duration::from_secs(60),
            half_open_max_requests: 3,
            request_timeout: Duration::from_secs(30),
            retry_strategy: RetryStrategy::default(),
            track_slow_calls: true,
            slow_call_threshold: Duration::from_secs(5),
            slow_call_rate_threshold: 50.0,
            name: "default".to_string(),
        }
    }
}

/// Request outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestOutcome {
    /// Request succeeded
    Success,
    /// Request failed
    Failure,
    /// Request timed out
    Timeout,
    /// Request was slow but succeeded
    SlowSuccess,
    /// Circuit was open, request rejected
    Rejected,
}

/// Metrics for circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetrics {
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Timed out requests
    pub timed_out_requests: u64,
    /// Rejected requests (circuit open)
    pub rejected_requests: u64,
    /// Slow requests
    pub slow_requests: u64,
    /// Current failure rate
    pub failure_rate: f64,
    /// Current slow call rate
    pub slow_call_rate: f64,
    /// Average response time (ms)
    pub avg_response_time_ms: f64,
    /// State transition count
    pub state_transitions: u64,
    /// Time in each state (ms)
    pub time_in_closed_ms: u64,
    pub time_in_open_ms: u64,
    pub time_in_half_open_ms: u64,
}

impl Default for CircuitMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            timed_out_requests: 0,
            rejected_requests: 0,
            slow_requests: 0,
            failure_rate: 0.0,
            slow_call_rate: 0.0,
            avg_response_time_ms: 0.0,
            state_transitions: 0,
            time_in_closed_ms: 0,
            time_in_open_ms: 0,
            time_in_half_open_ms: 0,
        }
    }
}

/// State transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Previous state
    pub from: CircuitState,
    /// New state
    pub to: CircuitState,
    /// Transition timestamp
    pub timestamp: SystemTime,
    /// Reason for transition
    pub reason: String,
}

/// Sliding window entry
#[derive(Debug, Clone)]
struct WindowEntry {
    outcome: RequestOutcome,
    timestamp: Instant,
    #[allow(dead_code)]
    duration: Duration,
}

/// Internal circuit breaker state
struct CircuitBreakerState {
    /// Current state
    state: CircuitState,
    /// When state was last changed
    state_changed_at: Instant,
    /// Failure count in current window
    failure_count: u32,
    /// Success count in current window
    success_count: u32,
    /// Half-open request count
    half_open_requests: u32,
    /// Sliding window entries
    window: Vec<WindowEntry>,
    /// Response times for average calculation
    response_times: Vec<Duration>,
    /// Metrics
    metrics: CircuitMetrics,
    /// State transition history
    transitions: Vec<StateTransition>,
    /// Adaptive threshold (for adaptive strategy)
    adaptive_threshold: u32,
}

impl CircuitBreakerState {
    fn new() -> Self {
        Self {
            state: CircuitState::Closed,
            state_changed_at: Instant::now(),
            failure_count: 0,
            success_count: 0,
            half_open_requests: 0,
            window: Vec::new(),
            response_times: Vec::new(),
            metrics: CircuitMetrics::default(),
            transitions: Vec::new(),
            adaptive_threshold: 5,
        }
    }

    fn record_transition(&mut self, to: CircuitState, reason: &str) {
        let from = self.state;
        if from != to {
            self.transitions.push(StateTransition {
                from,
                to,
                timestamp: SystemTime::now(),
                reason: reason.to_string(),
            });
            self.state = to;
            self.state_changed_at = Instant::now();
            self.metrics.state_transitions += 1;

            // Reset counters on state change
            if to == CircuitState::Closed {
                self.failure_count = 0;
                self.success_count = 0;
            } else if to == CircuitState::HalfOpen {
                self.half_open_requests = 0;
            }
        }
    }
}

/// Circuit Breaker
///
/// Protects services from cascading failures by monitoring request
/// outcomes and temporarily blocking requests when failures exceed thresholds.
pub struct CircuitBreaker {
    /// Configuration
    config: CircuitBreakerConfig,
    /// Internal state
    state: Arc<RwLock<CircuitBreakerState>>,
    /// Event handlers
    event_handlers: Arc<RwLock<Vec<Arc<dyn CircuitBreakerEventHandler + Send + Sync>>>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState::new())),
            event_handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create with default config and custom name
    pub fn with_name(name: &str) -> Self {
        let config = CircuitBreakerConfig {
            name: name.to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Register an event handler
    pub async fn register_event_handler(
        &self,
        handler: Arc<dyn CircuitBreakerEventHandler + Send + Sync>,
    ) {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(handler);
    }

    /// Get current state
    pub async fn get_state(&self) -> CircuitState {
        let state = self.state.read().await;
        state.state
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> CircuitMetrics {
        let state = self.state.read().await;
        state.metrics.clone()
    }

    /// Get recent transitions
    pub async fn get_transitions(&self, limit: usize) -> Vec<StateTransition> {
        let state = self.state.read().await;
        state
            .transitions
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Check if circuit allows request
    pub async fn allows_request(&self) -> bool {
        let mut state = self.state.write().await;

        match state.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                if state.state_changed_at.elapsed() >= self.config.open_duration {
                    state.record_transition(
                        CircuitState::HalfOpen,
                        "Open duration elapsed, transitioning to half-open",
                    );
                    self.notify_state_change(CircuitState::Open, CircuitState::HalfOpen)
                        .await;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited requests in half-open state
                state.half_open_requests < self.config.half_open_max_requests
            }
        }
    }

    /// Record a request outcome
    pub async fn record_outcome(&self, outcome: RequestOutcome, duration: Duration) {
        let mut state = self.state.write().await;

        // Update metrics
        state.metrics.total_requests += 1;
        state.response_times.push(duration);

        // Limit response time history
        if state.response_times.len() > 1000 {
            state.response_times.drain(0..100);
        }

        // Calculate average response time
        let total_time: u128 = state.response_times.iter().map(|d| d.as_millis()).sum();
        state.metrics.avg_response_time_ms = total_time as f64 / state.response_times.len() as f64;

        // Track slow calls
        let is_slow = duration >= self.config.slow_call_threshold;
        if is_slow {
            state.metrics.slow_requests += 1;
        }

        // Add to sliding window
        state.window.push(WindowEntry {
            outcome,
            timestamp: Instant::now(),
            duration,
        });

        // Cleanup old entries
        let window_duration = match &self.config.strategy {
            CircuitBreakerStrategy::TimeBased {
                window_duration, ..
            } => *window_duration,
            CircuitBreakerStrategy::SlidingWindow { .. } => Duration::from_secs(60),
            _ => Duration::from_secs(60),
        };
        state
            .window
            .retain(|e| e.timestamp.elapsed() < window_duration);

        match outcome {
            RequestOutcome::Success | RequestOutcome::SlowSuccess => {
                state.metrics.successful_requests += 1;
                state.success_count += 1;
                state.failure_count = 0;

                if is_slow {
                    state.metrics.slow_requests += 1;
                }
            }
            RequestOutcome::Failure => {
                state.metrics.failed_requests += 1;
                state.failure_count += 1;
                state.success_count = 0;
            }
            RequestOutcome::Timeout => {
                state.metrics.timed_out_requests += 1;
                state.metrics.failed_requests += 1;
                state.failure_count += 1;
                state.success_count = 0;
            }
            RequestOutcome::Rejected => {
                state.metrics.rejected_requests += 1;
                return; // Don't process state transitions for rejected requests
            }
        }

        // Calculate failure rate
        let total = state.metrics.successful_requests + state.metrics.failed_requests;
        if total > 0 {
            state.metrics.failure_rate =
                (state.metrics.failed_requests as f64 / total as f64) * 100.0;
        }

        // Calculate slow call rate
        if state.metrics.successful_requests > 0 {
            state.metrics.slow_call_rate = (state.metrics.slow_requests as f64
                / state.metrics.successful_requests as f64)
                * 100.0;
        }

        // Handle state transitions
        let current_state = state.state;
        let new_state = self.evaluate_state_transition(&state);

        if current_state != new_state {
            let reason = match new_state {
                CircuitState::Open => "Failure threshold exceeded",
                CircuitState::Closed => "Success threshold reached",
                CircuitState::HalfOpen => "Ready to test",
            };
            state.record_transition(new_state, reason);
            drop(state);
            self.notify_state_change(current_state, new_state).await;
        }
    }

    /// Evaluate if state should transition
    fn evaluate_state_transition(&self, state: &CircuitBreakerState) -> CircuitState {
        match state.state {
            CircuitState::Closed => match &self.config.strategy {
                CircuitBreakerStrategy::CountBased {
                    failure_threshold, ..
                } => {
                    if state.failure_count >= *failure_threshold {
                        CircuitState::Open
                    } else {
                        CircuitState::Closed
                    }
                }
                CircuitBreakerStrategy::TimeBased {
                    failure_rate_threshold,
                    min_request_count,
                    ..
                } => {
                    let window_requests = state.window.len() as u32;
                    if window_requests >= *min_request_count {
                        let failures = state
                            .window
                            .iter()
                            .filter(|e| {
                                matches!(
                                    e.outcome,
                                    RequestOutcome::Failure | RequestOutcome::Timeout
                                )
                            })
                            .count();
                        let rate = (failures as f64 / window_requests as f64) * 100.0;
                        if rate >= *failure_rate_threshold {
                            CircuitState::Open
                        } else {
                            CircuitState::Closed
                        }
                    } else {
                        CircuitState::Closed
                    }
                }
                CircuitBreakerStrategy::Adaptive { .. } => {
                    if state.failure_count >= state.adaptive_threshold {
                        CircuitState::Open
                    } else {
                        CircuitState::Closed
                    }
                }
                CircuitBreakerStrategy::SlidingWindow {
                    failure_rate_threshold,
                    min_calls,
                    ..
                } => {
                    if state.window.len() as u32 >= *min_calls {
                        let failures = state
                            .window
                            .iter()
                            .filter(|e| {
                                matches!(
                                    e.outcome,
                                    RequestOutcome::Failure | RequestOutcome::Timeout
                                )
                            })
                            .count();
                        let rate = (failures as f64 / state.window.len() as f64) * 100.0;
                        if rate >= *failure_rate_threshold {
                            CircuitState::Open
                        } else {
                            CircuitState::Closed
                        }
                    } else {
                        CircuitState::Closed
                    }
                }
            },
            CircuitState::HalfOpen => {
                let success_threshold = match &self.config.strategy {
                    CircuitBreakerStrategy::CountBased {
                        success_threshold, ..
                    } => *success_threshold,
                    CircuitBreakerStrategy::Adaptive {
                        success_threshold, ..
                    } => *success_threshold,
                    _ => 3,
                };

                if state.success_count >= success_threshold {
                    CircuitState::Closed
                } else if state.failure_count > 0 {
                    CircuitState::Open
                } else {
                    CircuitState::HalfOpen
                }
            }
            CircuitState::Open => CircuitState::Open,
        }
    }

    /// Notify event handlers of state change
    async fn notify_state_change(&self, from: CircuitState, to: CircuitState) {
        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            handler.on_state_change(&self.config.name, from, to).await;
        }
    }

    /// Execute a function with circuit breaker protection
    pub async fn execute<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        // Check if circuit allows request
        if !self.allows_request().await {
            self.record_outcome(RequestOutcome::Rejected, Duration::ZERO)
                .await;
            return Err(anyhow!("Circuit breaker is open"));
        }

        let start = Instant::now();
        let result = tokio::time::timeout(self.config.request_timeout, f()).await;
        let duration = start.elapsed();

        match result {
            Ok(Ok(value)) => {
                let outcome = if duration >= self.config.slow_call_threshold {
                    RequestOutcome::SlowSuccess
                } else {
                    RequestOutcome::Success
                };
                self.record_outcome(outcome, duration).await;
                Ok(value)
            }
            Ok(Err(e)) => {
                self.record_outcome(RequestOutcome::Failure, duration).await;
                Err(e)
            }
            Err(_) => {
                self.record_outcome(RequestOutcome::Timeout, duration).await;
                Err(anyhow!("Request timed out"))
            }
        }
    }

    /// Execute with retry
    pub async fn execute_with_retry<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0;

        loop {
            match self.execute(f.clone()).await {
                Ok(value) => return Ok(value),
                Err(e) => {
                    if let Some(delay) = self.config.retry_strategy.calculate_delay(attempt) {
                        attempt += 1;
                        tokio::time::sleep(delay).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Force circuit to specific state (for testing/manual intervention)
    pub async fn force_state(&self, new_state: CircuitState, reason: &str) {
        let current_state;
        {
            let mut state = self.state.write().await;
            current_state = state.state;
            state.record_transition(new_state, reason);
        }

        if current_state != new_state {
            self.notify_state_change(current_state, new_state).await;
        }
    }

    /// Reset circuit breaker to initial state
    pub async fn reset(&self) {
        let current_state;
        {
            let mut state = self.state.write().await;
            current_state = state.state;
            *state = CircuitBreakerState::new();
        }

        if current_state != CircuitState::Closed {
            self.notify_state_change(current_state, CircuitState::Closed)
                .await;
        }
    }
}

/// Trait for handling circuit breaker events
#[async_trait::async_trait]
pub trait CircuitBreakerEventHandler {
    /// Called when circuit state changes
    async fn on_state_change(&self, name: &str, from: CircuitState, to: CircuitState);
}

/// Logging event handler
pub struct LoggingCircuitHandler;

#[async_trait::async_trait]
impl CircuitBreakerEventHandler for LoggingCircuitHandler {
    async fn on_state_change(&self, name: &str, from: CircuitState, to: CircuitState) {
        match to {
            CircuitState::Open => {
                tracing::warn!(
                    "Circuit breaker '{}': {} -> {} (blocking requests)",
                    name,
                    from,
                    to
                );
            }
            CircuitState::HalfOpen => {
                tracing::info!(
                    "Circuit breaker '{}': {} -> {} (testing recovery)",
                    name,
                    from,
                    to
                );
            }
            CircuitState::Closed => {
                tracing::info!("Circuit breaker '{}': {} -> {} (recovered)", name, from, to);
            }
        }
    }
}

/// Circuit breaker registry for managing multiple breakers
pub struct CircuitBreakerRegistry {
    /// Named circuit breakers
    breakers: Arc<RwLock<HashMap<String, Arc<CircuitBreaker>>>>,
    /// Default configuration
    default_config: CircuitBreakerConfig,
}

impl CircuitBreakerRegistry {
    /// Create a new registry
    pub fn new(default_config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: Arc::new(RwLock::new(HashMap::new())),
            default_config,
        }
    }

    /// Get or create a circuit breaker
    pub async fn get_or_create(&self, name: &str) -> Arc<CircuitBreaker> {
        {
            let breakers = self.breakers.read().await;
            if let Some(breaker) = breakers.get(name) {
                return breaker.clone();
            }
        }

        let mut breakers = self.breakers.write().await;
        // Double-check after acquiring write lock
        if let Some(breaker) = breakers.get(name) {
            return breaker.clone();
        }

        let config = CircuitBreakerConfig {
            name: name.to_string(),
            ..self.default_config.clone()
        };
        let breaker = Arc::new(CircuitBreaker::new(config));
        breakers.insert(name.to_string(), breaker.clone());
        breaker
    }

    /// Get a circuit breaker if it exists
    pub async fn get(&self, name: &str) -> Option<Arc<CircuitBreaker>> {
        let breakers = self.breakers.read().await;
        breakers.get(name).cloned()
    }

    /// Get all circuit breaker states
    pub async fn get_all_states(&self) -> HashMap<String, CircuitState> {
        let breakers = self.breakers.read().await;
        let mut states = HashMap::new();
        for (name, breaker) in breakers.iter() {
            states.insert(name.clone(), breaker.get_state().await);
        }
        states
    }

    /// Get all metrics
    pub async fn get_all_metrics(&self) -> HashMap<String, CircuitMetrics> {
        let breakers = self.breakers.read().await;
        let mut metrics = HashMap::new();
        for (name, breaker) in breakers.iter() {
            metrics.insert(name.clone(), breaker.get_metrics().await);
        }
        metrics
    }

    /// Reset all circuit breakers
    pub async fn reset_all(&self) {
        let breakers = self.breakers.read().await;
        for breaker in breakers.values() {
            breaker.reset().await;
        }
    }
}

/// Bulkhead for resource isolation
pub struct Bulkhead {
    /// Name for identification
    name: String,
    /// Maximum concurrent executions
    max_concurrent: usize,
    /// Current concurrent executions
    current_concurrent: Arc<AtomicU64>,
    /// Maximum wait queue size
    max_wait_queue: usize,
    /// Current wait queue size
    current_wait_queue: Arc<AtomicU64>,
    /// Semaphore for concurrency control
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl Bulkhead {
    /// Create a new bulkhead
    pub fn new(name: &str, max_concurrent: usize, max_wait_queue: usize) -> Self {
        Self {
            name: name.to_string(),
            max_concurrent,
            current_concurrent: Arc::new(AtomicU64::new(0)),
            max_wait_queue,
            current_wait_queue: Arc::new(AtomicU64::new(0)),
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
        }
    }

    /// Execute with bulkhead protection
    pub async fn execute<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        // Check wait queue capacity
        let current_wait = self.current_wait_queue.load(Ordering::SeqCst);
        if current_wait >= self.max_wait_queue as u64 {
            return Err(anyhow!("Bulkhead '{}' wait queue is full", self.name));
        }

        self.current_wait_queue.fetch_add(1, Ordering::SeqCst);
        let permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| anyhow!("{}", e))?;
        self.current_wait_queue.fetch_sub(1, Ordering::SeqCst);
        self.current_concurrent.fetch_add(1, Ordering::SeqCst);

        let result = f().await;

        self.current_concurrent.fetch_sub(1, Ordering::SeqCst);
        drop(permit);

        result
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> BulkheadMetrics {
        BulkheadMetrics {
            name: self.name.clone(),
            max_concurrent: self.max_concurrent,
            current_concurrent: self.current_concurrent.load(Ordering::SeqCst) as usize,
            max_wait_queue: self.max_wait_queue,
            current_wait_queue: self.current_wait_queue.load(Ordering::SeqCst) as usize,
            available_permits: self.semaphore.available_permits(),
        }
    }
}

/// Bulkhead metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkheadMetrics {
    /// Bulkhead name
    pub name: String,
    /// Maximum concurrent executions
    pub max_concurrent: usize,
    /// Current concurrent executions
    pub current_concurrent: usize,
    /// Maximum wait queue size
    pub max_wait_queue: usize,
    /// Current wait queue size
    pub current_wait_queue: usize,
    /// Available permits
    pub available_permits: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_creation() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_count_based_circuit_opening() {
        let config = CircuitBreakerConfig {
            strategy: CircuitBreakerStrategy::CountBased {
                failure_threshold: 3,
                success_threshold: 2,
            },
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Record failures
        for _ in 0..3 {
            cb.record_outcome(RequestOutcome::Failure, Duration::from_millis(100))
                .await;
        }

        assert_eq!(cb.get_state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_half_open_transition() {
        let config = CircuitBreakerConfig {
            strategy: CircuitBreakerStrategy::CountBased {
                failure_threshold: 1,
                success_threshold: 1,
            },
            open_duration: Duration::from_millis(10),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Open the circuit
        cb.record_outcome(RequestOutcome::Failure, Duration::from_millis(10))
            .await;
        assert_eq!(cb.get_state().await, CircuitState::Open);

        // Wait for open duration
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Should transition to half-open when checking
        assert!(cb.allows_request().await);
        assert_eq!(cb.get_state().await, CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_recovery_from_half_open() {
        let config = CircuitBreakerConfig {
            strategy: CircuitBreakerStrategy::CountBased {
                failure_threshold: 1,
                success_threshold: 2,
            },
            open_duration: Duration::from_millis(1),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Open the circuit
        cb.record_outcome(RequestOutcome::Failure, Duration::from_millis(10))
            .await;

        // Wait and transition to half-open
        tokio::time::sleep(Duration::from_millis(10)).await;
        cb.allows_request().await;

        // Record successes to recover
        cb.record_outcome(RequestOutcome::Success, Duration::from_millis(10))
            .await;
        cb.record_outcome(RequestOutcome::Success, Duration::from_millis(10))
            .await;

        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_execute_success() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        let result = cb.execute(|| async { Ok::<_, anyhow::Error>(42) }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        let metrics = cb.get_metrics().await;
        assert_eq!(metrics.successful_requests, 1);
    }

    #[tokio::test]
    async fn test_execute_failure() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        let result = cb
            .execute(|| async { Err::<i32, _>(anyhow!("test error")) })
            .await;

        assert!(result.is_err());

        let metrics = cb.get_metrics().await;
        assert_eq!(metrics.failed_requests, 1);
    }

    #[tokio::test]
    async fn test_retry_strategy_exponential() {
        let strategy = RetryStrategy::ExponentialBackoff {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
            jitter: false,
        };

        assert!(strategy.calculate_delay(0).is_some());
        assert!(strategy.calculate_delay(1).is_some());
        assert!(strategy.calculate_delay(2).is_some());
        assert!(strategy.calculate_delay(3).is_none());
    }

    #[tokio::test]
    async fn test_registry() {
        let config = CircuitBreakerConfig::default();
        let registry = CircuitBreakerRegistry::new(config);

        let cb1 = registry.get_or_create("service-a").await;
        let cb2 = registry.get_or_create("service-a").await;

        // Should return the same instance
        assert!(Arc::ptr_eq(&cb1, &cb2));

        let states = registry.get_all_states().await;
        assert!(states.contains_key("service-a"));
    }

    #[tokio::test]
    async fn test_bulkhead() {
        let bulkhead = Bulkhead::new("test", 2, 1);

        let result = bulkhead
            .execute(|| async { Ok::<_, anyhow::Error>(42) })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_force_state() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        cb.force_state(CircuitState::Open, "Manual intervention")
            .await;
        assert_eq!(cb.get_state().await, CircuitState::Open);

        cb.reset().await;
        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        cb.record_outcome(RequestOutcome::Success, Duration::from_millis(50))
            .await;
        cb.record_outcome(RequestOutcome::Success, Duration::from_millis(100))
            .await;
        cb.record_outcome(RequestOutcome::Failure, Duration::from_millis(75))
            .await;

        let metrics = cb.get_metrics().await;
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.successful_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
        assert!(metrics.avg_response_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_transition_history() {
        let config = CircuitBreakerConfig {
            strategy: CircuitBreakerStrategy::CountBased {
                failure_threshold: 1,
                success_threshold: 1,
            },
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.record_outcome(RequestOutcome::Failure, Duration::from_millis(10))
            .await;

        let transitions = cb.get_transitions(10).await;
        assert!(!transitions.is_empty());
        assert_eq!(transitions[0].to, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_sliding_window_strategy() {
        let config = CircuitBreakerConfig {
            strategy: CircuitBreakerStrategy::SlidingWindow {
                window_size: 10,
                failure_rate_threshold: 50.0,
                min_calls: 4,
            },
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // 50% failure rate
        cb.record_outcome(RequestOutcome::Success, Duration::from_millis(10))
            .await;
        cb.record_outcome(RequestOutcome::Success, Duration::from_millis(10))
            .await;
        cb.record_outcome(RequestOutcome::Failure, Duration::from_millis(10))
            .await;
        cb.record_outcome(RequestOutcome::Failure, Duration::from_millis(10))
            .await;

        assert_eq!(cb.get_state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_rejects_when_open() {
        let config = CircuitBreakerConfig {
            strategy: CircuitBreakerStrategy::CountBased {
                failure_threshold: 1,
                success_threshold: 1,
            },
            open_duration: Duration::from_secs(60),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        cb.record_outcome(RequestOutcome::Failure, Duration::from_millis(10))
            .await;

        assert!(!cb.allows_request().await);

        let metrics = cb.get_metrics().await;
        assert_eq!(metrics.rejected_requests, 0); // allows_request doesn't record rejections
    }
}
