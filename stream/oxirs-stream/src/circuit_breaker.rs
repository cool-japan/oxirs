//! # Advanced Circuit Breaker Implementation
//!
//! Enterprise-grade circuit breaker functionality for fault tolerance and resilience
//! in distributed streaming systems with adaptive thresholds, failure classification,
//! metrics integration, and advanced recovery strategies.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Enhanced circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
    pub half_open_max_calls: u32,
    /// Enable adaptive thresholds based on historical performance
    pub adaptive_thresholds: bool,
    /// Minimum window size for adaptive calculations
    pub adaptive_window_size: usize,
    /// Maximum failure rate percentage for adaptive mode
    pub max_failure_rate: f64,
    /// Enable failure type classification
    pub classify_failures: bool,
    /// Timeout for specific failure types
    pub failure_type_timeouts: HashMap<String, Duration>,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Sliding window size for rate calculations
    pub sliding_window_size: usize,
    /// Recovery strategy configuration
    pub recovery_strategy: RecoveryStrategy,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            half_open_max_calls: 3,
            adaptive_thresholds: true,
            adaptive_window_size: 100,
            max_failure_rate: 50.0,
            classify_failures: true,
            failure_type_timeouts: HashMap::new(),
            enable_metrics: true,
            sliding_window_size: 60,
            recovery_strategy: RecoveryStrategy::Exponential,
        }
    }
}

/// Recovery strategies for circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Fixed timeout duration
    Fixed,
    /// Exponential backoff with jitter
    Exponential,
    /// Linear increase in timeout
    Linear,
    /// Adaptive based on recent performance
    Adaptive,
}

/// Failure classification for different error types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FailureType {
    /// Network connectivity issues
    NetworkError,
    /// Authentication/authorization failures
    AuthError,
    /// Service unavailable or overloaded
    ServiceUnavailable,
    /// Request timeout
    Timeout,
    /// Rate limiting
    RateLimited,
    /// Invalid request format
    BadRequest,
    /// Server-side errors
    ServerError,
    /// Unknown or unclassified error
    Unknown,
}

impl FailureType {
    /// Check if this failure type should contribute to circuit breaking
    pub fn is_circuit_breaking(&self) -> bool {
        match self {
            FailureType::NetworkError
            | FailureType::ServiceUnavailable
            | FailureType::Timeout
            | FailureType::ServerError => true,
            FailureType::AuthError
            | FailureType::RateLimited
            | FailureType::BadRequest
            | FailureType::Unknown => false,
        }
    }

    /// Get the weight of this failure type for threshold calculations
    pub fn weight(&self) -> f64 {
        match self {
            FailureType::NetworkError => 1.0,
            FailureType::ServiceUnavailable => 1.5,
            FailureType::Timeout => 0.8,
            FailureType::ServerError => 1.2,
            FailureType::AuthError => 0.3,
            FailureType::RateLimited => 0.5,
            FailureType::BadRequest => 0.1,
            FailureType::Unknown => 0.7,
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Enhanced circuit breaker implementation with advanced features
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitBreakerState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<Instant>,
    half_open_calls: u32,
    /// Sliding window of recent events for adaptive calculations
    event_window: VecDeque<CircuitBreakerEvent>,
    /// Failure counts by type for classification
    failure_by_type: HashMap<FailureType, u32>,
    /// Adaptive threshold calculator
    adaptive_threshold: AdaptiveThreshold,
    /// Recovery strategy handler
    recovery_handler: RecoveryHandler,
    /// Metrics collector
    metrics: CircuitBreakerMetrics,
    /// Circuit breaker ID for tracking
    id: String,
    /// Creation timestamp
    created_at: Instant,
}

/// Event recorded by circuit breaker
#[derive(Debug, Clone)]
struct CircuitBreakerEvent {
    timestamp: Instant,
    event_type: EventType,
    failure_type: Option<FailureType>,
    duration: Duration,
}

#[derive(Debug, Clone)]
enum EventType {
    Success,
    Failure,
    StateChange(CircuitBreakerState),
}

/// Adaptive threshold calculator
#[derive(Debug, Clone)]
struct AdaptiveThreshold {
    enabled: bool,
    window_size: usize,
    max_failure_rate: f64,
    current_threshold: f64,
    baseline_threshold: u32,
}

impl AdaptiveThreshold {
    fn new(config: &CircuitBreakerConfig) -> Self {
        Self {
            enabled: config.adaptive_thresholds,
            window_size: config.adaptive_window_size,
            max_failure_rate: config.max_failure_rate,
            current_threshold: config.failure_threshold as f64,
            baseline_threshold: config.failure_threshold,
        }
    }

    fn calculate_threshold(&mut self, events: &VecDeque<CircuitBreakerEvent>) -> u32 {
        if !self.enabled || events.len() < self.window_size {
            return self.baseline_threshold;
        }

        let recent_events: Vec<_> = events.iter().rev().take(self.window_size).collect();

        let total_events = recent_events.len();
        let failure_events = recent_events
            .iter()
            .filter(|e| matches!(e.event_type, EventType::Failure))
            .count();

        if total_events == 0 {
            return self.baseline_threshold;
        }

        let failure_rate = (failure_events as f64 / total_events as f64) * 100.0;

        // Adjust threshold based on recent failure rate
        let adjustment_factor = if failure_rate > self.max_failure_rate {
            0.8 // Lower threshold if failure rate is high
        } else if failure_rate < 10.0 {
            1.2 // Raise threshold if failure rate is low
        } else {
            1.0 // Keep current threshold
        };

        self.current_threshold = (self.baseline_threshold as f64 * adjustment_factor)
            .max(1.0)
            .min(self.baseline_threshold as f64 * 2.0);

        self.current_threshold as u32
    }
}

/// Recovery strategy handler
#[derive(Debug, Clone)]
struct RecoveryHandler {
    strategy: RecoveryStrategy,
    base_timeout: Duration,
    current_timeout: Duration,
    retry_count: u32,
    last_attempt: Option<Instant>,
}

impl RecoveryHandler {
    fn new(config: &CircuitBreakerConfig) -> Self {
        Self {
            strategy: config.recovery_strategy.clone(),
            base_timeout: config.timeout,
            current_timeout: config.timeout,
            retry_count: 0,
            last_attempt: None,
        }
    }

    fn calculate_timeout(&mut self) -> Duration {
        match self.strategy {
            RecoveryStrategy::Fixed => self.base_timeout,
            RecoveryStrategy::Exponential => {
                let multiplier = 2u64.pow(self.retry_count.min(10));
                let jitter = fastrand::f64() * 0.1 + 0.95; // 5% jitter
                Duration::from_millis(
                    (self.base_timeout.as_millis() as f64 * multiplier as f64 * jitter) as u64,
                )
            }
            RecoveryStrategy::Linear => Duration::from_millis(
                self.base_timeout.as_millis() as u64 * (self.retry_count as u64 + 1),
            ),
            RecoveryStrategy::Adaptive => {
                // Simplified adaptive strategy
                if self.retry_count == 0 {
                    self.base_timeout
                } else {
                    Duration::from_millis(
                        (self.base_timeout.as_millis() as f64
                            * 1.5f64.powi(self.retry_count as i32)) as u64,
                    )
                }
            }
        }
    }

    fn on_failure(&mut self) {
        self.retry_count += 1;
        self.current_timeout = self.calculate_timeout();
        self.last_attempt = Some(Instant::now());
    }

    fn on_success(&mut self) {
        self.retry_count = 0;
        self.current_timeout = self.base_timeout;
        self.last_attempt = None;
    }

    fn should_retry(&self) -> bool {
        if let Some(last_attempt) = self.last_attempt {
            last_attempt.elapsed() >= self.current_timeout
        } else {
            true
        }
    }
}

/// Circuit breaker metrics
#[derive(Debug, Clone, Default)]
pub struct CircuitBreakerMetrics {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    circuit_opened_count: u64,
    circuit_closed_count: u64,
    half_open_attempts: u64,
    average_response_time: Duration,
    last_state_change: Option<Instant>,
    state_durations: HashMap<CircuitBreakerState, Duration>,
}

impl CircuitBreakerMetrics {
    fn record_request(&mut self, success: bool, duration: Duration) {
        self.total_requests += 1;
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }

        // Update average response time
        let total_time =
            self.average_response_time.as_nanos() as f64 * (self.total_requests - 1) as f64;
        self.average_response_time = Duration::from_nanos(
            ((total_time + duration.as_nanos() as f64) / self.total_requests as f64) as u64,
        );
    }

    fn record_state_change(&mut self, from: CircuitBreakerState, to: CircuitBreakerState) {
        match to {
            CircuitBreakerState::Open => self.circuit_opened_count += 1,
            CircuitBreakerState::Closed => self.circuit_closed_count += 1,
            CircuitBreakerState::HalfOpen => self.half_open_attempts += 1,
        }

        if let Some(last_change) = self.last_state_change {
            let duration = last_change.elapsed();
            *self.state_durations.entry(from).or_insert(Duration::ZERO) += duration;
        }

        self.last_state_change = Some(Instant::now());
    }

    fn get_failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.failed_requests as f64 / self.total_requests as f64) * 100.0
        }
    }

    fn get_success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        }
    }
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            adaptive_threshold: AdaptiveThreshold::new(&config),
            recovery_handler: RecoveryHandler::new(&config),
            config,
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            half_open_calls: 0,
            event_window: VecDeque::new(),
            failure_by_type: HashMap::new(),
            metrics: CircuitBreakerMetrics::default(),
            id: Uuid::new_v4().to_string(),
            created_at: Instant::now(),
        }
    }

    /// Create circuit breaker with custom ID
    pub fn with_id(config: CircuitBreakerConfig, id: String) -> Self {
        let mut cb = Self::new(config);
        cb.id = id;
        cb
    }

    pub fn is_open(&self) -> bool {
        self.state == CircuitBreakerState::Open
    }

    pub fn state(&self) -> CircuitBreakerState {
        if !self.config.enabled {
            return CircuitBreakerState::Closed;
        }

        match self.state {
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.config.timeout {
                        CircuitBreakerState::HalfOpen
                    } else {
                        CircuitBreakerState::Open
                    }
                } else {
                    CircuitBreakerState::Open
                }
            }
            other => other,
        }
    }

    /// Record a successful operation
    pub fn record_success(&mut self) {
        self.record_success_with_duration(Duration::from_millis(100))
    }

    /// Record a successful operation with duration
    pub fn record_success_with_duration(&mut self, duration: Duration) {
        if !self.config.enabled {
            return;
        }

        let old_state = self.state;

        // Record event
        self.record_event(EventType::Success, None, duration);

        // Update metrics
        self.metrics.record_request(true, duration);

        // Update recovery handler
        self.recovery_handler.on_success();

        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count = 0;
            }
            CircuitBreakerState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.config.success_threshold {
                    self.transition_to_state(CircuitBreakerState::Closed);
                    self.failure_count = 0;
                    self.success_count = 0;
                    self.half_open_calls = 0;
                }
            }
            CircuitBreakerState::Open => {
                // Should not happen, but reset if it does
                self.transition_to_state(CircuitBreakerState::Closed);
                self.failure_count = 0;
                self.success_count = 0;
            }
        }

        if old_state != self.state {
            info!(
                "Circuit breaker {} transitioned from {:?} to {:?}",
                self.id, old_state, self.state
            );
        }
    }

    /// Record a failure with classification
    pub fn record_failure_with_type(&mut self, failure_type: FailureType) {
        self.record_failure_with_details(failure_type, Duration::from_millis(1000))
    }

    /// Record a failure with type and duration
    pub fn record_failure_with_details(&mut self, failure_type: FailureType, duration: Duration) {
        if !self.config.enabled {
            return;
        }

        let old_state = self.state;

        // Record event
        self.record_event(EventType::Failure, Some(failure_type.clone()), duration);

        // Update metrics
        self.metrics.record_request(false, duration);

        // Update recovery handler
        self.recovery_handler.on_failure();

        // Update failure counts by type
        *self
            .failure_by_type
            .entry(failure_type.clone())
            .or_insert(0) += 1;

        self.last_failure_time = Some(Instant::now());

        // Only count circuit-breaking failures
        if self.config.classify_failures && !failure_type.is_circuit_breaking() {
            debug!("Non-circuit-breaking failure recorded: {:?}", failure_type);
            return;
        }

        // Calculate weighted failure count
        let weighted_failure = if self.config.classify_failures {
            failure_type.weight()
        } else {
            1.0
        };

        // Get adaptive threshold
        let current_threshold = self
            .adaptive_threshold
            .calculate_threshold(&self.event_window);

        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count += weighted_failure as u32;
                if self.failure_count >= current_threshold {
                    self.transition_to_state(CircuitBreakerState::Open);
                }
            }
            CircuitBreakerState::HalfOpen => {
                self.transition_to_state(CircuitBreakerState::Open);
                self.half_open_calls = 0;
                self.success_count = 0;
            }
            CircuitBreakerState::Open => {
                // Already open, no change needed
            }
        }

        if old_state != self.state {
            warn!(
                "Circuit breaker {} opened due to {:?} failure",
                self.id, failure_type
            );
        }
    }

    /// Record a generic failure (backward compatibility)
    pub fn record_failure(&mut self) {
        self.record_failure_with_type(FailureType::Unknown)
    }

    /// Execute a function with circuit breaker protection
    pub fn execute<F, R, E>(&mut self, operation: F) -> Result<R, CircuitBreakerError>
    where
        F: FnOnce() -> std::result::Result<R, E>,
        E: std::fmt::Debug,
    {
        if !self.can_execute() {
            return Err(CircuitBreakerError::CircuitOpen {
                state: self.state,
                last_failure: self.last_failure_time,
            });
        }

        let start = Instant::now();
        match operation() {
            Ok(result) => {
                let duration = start.elapsed();
                self.record_success_with_duration(duration);
                Ok(result)
            }
            Err(error) => {
                let duration = start.elapsed();
                let failure_type = self.classify_error(&error);
                self.record_failure_with_details(failure_type, duration);
                Err(CircuitBreakerError::OperationFailed(format!("{error:?}")))
            }
        }
    }

    /// Execute an async function with circuit breaker protection
    pub async fn execute_async<F, Fut, R, E>(
        &mut self,
        operation: F,
    ) -> Result<R, CircuitBreakerError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<R, E>>,
        E: std::fmt::Debug,
    {
        if !self.can_execute() {
            return Err(CircuitBreakerError::CircuitOpen {
                state: self.state,
                last_failure: self.last_failure_time,
            });
        }

        let start = Instant::now();
        match operation().await {
            Ok(result) => {
                let duration = start.elapsed();
                self.record_success_with_duration(duration);
                Ok(result)
            }
            Err(error) => {
                let duration = start.elapsed();
                let failure_type = self.classify_error(&error);
                self.record_failure_with_details(failure_type, duration);
                Err(CircuitBreakerError::OperationFailed(format!("{error:?}")))
            }
        }
    }

    /// Classify error type for failure handling
    fn classify_error<E: std::fmt::Debug>(&self, error: &E) -> FailureType {
        let error_str = format!("{error:?}").to_lowercase();

        if error_str.contains("timeout") || error_str.contains("timed out") {
            FailureType::Timeout
        } else if error_str.contains("connection") || error_str.contains("network") {
            FailureType::NetworkError
        } else if error_str.contains("unauthorized") || error_str.contains("forbidden") {
            FailureType::AuthError
        } else if error_str.contains("unavailable") || error_str.contains("overload") {
            FailureType::ServiceUnavailable
        } else if error_str.contains("rate limit") || error_str.contains("throttle") {
            FailureType::RateLimited
        } else if error_str.contains("bad request") || error_str.contains("invalid") {
            FailureType::BadRequest
        } else if error_str.contains("server error") || error_str.contains("internal") {
            FailureType::ServerError
        } else {
            FailureType::Unknown
        }
    }

    /// Helper method to record events
    fn record_event(
        &mut self,
        event_type: EventType,
        failure_type: Option<FailureType>,
        duration: Duration,
    ) {
        let event = CircuitBreakerEvent {
            timestamp: Instant::now(),
            event_type,
            failure_type,
            duration,
        };

        self.event_window.push_back(event);

        // Maintain window size
        while self.event_window.len() > self.config.sliding_window_size {
            self.event_window.pop_front();
        }
    }

    /// Helper method to transition between states
    fn transition_to_state(&mut self, new_state: CircuitBreakerState) {
        let old_state = self.state;
        self.state = new_state;

        self.metrics.record_state_change(old_state, new_state);
        self.record_event(EventType::StateChange(new_state), None, Duration::ZERO);

        debug!(
            "Circuit breaker {} transitioned from {:?} to {:?}",
            self.id, old_state, new_state
        );
    }

    /// Check if the circuit breaker allows execution
    pub fn can_execute(&mut self) -> bool {
        if !self.config.enabled {
            return true;
        }

        let current_state = self.state();

        match current_state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if we should transition to half-open using recovery strategy
                if self.recovery_handler.should_retry() {
                    self.transition_to_state(CircuitBreakerState::HalfOpen);
                    self.half_open_calls = 0;
                    self.success_count = 0;
                    true
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => {
                if self.half_open_calls < self.config.half_open_max_calls {
                    self.half_open_calls += 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Get enhanced circuit breaker statistics
    pub fn get_enhanced_stats(&self) -> EnhancedCircuitBreakerStats {
        EnhancedCircuitBreakerStats {
            id: self.id.clone(),
            state: self.state,
            failure_count: self.failure_count,
            success_count: self.success_count,
            half_open_calls: self.half_open_calls,
            last_failure_time: self.last_failure_time,
            failure_by_type: self.failure_by_type.clone(),
            metrics: self.metrics.clone(),
            adaptive_threshold: self.adaptive_threshold.current_threshold,
            recovery_timeout: self.recovery_handler.current_timeout,
            created_at: self.created_at,
            uptime: self.created_at.elapsed(),
        }
    }

    /// Get current failure rate
    pub fn get_failure_rate(&self) -> f64 {
        self.metrics.get_failure_rate()
    }

    /// Get average response time
    pub fn get_average_response_time(&self) -> Duration {
        self.metrics.average_response_time
    }

    /// Reset circuit breaker to closed state
    pub fn reset(&mut self) {
        let old_state = self.state;
        self.transition_to_state(CircuitBreakerState::Closed);
        self.failure_count = 0;
        self.success_count = 0;
        self.half_open_calls = 0;
        self.last_failure_time = None;
        self.recovery_handler.on_success();

        info!(
            "Circuit breaker {} manually reset from {:?}",
            self.id, old_state
        );
    }

    /// Force circuit breaker to open state
    pub fn force_open(&mut self) {
        let old_state = self.state;
        self.transition_to_state(CircuitBreakerState::Open);
        self.last_failure_time = Some(Instant::now());

        warn!(
            "Circuit breaker {} manually forced open from {:?}",
            self.id, old_state
        );
    }

    /// Get circuit breaker ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Check if circuit breaker is healthy
    pub fn is_healthy(&self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::HalfOpen => self.success_count > 0,
            CircuitBreakerState::Open => false,
        }
    }

    /// Get circuit breaker statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.state(),
            failure_count: self.failure_count,
            success_count: self.success_count,
            half_open_calls: self.half_open_calls,
            last_failure_time: self.last_failure_time,
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerStats {
    pub state: CircuitBreakerState,
    pub failure_count: u32,
    pub success_count: u32,
    pub half_open_calls: u32,
    #[serde(skip)]
    pub last_failure_time: Option<Instant>,
}

/// Enhanced circuit breaker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnhancedCircuitBreakerStats {
    pub id: String,
    pub state: CircuitBreakerState,
    pub failure_count: u32,
    pub success_count: u32,
    pub half_open_calls: u32,
    #[serde(skip)]
    pub last_failure_time: Option<Instant>,
    pub failure_by_type: HashMap<FailureType, u32>,
    #[serde(skip)]
    pub metrics: CircuitBreakerMetrics,
    pub adaptive_threshold: f64,
    #[serde(skip)]
    pub recovery_timeout: Duration,
    #[serde(skip)]
    pub created_at: Instant,
    #[serde(skip)]
    pub uptime: Duration,
}

impl Default for EnhancedCircuitBreakerStats {
    fn default() -> Self {
        Self {
            id: String::new(),
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            half_open_calls: 0,
            last_failure_time: None,
            failure_by_type: HashMap::new(),
            metrics: CircuitBreakerMetrics::default(),
            adaptive_threshold: 0.0,
            recovery_timeout: Duration::ZERO,
            created_at: Instant::now(),
            uptime: Duration::ZERO,
        }
    }
}

/// Circuit breaker error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakerError {
    CircuitOpen {
        state: CircuitBreakerState,
        #[serde(skip)]
        last_failure: Option<Instant>,
    },
    OperationFailed(String),
    Timeout,
    ConfigurationError(String),
}

impl std::fmt::Display for CircuitBreakerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitBreakerError::CircuitOpen {
                state,
                last_failure,
            } => {
                write!(
                    f,
                    "Circuit breaker is {state:?}, last failure: {last_failure:?}"
                )
            }
            CircuitBreakerError::OperationFailed(msg) => {
                write!(f, "Operation failed: {msg}")
            }
            CircuitBreakerError::Timeout => {
                write!(f, "Operation timed out")
            }
            CircuitBreakerError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {msg}")
            }
        }
    }
}

impl std::error::Error for CircuitBreakerError {}

/// Shared circuit breaker for async usage
pub type SharedCircuitBreaker = Arc<RwLock<CircuitBreaker>>;

/// Create a new shared circuit breaker
pub fn new_shared_circuit_breaker(config: CircuitBreakerConfig) -> SharedCircuitBreaker {
    Arc::new(RwLock::new(CircuitBreaker::new(config)))
}

/// Create a new shared circuit breaker with custom ID
pub fn new_shared_circuit_breaker_with_id(
    config: CircuitBreakerConfig,
    id: String,
) -> SharedCircuitBreaker {
    Arc::new(RwLock::new(CircuitBreaker::with_id(config, id)))
}

/// Helper functions for working with shared circuit breakers
pub mod shared_helpers {
    use super::*;

    /// Execute an async operation with circuit breaker protection
    pub async fn execute_protected<F, Fut, R, E>(
        cb: &SharedCircuitBreaker,
        operation: F,
    ) -> Result<R, CircuitBreakerError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<R, E>>,
        E: std::fmt::Debug,
    {
        let mut circuit_breaker = cb.write().await;
        circuit_breaker.execute_async(operation).await
    }

    /// Get enhanced statistics
    pub async fn get_enhanced_stats(cb: &SharedCircuitBreaker) -> EnhancedCircuitBreakerStats {
        let circuit_breaker = cb.read().await;
        circuit_breaker.get_enhanced_stats()
    }
}

/// Extension trait for SharedCircuitBreaker to provide async interface
#[async_trait::async_trait]
pub trait SharedCircuitBreakerExt {
    async fn can_execute(&self) -> bool;
    async fn record_success_with_duration(&self, duration: Duration);
    async fn record_failure_with_type(&self, failure_type: FailureType);
    async fn is_healthy(&self) -> bool;
    async fn reset(&self);
    async fn get_enhanced_stats(&self) -> EnhancedCircuitBreakerStats;
}

#[async_trait::async_trait]
impl SharedCircuitBreakerExt for SharedCircuitBreaker {
    /// Check if the circuit breaker allows execution
    async fn can_execute(&self) -> bool {
        let mut cb = self.write().await;
        cb.can_execute()
    }

    /// Record a successful operation with execution duration
    async fn record_success_with_duration(&self, duration: Duration) {
        let mut cb = self.write().await;
        cb.record_success_with_duration(duration);
    }

    /// Record a failure with specific failure type
    async fn record_failure_with_type(&self, failure_type: FailureType) {
        let mut cb = self.write().await;
        cb.record_failure_with_type(failure_type);
    }

    /// Check if the circuit breaker is healthy
    async fn is_healthy(&self) -> bool {
        let cb = self.read().await;
        cb.is_healthy()
    }

    /// Reset the circuit breaker to closed state
    async fn reset(&self) {
        let mut cb = self.write().await;
        cb.reset();
    }

    /// Get enhanced statistics
    async fn get_enhanced_stats(&self) -> EnhancedCircuitBreakerStats {
        let cb = self.read().await;
        cb.get_enhanced_stats()
    }
}

/// Circuit breaker manager for handling multiple circuit breakers
pub struct CircuitBreakerManager {
    circuit_breakers: Arc<RwLock<HashMap<String, SharedCircuitBreaker>>>,
    default_config: CircuitBreakerConfig,
}

impl CircuitBreakerManager {
    pub fn new(default_config: CircuitBreakerConfig) -> Self {
        Self {
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            default_config,
        }
    }

    /// Get or create a circuit breaker
    pub async fn get_or_create(&self, name: String) -> SharedCircuitBreaker {
        let mut breakers = self.circuit_breakers.write().await;

        breakers
            .entry(name.clone())
            .or_insert_with(|| {
                new_shared_circuit_breaker_with_id(self.default_config.clone(), name)
            })
            .clone()
    }

    /// Get circuit breaker by name
    pub async fn get(&self, name: &str) -> Option<SharedCircuitBreaker> {
        let breakers = self.circuit_breakers.read().await;
        breakers.get(name).cloned()
    }

    /// Remove circuit breaker
    pub async fn remove(&self, name: &str) -> Option<SharedCircuitBreaker> {
        let mut breakers = self.circuit_breakers.write().await;
        breakers.remove(name)
    }

    /// Get all circuit breaker names
    pub async fn list_names(&self) -> Vec<String> {
        let breakers = self.circuit_breakers.read().await;
        breakers.keys().cloned().collect()
    }

    /// Get health summary of all circuit breakers
    pub async fn get_health_summary(&self) -> HashMap<String, bool> {
        let breakers = self.circuit_breakers.read().await;
        let mut summary = HashMap::new();

        for (name, cb) in breakers.iter() {
            summary.insert(name.clone(), cb.is_healthy().await);
        }

        summary
    }

    /// Reset all circuit breakers
    pub async fn reset_all(&self) {
        let breakers = self.circuit_breakers.read().await;

        for cb in breakers.values() {
            cb.reset().await;
        }
    }

    /// Get comprehensive statistics for all circuit breakers
    pub async fn get_all_stats(&self) -> HashMap<String, EnhancedCircuitBreakerStats> {
        let breakers = self.circuit_breakers.read().await;
        let mut stats = HashMap::new();

        for (name, cb) in breakers.iter() {
            stats.insert(name.clone(), cb.get_enhanced_stats().await);
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn test_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            enabled: true,
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
            half_open_max_calls: 2,
            adaptive_thresholds: false,
            adaptive_window_size: 50,
            max_failure_rate: 60.0,
            classify_failures: false,
            failure_type_timeouts: HashMap::new(),
            enable_metrics: false,
            sliding_window_size: 30,
            recovery_strategy: RecoveryStrategy::Linear,
        }
    }

    #[test]
    fn test_circuit_breaker_closed_state() {
        let mut cb = CircuitBreaker::new(test_config());

        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert!(cb.can_execute());

        // Record some successes
        cb.record_success();
        cb.record_success();

        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let mut cb = CircuitBreaker::new(test_config());

        // Record failures up to threshold
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);

        cb.record_failure(); // This should open the circuit
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        assert!(!cb.can_execute());
    }

    #[tokio::test]
    async fn test_circuit_breaker_transitions_to_half_open() {
        let mut cb = CircuitBreaker::new(test_config());

        // Open the circuit
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should transition to half-open
        assert!(cb.can_execute());
        assert_eq!(cb.state(), CircuitBreakerState::HalfOpen);
    }

    #[test]
    fn test_disabled_circuit_breaker() {
        let mut config = test_config();
        config.enabled = false;

        let mut cb = CircuitBreaker::new(config);

        // Should always allow execution when disabled
        assert!(cb.can_execute());

        cb.record_failure();
        cb.record_failure();
        cb.record_failure();

        assert!(cb.can_execute());
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }
}
