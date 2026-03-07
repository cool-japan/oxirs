//! Circuit breaker with half-open state and automatic reset logic.
//!
//! Implements the classic three-state circuit breaker pattern:
//!
//! ```text
//!  CLOSED ──(failure threshold exceeded)──► OPEN
//!    ▲                                         │
//!    │              (reset timeout elapsed)    │
//!    │◄──────────── HALF-OPEN ◄────────────────┘
//!    │                   │
//!    │  (probe succeeds) │ (probe fails)
//!    └───────────────────┘──► OPEN
//! ```
//!
//! # States
//!
//! - **Closed**: Normal operation.  Failures are counted.  If `failure_threshold`
//!   consecutive failures occur, the breaker trips to *Open*.
//! - **Open**: All requests are rejected immediately without being forwarded.
//!   After `reset_timeout` the breaker moves to *Half-Open* to try a probe.
//! - **Half-Open**: One probe request is allowed through.  On success, the
//!   breaker resets to *Closed*.  On failure, it returns to *Open*.
//!
//! # Thread safety
//!
//! `CircuitBreaker` uses internal `Mutex` synchronisation.  Wrap in `Arc` for
//! sharing across tasks.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ─── CircuitState ─────────────────────────────────────────────────────────────

/// The current state of a circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Normal operation: requests pass through.
    Closed,
    /// Tripped: all requests are rejected without forwarding.
    Open,
    /// Recovery probe: one test request is allowed through.
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitState::Closed => f.write_str("CLOSED"),
            CircuitState::Open => f.write_str("OPEN"),
            CircuitState::HalfOpen => f.write_str("HALF-OPEN"),
        }
    }
}

// ─── CircuitBreakerError ──────────────────────────────────────────────────────

/// Errors produced by the circuit breaker.
#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is OPEN: endpoint {endpoint_id} is unavailable")]
    Open { endpoint_id: String },

    #[error("Circuit breaker is HALF-OPEN and a probe is already in flight for {endpoint_id}")]
    ProbeInFlight { endpoint_id: String },
}

// ─── CircuitBreakerConfig ─────────────────────────────────────────────────────

/// Configuration for `CircuitBreaker`.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures required to trip the breaker.
    pub failure_threshold: u32,
    /// Duration the breaker stays Open before trying Half-Open.
    pub reset_timeout: Duration,
    /// Number of consecutive successes in Half-Open to return to Closed.
    pub success_threshold: u32,
    /// Optional human-readable name for the protected resource.
    pub name: String,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            reset_timeout: Duration::from_secs(30),
            success_threshold: 1,
            name: "unnamed".to_string(),
        }
    }
}

impl CircuitBreakerConfig {
    /// Create a configuration for the given endpoint.
    pub fn for_endpoint(endpoint_id: impl Into<String>) -> Self {
        Self {
            name: endpoint_id.into(),
            ..Default::default()
        }
    }
}

// ─── CircuitBreakerStats ──────────────────────────────────────────────────────

/// A snapshot of circuit breaker statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerStats {
    pub name: String,
    pub state: CircuitState,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub total_calls: u64,
    pub total_failures: u64,
    pub total_rejected: u64,
    pub trips: u64,
}

// ─── CircuitBreakerInner ──────────────────────────────────────────────────────

struct CircuitBreakerInner {
    state: CircuitState,
    consecutive_failures: u32,
    consecutive_successes: u32,
    last_failure: Option<Instant>,
    probe_in_flight: bool,
    total_calls: u64,
    total_failures: u64,
    total_rejected: u64,
    trips: u64,
    config: CircuitBreakerConfig,
}

impl CircuitBreakerInner {
    fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_failure: None,
            probe_in_flight: false,
            total_calls: 0,
            total_failures: 0,
            total_rejected: 0,
            trips: 0,
            config,
        }
    }

    /// Check whether the breaker allows a request to pass through.
    /// Returns `Ok(())` if allowed, or an error variant if blocked.
    fn check_and_acquire(&mut self) -> Result<(), CircuitBreakerError> {
        // Transition Open → Half-Open if reset timeout has elapsed.
        if self.state == CircuitState::Open {
            if let Some(last) = self.last_failure {
                if last.elapsed() >= self.config.reset_timeout {
                    self.state = CircuitState::HalfOpen;
                    self.probe_in_flight = false;
                    self.consecutive_successes = 0;
                } else {
                    self.total_rejected += 1;
                    return Err(CircuitBreakerError::Open {
                        endpoint_id: self.config.name.clone(),
                    });
                }
            } else {
                // No recorded failure time – should not normally happen.
                self.state = CircuitState::HalfOpen;
            }
        }

        match self.state {
            CircuitState::Closed => {
                self.total_calls += 1;
                Ok(())
            }
            CircuitState::HalfOpen => {
                if self.probe_in_flight {
                    self.total_rejected += 1;
                    Err(CircuitBreakerError::ProbeInFlight {
                        endpoint_id: self.config.name.clone(),
                    })
                } else {
                    self.probe_in_flight = true;
                    self.total_calls += 1;
                    Ok(())
                }
            }
            CircuitState::Open => {
                // Already handled above.
                self.total_rejected += 1;
                Err(CircuitBreakerError::Open {
                    endpoint_id: self.config.name.clone(),
                })
            }
        }
    }

    fn record_success(&mut self) {
        self.consecutive_failures = 0;
        match self.state {
            CircuitState::HalfOpen => {
                self.consecutive_successes += 1;
                self.probe_in_flight = false;
                if self.consecutive_successes >= self.config.success_threshold {
                    self.state = CircuitState::Closed;
                    self.consecutive_successes = 0;
                }
            }
            CircuitState::Closed => {
                // Nothing extra to do.
            }
            CircuitState::Open => {
                // Should not receive successes in Open state.
            }
        }
    }

    fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.total_failures += 1;
        self.last_failure = Some(Instant::now());
        self.probe_in_flight = false;

        match self.state {
            CircuitState::Closed => {
                if self.consecutive_failures >= self.config.failure_threshold {
                    self.state = CircuitState::Open;
                    self.trips += 1;
                }
            }
            CircuitState::HalfOpen => {
                // Probe failed – go back to Open.
                self.state = CircuitState::Open;
                self.trips += 1;
            }
            CircuitState::Open => {}
        }
    }

    fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            name: self.config.name.clone(),
            state: self.state,
            consecutive_failures: self.consecutive_failures,
            consecutive_successes: self.consecutive_successes,
            total_calls: self.total_calls,
            total_failures: self.total_failures,
            total_rejected: self.total_rejected,
            trips: self.trips,
        }
    }

    fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.consecutive_failures = 0;
        self.consecutive_successes = 0;
        self.last_failure = None;
        self.probe_in_flight = false;
    }
}

// ─── CircuitBreaker ───────────────────────────────────────────────────────────

/// Thread-safe circuit breaker protecting a single endpoint or resource.
#[derive(Clone)]
pub struct CircuitBreaker {
    inner: Arc<Mutex<CircuitBreakerInner>>,
}

impl CircuitBreaker {
    /// Create a new breaker with the default configuration.
    pub fn new(name: impl Into<String>) -> Self {
        let config = CircuitBreakerConfig {
            name: name.into(),
            ..Default::default()
        };
        Self {
            inner: Arc::new(Mutex::new(CircuitBreakerInner::new(config))),
        }
    }

    /// Create a new breaker with a custom configuration.
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CircuitBreakerInner::new(config))),
        }
    }

    /// Check whether the breaker allows a call through.
    ///
    /// Returns `Ok(())` when the call should proceed.
    /// Returns `Err(CircuitBreakerError::Open)` or `Err(CircuitBreakerError::ProbeInFlight)`
    /// when the call should be rejected.
    pub fn check(&self) -> Result<(), CircuitBreakerError> {
        self.inner
            .lock()
            .expect("circuit breaker lock poisoned")
            .check_and_acquire()
    }

    /// Record a successful call outcome.
    pub fn record_success(&self) {
        self.inner
            .lock()
            .expect("circuit breaker lock poisoned")
            .record_success();
    }

    /// Record a failed call outcome.
    pub fn record_failure(&self) {
        self.inner
            .lock()
            .expect("circuit breaker lock poisoned")
            .record_failure();
    }

    /// Execute a closure through the circuit breaker.
    ///
    /// The closure is called only when the breaker is Closed or HalfOpen.
    /// The return value of `f` is `Ok(T)` for success, `Err(E)` for failure.
    /// The breaker's success/failure counters are updated automatically.
    pub fn call<T, E, F>(&self, f: F) -> Result<T, CallError<E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        self.check().map_err(CallError::Rejected)?;
        match f() {
            Ok(v) => {
                self.record_success();
                Ok(v)
            }
            Err(e) => {
                self.record_failure();
                Err(CallError::Failed(e))
            }
        }
    }

    /// Return the current circuit state.
    pub fn state(&self) -> CircuitState {
        self.inner
            .lock()
            .expect("circuit breaker lock poisoned")
            .state
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> CircuitBreakerStats {
        self.inner
            .lock()
            .expect("circuit breaker lock poisoned")
            .stats()
    }

    /// Force-reset the breaker to Closed state (useful for testing / admin).
    pub fn reset(&self) {
        self.inner
            .lock()
            .expect("circuit breaker lock poisoned")
            .reset();
    }

    /// Return whether the breaker is currently allowing requests.
    pub fn is_closed(&self) -> bool {
        self.state() == CircuitState::Closed
    }

    /// Return whether the breaker is currently open (rejecting all requests).
    pub fn is_open(&self) -> bool {
        self.state() == CircuitState::Open
    }
}

impl fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        write!(
            f,
            "CircuitBreaker {{ name: {}, state: {} }}",
            stats.name, stats.state
        )
    }
}

// ─── CallError ────────────────────────────────────────────────────────────────

/// Error returned from `CircuitBreaker::call`.
#[derive(Debug, thiserror::Error)]
pub enum CallError<E> {
    /// The circuit breaker rejected the call without forwarding it.
    #[error("Request rejected by circuit breaker: {0}")]
    Rejected(#[source] CircuitBreakerError),

    /// The call was forwarded but the underlying operation failed.
    #[error("Call failed")]
    Failed(E),
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn fast_breaker() -> CircuitBreaker {
        CircuitBreaker::with_config(CircuitBreakerConfig {
            name: "test-ep".to_string(),
            failure_threshold: 3,
            reset_timeout: Duration::from_millis(50),
            success_threshold: 1,
        })
    }

    // ── State transitions ─────────────────────────────────────────────────

    #[test]
    fn test_initial_state_is_closed() {
        let cb = fast_breaker();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_closed());
        assert!(!cb.is_open());
    }

    #[test]
    fn test_trip_on_failure_threshold() {
        let cb = fast_breaker();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed); // not yet tripped
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure(); // 3rd failure – trip
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.is_open());
    }

    #[test]
    fn test_open_rejects_requests() {
        let cb = fast_breaker();
        for _ in 0..3 {
            cb.record_failure();
        }
        assert!(matches!(cb.check(), Err(CircuitBreakerError::Open { .. })));
    }

    #[test]
    fn test_open_to_half_open_after_timeout() {
        let cb = CircuitBreaker::with_config(CircuitBreakerConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_millis(10),
            success_threshold: 1,
            name: "ep".to_string(),
        });
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(20));
        // First check should trigger transition to HalfOpen
        let _ = cb.check();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_half_open_probe_success_closes() {
        let cb = CircuitBreaker::with_config(CircuitBreakerConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_millis(10),
            success_threshold: 1,
            name: "ep".to_string(),
        });
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(20));
        let _ = cb.check(); // transitions to HalfOpen
        cb.record_success(); // probe succeeds
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_half_open_probe_failure_reopens() {
        let cb = CircuitBreaker::with_config(CircuitBreakerConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_millis(10),
            success_threshold: 1,
            name: "ep".to_string(),
        });
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(20));
        let _ = cb.check(); // transitions to HalfOpen
        cb.record_failure(); // probe fails
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_half_open_second_probe_blocked() {
        let cb = CircuitBreaker::with_config(CircuitBreakerConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_millis(10),
            success_threshold: 1,
            name: "ep".to_string(),
        });
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(20));
        let first = cb.check();
        assert!(first.is_ok()); // probe acquired
                                // Second check while probe is in flight should be rejected
        let second = cb.check();
        assert!(matches!(
            second,
            Err(CircuitBreakerError::ProbeInFlight { .. })
        ));
    }

    #[test]
    fn test_success_resets_consecutive_failures() {
        let cb = fast_breaker();
        cb.record_failure();
        cb.record_failure();
        cb.record_success(); // reset
        cb.record_failure(); // starts fresh count
        assert_eq!(cb.state(), CircuitState::Closed); // only 1 failure after reset
    }

    // ── call() helper ─────────────────────────────────────────────────────

    #[test]
    fn test_call_success() {
        let cb = fast_breaker();
        let result: Result<i32, CallError<&str>> = cb.call(|| Ok(42));
        assert!(matches!(result, Ok(42)));
        assert_eq!(cb.stats().consecutive_failures, 0);
    }

    #[test]
    fn test_call_failure() {
        let cb = fast_breaker();
        let result: Result<i32, CallError<&str>> = cb.call(|| Err("oops"));
        assert!(matches!(result, Err(CallError::Failed("oops"))));
        assert_eq!(cb.stats().consecutive_failures, 1);
    }

    #[test]
    fn test_call_rejected_when_open() {
        let cb = fast_breaker();
        for _ in 0..3 {
            cb.record_failure();
        }
        let result: Result<i32, CallError<&str>> = cb.call(|| Ok(1));
        assert!(matches!(result, Err(CallError::Rejected(_))));
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_calls() {
        let cb = fast_breaker();
        cb.check().expect("should be allowed");
        cb.check().expect("should be allowed");
        assert_eq!(cb.stats().total_calls, 2);
    }

    #[test]
    fn test_stats_total_failures() {
        let cb = fast_breaker();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.stats().total_failures, 2);
    }

    #[test]
    fn test_stats_total_rejected() {
        let cb = fast_breaker();
        for _ in 0..3 {
            cb.record_failure();
        }
        let _ = cb.check(); // rejected
        let _ = cb.check(); // rejected
        assert_eq!(cb.stats().total_rejected, 2);
    }

    #[test]
    fn test_stats_trips() {
        let cb = fast_breaker();
        for _ in 0..3 {
            cb.record_failure();
        }
        assert_eq!(cb.stats().trips, 1);
    }

    // ── reset ─────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_returns_to_closed() {
        let cb = fast_breaker();
        for _ in 0..3 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.check().is_ok());
    }

    // ── Clone / Debug ─────────────────────────────────────────────────────

    #[test]
    fn test_debug_output() {
        let cb = fast_breaker();
        let s = format!("{:?}", cb);
        assert!(s.contains("CircuitBreaker"));
        assert!(s.contains("CLOSED"));
    }

    #[test]
    fn test_clone_shares_state() {
        let cb = fast_breaker();
        let cb2 = cb.clone();
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        // cb2 should see the same state since they share Arc
        assert_eq!(cb2.state(), CircuitState::Open);
    }

    #[test]
    fn test_state_display() {
        assert_eq!(format!("{}", CircuitState::Closed), "CLOSED");
        assert_eq!(format!("{}", CircuitState::Open), "OPEN");
        assert_eq!(format!("{}", CircuitState::HalfOpen), "HALF-OPEN");
    }
}
