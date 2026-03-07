//! Circuit breaker pattern for fault tolerance
//!
//! Implements the circuit breaker pattern with three states:
//! - Closed: requests flow normally
//! - Open: requests are rejected immediately
//! - Half-Open: limited requests allowed to test recovery

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected immediately
    Open,
    /// Circuit is half-open, limited requests allowed to probe recovery
    HalfOpen,
}

/// Configuration for the circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit
    pub failure_threshold: usize,
    /// Number of consecutive successes needed to close from half-open
    pub success_threshold: usize,
    /// How long to wait before transitioning from Open to HalfOpen
    pub timeout: Duration,
    /// Maximum concurrent calls allowed in half-open state
    pub half_open_max_calls: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_calls: 1,
        }
    }
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitState,
    failure_count: usize,
    success_count: usize,
    half_open_calls: usize,
    /// When the circuit was last opened (for timeout tracking)
    opened_at: Option<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            half_open_calls: 0,
            opened_at: None,
        }
    }

    /// Check whether a request can currently be executed.
    ///
    /// For `Open` circuits, this also checks whether the timeout has elapsed
    /// and transitions to `HalfOpen` if so.
    pub fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,

            CircuitState::Open => {
                if let Some(opened_at) = self.opened_at {
                    if opened_at.elapsed() >= self.config.timeout {
                        // Transition to half-open
                        self.state = CircuitState::HalfOpen;
                        self.half_open_calls = 0;
                        self.success_count = 0;
                        return true;
                    }
                }
                false
            }

            CircuitState::HalfOpen => {
                if self.half_open_calls < self.config.half_open_max_calls {
                    self.half_open_calls += 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Record a successful call
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                // Reset failure count on any success while closed
                self.failure_count = 0;
            }

            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.config.success_threshold {
                    // Enough successes — close the circuit
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                    self.opened_at = None;
                }
            }

            CircuitState::Open => {
                // Shouldn't normally happen, but reset defensively
                self.state = CircuitState::Closed;
                self.failure_count = 0;
                self.opened_at = None;
            }
        }
    }

    /// Record a failed call
    pub fn record_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.config.failure_threshold {
                    self.state = CircuitState::Open;
                    self.opened_at = Some(Instant::now());
                }
            }

            CircuitState::HalfOpen => {
                // Any failure in half-open reopens the circuit
                self.state = CircuitState::Open;
                self.opened_at = Some(Instant::now());
                self.success_count = 0;
                self.half_open_calls = 0;
            }

            CircuitState::Open => {
                // Update opened_at to restart the timeout
                self.opened_at = Some(Instant::now());
            }
        }
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Reset circuit breaker to closed state
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.half_open_calls = 0;
        self.opened_at = None;
    }

    /// Get current failure count
    pub fn failure_count(&self) -> usize {
        self.failure_count
    }

    /// Get current success count (meaningful in half-open state)
    pub fn success_count(&self) -> usize {
        self.success_count
    }

    /// Get metrics for monitoring
    pub fn metrics(&self) -> CircuitBreakerMetrics {
        CircuitBreakerMetrics {
            state: self.state,
            failure_count: self.failure_count,
            success_count: self.success_count,
            half_open_calls: self.half_open_calls,
        }
    }
}

/// Snapshot of circuit breaker metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerMetrics {
    pub state: CircuitState,
    pub failure_count: usize,
    pub success_count: usize,
    pub half_open_calls: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let mut breaker = CircuitBreaker::new(config);

        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let mut breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.failure_count(), 2);

        breaker.record_success();
        assert_eq!(breaker.failure_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_transition() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
            half_open_max_calls: 1,
        };
        let mut breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        std::thread::sleep(Duration::from_millis(150));

        assert!(breaker.can_execute());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_closes_after_successes() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
            half_open_max_calls: 3,
        };
        let mut breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        std::thread::sleep(Duration::from_millis(150));
        assert!(breaker.can_execute());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_reopens_on_half_open_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_millis(100),
            half_open_max_calls: 1,
            ..Default::default()
        };
        let mut breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        breaker.record_failure();

        std::thread::sleep(Duration::from_millis(150));
        assert!(breaker.can_execute());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let mut breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        breaker.reset();
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert_eq!(breaker.failure_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_metrics() {
        let mut breaker = CircuitBreaker::new(CircuitBreakerConfig::default());

        breaker.record_failure();
        breaker.record_failure();

        let metrics = breaker.metrics();
        assert_eq!(metrics.failure_count, 2);
        assert_eq!(metrics.state, CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_rejects_when_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout: Duration::from_secs(60),
            ..Default::default()
        };
        let mut breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.can_execute());
    }
}
