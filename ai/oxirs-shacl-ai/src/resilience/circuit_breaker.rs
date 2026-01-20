//! Circuit breaker pattern for fault tolerance

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,

    /// Circuit is open, requests are rejected
    Open,

    /// Circuit is half-open, limited requests allowed to test recovery
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening circuit
    pub failure_threshold: usize,

    /// Number of consecutive successes to close circuit from half-open
    pub success_threshold: usize,

    /// Time to wait before transitioning from open to half-open
    pub timeout: Duration,

    /// Maximum number of calls allowed in half-open state
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
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    last_failure_time: AtomicU64,
    half_open_calls: AtomicUsize,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed,
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            last_failure_time: AtomicU64::new(0),
            half_open_calls: AtomicUsize::new(0),
        }
    }

    /// Check if a call can be executed
    pub fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,

            CircuitState::Open => {
                // Check if timeout has elapsed
                let last_failure = self.last_failure_time.load(Ordering::Relaxed);
                if last_failure == 0 {
                    return false;
                }

                let elapsed = Instant::now()
                    .duration_since(Instant::now() - Duration::from_secs(last_failure))
                    .as_secs();

                if elapsed >= self.config.timeout.as_secs() {
                    // Transition to half-open
                    self.state = CircuitState::HalfOpen;
                    self.half_open_calls.store(0, Ordering::Relaxed);
                    self.success_count.store(0, Ordering::Relaxed);
                    true
                } else {
                    false
                }
            }

            CircuitState::HalfOpen => {
                // Allow limited number of calls
                let calls = self.half_open_calls.fetch_add(1, Ordering::Relaxed);
                calls < self.config.half_open_max_calls
            }
        }
    }

    /// Record a successful call
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }

            CircuitState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.config.success_threshold {
                    // Close the circuit
                    self.state = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::Relaxed);
                    self.success_count.store(0, Ordering::Relaxed);
                }
            }

            CircuitState::Open => {
                // Should not happen, but reset anyway
                self.state = CircuitState::Closed;
                self.failure_count.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Record a failed call
    pub fn record_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if failures >= self.config.failure_threshold {
                    // Open the circuit
                    self.state = CircuitState::Open;
                    let now = Instant::now()
                        .duration_since(Instant::now() - Duration::from_secs(0))
                        .as_secs();
                    self.last_failure_time.store(now, Ordering::Relaxed);
                }
            }

            CircuitState::HalfOpen => {
                // Any failure in half-open state reopens the circuit
                self.state = CircuitState::Open;
                let now = Instant::now()
                    .duration_since(Instant::now() - Duration::from_secs(0))
                    .as_secs();
                self.last_failure_time.store(now, Ordering::Relaxed);
                self.success_count.store(0, Ordering::Relaxed);
            }

            CircuitState::Open => {
                // Update last failure time
                let now = Instant::now()
                    .duration_since(Instant::now() - Duration::from_secs(0))
                    .as_secs();
                self.last_failure_time.store(now, Ordering::Relaxed);
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
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.last_failure_time.store(0, Ordering::Relaxed);
        self.half_open_calls.store(0, Ordering::Relaxed);
    }

    /// Get current failure count
    pub fn failure_count(&self) -> usize {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// Get current success count (in half-open state)
    pub fn success_count(&self) -> usize {
        self.success_count.load(Ordering::Relaxed)
    }

    /// Get metrics for monitoring
    pub fn metrics(&self) -> CircuitBreakerMetrics {
        CircuitBreakerMetrics {
            state: self.state,
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            half_open_calls: self.half_open_calls.load(Ordering::Relaxed),
        }
    }
}

/// Circuit breaker metrics
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

        // Record failures
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

        // Record some failures
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.failure_count(), 2);

        // Success resets counter
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

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));

        // Check if transition to half-open
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

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait and transition to half-open
        std::thread::sleep(Duration::from_millis(150));
        assert!(breaker.can_execute());

        // Record successes
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

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();

        // Wait and transition to half-open
        std::thread::sleep(Duration::from_millis(150));
        assert!(breaker.can_execute());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Failure in half-open reopens circuit
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

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Reset
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
}
