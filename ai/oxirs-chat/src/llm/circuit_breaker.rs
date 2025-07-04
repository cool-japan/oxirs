//! Circuit Breaker Implementation for LLM Provider Resilience
//!
//! Provides circuit breaker functionality to prevent cascading failures
//! when LLM providers are experiencing issues.

use serde::{Deserialize, Serialize};
use std::{
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;
use tracing::{info, warn};

use super::config::{CircuitBreakerConfig, CircuitBreakerState};

/// Circuit breaker for LLM provider calls
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitBreakerState>>,
    config: CircuitBreakerConfig,
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    last_failure_time: AtomicU64,
    call_history: Arc<RwLock<Vec<CallResult>>>,
}

#[derive(Debug, Clone)]
struct CallResult {
    timestamp: Instant,
    success: bool,
    duration: Duration,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            config,
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            last_failure_time: AtomicU64::new(0),
            call_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Check if a call should be allowed through the circuit breaker
    pub async fn can_execute(&self) -> bool {
        let state = self.state.read().await;
        match *state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if timeout has passed
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let last_failure = self.last_failure_time.load(Ordering::Relaxed);

                if now - last_failure >= self.config.timeout_duration.as_secs() {
                    drop(state);
                    self.transition_to_half_open().await;
                    true
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Allow limited calls to test recovery
                self.success_count.load(Ordering::Relaxed) < self.config.recovery_threshold
            }
        }
    }

    /// Record the result of a call
    pub async fn record_result(&self, success: bool, duration: Duration) {
        let now = Instant::now();

        // Add to call history
        {
            let mut history = self.call_history.write().await;
            history.push(CallResult {
                timestamp: now,
                success,
                duration,
            });

            // Keep only recent calls
            let cutoff = now - Duration::from_secs(300); // 5 minutes
            history.retain(|call| call.timestamp > cutoff);

            // Maintain sliding window size
            if history.len() > self.config.sliding_window_size {
                history.remove(0);
            }
        }

        let current_state = self.state.read().await.clone();

        if success {
            self.success_count.fetch_add(1, Ordering::Relaxed);
            self.failure_count.store(0, Ordering::Relaxed);

            // Transition to closed if we're in half-open and have enough successes
            if current_state == CircuitBreakerState::HalfOpen
                && self.success_count.load(Ordering::Relaxed) >= self.config.recovery_threshold
            {
                drop(current_state);
                self.transition_to_closed().await;
            }
        } else {
            self.failure_count.fetch_add(1, Ordering::Relaxed);
            self.success_count.store(0, Ordering::Relaxed);
            self.last_failure_time.store(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                Ordering::Relaxed,
            );

            // Check if we should open the circuit
            if self.should_open_circuit().await {
                drop(current_state);
                self.transition_to_open().await;
            }
        }
    }

    async fn should_open_circuit(&self) -> bool {
        let failure_count = self.failure_count.load(Ordering::Relaxed);

        // Simple failure threshold check
        if failure_count >= self.config.failure_threshold {
            return true;
        }

        // Check slow call rate
        let history = self.call_history.read().await;
        if history.len() < self.config.sliding_window_size {
            return false; // Not enough data
        }

        let slow_calls = history
            .iter()
            .filter(|call| !call.success || call.duration > self.config.slow_call_threshold)
            .count();

        let slow_call_rate = slow_calls as f32 / history.len() as f32;
        slow_call_rate >= self.config.slow_call_rate_threshold
    }

    async fn transition_to_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::Open;
        warn!("Circuit breaker opened - failing fast for LLM calls");
    }

    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::HalfOpen;
        self.success_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker transitioned to half-open - testing recovery");
    }

    async fn transition_to_closed(&self) {
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker closed - normal operation resumed");
    }

    /// Get current circuit breaker statistics
    pub async fn get_stats(&self) -> CircuitBreakerStats {
        let state = self.state.read().await.clone();
        let history = self.call_history.read().await;

        let total_calls = history.len();
        let successful_calls = history.iter().filter(|call| call.success).count();
        let failed_calls = total_calls - successful_calls;

        let avg_response_time = if !history.is_empty() {
            history
                .iter()
                .map(|call| call.duration.as_millis())
                .sum::<u128>()
                / history.len() as u128
        } else {
            0
        };

        CircuitBreakerStats {
            state,
            total_calls,
            successful_calls,
            failed_calls,
            failure_rate: if total_calls > 0 {
                failed_calls as f32 / total_calls as f32
            } else {
                0.0
            },
            avg_response_time_ms: avg_response_time as u64,
            consecutive_failures: self.failure_count.load(Ordering::Relaxed),
        }
    }

    /// Reset the circuit breaker to closed state
    pub async fn reset(&self) -> Result<(), anyhow::Error> {
        self.transition_to_closed().await;
        
        // Clear call history
        {
            let mut history = self.call_history.write().await;
            history.clear();
        }
        
        info!("Circuit breaker has been manually reset");
        Ok(())
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerStats {
    pub state: CircuitBreakerState,
    pub total_calls: usize,
    pub successful_calls: usize,
    pub failed_calls: usize,
    pub failure_rate: f32,
    pub avg_response_time_ms: u64,
    pub consecutive_failures: usize,
}