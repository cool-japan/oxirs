//! # Circuit Breaker Pattern
//!
//! Implements the circuit breaker pattern for fault tolerance and preventing
//! cascading failures in distributed systems.
//!
//! ## Overview
//!
//! The circuit breaker acts as a proxy for operations that might fail. It monitors
//! failures and opens the circuit when failures exceed a threshold, preventing
//! further requests until the system recovers.
//!
//! ## States
//!
//! - **Closed**: Normal operation, requests pass through
//! - **Open**: Too many failures, requests are rejected immediately
//! - **Half-Open**: Testing if the system has recovered
//!
//! ## Features
//!
//! - Configurable failure thresholds
//! - Timeout-based recovery
//! - Metrics collection using SciRS2
//! - Adaptive thresholds based on historical data
//! - Per-node and per-operation circuit breakers

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::error::{ClusterError, Result};
use crate::raft::OxirsNodeId;

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed, requests pass through normally
    Closed,
    /// Circuit is open, requests are rejected immediately
    Open,
    /// Circuit is testing if the system has recovered
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open the circuit
    pub failure_threshold: u32,
    /// Success threshold to close the circuit from half-open
    pub success_threshold: u32,
    /// Timeout before attempting to close the circuit (milliseconds)
    pub timeout_ms: u64,
    /// Window size for tracking failures (seconds)
    pub window_size_secs: u64,
    /// Maximum number of half-open requests
    pub half_open_requests: u32,
    /// Enable adaptive thresholds based on historical data
    pub adaptive_thresholds: bool,
    /// Minimum failure rate (0.0-1.0) to open circuit
    pub min_failure_rate: f64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout_ms: 5000,
            window_size_secs: 60,
            half_open_requests: 3,
            adaptive_thresholds: true,
            min_failure_rate: 0.5,
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitBreakerStats {
    /// Total requests
    pub total_requests: u64,
    /// Total failures
    pub total_failures: u64,
    /// Total successes
    pub total_successes: u64,
    /// Requests rejected by open circuit
    pub rejected_requests: u64,
    /// Number of times circuit opened
    pub times_opened: u64,
    /// Number of times circuit closed
    pub times_closed: u64,
    /// Current failure rate (0.0-1.0)
    pub current_failure_rate: f64,
    /// Average response time (milliseconds)
    pub avg_response_time_ms: f64,
}

/// Circuit breaker internal state
#[derive(Debug, Clone)]
struct CircuitBreakerState {
    /// Current state
    state: CircuitState,
    /// Failure count in current window
    failure_count: u32,
    /// Success count in half-open state
    success_count: u32,
    /// When the circuit was opened
    opened_at: Option<Instant>,
    /// Recent request results (true = success, false = failure)
    recent_results: Vec<(Instant, bool)>,
    /// Statistics
    stats: CircuitBreakerStats,
    /// Half-open request count
    half_open_request_count: u32,
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            opened_at: None,
            recent_results: Vec::new(),
            stats: CircuitBreakerStats::default(),
            half_open_request_count: 0,
        }
    }
}

/// Circuit breaker for a single resource
#[derive(Clone)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitBreakerState>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState::default())),
        }
    }

    /// Check if the circuit allows the request
    pub async fn can_execute(&self) -> Result<()> {
        let mut state = self.state.write().await;

        match state.state {
            CircuitState::Closed => Ok(()),
            CircuitState::Open => {
                // Check if timeout has elapsed
                if let Some(opened_at) = state.opened_at {
                    let elapsed = opened_at.elapsed();
                    if elapsed.as_millis() >= self.config.timeout_ms as u128 {
                        // Transition to half-open
                        state.state = CircuitState::HalfOpen;
                        state.success_count = 0;
                        state.failure_count = 0;
                        state.half_open_request_count = 0;
                        info!("Circuit breaker transitioned to HalfOpen state");
                        Ok(())
                    } else {
                        state.stats.rejected_requests += 1;
                        Err(ClusterError::CircuitOpen)
                    }
                } else {
                    state.stats.rejected_requests += 1;
                    Err(ClusterError::CircuitOpen)
                }
            }
            CircuitState::HalfOpen => {
                if state.half_open_request_count < self.config.half_open_requests {
                    state.half_open_request_count += 1;
                    Ok(())
                } else {
                    state.stats.rejected_requests += 1;
                    Err(ClusterError::CircuitOpen)
                }
            }
        }
    }

    /// Record a successful execution
    pub async fn record_success(&self, response_time_ms: f64) {
        let mut state = self.state.write().await;

        state.stats.total_requests += 1;
        state.stats.total_successes += 1;

        // Update average response time
        let total_responses = state.stats.total_successes + state.stats.total_failures;
        if total_responses > 1 {
            state.stats.avg_response_time_ms = (state.stats.avg_response_time_ms
                * (total_responses - 1) as f64
                + response_time_ms)
                / total_responses as f64;
        } else {
            state.stats.avg_response_time_ms = response_time_ms;
        }

        // Add to recent results
        state.recent_results.push((Instant::now(), true));
        self.cleanup_old_results(&mut state);

        match state.state {
            CircuitState::Closed => {
                // Reset failure count on success
                state.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                state.success_count += 1;
                if state.success_count >= self.config.success_threshold {
                    // Close the circuit
                    state.state = CircuitState::Closed;
                    state.failure_count = 0;
                    state.success_count = 0;
                    state.stats.times_closed += 1;
                    info!("Circuit breaker closed after successful recovery");
                }
            }
            CircuitState::Open => {
                // Should not happen, but transition to half-open just in case
                state.state = CircuitState::HalfOpen;
                state.success_count = 1;
                state.failure_count = 0;
            }
        }

        self.update_failure_rate(&mut state);
    }

    /// Record a failed execution
    pub async fn record_failure(&self) {
        let mut state = self.state.write().await;

        state.stats.total_requests += 1;
        state.stats.total_failures += 1;

        // Add to recent results
        state.recent_results.push((Instant::now(), false));
        self.cleanup_old_results(&mut state);

        match state.state {
            CircuitState::Closed => {
                state.failure_count += 1;

                // Check if we should open the circuit
                if self.should_open_circuit(&state) {
                    state.state = CircuitState::Open;
                    state.opened_at = Some(Instant::now());
                    state.stats.times_opened += 1;
                    warn!(
                        "Circuit breaker opened after {} failures (rate: {:.2}%)",
                        state.failure_count,
                        state.stats.current_failure_rate * 100.0
                    );
                }
            }
            CircuitState::HalfOpen => {
                // Immediately open the circuit on failure in half-open state
                state.state = CircuitState::Open;
                state.opened_at = Some(Instant::now());
                state.failure_count = 1;
                state.success_count = 0;
                state.stats.times_opened += 1;
                warn!("Circuit breaker re-opened after failure in HalfOpen state");
            }
            CircuitState::Open => {
                // Already open, just increment failure count
                state.failure_count += 1;
            }
        }

        self.update_failure_rate(&mut state);
    }

    /// Get current circuit state
    pub async fn get_state(&self) -> CircuitState {
        self.state.read().await.state
    }

    /// Get circuit breaker statistics
    pub async fn get_stats(&self) -> CircuitBreakerStats {
        self.state.read().await.stats.clone()
    }

    /// Reset the circuit breaker
    pub async fn reset(&self) {
        let mut state = self.state.write().await;
        state.state = CircuitState::Closed;
        state.failure_count = 0;
        state.success_count = 0;
        state.opened_at = None;
        state.recent_results.clear();
        info!("Circuit breaker reset to Closed state");
    }

    /// Check if circuit should open based on failures
    fn should_open_circuit(&self, state: &CircuitBreakerState) -> bool {
        // Check failure count threshold
        if state.failure_count >= self.config.failure_threshold {
            return true;
        }

        // Check failure rate if adaptive thresholds are enabled
        if self.config.adaptive_thresholds {
            let recent_failures = state
                .recent_results
                .iter()
                .filter(|(_, success)| !success)
                .count();
            let total_recent = state.recent_results.len();

            if total_recent > 0 {
                let failure_rate = recent_failures as f64 / total_recent as f64;
                if failure_rate >= self.config.min_failure_rate {
                    return true;
                }
            }
        }

        false
    }

    /// Update failure rate statistics
    fn update_failure_rate(&self, state: &mut CircuitBreakerState) {
        if state.stats.total_requests > 0 {
            state.stats.current_failure_rate =
                state.stats.total_failures as f64 / state.stats.total_requests as f64;
        }
    }

    /// Clean up old results outside the window
    fn cleanup_old_results(&self, state: &mut CircuitBreakerState) {
        let window = Duration::from_secs(self.config.window_size_secs);
        let now = Instant::now();

        state
            .recent_results
            .retain(|(timestamp, _)| now.duration_since(*timestamp) < window);
    }
}

/// Circuit breaker manager for multiple resources
pub struct CircuitBreakerManager {
    /// Per-node circuit breakers
    node_breakers: Arc<RwLock<HashMap<OxirsNodeId, CircuitBreaker>>>,
    /// Per-operation circuit breakers
    operation_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    /// Default configuration
    default_config: CircuitBreakerConfig,
}

impl CircuitBreakerManager {
    /// Create a new circuit breaker manager
    pub fn new(default_config: CircuitBreakerConfig) -> Self {
        Self {
            node_breakers: Arc::new(RwLock::new(HashMap::new())),
            operation_breakers: Arc::new(RwLock::new(HashMap::new())),
            default_config,
        }
    }

    /// Get or create circuit breaker for a node
    pub async fn get_node_breaker(&self, node_id: OxirsNodeId) -> CircuitBreaker {
        let mut breakers = self.node_breakers.write().await;
        breakers
            .entry(node_id)
            .or_insert_with(|| CircuitBreaker::new(self.default_config.clone()))
            .clone()
    }

    /// Get or create circuit breaker for an operation
    pub async fn get_operation_breaker(&self, operation: &str) -> CircuitBreaker {
        let mut breakers = self.operation_breakers.write().await;
        breakers
            .entry(operation.to_string())
            .or_insert_with(|| CircuitBreaker::new(self.default_config.clone()))
            .clone()
    }

    /// Execute an operation with circuit breaker protection for a node
    pub async fn execute_with_node_breaker<F, T>(&self, node_id: OxirsNodeId, f: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let breaker = self.get_node_breaker(node_id).await;

        // Check if circuit allows execution
        breaker.can_execute().await?;

        let start = Instant::now();

        // Execute the operation
        match f.await {
            Ok(result) => {
                let elapsed = start.elapsed().as_millis() as f64;
                breaker.record_success(elapsed).await;
                Ok(result)
            }
            Err(e) => {
                breaker.record_failure().await;
                Err(e)
            }
        }
    }

    /// Execute an operation with circuit breaker protection for an operation type
    pub async fn execute_with_operation_breaker<F, T>(&self, operation: &str, f: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let breaker = self.get_operation_breaker(operation).await;

        // Check if circuit allows execution
        breaker.can_execute().await?;

        let start = Instant::now();

        // Execute the operation
        match f.await {
            Ok(result) => {
                let elapsed = start.elapsed().as_millis() as f64;
                breaker.record_success(elapsed).await;
                Ok(result)
            }
            Err(e) => {
                breaker.record_failure().await;
                Err(e)
            }
        }
    }

    /// Get all node breaker states
    pub async fn get_all_node_states(&self) -> HashMap<OxirsNodeId, CircuitState> {
        let breakers = self.node_breakers.read().await;
        let mut states = HashMap::new();

        for (node_id, breaker) in breakers.iter() {
            states.insert(*node_id, breaker.get_state().await);
        }

        states
    }

    /// Get all operation breaker states
    pub async fn get_all_operation_states(&self) -> HashMap<String, CircuitState> {
        let breakers = self.operation_breakers.read().await;
        let mut states = HashMap::new();

        for (operation, breaker) in breakers.iter() {
            states.insert(operation.clone(), breaker.get_state().await);
        }

        states
    }

    /// Get statistics for a specific node
    pub async fn get_node_stats(&self, node_id: OxirsNodeId) -> Option<CircuitBreakerStats> {
        let breakers = self.node_breakers.read().await;
        if let Some(breaker) = breakers.get(&node_id) {
            Some(breaker.get_stats().await)
        } else {
            None
        }
    }

    /// Get statistics for all nodes
    pub async fn get_all_node_stats(&self) -> HashMap<OxirsNodeId, CircuitBreakerStats> {
        let breakers = self.node_breakers.read().await;
        let mut stats = HashMap::new();

        for (node_id, breaker) in breakers.iter() {
            stats.insert(*node_id, breaker.get_stats().await);
        }

        stats
    }

    /// Reset all circuit breakers
    pub async fn reset_all(&self) {
        debug!("Resetting all circuit breakers");

        let node_breakers = self.node_breakers.read().await;
        for breaker in node_breakers.values() {
            breaker.reset().await;
        }

        let operation_breakers = self.operation_breakers.read().await;
        for breaker in operation_breakers.values() {
            breaker.reset().await;
        }

        info!("All circuit breakers reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_initial_state() {
        let config = CircuitBreakerConfig::default();
        let breaker = CircuitBreaker::new(config);

        assert_eq!(breaker.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let mut config = CircuitBreakerConfig::default();
        config.failure_threshold = 3;
        let breaker = CircuitBreaker::new(config);

        // Record failures
        for _ in 0..3 {
            breaker.record_failure().await;
        }

        assert_eq!(breaker.get_state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_transition() {
        let mut config = CircuitBreakerConfig::default();
        config.failure_threshold = 2;
        config.timeout_ms = 100;
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure().await;
        breaker.record_failure().await;
        assert_eq!(breaker.get_state().await, CircuitState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should transition to half-open
        let result = breaker.can_execute().await;
        assert!(result.is_ok());
        assert_eq!(breaker.get_state().await, CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_closes_on_success() {
        let mut config = CircuitBreakerConfig::default();
        config.failure_threshold = 2;
        config.success_threshold = 2;
        config.timeout_ms = 100;
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure().await;
        breaker.record_failure().await;
        assert_eq!(breaker.get_state().await, CircuitState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Transition to half-open
        breaker.can_execute().await.unwrap();

        // Record successes
        breaker.record_success(10.0).await;
        breaker.record_success(10.0).await;

        assert_eq!(breaker.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_stats() {
        let config = CircuitBreakerConfig::default();
        let breaker = CircuitBreaker::new(config);

        breaker.record_success(10.0).await;
        breaker.record_success(20.0).await;
        breaker.record_failure().await;

        let stats = breaker.get_stats().await;
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.total_successes, 2);
        assert_eq!(stats.total_failures, 1);
        assert_eq!(stats.current_failure_rate, 1.0 / 3.0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_manager() {
        let config = CircuitBreakerConfig::default();
        let manager = CircuitBreakerManager::new(config);

        // Test node breaker
        let breaker = manager.get_node_breaker(1).await;
        assert_eq!(breaker.get_state().await, CircuitState::Closed);

        // Test operation breaker
        let breaker = manager.get_operation_breaker("query").await;
        assert_eq!(breaker.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_execute_with_protection() {
        let config = CircuitBreakerConfig::default();
        let manager = CircuitBreakerManager::new(config);

        // Successful execution
        let result = manager
            .execute_with_node_breaker(1, async { Ok::<_, ClusterError>(42) })
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Failed execution
        let result = manager
            .execute_with_node_breaker(2, async {
                Err::<i32, _>(ClusterError::Network("test error".to_string()))
            })
            .await;
        assert!(result.is_err());
    }
}
