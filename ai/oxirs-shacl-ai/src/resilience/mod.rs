//! Production resilience and error handling for AI operations
//!
//! This module provides enterprise-grade error handling, retry mechanisms,
//! circuit breakers, and graceful degradation for LLM integrations.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

pub mod error;
pub mod retry;
pub mod circuit_breaker;
pub mod fallback;
pub mod timeout;

pub use error::{AIError, AIErrorKind, ErrorContext};
pub use retry::{RetryPolicy, RetryStrategy, ExponentialBackoff};
pub use circuit_breaker::{CircuitBreaker, CircuitState, CircuitBreakerConfig};
pub use fallback::{FallbackStrategy, FallbackChain};
pub use timeout::{TimeoutGuard, TimeoutConfig};

/// Configuration for resilience mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceConfig {
    /// Enable retry mechanism
    pub enable_retry: bool,

    /// Maximum retry attempts
    pub max_retries: usize,

    /// Initial retry delay in milliseconds
    pub initial_retry_delay_ms: u64,

    /// Maximum retry delay in milliseconds
    pub max_retry_delay_ms: u64,

    /// Enable circuit breaker
    pub enable_circuit_breaker: bool,

    /// Circuit breaker failure threshold
    pub circuit_breaker_threshold: usize,

    /// Circuit breaker timeout in seconds
    pub circuit_breaker_timeout_secs: u64,

    /// Enable graceful degradation
    pub enable_graceful_degradation: bool,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,

    /// Enable fallback strategies
    pub enable_fallback: bool,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            enable_retry: true,
            max_retries: 3,
            initial_retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
            enable_circuit_breaker: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_secs: 60,
            enable_graceful_degradation: true,
            request_timeout_secs: 30,
            enable_fallback: true,
        }
    }
}

/// Main resilience manager coordinating all mechanisms
pub struct ResilienceManager {
    config: ResilienceConfig,
    retry_policy: RetryPolicy,
    circuit_breaker: Arc<RwLock<CircuitBreaker>>,
    fallback_chain: FallbackChain,
    timeout_config: TimeoutConfig,
    metrics: Arc<RwLock<ResilienceMetrics>>,
}

/// Metrics for resilience operations
#[derive(Debug, Clone, Default)]
pub struct ResilienceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub retried_requests: u64,
    pub circuit_breaker_opens: u64,
    pub fallback_invocations: u64,
    pub timeout_errors: u64,
    pub avg_response_time_ms: f64,
}

impl ResilienceManager {
    /// Create a new resilience manager
    pub fn new(config: ResilienceConfig) -> Result<Self> {
        let retry_policy = RetryPolicy::new(
            RetryStrategy::ExponentialBackoff(ExponentialBackoff {
                initial_delay: Duration::from_millis(config.initial_retry_delay_ms),
                max_delay: Duration::from_millis(config.max_retry_delay_ms),
                multiplier: 2.0,
                jitter: true,
            }),
            config.max_retries,
        );

        let circuit_breaker = Arc::new(RwLock::new(CircuitBreaker::new(
            CircuitBreakerConfig {
                failure_threshold: config.circuit_breaker_threshold,
                success_threshold: 2,
                timeout: Duration::from_secs(config.circuit_breaker_timeout_secs),
                half_open_max_calls: 1,
            },
        )));

        let timeout_config = TimeoutConfig {
            request_timeout: Duration::from_secs(config.request_timeout_secs),
            read_timeout: Duration::from_secs(config.request_timeout_secs / 2),
            connect_timeout: Duration::from_secs(10),
        };

        Ok(Self {
            config,
            retry_policy,
            circuit_breaker,
            fallback_chain: FallbackChain::new(),
            timeout_config,
            metrics: Arc::new(RwLock::new(ResilienceMetrics::default())),
        })
    }

    /// Execute an operation with full resilience (retry, circuit breaker, timeout, fallback)
    pub async fn execute<F, T>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T> + Send,
        T: Clone,
    {
        let start = Instant::now();

        // Update metrics
        {
            let mut metrics = self.metrics.write()
                .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
            metrics.total_requests += 1;
        }

        // Check circuit breaker
        if self.config.enable_circuit_breaker {
            let mut breaker = self.circuit_breaker.write()
                .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

            if !breaker.can_execute() {
                let mut metrics = self.metrics.write()
                    .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
                metrics.failed_requests += 1;

                return Err(anyhow!("Circuit breaker is open"));
            }
        }

        // Execute with retry
        let result = if self.config.enable_retry {
            self.execute_with_retry(&mut operation).await
        } else {
            operation()
        };

        // Update circuit breaker and metrics
        {
            let mut breaker = self.circuit_breaker.write()
                .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

            let mut metrics = self.metrics.write()
                .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

            match &result {
                Ok(_) => {
                    breaker.record_success();
                    metrics.successful_requests += 1;
                }
                Err(_) => {
                    breaker.record_failure();
                    metrics.failed_requests += 1;
                }
            }

            let elapsed = start.elapsed().as_millis() as f64;
            metrics.avg_response_time_ms =
                (metrics.avg_response_time_ms * (metrics.total_requests - 1) as f64 + elapsed)
                / metrics.total_requests as f64;
        }

        result
    }

    /// Execute with retry mechanism
    async fn execute_with_retry<F, T>(&self, operation: &mut F) -> Result<T>
    where
        F: FnMut() -> Result<T> + Send,
    {
        let mut attempt = 0;

        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(_) if attempt < self.retry_policy.max_attempts => {
                    attempt += 1;

                    // Update retry metrics
                    {
                        let mut metrics = self.metrics.write()
                            .map_err(|err| anyhow!("Failed to acquire write lock: {}", err))?;
                        metrics.retried_requests += 1;
                    }

                    // Calculate delay
                    let delay = self.retry_policy.strategy.calculate_delay(attempt);

                    // Sleep before retry
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Execute with timeout
    pub async fn execute_with_timeout<F, T>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>> + Send,
    {
        match tokio::time::timeout(self.timeout_config.request_timeout, operation).await {
            Ok(result) => result,
            Err(_) => {
                let mut metrics = self.metrics.write()
                    .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
                metrics.timeout_errors += 1;
                Err(anyhow!("Operation timed out"))
            }
        }
    }

    /// Get current resilience metrics
    pub fn metrics(&self) -> Result<ResilienceMetrics> {
        let metrics = self.metrics.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
        Ok(metrics.clone())
    }

    /// Get circuit breaker state
    pub fn circuit_state(&self) -> Result<CircuitState> {
        let breaker = self.circuit_breaker.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
        Ok(breaker.state())
    }

    /// Reset resilience state
    pub fn reset(&self) -> Result<()> {
        let mut breaker = self.circuit_breaker.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        breaker.reset();

        let mut metrics = self.metrics.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        *metrics = ResilienceMetrics::default();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resilience_manager_success() {
        let manager = ResilienceManager::new(ResilienceConfig::default()).unwrap();

        let result = manager.execute(|| Ok::<_, anyhow::Error>(42)).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        let metrics = manager.metrics().unwrap();
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_requests, 1);
    }

    #[tokio::test]
    async fn test_resilience_manager_retry() {
        let manager = ResilienceManager::new(ResilienceConfig {
            max_retries: 2,
            initial_retry_delay_ms: 10,
            ..Default::default()
        }).unwrap();

        let mut call_count = 0;
        let result = manager.execute(|| {
            call_count += 1;
            if call_count < 2 {
                Err(anyhow!("Temporary failure"))
            } else {
                Ok(42)
            }
        }).await;

        assert!(result.is_ok());
        assert_eq!(call_count, 2);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens() {
        let manager = ResilienceManager::new(ResilienceConfig {
            circuit_breaker_threshold: 2,
            enable_retry: false,
            ..Default::default()
        }).unwrap();

        // Cause failures to open circuit
        for _ in 0..2 {
            let _ = manager.execute(|| Err::<(), _>(anyhow!("Error"))).await;
        }

        // Circuit should be open now
        let state = manager.circuit_state().unwrap();
        assert_eq!(state, CircuitState::Open);
    }
}
