//! Retry mechanisms with exponential backoff

use anyhow::Result;
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Retry strategy
#[derive(Debug, Clone)]
pub enum RetryStrategy {
    /// Fixed delay between retries
    Fixed {
        delay: Duration,
    },

    /// Linear backoff (delay increases linearly)
    Linear {
        initial_delay: Duration,
        increment: Duration,
    },

    /// Exponential backoff (delay doubles each time)
    ExponentialBackoff(ExponentialBackoff),
}

/// Exponential backoff configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialBackoff {
    /// Initial delay before first retry
    pub initial_delay: Duration,

    /// Maximum delay between retries
    pub max_delay: Duration,

    /// Multiplier for each retry (typically 2.0)
    pub multiplier: f64,

    /// Add random jitter to prevent thundering herd
    pub jitter: bool,
}

impl RetryStrategy {
    /// Calculate delay for the given attempt number
    pub fn calculate_delay(&self, attempt: usize) -> Duration {
        match self {
            Self::Fixed { delay } => *delay,

            Self::Linear { initial_delay, increment } => {
                *initial_delay + *increment * (attempt as u32)
            }

            Self::ExponentialBackoff(backoff) => {
                let base_delay = backoff.initial_delay.as_millis() as f64
                    * backoff.multiplier.powi(attempt as i32 - 1);

                let delay = base_delay.min(backoff.max_delay.as_millis() as f64);

                let final_delay = if backoff.jitter {
                    let mut rng = Random::default();
                    let jitter_factor = 0.5 + rng.random::<f64>() * 0.5; // 0.5 to 1.0
                    delay * jitter_factor
                } else {
                    delay
                };

                Duration::from_millis(final_delay as u64)
            }
        }
    }
}

/// Retry policy combining strategy and limits
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub strategy: RetryStrategy,
    pub max_attempts: usize,
}

impl RetryPolicy {
    pub fn new(strategy: RetryStrategy, max_attempts: usize) -> Self {
        Self {
            strategy,
            max_attempts,
        }
    }

    /// Create exponential backoff policy with defaults
    pub fn exponential_default() -> Self {
        Self::new(
            RetryStrategy::ExponentialBackoff(ExponentialBackoff {
                initial_delay: Duration::from_millis(1000),
                max_delay: Duration::from_millis(30000),
                multiplier: 2.0,
                jitter: true,
            }),
            3,
        )
    }

    /// Create fixed delay policy
    pub fn fixed(delay: Duration, max_attempts: usize) -> Self {
        Self::new(RetryStrategy::Fixed { delay }, max_attempts)
    }

    /// Create linear backoff policy
    pub fn linear(initial: Duration, increment: Duration, max_attempts: usize) -> Self {
        Self::new(
            RetryStrategy::Linear {
                initial_delay: initial,
                increment,
            },
            max_attempts,
        )
    }
}

/// Execute an operation with retry logic
pub async fn execute_with_retry<F, T, E>(
    policy: &RetryPolicy,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    let mut attempt = 0;

    loop {
        match operation() {
            Ok(result) => return Ok(result),
            Err(e) if attempt < policy.max_attempts => {
                attempt += 1;
                let delay = policy.strategy.calculate_delay(attempt);
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Retry with custom should_retry predicate
pub async fn execute_with_retry_if<F, T, E, P>(
    policy: &RetryPolicy,
    mut operation: F,
    mut should_retry: P,
) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
    P: FnMut(&E) -> bool,
{
    let mut attempt = 0;

    loop {
        match operation() {
            Ok(result) => return Ok(result),
            Err(e) if attempt < policy.max_attempts && should_retry(&e) => {
                attempt += 1;
                let delay = policy.strategy.calculate_delay(attempt);
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_delay() {
        let strategy = RetryStrategy::Fixed {
            delay: Duration::from_millis(100),
        };

        assert_eq!(strategy.calculate_delay(1).as_millis(), 100);
        assert_eq!(strategy.calculate_delay(5).as_millis(), 100);
    }

    #[test]
    fn test_linear_backoff() {
        let strategy = RetryStrategy::Linear {
            initial_delay: Duration::from_millis(100),
            increment: Duration::from_millis(50),
        };

        assert_eq!(strategy.calculate_delay(1).as_millis(), 150); // 100 + 50*1
        assert_eq!(strategy.calculate_delay(2).as_millis(), 200); // 100 + 50*2
        assert_eq!(strategy.calculate_delay(3).as_millis(), 250); // 100 + 50*3
    }

    #[test]
    fn test_exponential_backoff_no_jitter() {
        let strategy = RetryStrategy::ExponentialBackoff(ExponentialBackoff {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(10000),
            multiplier: 2.0,
            jitter: false,
        });

        assert_eq!(strategy.calculate_delay(1).as_millis(), 100); // 100 * 2^0
        assert_eq!(strategy.calculate_delay(2).as_millis(), 200); // 100 * 2^1
        assert_eq!(strategy.calculate_delay(3).as_millis(), 400); // 100 * 2^2
        assert_eq!(strategy.calculate_delay(4).as_millis(), 800); // 100 * 2^3
    }

    #[test]
    fn test_exponential_backoff_max_delay() {
        let strategy = RetryStrategy::ExponentialBackoff(ExponentialBackoff {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(500),
            multiplier: 2.0,
            jitter: false,
        });

        assert_eq!(strategy.calculate_delay(1).as_millis(), 100);
        assert_eq!(strategy.calculate_delay(2).as_millis(), 200);
        assert_eq!(strategy.calculate_delay(3).as_millis(), 400);
        assert_eq!(strategy.calculate_delay(4).as_millis(), 500); // Capped at max
        assert_eq!(strategy.calculate_delay(5).as_millis(), 500); // Capped at max
    }

    #[test]
    fn test_exponential_backoff_with_jitter() {
        let strategy = RetryStrategy::ExponentialBackoff(ExponentialBackoff {
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_millis(10000),
            multiplier: 2.0,
            jitter: true,
        });

        // With jitter, delay should be between 0.5x and 1.0x the base delay
        let delay1 = strategy.calculate_delay(1).as_millis();
        assert!(delay1 >= 500 && delay1 <= 1000);

        let delay2 = strategy.calculate_delay(2).as_millis();
        assert!(delay2 >= 1000 && delay2 <= 2000);
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let policy = RetryPolicy::fixed(Duration::from_millis(10), 3);

        let mut call_count = 0;
        let result = execute_with_retry(&policy, || {
            call_count += 1;
            if call_count < 3 {
                Err("Temporary failure")
            } else {
                Ok(42)
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count, 3);
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let policy = RetryPolicy::fixed(Duration::from_millis(10), 2);

        let mut call_count = 0;
        let result = execute_with_retry(&policy, || {
            call_count += 1;
            Err::<i32, _>("Always fails")
        })
        .await;

        assert!(result.is_err());
        assert_eq!(call_count, 3); // Initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_with_predicate() {
        let policy = RetryPolicy::fixed(Duration::from_millis(10), 5);

        let mut call_count = 0;
        let result: Result<i32, &str> = execute_with_retry_if(
            &policy,
            || {
                call_count += 1;
                if call_count < 3 {
                    Err("retryable")
                } else {
                    Err("non-retryable")
                }
            },
            |e| *e == "retryable",
        )
        .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "non-retryable");
        assert_eq!(call_count, 3); // Stopped when non-retryable error occurred
    }
}
