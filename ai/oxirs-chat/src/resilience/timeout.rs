//! Timeout management for operations

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Overall request timeout
    pub request_timeout: Duration,

    /// Read timeout for streaming operations
    pub read_timeout: Duration,

    /// Connection timeout
    pub connect_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(15),
            connect_timeout: Duration::from_secs(10),
        }
    }
}

/// Timeout guard for operations
pub struct TimeoutGuard {
    config: TimeoutConfig,
    start_time: std::time::Instant,
}

impl TimeoutGuard {
    pub fn new(config: TimeoutConfig) -> Self {
        Self {
            config,
            start_time: std::time::Instant::now(),
        }
    }

    /// Check if operation has timed out
    pub fn is_expired(&self) -> bool {
        self.start_time.elapsed() >= self.config.request_timeout
    }

    /// Get remaining time
    pub fn remaining(&self) -> Option<Duration> {
        let elapsed = self.start_time.elapsed();
        if elapsed < self.config.request_timeout {
            Some(self.config.request_timeout - elapsed)
        } else {
            None
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Execute operation with timeout
pub async fn execute_with_timeout<F, T>(
    duration: Duration,
    operation: F,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    match timeout(duration, operation).await {
        Ok(result) => result,
        Err(_) => Err(anyhow!("Operation timed out after {:?}", duration)),
    }
}

/// Execute with timeout and custom error message
pub async fn execute_with_timeout_message<F, T>(
    duration: Duration,
    operation: F,
    operation_name: &str,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    match timeout(duration, operation).await {
        Ok(result) => result,
        Err(_) => Err(anyhow!(
            "{} timed out after {:?}",
            operation_name,
            duration
        )),
    }
}

/// Timeout wrapper for streaming operations
pub struct StreamTimeout {
    last_activity: std::sync::Arc<std::sync::RwLock<std::time::Instant>>,
    timeout: Duration,
}

impl StreamTimeout {
    pub fn new(timeout: Duration) -> Self {
        Self {
            last_activity: std::sync::Arc::new(std::sync::RwLock::new(std::time::Instant::now())),
            timeout,
        }
    }

    /// Update last activity timestamp
    pub fn update(&self) -> Result<()> {
        let mut last = self.last_activity.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        *last = std::time::Instant::now();
        Ok(())
    }

    /// Check if stream has timed out
    pub fn is_expired(&self) -> Result<bool> {
        let last = self.last_activity.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
        Ok(last.elapsed() >= self.timeout)
    }

    /// Get time since last activity
    pub fn idle_time(&self) -> Result<Duration> {
        let last = self.last_activity.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
        Ok(last.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_timeout_success() {
        let result = execute_with_timeout(
            Duration::from_millis(100),
            async { Ok::<_, anyhow::Error>(42) },
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_timeout_expires() {
        let result = execute_with_timeout(
            Duration::from_millis(10),
            async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok::<_, anyhow::Error>(42)
            },
        )
        .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn test_timeout_with_message() {
        let result = execute_with_timeout_message(
            Duration::from_millis(10),
            async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok::<_, anyhow::Error>(42)
            },
            "LLM API call",
        )
        .await;

        assert!(result.is_err());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("LLM API call"));
        assert!(error.contains("timed out"));
    }

    #[test]
    fn test_timeout_guard() {
        let guard = TimeoutGuard::new(TimeoutConfig {
            request_timeout: Duration::from_millis(100),
            ..Default::default()
        });

        assert!(!guard.is_expired());
        assert!(guard.remaining().is_some());

        std::thread::sleep(Duration::from_millis(150));

        assert!(guard.is_expired());
        assert!(guard.remaining().is_none());
    }

    #[test]
    fn test_timeout_guard_elapsed() {
        let guard = TimeoutGuard::new(TimeoutConfig::default());

        std::thread::sleep(Duration::from_millis(50));

        let elapsed = guard.elapsed();
        assert!(elapsed >= Duration::from_millis(50));
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_stream_timeout() {
        let stream_timeout = StreamTimeout::new(Duration::from_millis(100));

        assert!(!stream_timeout.is_expired().unwrap());

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));
        assert!(stream_timeout.is_expired().unwrap());

        // Update activity
        stream_timeout.update().unwrap();
        assert!(!stream_timeout.is_expired().unwrap());
    }

    #[test]
    fn test_stream_timeout_idle_time() {
        let stream_timeout = StreamTimeout::new(Duration::from_secs(60));

        std::thread::sleep(Duration::from_millis(50));

        let idle = stream_timeout.idle_time().unwrap();
        assert!(idle >= Duration::from_millis(50));
        assert!(idle < Duration::from_millis(100));
    }
}
