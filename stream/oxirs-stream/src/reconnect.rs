//! Automatic reconnection logic with exponential backoff
//!
//! Provides resilient connection management with configurable retry strategies,
//! connection failure callbacks, and comprehensive error handling.

use crate::connection_pool::{ConnectionFactory, PooledConnection};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tokio::time::sleep;
use tracing::{error, info, warn};

/// Reconnection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconnectConfig {
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Exponential backoff multiplier
    pub multiplier: f64,
    /// Maximum retry attempts (0 for unlimited)
    pub max_attempts: u32,
    /// Jitter factor (0.0 to 1.0)
    pub jitter_factor: f64,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Enable connection failure callbacks
    pub enable_callbacks: bool,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            max_attempts: 10,
            jitter_factor: 0.1,
            connection_timeout: Duration::from_secs(30),
            enable_callbacks: true,
        }
    }
}

/// Reconnection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReconnectStrategy {
    /// Exponential backoff with jitter
    ExponentialBackoff,
    /// Fixed delay between attempts
    FixedDelay(Duration),
    /// Linear backoff
    LinearBackoff(Duration),
    /// Custom strategy with callback
    Custom,
}

/// Reconnection event
#[derive(Debug, Clone)]
pub enum ReconnectEvent {
    /// Reconnection attempt started
    AttemptStarted {
        connection_id: String,
        attempt: u32,
        delay: Duration,
    },
    /// Reconnection attempt succeeded
    AttemptSucceeded {
        connection_id: String,
        attempt: u32,
        total_time: Duration,
    },
    /// Reconnection attempt failed
    AttemptFailed {
        connection_id: String,
        attempt: u32,
        error: String,
        next_delay: Option<Duration>,
    },
    /// All reconnection attempts exhausted
    ReconnectionExhausted {
        connection_id: String,
        total_attempts: u32,
        total_time: Duration,
    },
}

/// Reconnection statistics
#[derive(Debug, Clone, Default)]
pub struct ReconnectStatistics {
    pub total_attempts: u64,
    pub successful_reconnects: u64,
    pub failed_reconnects: u64,
    pub current_streak: u32,
    pub longest_streak: u32,
    pub total_reconnect_time: Duration,
    pub avg_reconnect_time: Duration,
    pub last_reconnect_attempt: Option<Instant>,
    pub last_successful_reconnect: Option<Instant>,
}

/// Callback for connection failures
pub type ConnectionFailureCallback =
    Arc<dyn Fn(String, String, u32) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Automatic reconnection manager
pub struct ReconnectManager<T: PooledConnection> {
    config: ReconnectConfig,
    strategy: ReconnectStrategy,
    statistics: Arc<RwLock<ReconnectStatistics>>,
    event_sender: broadcast::Sender<ReconnectEvent>,
    failure_callbacks: Arc<RwLock<Vec<ConnectionFailureCallback>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PooledConnection + Clone> ReconnectManager<T> {
    /// Create a new reconnection manager
    pub fn new(config: ReconnectConfig, strategy: ReconnectStrategy) -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            config,
            strategy,
            statistics: Arc::new(RwLock::new(ReconnectStatistics::default())),
            event_sender,
            failure_callbacks: Arc::new(RwLock::new(Vec::new())),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Attempt to reconnect with configured strategy
    pub async fn reconnect(
        &self,
        connection_id: String,
        factory: Arc<dyn ConnectionFactory<T>>,
    ) -> Result<T> {
        let start_time = Instant::now();
        let mut attempt = 0;
        let mut current_delay = self.config.initial_delay;

        loop {
            attempt += 1;

            // Check max attempts
            if self.config.max_attempts > 0 && attempt > self.config.max_attempts {
                let total_time = start_time.elapsed();

                let _ = self
                    .event_sender
                    .send(ReconnectEvent::ReconnectionExhausted {
                        connection_id: connection_id.clone(),
                        total_attempts: attempt - 1,
                        total_time,
                    });

                // Update statistics
                let mut stats = self.statistics.write().await;
                stats.failed_reconnects += 1;
                stats.current_streak = 0;

                // Call failure callbacks
                if self.config.enable_callbacks {
                    self.invoke_failure_callbacks(
                        connection_id.clone(),
                        "Maximum retry attempts exhausted".to_string(),
                        attempt - 1,
                    )
                    .await;
                }

                return Err(anyhow!(
                    "Failed to reconnect after {} attempts",
                    self.config.max_attempts
                ));
            }

            // Calculate delay with jitter
            let jittered_delay = self.apply_jitter(current_delay);

            if attempt > 1 {
                info!(
                    "Reconnection attempt {} for {} in {:?}",
                    attempt, connection_id, jittered_delay
                );

                let _ = self.event_sender.send(ReconnectEvent::AttemptStarted {
                    connection_id: connection_id.clone(),
                    attempt,
                    delay: jittered_delay,
                });

                sleep(jittered_delay).await;
            }

            // Update statistics
            {
                let mut stats = self.statistics.write().await;
                stats.total_attempts += 1;
                stats.last_reconnect_attempt = Some(Instant::now());
            }

            // Attempt connection with timeout
            match tokio::time::timeout(self.config.connection_timeout, factory.create_connection())
                .await
            {
                Ok(Ok(connection)) => {
                    let total_time = start_time.elapsed();

                    info!(
                        "Successfully reconnected {} after {} attempts in {:?}",
                        connection_id, attempt, total_time
                    );

                    let _ = self.event_sender.send(ReconnectEvent::AttemptSucceeded {
                        connection_id: connection_id.clone(),
                        attempt,
                        total_time,
                    });

                    // Update statistics
                    let mut stats = self.statistics.write().await;
                    stats.successful_reconnects += 1;
                    stats.current_streak += 1;
                    stats.longest_streak = stats.longest_streak.max(stats.current_streak);
                    stats.total_reconnect_time += total_time;
                    stats.last_successful_reconnect = Some(Instant::now());

                    if stats.successful_reconnects > 0 {
                        stats.avg_reconnect_time =
                            stats.total_reconnect_time / stats.successful_reconnects as u32;
                    }

                    return Ok(connection);
                }
                Ok(Err(e)) => {
                    warn!(
                        "Reconnection attempt {} for {} failed: {}",
                        attempt, connection_id, e
                    );

                    // Calculate next delay
                    current_delay = self.calculate_next_delay(current_delay, attempt);
                    let next_delay = if attempt < self.config.max_attempts {
                        Some(current_delay)
                    } else {
                        None
                    };

                    let _ = self.event_sender.send(ReconnectEvent::AttemptFailed {
                        connection_id: connection_id.clone(),
                        attempt,
                        error: e.to_string(),
                        next_delay,
                    });

                    // Call failure callbacks for each attempt if enabled
                    if self.config.enable_callbacks && attempt % 3 == 0 {
                        self.invoke_failure_callbacks(
                            connection_id.clone(),
                            e.to_string(),
                            attempt,
                        )
                        .await;
                    }
                }
                Err(_) => {
                    error!(
                        "Reconnection attempt {} for {} timed out",
                        attempt, connection_id
                    );

                    current_delay = self.calculate_next_delay(current_delay, attempt);

                    let _ = self.event_sender.send(ReconnectEvent::AttemptFailed {
                        connection_id: connection_id.clone(),
                        attempt,
                        error: "Connection timeout".to_string(),
                        next_delay: Some(current_delay),
                    });
                }
            }
        }
    }

    /// Calculate next delay based on strategy
    fn calculate_next_delay(&self, current_delay: Duration, attempt: u32) -> Duration {
        match &self.strategy {
            ReconnectStrategy::ExponentialBackoff => {
                let next_delay = current_delay.mul_f64(self.config.multiplier);
                next_delay.min(self.config.max_delay)
            }
            ReconnectStrategy::FixedDelay(delay) => *delay,
            ReconnectStrategy::LinearBackoff(increment) => {
                let next_delay = self.config.initial_delay + (*increment * attempt);
                next_delay.min(self.config.max_delay)
            }
            ReconnectStrategy::Custom => {
                // For custom strategy, use exponential backoff as fallback
                let next_delay = current_delay.mul_f64(self.config.multiplier);
                next_delay.min(self.config.max_delay)
            }
        }
    }

    /// Apply jitter to delay
    fn apply_jitter(&self, delay: Duration) -> Duration {
        if self.config.jitter_factor <= 0.0 {
            return delay;
        }

        let jitter_range = delay.as_millis() as f64 * self.config.jitter_factor;
        let jitter = (fastrand::f64() - 0.5) * 2.0 * jitter_range;
        let jittered_millis = (delay.as_millis() as f64 + jitter).max(0.0) as u64;

        Duration::from_millis(jittered_millis)
    }

    /// Register a connection failure callback
    pub async fn register_failure_callback<F>(&self, callback: F)
    where
        F: Fn(String, String, u32) -> Pin<Box<dyn Future<Output = ()> + Send>>
            + Send
            + Sync
            + 'static,
    {
        let mut callbacks = self.failure_callbacks.write().await;
        callbacks.push(Arc::new(callback));
    }

    /// Invoke all registered failure callbacks
    async fn invoke_failure_callbacks(&self, connection_id: String, error: String, attempt: u32) {
        let callbacks = self.failure_callbacks.read().await;

        for callback in callbacks.iter() {
            let fut = callback(connection_id.clone(), error.clone(), attempt);
            tokio::spawn(async move {
                fut.await;
            });
        }
    }

    /// Get reconnection statistics
    pub async fn get_statistics(&self) -> ReconnectStatistics {
        self.statistics.read().await.clone()
    }

    /// Reset reconnection statistics
    pub async fn reset_statistics(&self) {
        *self.statistics.write().await = ReconnectStatistics::default();
    }

    /// Subscribe to reconnection events
    pub fn subscribe(&self) -> broadcast::Receiver<ReconnectEvent> {
        self.event_sender.subscribe()
    }
}

/// Helper for creating reconnection-aware connections
pub struct ResilientConnection<T: PooledConnection> {
    connection: Option<T>,
    connection_id: String,
    factory: Arc<dyn ConnectionFactory<T>>,
    reconnect_manager: Arc<ReconnectManager<T>>,
    last_error: Option<String>,
}

impl<T: PooledConnection + Clone> ResilientConnection<T> {
    /// Create a new resilient connection
    pub async fn new(
        connection_id: String,
        factory: Arc<dyn ConnectionFactory<T>>,
        reconnect_manager: Arc<ReconnectManager<T>>,
    ) -> Result<Self> {
        let connection = factory.create_connection().await?;

        Ok(Self {
            connection: Some(connection),
            connection_id,
            factory,
            reconnect_manager,
            last_error: None,
        })
    }

    /// Get the underlying connection, reconnecting if necessary
    pub async fn get_connection(&mut self) -> Result<&mut T> {
        // Check if we have a healthy connection
        let needs_reconnection = match self.connection {
            Some(ref mut conn) => !conn.is_healthy().await,
            None => true,
        };

        if !needs_reconnection {
            // Return the healthy connection
            return self
                .connection
                .as_mut()
                .ok_or_else(|| anyhow!("Connection unexpectedly None"));
        }

        // Connection is unhealthy or missing, attempt reconnection
        info!(
            "Connection {} is unhealthy, attempting reconnection",
            self.connection_id
        );

        match self
            .reconnect_manager
            .reconnect(self.connection_id.clone(), self.factory.clone())
            .await
        {
            Ok(new_conn) => {
                self.connection = Some(new_conn);
                self.last_error = None;
                self.connection
                    .as_mut()
                    .ok_or_else(|| anyhow!("Connection unexpectedly None"))
            }
            Err(e) => {
                self.last_error = Some(e.to_string());
                Err(e)
            }
        }
    }

    /// Check if connection is currently healthy
    pub async fn is_healthy(&self) -> bool {
        if let Some(ref conn) = self.connection {
            conn.is_healthy().await
        } else {
            false
        }
    }

    /// Get the last error if any
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Manually trigger reconnection
    pub async fn reconnect(&mut self) -> Result<()> {
        let new_conn = self
            .reconnect_manager
            .reconnect(self.connection_id.clone(), self.factory.clone())
            .await?;

        // Close old connection if exists
        if let Some(mut old_conn) = self.connection.take() {
            let _ = old_conn.close().await;
        }

        self.connection = Some(new_conn);
        self.last_error = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    #[derive(Clone)]
    struct TestConnection {
        id: u32,
        healthy: Arc<AtomicBool>,
        created_at: Instant,
    }

    #[async_trait::async_trait]
    impl PooledConnection for TestConnection {
        async fn is_healthy(&self) -> bool {
            self.healthy.load(Ordering::Relaxed)
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }

        fn clone_connection(&self) -> Box<dyn PooledConnection> {
            Box::new(TestConnection {
                id: self.id,
                healthy: Arc::new(AtomicBool::new(self.healthy.load(Ordering::Relaxed))),
                created_at: self.created_at,
            })
        }

        fn created_at(&self) -> Instant {
            self.created_at
        }

        fn last_activity(&self) -> Instant {
            Instant::now()
        }

        fn update_activity(&mut self) {}
    }

    struct TestConnectionFactory {
        counter: Arc<AtomicU32>,
        should_fail: Arc<AtomicBool>,
        fail_count: Arc<AtomicU32>,
    }

    #[async_trait::async_trait]
    impl ConnectionFactory<TestConnection> for TestConnectionFactory {
        async fn create_connection(&self) -> Result<TestConnection> {
            let current_fails = self.fail_count.load(Ordering::Relaxed);

            if self.should_fail.load(Ordering::Relaxed) && current_fails > 0 {
                self.fail_count.fetch_sub(1, Ordering::Relaxed);
                return Err(anyhow!("Simulated connection failure"));
            }

            let id = self.counter.fetch_add(1, Ordering::Relaxed);
            Ok(TestConnection {
                id,
                healthy: Arc::new(AtomicBool::new(true)),
                created_at: Instant::now(),
            })
        }
    }

    #[tokio::test]
    async fn test_exponential_backoff() {
        let config = ReconnectConfig {
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            multiplier: 2.0,
            max_attempts: 5,
            jitter_factor: 0.0,
            ..Default::default()
        };

        let manager =
            ReconnectManager::<TestConnection>::new(config, ReconnectStrategy::ExponentialBackoff);

        let factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(0)),
            should_fail: Arc::new(AtomicBool::new(true)),
            fail_count: Arc::new(AtomicU32::new(3)), // Fail first 3 attempts
        });

        let start = Instant::now();
        let result = manager.reconnect("test-conn".to_string(), factory).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());

        // Should have delays: 0ms, 10ms, 20ms, 40ms (total ~70ms)
        // Allow for more timing variance during parallel test execution
        assert!(elapsed >= Duration::from_millis(50));
        assert!(elapsed < Duration::from_millis(300));

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_attempts, 4);
        assert_eq!(stats.successful_reconnects, 1);
    }

    #[tokio::test]
    async fn test_max_attempts() {
        let config = ReconnectConfig {
            initial_delay: Duration::from_millis(1),
            max_attempts: 3,
            ..Default::default()
        };

        let manager =
            ReconnectManager::<TestConnection>::new(config, ReconnectStrategy::ExponentialBackoff);

        let factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(0)),
            should_fail: Arc::new(AtomicBool::new(true)),
            fail_count: Arc::new(AtomicU32::new(100)), // Always fail
        });

        let result = manager.reconnect("test-conn".to_string(), factory).await;
        assert!(result.is_err());

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_attempts, 3);
        assert_eq!(stats.failed_reconnects, 1);
    }

    #[tokio::test]
    async fn test_failure_callbacks() {
        let config = ReconnectConfig {
            initial_delay: Duration::from_millis(1),
            max_attempts: 3,
            enable_callbacks: true,
            ..Default::default()
        };

        let manager = ReconnectManager::<TestConnection>::new(
            config,
            ReconnectStrategy::FixedDelay(Duration::from_millis(1)),
        );

        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();

        manager
            .register_failure_callback(move |_id, _error, _attempt| {
                let called = callback_called_clone.clone();
                Box::pin(async move {
                    called.store(true, Ordering::Relaxed);
                })
            })
            .await;

        let factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(0)),
            should_fail: Arc::new(AtomicBool::new(true)),
            fail_count: Arc::new(AtomicU32::new(100)),
        });

        let _ = manager.reconnect("test-conn".to_string(), factory).await;

        // Give callback time to execute
        tokio::time::sleep(Duration::from_millis(10)).await;

        assert!(callback_called.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_resilient_connection() {
        let config = ReconnectConfig::default();
        let manager = Arc::new(ReconnectManager::<TestConnection>::new(
            config,
            ReconnectStrategy::ExponentialBackoff,
        ));

        let _healthy_flag = Arc::new(AtomicBool::new(true));
        let factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(0)),
            should_fail: Arc::new(AtomicBool::new(false)),
            fail_count: Arc::new(AtomicU32::new(0)),
        });

        let mut resilient = ResilientConnection::new("test-conn".to_string(), factory, manager)
            .await
            .unwrap();

        // Should work normally
        assert!(resilient.is_healthy().await);
        let conn = resilient.get_connection().await.unwrap();
        assert!(conn.is_healthy().await);
    }
}
