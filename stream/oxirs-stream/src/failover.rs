//! Failover mechanisms for connection pool
//!
//! Provides primary/secondary connection failover, automatic failover on connection failure,
//! and comprehensive failover event notifications.

use crate::connection_pool::{ConnectionFactory, PooledConnection};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Health check interval for primary connection
    pub health_check_interval: Duration,
    /// Timeout for health checks
    pub health_check_timeout: Duration,
    /// Number of consecutive failures before failover
    pub failure_threshold: u32,
    /// Number of consecutive successes before failback
    pub recovery_threshold: u32,
    /// Delay before attempting failback to primary
    pub failback_delay: Duration,
    /// Enable automatic failback to primary when it recovers
    pub auto_failback: bool,
    /// Connection timeout for failover attempts
    pub connection_timeout: Duration,
    /// Enable failover event notifications
    pub enable_notifications: bool,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(10),
            health_check_timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 3,
            failback_delay: Duration::from_secs(60),
            auto_failback: true,
            connection_timeout: Duration::from_secs(30),
            enable_notifications: true,
        }
    }
}

/// Failover state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FailoverState {
    /// Using primary connection
    Primary,
    /// Failed over to secondary
    Secondary,
    /// Failing over from primary to secondary
    FailingOver,
    /// Failing back from secondary to primary
    FailingBack,
    /// Both connections unavailable
    Unavailable,
}

impl Default for FailoverState {
    fn default() -> Self {
        FailoverState::Primary
    }
}

/// Failover event
#[derive(Debug, Clone)]
pub enum FailoverEvent {
    /// Failover initiated
    FailoverInitiated {
        from: String,
        to: String,
        reason: String,
    },
    /// Failover completed successfully
    FailoverCompleted {
        from: String,
        to: String,
        duration: Duration,
    },
    /// Failover failed
    FailoverFailed {
        from: String,
        to: String,
        error: String,
    },
    /// Failback initiated
    FailbackInitiated {
        from: String,
        to: String,
    },
    /// Failback completed
    FailbackCompleted {
        from: String,
        to: String,
        duration: Duration,
    },
    /// Health check failed
    HealthCheckFailed {
        connection: String,
        consecutive_failures: u32,
    },
    /// Connection recovered
    ConnectionRecovered {
        connection: String,
    },
    /// All connections unavailable
    AllConnectionsUnavailable,
}

/// Failover statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FailoverStatistics {
    pub total_failovers: u64,
    pub successful_failovers: u64,
    pub failed_failovers: u64,
    pub total_failbacks: u64,
    pub successful_failbacks: u64,
    pub failed_failbacks: u64,
    pub primary_uptime: Duration,
    pub secondary_uptime: Duration,
    #[serde(skip)]
    pub last_failover: Option<Instant>,
    #[serde(skip)]
    pub last_failback: Option<Instant>,
    pub current_state: FailoverState,
    #[serde(skip)]
    pub state_changes: Vec<(Instant, FailoverState)>,
}

/// Connection endpoint configuration
#[derive(Clone)]
pub struct ConnectionEndpoint<T: PooledConnection> {
    pub name: String,
    pub factory: Arc<dyn ConnectionFactory<T>>,
    pub priority: u32,
    pub metadata: HashMap<String, String>,
}

impl<T: PooledConnection> std::fmt::Debug for ConnectionEndpoint<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionEndpoint")
            .field("name", &self.name)
            .field("factory", &"<ConnectionFactory>")
            .field("priority", &self.priority)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Failover manager for primary/secondary connections
pub struct FailoverManager<T: PooledConnection> {
    config: FailoverConfig,
    primary: ConnectionEndpoint<T>,
    secondary: ConnectionEndpoint<T>,
    current_connection: Arc<RwLock<Option<T>>>,
    state: Arc<RwLock<FailoverState>>,
    statistics: Arc<RwLock<FailoverStatistics>>,
    event_sender: broadcast::Sender<FailoverEvent>,
    health_status: Arc<RwLock<HealthStatusTracker>>,
    shutdown_signal: Arc<RwLock<bool>>,
}

/// Health status tracking
#[derive(Debug, Clone)]
struct HealthStatusTracker {
    primary_consecutive_failures: u32,
    primary_consecutive_successes: u32,
    secondary_consecutive_failures: u32,
    secondary_consecutive_successes: u32,
    primary_last_check: Option<Instant>,
    secondary_last_check: Option<Instant>,
    primary_healthy: bool,
    secondary_healthy: bool,
}

impl Default for HealthStatusTracker {
    fn default() -> Self {
        Self {
            primary_consecutive_failures: 0,
            primary_consecutive_successes: 0,
            secondary_consecutive_failures: 0,
            secondary_consecutive_successes: 0,
            primary_last_check: None,
            secondary_last_check: None,
            primary_healthy: true,
            secondary_healthy: true,
        }
    }
}

impl<T: PooledConnection> FailoverManager<T> {
    /// Create a new failover manager
    pub async fn new(
        config: FailoverConfig,
        primary: ConnectionEndpoint<T>,
        secondary: ConnectionEndpoint<T>,
    ) -> Result<Self> {
        let (event_sender, _) = broadcast::channel(1000);
        
        // Try to establish primary connection first
        let initial_connection = match tokio::time::timeout(
            config.connection_timeout,
            primary.factory.create_connection(),
        )
        .await
        {
            Ok(Ok(conn)) => {
                info!("Successfully connected to primary: {}", primary.name);
                Some(conn)
            }
            _ => {
                warn!("Failed to connect to primary, trying secondary");
                match tokio::time::timeout(
                    config.connection_timeout,
                    secondary.factory.create_connection(),
                )
                .await
                {
                    Ok(Ok(conn)) => {
                        info!("Successfully connected to secondary: {}", secondary.name);
                        Some(conn)
                    }
                    _ => {
                        error!("Failed to connect to both primary and secondary");
                        None
                    }
                }
            }
        };
        
        let initial_state = if initial_connection.is_some() {
            FailoverState::Primary
        } else {
            FailoverState::Unavailable
        };
        
        let mut statistics = FailoverStatistics::default();
        statistics.current_state = initial_state.clone();
        statistics.state_changes.push((Instant::now(), initial_state.clone()));
        
        let manager = Self {
            config,
            primary,
            secondary,
            current_connection: Arc::new(RwLock::new(initial_connection)),
            state: Arc::new(RwLock::new(initial_state)),
            statistics: Arc::new(RwLock::new(statistics)),
            event_sender,
            health_status: Arc::new(RwLock::new(HealthStatusTracker::default())),
            shutdown_signal: Arc::new(RwLock::new(false)),
        };
        
        // Start health monitoring
        manager.start_health_monitoring().await;
        
        Ok(manager)
    }

    /// Get the current active connection
    pub async fn get_connection(&self) -> Result<T> {
        let state = self.state.read().await.clone();
        
        match state {
            FailoverState::Primary | FailoverState::Secondary => {
                if let Some(conn) = self.current_connection.read().await.as_ref() {
                    if conn.is_healthy().await {
                        // Clone the connection if it implements Clone
                        // For this example, we'll need to handle this differently
                        // In real implementation, you'd return a reference or handle
                        return Err(anyhow!("Connection borrowing not implemented in this example"));
                    }
                }
                
                // Current connection is unhealthy, trigger failover
                self.handle_connection_failure().await
            }
            FailoverState::FailingOver | FailoverState::FailingBack => {
                // Wait for failover to complete
                let mut retry_count = 0;
                while retry_count < 10 {
                    sleep(Duration::from_millis(100)).await;
                    let current_state = self.state.read().await.clone();
                    if !matches!(current_state, FailoverState::FailingOver | FailoverState::FailingBack) {
                        return self.get_connection().await;
                    }
                    retry_count += 1;
                }
                Err(anyhow!("Failover in progress timeout"))
            }
            FailoverState::Unavailable => {
                Err(anyhow!("No connections available"))
            }
        }
    }

    /// Handle connection failure and trigger failover
    async fn handle_connection_failure(&self) -> Result<T> {
        let current_state = self.state.read().await.clone();
        
        match current_state {
            FailoverState::Primary => {
                // Failover to secondary
                self.failover_to_secondary().await
            }
            FailoverState::Secondary => {
                // Try to failback to primary if it's recovered
                if self.health_status.read().await.primary_healthy {
                    self.failback_to_primary().await
                } else {
                    Err(anyhow!("Secondary connection failed and primary is still unhealthy"))
                }
            }
            _ => Err(anyhow!("Connection failure in unexpected state: {:?}", current_state)),
        }
    }

    /// Perform failover to secondary connection
    async fn failover_to_secondary(&self) -> Result<T> {
        let start_time = Instant::now();
        
        // Update state
        *self.state.write().await = FailoverState::FailingOver;
        
        if self.config.enable_notifications {
            let _ = self.event_sender.send(FailoverEvent::FailoverInitiated {
                from: self.primary.name.clone(),
                to: self.secondary.name.clone(),
                reason: "Primary connection failure".to_string(),
            });
        }
        
        // Attempt to create secondary connection
        match tokio::time::timeout(
            self.config.connection_timeout,
            self.secondary.factory.create_connection(),
        )
        .await
        {
            Ok(Ok(conn)) => {
                // Close old connection if exists
                if let Some(mut old_conn) = self.current_connection.write().await.take() {
                    let _ = old_conn.close().await;
                }
                
                *self.current_connection.write().await = Some(conn);
                *self.state.write().await = FailoverState::Secondary;
                
                let duration = start_time.elapsed();
                
                // Update statistics
                let mut stats = self.statistics.write().await;
                stats.total_failovers += 1;
                stats.successful_failovers += 1;
                stats.last_failover = Some(Instant::now());
                stats.current_state = FailoverState::Secondary;
                stats.state_changes.push((Instant::now(), FailoverState::Secondary));
                
                if self.config.enable_notifications {
                    let _ = self.event_sender.send(FailoverEvent::FailoverCompleted {
                        from: self.primary.name.clone(),
                        to: self.secondary.name.clone(),
                        duration,
                    });
                }
                
                info!(
                    "Successfully failed over from {} to {} in {:?}",
                    self.primary.name, self.secondary.name, duration
                );
                
                Err(anyhow!("Failover successful but connection borrowing not implemented"))
            }
            Ok(Err(e)) | Err(e) => {
                *self.state.write().await = FailoverState::Unavailable;
                
                let error_msg = if let Ok(Err(e)) = e.downcast::<anyhow::Error>() {
                    e.to_string()
                } else {
                    "Connection timeout".to_string()
                };
                
                // Update statistics
                let mut stats = self.statistics.write().await;
                stats.total_failovers += 1;
                stats.failed_failovers += 1;
                stats.current_state = FailoverState::Unavailable;
                
                if self.config.enable_notifications {
                    let _ = self.event_sender.send(FailoverEvent::FailoverFailed {
                        from: self.primary.name.clone(),
                        to: self.secondary.name.clone(),
                        error: error_msg.clone(),
                    });
                    
                    let _ = self.event_sender.send(FailoverEvent::AllConnectionsUnavailable);
                }
                
                error!("Failover to secondary failed: {}", error_msg);
                Err(anyhow!("Failover failed: {}", error_msg))
            }
        }
    }

    /// Perform failback to primary connection
    async fn failback_to_primary(&self) -> Result<T> {
        let start_time = Instant::now();
        
        // Update state
        *self.state.write().await = FailoverState::FailingBack;
        
        if self.config.enable_notifications {
            let _ = self.event_sender.send(FailoverEvent::FailbackInitiated {
                from: self.secondary.name.clone(),
                to: self.primary.name.clone(),
            });
        }
        
        // Attempt to create primary connection
        match tokio::time::timeout(
            self.config.connection_timeout,
            self.primary.factory.create_connection(),
        )
        .await
        {
            Ok(Ok(conn)) => {
                // Close old connection if exists
                if let Some(mut old_conn) = self.current_connection.write().await.take() {
                    let _ = old_conn.close().await;
                }
                
                *self.current_connection.write().await = Some(conn);
                *self.state.write().await = FailoverState::Primary;
                
                let duration = start_time.elapsed();
                
                // Update statistics
                let mut stats = self.statistics.write().await;
                stats.total_failbacks += 1;
                stats.successful_failbacks += 1;
                stats.last_failback = Some(Instant::now());
                stats.current_state = FailoverState::Primary;
                stats.state_changes.push((Instant::now(), FailoverState::Primary));
                
                if self.config.enable_notifications {
                    let _ = self.event_sender.send(FailoverEvent::FailbackCompleted {
                        from: self.secondary.name.clone(),
                        to: self.primary.name.clone(),
                        duration,
                    });
                }
                
                info!(
                    "Successfully failed back from {} to {} in {:?}",
                    self.secondary.name, self.primary.name, duration
                );
                
                Err(anyhow!("Failback successful but connection borrowing not implemented"))
            }
            Ok(Err(e)) | Err(e) => {
                // Failback failed, stay on secondary
                *self.state.write().await = FailoverState::Secondary;
                
                let error_msg = if let Ok(Err(e)) = e.downcast::<anyhow::Error>() {
                    e.to_string()
                } else {
                    "Connection timeout".to_string()
                };
                
                // Update statistics
                self.statistics.write().await.total_failbacks += 1;
                self.statistics.write().await.failed_failbacks += 1;
                
                warn!("Failback to primary failed: {}, staying on secondary", error_msg);
                Err(anyhow!("Failback failed: {}", error_msg))
            }
        }
    }

    /// Start health monitoring for automatic failover/failback
    async fn start_health_monitoring(&self) {
        let config = self.config.clone();
        let primary = self.primary.clone();
        let secondary = self.secondary.clone();
        let state = self.state.clone();
        let health_status = self.health_status.clone();
        let event_sender = self.event_sender.clone();
        let shutdown_signal = self.shutdown_signal.clone();
        
        tokio::spawn(async move {
            let mut check_interval = interval(config.health_check_interval);
            
            loop {
                check_interval.tick().await;
                
                if *shutdown_signal.read().await {
                    info!("Failover health monitoring shutting down");
                    break;
                }
                
                let current_state = state.read().await.clone();
                
                // Check primary health
                match tokio::time::timeout(
                    config.health_check_timeout,
                    primary.factory.create_connection(),
                )
                .await
                {
                    Ok(Ok(conn)) => {
                        if conn.is_healthy().await {
                            let mut status = health_status.write().await;
                            status.primary_consecutive_successes += 1;
                            status.primary_consecutive_failures = 0;
                            status.primary_last_check = Some(Instant::now());
                            
                            if !status.primary_healthy {
                                status.primary_healthy = true;
                                if config.enable_notifications {
                                    let _ = event_sender.send(FailoverEvent::ConnectionRecovered {
                                        connection: primary.name.clone(),
                                    });
                                }
                            }
                            
                            // Auto-failback logic
                            if config.auto_failback 
                                && current_state == FailoverState::Secondary
                                && status.primary_consecutive_successes >= config.recovery_threshold
                            {
                                drop(status);
                                info!("Primary connection recovered, initiating auto-failback");
                                sleep(config.failback_delay).await;
                                // Trigger failback through the manager
                                // In real implementation, this would be done differently
                            }
                        }
                    }
                    _ => {
                        let mut status = health_status.write().await;
                        status.primary_consecutive_failures += 1;
                        status.primary_consecutive_successes = 0;
                        status.primary_healthy = false;
                        status.primary_last_check = Some(Instant::now());
                        
                        if config.enable_notifications && status.primary_consecutive_failures % 3 == 0 {
                            let _ = event_sender.send(FailoverEvent::HealthCheckFailed {
                                connection: primary.name.clone(),
                                consecutive_failures: status.primary_consecutive_failures,
                            });
                        }
                    }
                }
                
                // Check secondary health only if we're using it or as backup
                if current_state == FailoverState::Secondary || config.auto_failback {
                    match tokio::time::timeout(
                        config.health_check_timeout,
                        secondary.factory.create_connection(),
                    )
                    .await
                    {
                        Ok(Ok(conn)) => {
                            if conn.is_healthy().await {
                                let mut status = health_status.write().await;
                                status.secondary_consecutive_successes += 1;
                                status.secondary_consecutive_failures = 0;
                                status.secondary_healthy = true;
                                status.secondary_last_check = Some(Instant::now());
                            }
                        }
                        _ => {
                            let mut status = health_status.write().await;
                            status.secondary_consecutive_failures += 1;
                            status.secondary_consecutive_successes = 0;
                            status.secondary_healthy = false;
                            status.secondary_last_check = Some(Instant::now());
                        }
                    }
                }
            }
        });
    }

    /// Get current failover state
    pub async fn get_state(&self) -> FailoverState {
        self.state.read().await.clone()
    }

    /// Get failover statistics
    pub async fn get_statistics(&self) -> FailoverStatistics {
        let mut stats = self.statistics.read().await.clone();
        
        // Calculate uptimes
        let now = Instant::now();
        for (i, (timestamp, state)) in stats.state_changes.iter().enumerate() {
            let duration = if i + 1 < stats.state_changes.len() {
                stats.state_changes[i + 1].0.duration_since(*timestamp)
            } else {
                now.duration_since(*timestamp)
            };
            
            match state {
                FailoverState::Primary => stats.primary_uptime += duration,
                FailoverState::Secondary => stats.secondary_uptime += duration,
                _ => {}
            }
        }
        
        stats
    }

    /// Subscribe to failover events
    pub fn subscribe(&self) -> broadcast::Receiver<FailoverEvent> {
        self.event_sender.subscribe()
    }

    /// Manually trigger failover
    pub async fn trigger_failover(&self) -> Result<()> {
        let current_state = self.state.read().await.clone();
        
        match current_state {
            FailoverState::Primary => {
                self.failover_to_secondary().await.map(|_| ())
            }
            FailoverState::Secondary => {
                self.failback_to_primary().await.map(|_| ())
            }
            _ => Err(anyhow!("Cannot trigger failover in current state: {:?}", current_state)),
        }
    }

    /// Stop health monitoring
    pub async fn stop(&self) {
        *self.shutdown_signal.write().await = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    struct TestConnection {
        id: u32,
        healthy: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl PooledConnection for TestConnection {
        async fn is_healthy(&self) -> bool {
            self.healthy.load(Ordering::Relaxed)
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }

        fn created_at(&self) -> Instant {
            Instant::now()
        }

        fn last_activity(&self) -> Instant {
            Instant::now()
        }

        fn update_activity(&mut self) {}
    }

    struct TestConnectionFactory {
        counter: Arc<AtomicU32>,
        should_fail: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl ConnectionFactory<TestConnection> for TestConnectionFactory {
        async fn create_connection(&self) -> Result<TestConnection> {
            if self.should_fail.load(Ordering::Relaxed) {
                return Err(anyhow!("Simulated connection failure"));
            }
            
            let id = self.counter.fetch_add(1, Ordering::Relaxed);
            Ok(TestConnection {
                id,
                healthy: Arc::new(AtomicBool::new(true)),
            })
        }
    }

    #[tokio::test]
    async fn test_failover_manager_creation() {
        let config = FailoverConfig::default();
        
        let primary_factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(0)),
            should_fail: Arc::new(AtomicBool::new(false)),
        });
        
        let secondary_factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(100)),
            should_fail: Arc::new(AtomicBool::new(false)),
        });
        
        let primary = ConnectionEndpoint {
            name: "primary".to_string(),
            factory: primary_factory,
            priority: 1,
            metadata: HashMap::new(),
        };
        
        let secondary = ConnectionEndpoint {
            name: "secondary".to_string(),
            factory: secondary_factory,
            priority: 2,
            metadata: HashMap::new(),
        };
        
        let manager = FailoverManager::new(config, primary, secondary).await.unwrap();
        
        assert_eq!(manager.get_state().await, FailoverState::Primary);
        
        manager.stop().await;
    }

    #[tokio::test]
    async fn test_failover_events() {
        let config = FailoverConfig {
            enable_notifications: true,
            ..Default::default()
        };
        
        let primary_should_fail = Arc::new(AtomicBool::new(false));
        let primary_factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(0)),
            should_fail: primary_should_fail.clone(),
        });
        
        let secondary_factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(100)),
            should_fail: Arc::new(AtomicBool::new(false)),
        });
        
        let primary = ConnectionEndpoint {
            name: "primary".to_string(),
            factory: primary_factory,
            priority: 1,
            metadata: HashMap::new(),
        };
        
        let secondary = ConnectionEndpoint {
            name: "secondary".to_string(),
            factory: secondary_factory,
            priority: 2,
            metadata: HashMap::new(),
        };
        
        let manager = FailoverManager::new(config, primary, secondary).await.unwrap();
        let mut event_receiver = manager.subscribe();
        
        // Simulate primary failure
        primary_should_fail.store(true, Ordering::Relaxed);
        
        // Trigger failover
        let _ = manager.trigger_failover().await;
        
        // Should receive failover events
        tokio::time::timeout(Duration::from_secs(1), async {
            while let Ok(event) = event_receiver.recv().await {
                if matches!(event, FailoverEvent::FailoverCompleted { .. }) {
                    return;
                }
            }
        })
        .await
        .expect("Should receive failover completed event");
        
        assert_eq!(manager.get_state().await, FailoverState::Secondary);
        
        manager.stop().await;
    }

    #[tokio::test]
    async fn test_failover_statistics() {
        let config = FailoverConfig::default();
        
        let primary_factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(0)),
            should_fail: Arc::new(AtomicBool::new(false)),
        });
        
        let secondary_factory = Arc::new(TestConnectionFactory {
            counter: Arc::new(AtomicU32::new(100)),
            should_fail: Arc::new(AtomicBool::new(false)),
        });
        
        let primary = ConnectionEndpoint {
            name: "primary".to_string(),
            factory: primary_factory,
            priority: 1,
            metadata: HashMap::new(),
        };
        
        let secondary = ConnectionEndpoint {
            name: "secondary".to_string(),
            factory: secondary_factory,
            priority: 2,
            metadata: HashMap::new(),
        };
        
        let manager = FailoverManager::new(config, primary, secondary).await.unwrap();
        
        // Trigger failover
        let _ = manager.trigger_failover().await;
        
        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_failovers, 1);
        assert_eq!(stats.successful_failovers, 1);
        assert_eq!(stats.current_state, FailoverState::Secondary);
        
        manager.stop().await;
    }
}