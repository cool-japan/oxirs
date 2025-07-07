//! Connection health monitoring and tracking
//!
//! Provides advanced health monitoring capabilities including periodic checks,
//! dead connection detection, and comprehensive statistics tracking.

use crate::connection_pool::PooledConnection;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Interval between health checks
    pub check_interval: Duration,
    /// Timeout for individual health checks
    pub check_timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub recovery_threshold: u32,
    /// Enable detailed statistics tracking
    pub enable_statistics: bool,
    /// Health check retry attempts
    pub retry_attempts: u32,
    /// Delay between retry attempts
    pub retry_delay: Duration,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 2,
            enable_statistics: true,
            retry_attempts: 2,
            retry_delay: Duration::from_millis(500),
        }
    }
}

/// Connection health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Connection is healthy and operational
    Healthy,
    /// Connection is degraded but still operational
    Degraded,
    /// Connection is unhealthy and should not be used
    Unhealthy,
    /// Connection is dead and should be removed
    Dead,
    /// Health status is unknown (not checked yet)
    Unknown,
}

/// Connection health statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatistics {
    /// Total health checks performed
    pub total_checks: u64,
    /// Successful health checks
    pub successful_checks: u64,
    /// Failed health checks
    pub failed_checks: u64,
    /// Average response time for health checks
    pub avg_response_time_ms: f64,
    /// Minimum response time
    pub min_response_time_ms: f64,
    /// Maximum response time
    pub max_response_time_ms: f64,
    /// Current consecutive failures
    pub consecutive_failures: u32,
    /// Current consecutive successes
    pub consecutive_successes: u32,
    /// Last health check timestamp
    #[serde(skip)]
    pub last_check: Option<Instant>,
    /// Last successful check timestamp
    #[serde(skip)]
    pub last_success: Option<Instant>,
    /// Last failure timestamp
    #[serde(skip)]
    pub last_failure: Option<Instant>,
    /// Error types encountered
    pub error_counts: HashMap<String, u64>,
}

impl Default for HealthStatistics {
    fn default() -> Self {
        Self {
            total_checks: 0,
            successful_checks: 0,
            failed_checks: 0,
            avg_response_time_ms: 0.0,
            min_response_time_ms: f64::MAX,
            max_response_time_ms: 0.0,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_check: None,
            last_success: None,
            last_failure: None,
            error_counts: HashMap::new(),
        }
    }
}

/// Connection health record
#[derive(Debug, Clone)]
pub struct ConnectionHealthRecord {
    /// Connection identifier
    pub connection_id: String,
    /// Current health status
    pub status: HealthStatus,
    /// Health statistics
    pub statistics: HealthStatistics,
    /// Connection metadata
    pub metadata: HashMap<String, String>,
    /// Health check history (limited to last N entries)
    pub history: Vec<HealthCheckResult>,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub success: bool,
    pub response_time_ms: f64,
    pub error: Option<String>,
}

/// Health monitor for connection pool
pub struct HealthMonitor<T: PooledConnection> {
    config: HealthCheckConfig,
    /// Health records for all connections
    health_records: Arc<RwLock<HashMap<String, ConnectionHealthRecord>>>,
    /// Event broadcaster for health status changes
    event_sender: broadcast::Sender<HealthEvent>,
    /// Shutdown signal
    shutdown_signal: Arc<RwLock<bool>>,
    /// Type marker
    _phantom: std::marker::PhantomData<T>,
}

/// Health monitoring events
#[derive(Debug, Clone)]
pub enum HealthEvent {
    /// Connection status changed
    StatusChanged {
        connection_id: String,
        old_status: HealthStatus,
        new_status: HealthStatus,
    },
    /// Connection marked as dead
    ConnectionDead {
        connection_id: String,
        reason: String,
    },
    /// Connection recovered
    ConnectionRecovered { connection_id: String },
    /// Health check failed
    HealthCheckFailed {
        connection_id: String,
        error: String,
    },
}

impl<T: PooledConnection> HealthMonitor<T> {
    /// Create a new health monitor
    pub fn new(config: HealthCheckConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            config,
            health_records: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            shutdown_signal: Arc::new(RwLock::new(false)),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Register a connection for health monitoring
    pub async fn register_connection(
        &self,
        connection_id: String,
        metadata: HashMap<String, String>,
    ) {
        let mut records = self.health_records.write().await;

        let record = ConnectionHealthRecord {
            connection_id: connection_id.clone(),
            status: HealthStatus::Unknown,
            statistics: HealthStatistics::default(),
            metadata,
            history: Vec::with_capacity(100),
        };

        records.insert(connection_id.clone(), record);
        info!(
            "Registered connection {} for health monitoring",
            connection_id
        );
    }

    /// Unregister a connection from health monitoring
    pub async fn unregister_connection(&self, connection_id: &str) {
        let mut records = self.health_records.write().await;
        if records.remove(connection_id).is_some() {
            info!(
                "Unregistered connection {} from health monitoring",
                connection_id
            );
        }
    }

    /// Perform health check on a specific connection
    pub async fn check_connection_health(
        &self,
        connection_id: &str,
        connection: &T,
    ) -> Result<HealthStatus> {
        let start_time = Instant::now();
        let mut attempts = 0;
        let mut last_error = None;

        // Retry logic for health checks
        while attempts < self.config.retry_attempts {
            attempts += 1;

            match tokio::time::timeout(self.config.check_timeout, connection.is_healthy()).await {
                Ok(true) => {
                    let response_time = start_time.elapsed();
                    self.record_health_check_result(connection_id, true, response_time, None)
                        .await?;

                    return Ok(self.determine_health_status(connection_id).await);
                }
                Ok(false) => {
                    last_error = Some("Health check returned false".to_string());
                }
                Err(_) => {
                    last_error = Some("Health check timed out".to_string());
                }
            }

            if attempts < self.config.retry_attempts {
                tokio::time::sleep(self.config.retry_delay).await;
            }
        }

        // All attempts failed
        let response_time = start_time.elapsed();
        self.record_health_check_result(connection_id, false, response_time, last_error.clone())
            .await?;

        let status = self.determine_health_status(connection_id).await;

        if let Some(error) = last_error {
            let _ = self.event_sender.send(HealthEvent::HealthCheckFailed {
                connection_id: connection_id.to_string(),
                error,
            });
        }

        Ok(status)
    }

    /// Record health check result and update statistics
    async fn record_health_check_result(
        &self,
        connection_id: &str,
        success: bool,
        response_time: Duration,
        error: Option<String>,
    ) -> Result<()> {
        let mut records = self.health_records.write().await;

        if let Some(record) = records.get_mut(connection_id) {
            let response_time_ms = response_time.as_millis() as f64;
            let stats = &mut record.statistics;

            // Update basic counters
            stats.total_checks += 1;
            stats.last_check = Some(Instant::now());

            if success {
                stats.successful_checks += 1;
                stats.consecutive_successes += 1;
                stats.consecutive_failures = 0;
                stats.last_success = Some(Instant::now());
            } else {
                stats.failed_checks += 1;
                stats.consecutive_failures += 1;
                stats.consecutive_successes = 0;
                stats.last_failure = Some(Instant::now());

                if let Some(ref err) = error {
                    *stats.error_counts.entry(err.clone()).or_insert(0) += 1;
                }
            }

            // Update response time statistics
            stats.min_response_time_ms = stats.min_response_time_ms.min(response_time_ms);
            stats.max_response_time_ms = stats.max_response_time_ms.max(response_time_ms);

            // Update average with exponential moving average
            let alpha = 0.1;
            if stats.total_checks == 1 {
                stats.avg_response_time_ms = response_time_ms;
            } else {
                stats.avg_response_time_ms =
                    alpha * response_time_ms + (1.0 - alpha) * stats.avg_response_time_ms;
            }

            // Add to history
            let result = HealthCheckResult {
                timestamp: Instant::now(),
                success,
                response_time_ms,
                error,
            };

            record.history.push(result);
            if record.history.len() > 100 {
                record.history.remove(0);
            }
        }

        Ok(())
    }

    /// Determine health status based on statistics
    async fn determine_health_status(&self, connection_id: &str) -> HealthStatus {
        let records = self.health_records.read().await;

        if let Some(record) = records.get(connection_id) {
            let stats = &record.statistics;
            let old_status = record.status.clone();
            let consecutive_failures = stats.consecutive_failures; // Copy the value we need

            let new_status = if stats.consecutive_failures >= self.config.failure_threshold * 2 {
                HealthStatus::Dead
            } else if stats.consecutive_failures >= self.config.failure_threshold {
                HealthStatus::Unhealthy
            } else if stats.consecutive_successes >= self.config.recovery_threshold {
                HealthStatus::Healthy
            } else if stats.consecutive_failures > 0 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Unknown
            };

            // Send event if status changed
            if old_status != new_status {
                drop(records); // Release read lock before sending event

                let _ = self.event_sender.send(HealthEvent::StatusChanged {
                    connection_id: connection_id.to_string(),
                    old_status: old_status.clone(), // Clone instead of moving
                    new_status: new_status.clone(),
                });

                match new_status {
                    HealthStatus::Dead => {
                        let _ = self.event_sender.send(HealthEvent::ConnectionDead {
                            connection_id: connection_id.to_string(),
                            reason: format!("{consecutive_failures} consecutive failures"), // Use copied value
                        });
                    }
                    HealthStatus::Healthy if old_status == HealthStatus::Unhealthy => {
                        let _ = self.event_sender.send(HealthEvent::ConnectionRecovered {
                            connection_id: connection_id.to_string(),
                        });
                    }
                    _ => {}
                }

                // Update the status
                let mut records = self.health_records.write().await;
                if let Some(record) = records.get_mut(connection_id) {
                    record.status = new_status.clone();
                }
            }

            new_status
        } else {
            HealthStatus::Unknown
        }
    }

    /// Start periodic health monitoring
    pub async fn start_monitoring(&self, connections: Arc<RwLock<HashMap<String, T>>>) {
        let health_records = self.health_records.clone();
        let config = self.config.clone();
        let shutdown_signal = self.shutdown_signal.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut check_interval = interval(config.check_interval);

            loop {
                check_interval.tick().await;

                // Check shutdown signal
                if *shutdown_signal.read().await {
                    info!("Health monitor shutting down");
                    break;
                }

                // Get all connections to check
                let connections_guard = connections.read().await;
                let connection_ids: Vec<String> = connections_guard.keys().cloned().collect();
                drop(connections_guard);

                for conn_id in connection_ids {
                    let start_time = Instant::now();

                    // Get connection for health check - avoid holding the lock during async operations
                    let health_check_result = {
                        let connection_guard = connections.read().await;
                        let connection = match connection_guard.get(&conn_id) {
                            Some(conn) => conn,
                            None => continue, // Connection was removed
                        };

                        // Call the health check while holding the guard briefly
                        tokio::time::timeout(config.check_timeout, connection.is_healthy()).await
                    };

                    match health_check_result {
                        Ok(healthy) => {
                            let response_time = start_time.elapsed();
                            let response_time_ms = response_time.as_millis() as f64;

                            // Update health record
                            let mut records = health_records.write().await;
                            if let Some(record) = records.get_mut(&conn_id) {
                                let stats = &mut record.statistics;
                                stats.total_checks += 1;
                                stats.last_check = Some(Instant::now());

                                if healthy {
                                    stats.successful_checks += 1;
                                    stats.consecutive_successes += 1;
                                    stats.consecutive_failures = 0;
                                    stats.last_success = Some(Instant::now());

                                    debug!(
                                        "Connection {} health check passed in {:.2}ms",
                                        conn_id, response_time_ms
                                    );
                                } else {
                                    stats.failed_checks += 1;
                                    stats.consecutive_failures += 1;
                                    stats.consecutive_successes = 0;
                                    stats.last_failure = Some(Instant::now());

                                    warn!("Connection {} health check failed", conn_id);
                                }

                                // Check if status needs to change
                                let old_status = record.status.clone();
                                let new_status = if stats.consecutive_failures
                                    >= config.failure_threshold * 2
                                {
                                    HealthStatus::Dead
                                } else if stats.consecutive_failures >= config.failure_threshold {
                                    HealthStatus::Unhealthy
                                } else if stats.consecutive_successes >= config.recovery_threshold {
                                    HealthStatus::Healthy
                                } else {
                                    old_status.clone()
                                };

                                if old_status != new_status {
                                    record.status = new_status.clone();
                                    let _ = event_sender.send(HealthEvent::StatusChanged {
                                        connection_id: conn_id.clone(),
                                        old_status,
                                        new_status,
                                    });
                                }
                            }
                        }
                        Err(_) => {
                            error!("Health check timeout for connection {}", conn_id);

                            let mut records = health_records.write().await;
                            if let Some(record) = records.get_mut(&conn_id) {
                                record.statistics.failed_checks += 1;
                                record.statistics.consecutive_failures += 1;
                                record.statistics.consecutive_successes = 0;
                                *record
                                    .statistics
                                    .error_counts
                                    .entry("timeout".to_string())
                                    .or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        });
    }

    /// Stop health monitoring
    pub async fn stop_monitoring(&self) {
        *self.shutdown_signal.write().await = true;
    }

    /// Get health status for a specific connection
    pub async fn get_connection_health(
        &self,
        connection_id: &str,
    ) -> Option<ConnectionHealthRecord> {
        self.health_records.read().await.get(connection_id).cloned()
    }

    /// Get all unhealthy connections
    pub async fn get_unhealthy_connections(&self) -> Vec<String> {
        self.health_records
            .read()
            .await
            .iter()
            .filter(|(_, record)| {
                matches!(
                    record.status,
                    HealthStatus::Unhealthy | HealthStatus::Dead | HealthStatus::Degraded
                )
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get dead connections that should be removed
    pub async fn get_dead_connections(&self) -> Vec<String> {
        self.health_records
            .read()
            .await
            .iter()
            .filter(|(_, record)| record.status == HealthStatus::Dead)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get overall health statistics
    pub async fn get_overall_statistics(&self) -> OverallHealthStatistics {
        let records = self.health_records.read().await;

        let total_connections = records.len();
        let healthy_connections = records
            .values()
            .filter(|r| r.status == HealthStatus::Healthy)
            .count();
        let degraded_connections = records
            .values()
            .filter(|r| r.status == HealthStatus::Degraded)
            .count();
        let unhealthy_connections = records
            .values()
            .filter(|r| r.status == HealthStatus::Unhealthy)
            .count();
        let dead_connections = records
            .values()
            .filter(|r| r.status == HealthStatus::Dead)
            .count();

        let total_checks: u64 = records.values().map(|r| r.statistics.total_checks).sum();
        let successful_checks: u64 = records
            .values()
            .map(|r| r.statistics.successful_checks)
            .sum();
        let failed_checks: u64 = records.values().map(|r| r.statistics.failed_checks).sum();

        let avg_response_time_ms = if total_connections > 0 {
            records
                .values()
                .map(|r| r.statistics.avg_response_time_ms)
                .sum::<f64>()
                / total_connections as f64
        } else {
            0.0
        };

        OverallHealthStatistics {
            total_connections,
            healthy_connections,
            degraded_connections,
            unhealthy_connections,
            dead_connections,
            total_checks,
            successful_checks,
            failed_checks,
            success_rate: if total_checks > 0 {
                (successful_checks as f64 / total_checks as f64) * 100.0
            } else {
                0.0
            },
            avg_response_time_ms,
        }
    }

    /// Subscribe to health events
    pub fn subscribe(&self) -> broadcast::Receiver<HealthEvent> {
        self.event_sender.subscribe()
    }
}

/// Overall health statistics across all connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallHealthStatistics {
    pub total_connections: usize,
    pub healthy_connections: usize,
    pub degraded_connections: usize,
    pub unhealthy_connections: usize,
    pub dead_connections: usize,
    pub total_checks: u64,
    pub successful_checks: u64,
    pub failed_checks: u64,
    pub success_rate: f64,
    pub avg_response_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Clone)]
    struct TestConnection {
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

        fn clone_connection(&self) -> Box<dyn PooledConnection> {
            Box::new(TestConnection {
                healthy: Arc::new(AtomicBool::new(self.healthy.load(Ordering::Relaxed))),
            })
        }

        fn created_at(&self) -> Instant {
            Instant::now()
        }

        fn last_activity(&self) -> Instant {
            Instant::now()
        }

        fn update_activity(&mut self) {}
    }

    #[tokio::test]
    async fn test_health_monitoring() {
        let config = HealthCheckConfig::default();
        let monitor = HealthMonitor::<TestConnection>::new(config);

        // Register a connection
        let metadata = HashMap::new();
        monitor
            .register_connection("test-conn-1".to_string(), metadata)
            .await;

        // Create test connection
        let healthy_flag = Arc::new(AtomicBool::new(true));
        let connection = TestConnection {
            healthy: healthy_flag.clone(),
        };

        // Check healthy connection
        let status = monitor
            .check_connection_health("test-conn-1", &connection)
            .await
            .unwrap();
        assert_eq!(status, HealthStatus::Unknown); // First check, no history

        // Multiple successful checks should mark as healthy
        for _ in 0..3 {
            monitor
                .check_connection_health("test-conn-1", &connection)
                .await
                .unwrap();
        }

        let health = monitor.get_connection_health("test-conn-1").await.unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.statistics.consecutive_successes, 4);

        // Make connection unhealthy
        healthy_flag.store(false, Ordering::Relaxed);

        // Multiple failed checks should mark as unhealthy
        for _ in 0..3 {
            monitor
                .check_connection_health("test-conn-1", &connection)
                .await
                .unwrap();
        }

        let health = monitor.get_connection_health("test-conn-1").await.unwrap();
        assert_eq!(health.status, HealthStatus::Unhealthy);
        assert_eq!(health.statistics.consecutive_failures, 3);

        // Check unhealthy connections list
        let unhealthy = monitor.get_unhealthy_connections().await;
        assert!(unhealthy.contains(&"test-conn-1".to_string()));
    }

    #[tokio::test]
    async fn test_dead_connection_detection() {
        let config = HealthCheckConfig { failure_threshold: 2, ..Default::default() };

        let monitor = HealthMonitor::<TestConnection>::new(config);
        monitor
            .register_connection("test-conn-1".to_string(), HashMap::new())
            .await;

        let connection = TestConnection {
            healthy: Arc::new(AtomicBool::new(false)),
        };

        // Enough failures to mark as dead
        for _ in 0..5 {
            monitor
                .check_connection_health("test-conn-1", &connection)
                .await
                .unwrap();
        }

        let health = monitor.get_connection_health("test-conn-1").await.unwrap();
        assert_eq!(health.status, HealthStatus::Dead);

        let dead = monitor.get_dead_connections().await;
        assert!(dead.contains(&"test-conn-1".to_string()));
    }

    #[tokio::test]
    async fn test_health_events() {
        let config = HealthCheckConfig::default();
        let monitor = HealthMonitor::<TestConnection>::new(config);

        let mut event_receiver = monitor.subscribe();

        monitor
            .register_connection("test-conn-1".to_string(), HashMap::new())
            .await;

        let healthy_flag = Arc::new(AtomicBool::new(true));
        let connection = TestConnection {
            healthy: healthy_flag.clone(),
        };

        // Generate health status change
        for _ in 0..3 {
            monitor
                .check_connection_health("test-conn-1", &connection)
                .await
                .unwrap();
        }

        healthy_flag.store(false, Ordering::Relaxed);

        for _ in 0..3 {
            monitor
                .check_connection_health("test-conn-1", &connection)
                .await
                .unwrap();
        }

        // Should receive status change event
        tokio::time::timeout(Duration::from_secs(1), async {
            while let Ok(event) = event_receiver.recv().await {
                if matches!(event, HealthEvent::StatusChanged { .. }) {
                    return;
                }
            }
        })
        .await
        .expect("Should receive status change event");
    }
}
