//! Connection Pool — Pool lifecycle management
//!
//! Acquire/release, connection reuse, idle cleanup, adaptive sizing,
//! the ConnectionPool struct and PooledConnectionHandle.

use crate::{
    circuit_breaker::{
        new_shared_circuit_breaker, FailureType, SharedCircuitBreaker, SharedCircuitBreakerExt,
    },
    failover::{ConnectionEndpoint, FailoverConfig, FailoverManager},
    health_monitor::{HealthCheckConfig, HealthMonitor},
    reconnect::{ReconnectConfig, ReconnectManager, ReconnectStrategy},
    StreamConfig,
};
use anyhow::{anyhow, Result};
use fastrand;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, RwLock, Semaphore};
use tracing::{debug, error, info, warn};

use super::connection_pool_types::{
    AdaptiveController, ConnectionFactory, DetailedPoolMetrics, LoadBalancingStrategy, PoolConfig,
    PoolMetrics, PoolStats, PoolStatus, PooledConnection, PooledConnectionWrapper,
};

/// Advanced connection pool implementation with monitoring and load balancing
pub struct ConnectionPool<T: PooledConnection + Clone> {
    pub(super) config: PoolConfig,
    pub(super) connections: Arc<Mutex<VecDeque<PooledConnectionWrapper<T>>>>,
    pub(super) active_count: Arc<Mutex<usize>>,
    pub(super) semaphore: Arc<Semaphore>,
    pub(super) stats: Arc<RwLock<PoolStats>>,
    pub(super) connection_factory: Arc<dyn ConnectionFactory<T>>,
    pub(super) circuit_breaker: Option<SharedCircuitBreaker>,
    pub(super) round_robin_counter: Arc<AtomicUsize>,
    pub(super) metrics: Arc<RwLock<PoolMetrics>>,
    pub(super) pending_requests: Arc<AtomicUsize>,
    pub(super) created_at: Instant,
    pub(super) adaptive_controller: Arc<RwLock<AdaptiveController>>,
    pub(super) health_monitor: Arc<HealthMonitor<T>>,
    pub(super) reconnect_manager: Arc<ReconnectManager<T>>,
    pub(super) failover_manager: Option<Arc<FailoverManager<T>>>,
}

impl<T: PooledConnection + Clone> ConnectionPool<T> {
    /// Create a new advanced connection pool
    pub async fn new(config: PoolConfig, factory: Arc<dyn ConnectionFactory<T>>) -> Result<Self> {
        let circuit_breaker = if config.enable_circuit_breaker {
            Some(new_shared_circuit_breaker(
                config.circuit_breaker_config.clone().unwrap_or_default(),
            ))
        } else {
            None
        };

        let adaptive_controller = AdaptiveController {
            enabled: config.adaptive_sizing,
            target_response_time: Duration::from_millis(config.target_response_time_ms),
            current_target_size: config.min_connections,
            ..Default::default()
        };

        let health_check_config = HealthCheckConfig {
            check_interval: config.health_check_interval,
            check_timeout: config.validation_timeout,
            enable_statistics: config.enable_metrics,
            ..Default::default()
        };
        let health_monitor = Arc::new(HealthMonitor::new(health_check_config));

        let reconnect_config = ReconnectConfig {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            max_attempts: config.retry_attempts,
            connection_timeout: config.connection_timeout,
            ..Default::default()
        };
        let reconnect_manager = Arc::new(ReconnectManager::new(
            reconnect_config,
            ReconnectStrategy::ExponentialBackoff,
        ));

        let pool = Self {
            semaphore: Arc::new(Semaphore::new(config.max_connections)),
            connections: Arc::new(Mutex::new(VecDeque::new())),
            active_count: Arc::new(Mutex::new(0)),
            stats: Arc::new(RwLock::new(PoolStats::default())),
            connection_factory: factory,
            circuit_breaker,
            round_robin_counter: Arc::new(AtomicUsize::new(0)),
            metrics: Arc::new(RwLock::new(PoolMetrics::default())),
            pending_requests: Arc::new(AtomicUsize::new(0)),
            created_at: Instant::now(),
            adaptive_controller: Arc::new(RwLock::new(adaptive_controller)),
            health_monitor,
            reconnect_manager,
            failover_manager: None,
            config,
        };

        pool.ensure_min_connections().await?;
        pool.start_maintenance_task().await;

        if pool.config.adaptive_sizing {
            pool.start_adaptive_sizing_task().await;
        }

        pool.start_health_monitoring().await;

        info!(
            "Created advanced connection pool with health monitoring, automatic reconnection, and {} features",
            if pool.circuit_breaker.is_some() { "circuit breaker" } else { "standard" }
        );

        Ok(pool)
    }

    /// Get a connection from the pool
    pub async fn get_connection(&self) -> Result<PooledConnectionHandle<T>> {
        let start_time = Instant::now();
        self.pending_requests.fetch_add(1, Ordering::Relaxed);

        if let Some(cb) = &self.circuit_breaker {
            if !cb.can_execute().await {
                self.pending_requests.fetch_sub(1, Ordering::Relaxed);
                return Err(anyhow!(
                    "Circuit breaker is open - connection pool unavailable"
                ));
            }
        }

        let _permit = tokio::time::timeout(self.config.acquire_timeout, self.semaphore.acquire())
            .await
            .map_err(|_| anyhow!("Timeout acquiring connection from pool"))?
            .map_err(|_| anyhow!("Failed to acquire semaphore permit"))?;

        let connection = match self.try_get_existing_connection_with_lb().await {
            Some(conn) => {
                if let Some(cb) = &self.circuit_breaker {
                    cb.record_success_with_duration(start_time.elapsed()).await;
                }
                conn
            }
            None => match self.create_new_connection().await {
                Ok(conn) => {
                    if let Some(cb) = &self.circuit_breaker {
                        cb.record_success_with_duration(start_time.elapsed()).await;
                    }
                    conn
                }
                Err(e) => {
                    if let Some(cb) = &self.circuit_breaker {
                        cb.record_failure_with_type(FailureType::NetworkError).await;
                    }
                    self.pending_requests.fetch_sub(1, Ordering::Relaxed);
                    return Err(e);
                }
            },
        };

        *self.active_count.lock().await += 1;
        let mut stats = self.stats.write().await;
        stats.total_borrowed += 1;
        stats.load_balancing_decisions += 1;
        drop(stats);

        let wait_time = start_time.elapsed();
        self.update_metrics(wait_time).await;
        self.pending_requests.fetch_sub(1, Ordering::Relaxed);

        Ok(PooledConnectionHandle::new(
            connection,
            self.connections.clone(),
            self.active_count.clone(),
            self.stats.clone(),
            self.metrics.clone(),
            self.adaptive_controller.clone(),
        ))
    }

    /// Try to get an existing healthy connection with load balancing
    async fn try_get_existing_connection_with_lb(&self) -> Option<T> {
        let mut connections = self.connections.lock().await;

        if connections.is_empty() {
            return None;
        }

        let selected_index = match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % connections.len()
            }
            LoadBalancingStrategy::Random => fastrand::usize(..connections.len()),
            LoadBalancingStrategy::LeastRecentlyUsed => connections
                .iter()
                .enumerate()
                .min_by_key(|(_, wrapper)| wrapper.last_activity)
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            LoadBalancingStrategy::LeastConnections => connections
                .iter()
                .enumerate()
                .min_by_key(|(_, wrapper)| wrapper.usage_count)
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            LoadBalancingStrategy::WeightedRoundRobin => connections
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.efficiency_score()
                        .partial_cmp(&b.efficiency_score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0),
        };

        for attempt in 0..connections.len() {
            let index = (selected_index + attempt) % connections.len();

            if let Some(mut wrapper) = connections.remove(index) {
                if wrapper.is_expired(self.config.max_lifetime, self.config.idle_timeout) {
                    if let Err(e) = wrapper.connection.close().await {
                        warn!("Failed to close expired connection: {}", e);
                    }
                    self.stats.write().await.total_destroyed += 1;
                    continue;
                }

                let health_check =
                    tokio::time::timeout(self.config.validation_timeout, wrapper.is_healthy())
                        .await;

                match health_check {
                    Ok(true) => {
                        wrapper.is_in_use = true;
                        wrapper.last_activity = Instant::now();
                        wrapper.last_health_check = Some((Instant::now(), true));
                        debug!(
                            "Selected connection {} using {:?} strategy",
                            wrapper.connection_id, self.config.load_balancing
                        );
                        return Some(wrapper.connection);
                    }
                    Ok(false) | Err(_) => {
                        if let Err(e) = wrapper.connection.close().await {
                            warn!("Failed to close unhealthy connection: {}", e);
                        }
                        let mut stats = self.stats.write().await;
                        stats.health_check_failures += 1;
                        stats.total_destroyed += 1;
                        continue;
                    }
                }
            }
        }

        None
    }

    /// Create a new connection
    pub(super) async fn create_new_connection(&self) -> Result<T> {
        match self.connection_factory.create_connection().await {
            Ok(connection) => {
                self.stats.write().await.total_created += 1;
                debug!("Created new connection");
                Ok(connection)
            }
            Err(e) => {
                self.stats.write().await.creation_failures += 1;
                error!("Failed to create connection: {}", e);
                Err(e)
            }
        }
    }

    /// Return a connection to the pool with usage tracking
    async fn return_connection_with_metrics(
        &self,
        mut connection: T,
        execution_time: Duration,
        success: bool,
    ) {
        connection.update_activity();

        let mut wrapper = PooledConnectionWrapper::new(connection);
        wrapper.record_usage(execution_time, success);
        wrapper.is_in_use = false;

        self.connections.lock().await.push_back(wrapper);

        let mut active_count = self.active_count.lock().await;
        if *active_count > 0 {
            *active_count -= 1;
        }

        self.stats.write().await.total_returned += 1;

        let mut controller = self.adaptive_controller.write().await;
        let utilization = (*active_count as f64) / (self.config.max_connections as f64);
        controller.record_metrics(execution_time, utilization);

        debug!(
            "Returned connection to pool with metrics: exec_time={:?}, success={}",
            execution_time, success
        );
    }

    /// Legacy method for backward compatibility
    #[allow(dead_code)]
    async fn return_connection(&self, connection: T) {
        self.return_connection_with_metrics(connection, Duration::from_millis(100), true)
            .await;
    }

    /// Ensure minimum connections are available
    pub(super) async fn ensure_min_connections(&self) -> Result<()> {
        let current_count = self.connections.lock().await.len();
        let active_count = *self.active_count.lock().await;
        let total_count = current_count + active_count;

        if total_count < self.config.min_connections {
            let needed = self.config.min_connections - total_count;
            for _ in 0..needed {
                match self.create_new_connection().await {
                    Ok(connection) => {
                        let wrapper = PooledConnectionWrapper::new(connection);
                        self.connections.lock().await.push_back(wrapper);
                    }
                    Err(e) => {
                        warn!("Failed to create minimum connection: {}", e);
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Update pool metrics
    pub(super) async fn update_metrics(&self, wait_time: Duration) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        let wait_time_ms = wait_time.as_millis() as f64;
        let alpha = 0.1;
        metrics.avg_wait_time_ms = alpha * wait_time_ms + (1.0 - alpha) * metrics.avg_wait_time_ms;

        let connections = self.connections.lock().await;
        let active_count = *self.active_count.lock().await;
        let utilization = (active_count as f64) / (self.config.max_connections as f64);

        metrics
            .utilization_history
            .push_back((Instant::now(), utilization));
        if metrics.utilization_history.len() > 1000 {
            metrics.utilization_history.pop_front();
        }

        metrics.current_size = connections.len() + active_count;
        metrics.peak_size = metrics.peak_size.max(metrics.current_size);
        metrics.last_updated = Instant::now();
    }

    /// Get comprehensive pool status
    pub async fn status(&self) -> PoolStatus {
        let connections = self.connections.lock().await;
        let active_count = *self.active_count.lock().await;
        let metrics = self.metrics.read().await;
        let pending = self.pending_requests.load(Ordering::Relaxed);

        let total_connections = connections.len() + active_count;
        let utilization = if self.config.max_connections > 0 {
            (total_connections as f64 / self.config.max_connections as f64) * 100.0
        } else {
            0.0
        };

        let circuit_breaker_open = if let Some(cb) = &self.circuit_breaker {
            !cb.is_healthy().await
        } else {
            false
        };

        let is_healthy =
            !circuit_breaker_open && utilization < 95.0 && metrics.avg_wait_time_ms < 1000.0;

        PoolStatus {
            total_connections,
            active_connections: active_count,
            idle_connections: connections.len(),
            pending_requests: pending,
            is_healthy,
            last_health_check: Some(Instant::now()),
            utilization_percent: utilization,
            avg_response_time_ms: metrics.avg_wait_time_ms,
            load_balancing_strategy: self.config.load_balancing.clone(),
            circuit_breaker_open,
            config_hash: self.calculate_config_hash(),
        }
    }

    fn calculate_config_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.config.min_connections.hash(&mut hasher);
        self.config.max_connections.hash(&mut hasher);
        self.config.adaptive_sizing.hash(&mut hasher);
        hasher.finish()
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        self.stats.read().await.clone()
    }

    /// Create a connection pool from stream configuration
    pub async fn new_from_config(
        config: &StreamConfig,
        factory: Arc<dyn ConnectionFactory<T>>,
    ) -> Result<Self> {
        let pool_config = PoolConfig {
            min_connections: 1,
            max_connections: config.max_connections,
            connection_timeout: config.connection_timeout,
            adaptive_sizing: true,
            enable_circuit_breaker: true,
            enable_metrics: true,
            ..Default::default()
        };
        Self::new(pool_config, factory).await
    }

    /// Health check with enhanced status
    pub async fn health_check(&self) -> PoolStatus {
        self.status().await
    }

    /// Get detailed pool metrics
    pub async fn get_detailed_metrics(&self) -> DetailedPoolMetrics {
        let status = self.status().await;
        let metrics = self.metrics.read().await;
        let stats = self.stats.read().await;
        let controller = self.adaptive_controller.read().await;

        DetailedPoolMetrics {
            status,
            total_requests: metrics.total_requests,
            peak_size: metrics.peak_size,
            avg_wait_time_ms: metrics.avg_wait_time_ms,
            response_time_p50: metrics.response_time_p50,
            response_time_p95: metrics.response_time_p95,
            response_time_p99: metrics.response_time_p99,
            adaptive_scaling_events: stats.adaptive_scaling_events,
            circuit_breaker_failures: stats.circuit_breaker_failures,
            load_balancing_decisions: stats.load_balancing_decisions,
            current_target_size: controller.current_target_size,
            pool_uptime: self.created_at.elapsed(),
        }
    }

    /// Reset pool statistics
    pub async fn reset_statistics(&self) {
        *self.stats.write().await = PoolStats::default();
        *self.metrics.write().await = PoolMetrics::default();
        info!("Pool statistics reset");
    }

    /// Force resize the pool
    pub async fn resize(&self, new_size: usize) -> Result<()> {
        if new_size < self.config.min_connections || new_size > self.config.max_connections {
            return Err(anyhow!(
                "New size {} outside allowed range [{}, {}]",
                new_size,
                self.config.min_connections,
                self.config.max_connections
            ));
        }
        let mut controller = self.adaptive_controller.write().await;
        controller.current_target_size = new_size;
        controller.last_adjustment = Instant::now();
        info!("Pool manually resized to {}", new_size);
        Ok(())
    }

    /// Create a connection pool with failover support
    pub async fn new_with_failover(
        config: PoolConfig,
        primary_factory: Arc<dyn ConnectionFactory<T>>,
        secondary_factory: Arc<dyn ConnectionFactory<T>>,
        failover_config: FailoverConfig,
    ) -> Result<Self> {
        let primary_endpoint = ConnectionEndpoint {
            name: "primary".to_string(),
            factory: primary_factory.clone(),
            priority: 1,
            metadata: HashMap::new(),
        };
        let secondary_endpoint = ConnectionEndpoint {
            name: "secondary".to_string(),
            factory: secondary_factory,
            priority: 2,
            metadata: HashMap::new(),
        };

        let failover_manager = Arc::new(
            FailoverManager::new(failover_config, primary_endpoint, secondary_endpoint).await?,
        );

        let mut pool = Self::new(config, primary_factory).await?;
        pool.failover_manager = Some(failover_manager.clone());

        let mut failover_events = failover_manager.subscribe();
        let stats = pool.stats.clone();

        tokio::spawn(async move {
            while let Ok(event) = failover_events.recv().await {
                match event {
                    crate::failover::FailoverEvent::FailoverCompleted { from, to, duration } => {
                        info!(
                            "Failover completed from {} to {} in {:?}",
                            from, to, duration
                        );
                        stats.write().await.failover_count += 1;
                    }
                    crate::failover::FailoverEvent::FailbackCompleted { from, to, duration } => {
                        info!(
                            "Failback completed from {} to {} in {:?}",
                            from, to, duration
                        );
                    }
                    crate::failover::FailoverEvent::AllConnectionsUnavailable => {
                        error!("All connections unavailable!");
                    }
                    _ => {}
                }
            }
        });

        Ok(pool)
    }

    /// Get health statistics from the health monitor
    pub async fn get_health_statistics(&self) -> crate::health_monitor::OverallHealthStatistics {
        self.health_monitor.get_overall_statistics().await
    }

    /// Get reconnection statistics
    pub async fn get_reconnection_statistics(&self) -> crate::reconnect::ReconnectStatistics {
        self.reconnect_manager.get_statistics().await
    }

    /// Get failover statistics if failover is enabled
    pub async fn get_failover_statistics(&self) -> Option<crate::failover::FailoverStatistics> {
        if let Some(fm) = &self.failover_manager {
            Some(fm.get_statistics().await)
        } else {
            None
        }
    }

    /// Register a connection failure callback for automatic reconnection
    pub async fn register_failure_callback<F>(&self, callback: F)
    where
        F: Fn(String, String, u32) -> Pin<Box<dyn Future<Output = ()> + Send>>
            + Send
            + Sync
            + 'static,
    {
        self.reconnect_manager
            .register_failure_callback(callback)
            .await;
    }

    /// Manually trigger failover (if configured)
    pub async fn trigger_failover(&self) -> Result<()> {
        if let Some(fm) = &self.failover_manager {
            fm.trigger_failover().await
        } else {
            Err(anyhow!("Failover not configured for this pool"))
        }
    }

    /// Check if the pool has failover configured
    pub fn has_failover(&self) -> bool {
        self.failover_manager.is_some()
    }

    /// Get unhealthy connections from health monitor
    pub async fn get_unhealthy_connections(&self) -> Vec<String> {
        self.health_monitor.get_unhealthy_connections().await
    }

    /// Subscribe to health monitoring events
    pub fn subscribe_health_events(
        &self,
    ) -> broadcast::Receiver<crate::health_monitor::HealthEvent> {
        self.health_monitor.subscribe()
    }

    /// Subscribe to reconnection events
    pub fn subscribe_reconnect_events(
        &self,
    ) -> broadcast::Receiver<crate::reconnect::ReconnectEvent> {
        self.reconnect_manager.subscribe()
    }
}

// ── PooledConnectionHandle ────────────────────────────────────────────────────

/// Enhanced connection handle with usage tracking and metrics
pub struct PooledConnectionHandle<T: PooledConnection> {
    connection: Option<T>,
    pool_connections: Arc<Mutex<VecDeque<PooledConnectionWrapper<T>>>>,
    active_count: Arc<Mutex<usize>>,
    stats: Arc<RwLock<PoolStats>>,
    metrics: Arc<RwLock<PoolMetrics>>,
    adaptive_controller: Arc<RwLock<AdaptiveController>>,
    acquired_at: Instant,
    execution_times: Vec<Duration>,
    operation_count: u32,
    success_count: u32,
}

impl<T: PooledConnection> PooledConnectionHandle<T> {
    pub(super) fn new(
        connection: T,
        pool_connections: Arc<Mutex<VecDeque<PooledConnectionWrapper<T>>>>,
        active_count: Arc<Mutex<usize>>,
        stats: Arc<RwLock<PoolStats>>,
        metrics: Arc<RwLock<PoolMetrics>>,
        adaptive_controller: Arc<RwLock<AdaptiveController>>,
    ) -> Self {
        Self {
            connection: Some(connection),
            pool_connections,
            active_count,
            stats,
            metrics,
            adaptive_controller,
            acquired_at: Instant::now(),
            execution_times: Vec::new(),
            operation_count: 0,
            success_count: 0,
        }
    }

    pub fn record_operation(&mut self, execution_time: Duration, success: bool) {
        self.execution_times.push(execution_time);
        self.operation_count += 1;
        if success {
            self.success_count += 1;
        }
        debug!(
            "Recorded operation: time={:?}, success={}, total_ops={}",
            execution_time, success, self.operation_count
        );
    }

    pub fn get_operation_stats(&self) -> (u32, u32, Duration) {
        let avg_time = if !self.execution_times.is_empty() {
            self.execution_times.iter().sum::<Duration>() / self.execution_times.len() as u32
        } else {
            Duration::ZERO
        };
        (self.operation_count, self.success_count, avg_time)
    }

    pub fn held_duration(&self) -> Duration {
        self.acquired_at.elapsed()
    }

    pub fn as_ref(&self) -> Option<&T> {
        self.connection.as_ref()
    }

    pub fn as_mut(&mut self) -> Option<&mut T> {
        self.connection.as_mut()
    }

    pub fn take(mut self) -> Option<T> {
        self.connection.take()
    }
}

impl<T: PooledConnection> Drop for PooledConnectionHandle<T> {
    fn drop(&mut self) {
        if let Some(connection) = self.connection.take() {
            let pool_connections = self.pool_connections.clone();
            let active_count = self.active_count.clone();
            let stats = self.stats.clone();
            let _metrics = self.metrics.clone();
            let adaptive_controller = self.adaptive_controller.clone();

            let total_held_time = self.acquired_at.elapsed();
            let avg_execution_time = if !self.execution_times.is_empty() {
                self.execution_times.iter().sum::<Duration>() / self.execution_times.len() as u32
            } else {
                Duration::from_millis(50)
            };

            let success_rate = if self.operation_count > 0 {
                self.success_count as f64 / self.operation_count as f64
            } else {
                1.0
            };
            let overall_success = success_rate > 0.8;

            tokio::spawn(async move {
                let mut wrapper = PooledConnectionWrapper::new(connection);
                wrapper.record_usage(avg_execution_time, overall_success);
                wrapper.is_in_use = false;

                let usage_count = wrapper.usage_count;
                pool_connections.lock().await.push_back(wrapper);

                let mut active = active_count.lock().await;
                if *active > 0 {
                    *active -= 1;
                }

                stats.write().await.total_returned += 1;

                let utilization = (*active as f64) / 10.0;
                adaptive_controller
                    .write()
                    .await
                    .record_metrics(avg_execution_time, utilization);

                debug!(
                    "Returned connection to pool: held_time={:?}, ops={}, success_rate={:.2}",
                    total_held_time, usage_count, success_rate
                );
            });
        }
    }
}
