//! # Advanced Connection Pool Implementation
//!
//! Enterprise-grade connection pooling with adaptive sizing, health monitoring,
//! metrics collection, circuit breaker integration, and intelligent load balancing
//! for high-throughput streaming scenarios.

use crate::{
    circuit_breaker::{
        new_shared_circuit_breaker, CircuitBreaker, CircuitBreakerConfig, FailureType,
        SharedCircuitBreaker, SharedCircuitBreakerExt,
    },
    failover::{ConnectionEndpoint, FailoverConfig, FailoverManager},
    health_monitor::{HealthCheckConfig, HealthMonitor, HealthStatus},
    reconnect::{ReconnectConfig, ReconnectManager, ReconnectStrategy},
    StreamConfig,
};
use anyhow::{anyhow, Result};
use fastrand;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
};

use chrono::{DateTime, Utc};
#[cfg(test)]
use futures_util;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, RwLock, Semaphore};
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Connection pool configuration with advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub min_connections: usize,
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
    pub health_check_interval: Duration,
    pub retry_attempts: u32,
    /// Enable adaptive pool sizing based on load
    pub adaptive_sizing: bool,
    /// Target response time for adaptive sizing (milliseconds)
    pub target_response_time_ms: u64,
    /// Load balancing strategy for connection distribution
    pub load_balancing: LoadBalancingStrategy,
    /// Enable circuit breaker for connection failures
    pub enable_circuit_breaker: bool,
    /// Circuit breaker configuration
    pub circuit_breaker_config: Option<CircuitBreakerConfig>,
    /// Enable comprehensive metrics collection
    pub enable_metrics: bool,
    /// Connection validation timeout
    pub validation_timeout: Duration,
    /// Maximum wait time for acquiring a connection
    pub acquire_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 1,
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300), // 5 minutes
            max_lifetime: Duration::from_secs(1800), // 30 minutes
            health_check_interval: Duration::from_secs(60),
            retry_attempts: 3,
            adaptive_sizing: true,
            target_response_time_ms: 100,
            load_balancing: LoadBalancingStrategy::RoundRobin,
            enable_circuit_breaker: true,
            circuit_breaker_config: Some(CircuitBreakerConfig::default()),
            enable_metrics: true,
            validation_timeout: Duration::from_secs(5),
            acquire_timeout: Duration::from_secs(30),
        }
    }
}

/// Load balancing strategies for connection distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin selection
    RoundRobin,
    /// Least recently used
    LeastRecentlyUsed,
    /// Random selection
    Random,
    /// Least connections (best for varying load)
    LeastConnections,
    /// Weighted round-robin based on response times
    WeightedRoundRobin,
}

/// Connection pool status with comprehensive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStatus {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub pending_requests: usize,
    pub is_healthy: bool,
    #[serde(skip)]
    pub last_health_check: Option<Instant>,
    /// Current pool utilization percentage
    pub utilization_percent: f64,
    /// Average response time across all connections
    pub avg_response_time_ms: f64,
    /// Current load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Circuit breaker status
    pub circuit_breaker_open: bool,
    /// Pool configuration hash for validation
    pub config_hash: u64,
}

/// Generic connection trait
#[async_trait::async_trait]
pub trait PooledConnection: Send + Sync + 'static {
    /// Check if connection is healthy
    async fn is_healthy(&self) -> bool;

    /// Close the connection
    async fn close(&mut self) -> Result<()>;

    /// Get connection creation time
    fn created_at(&self) -> Instant;

    /// Get last activity time
    fn last_activity(&self) -> Instant;

    /// Update last activity time
    fn update_activity(&mut self);
}

/// Implement PooledConnection for Box<dyn PooledConnection> to enable trait object usage
#[async_trait::async_trait]
impl PooledConnection for Box<dyn PooledConnection> {
    async fn is_healthy(&self) -> bool {
        self.as_ref().is_healthy().await
    }

    async fn close(&mut self) -> Result<()> {
        self.as_mut().close().await
    }

    fn created_at(&self) -> Instant {
        self.as_ref().created_at()
    }

    fn last_activity(&self) -> Instant {
        self.as_ref().last_activity()
    }

    fn update_activity(&mut self) {
        self.as_mut().update_activity()
    }
}

/// Connection wrapper with comprehensive metadata and monitoring
struct PooledConnectionWrapper<T: PooledConnection> {
    connection: T,
    created_at: Instant,
    last_activity: Instant,
    is_in_use: bool,
    /// Unique connection identifier
    connection_id: String,
    /// Connection usage statistics
    usage_count: u64,
    /// Total time spent executing operations
    total_execution_time: Duration,
    /// Average response time for this connection
    avg_response_time: Duration,
    /// Number of failures on this connection
    failure_count: u32,
    /// Last known health status
    last_health_check: Option<(Instant, bool)>,
    /// Connection weight for load balancing
    weight: f64,
}

impl<T: PooledConnection> PooledConnectionWrapper<T> {
    fn new(connection: T) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_activity: now,
            is_in_use: false,
            connection_id: Uuid::new_v4().to_string(),
            usage_count: 0,
            total_execution_time: Duration::ZERO,
            avg_response_time: Duration::from_millis(50), // Default baseline
            failure_count: 0,
            last_health_check: None,
            weight: 1.0, // Default weight
        }
    }

    /// Update connection usage statistics
    fn record_usage(&mut self, execution_time: Duration, success: bool) {
        self.usage_count += 1;
        self.last_activity = Instant::now();
        self.total_execution_time += execution_time;

        // Update average response time with exponential moving average
        let alpha = 0.1; // Smoothing factor
        let new_time_ms = execution_time.as_millis() as f64;
        let current_avg_ms = self.avg_response_time.as_millis() as f64;
        let updated_avg_ms = alpha * new_time_ms + (1.0 - alpha) * current_avg_ms;
        self.avg_response_time = Duration::from_millis(updated_avg_ms as u64);

        if !success {
            self.failure_count += 1;
            // Reduce weight for failing connections
            self.weight = (self.weight * 0.9).max(0.1);
        } else if self.failure_count > 0 {
            // Gradually restore weight for recovering connections
            self.weight = (self.weight * 1.01).min(1.0);
        }
    }

    /// Get connection efficiency score for load balancing
    fn efficiency_score(&self) -> f64 {
        if self.usage_count == 0 {
            return 1.0;
        }

        let failure_rate = self.failure_count as f64 / self.usage_count as f64;
        let response_time_penalty = (self.avg_response_time.as_millis() as f64).ln() / 10.0;

        (1.0 - failure_rate) * self.weight / (1.0 + response_time_penalty)
    }

    fn is_expired(&self, max_lifetime: Duration, idle_timeout: Duration) -> bool {
        let now = Instant::now();
        now.duration_since(self.created_at) > max_lifetime
            || (!self.is_in_use && now.duration_since(self.last_activity) > idle_timeout)
    }

    async fn is_healthy(&self) -> bool {
        self.connection.is_healthy().await
    }
}

/// Advanced connection pool implementation with monitoring and load balancing
pub struct ConnectionPool<T: PooledConnection> {
    config: PoolConfig,
    connections: Arc<Mutex<VecDeque<PooledConnectionWrapper<T>>>>,
    active_count: Arc<Mutex<usize>>,
    semaphore: Arc<Semaphore>,
    stats: Arc<RwLock<PoolStats>>,
    connection_factory: Arc<dyn ConnectionFactory<T>>,
    /// Circuit breaker for connection failures
    circuit_breaker: Option<SharedCircuitBreaker>,
    /// Round-robin counter for load balancing
    round_robin_counter: Arc<AtomicUsize>,
    /// Pool metrics and monitoring
    metrics: Arc<RwLock<PoolMetrics>>,
    /// Pending connection requests queue
    pending_requests: Arc<AtomicUsize>,
    /// Pool creation timestamp
    created_at: Instant,
    /// Adaptive sizing controller
    adaptive_controller: Arc<RwLock<AdaptiveController>>,
    /// Health monitor for connection health tracking
    health_monitor: Arc<HealthMonitor<T>>,
    /// Reconnection manager for automatic reconnection
    reconnect_manager: Arc<ReconnectManager<T>>,
    /// Failover manager for primary/secondary failover
    failover_manager: Option<Arc<FailoverManager<T>>>,
}

/// Connection factory trait
#[async_trait::async_trait]
pub trait ConnectionFactory<T: PooledConnection>: Send + Sync {
    async fn create_connection(&self) -> Result<T>;
}

/// Pool statistics with enhanced metrics
#[derive(Debug, Default, Clone)]
struct PoolStats {
    total_created: u64,
    total_destroyed: u64,
    total_borrowed: u64,
    total_returned: u64,
    creation_failures: u64,
    health_check_failures: u64,
    timeouts: u64,
    circuit_breaker_failures: u64,
    adaptive_scaling_events: u64,
    load_balancing_decisions: u64,
    failover_count: u64,
}

/// Comprehensive pool metrics for monitoring
#[derive(Debug, Clone)]
struct PoolMetrics {
    /// Current pool size
    current_size: usize,
    /// Peak pool size reached
    peak_size: usize,
    /// Total connection requests
    total_requests: u64,
    /// Average wait time for connections
    avg_wait_time_ms: f64,
    /// Pool utilization over time
    utilization_history: VecDeque<(Instant, f64)>,
    /// Response time percentiles
    response_time_p50: Duration,
    response_time_p95: Duration,
    response_time_p99: Duration,
    /// Error rates by type
    error_rates: HashMap<String, f64>,
    /// Last update timestamp
    last_updated: Instant,
}

impl Default for PoolMetrics {
    fn default() -> Self {
        Self {
            current_size: 0,
            peak_size: 0,
            total_requests: 0,
            avg_wait_time_ms: 0.0,
            utilization_history: VecDeque::new(),
            response_time_p50: Duration::ZERO,
            response_time_p95: Duration::ZERO,
            response_time_p99: Duration::ZERO,
            error_rates: HashMap::new(),
            last_updated: Instant::now(),
        }
    }
}

/// Adaptive sizing controller
#[derive(Debug, Clone)]
struct AdaptiveController {
    enabled: bool,
    target_response_time: Duration,
    last_adjustment: Instant,
    adjustment_cooldown: Duration,
    current_target_size: usize,
    response_time_samples: VecDeque<Duration>,
    utilization_samples: VecDeque<f64>,
}

impl Default for AdaptiveController {
    fn default() -> Self {
        Self {
            enabled: false,
            target_response_time: Duration::from_millis(100),
            last_adjustment: Instant::now(),
            adjustment_cooldown: Duration::from_secs(60),
            current_target_size: 1,
            response_time_samples: VecDeque::with_capacity(100),
            utilization_samples: VecDeque::with_capacity(100),
        }
    }
}

impl AdaptiveController {
    fn should_scale_up(
        &self,
        current_size: usize,
        avg_response_time: Duration,
        utilization: f64,
    ) -> bool {
        if !self.enabled || self.last_adjustment.elapsed() < self.adjustment_cooldown {
            return false;
        }

        avg_response_time > self.target_response_time && utilization > 0.8
    }

    fn should_scale_down(
        &self,
        current_size: usize,
        avg_response_time: Duration,
        utilization: f64,
    ) -> bool {
        if !self.enabled
            || self.last_adjustment.elapsed() < self.adjustment_cooldown
            || current_size <= 1
        {
            return false;
        }

        avg_response_time < self.target_response_time / 2 && utilization < 0.3
    }

    fn record_metrics(&mut self, response_time: Duration, utilization: f64) {
        self.response_time_samples.push_back(response_time);
        if self.response_time_samples.len() > 100 {
            self.response_time_samples.pop_front();
        }

        self.utilization_samples.push_back(utilization);
        if self.utilization_samples.len() > 100 {
            self.utilization_samples.pop_front();
        }
    }
}

impl<T: PooledConnection> ConnectionPool<T> {
    /// Create a new advanced connection pool with monitoring and circuit breaker
    pub async fn new(config: PoolConfig, factory: Arc<dyn ConnectionFactory<T>>) -> Result<Self> {
        // Initialize circuit breaker if enabled
        let circuit_breaker = if config.enable_circuit_breaker {
            Some(new_shared_circuit_breaker(
                config.circuit_breaker_config.clone().unwrap_or_default(),
            ))
        } else {
            None
        };

        // Initialize adaptive controller
        let mut adaptive_controller = AdaptiveController::default();
        adaptive_controller.enabled = config.adaptive_sizing;
        adaptive_controller.target_response_time =
            Duration::from_millis(config.target_response_time_ms);
        adaptive_controller.current_target_size = config.min_connections;

        // Initialize health monitor
        let health_check_config = HealthCheckConfig {
            check_interval: config.health_check_interval,
            check_timeout: config.validation_timeout,
            enable_statistics: config.enable_metrics,
            ..Default::default()
        };
        let health_monitor = Arc::new(HealthMonitor::new(health_check_config));

        // Initialize reconnection manager
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

        // Initialize minimum connections
        pool.ensure_min_connections().await?;

        // Start background maintenance task
        pool.start_maintenance_task().await;

        // Start adaptive sizing task if enabled
        if pool.config.adaptive_sizing {
            pool.start_adaptive_sizing_task().await;
        }

        // Start health monitoring
        pool.start_health_monitoring().await;

        info!(
            "Created advanced connection pool with health monitoring, automatic reconnection, and {} features",
            if pool.circuit_breaker.is_some() {
                "circuit breaker"
            } else {
                "standard"
            }
        );

        Ok(pool)
    }

    /// Get a connection from the pool with advanced load balancing and monitoring
    pub async fn get_connection(&self) -> Result<PooledConnectionHandle<T>> {
        let start_time = Instant::now();
        self.pending_requests.fetch_add(1, Ordering::Relaxed);

        // Check circuit breaker if enabled
        if let Some(cb) = &self.circuit_breaker {
            if !cb.can_execute().await {
                self.pending_requests.fetch_sub(1, Ordering::Relaxed);
                return Err(anyhow!(
                    "Circuit breaker is open - connection pool unavailable"
                ));
            }
        }

        // Acquire permit with timeout
        let permit = tokio::time::timeout(self.config.acquire_timeout, self.semaphore.acquire())
            .await
            .map_err(|_| anyhow!("Timeout acquiring connection from pool"))?
            .map_err(|_| anyhow!("Failed to acquire semaphore permit"))?;

        let connection = match self.try_get_existing_connection_with_lb().await {
            Some(conn) => {
                // Record successful circuit breaker operation
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

        // Update metrics
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

        // Apply load balancing strategy
        let selected_index = match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % connections.len()
            }
            LoadBalancingStrategy::Random => fastrand::usize(..connections.len()),
            LoadBalancingStrategy::LeastRecentlyUsed => {
                // Find connection with oldest last_activity
                connections
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, wrapper)| wrapper.last_activity)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
            LoadBalancingStrategy::LeastConnections => {
                // Find connection with lowest usage_count
                connections
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, wrapper)| wrapper.usage_count)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                // Select based on efficiency score
                connections
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.efficiency_score()
                            .partial_cmp(&b.efficiency_score())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
        };

        // Try to get the selected connection, falling back to linear search
        for attempt in 0..connections.len() {
            let index = (selected_index + attempt) % connections.len();

            if let Some(mut wrapper) = connections.remove(index) {
                if wrapper.is_expired(self.config.max_lifetime, self.config.idle_timeout) {
                    // Connection expired, destroy it
                    if let Err(e) = wrapper.connection.close().await {
                        warn!("Failed to close expired connection: {}", e);
                    }
                    self.stats.write().await.total_destroyed += 1;
                    continue;
                }

                // Validate connection health with timeout
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
                        // Connection unhealthy or timed out, destroy it
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
    async fn create_new_connection(&self) -> Result<T> {
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

        // Record metrics for adaptive sizing
        let mut controller = self.adaptive_controller.write().await;
        let utilization = (*active_count as f64) / (self.config.max_connections as f64);
        controller.record_metrics(execution_time, utilization);

        debug!(
            "Returned connection to pool with metrics: exec_time={:?}, success={}",
            execution_time, success
        );
    }

    /// Legacy method for backward compatibility
    async fn return_connection(&self, connection: T) {
        self.return_connection_with_metrics(connection, Duration::from_millis(100), true)
            .await;
    }

    /// Ensure minimum connections are available
    async fn ensure_min_connections(&self) -> Result<()> {
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

    /// Start background maintenance task with health monitoring
    async fn start_maintenance_task(&self) {
        let connections = self.connections.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();
        let health_monitor = self.health_monitor.clone();
        let reconnect_manager = self.reconnect_manager.clone();
        let connection_factory = self.connection_factory.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.health_check_interval);

            loop {
                interval.tick().await;

                let mut connections_guard = connections.lock().await;
                let mut to_remove = Vec::new();
                let mut to_reconnect = Vec::new();

                for (index, wrapper) in connections_guard.iter().enumerate() {
                    let conn_id = wrapper.connection_id.clone();

                    // Check if connection is expired
                    if wrapper.is_expired(config.max_lifetime, config.idle_timeout) {
                        to_remove.push(index);
                        health_monitor.unregister_connection(&conn_id).await;
                    } else {
                        // Perform health check via health monitor
                        let health_status = health_monitor
                            .check_connection_health(&conn_id, &wrapper.connection)
                            .await
                            .unwrap_or(HealthStatus::Unknown);

                        match health_status {
                            HealthStatus::Dead => {
                                to_remove.push(index);
                                health_monitor.unregister_connection(&conn_id).await;
                            }
                            HealthStatus::Unhealthy => {
                                to_reconnect.push((index, conn_id.clone()));
                            }
                            _ => {}
                        }
                    }
                }

                // Remove dead connections in reverse order
                for &index in to_remove.iter().rev() {
                    if let Some(mut wrapper) = connections_guard.remove(index) {
                        if let Err(e) = wrapper.connection.close().await {
                            warn!("Failed to close connection during maintenance: {}", e);
                        }
                        stats.write().await.total_destroyed += 1;
                    }
                }

                // Attempt to reconnect unhealthy connections
                for (index, conn_id) in to_reconnect {
                    if index < connections_guard.len() {
                        match reconnect_manager
                            .reconnect(conn_id.clone(), connection_factory.clone())
                            .await
                        {
                            Ok(new_conn) => {
                                let mut new_wrapper = PooledConnectionWrapper::new(new_conn);
                                new_wrapper.connection_id = conn_id.clone();

                                // Register new connection with health monitor
                                let mut metadata = HashMap::new();
                                metadata.insert("pool_id".to_string(), "main".to_string());
                                health_monitor.register_connection(conn_id, metadata).await;

                                connections_guard[index] = new_wrapper;
                                info!("Successfully reconnected connection {}", conn_id);
                            }
                            Err(e) => {
                                warn!("Failed to reconnect connection {}: {}", conn_id, e);
                                // Remove the failed connection
                                connections_guard.remove(index);
                                stats.write().await.total_destroyed += 1;
                            }
                        }
                    }
                }

                // Get dead connections from health monitor
                let dead_connections = health_monitor.get_dead_connections().await;
                if !dead_connections.is_empty() {
                    warn!(
                        "Health monitor detected {} dead connections",
                        dead_connections.len()
                    );
                }

                debug!(
                    "Pool maintenance completed, removed {} connections, attempted {} reconnections",
                    to_remove.len(),
                    to_reconnect.len()
                );
            }
        });
    }

    /// Start health monitoring for all connections
    async fn start_health_monitoring(&self) {
        // Register existing connections with health monitor
        let connections = self.connections.lock().await;
        for wrapper in connections.iter() {
            let mut metadata = HashMap::new();
            metadata.insert("pool_id".to_string(), "main".to_string());
            metadata.insert(
                "created_at".to_string(),
                wrapper.created_at.elapsed().as_secs().to_string(),
            );

            self.health_monitor
                .register_connection(wrapper.connection_id.clone(), metadata)
                .await;
        }

        // Subscribe to health events
        let mut health_events = self.health_monitor.subscribe();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            while let Ok(event) = health_events.recv().await {
                match event {
                    crate::health_monitor::HealthEvent::ConnectionDead {
                        connection_id,
                        reason,
                    } => {
                        error!("Connection {} marked as dead: {}", connection_id, reason);
                        stats.write().await.health_check_failures += 1;
                    }
                    crate::health_monitor::HealthEvent::ConnectionRecovered { connection_id } => {
                        info!("Connection {} recovered", connection_id);
                    }
                    crate::health_monitor::HealthEvent::StatusChanged {
                        connection_id,
                        old_status,
                        new_status,
                    } => {
                        debug!(
                            "Connection {} status changed from {:?} to {:?}",
                            connection_id, old_status, new_status
                        );
                    }
                    _ => {}
                }
            }
        });
    }

    /// Get comprehensive pool status with advanced metrics
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

    /// Calculate hash of current configuration for validation
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

    /// Update pool metrics for monitoring
    async fn update_metrics(&self, wait_time: Duration) {
        let mut metrics = self.metrics.write().await;

        metrics.total_requests += 1;
        let wait_time_ms = wait_time.as_millis() as f64;

        // Update average wait time with exponential moving average
        let alpha = 0.1;
        metrics.avg_wait_time_ms = alpha * wait_time_ms + (1.0 - alpha) * metrics.avg_wait_time_ms;

        // Update utilization history
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

    /// Start adaptive sizing background task
    async fn start_adaptive_sizing_task(&self) {
        let pool_metrics = self.metrics.clone();
        let adaptive_controller = self.adaptive_controller.clone();
        let pool_config = self.config.clone();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                let metrics = pool_metrics.read().await;
                let mut controller = adaptive_controller.write().await;

                if !controller.enabled {
                    continue;
                }

                let avg_response_time = Duration::from_millis(metrics.avg_wait_time_ms as u64);
                let current_utilization =
                    if let Some((_, util)) = metrics.utilization_history.back() {
                        *util
                    } else {
                        0.0
                    };

                let should_scale_up = controller.should_scale_up(
                    metrics.current_size,
                    avg_response_time,
                    current_utilization,
                );

                let should_scale_down = controller.should_scale_down(
                    metrics.current_size,
                    avg_response_time,
                    current_utilization,
                );

                if should_scale_up && metrics.current_size < pool_config.max_connections {
                    controller.current_target_size =
                        (controller.current_target_size + 1).min(pool_config.max_connections);
                    controller.last_adjustment = Instant::now();
                    stats.write().await.adaptive_scaling_events += 1;
                    info!(
                        "Adaptive scaling: scaling UP to {}",
                        controller.current_target_size
                    );
                } else if should_scale_down && metrics.current_size > pool_config.min_connections {
                    controller.current_target_size =
                        (controller.current_target_size.saturating_sub(1))
                            .max(pool_config.min_connections);
                    controller.last_adjustment = Instant::now();
                    stats.write().await.adaptive_scaling_events += 1;
                    info!(
                        "Adaptive scaling: scaling DOWN to {}",
                        controller.current_target_size
                    );
                }
            }
        });
    }
}

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
    fn new(
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

    /// Record an operation execution time for metrics
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

    /// Get operation statistics for this handle
    pub fn get_operation_stats(&self) -> (u32, u32, Duration) {
        let avg_time = if !self.execution_times.is_empty() {
            self.execution_times.iter().sum::<Duration>() / self.execution_times.len() as u32
        } else {
            Duration::ZERO
        };

        (self.operation_count, self.success_count, avg_time)
    }

    /// Get the total time this connection has been held
    pub fn held_duration(&self) -> Duration {
        self.acquired_at.elapsed()
    }

    /// Get reference to the connection
    pub fn as_ref(&self) -> Option<&T> {
        self.connection.as_ref()
    }

    /// Get mutable reference to the connection
    pub fn as_mut(&mut self) -> Option<&mut T> {
        self.connection.as_mut()
    }

    /// Take the connection out of the handle (won't be returned to pool)
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
            let metrics = self.metrics.clone();
            let adaptive_controller = self.adaptive_controller.clone();

            // Calculate metrics for this connection usage
            let total_held_time = self.acquired_at.elapsed();
            let avg_execution_time = if !self.execution_times.is_empty() {
                self.execution_times.iter().sum::<Duration>() / self.execution_times.len() as u32
            } else {
                Duration::from_millis(50) // Default assumption
            };

            let success_rate = if self.operation_count > 0 {
                self.success_count as f64 / self.operation_count as f64
            } else {
                1.0 // Assume success if no operations recorded
            };

            let overall_success = success_rate > 0.8; // Consider successful if >80% operations succeeded

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

                // Update pool statistics
                stats.write().await.total_returned += 1;

                // Update adaptive controller metrics
                let utilization = (*active as f64) / 10.0; // Simplified calculation
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

/// Helper for creating connection pools from stream config
impl<T: PooledConnection> ConnectionPool<T> {
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

    /// Get detailed pool metrics for monitoring
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
        // Create primary and secondary endpoints
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

        // Create failover manager
        let failover_manager = Arc::new(
            FailoverManager::new(failover_config, primary_endpoint, secondary_endpoint).await?,
        );

        // Create pool with primary factory
        let mut pool = Self::new(config, primary_factory).await?;
        pool.failover_manager = Some(failover_manager.clone());

        // Subscribe to failover events
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

/// Detailed pool metrics for comprehensive monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedPoolMetrics {
    pub status: PoolStatus,
    pub total_requests: u64,
    pub peak_size: usize,
    pub avg_wait_time_ms: f64,
    #[serde(skip)]
    pub response_time_p50: Duration,
    #[serde(skip)]
    pub response_time_p95: Duration,
    #[serde(skip)]
    pub response_time_p99: Duration,
    pub adaptive_scaling_events: u64,
    pub circuit_breaker_failures: u64,
    pub load_balancing_decisions: u64,
    pub current_target_size: usize,
    #[serde(skip)]
    pub pool_uptime: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug)]
    struct TestConnection {
        id: u32,
        created_at: Instant,
        last_activity: Instant,
        is_healthy: Arc<AtomicBool>,
        is_closed: bool,
    }

    impl TestConnection {
        fn new(id: u32) -> Self {
            let now = Instant::now();
            Self {
                id,
                created_at: now,
                last_activity: now,
                is_healthy: Arc::new(AtomicBool::new(true)),
                is_closed: false,
            }
        }
    }

    #[async_trait::async_trait]
    impl PooledConnection for TestConnection {
        async fn is_healthy(&self) -> bool {
            !self.is_closed && self.is_healthy.load(Ordering::Relaxed)
        }

        async fn close(&mut self) -> Result<()> {
            self.is_closed = true;
            Ok(())
        }

        fn created_at(&self) -> Instant {
            self.created_at
        }

        fn last_activity(&self) -> Instant {
            self.last_activity
        }

        fn update_activity(&mut self) {
            self.last_activity = Instant::now();
        }
    }

    struct TestConnectionFactory {
        counter: Arc<Mutex<u32>>,
    }

    impl TestConnectionFactory {
        fn new() -> Self {
            Self {
                counter: Arc::new(Mutex::new(0)),
            }
        }
    }

    #[async_trait::async_trait]
    impl ConnectionFactory<TestConnection> for TestConnectionFactory {
        async fn create_connection(&self) -> Result<TestConnection> {
            let mut counter = self.counter.lock().await;
            *counter += 1;
            Ok(TestConnection::new(*counter))
        }
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 5,
            ..Default::default()
        };

        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();

        let status = pool.status().await;
        assert_eq!(status.idle_connections, 2);
        assert_eq!(status.active_connections, 0);
    }

    #[tokio::test]
    async fn test_connection_borrowing() {
        let config = PoolConfig {
            min_connections: 1,
            max_connections: 3,
            ..Default::default()
        };

        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();

        let mut handle = pool.get_connection().await.unwrap();

        let status = pool.status().await;
        assert_eq!(status.active_connections, 1);
        assert_eq!(status.idle_connections, 0);
        assert!(status.is_healthy);

        // Record some operations
        handle.record_operation(Duration::from_millis(50), true);
        handle.record_operation(Duration::from_millis(75), true);

        let (ops, successes, avg_time) = handle.get_operation_stats();
        assert_eq!(ops, 2);
        assert_eq!(successes, 2);
        assert!(avg_time > Duration::ZERO);

        drop(handle);

        // Wait for the connection to be returned
        tokio::time::sleep(Duration::from_millis(10)).await;

        let status = pool.status().await;
        assert_eq!(status.active_connections, 0);
        assert_eq!(status.idle_connections, 1);
    }

    #[tokio::test]
    async fn test_load_balancing_strategies() {
        for strategy in [
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::Random,
            LoadBalancingStrategy::LeastRecentlyUsed,
            LoadBalancingStrategy::LeastConnections,
            LoadBalancingStrategy::WeightedRoundRobin,
        ] {
            let config = PoolConfig {
                min_connections: 3,
                max_connections: 5,
                load_balancing: strategy.clone(),
                ..Default::default()
            };

            let factory = Arc::new(TestConnectionFactory::new());
            let pool = ConnectionPool::new(config, factory).await.unwrap();

            // Get multiple connections to test load balancing
            let handles: Vec<_> =
                futures_util::future::join_all((0..3).map(|_| pool.get_connection()))
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();

            let status = pool.status().await;
            assert_eq!(status.active_connections, 3);
            assert_eq!(status.load_balancing_strategy, strategy);

            drop(handles);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker_integration() {
        let config = PoolConfig {
            min_connections: 1,
            max_connections: 2,
            enable_circuit_breaker: true,
            circuit_breaker_config: Some(CircuitBreakerConfig {
                failure_threshold: 2,
                timeout: Duration::from_millis(50),
                ..Default::default()
            }),
            ..Default::default()
        };

        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();

        // Should work normally initially
        let handle = pool.get_connection().await;
        assert!(handle.is_ok());
        drop(handle);

        // Test that circuit breaker status is included in pool status
        let status = pool.status().await;
        assert!(!status.circuit_breaker_open);
    }

    #[tokio::test]
    async fn test_adaptive_sizing() {
        let config = PoolConfig {
            min_connections: 1,
            max_connections: 5,
            adaptive_sizing: true,
            target_response_time_ms: 50,
            ..Default::default()
        };

        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();

        let metrics = pool.get_detailed_metrics().await;
        assert_eq!(metrics.current_target_size, 1); // Should start at min_connections
        assert!(metrics.adaptive_scaling_events == 0);

        // Test pool resizing
        pool.resize(3).await.unwrap();
        let metrics = pool.get_detailed_metrics().await;
        assert_eq!(metrics.current_target_size, 3);
    }

    #[tokio::test]
    async fn test_detailed_metrics() {
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 4,
            enable_metrics: true,
            ..Default::default()
        };

        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();

        // Generate some activity
        let handles: Vec<_> = futures_util::future::join_all((0..3).map(|_| pool.get_connection()))
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let metrics = pool.get_detailed_metrics().await;
        assert!(metrics.total_requests >= 3);
        assert!(metrics.status.utilization_percent > 0.0);
        assert!(metrics.pool_uptime > Duration::ZERO);
        assert_eq!(metrics.status.active_connections, 3);

        drop(handles);

        // Test statistics reset
        pool.reset_statistics().await;
        let metrics = pool.get_detailed_metrics().await;
        assert_eq!(metrics.adaptive_scaling_events, 0);
    }
}
