//! Connection Pool — Type definitions
//!
//! Pool configuration, connection state types, health check types,
//! load balancing strategies, and pool status/metrics structures.

use crate::circuit_breaker::CircuitBreakerConfig;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

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
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(1800),
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Pool statistics with enhanced metrics
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    pub total_created: u64,
    pub total_destroyed: u64,
    pub total_borrowed: u64,
    pub total_returned: u64,
    pub creation_failures: u64,
    pub health_check_failures: u64,
    pub timeouts: u64,
    pub circuit_breaker_failures: u64,
    pub adaptive_scaling_events: u64,
    pub load_balancing_decisions: u64,
    pub failover_count: u64,
}

/// Comprehensive pool metrics for monitoring (crate-internal)
#[derive(Debug, Clone)]
pub(crate) struct PoolMetrics {
    pub(crate) current_size: usize,
    pub(crate) peak_size: usize,
    pub(crate) total_requests: u64,
    pub(crate) avg_wait_time_ms: f64,
    pub(crate) utilization_history: VecDeque<(Instant, f64)>,
    pub(crate) response_time_p50: Duration,
    pub(crate) response_time_p95: Duration,
    pub(crate) response_time_p99: Duration,
    pub(crate) error_rates: HashMap<String, f64>,
    pub(crate) last_updated: Instant,
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

/// Adaptive sizing controller (crate-internal)
#[derive(Debug, Clone)]
pub(crate) struct AdaptiveController {
    pub(crate) enabled: bool,
    pub(crate) target_response_time: Duration,
    pub(crate) last_adjustment: Instant,
    pub(crate) adjustment_cooldown: Duration,
    pub(crate) current_target_size: usize,
    pub(crate) response_time_samples: VecDeque<Duration>,
    pub(crate) utilization_samples: VecDeque<f64>,
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
    pub(crate) fn should_scale_up(
        &self,
        _current_size: usize,
        avg_response_time: Duration,
        utilization: f64,
    ) -> bool {
        if !self.enabled || self.last_adjustment.elapsed() < self.adjustment_cooldown {
            return false;
        }
        avg_response_time > self.target_response_time && utilization > 0.8
    }

    pub(crate) fn should_scale_down(
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

    pub(crate) fn record_metrics(&mut self, response_time: Duration, utilization: f64) {
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

/// Generic connection trait
#[async_trait::async_trait]
pub trait PooledConnection: Send + Sync + 'static {
    async fn is_healthy(&self) -> bool;
    async fn close(&mut self) -> anyhow::Result<()>;
    fn created_at(&self) -> Instant;
    fn last_activity(&self) -> Instant;
    fn update_activity(&mut self);
    fn clone_connection(&self) -> Box<dyn PooledConnection>;
}

/// Implement PooledConnection for `Box<dyn PooledConnection>`
#[async_trait::async_trait]
impl PooledConnection for Box<dyn PooledConnection> {
    async fn is_healthy(&self) -> bool {
        self.as_ref().is_healthy().await
    }
    async fn close(&mut self) -> anyhow::Result<()> {
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
    fn clone_connection(&self) -> Box<dyn PooledConnection> {
        self.as_ref().clone_connection()
    }
}

/// Connection factory trait
#[async_trait::async_trait]
pub trait ConnectionFactory<T: PooledConnection + Clone>: Send + Sync {
    async fn create_connection(&self) -> anyhow::Result<T>;
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

/// Connection wrapper with comprehensive metadata and monitoring (crate-internal)
pub(crate) struct PooledConnectionWrapper<T: PooledConnection> {
    pub(crate) connection: T,
    pub(crate) created_at: Instant,
    pub(crate) last_activity: Instant,
    pub(crate) is_in_use: bool,
    pub(crate) connection_id: String,
    pub(crate) usage_count: u64,
    pub(crate) total_execution_time: Duration,
    pub(crate) avg_response_time: Duration,
    pub(crate) failure_count: u32,
    pub(crate) last_health_check: Option<(Instant, bool)>,
    pub(crate) weight: f64,
}

impl<T: PooledConnection> PooledConnectionWrapper<T> {
    pub(crate) fn new(connection: T) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_activity: now,
            is_in_use: false,
            connection_id: uuid::Uuid::new_v4().to_string(),
            usage_count: 0,
            total_execution_time: Duration::ZERO,
            avg_response_time: Duration::from_millis(50),
            failure_count: 0,
            last_health_check: None,
            weight: 1.0,
        }
    }

    pub(crate) fn record_usage(&mut self, execution_time: Duration, success: bool) {
        self.usage_count += 1;
        self.last_activity = Instant::now();
        self.total_execution_time += execution_time;

        let alpha = 0.1;
        let new_time_ms = execution_time.as_millis() as f64;
        let current_avg_ms = self.avg_response_time.as_millis() as f64;
        let updated_avg_ms = alpha * new_time_ms + (1.0 - alpha) * current_avg_ms;
        self.avg_response_time = Duration::from_millis(updated_avg_ms as u64);

        if !success {
            self.failure_count += 1;
            self.weight = (self.weight * 0.9).max(0.1);
        } else if self.failure_count > 0 {
            self.weight = (self.weight * 1.01).min(1.0);
        }
    }

    pub(crate) fn efficiency_score(&self) -> f64 {
        if self.usage_count == 0 {
            return 1.0;
        }
        let failure_rate = self.failure_count as f64 / self.usage_count as f64;
        let response_time_penalty = (self.avg_response_time.as_millis() as f64).ln() / 10.0;
        (1.0 - failure_rate) * self.weight / (1.0 + response_time_penalty)
    }

    pub(crate) fn is_expired(&self, max_lifetime: Duration, idle_timeout: Duration) -> bool {
        let now = Instant::now();
        now.duration_since(self.created_at) > max_lifetime
            || (!self.is_in_use && now.duration_since(self.last_activity) > idle_timeout)
    }

    pub(crate) async fn is_healthy(&self) -> bool {
        self.connection.is_healthy().await
    }
}
