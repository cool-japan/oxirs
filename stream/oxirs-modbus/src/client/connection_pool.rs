//! Connection pooling for Modbus clients
//!
//! Provides efficient connection management for multiple Modbus devices
//! with automatic reconnection and health monitoring.

use crate::error::{ModbusError, ModbusResult};
use crate::protocol::ModbusTcpClient;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::timeout;
use tracing::{debug, warn};

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections per device
    pub max_connections: usize,
    /// Minimum number of idle connections
    pub min_idle: usize,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Maximum time to wait for a connection from pool (milliseconds)
    pub acquire_timeout_ms: u64,
    /// Enable automatic reconnection on failure
    pub auto_reconnect: bool,
    /// Connection health check interval (seconds)
    pub health_check_interval_secs: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_idle: 2,
            connect_timeout_ms: 5000,
            acquire_timeout_ms: 30000,
            auto_reconnect: true,
            health_check_interval_secs: 60,
        }
    }
}

/// Connection wrapper with health tracking
struct PooledConnection {
    /// Underlying Modbus client
    client: ModbusTcpClient,
    /// Connection creation timestamp
    created_at: std::time::Instant,
    /// Last successful operation timestamp
    last_used: std::time::Instant,
    /// Number of successful operations
    successful_ops: u64,
    /// Number of failed operations
    failed_ops: u64,
}

impl PooledConnection {
    fn new(client: ModbusTcpClient) -> Self {
        let now = std::time::Instant::now();
        Self {
            client,
            created_at: now,
            last_used: now,
            successful_ops: 0,
            failed_ops: 0,
        }
    }

    fn record_success(&mut self) {
        self.last_used = std::time::Instant::now();
        self.successful_ops += 1;
    }

    fn record_failure(&mut self) {
        self.last_used = std::time::Instant::now();
        self.failed_ops += 1;
    }

    fn is_healthy(&self) -> bool {
        // Consider unhealthy if >50% failure rate
        let total_ops = self.successful_ops + self.failed_ops;
        if total_ops > 10 {
            let failure_rate = self.failed_ops as f64 / total_ops as f64;
            failure_rate < 0.5
        } else {
            true // Not enough data yet
        }
    }
}

/// Modbus connection pool
///
/// Manages a pool of Modbus TCP connections to a single device.
pub struct ModbusConnectionPool {
    /// Device address (host:port)
    address: String,
    /// Unit ID for Modbus requests
    unit_id: u8,
    /// Pool configuration
    config: PoolConfig,
    /// Available connections
    connections: Arc<Mutex<Vec<PooledConnection>>>,
    /// Semaphore to limit total connections
    semaphore: Arc<Semaphore>,
    /// Pool statistics
    stats: Arc<Mutex<PoolStats>>,
}

/// Pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total connections created
    pub total_created: u64,
    /// Total connections destroyed
    pub total_destroyed: u64,
    /// Total acquire operations
    pub total_acquires: u64,
    /// Total acquire timeouts
    pub acquire_timeouts: u64,
    /// Total connection errors
    pub connection_errors: u64,
}

impl ModbusConnectionPool {
    /// Create a new connection pool
    pub fn new(address: impl Into<String>, unit_id: u8, config: PoolConfig) -> Self {
        let max_connections = config.max_connections;

        Self {
            address: address.into(),
            unit_id,
            config,
            connections: Arc::new(Mutex::new(Vec::new())),
            semaphore: Arc::new(Semaphore::new(max_connections)),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Create pool with default configuration
    pub fn with_defaults(address: impl Into<String>, unit_id: u8) -> Self {
        Self::new(address, unit_id, PoolConfig::default())
    }

    /// Acquire a connection from the pool
    ///
    /// If no idle connection is available, creates a new one up to max_connections.
    pub async fn acquire(&self) -> ModbusResult<PooledModbusClient> {
        let acquire_timeout = Duration::from_millis(self.config.acquire_timeout_ms);

        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.total_acquires += 1;
        }

        // Try to acquire permit with timeout
        let permit = match timeout(acquire_timeout, self.semaphore.acquire()).await {
            Ok(Ok(p)) => p,
            Ok(Err(_)) => {
                let mut stats = self.stats.lock().await;
                stats.acquire_timeouts += 1;
                return Err(ModbusError::Timeout(acquire_timeout));
            }
            Err(_) => {
                let mut stats = self.stats.lock().await;
                stats.acquire_timeouts += 1;
                return Err(ModbusError::Timeout(acquire_timeout));
            }
        };

        // Try to get existing connection
        {
            let mut connections = self.connections.lock().await;
            if let Some(conn) = connections.pop() {
                debug!("Reusing existing connection from pool");
                permit.forget(); // Release permit since we're using existing connection
                return Ok(PooledModbusClient::new(
                    conn,
                    self.connections.clone(),
                    self.semaphore.clone(),
                ));
            }
        }

        // No idle connection, create new one
        debug!(address = %self.address, "Creating new pool connection");

        let connect_timeout = Duration::from_millis(self.config.connect_timeout_ms);
        let client = match timeout(
            connect_timeout,
            ModbusTcpClient::connect(&self.address, self.unit_id),
        )
        .await
        {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                let mut stats = self.stats.lock().await;
                stats.connection_errors += 1;
                permit.forget();
                return Err(e);
            }
            Err(_) => {
                let mut stats = self.stats.lock().await;
                stats.connection_errors += 1;
                permit.forget();
                return Err(ModbusError::Timeout(connect_timeout));
            }
        };

        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.total_created += 1;
        }

        let pooled_conn = PooledConnection::new(client);

        Ok(PooledModbusClient::new(
            pooled_conn,
            self.connections.clone(),
            self.semaphore.clone(),
        ))
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        self.stats.lock().await.clone()
    }

    /// Get number of idle connections
    pub async fn idle_count(&self) -> usize {
        self.connections.lock().await.len()
    }

    /// Get number of active connections
    pub async fn active_count(&self) -> usize {
        self.config.max_connections - self.semaphore.available_permits()
    }

    /// Prune unhealthy connections
    pub async fn prune_unhealthy(&self) -> usize {
        let mut connections = self.connections.lock().await;
        let original_len = connections.len();

        connections.retain(|conn| {
            let healthy = conn.is_healthy();
            if !healthy {
                warn!("Removing unhealthy connection from pool");
            }
            healthy
        });

        let removed = original_len - connections.len();
        if removed > 0 {
            let mut stats = self.stats.lock().await;
            stats.total_destroyed += removed as u64;
        }

        removed
    }
}

/// Pooled Modbus client with automatic return to pool
pub struct PooledModbusClient {
    /// Wrapped connection
    connection: Option<PooledConnection>,
    /// Pool to return connection to
    pool: Arc<Mutex<Vec<PooledConnection>>>,
    /// Semaphore to release permit
    semaphore: Arc<Semaphore>,
}

impl PooledModbusClient {
    fn new(
        connection: PooledConnection,
        pool: Arc<Mutex<Vec<PooledConnection>>>,
        semaphore: Arc<Semaphore>,
    ) -> Self {
        Self {
            connection: Some(connection),
            pool,
            semaphore,
        }
    }

    /// Read holding registers
    pub async fn read_holding_registers(
        &mut self,
        start_addr: u16,
        count: u16,
    ) -> ModbusResult<Vec<u16>> {
        let conn = self
            .connection
            .as_mut()
            .ok_or_else(|| ModbusError::Config("Connection already returned".into()))?;

        match conn.client.read_holding_registers(start_addr, count).await {
            Ok(regs) => {
                conn.record_success();
                Ok(regs)
            }
            Err(e) => {
                conn.record_failure();
                Err(e)
            }
        }
    }

    /// Read input registers
    pub async fn read_input_registers(
        &mut self,
        start_addr: u16,
        count: u16,
    ) -> ModbusResult<Vec<u16>> {
        let conn = self
            .connection
            .as_mut()
            .ok_or_else(|| ModbusError::Config("Connection already returned".into()))?;

        match conn.client.read_input_registers(start_addr, count).await {
            Ok(regs) => {
                conn.record_success();
                Ok(regs)
            }
            Err(e) => {
                conn.record_failure();
                Err(e)
            }
        }
    }

    /// Write single register
    pub async fn write_single_register(&mut self, addr: u16, value: u16) -> ModbusResult<()> {
        let conn = self
            .connection
            .as_mut()
            .ok_or_else(|| ModbusError::Config("Connection already returned".into()))?;

        match conn.client.write_single_register(addr, value).await {
            Ok(()) => {
                conn.record_success();
                Ok(())
            }
            Err(e) => {
                conn.record_failure();
                Err(e)
            }
        }
    }

    /// Get connection statistics
    pub fn connection_stats(&self) -> Option<ConnectionStats> {
        self.connection.as_ref().map(|conn| ConnectionStats {
            age_secs: conn.created_at.elapsed().as_secs(),
            successful_ops: conn.successful_ops,
            failed_ops: conn.failed_ops,
            failure_rate: if conn.successful_ops + conn.failed_ops > 0 {
                conn.failed_ops as f64 / (conn.successful_ops + conn.failed_ops) as f64
            } else {
                0.0
            },
        })
    }
}

/// Connection statistics
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    /// Connection age in seconds
    pub age_secs: u64,
    /// Number of successful operations
    pub successful_ops: u64,
    /// Number of failed operations
    pub failed_ops: u64,
    /// Failure rate (0.0 to 1.0)
    pub failure_rate: f64,
}

impl Drop for PooledModbusClient {
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            // Return connection to pool
            let pool = self.pool.clone();
            let semaphore = self.semaphore.clone();

            tokio::spawn(async move {
                if conn.is_healthy() {
                    let mut connections = pool.lock().await;
                    connections.push(conn);
                    debug!("Returned connection to pool");
                } else {
                    warn!("Discarding unhealthy connection");
                }
                // Release semaphore permit
                semaphore.add_permits(1);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_connections, 10);
        assert_eq!(config.min_idle, 2);
        assert!(config.auto_reconnect);
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let pool = ModbusConnectionPool::with_defaults("127.0.0.1:502", 1);
        assert_eq!(pool.idle_count().await, 0);
        assert_eq!(pool.active_count().await, 0);
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let pool = ModbusConnectionPool::with_defaults("127.0.0.1:502", 1);
        let stats = pool.stats().await;
        assert_eq!(stats.total_created, 0);
        assert_eq!(stats.total_acquires, 0);
    }

    #[tokio::test]
    async fn test_connection_health() {
        // Mock connection for testing
        let address = "127.0.0.1:502".to_string();
        let client = match ModbusTcpClient::connect(&address, 1).await {
            Ok(c) => c,
            Err(_) => return, // Skip if cannot connect (expected in CI)
        };

        let mut conn = PooledConnection::new(client);
        assert!(conn.is_healthy());

        // Record some operations
        conn.record_success();
        conn.record_success();
        assert!(conn.is_healthy());

        // Record many failures
        for _ in 0..20 {
            conn.record_failure();
        }
        assert!(!conn.is_healthy());
    }
}
