//! Connection Pooling Optimization
//!
//! This module provides high-performance connection pooling for:
//! - Database connections with adaptive sizing
//! - HTTP client connection reuse
//! - Connection health monitoring
//! - Automatic connection recycling
//! - Pool statistics and monitoring

use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn};

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of connections
    pub min_size: usize,
    /// Maximum number of connections
    pub max_size: usize,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
    /// Connection lifetime in seconds (for recycling)
    pub max_lifetime_secs: u64,
    /// Idle timeout in seconds
    pub idle_timeout_secs: u64,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Enable adaptive pool sizing
    pub enable_adaptive_sizing: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            min_size: 5,
            max_size: 100,
            connection_timeout_secs: 30,
            max_lifetime_secs: 3600, // 1 hour
            idle_timeout_secs: 600,  // 10 minutes
            health_check_interval_secs: 60,
            enable_adaptive_sizing: true,
        }
    }
}

/// Connection wrapper with metadata
pub struct PooledConnection<T> {
    /// The actual connection
    pub connection: T,
    /// Connection ID
    pub id: String,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last used timestamp
    pub last_used_at: DateTime<Utc>,
    /// Usage count
    pub usage_count: u64,
    /// Is healthy
    pub is_healthy: bool,
}

impl<T> PooledConnection<T> {
    /// Create new pooled connection
    pub fn new(connection: T, id: String) -> Self {
        let now = Utc::now();
        PooledConnection {
            connection,
            id,
            created_at: now,
            last_used_at: now,
            usage_count: 0,
            is_healthy: true,
        }
    }

    /// Check if connection should be recycled
    pub fn should_recycle(&self, max_lifetime: Duration, idle_timeout: Duration) -> bool {
        let now = Utc::now();
        let age = now - self.created_at;
        let idle_time = now - self.last_used_at;

        age > max_lifetime || idle_time > idle_timeout || !self.is_healthy
    }

    /// Mark as used
    pub fn mark_used(&mut self) {
        self.last_used_at = Utc::now();
        self.usage_count += 1;
    }
}

/// Generic connection pool
pub struct ConnectionPool<T: Send + 'static> {
    /// Available connections
    available: Arc<RwLock<VecDeque<PooledConnection<T>>>>,
    /// Active connections (being used)
    active: Arc<DashMap<String, DateTime<Utc>>>,
    /// Pool configuration
    config: PoolConfig,
    /// Semaphore for max connections limit
    semaphore: Arc<Semaphore>,
    /// Connection factory
    factory: Arc<dyn Fn() -> FusekiResult<T> + Send + Sync>,
    /// Pool statistics
    stats: PoolStats,
}

/// Pool statistics
struct PoolStats {
    total_created: AtomicU64,
    total_recycled: AtomicU64,
    total_borrowed: AtomicU64,
    total_returned: AtomicU64,
    total_timeouts: AtomicU64,
    total_errors: AtomicU64,
}

impl PoolStats {
    fn new() -> Self {
        PoolStats {
            total_created: AtomicU64::new(0),
            total_recycled: AtomicU64::new(0),
            total_borrowed: AtomicU64::new(0),
            total_returned: AtomicU64::new(0),
            total_timeouts: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
        }
    }
}

/// Pool statistics snapshot
#[derive(Debug, Clone, Serialize)]
pub struct PoolStatsSnapshot {
    pub available_connections: usize,
    pub active_connections: usize,
    pub total_created: u64,
    pub total_recycled: u64,
    pub total_borrowed: u64,
    pub total_returned: u64,
    pub total_timeouts: u64,
    pub total_errors: u64,
    pub pool_utilization: f64,
}

impl<T: Send + Sync + 'static> ConnectionPool<T> {
    /// Create new connection pool
    pub async fn new<F>(config: PoolConfig, factory: F) -> FusekiResult<Arc<Self>>
    where
        F: Fn() -> FusekiResult<T> + Send + Sync + 'static,
    {
        let pool = Arc::new(ConnectionPool {
            available: Arc::new(RwLock::new(VecDeque::new())),
            active: Arc::new(DashMap::new()),
            semaphore: Arc::new(Semaphore::new(config.max_size)),
            config: config.clone(),
            factory: Arc::new(factory),
            stats: PoolStats::new(),
        });

        // Pre-populate with minimum connections
        pool.ensure_minimum_connections().await?;

        info!(
            "Created connection pool (min: {}, max: {})",
            config.min_size, config.max_size
        );

        Ok(pool)
    }

    /// Borrow a connection from the pool
    pub async fn acquire(&self) -> FusekiResult<PooledConnection<T>> {
        // Wait for available slot (with timeout)
        let permit = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.connection_timeout_secs),
            self.semaphore.acquire(),
        )
        .await
        .map_err(|_| {
            self.stats.total_timeouts.fetch_add(1, Ordering::Relaxed);
            FusekiError::service_unavailable("Connection pool timeout")
        })?
        .map_err(|_| FusekiError::service_unavailable("Connection pool closed"))?;

        permit.forget(); // We'll manually release when connection is returned

        self.stats.total_borrowed.fetch_add(1, Ordering::Relaxed);

        // Try to get an existing connection
        loop {
            let conn_opt = {
                let mut available = self.available.write().await;
                available.pop_front()
            };

            if let Some(mut c) = conn_opt {
                // Check if connection should be recycled
                if c.should_recycle(
                    Duration::seconds(self.config.max_lifetime_secs as i64),
                    Duration::seconds(self.config.idle_timeout_secs as i64),
                ) {
                    debug!("Recycling old connection: {}", c.id);
                    self.stats.total_recycled.fetch_add(1, Ordering::Relaxed);
                    continue; // Drop this connection and try next
                }

                // Mark as active
                c.mark_used();
                self.active.insert(c.id.clone(), Utc::now());

                debug!("Acquired connection: {}", c.id);
                return Ok(c);
            } else {
                // No available connections, create a new one
                break;
            }
        }

        // Create new connection
        let conn = self.create_connection().await?;

        Ok(conn)
    }

    /// Return a connection to the pool
    pub async fn release(&self, mut conn: PooledConnection<T>) {
        self.stats.total_returned.fetch_add(1, Ordering::Relaxed);

        // Remove from active
        self.active.remove(&conn.id);

        // Check if connection should be recycled
        if conn.should_recycle(
            Duration::seconds(self.config.max_lifetime_secs as i64),
            Duration::seconds(self.config.idle_timeout_secs as i64),
        ) {
            debug!(
                "Not returning connection to pool (needs recycling): {}",
                conn.id
            );
            self.stats.total_recycled.fetch_add(1, Ordering::Relaxed);
        } else {
            // Return to pool
            conn.last_used_at = Utc::now();
            let mut available = self.available.write().await;
            available.push_back(conn);
        }

        // Release semaphore permit
        self.semaphore.add_permits(1);
    }

    /// Create a new connection
    async fn create_connection(&self) -> FusekiResult<PooledConnection<T>> {
        let connection = (self.factory)().map_err(|e| {
            self.stats.total_errors.fetch_add(1, Ordering::Relaxed);
            e
        })?;

        let id = uuid::Uuid::new_v4().to_string();
        let mut pooled = PooledConnection::new(connection, id.clone());

        self.active.insert(id.clone(), Utc::now());
        self.stats.total_created.fetch_add(1, Ordering::Relaxed);

        pooled.mark_used();

        debug!("Created new connection: {}", id);

        Ok(pooled)
    }

    /// Ensure minimum number of connections
    async fn ensure_minimum_connections(&self) -> FusekiResult<()> {
        let current = self.get_total_connections();

        if current < self.config.min_size {
            let needed = self.config.min_size - current;
            debug!("Pre-populating pool with {} connections", needed);

            for _ in 0..needed {
                let connection = (self.factory)()?;
                let id = uuid::Uuid::new_v4().to_string();
                let pooled = PooledConnection::new(connection, id);

                self.stats.total_created.fetch_add(1, Ordering::Relaxed);

                let mut available = self.available.write().await;
                available.push_back(pooled);
            }
        }

        Ok(())
    }

    /// Get total number of connections (available + active)
    pub fn get_total_connections(&self) -> usize {
        self.config.max_size - self.semaphore.available_permits()
    }

    /// Get pool statistics
    pub async fn get_stats(&self) -> PoolStatsSnapshot {
        let available = self.available.read().await.len();
        let active = self.active.len();
        let total = self.get_total_connections();

        let utilization = if self.config.max_size > 0 {
            (total as f64 / self.config.max_size as f64) * 100.0
        } else {
            0.0
        };

        PoolStatsSnapshot {
            available_connections: available,
            active_connections: active,
            total_created: self.stats.total_created.load(Ordering::Relaxed),
            total_recycled: self.stats.total_recycled.load(Ordering::Relaxed),
            total_borrowed: self.stats.total_borrowed.load(Ordering::Relaxed),
            total_returned: self.stats.total_returned.load(Ordering::Relaxed),
            total_timeouts: self.stats.total_timeouts.load(Ordering::Relaxed),
            total_errors: self.stats.total_errors.load(Ordering::Relaxed),
            pool_utilization: utilization,
        }
    }

    /// Cleanup old connections
    pub async fn cleanup(&self) {
        let max_lifetime = Duration::seconds(self.config.max_lifetime_secs as i64);
        let idle_timeout = Duration::seconds(self.config.idle_timeout_secs as i64);

        let mut available = self.available.write().await;
        let before = available.len();

        available.retain(|conn| !conn.should_recycle(max_lifetime, idle_timeout));

        let recycled = before - available.len();

        if recycled > 0 {
            self.stats
                .total_recycled
                .fetch_add(recycled as u64, Ordering::Relaxed);
            debug!("Recycled {} old connections", recycled);
        }

        // Ensure minimum connections
        drop(available); // Release lock
        if let Err(e) = self.ensure_minimum_connections().await {
            warn!("Failed to ensure minimum connections: {}", e);
        }
    }

    /// Start background maintenance task
    pub fn start_maintenance_task(self: Arc<Self>) {
        let pool = Arc::clone(&self);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(
                pool.config.health_check_interval_secs,
            ));

            loop {
                interval.tick().await;
                pool.cleanup().await;

                let stats = pool.get_stats().await;
                debug!(
                    "Pool stats: available={}, active={}, utilization={:.1}%",
                    stats.available_connections, stats.active_connections, stats.pool_utilization
                );
            }
        });

        info!(
            "Started pool maintenance task (interval: {}s)",
            self.config.health_check_interval_secs
        );
    }
}

/// HTTP client pool for reusing HTTP connections
pub struct HttpClientPool {
    /// Reusable HTTP client with connection pooling
    client: reqwest::Client,
    /// Request statistics
    total_requests: Arc<AtomicU64>,
    total_errors: Arc<AtomicU64>,
}

impl HttpClientPool {
    /// Create new HTTP client pool
    pub fn new(max_idle_per_host: usize, timeout_secs: u64) -> FusekiResult<Self> {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(max_idle_per_host)
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .tcp_keepalive(Some(std::time::Duration::from_secs(60)))
            .http2_keep_alive_interval(Some(std::time::Duration::from_secs(30)))
            .build()
            .map_err(|e| {
                FusekiError::configuration(format!("Failed to create HTTP client: {}", e))
            })?;

        Ok(HttpClientPool {
            client,
            total_requests: Arc::new(AtomicU64::new(0)),
            total_errors: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Get the HTTP client
    pub fn client(&self) -> &reqwest::Client {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        &self.client
    }

    /// Record an error
    pub fn record_error(&self) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn get_stats(&self) -> (u64, u64) {
        (
            self.total_requests.load(Ordering::Relaxed),
            self.total_errors.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test connection type
    struct TestConnection {
        id: usize,
    }

    fn create_test_factory(counter: Arc<AtomicUsize>) -> impl Fn() -> FusekiResult<TestConnection> {
        move || {
            let id = counter.fetch_add(1, Ordering::Relaxed);
            Ok(TestConnection { id })
        }
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let counter = Arc::new(AtomicUsize::new(0));
        let factory = create_test_factory(counter.clone());

        let config = PoolConfig {
            min_size: 2,
            max_size: 10,
            ..Default::default()
        };

        let pool = ConnectionPool::new(config, factory).await.unwrap();

        let stats = pool.get_stats().await;
        assert_eq!(stats.available_connections, 2); // Min size
    }

    #[tokio::test]
    async fn test_acquire_and_release() {
        let counter = Arc::new(AtomicUsize::new(0));
        let factory = create_test_factory(counter.clone());

        let config = PoolConfig {
            min_size: 1,
            max_size: 5,
            ..Default::default()
        };

        let pool = ConnectionPool::new(config, factory).await.unwrap();

        let conn = pool.acquire().await.unwrap();
        let conn_id = conn.id.clone();

        pool.release(conn).await;

        let stats = pool.get_stats().await;
        assert_eq!(stats.active_connections, 0);

        // Acquire again, should get the same connection
        let conn2 = pool.acquire().await.unwrap();
        assert_eq!(conn2.id, conn_id);
    }

    #[tokio::test]
    async fn test_pool_max_size() {
        let counter = Arc::new(AtomicUsize::new(0));
        let factory = create_test_factory(counter.clone());

        let config = PoolConfig {
            min_size: 0,
            max_size: 3,
            connection_timeout_secs: 1,
            ..Default::default()
        };

        let pool = Arc::new(ConnectionPool::new(config, factory).await.unwrap());

        // Acquire max connections
        let _c1 = pool.acquire().await.unwrap();
        let _c2 = pool.acquire().await.unwrap();
        let _c3 = pool.acquire().await.unwrap();

        // Next acquire should timeout
        let result = pool.acquire().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_http_client_pool() {
        let pool = HttpClientPool::new(10, 30).unwrap();

        let client = pool.client();
        assert!(client.get("https://example.com").build().is_ok());

        let (requests, errors) = pool.get_stats();
        assert_eq!(requests, 1);
        assert_eq!(errors, 0);
    }
}
