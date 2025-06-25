//! # Connection Pool Implementation
//!
//! Provides connection pooling functionality for managing backend connections
//! efficiently in high-throughput streaming scenarios.

use crate::StreamConfig;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::{debug, error, info, warn};

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub min_connections: usize,
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
    pub health_check_interval: Duration,
    pub retry_attempts: u32,
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
        }
    }
}

/// Connection pool status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStatus {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub pending_requests: usize,
    pub is_healthy: bool,
    pub last_health_check: Option<Instant>,
}

/// Generic connection trait
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

/// Connection wrapper with metadata
struct PooledConnectionWrapper<T: PooledConnection> {
    connection: T,
    created_at: Instant,
    last_activity: Instant,
    is_in_use: bool,
}

impl<T: PooledConnection> PooledConnectionWrapper<T> {
    fn new(connection: T) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_activity: now,
            is_in_use: false,
        }
    }
    
    fn is_expired(&self, max_lifetime: Duration, idle_timeout: Duration) -> bool {
        let now = Instant::now();
        now.duration_since(self.created_at) > max_lifetime ||
        (!self.is_in_use && now.duration_since(self.last_activity) > idle_timeout)
    }
    
    async fn is_healthy(&self) -> bool {
        self.connection.is_healthy().await
    }
}

/// Connection pool implementation
pub struct ConnectionPool<T: PooledConnection> {
    config: PoolConfig,
    connections: Arc<Mutex<VecDeque<PooledConnectionWrapper<T>>>>,
    active_count: Arc<Mutex<usize>>,
    semaphore: Arc<Semaphore>,
    stats: Arc<RwLock<PoolStats>>,
    connection_factory: Arc<dyn ConnectionFactory<T>>,
}

/// Connection factory trait
#[async_trait::async_trait]
pub trait ConnectionFactory<T: PooledConnection>: Send + Sync {
    async fn create_connection(&self) -> Result<T>;
}

/// Pool statistics
#[derive(Debug, Default, Clone)]
struct PoolStats {
    total_created: u64,
    total_destroyed: u64,
    total_borrowed: u64,
    total_returned: u64,
    creation_failures: u64,
    health_check_failures: u64,
    timeouts: u64,
}

impl<T: PooledConnection> ConnectionPool<T> {
    /// Create a new connection pool
    pub async fn new(
        config: PoolConfig,
        factory: Arc<dyn ConnectionFactory<T>>,
    ) -> Result<Self> {
        let pool = Self {
            semaphore: Arc::new(Semaphore::new(config.max_connections)),
            connections: Arc::new(Mutex::new(VecDeque::new())),
            active_count: Arc::new(Mutex::new(0)),
            stats: Arc::new(RwLock::new(PoolStats::default())),
            connection_factory: factory,
            config,
        };
        
        // Initialize minimum connections
        pool.ensure_min_connections().await?;
        
        // Start background maintenance task
        pool.start_maintenance_task().await;
        
        Ok(pool)
    }
    
    /// Get a connection from the pool
    pub async fn get_connection(&self) -> Result<PooledConnectionHandle<T>> {
        let _permit = self.semaphore.acquire().await
            .map_err(|_| anyhow!("Failed to acquire semaphore permit"))?;
        
        let connection = match self.try_get_existing_connection().await {
            Some(conn) => conn,
            None => self.create_new_connection().await?,
        };
        
        *self.active_count.lock().await += 1;
        self.stats.write().await.total_borrowed += 1;
        
        Ok(PooledConnectionHandle::new(
            connection,
            self.connections.clone(),
            self.active_count.clone(),
            self.stats.clone(),
        ))
    }
    
    /// Try to get an existing healthy connection
    async fn try_get_existing_connection(&self) -> Option<T> {
        let mut connections = self.connections.lock().await;
        
        while let Some(mut wrapper) = connections.pop_front() {
            if wrapper.is_expired(self.config.max_lifetime, self.config.idle_timeout) {
                // Connection expired, destroy it
                if let Err(e) = wrapper.connection.close().await {
                    warn!("Failed to close expired connection: {}", e);
                }
                self.stats.write().await.total_destroyed += 1;
                continue;
            }
            
            if !wrapper.is_healthy().await {
                // Connection unhealthy, destroy it
                if let Err(e) = wrapper.connection.close().await {
                    warn!("Failed to close unhealthy connection: {}", e);
                }
                self.stats.write().await.health_check_failures += 1;
                self.stats.write().await.total_destroyed += 1;
                continue;
            }
            
            wrapper.is_in_use = true;
            wrapper.last_activity = Instant::now();
            return Some(wrapper.connection);
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
    
    /// Return a connection to the pool
    async fn return_connection(&self, mut connection: T) {
        connection.update_activity();
        
        let wrapper = PooledConnectionWrapper {
            connection,
            created_at: Instant::now(),
            last_activity: Instant::now(),
            is_in_use: false,
        };
        
        self.connections.lock().await.push_back(wrapper);
        
        let mut active_count = self.active_count.lock().await;
        if *active_count > 0 {
            *active_count -= 1;
        }
        
        self.stats.write().await.total_returned += 1;
        debug!("Returned connection to pool");
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
    
    /// Start background maintenance task
    async fn start_maintenance_task(&self) {
        let connections = self.connections.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.health_check_interval);
            
            loop {
                interval.tick().await;
                
                let mut connections_guard = connections.lock().await;
                let mut to_remove = Vec::new();
                
                for (index, wrapper) in connections_guard.iter().enumerate() {
                    if wrapper.is_expired(config.max_lifetime, config.idle_timeout) {
                        to_remove.push(index);
                    } else if !wrapper.is_healthy().await {
                        to_remove.push(index);
                        stats.write().await.health_check_failures += 1;
                    }
                }
                
                // Remove expired/unhealthy connections in reverse order
                for &index in to_remove.iter().rev() {
                    if let Some(mut wrapper) = connections_guard.remove(index) {
                        if let Err(e) = wrapper.connection.close().await {
                            warn!("Failed to close connection during maintenance: {}", e);
                        }
                        stats.write().await.total_destroyed += 1;
                    }
                }
                
                debug!("Pool maintenance completed, removed {} connections", to_remove.len());
            }
        });
    }
    
    /// Get pool status
    pub async fn status(&self) -> PoolStatus {
        let connections = self.connections.lock().await;
        let active_count = *self.active_count.lock().await;
        
        PoolStatus {
            total_connections: connections.len() + active_count,
            active_connections: active_count,
            idle_connections: connections.len(),
            pending_requests: 0, // Would need to track this separately
            is_healthy: true,   // Simplified health check
            last_health_check: Some(Instant::now()),
        }
    }
    
    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        self.stats.read().await.clone()
    }
    
    /// Health check for the entire pool
    pub async fn health_check(&self) -> PoolStatus {
        self.status().await
    }
}

/// Connection handle that automatically returns connection to pool when dropped
pub struct PooledConnectionHandle<T: PooledConnection> {
    connection: Option<T>,
    pool_connections: Arc<Mutex<VecDeque<PooledConnectionWrapper<T>>>>,
    active_count: Arc<Mutex<usize>>,
    stats: Arc<RwLock<PoolStats>>,
}

impl<T: PooledConnection> PooledConnectionHandle<T> {
    fn new(
        connection: T,
        pool_connections: Arc<Mutex<VecDeque<PooledConnectionWrapper<T>>>>,
        active_count: Arc<Mutex<usize>>,
        stats: Arc<RwLock<PoolStats>>,
    ) -> Self {
        Self {
            connection: Some(connection),
            pool_connections,
            active_count,
            stats,
        }
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
            
            tokio::spawn(async move {
                let wrapper = PooledConnectionWrapper {
                    connection,
                    created_at: Instant::now(),
                    last_activity: Instant::now(),
                    is_in_use: false,
                };
                
                pool_connections.lock().await.push_back(wrapper);
                
                let mut active = active_count.lock().await;
                if *active > 0 {
                    *active -= 1;
                }
                
                stats.write().await.total_returned += 1;
            });
        }
    }
}

/// Helper for creating connection pools from stream config
impl ConnectionPool<()> {
    /// Create a connection pool from stream configuration
    pub async fn new_from_config(config: &StreamConfig) -> Result<Self> {
        let pool_config = PoolConfig {
            min_connections: 1,
            max_connections: config.max_connections,
            connection_timeout: config.connection_timeout,
            ..Default::default()
        };
        
        // This is a placeholder - in reality you'd create appropriate factory
        // based on the backend type
        let factory = Arc::new(DummyConnectionFactory);
        
        Self::new(pool_config, factory).await
    }
    
    /// Health check (simplified version)
    pub async fn health_check(&self) -> PoolStatus {
        PoolStatus {
            total_connections: 0,
            active_connections: 0,
            idle_connections: 0,
            pending_requests: 0,
            is_healthy: true,
            last_health_check: Some(Instant::now()),
        }
    }
}

/// Dummy connection factory for testing
struct DummyConnectionFactory;

#[async_trait::async_trait]
impl ConnectionFactory<()> for DummyConnectionFactory {
    async fn create_connection(&self) -> Result<()> {
        Ok(())
    }
}

impl PooledConnection for () {
    async fn is_healthy(&self) -> bool {
        true
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
    
    fn update_activity(&mut self) {
        // No-op for unit type
    }
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
        
        let handle = pool.get_connection().await.unwrap();
        
        let status = pool.status().await;
        assert_eq!(status.active_connections, 1);
        assert_eq!(status.idle_connections, 0);
        
        drop(handle);
        
        // Wait for the connection to be returned
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let status = pool.status().await;
        assert_eq!(status.active_connections, 0);
        assert_eq!(status.idle_connections, 1);
    }
}