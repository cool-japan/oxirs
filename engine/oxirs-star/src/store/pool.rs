//! Connection pool for managing concurrent access to the store.
//!
//! This module provides connection pooling functionality to efficiently manage
//! concurrent access to RDF-star storage, with automatic connection lifecycle management.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use crate::{StarConfig, StarError, StarResult};

// Forward declaration - StarStore will be defined in parent module
use super::StarStore;

/// Connection pool for managing concurrent access to the store
pub struct ConnectionPool {
    /// Pool of available store connections
    available_connections: Arc<Mutex<VecDeque<Arc<StarStore>>>>,
    /// Condition variable for waiting on available connections
    connection_available: Arc<Condvar>,
    /// Maximum number of connections in the pool
    max_connections: usize,
    /// Current number of created connections
    active_connections: Arc<Mutex<usize>>,
    /// Configuration for creating new connections
    config: StarConfig,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(max_connections: usize, config: StarConfig) -> Self {
        Self {
            available_connections: Arc::new(Mutex::new(VecDeque::new())),
            connection_available: Arc::new(Condvar::new()),
            max_connections,
            active_connections: Arc::new(Mutex::new(0)),
            config,
        }
    }

    /// Get a connection from the pool (blocks if none available)
    pub fn get_connection(&self) -> StarResult<PooledConnection> {
        let mut available = self.available_connections.lock().unwrap();

        // Try to get an existing connection
        if let Some(store) = available.pop_front() {
            return Ok(PooledConnection::new(store, self.clone()));
        }

        // Check if we can create a new connection
        let mut active_count = self.active_connections.lock().unwrap();
        if *active_count < self.max_connections {
            *active_count += 1;
            drop(active_count);
            drop(available);

            let store = Arc::new(StarStore::with_config(self.config.clone()));
            return Ok(PooledConnection::new(store, self.clone()));
        }

        // Wait for a connection to become available
        drop(active_count);
        available = self.connection_available.wait(available).unwrap();

        match available.pop_front() {
            Some(store) => Ok(PooledConnection::new(store, self.clone())),
            _ => Err(StarError::query_error(
                "No connections available".to_string(),
            )),
        }
    }

    /// Try to get a connection without blocking
    pub fn try_get_connection(&self) -> Option<PooledConnection> {
        let mut available = self.available_connections.lock().ok()?;

        if let Some(store) = available.pop_front() {
            return Some(PooledConnection::new(store, self.clone()));
        }

        let mut active_count = self.active_connections.lock().ok()?;
        if *active_count < self.max_connections {
            *active_count += 1;
            drop(active_count);
            drop(available);

            let store = Arc::new(StarStore::with_config(self.config.clone()));
            Some(PooledConnection::new(store, self.clone()))
        } else {
            None
        }
    }

    /// Return a connection to the pool
    pub(super) fn return_connection(&self, store: Arc<StarStore>) {
        let mut available = self.available_connections.lock().unwrap();
        available.push_back(store);
        self.connection_available.notify_one();
    }

    /// Get pool statistics
    pub fn get_statistics(&self) -> PoolStatistics {
        let available = self.available_connections.lock().unwrap();
        let active_count = self.active_connections.lock().unwrap();

        PoolStatistics {
            available_connections: available.len(),
            active_connections: *active_count,
            max_connections: self.max_connections,
            utilization: (*active_count as f64 / self.max_connections as f64) * 100.0,
        }
    }
}

impl Clone for ConnectionPool {
    fn clone(&self) -> Self {
        Self {
            available_connections: Arc::clone(&self.available_connections),
            connection_available: Arc::clone(&self.connection_available),
            max_connections: self.max_connections,
            active_connections: Arc::clone(&self.active_connections),
            config: self.config.clone(),
        }
    }
}

/// A pooled connection that automatically returns to the pool when dropped
pub struct PooledConnection {
    store: Option<Arc<StarStore>>,
    pool: ConnectionPool,
}

impl PooledConnection {
    fn new(store: Arc<StarStore>, pool: ConnectionPool) -> Self {
        Self {
            store: Some(store),
            pool,
        }
    }

    /// Get access to the underlying store
    pub fn store(&self) -> &StarStore {
        self.store.as_ref().expect("Connection has been dropped")
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(store) = self.store.take() {
            self.pool.return_connection(store);
        }
    }
}

/// Statistics about connection pool usage
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    pub available_connections: usize,
    pub active_connections: usize,
    pub max_connections: usize,
    pub utilization: f64,
}
