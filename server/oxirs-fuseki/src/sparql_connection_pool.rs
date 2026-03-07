//! SPARQL endpoint connection pool.
//!
//! Manages a pool of connections with lifecycle tracking, health checks, and eviction.

use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for a SPARQL connection pool.
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub min_idle: usize,
    pub connect_timeout_ms: u64,
    pub idle_timeout_ms: u64,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 3030,
            max_connections: 10,
            min_idle: 2,
            connect_timeout_ms: 5000,
            idle_timeout_ms: 30_000,
        }
    }
}

/// Represents a single connection in the pool.
#[derive(Debug, Clone)]
pub struct SparqlConnection {
    pub id: u64,
    pub created_at: u64,
    pub last_used: u64,
    pub request_count: u64,
    pub is_healthy: bool,
}

impl SparqlConnection {
    fn new(id: u64, current_time_ms: u64) -> Self {
        Self {
            id,
            created_at: current_time_ms,
            last_used: current_time_ms,
            request_count: 0,
            is_healthy: true,
        }
    }
}

/// Statistics for a connection pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total: usize,
    pub idle: usize,
    pub active: usize,
    pub total_acquired: u64,
    pub total_released: u64,
    pub total_errors: u64,
}

/// Result of attempting to acquire a connection.
#[derive(Debug, Clone, PartialEq)]
pub enum AcquireResult {
    /// A connection was successfully acquired; contains the connection ID.
    Acquired(u64),
    /// All connections are in use and the pool is at capacity.
    PoolExhausted,
    /// Acquisition timed out (reserved for async usage).
    Timeout,
}

/// A pool of SPARQL endpoint connections.
pub struct SparqlConnectionPool {
    config: ConnectionConfig,
    connections: HashMap<u64, SparqlConnection>,
    idle: VecDeque<u64>,
    active: HashSet<u64>,
    next_id: u64,
    stats: PoolStats,
}

impl SparqlConnectionPool {
    /// Create a new connection pool with the given configuration.
    pub fn new(config: ConnectionConfig) -> Self {
        Self {
            config,
            connections: HashMap::new(),
            idle: VecDeque::new(),
            active: HashSet::new(),
            next_id: 1,
            stats: PoolStats::default(),
        }
    }

    /// Attempt to acquire a connection from the pool.
    ///
    /// If an idle connection is available, it is returned. Otherwise a new connection
    /// is created if below max_connections. Returns `PoolExhausted` if at capacity.
    pub fn acquire(&mut self, current_time_ms: u64) -> AcquireResult {
        // Try to pop a healthy idle connection
        while let Some(id) = self.idle.pop_front() {
            if let Some(conn) = self.connections.get_mut(&id) {
                if conn.is_healthy {
                    conn.last_used = current_time_ms;
                    conn.request_count += 1;
                    self.active.insert(id);
                    self.stats.total_acquired += 1;
                    self.stats.idle = self.idle.len();
                    self.stats.active = self.active.len();
                    self.stats.total = self.connections.len();
                    return AcquireResult::Acquired(id);
                } else {
                    // Remove unhealthy connection
                    self.connections.remove(&id);
                    self.stats.total_errors += 1;
                }
            }
        }

        // Create new connection if below max
        let total = self.connections.len();
        if total >= self.config.max_connections {
            return AcquireResult::PoolExhausted;
        }

        let id = self.next_id;
        self.next_id += 1;

        let mut conn = SparqlConnection::new(id, current_time_ms);
        conn.request_count += 1;
        self.connections.insert(id, conn);
        self.active.insert(id);
        self.stats.total_acquired += 1;
        self.stats.total = self.connections.len();
        self.stats.active = self.active.len();
        self.stats.idle = self.idle.len();

        AcquireResult::Acquired(id)
    }

    /// Release a connection back to the idle pool.
    ///
    /// Returns `true` if the connection was found and released successfully.
    pub fn release(&mut self, conn_id: u64, current_time_ms: u64) -> bool {
        if !self.active.remove(&conn_id) {
            return false;
        }

        if let Some(conn) = self.connections.get_mut(&conn_id) {
            conn.last_used = current_time_ms;
            if conn.is_healthy {
                self.idle.push_back(conn_id);
            } else {
                self.connections.remove(&conn_id);
                self.stats.total_errors += 1;
            }
        } else {
            return false;
        }

        self.stats.total_released += 1;
        self.stats.total = self.connections.len();
        self.stats.active = self.active.len();
        self.stats.idle = self.idle.len();
        true
    }

    /// Mark a connection as unhealthy. It will be removed on next release or eviction.
    pub fn mark_unhealthy(&mut self, conn_id: u64) {
        if let Some(conn) = self.connections.get_mut(&conn_id) {
            conn.is_healthy = false;
        }
    }

    /// Evict idle connections that have been idle longer than `idle_timeout_ms`.
    ///
    /// Returns the number of connections evicted.
    pub fn evict_idle(&mut self, current_time_ms: u64) -> usize {
        let timeout = self.config.idle_timeout_ms;
        let mut evicted = 0;

        let to_remove: Vec<u64> = self
            .idle
            .iter()
            .filter(|&&id| {
                if let Some(conn) = self.connections.get(&id) {
                    let idle_duration = current_time_ms.saturating_sub(conn.last_used);
                    idle_duration > timeout || !conn.is_healthy
                } else {
                    true
                }
            })
            .copied()
            .collect();

        for id in to_remove {
            self.idle.retain(|&x| x != id);
            self.connections.remove(&id);
            evicted += 1;
        }

        self.stats.total = self.connections.len();
        self.stats.idle = self.idle.len();

        evicted
    }

    /// Ensure the pool has at least `min_idle` idle connections, creating new ones as needed.
    pub fn ensure_min_idle(&mut self, current_time_ms: u64) {
        while self.idle.len() < self.config.min_idle
            && self.connections.len() < self.config.max_connections
        {
            let id = self.next_id;
            self.next_id += 1;
            let conn = SparqlConnection::new(id, current_time_ms);
            self.connections.insert(id, conn);
            self.idle.push_back(id);
        }

        self.stats.total = self.connections.len();
        self.stats.idle = self.idle.len();
        self.stats.active = self.active.len();
    }

    /// Get pool statistics.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Check if a specific connection is healthy.
    pub fn is_healthy(&self, conn_id: u64) -> bool {
        self.connections
            .get(&conn_id)
            .map(|c| c.is_healthy)
            .unwrap_or(false)
    }

    /// Resize the pool's maximum number of connections.
    pub fn resize(&mut self, new_max: usize) {
        self.config.max_connections = new_max;

        // Evict excess idle connections if needed
        while self.connections.len() > new_max && !self.idle.is_empty() {
            if let Some(id) = self.idle.pop_back() {
                self.connections.remove(&id);
            }
        }

        self.stats.total = self.connections.len();
        self.stats.idle = self.idle.len();
        self.stats.active = self.active.len();
    }

    /// Drain all connections, closing both idle and active.
    pub fn drain(&mut self) {
        self.connections.clear();
        self.idle.clear();
        self.active.clear();
        self.stats.total = 0;
        self.stats.idle = 0;
        self.stats.active = 0;
    }

    /// Get the number of total connections.
    pub fn total(&self) -> usize {
        self.connections.len()
    }

    /// Get the number of idle connections.
    pub fn idle_count(&self) -> usize {
        self.idle.len()
    }

    /// Get the number of active connections.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Get a reference to a connection by ID.
    pub fn get_connection(&self, conn_id: u64) -> Option<&SparqlConnection> {
        self.connections.get(&conn_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(max: usize, min_idle: usize) -> ConnectionConfig {
        ConnectionConfig {
            host: "localhost".to_string(),
            port: 3030,
            max_connections: max,
            min_idle,
            connect_timeout_ms: 1000,
            idle_timeout_ms: 10_000,
        }
    }

    #[test]
    fn test_acquire_creates_connection() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 1));
        let result = pool.acquire(100);
        assert!(matches!(result, AcquireResult::Acquired(_)));
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn test_acquire_returns_idle_connection_first() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 2));
        pool.ensure_min_idle(100);
        assert_eq!(pool.idle_count(), 2);
        let result = pool.acquire(200);
        match result {
            AcquireResult::Acquired(id) => {
                assert!(pool.active.contains(&id));
            }
            _ => panic!("Expected Acquired"),
        }
        assert_eq!(pool.idle_count(), 1);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn test_max_connections_limit_pool_exhausted() {
        let mut pool = SparqlConnectionPool::new(make_config(3, 0));
        let _r1 = pool.acquire(100);
        let _r2 = pool.acquire(100);
        let _r3 = pool.acquire(100);
        let r4 = pool.acquire(100);
        assert_eq!(r4, AcquireResult::PoolExhausted);
    }

    #[test]
    fn test_release_returns_to_idle() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!("Expected Acquired"),
        };
        assert_eq!(pool.idle_count(), 0);
        let released = pool.release(id, 200);
        assert!(released);
        assert_eq!(pool.idle_count(), 1);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn test_release_unknown_id_returns_false() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        assert!(!pool.release(999, 100));
    }

    #[test]
    fn test_release_updates_last_used() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        pool.release(id, 500);
        let conn = pool.get_connection(id).unwrap();
        assert_eq!(conn.last_used, 500);
    }

    #[test]
    fn test_evict_idle_removes_stale_connections() {
        let config = ConnectionConfig {
            idle_timeout_ms: 1000,
            ..make_config(5, 0)
        };
        let mut pool = SparqlConnectionPool::new(config);

        // Create and release connections at time 0
        let id1 = match pool.acquire(0) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        pool.release(id1, 0);

        // Evict at time 2000 (> idle_timeout_ms of 1000)
        let evicted = pool.evict_idle(2000);
        assert_eq!(evicted, 1);
        assert_eq!(pool.idle_count(), 0);
        assert_eq!(pool.total(), 0);
    }

    #[test]
    fn test_evict_idle_keeps_recent_connections() {
        let config = ConnectionConfig {
            idle_timeout_ms: 10_000,
            ..make_config(5, 0)
        };
        let mut pool = SparqlConnectionPool::new(config);

        let id1 = match pool.acquire(1000) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        pool.release(id1, 1000);

        // Evict at time 5000 (< idle_timeout_ms of 10000)
        let evicted = pool.evict_idle(5000);
        assert_eq!(evicted, 0);
        assert_eq!(pool.idle_count(), 1);
    }

    #[test]
    fn test_ensure_min_idle_creates_connections() {
        let mut pool = SparqlConnectionPool::new(make_config(10, 3));
        pool.ensure_min_idle(100);
        assert!(pool.idle_count() >= 3);
    }

    #[test]
    fn test_ensure_min_idle_respects_max_connections() {
        let mut pool = SparqlConnectionPool::new(make_config(2, 5));
        pool.ensure_min_idle(100);
        assert_eq!(pool.idle_count(), 2);
        assert_eq!(pool.total(), 2);
    }

    #[test]
    fn test_mark_unhealthy_is_reflected() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        assert!(pool.is_healthy(id));
        pool.mark_unhealthy(id);
        assert!(!pool.is_healthy(id));
    }

    #[test]
    fn test_mark_unhealthy_removes_on_release() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        pool.mark_unhealthy(id);
        pool.release(id, 200);
        // Should not be in idle
        assert_eq!(pool.idle_count(), 0);
        // Should be removed from connections
        assert!(pool.get_connection(id).is_none());
    }

    #[test]
    fn test_drain_empties_all() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 3));
        pool.ensure_min_idle(100);
        let _id = pool.acquire(100);
        assert!(pool.total() > 0);
        pool.drain();
        assert_eq!(pool.total(), 0);
        assert_eq!(pool.idle_count(), 0);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn test_stats_total_acquired_increments() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        pool.acquire(100);
        pool.acquire(100);
        assert_eq!(pool.stats().total_acquired, 2);
    }

    #[test]
    fn test_stats_total_released_increments() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        pool.release(id, 200);
        assert_eq!(pool.stats().total_released, 1);
    }

    #[test]
    fn test_stats_active_idle_counts() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id1 = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        let _id2 = pool.acquire(100);
        assert_eq!(pool.stats().active, 2);
        pool.release(id1, 200);
        assert_eq!(pool.stats().idle, 1);
        assert_eq!(pool.stats().active, 1);
    }

    #[test]
    fn test_resize_reduces_max() {
        let mut pool = SparqlConnectionPool::new(make_config(10, 0));
        pool.ensure_min_idle(100);
        pool.resize(2);
        assert!(pool.total() <= 2);
        let r = pool.acquire(100);
        assert!(matches!(r, AcquireResult::Acquired(_)));
    }

    #[test]
    fn test_resize_allows_more_connections() {
        let mut pool = SparqlConnectionPool::new(make_config(2, 0));
        pool.acquire(100);
        pool.acquire(100);
        assert_eq!(pool.acquire(100), AcquireResult::PoolExhausted);
        pool.resize(5);
        assert!(matches!(pool.acquire(100), AcquireResult::Acquired(_)));
    }

    #[test]
    fn test_acquire_after_release() {
        let mut pool = SparqlConnectionPool::new(make_config(1, 0));
        let id = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        assert_eq!(pool.acquire(100), AcquireResult::PoolExhausted);
        pool.release(id, 200);
        let result = pool.acquire(300);
        assert!(matches!(result, AcquireResult::Acquired(_)));
    }

    #[test]
    fn test_request_count_increments() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        pool.release(id, 200);
        // Acquire again (same id from idle)
        let id2 = match pool.acquire(300) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        let conn = pool.get_connection(id2).unwrap();
        assert!(conn.request_count >= 2);
    }

    #[test]
    fn test_is_healthy_unknown_id() {
        let pool = SparqlConnectionPool::new(make_config(5, 0));
        assert!(!pool.is_healthy(999));
    }

    #[test]
    fn test_connection_created_at_set() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id = match pool.acquire(12345) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        let conn = pool.get_connection(id).unwrap();
        assert_eq!(conn.created_at, 12345);
    }

    #[test]
    fn test_evict_unhealthy_idle_connections() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        pool.release(id, 100);
        pool.mark_unhealthy(id);
        let evicted = pool.evict_idle(200);
        assert_eq!(evicted, 1);
    }

    #[test]
    fn test_multiple_acquire_release_cycles() {
        let mut pool = SparqlConnectionPool::new(make_config(3, 0));
        for i in 0..5u64 {
            let id = match pool.acquire(i * 100) {
                AcquireResult::Acquired(id) => id,
                _ => panic!("acquire {i} failed"),
            };
            pool.release(id, i * 100 + 50);
        }
        assert_eq!(pool.stats().total_acquired, 5);
        assert_eq!(pool.stats().total_released, 5);
    }

    #[test]
    fn test_min_idle_does_not_exceed_max_connections() {
        let mut pool = SparqlConnectionPool::new(make_config(3, 10));
        pool.ensure_min_idle(100);
        assert!(pool.idle_count() <= 3);
        assert!(pool.total() <= 3);
    }

    #[test]
    fn test_pool_stats_total_errors() {
        let mut pool = SparqlConnectionPool::new(make_config(5, 0));
        let id = match pool.acquire(100) {
            AcquireResult::Acquired(id) => id,
            _ => panic!(),
        };
        pool.mark_unhealthy(id);
        pool.release(id, 200);
        // total_errors should have incremented
        assert!(pool.stats().total_errors > 0);
    }
}
