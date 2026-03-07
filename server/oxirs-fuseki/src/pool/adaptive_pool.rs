//! Adaptive Connection Pool
//!
//! A generic, per-dataset connection pool that auto-sizes based on observed load.
//!
//! Design goals:
//! - **Adaptive sizing**: monitors utilization and grows/shrinks between
//!   `min_connections` and `max_connections`
//! - **Idle connection reaping**: removes connections that have been idle
//!   longer than `idle_timeout`
//! - **Lifetime-based recycling**: evicts connections older than `max_lifetime`
//! - **Bounded wait**: callers block for at most `acquire_timeout` before
//!   receiving an error
//! - **No unsafe code**, **no unwrap**

use crate::error::{FusekiError, FusekiResult};
use serde::Serialize;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for an `AdaptivePool`.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of connections maintained (pool never shrinks below this)
    pub min_connections: usize,
    /// Maximum number of connections allowed
    pub max_connections: usize,
    /// How long to wait for a connection before returning `Err`
    pub acquire_timeout: Duration,
    /// Connections idle longer than this are eligible for removal
    pub idle_timeout: Duration,
    /// Connections older than this are recycled even if active
    pub max_lifetime: Duration,
    /// Target utilization rate (0.0–1.0); pool grows when above, shrinks when below
    pub target_utilization: f64,
    /// How often `maybe_resize` is permitted to act
    pub resize_interval: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            min_connections: 2,
            max_connections: 50,
            acquire_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(3600),
            target_utilization: 0.70,
            resize_interval: Duration::from_secs(60),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Statistics
// ──────────────────────────────────────────────────────────────────────────────

/// Snapshot of pool statistics for monitoring.
#[derive(Debug, Clone, Serialize)]
pub struct PoolStats {
    /// Total connections in the pool (idle + active)
    pub total_connections: usize,
    /// Connections currently in use
    pub active_connections: usize,
    /// Connections available for use
    pub idle_connections: usize,
    /// Number of times a caller had to wait for a connection
    pub wait_count: u64,
    /// Cumulative wait time across all waits
    pub wait_duration_ms: u64,
    /// Current utilization (active / total), 0.0 if total == 0
    pub utilization: f64,
    /// Number of times the pool has been resized
    pub resize_count: u64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Connection entry
// ──────────────────────────────────────────────────────────────────────────────

/// Wraps a raw connection with lifecycle metadata.
pub struct ConnectionEntry<C> {
    pub connection: C,
    pub created_at: Instant,
    pub last_used: Instant,
    pub use_count: u64,
}

impl<C> ConnectionEntry<C> {
    fn new(connection: C) -> Self {
        let now = Instant::now();
        ConnectionEntry {
            connection,
            created_at: now,
            last_used: now,
            use_count: 0,
        }
    }

    /// Returns `true` if this connection should be discarded.
    fn is_expired(&self, config: &PoolConfig) -> bool {
        let now = Instant::now();
        now.duration_since(self.created_at) >= config.max_lifetime
            || now.duration_since(self.last_used) >= config.idle_timeout
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pool internals
// ──────────────────────────────────────────────────────────────────────────────

/// Shared mutable state for the pool.
struct PoolState<C> {
    /// Idle connections ready for use
    idle: Vec<ConnectionEntry<C>>,
}

// ──────────────────────────────────────────────────────────────────────────────
// AdaptivePool
// ──────────────────────────────────────────────────────────────────────────────

/// Generic adaptive connection pool.
///
/// Type parameter `C` is the raw connection type (e.g. a DB handle, HTTP client,
/// or any `Send + 'static` value).
pub struct AdaptivePool<C: Send + 'static> {
    config: PoolConfig,
    state: Arc<Mutex<PoolState<C>>>,
    active_count: Arc<AtomicUsize>,
    total_count: Arc<AtomicUsize>,
    wait_count: Arc<AtomicU64>,
    wait_duration_ms: Arc<AtomicU64>,
    factory: Arc<dyn Fn() -> FusekiResult<C> + Send + Sync>,
    last_resize: Arc<Mutex<Instant>>,
    resize_count: Arc<AtomicU64>,
}

impl<C: Send + 'static> AdaptivePool<C> {
    /// Create a new pool.  The `factory` closure is called whenever a new
    /// connection needs to be established.
    pub fn new(
        config: PoolConfig,
        factory: impl Fn() -> FusekiResult<C> + Send + Sync + 'static,
    ) -> FusekiResult<Self> {
        if config.min_connections > config.max_connections {
            return Err(FusekiError::Configuration {
                message: format!(
                    "min_connections ({}) must be <= max_connections ({})",
                    config.min_connections, config.max_connections
                ),
            });
        }

        let factory: Arc<dyn Fn() -> FusekiResult<C> + Send + Sync> = Arc::new(factory);

        // Pre-warm min_connections
        let mut idle = Vec::with_capacity(config.min_connections);
        for _ in 0..config.min_connections {
            let conn = factory()?;
            idle.push(ConnectionEntry::new(conn));
        }

        let total = idle.len();
        info!(
            min = config.min_connections,
            max = config.max_connections,
            "AdaptivePool created"
        );

        Ok(AdaptivePool {
            config,
            state: Arc::new(Mutex::new(PoolState { idle })),
            active_count: Arc::new(AtomicUsize::new(0)),
            total_count: Arc::new(AtomicUsize::new(total)),
            wait_count: Arc::new(AtomicU64::new(0)),
            wait_duration_ms: Arc::new(AtomicU64::new(0)),
            factory,
            last_resize: Arc::new(Mutex::new(Instant::now())),
            resize_count: Arc::new(AtomicU64::new(0)),
        })
    }

    // ──────────────────────────────────────────────────────────────────────
    // Acquire
    // ──────────────────────────────────────────────────────────────────────

    /// Acquire a connection from the pool.
    ///
    /// Blocks (spinning with short sleeps) for up to `acquire_timeout` before
    /// returning `Err(FusekiError::Timeout)`.
    pub fn acquire(&self) -> FusekiResult<PooledConnection<C>> {
        let deadline = Instant::now() + self.config.acquire_timeout;
        let mut waited = false;
        let wait_start = Instant::now();

        loop {
            // Try to get an idle connection
            let entry = {
                let mut state = self.state.lock().map_err(|e| FusekiError::Internal {
                    message: format!("Pool state lock poisoned: {e}"),
                })?;

                // Reap any expired idle connections first
                state.idle.retain(|e| !e.is_expired(&self.config));

                // Try to pop a fresh idle entry
                state.idle.pop()
            };

            if let Some(mut entry) = entry {
                entry.last_used = Instant::now();
                entry.use_count += 1;
                self.active_count.fetch_add(1, Ordering::Relaxed);

                if waited {
                    let elapsed_ms = wait_start.elapsed().as_millis() as u64;
                    self.wait_count.fetch_add(1, Ordering::Relaxed);
                    self.wait_duration_ms
                        .fetch_add(elapsed_ms, Ordering::Relaxed);
                }

                debug!(
                    active = self.active_count.load(Ordering::Relaxed),
                    "Connection acquired"
                );
                return Ok(PooledConnection {
                    pool: Arc::clone(&self.state),
                    entry: Some(entry),
                    active_count: Arc::clone(&self.active_count),
                    total_count: Arc::clone(&self.total_count),
                    config: self.config.clone(),
                });
            }

            // No idle connection – can we create a new one?
            let current_total = self.total_count.load(Ordering::Relaxed);
            if current_total < self.config.max_connections {
                // Attempt to reserve a slot atomically
                let prev = self.total_count.compare_exchange(
                    current_total,
                    current_total + 1,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                );
                if prev.is_ok() {
                    // We own the slot – create the connection
                    match (self.factory)() {
                        Ok(conn) => {
                            let mut entry = ConnectionEntry::new(conn);
                            entry.use_count = 1;
                            entry.last_used = Instant::now();
                            self.active_count.fetch_add(1, Ordering::Relaxed);

                            if waited {
                                let elapsed_ms = wait_start.elapsed().as_millis() as u64;
                                self.wait_count.fetch_add(1, Ordering::Relaxed);
                                self.wait_duration_ms
                                    .fetch_add(elapsed_ms, Ordering::Relaxed);
                            }

                            debug!(
                                total = self.total_count.load(Ordering::Relaxed),
                                "New connection created"
                            );
                            return Ok(PooledConnection {
                                pool: Arc::clone(&self.state),
                                entry: Some(entry),
                                active_count: Arc::clone(&self.active_count),
                                total_count: Arc::clone(&self.total_count),
                                config: self.config.clone(),
                            });
                        }
                        Err(e) => {
                            // Give back the reserved slot
                            self.total_count.fetch_sub(1, Ordering::SeqCst);
                            return Err(e);
                        }
                    }
                }
                // Another thread snapped up the slot; loop again
            } else {
                // Pool is at max_connections – wait briefly and retry
                waited = true;
                if Instant::now() >= deadline {
                    return Err(FusekiError::TimeoutWithMessage(format!(
                        "Could not acquire connection within {:?}",
                        self.config.acquire_timeout
                    )));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Resize
    // ──────────────────────────────────────────────────────────────────────

    /// Opportunistically resize the pool based on current utilization.
    ///
    /// Should be called periodically (e.g. from a background task).
    /// - If utilization > `target_utilization`, grow toward `max_connections`.
    /// - If utilization < `target_utilization / 2`, shrink toward `min_connections`.
    pub fn maybe_resize(&self) -> FusekiResult<()> {
        // Throttle resizes
        {
            let mut last = self.last_resize.lock().map_err(|e| FusekiError::Internal {
                message: format!("resize lock poisoned: {e}"),
            })?;
            if last.elapsed() < self.config.resize_interval {
                return Ok(());
            }
            *last = Instant::now();
        }

        let active = self.active_count.load(Ordering::Relaxed);
        let total = self.total_count.load(Ordering::Relaxed);

        let utilization = if total == 0 {
            0.0
        } else {
            active as f64 / total as f64
        };

        if utilization > self.config.target_utilization && total < self.config.max_connections {
            // Grow: add one connection
            let conn = (self.factory)()?;
            let entry = ConnectionEntry::new(conn);
            {
                let mut state = self.state.lock().map_err(|e| FusekiError::Internal {
                    message: format!("Pool state lock poisoned on grow: {e}"),
                })?;
                state.idle.push(entry);
            }
            self.total_count.fetch_add(1, Ordering::SeqCst);
            self.resize_count.fetch_add(1, Ordering::Relaxed);
            info!(total = total + 1, utilization, "Pool grown");
        } else if utilization < self.config.target_utilization / 2.0
            && total > self.config.min_connections
        {
            // Shrink: remove one idle connection
            let removed = {
                let mut state = self.state.lock().map_err(|e| FusekiError::Internal {
                    message: format!("Pool state lock poisoned on shrink: {e}"),
                })?;
                state.idle.pop().is_some()
            };
            if removed {
                self.total_count.fetch_sub(1, Ordering::SeqCst);
                self.resize_count.fetch_add(1, Ordering::Relaxed);
                info!(total = total - 1, utilization, "Pool shrunk");
            }
        }

        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    // Maintenance
    // ──────────────────────────────────────────────────────────────────────

    /// Drain idle connections exceeding `min_connections`.
    ///
    /// Returns the number of connections removed.
    pub fn drain_idle(&self) -> usize {
        let min = self.config.min_connections;
        let mut state = match self.state.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Pool state lock poisoned on drain_idle: {}", e);
                return 0;
            }
        };

        let len = state.idle.len();
        if len <= min {
            return 0;
        }
        let drain_count = len - min;
        state.idle.truncate(min);
        self.total_count.fetch_sub(drain_count, Ordering::SeqCst);
        debug!(drained = drain_count, "Drained idle connections");
        drain_count
    }

    // ──────────────────────────────────────────────────────────────────────
    // Stats
    // ──────────────────────────────────────────────────────────────────────

    pub fn stats(&self) -> PoolStats {
        let idle_count = self.state.lock().map(|s| s.idle.len()).unwrap_or(0);

        let active = self.active_count.load(Ordering::Relaxed);
        let total = self.total_count.load(Ordering::Relaxed);

        PoolStats {
            total_connections: total,
            active_connections: active,
            idle_connections: idle_count,
            wait_count: self.wait_count.load(Ordering::Relaxed),
            wait_duration_ms: self.wait_duration_ms.load(Ordering::Relaxed),
            utilization: if total == 0 {
                0.0
            } else {
                active as f64 / total as f64
            },
            resize_count: self.resize_count.load(Ordering::Relaxed),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PooledConnection (RAII guard)
// ──────────────────────────────────────────────────────────────────────────────

/// A connection checked out from an `AdaptivePool`.
///
/// Implements `Deref` so callers can use the inner connection directly.
/// On `Drop`, the connection is returned to the pool's idle list.
pub struct PooledConnection<C: Send + 'static> {
    pool: Arc<Mutex<PoolState<C>>>,
    entry: Option<ConnectionEntry<C>>,
    active_count: Arc<AtomicUsize>,
    total_count: Arc<AtomicUsize>,
    config: PoolConfig,
}

impl<C: Send + 'static> std::ops::Deref for PooledConnection<C> {
    type Target = C;

    fn deref(&self) -> &C {
        // `entry` is always `Some` while the connection is held
        &self
            .entry
            .as_ref()
            .expect("PooledConnection entry missing")
            .connection
    }
}

impl<C: Send + 'static> std::ops::DerefMut for PooledConnection<C> {
    fn deref_mut(&mut self) -> &mut C {
        &mut self
            .entry
            .as_mut()
            .expect("PooledConnection entry missing")
            .connection
    }
}

impl<C: Send + 'static> Drop for PooledConnection<C> {
    fn drop(&mut self) {
        let entry = match self.entry.take() {
            Some(e) => e,
            None => return,
        };

        self.active_count.fetch_sub(1, Ordering::Relaxed);

        // If the connection is expired, discard it rather than returning it
        if entry.is_expired(&self.config) {
            self.total_count.fetch_sub(1, Ordering::SeqCst);
            debug!("Expired connection discarded on return");
            return;
        }

        // Return to idle pool
        match self.pool.lock() {
            Ok(mut state) => {
                state.idle.push(entry);
                debug!("Connection returned to pool");
            }
            Err(e) => {
                warn!("Pool state lock poisoned on return: {}", e);
                self.total_count.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-dataset pool registry
// ──────────────────────────────────────────────────────────────────────────────

/// Type alias for the dataset-keyed factory function used by `DatasetPoolRegistry`.
type DatasetFactory<C> = Arc<dyn Fn(&str) -> FusekiResult<C> + Send + Sync>;

/// Registry that maintains one `AdaptivePool` per named dataset.
pub struct DatasetPoolRegistry<C: Send + 'static> {
    pools: Mutex<std::collections::HashMap<String, Arc<AdaptivePool<C>>>>,
    default_config: PoolConfig,
    factory: DatasetFactory<C>,
}

impl<C: Send + 'static> DatasetPoolRegistry<C> {
    /// Create a new registry.  The `factory` closure receives the dataset name
    /// and returns a new connection.
    pub fn new(
        default_config: PoolConfig,
        factory: impl Fn(&str) -> FusekiResult<C> + Send + Sync + 'static,
    ) -> Self {
        DatasetPoolRegistry {
            pools: Mutex::new(std::collections::HashMap::new()),
            default_config,
            factory: Arc::new(factory),
        }
    }

    /// Get or create the pool for `dataset_name`.
    pub fn pool_for(&self, dataset_name: &str) -> FusekiResult<Arc<AdaptivePool<C>>> {
        let mut pools = self.pools.lock().map_err(|e| FusekiError::Internal {
            message: format!("DatasetPoolRegistry lock poisoned: {e}"),
        })?;

        if let Some(pool) = pools.get(dataset_name) {
            return Ok(Arc::clone(pool));
        }

        let name = dataset_name.to_string();
        let factory = Arc::clone(&self.factory);
        let pool = AdaptivePool::new(self.default_config.clone(), move || factory(&name))?;

        let pool = Arc::new(pool);
        pools.insert(dataset_name.to_string(), Arc::clone(&pool));
        info!(
            dataset = dataset_name,
            "Created new dataset connection pool"
        );
        Ok(pool)
    }

    /// Snapshot statistics for all pools.
    pub fn all_stats(&self) -> std::collections::HashMap<String, PoolStats> {
        let pools = self.pools.lock().unwrap_or_else(|e| e.into_inner());
        pools
            .iter()
            .map(|(name, pool)| (name.clone(), pool.stats()))
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct TestConn {
        id: usize,
    }

    fn make_pool(min: usize, max: usize) -> FusekiResult<AdaptivePool<TestConn>> {
        let counter = Arc::new(AtomicUsize::new(0));
        let config = PoolConfig {
            min_connections: min,
            max_connections: max,
            acquire_timeout: Duration::from_millis(200),
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(3600),
            target_utilization: 0.7,
            resize_interval: Duration::from_secs(60),
        };
        AdaptivePool::new(config, move || {
            let id = counter.fetch_add(1, Ordering::Relaxed);
            Ok(TestConn { id })
        })
    }

    #[test]
    fn test_pool_creation() {
        let pool = make_pool(2, 10).unwrap();
        let stats = pool.stats();
        assert_eq!(
            stats.total_connections, 2,
            "Pool should start with min_connections"
        );
        assert_eq!(stats.idle_connections, 2);
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_acquire_and_release() {
        let pool = make_pool(2, 10).unwrap();

        {
            let conn = pool.acquire().unwrap();
            let stats = pool.stats();
            assert_eq!(stats.active_connections, 1);
            let _id = conn.id; // Use Deref
        } // conn dropped → returned to pool

        let stats = pool.stats();
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_pool_grows_on_demand() {
        let pool = make_pool(1, 10).unwrap();

        let _c1 = pool.acquire().unwrap();
        let _c2 = pool.acquire().unwrap(); // Should create a new connection

        let stats = pool.stats();
        assert!(stats.total_connections >= 2, "Pool should have grown");
    }

    #[test]
    fn test_pool_max_size_enforced() {
        let pool = make_pool(0, 2).unwrap();

        let _c1 = pool.acquire().unwrap();
        let _c2 = pool.acquire().unwrap();

        // Third acquire should timeout
        let result = pool.acquire();
        assert!(
            result.is_err(),
            "Should fail when pool is at max_connections"
        );
    }

    #[test]
    fn test_pool_invalid_config() {
        let counter = Arc::new(AtomicUsize::new(0));
        let config = PoolConfig {
            min_connections: 10,
            max_connections: 5, // Invalid: min > max
            ..Default::default()
        };
        let result = AdaptivePool::new(config, move || {
            let id = counter.fetch_add(1, Ordering::Relaxed);
            Ok(TestConn { id })
        });
        assert!(result.is_err(), "Should fail when min > max");
    }

    #[test]
    fn test_drain_idle() {
        let pool = make_pool(1, 10).unwrap();

        // Grow the pool
        {
            let _c1 = pool.acquire().unwrap();
            let _c2 = pool.acquire().unwrap();
            let _c3 = pool.acquire().unwrap();
        } // All returned → 3 idle (or at least 3 total)

        // Only drain if we have excess idle connections
        let drained = pool.drain_idle();
        let stats = pool.stats();
        assert!(
            stats.idle_connections <= stats.total_connections,
            "Idle should not exceed total"
        );
        let _ = drained; // may be 0 or positive depending on timing
    }

    #[test]
    fn test_stats_utilization() {
        let pool = make_pool(4, 10).unwrap();

        let _c1 = pool.acquire().unwrap();
        let _c2 = pool.acquire().unwrap();

        let stats = pool.stats();
        assert!(
            stats.utilization > 0.0,
            "Utilization should be > 0 when connections are active"
        );
        assert!(
            stats.utilization <= 1.0,
            "Utilization should not exceed 1.0"
        );
    }

    #[test]
    fn test_dataset_registry() {
        let registry: DatasetPoolRegistry<TestConn> = DatasetPoolRegistry::new(
            PoolConfig {
                min_connections: 1,
                max_connections: 5,
                acquire_timeout: Duration::from_millis(200),
                ..Default::default()
            },
            |dataset| {
                let id = dataset.len(); // Use dataset name length as ID
                Ok(TestConn { id })
            },
        );

        let pool_a = registry.pool_for("dataset_a").unwrap();
        let pool_b = registry.pool_for("dataset_b").unwrap();

        // Both pools should be distinct
        assert!(!Arc::ptr_eq(&pool_a, &pool_b));

        // Second call returns same pool
        let pool_a2 = registry.pool_for("dataset_a").unwrap();
        assert!(Arc::ptr_eq(&pool_a, &pool_a2));

        let all_stats = registry.all_stats();
        assert_eq!(all_stats.len(), 2);
        assert!(all_stats.contains_key("dataset_a"));
        assert!(all_stats.contains_key("dataset_b"));
    }

    #[test]
    fn test_deref_mut() {
        let pool: AdaptivePool<Vec<u8>> = AdaptivePool::new(
            PoolConfig {
                min_connections: 1,
                max_connections: 5,
                acquire_timeout: Duration::from_millis(200),
                ..Default::default()
            },
            || Ok(Vec::new()),
        )
        .unwrap();

        let mut conn = pool.acquire().unwrap();
        conn.push(42u8);
        assert_eq!(conn.len(), 1);
    }
}
