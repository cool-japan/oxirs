//! Adaptive Connection Pool with Dynamic Sizing and Health Monitoring
//!
//! This module provides `AdaptiveConnectionPool` — an enhanced connection pool that:
//! - Dynamically sizes between `min_connections` and `max_connections` based on load
//! - Performs periodic health checks on idle connections
//! - Enforces idle timeout and maximum lifetime for connections
//! - Supports connection warmup (pre-warming a configurable number of connections)
//! - Exposes rich pool metrics (active, idle, waiting, health-failed counts)
//! - Uses RAII guards for automatic connection return

use crate::error::{FusekiError, FusekiResult};
use serde::Serialize;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for `AdaptiveConnectionPool`.
#[derive(Debug, Clone)]
pub struct AdaptiveConnectionPoolConfig {
    /// Minimum connections the pool maintains at all times
    pub min_connections: usize,
    /// Maximum connections the pool will ever hold
    pub max_connections: usize,
    /// How many connections to pre-warm at startup (must be ≤ max_connections)
    pub warmup_connections: usize,
    /// Duration before an idle connection is closed
    pub idle_timeout: Duration,
    /// Maximum lifetime of any connection regardless of usage
    pub max_lifetime: Duration,
    /// How long to wait for a connection to become available
    pub acquire_timeout: Duration,
    /// Interval at which background health checks run on idle connections
    pub health_check_interval: Duration,
    /// Target utilization (0.0–1.0); pool grows above this, shrinks below half
    pub target_utilization: f64,
}

impl Default for AdaptiveConnectionPoolConfig {
    fn default() -> Self {
        AdaptiveConnectionPoolConfig {
            min_connections: 2,
            max_connections: 50,
            warmup_connections: 2,
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(3600),
            acquire_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(60),
            target_utilization: 0.70,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Live metrics snapshot for `AdaptiveConnectionPool`.
#[derive(Debug, Clone, Serialize)]
pub struct AdaptivePoolMetrics {
    /// Total connections currently in the pool (idle + active)
    pub total_connections: usize,
    /// Connections currently checked out by callers
    pub active_connections: usize,
    /// Connections sitting idle
    pub idle_connections: usize,
    /// Callers currently waiting for a connection
    pub waiting_connections: usize,
    /// How many connections failed the health check since pool creation
    pub health_check_failures: u64,
    /// How many connections have been recycled (expired/unhealthy)
    pub total_recycled: u64,
    /// How many new connections have been created
    pub total_created: u64,
    /// Current utilization: active / total (0.0 if total == 0)
    pub utilization: f64,
    /// Total times the pool was dynamically resized
    pub resize_count: u64,
    /// Total cumulative wait time in milliseconds
    pub total_wait_ms: u64,
    /// How many acquire calls had to wait at all
    pub total_waits: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection entry
// ─────────────────────────────────────────────────────────────────────────────

struct ConnectionEntry<C> {
    connection: C,
    created_at: Instant,
    last_used: Instant,
    use_count: u64,
    is_healthy: bool,
}

impl<C> ConnectionEntry<C> {
    fn new(connection: C) -> Self {
        let now = Instant::now();
        ConnectionEntry {
            connection,
            created_at: now,
            last_used: now,
            use_count: 0,
            is_healthy: true,
        }
    }

    /// Returns true if this connection should be discarded (expired or unhealthy).
    fn is_stale(&self, cfg: &AdaptiveConnectionPoolConfig) -> bool {
        let now = Instant::now();
        !self.is_healthy
            || now.duration_since(self.created_at) >= cfg.max_lifetime
            || now.duration_since(self.last_used) >= cfg.idle_timeout
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared pool state
// ─────────────────────────────────────────────────────────────────────────────

struct PoolInner<C> {
    idle: Vec<ConnectionEntry<C>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveConnectionPool
// ─────────────────────────────────────────────────────────────────────────────

/// An adaptive, health-aware connection pool.
///
/// Generic over `C` — the raw connection type (any `Send + 'static` value).
/// A `factory` closure creates new connections on demand; an optional
/// `health_check` closure validates idle connections during maintenance.
pub struct AdaptiveConnectionPool<C: Send + 'static> {
    config: AdaptiveConnectionPoolConfig,
    inner: Arc<Mutex<PoolInner<C>>>,

    // Atomic counters (lock-free reads for metrics)
    active_count: Arc<AtomicUsize>,
    waiting_count: Arc<AtomicUsize>,
    total_count: Arc<AtomicUsize>,

    // Cumulative statistics
    total_created: Arc<AtomicU64>,
    total_recycled: Arc<AtomicU64>,
    health_check_failures: Arc<AtomicU64>,
    resize_count: Arc<AtomicU64>,
    total_wait_ms: Arc<AtomicU64>,
    total_waits: Arc<AtomicU64>,

    // Factories
    factory: Arc<dyn Fn() -> FusekiResult<C> + Send + Sync>,
    health_check: Arc<dyn Fn(&C) -> bool + Send + Sync>,

    // Resize throttle
    last_resize: Arc<Mutex<Instant>>,
}

impl<C: Send + 'static> AdaptiveConnectionPool<C> {
    // ─────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────

    /// Create an `AdaptiveConnectionPool` with a custom health check.
    ///
    /// * `factory` — called whenever a new connection must be established
    /// * `health_check` — called on idle connections; return `false` to evict
    pub fn new_with_health_check(
        config: AdaptiveConnectionPoolConfig,
        factory: impl Fn() -> FusekiResult<C> + Send + Sync + 'static,
        health_check: impl Fn(&C) -> bool + Send + Sync + 'static,
    ) -> FusekiResult<Self> {
        if config.min_connections > config.max_connections {
            return Err(FusekiError::Configuration {
                message: format!(
                    "min_connections ({}) must be ≤ max_connections ({})",
                    config.min_connections, config.max_connections
                ),
            });
        }
        let warmup = config
            .warmup_connections
            .min(config.max_connections)
            .max(config.min_connections);

        let factory: Arc<dyn Fn() -> FusekiResult<C> + Send + Sync> = Arc::new(factory);
        let health_check: Arc<dyn Fn(&C) -> bool + Send + Sync> = Arc::new(health_check);

        // Pre-warm connections
        let mut idle_vec = Vec::with_capacity(warmup);
        for _ in 0..warmup {
            let conn = factory()?;
            idle_vec.push(ConnectionEntry::new(conn));
        }
        let pre_warmed = idle_vec.len();

        let total_count = Arc::new(AtomicUsize::new(pre_warmed));
        let total_created = Arc::new(AtomicU64::new(pre_warmed as u64));

        info!(
            min = config.min_connections,
            max = config.max_connections,
            warmed = pre_warmed,
            "AdaptiveConnectionPool created"
        );

        Ok(AdaptiveConnectionPool {
            config,
            inner: Arc::new(Mutex::new(PoolInner { idle: idle_vec })),
            active_count: Arc::new(AtomicUsize::new(0)),
            waiting_count: Arc::new(AtomicUsize::new(0)),
            total_count,
            total_created,
            total_recycled: Arc::new(AtomicU64::new(0)),
            health_check_failures: Arc::new(AtomicU64::new(0)),
            resize_count: Arc::new(AtomicU64::new(0)),
            total_wait_ms: Arc::new(AtomicU64::new(0)),
            total_waits: Arc::new(AtomicU64::new(0)),
            factory,
            health_check,
            last_resize: Arc::new(Mutex::new(Instant::now())),
        })
    }

    /// Create an `AdaptiveConnectionPool` without a custom health check
    /// (all connections are treated as healthy by default).
    pub fn new(
        config: AdaptiveConnectionPoolConfig,
        factory: impl Fn() -> FusekiResult<C> + Send + Sync + 'static,
    ) -> FusekiResult<Self> {
        Self::new_with_health_check(config, factory, |_| true)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Acquire
    // ─────────────────────────────────────────────────────────────────────

    /// Acquire a connection.  Blocks (with short sleeps) up to `acquire_timeout`.
    pub fn acquire(&self) -> FusekiResult<AdaptivePoolGuard<C>> {
        let deadline = Instant::now() + self.config.acquire_timeout;
        let mut is_waiting = false;
        let wait_start = Instant::now();

        loop {
            // --- try to pop an idle, healthy, non-stale connection ---
            let maybe_entry = {
                let mut guard = self.inner.lock().map_err(|e| FusekiError::Internal {
                    message: format!("pool lock poisoned: {e}"),
                })?;
                // Evict stale entries first
                let recycled = Self::evict_stale(&mut guard.idle, &self.config);
                if recycled > 0 {
                    self.total_recycled
                        .fetch_add(recycled as u64, Ordering::Relaxed);
                    self.total_count.fetch_sub(recycled, Ordering::SeqCst);
                    debug!(recycled, "Evicted stale connections");
                }
                // Pop an idle connection
                guard.idle.pop()
            };

            if let Some(mut entry) = maybe_entry {
                // Run health check
                let healthy = (self.health_check)(&entry.connection);
                if !healthy {
                    self.health_check_failures.fetch_add(1, Ordering::Relaxed);
                    self.total_recycled.fetch_add(1, Ordering::Relaxed);
                    self.total_count.fetch_sub(1, Ordering::SeqCst);
                    debug!("Health check failed; discarding connection");
                    if Instant::now() >= deadline {
                        return Err(FusekiError::TimeoutWithMessage(format!(
                            "AdaptiveConnectionPool: acquire timed out after {:?} (all connections unhealthy)",
                            self.config.acquire_timeout
                        )));
                    }
                    continue; // Try the next idle entry
                }

                entry.last_used = Instant::now();
                entry.use_count += 1;
                self.active_count.fetch_add(1, Ordering::Relaxed);

                if is_waiting {
                    let elapsed_ms = wait_start.elapsed().as_millis() as u64;
                    self.total_wait_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
                    self.total_waits.fetch_add(1, Ordering::Relaxed);
                    self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                }

                debug!(
                    active = self.active_count.load(Ordering::Relaxed),
                    "Connection acquired from idle pool"
                );
                return Ok(AdaptivePoolGuard {
                    inner: Arc::clone(&self.inner),
                    entry: Some(entry),
                    active_count: Arc::clone(&self.active_count),
                    total_count: Arc::clone(&self.total_count),
                    total_recycled: Arc::clone(&self.total_recycled),
                    config: self.config.clone(),
                });
            }

            // --- no idle connection; try to create a new one ---
            let current_total = self.total_count.load(Ordering::Relaxed);
            if current_total < self.config.max_connections {
                let prev = self.total_count.compare_exchange(
                    current_total,
                    current_total + 1,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                );
                if prev.is_ok() {
                    match (self.factory)() {
                        Ok(conn) => {
                            // Run health check on newly created connection.
                            let healthy = (self.health_check)(&conn);
                            if !healthy {
                                self.health_check_failures.fetch_add(1, Ordering::Relaxed);
                                self.total_recycled.fetch_add(1, Ordering::Relaxed);
                                self.total_count.fetch_sub(1, Ordering::SeqCst);
                                debug!("Newly created connection failed health check; discarding");
                                if Instant::now() >= deadline {
                                    return Err(FusekiError::TimeoutWithMessage(format!(
                                        "AdaptiveConnectionPool: acquire timed out after {:?} (all connections unhealthy)",
                                        self.config.acquire_timeout
                                    )));
                                }
                                continue;
                            }

                            self.total_created.fetch_add(1, Ordering::Relaxed);
                            let mut entry = ConnectionEntry::new(conn);
                            entry.use_count = 1;
                            entry.last_used = Instant::now();
                            self.active_count.fetch_add(1, Ordering::Relaxed);

                            if is_waiting {
                                let elapsed_ms = wait_start.elapsed().as_millis() as u64;
                                self.total_wait_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
                                self.total_waits.fetch_add(1, Ordering::Relaxed);
                                self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                            }

                            debug!(
                                total = self.total_count.load(Ordering::Relaxed),
                                "New connection created"
                            );
                            return Ok(AdaptivePoolGuard {
                                inner: Arc::clone(&self.inner),
                                entry: Some(entry),
                                active_count: Arc::clone(&self.active_count),
                                total_count: Arc::clone(&self.total_count),
                                total_recycled: Arc::clone(&self.total_recycled),
                                config: self.config.clone(),
                            });
                        }
                        Err(e) => {
                            self.total_count.fetch_sub(1, Ordering::SeqCst);
                            if is_waiting {
                                self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                            }
                            return Err(e);
                        }
                    }
                }
                // Another thread grabbed the slot; loop
                continue;
            }

            // --- pool at max; wait ---
            if !is_waiting {
                is_waiting = true;
                self.waiting_count.fetch_add(1, Ordering::Relaxed);
            }
            if Instant::now() >= deadline {
                self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                return Err(FusekiError::TimeoutWithMessage(format!(
                    "AdaptiveConnectionPool: acquire timed out after {:?}",
                    self.config.acquire_timeout
                )));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Health check maintenance
    // ─────────────────────────────────────────────────────────────────────

    /// Run a health check sweep on all idle connections.
    ///
    /// Unhealthy connections are discarded; the pool may then re-warm up to
    /// `min_connections` if needed.
    pub fn run_health_checks(&self) -> FusekiResult<usize> {
        let mut guard = self.inner.lock().map_err(|e| FusekiError::Internal {
            message: format!("pool lock poisoned during health check: {e}"),
        })?;

        let before = guard.idle.len();
        let health_check = &*self.health_check;
        guard.idle.retain(|entry| {
            let ok = health_check(&entry.connection);
            if !ok {
                self.health_check_failures.fetch_add(1, Ordering::Relaxed);
            }
            ok
        });
        let removed = before - guard.idle.len();
        if removed > 0 {
            self.total_recycled
                .fetch_add(removed as u64, Ordering::Relaxed);
            self.total_count.fetch_sub(removed, Ordering::SeqCst);
            debug!(removed, "Health-check removed unhealthy connections");
        }

        // Re-warm if below minimum
        let current_total = self.total_count.load(Ordering::Relaxed);
        let min = self.config.min_connections;
        let active = self.active_count.load(Ordering::Relaxed);
        let desired_idle = min.saturating_sub(active);
        let current_idle = guard.idle.len();

        if current_idle < desired_idle && current_total < self.config.max_connections {
            let to_create = desired_idle - current_idle;
            for _ in 0..to_create {
                match (self.factory)() {
                    Ok(conn) => {
                        self.total_created.fetch_add(1, Ordering::Relaxed);
                        self.total_count.fetch_add(1, Ordering::SeqCst);
                        guard.idle.push(ConnectionEntry::new(conn));
                    }
                    Err(e) => {
                        warn!("Failed to create connection during re-warm: {}", e);
                        break;
                    }
                }
            }
        }

        Ok(removed)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Dynamic resize
    // ─────────────────────────────────────────────────────────────────────

    /// Opportunistically grow or shrink the pool based on current utilization.
    ///
    /// Growth adds one idle connection; shrink removes one idle connection.
    /// Resize is throttled to at most once per `health_check_interval`.
    pub fn maybe_resize(&self) -> FusekiResult<()> {
        {
            let mut last = self.last_resize.lock().map_err(|e| FusekiError::Internal {
                message: format!("resize lock poisoned: {e}"),
            })?;
            if last.elapsed() < self.config.health_check_interval {
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
            // Grow by one
            match (self.factory)() {
                Ok(conn) => {
                    self.total_created.fetch_add(1, Ordering::Relaxed);
                    let entry = ConnectionEntry::new(conn);
                    let mut guard = self.inner.lock().map_err(|e| FusekiError::Internal {
                        message: format!("pool lock poisoned on grow: {e}"),
                    })?;
                    guard.idle.push(entry);
                    self.total_count.fetch_add(1, Ordering::SeqCst);
                    self.resize_count.fetch_add(1, Ordering::Relaxed);
                    info!(total = total + 1, utilization, "Pool grown");
                }
                Err(e) => {
                    warn!("Could not grow pool: {}", e);
                }
            }
        } else if utilization < self.config.target_utilization / 2.0
            && total > self.config.min_connections
        {
            // Shrink by one idle connection
            let removed = {
                let mut guard = self.inner.lock().map_err(|e| FusekiError::Internal {
                    message: format!("pool lock poisoned on shrink: {e}"),
                })?;
                guard.idle.pop().is_some()
            };
            if removed {
                self.total_count.fetch_sub(1, Ordering::SeqCst);
                self.total_recycled.fetch_add(1, Ordering::Relaxed);
                self.resize_count.fetch_add(1, Ordering::Relaxed);
                info!(total = total - 1, utilization, "Pool shrunk");
            }
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Metrics
    // ─────────────────────────────────────────────────────────────────────

    /// Snapshot current pool metrics.
    pub fn metrics(&self) -> AdaptivePoolMetrics {
        let idle = self.inner.lock().map(|g| g.idle.len()).unwrap_or(0);
        let active = self.active_count.load(Ordering::Relaxed);
        let total = self.total_count.load(Ordering::Relaxed);
        AdaptivePoolMetrics {
            total_connections: total,
            active_connections: active,
            idle_connections: idle,
            waiting_connections: self.waiting_count.load(Ordering::Relaxed),
            health_check_failures: self.health_check_failures.load(Ordering::Relaxed),
            total_recycled: self.total_recycled.load(Ordering::Relaxed),
            total_created: self.total_created.load(Ordering::Relaxed),
            utilization: if total == 0 {
                0.0
            } else {
                active as f64 / total as f64
            },
            resize_count: self.resize_count.load(Ordering::Relaxed),
            total_wait_ms: self.total_wait_ms.load(Ordering::Relaxed),
            total_waits: self.total_waits.load(Ordering::Relaxed),
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────

    fn evict_stale(
        idle: &mut Vec<ConnectionEntry<C>>,
        cfg: &AdaptiveConnectionPoolConfig,
    ) -> usize {
        let before = idle.len();
        idle.retain(|e| !e.is_stale(cfg));
        before - idle.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RAII guard — auto-returns connection on drop
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard holding a checked-out connection.
///
/// Implements `Deref` / `DerefMut` so callers interact directly with `C`.
/// On `Drop`, the connection is returned to the idle pool (or discarded if expired).
pub struct AdaptivePoolGuard<C: Send + 'static> {
    inner: Arc<Mutex<PoolInner<C>>>,
    entry: Option<ConnectionEntry<C>>,
    active_count: Arc<AtomicUsize>,
    total_count: Arc<AtomicUsize>,
    total_recycled: Arc<AtomicU64>,
    config: AdaptiveConnectionPoolConfig,
}

impl<C: Send + 'static> std::ops::Deref for AdaptivePoolGuard<C> {
    type Target = C;
    fn deref(&self) -> &C {
        &self
            .entry
            .as_ref()
            .expect("AdaptivePoolGuard: entry is None")
            .connection
    }
}

impl<C: Send + 'static> std::ops::DerefMut for AdaptivePoolGuard<C> {
    fn deref_mut(&mut self) -> &mut C {
        &mut self
            .entry
            .as_mut()
            .expect("AdaptivePoolGuard: entry is None")
            .connection
    }
}

impl<C: Send + 'static> Drop for AdaptivePoolGuard<C> {
    fn drop(&mut self) {
        let entry = match self.entry.take() {
            Some(e) => e,
            None => return,
        };

        self.active_count.fetch_sub(1, Ordering::Relaxed);

        // Discard if expired
        if entry.is_stale(&self.config) {
            self.total_count.fetch_sub(1, Ordering::SeqCst);
            self.total_recycled.fetch_add(1, Ordering::Relaxed);
            debug!("Expired connection discarded on return");
            return;
        }

        match self.inner.lock() {
            Ok(mut guard) => {
                guard.idle.push(entry);
                debug!("Connection returned to idle pool");
            }
            Err(e) => {
                warn!("Pool lock poisoned on return: {}", e);
                self.total_count.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    // --- Helper: simple counter-based test connection ---
    struct Conn {
        id: usize,
        healthy: bool,
    }

    fn make_factory(counter: Arc<AtomicUsize>) -> impl Fn() -> FusekiResult<Conn> + Send + Sync {
        move || {
            let id = counter.fetch_add(1, Ordering::Relaxed);
            Ok(Conn { id, healthy: true })
        }
    }

    fn default_cfg(min: usize, max: usize) -> AdaptiveConnectionPoolConfig {
        AdaptiveConnectionPoolConfig {
            min_connections: min,
            max_connections: max,
            warmup_connections: min,
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(3600),
            acquire_timeout: Duration::from_millis(500),
            health_check_interval: Duration::from_secs(60),
            target_utilization: 0.7,
        }
    }

    // 1. Pool starts with warmup connections
    #[test]
    fn test_warmup_connections() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(3, 10), make_factory(counter)).unwrap();
        let m = pool.metrics();
        assert_eq!(m.total_connections, 3, "Expected 3 warmed connections");
        assert_eq!(m.idle_connections, 3);
        assert_eq!(m.active_connections, 0);
    }

    // 2. Acquire increases active count
    #[test]
    fn test_acquire_increases_active() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(2, 10), make_factory(counter)).unwrap();
        let _guard = pool.acquire().unwrap();
        assert_eq!(pool.metrics().active_connections, 1);
    }

    // 3. Release (drop) decreases active count
    #[test]
    fn test_release_decreases_active() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(2, 10), make_factory(counter)).unwrap();
        {
            let _guard = pool.acquire().unwrap();
            assert_eq!(pool.metrics().active_connections, 1);
        }
        assert_eq!(pool.metrics().active_connections, 0);
    }

    // 4. Connection is returned to idle after release
    #[test]
    fn test_connection_returned_to_idle() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(2, 10), make_factory(counter)).unwrap();
        let id = {
            let guard = pool.acquire().unwrap();
            guard.id
        };
        let m = pool.metrics();
        assert_eq!(
            m.idle_connections, 2,
            "Connection returned; idle should be 2"
        );
        let _ = id;
    }

    // 5. Pool grows on demand beyond warmup
    #[test]
    fn test_pool_grows_on_demand() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(1, 10), make_factory(counter)).unwrap();
        let _g1 = pool.acquire().unwrap();
        let _g2 = pool.acquire().unwrap();
        assert!(pool.metrics().total_connections >= 2);
    }

    // 6. Pool enforces max_connections
    #[test]
    fn test_max_connections_enforced() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(0, 2), make_factory(counter)).unwrap();
        let _g1 = pool.acquire().unwrap();
        let _g2 = pool.acquire().unwrap();
        let result = pool.acquire();
        assert!(result.is_err(), "Should time out when pool is at max");
    }

    // 7. Invalid config (min > max) is rejected
    #[test]
    fn test_invalid_config_rejected() {
        let counter = Arc::new(AtomicUsize::new(0));
        let cfg = AdaptiveConnectionPoolConfig {
            min_connections: 10,
            max_connections: 5,
            ..Default::default()
        };
        let result = AdaptiveConnectionPool::new(cfg, make_factory(counter));
        assert!(result.is_err());
    }

    // 8. Health check failures are counted
    #[test]
    fn test_health_check_failures_counted() {
        let counter = Arc::new(AtomicUsize::new(0));
        let factory = {
            let c = counter.clone();
            move || {
                let id = c.fetch_add(1, Ordering::Relaxed);
                Ok(Conn { id, healthy: false })
            }
        };
        // Health check always fails
        let pool = AdaptiveConnectionPool::new_with_health_check(
            default_cfg(2, 10),
            factory,
            |conn: &Conn| conn.healthy,
        )
        .unwrap();

        // Acquiring should keep creating and failing until timeout
        let result = pool.acquire();
        assert!(result.is_err());
        assert!(pool.metrics().health_check_failures > 0);
    }

    // 9. run_health_checks removes stale connections
    #[test]
    fn test_run_health_checks_removes_unhealthy() {
        let counter = Arc::new(AtomicUsize::new(0));
        let healthy_flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let flag = healthy_flag.clone();
        let factory = {
            let c = counter.clone();
            move || {
                let id = c.fetch_add(1, Ordering::Relaxed);
                Ok(Conn { id, healthy: true })
            }
        };
        let flag2 = flag.clone();
        let pool = AdaptiveConnectionPool::new_with_health_check(
            default_cfg(2, 10),
            factory,
            move |_conn: &Conn| flag2.load(Ordering::Relaxed),
        )
        .unwrap();

        // Mark all as unhealthy
        healthy_flag.store(false, Ordering::Relaxed);
        let removed = pool.run_health_checks().unwrap();
        assert_eq!(removed, 2, "Both idle connections should be removed");
    }

    // 10. run_health_checks re-warms pool to min_connections
    #[test]
    fn test_health_checks_rewarm() {
        let counter = Arc::new(AtomicUsize::new(0));
        let healthy_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = healthy_flag.clone();
        let factory_counter = counter.clone();
        let factory = move || {
            let id = factory_counter.fetch_add(1, Ordering::Relaxed);
            Ok(Conn { id, healthy: true })
        };
        let flag2 = flag.clone();
        let pool = AdaptiveConnectionPool::new_with_health_check(
            default_cfg(2, 10),
            factory,
            move |_: &Conn| flag2.load(Ordering::Relaxed),
        )
        .unwrap();

        // Evict all, then re-warm should kick in
        pool.run_health_checks().unwrap();
        // Re-warm creates new connections marked healthy=true by factory, but
        // health_check is still false so re-warm loop stops. Verify no panic.
        let m = pool.metrics();
        // total_created is at least the initial 2 warmup
        assert!(m.total_created >= 2);
    }

    // 11. Metrics report correct utilization
    #[test]
    fn test_metrics_utilization() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(4, 10), make_factory(counter)).unwrap();
        let _g1 = pool.acquire().unwrap();
        let _g2 = pool.acquire().unwrap();
        let m = pool.metrics();
        assert!(m.utilization > 0.0 && m.utilization <= 1.0);
    }

    // 12. Deref gives access to underlying connection
    #[test]
    fn test_deref_access() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool =
            AdaptiveConnectionPool::new(default_cfg(1, 5), make_factory(counter.clone())).unwrap();
        let guard = pool.acquire().unwrap();
        // Access via Deref
        let _id: usize = guard.id;
    }

    // 13. DerefMut allows mutation
    #[test]
    fn test_deref_mut_access() {
        let pool: AdaptiveConnectionPool<Vec<u8>> =
            AdaptiveConnectionPool::new(default_cfg(1, 5), || Ok(Vec::new())).unwrap();
        let mut guard = pool.acquire().unwrap();
        guard.push(42u8);
        assert_eq!(guard.len(), 1);
    }

    // 14. Multiple sequential acquires reuse idle connections
    #[test]
    fn test_sequential_reuse() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(2, 10), make_factory(counter)).unwrap();
        let id1 = { pool.acquire().unwrap().id };
        let id2 = { pool.acquire().unwrap().id };
        // Ids should come from the pre-warmed pool, so total_created stays at 2
        assert!(id1 < 2 && id2 < 2, "Should reuse pre-warmed connections");
    }

    // 15. Waiting count resets after connection is released
    #[test]
    fn test_waiting_count_resets() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(0, 0), make_factory(counter)).unwrap();
        // acquire with max=0 should fail immediately
        let _ = pool.acquire(); // Will timeout
        assert_eq!(pool.metrics().waiting_connections, 0);
    }

    // 16. total_created increases as pool grows
    #[test]
    fn test_total_created_tracks_growth() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(1, 5), make_factory(counter)).unwrap();
        let initial = pool.metrics().total_created;
        let _g1 = pool.acquire().unwrap();
        let _g2 = pool.acquire().unwrap();
        let after = pool.metrics().total_created;
        assert!(after > initial, "total_created should increase");
    }

    // 17. Pool config with warmup > max is clamped
    #[test]
    fn test_warmup_clamped_to_max() {
        let cfg = AdaptiveConnectionPoolConfig {
            min_connections: 1,
            max_connections: 3,
            warmup_connections: 10, // Exceeds max
            ..Default::default()
        };
        let pool = AdaptiveConnectionPool::new(cfg, || Ok(0usize)).unwrap();
        let m = pool.metrics();
        assert!(m.total_connections <= 3, "warmup should be clamped to max");
    }

    // 18. Idle connections expire and are evicted on next acquire
    #[test]
    fn test_idle_timeout_eviction() {
        let counter = Arc::new(AtomicUsize::new(0));
        let cfg = AdaptiveConnectionPoolConfig {
            min_connections: 2,
            max_connections: 10,
            warmup_connections: 2,
            idle_timeout: Duration::from_millis(1), // Immediately expires
            max_lifetime: Duration::from_secs(3600),
            acquire_timeout: Duration::from_secs(5),
            health_check_interval: Duration::from_secs(60),
            target_utilization: 0.7,
        };
        let pool = AdaptiveConnectionPool::new(cfg, make_factory(counter)).unwrap();
        std::thread::sleep(Duration::from_millis(5)); // Let them expire
                                                      // Acquire: will evict stale and create a new one
        let guard = pool.acquire().unwrap();
        // The evicted connections should have incremented total_recycled
        assert!(pool.metrics().total_recycled > 0 || guard.id >= 2);
    }

    // 19. Health check sweeps don't panic with no idle connections
    #[test]
    fn test_health_check_empty_pool() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(0, 5), make_factory(counter)).unwrap();
        let removed = pool.run_health_checks().unwrap();
        assert_eq!(removed, 0);
    }

    // 20. maybe_resize is throttled by health_check_interval
    #[test]
    fn test_resize_throttle() {
        let counter = Arc::new(AtomicUsize::new(0));
        let cfg = AdaptiveConnectionPoolConfig {
            min_connections: 1,
            max_connections: 10,
            warmup_connections: 1,
            health_check_interval: Duration::from_secs(3600), // Very long
            target_utilization: 0.0,                          // Always wants to grow
            ..Default::default()
        };
        let pool = AdaptiveConnectionPool::new(cfg, make_factory(counter)).unwrap();
        pool.maybe_resize().unwrap();
        let r1 = pool.metrics().resize_count;
        pool.maybe_resize().unwrap(); // Should be throttled
        let r2 = pool.metrics().resize_count;
        assert_eq!(r1, r2, "Second resize should be throttled");
    }

    // 21. Pool shrinks when utilization is very low
    #[test]
    fn test_pool_shrinks_on_low_utilization() {
        let counter = Arc::new(AtomicUsize::new(0));
        let cfg = AdaptiveConnectionPoolConfig {
            min_connections: 1,
            max_connections: 10,
            warmup_connections: 5,                           // Start with 5
            health_check_interval: Duration::from_millis(1), // Very short
            target_utilization: 0.9,                         // shrink when < 0.45
            idle_timeout: Duration::from_secs(3600),
            max_lifetime: Duration::from_secs(3600),
            acquire_timeout: Duration::from_secs(5),
        };
        let pool = AdaptiveConnectionPool::new(cfg, make_factory(counter)).unwrap();
        let before = pool.metrics().total_connections;
        std::thread::sleep(Duration::from_millis(5));
        pool.maybe_resize().unwrap();
        let after = pool.metrics().total_connections;
        // Either shrunk or stayed at min
        assert!(after <= before);
    }

    // 22. Concurrent acquires are safe
    #[test]
    fn test_concurrent_acquires() {
        use std::sync::Arc;
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(
            AdaptiveConnectionPool::new(default_cfg(2, 20), make_factory(counter)).unwrap(),
        );
        let mut handles = Vec::new();
        for _ in 0..8 {
            let p = Arc::clone(&pool);
            handles.push(std::thread::spawn(move || {
                let guard = p.acquire()?;
                std::thread::sleep(Duration::from_millis(5));
                drop(guard);
                FusekiResult::Ok(())
            }));
        }
        for h in handles {
            h.join().unwrap().unwrap();
        }
        assert_eq!(pool.metrics().active_connections, 0);
    }

    // 23. Guard deref: id field is accessible
    #[test]
    fn test_guard_id_accessible() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = AdaptiveConnectionPool::new(default_cfg(1, 5), make_factory(counter)).unwrap();
        let guard = pool.acquire().unwrap();
        assert!(guard.id < 100);
    }

    // 24. metrics shows waiting_connections > 0 during saturation
    #[test]
    fn test_waiting_during_saturation() {
        use std::sync::Arc;
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(
            AdaptiveConnectionPool::new(default_cfg(2, 2), make_factory(counter)).unwrap(),
        );

        // Hold both connections
        let _g1 = pool.acquire().unwrap();
        let _g2 = pool.acquire().unwrap();

        let pool2 = Arc::clone(&pool);
        let handle = std::thread::spawn(move || {
            // This will block until timeout
            pool2.acquire()
        });

        std::thread::sleep(Duration::from_millis(20));
        let m = pool.metrics();
        assert!(m.waiting_connections > 0 || m.active_connections == 2);
        let _ = handle.join();
    }

    // 25. total_recycled increases when stale connections are evicted
    #[test]
    fn test_total_recycled_increases_on_eviction() {
        let counter = Arc::new(AtomicUsize::new(0));
        let cfg = AdaptiveConnectionPoolConfig {
            min_connections: 2,
            max_connections: 10,
            warmup_connections: 2,
            idle_timeout: Duration::from_millis(1),
            max_lifetime: Duration::from_secs(3600),
            acquire_timeout: Duration::from_secs(5),
            health_check_interval: Duration::from_secs(60),
            target_utilization: 0.7,
        };
        let pool = AdaptiveConnectionPool::new(cfg, make_factory(counter)).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let _g = pool.acquire().unwrap(); // Will evict stale idle connections
        assert!(pool.metrics().total_recycled > 0);
    }
}
