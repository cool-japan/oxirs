//! # Consistency Protocols
//!
//! Configurable consistency levels for stream operations, including:
//!
//! - [`StreamConsistencyLevel`]: Consistency semantics (Eventual, ReadYourWrites, etc.)
//! - [`ConsistencyConfig`]: Configuration for read/write consistency with retries
//! - [`VersionedValue`]: A value tagged with version, timestamp, and origin node
//! - [`ConsistencyManager`]: Enforces monotonic-read and monotonic-write guarantees per session
//! - [`EventualConsistencyBuffer`]: Batches writes for eventual synchronisation

use std::collections::HashMap;

// ─── StreamConsistencyLevel ───────────────────────────────────────────────────

/// Consistency semantics for stream read and write operations.
///
/// These map to well-known distributed-systems consistency models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamConsistencyLevel {
    /// Replicas converge eventually; no ordering guarantees.
    Eventual,
    /// A client always reads values it previously wrote.
    ReadYourWrites,
    /// Read responses are non-decreasing in version for a session.
    MonotonicRead,
    /// Write responses are non-decreasing in version for a session.
    MonotonicWrite,
    /// All reads reflect all preceding writes globally.
    Strong,
    /// Operations appear instantaneous and globally ordered.
    Linearizable,
}

// ─── ConsistencyConfig ────────────────────────────────────────────────────────

/// Configures read/write consistency, timeouts, and retry behaviour.
#[derive(Debug, Clone)]
pub struct ConsistencyConfig {
    pub read_level: StreamConsistencyLevel,
    pub write_level: StreamConsistencyLevel,
    pub timeout_ms: u64,
    pub retry_count: usize,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            read_level: StreamConsistencyLevel::Eventual,
            write_level: StreamConsistencyLevel::Eventual,
            timeout_ms: 1000,
            retry_count: 3,
        }
    }
}

// ─── VersionedValue ───────────────────────────────────────────────────────────

/// A value tagged with a logical version, wall-clock timestamp, and originating node.
#[derive(Debug, Clone)]
pub struct VersionedValue<T> {
    pub value: T,
    pub version: u64,
    pub timestamp: i64,
    pub node_id: String,
}

impl<T> VersionedValue<T> {
    pub fn new(value: T, version: u64, timestamp: i64, node_id: impl Into<String>) -> Self {
        Self {
            value,
            version,
            timestamp,
            node_id: node_id.into(),
        }
    }
}

// ─── ConsistencyManager ───────────────────────────────────────────────────────

/// Enforces per-session monotonic-read and monotonic-write guarantees.
///
/// Sessions are identified by arbitrary string tokens (e.g., client IDs).
pub struct ConsistencyManager {
    config: ConsistencyConfig,
    /// session_id → minimum version that reads must satisfy (MonotonicRead)
    monotonic_read_version: HashMap<String, u64>,
    /// session_id → last write version (MonotonicWrite)
    monotonic_write_version: HashMap<String, u64>,
}

impl ConsistencyManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: ConsistencyConfig) -> Self {
        Self {
            config,
            monotonic_read_version: HashMap::new(),
            monotonic_write_version: HashMap::new(),
        }
    }

    /// Determine whether a read of `value` is acceptable for `session_id`.
    ///
    /// - `Eventual` / `ReadYourWrites` / `Strong` / `Linearizable`: always `true`
    /// - `MonotonicRead`: `value.version >= session's last-seen version`
    pub fn can_read<T>(&mut self, session_id: &str, value: &VersionedValue<T>) -> bool {
        match self.config.read_level {
            StreamConsistencyLevel::MonotonicRead => {
                let min_ver = self
                    .monotonic_read_version
                    .get(session_id)
                    .copied()
                    .unwrap_or(0);
                value.version >= min_ver
            }
            _ => true,
        }
    }

    /// Record that `session_id` has observed `value` (update monotonic-read floor).
    pub fn after_read<T>(&mut self, session_id: &str, value: &VersionedValue<T>) {
        let entry = self
            .monotonic_read_version
            .entry(session_id.to_string())
            .or_insert(0);
        if value.version > *entry {
            *entry = value.version;
        }
    }

    /// Determine whether a write of `value` is valid for `session_id`.
    ///
    /// - `MonotonicWrite`: `value.version >= session's last-write version`
    /// - everything else: always `true`
    pub fn can_write<T>(&mut self, session_id: &str, value: &VersionedValue<T>) -> bool {
        match self.config.write_level {
            StreamConsistencyLevel::MonotonicWrite => {
                let last_ver = self
                    .monotonic_write_version
                    .get(session_id)
                    .copied()
                    .unwrap_or(0);
                value.version >= last_ver
            }
            _ => true,
        }
    }

    /// Record that `session_id` performed a write at `value`'s version.
    pub fn after_write<T>(&mut self, session_id: &str, value: &VersionedValue<T>) {
        let entry = self
            .monotonic_write_version
            .entry(session_id.to_string())
            .or_insert(0);
        if value.version > *entry {
            *entry = value.version;
        }
    }

    /// Return `true` if `value.version` is strictly behind `current_version`
    /// (i.e., the value is stale).
    pub fn is_stale<T>(&self, value: &VersionedValue<T>, current_version: u64) -> bool {
        value.version < current_version
    }

    /// Reference to the active configuration.
    pub fn config(&self) -> &ConsistencyConfig {
        &self.config
    }

    /// How many sessions are tracked for monotonic-read.
    pub fn monotonic_read_session_count(&self) -> usize {
        self.monotonic_read_version.len()
    }

    /// How many sessions are tracked for monotonic-write.
    pub fn monotonic_write_session_count(&self) -> usize {
        self.monotonic_write_version.len()
    }
}

// ─── EventualConsistencyBuffer ────────────────────────────────────────────────

/// Buffers writes for eventual propagation, flushing when `max_lag_ms` has passed.
pub struct EventualConsistencyBuffer {
    pending: Vec<(String, Vec<u8>)>,
    max_lag_ms: u64,
    last_sync_ms: i64,
}

impl EventualConsistencyBuffer {
    /// Create a buffer that flushes after at most `max_lag_ms` milliseconds.
    pub fn new(max_lag_ms: u64) -> Self {
        Self {
            pending: Vec::new(),
            max_lag_ms,
            last_sync_ms: current_ms(),
        }
    }

    /// Stage a key-value write for eventual propagation.
    pub fn buffer(&mut self, key: &str, value: Vec<u8>) {
        self.pending.push((key.to_string(), value));
    }

    /// Return `true` when the lag since last sync exceeds `max_lag_ms`.
    pub fn should_sync(&self, now_ms: i64) -> bool {
        now_ms - self.last_sync_ms >= self.max_lag_ms as i64
    }

    /// Drain all pending writes and reset the sync timer.
    pub fn drain(&mut self) -> Vec<(String, Vec<u8>)> {
        self.last_sync_ms = current_ms();
        std::mem::take(&mut self.pending)
    }

    /// Number of writes currently buffered.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Helper: current wall-clock time in milliseconds.
fn current_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_value<T>(value: T, version: u64) -> VersionedValue<T> {
        VersionedValue::new(value, version, 1_700_000_000_000, "node-1")
    }

    // ── ConsistencyConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = ConsistencyConfig::default();
        assert_eq!(cfg.read_level, StreamConsistencyLevel::Eventual);
        assert_eq!(cfg.write_level, StreamConsistencyLevel::Eventual);
        assert_eq!(cfg.timeout_ms, 1000);
        assert_eq!(cfg.retry_count, 3);
    }

    #[test]
    fn test_custom_config() {
        let cfg = ConsistencyConfig {
            read_level: StreamConsistencyLevel::Strong,
            write_level: StreamConsistencyLevel::Linearizable,
            timeout_ms: 500,
            retry_count: 5,
        };
        assert_eq!(cfg.read_level, StreamConsistencyLevel::Strong);
        assert_eq!(cfg.write_level, StreamConsistencyLevel::Linearizable);
    }

    // ── VersionedValue ─────────────────────────────────────────────────────────

    #[test]
    fn test_versioned_value_fields() {
        let vv = make_value("hello", 42);
        assert_eq!(vv.value, "hello");
        assert_eq!(vv.version, 42);
        assert_eq!(vv.node_id, "node-1");
    }

    // ── ConsistencyManager (Eventual) ─────────────────────────────────────────

    #[test]
    fn test_eventual_always_allows_read_and_write() {
        let mut mgr = ConsistencyManager::new(ConsistencyConfig::default());
        let vv = make_value(1u32, 1);
        assert!(mgr.can_read("s1", &vv));
        assert!(mgr.can_write("s1", &vv));
    }

    // ── ConsistencyManager (MonotonicRead) ────────────────────────────────────

    #[test]
    fn test_monotonic_read_initial_allows_any_version() {
        let mut mgr = ConsistencyManager::new(ConsistencyConfig {
            read_level: StreamConsistencyLevel::MonotonicRead,
            ..Default::default()
        });
        let vv = make_value(0u32, 0);
        assert!(mgr.can_read("sess", &vv));
    }

    #[test]
    fn test_monotonic_read_blocks_regression() {
        let mut mgr = ConsistencyManager::new(ConsistencyConfig {
            read_level: StreamConsistencyLevel::MonotonicRead,
            ..Default::default()
        });
        let v5 = make_value(0u32, 5);
        mgr.after_read("sess", &v5);

        // version 4 is a regression – must be blocked
        let v4 = make_value(0u32, 4);
        assert!(!mgr.can_read("sess", &v4));

        // version 5 is same level – must be allowed
        assert!(mgr.can_read("sess", &v5));

        // version 6 advances – must be allowed
        let v6 = make_value(0u32, 6);
        assert!(mgr.can_read("sess", &v6));
    }

    #[test]
    fn test_monotonic_read_sessions_independent() {
        let mut mgr = ConsistencyManager::new(ConsistencyConfig {
            read_level: StreamConsistencyLevel::MonotonicRead,
            ..Default::default()
        });
        let v10 = make_value(0u32, 10);
        mgr.after_read("session-A", &v10);

        // session-B has no history; version 1 is fine
        let v1 = make_value(0u32, 1);
        assert!(mgr.can_read("session-B", &v1));
    }

    #[test]
    fn test_after_read_tracks_max_version() {
        let mut mgr = ConsistencyManager::new(ConsistencyConfig {
            read_level: StreamConsistencyLevel::MonotonicRead,
            ..Default::default()
        });
        mgr.after_read("s", &make_value(0u32, 3));
        mgr.after_read("s", &make_value(0u32, 7));
        mgr.after_read("s", &make_value(0u32, 5)); // lower – should be ignored

        // After seeing version 7, the monotonic floor is 7.
        // v7 (== floor) must be allowed
        let v7 = make_value(0u32, 7);
        assert!(mgr.can_read("s", &v7));
        // v8 (> floor) must be allowed
        let v8 = make_value(0u32, 8);
        assert!(mgr.can_read("s", &v8));
        // v4 (< floor) must be blocked
        let v4 = make_value(0u32, 4);
        assert!(!mgr.can_read("s", &v4));
        // v6 (< floor) must also be blocked
        let v6 = make_value(0u32, 6);
        assert!(!mgr.can_read("s", &v6));
    }

    // ── ConsistencyManager (MonotonicWrite) ───────────────────────────────────

    #[test]
    fn test_monotonic_write_initial_allows_any_version() {
        let mut mgr = ConsistencyManager::new(ConsistencyConfig {
            write_level: StreamConsistencyLevel::MonotonicWrite,
            ..Default::default()
        });
        let vv = make_value(0u32, 0);
        assert!(mgr.can_write("sess", &vv));
    }

    #[test]
    fn test_monotonic_write_blocks_regression() {
        let mut mgr = ConsistencyManager::new(ConsistencyConfig {
            write_level: StreamConsistencyLevel::MonotonicWrite,
            ..Default::default()
        });
        let v5 = make_value(0u32, 5);
        mgr.after_write("sess", &v5);

        let v4 = make_value(0u32, 4);
        assert!(!mgr.can_write("sess", &v4));

        let v5b = make_value(0u32, 5);
        assert!(mgr.can_write("sess", &v5b));

        let v6 = make_value(0u32, 6);
        assert!(mgr.can_write("sess", &v6));
    }

    // ── ConsistencyManager (is_stale) ─────────────────────────────────────────

    #[test]
    fn test_is_stale_behind_current() {
        let mgr = ConsistencyManager::new(ConsistencyConfig::default());
        let vv = make_value(0u32, 4);
        assert!(mgr.is_stale(&vv, 10));
    }

    #[test]
    fn test_is_not_stale_at_current() {
        let mgr = ConsistencyManager::new(ConsistencyConfig::default());
        let vv = make_value(0u32, 10);
        assert!(!mgr.is_stale(&vv, 10));
    }

    #[test]
    fn test_is_not_stale_ahead_current() {
        let mgr = ConsistencyManager::new(ConsistencyConfig::default());
        let vv = make_value(0u32, 12);
        assert!(!mgr.is_stale(&vv, 10));
    }

    // ── ConsistencyManager (session counts) ───────────────────────────────────

    #[test]
    fn test_session_tracking_counts() {
        let mut mgr = ConsistencyManager::new(ConsistencyConfig {
            read_level: StreamConsistencyLevel::MonotonicRead,
            write_level: StreamConsistencyLevel::MonotonicWrite,
            ..Default::default()
        });
        assert_eq!(mgr.monotonic_read_session_count(), 0);
        assert_eq!(mgr.monotonic_write_session_count(), 0);

        mgr.after_read("r-sess", &make_value(0u32, 1));
        mgr.after_write("w-sess", &make_value(0u32, 1));

        assert_eq!(mgr.monotonic_read_session_count(), 1);
        assert_eq!(mgr.monotonic_write_session_count(), 1);
    }

    // ── EventualConsistencyBuffer ──────────────────────────────────────────────

    #[test]
    fn test_buffer_initial_empty() {
        let buf = EventualConsistencyBuffer::new(500);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_buffer_and_count() {
        let mut buf = EventualConsistencyBuffer::new(500);
        buf.buffer("key1", b"val1".to_vec());
        buf.buffer("key2", b"val2".to_vec());
        assert_eq!(buf.pending_count(), 2);
    }

    #[test]
    fn test_buffer_drain() {
        let mut buf = EventualConsistencyBuffer::new(500);
        buf.buffer("k", b"v".to_vec());
        let drained = buf.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, "k");
        assert_eq!(drained[0].1, b"v");
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_should_sync_past_deadline() {
        let buf = EventualConsistencyBuffer::new(100);
        let far_future = current_ms() + 10_000; // 10 seconds in the future
        assert!(buf.should_sync(far_future));
    }

    #[test]
    fn test_should_not_sync_before_deadline() {
        let buf = EventualConsistencyBuffer::new(60_000); // 60 second lag
        let now = current_ms();
        assert!(!buf.should_sync(now));
    }

    #[test]
    fn test_drain_resets_timer() {
        let mut buf = EventualConsistencyBuffer::new(100);
        buf.buffer("x", b"1".to_vec());
        let _ = buf.drain();
        // immediately after drain the timer is reset; now_ms == last_sync_ms ≈ now
        let now = current_ms();
        assert!(!buf.should_sync(now));
    }
}
