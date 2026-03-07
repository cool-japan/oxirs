//! Enhanced cross-DC replication types
//!
//! Provides:
//! - `CrossDcReplicationPolicy` — configurable replication behaviour
//! - `DcTopology` — datacenter topology with latency matrix
//! - `CrossDcSyncManager` — orchestrates cross-DC synchronisation
//! - `BatchReplicator` — efficient bulk replication via write batches

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// CrossDcReplicationPolicy
// ---------------------------------------------------------------------------

/// Configuration policy for cross-datacenter replication behaviour.
///
/// Controls async vs. synchronous replication, maximum acceptable lag, and
/// whether quorum acknowledgement is required before confirming a write.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDcReplicationPolicy {
    /// When `true` replication proceeds asynchronously (fire-and-forget to remote DCs).
    /// When `false` the local DC waits for at least one remote DC to acknowledge.
    pub async_replication: bool,
    /// Maximum acceptable replication lag in milliseconds.
    /// Exceeding this threshold marks the stream as unhealthy.
    pub max_lag_ms: u64,
    /// When `true` a quorum of remote DCs must acknowledge before the write is
    /// considered durable. Only meaningful when `async_replication = false`.
    pub quorum_required: bool,
}

impl CrossDcReplicationPolicy {
    /// Create a policy with explicit settings
    pub fn new(async_replication: bool, max_lag_ms: u64, quorum_required: bool) -> Self {
        Self {
            async_replication,
            max_lag_ms,
            quorum_required,
        }
    }

    /// Create a fully asynchronous policy (lowest latency, highest risk)
    pub fn async_policy() -> Self {
        Self {
            async_replication: true,
            max_lag_ms: 10_000,
            quorum_required: false,
        }
    }

    /// Create a synchronous quorum policy (highest durability)
    pub fn sync_quorum_policy() -> Self {
        Self {
            async_replication: false,
            max_lag_ms: 1_000,
            quorum_required: true,
        }
    }

    /// Validate the policy configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_lag_ms == 0 {
            return Err(ClusterError::Config("max_lag_ms must be > 0".into()));
        }
        Ok(())
    }
}

impl Default for CrossDcReplicationPolicy {
    fn default() -> Self {
        Self::async_policy()
    }
}

// ---------------------------------------------------------------------------
// DcTopology
// ---------------------------------------------------------------------------

/// Datacenter topology with latency matrix for routing decisions.
///
/// Tracks the primary DC, replica DCs, and estimated round-trip latency
/// between all pairs of datacenters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcTopology {
    /// The primary / source datacenter
    pub primary_dc: String,
    /// List of replica datacenters
    pub replica_dcs: Vec<String>,
    /// Estimated latency in milliseconds between DC pairs.
    /// Key is `(from_dc, to_dc)`.
    pub latency_matrix: HashMap<(String, String), u64>,
}

impl DcTopology {
    /// Create a new topology with a given primary DC
    pub fn new(primary_dc: impl Into<String>) -> Self {
        Self {
            primary_dc: primary_dc.into(),
            replica_dcs: Vec::new(),
            latency_matrix: HashMap::new(),
        }
    }

    /// Add a replica DC
    pub fn add_replica_dc(&mut self, dc_id: impl Into<String>) {
        let id: String = dc_id.into();
        if !self.replica_dcs.contains(&id) {
            self.replica_dcs.push(id);
        }
    }

    /// Register estimated latency between two DCs (symmetric)
    pub fn set_latency(&mut self, from: impl Into<String>, to: impl Into<String>, latency_ms: u64) {
        let from: String = from.into();
        let to: String = to.into();
        self.latency_matrix
            .insert((from.clone(), to.clone()), latency_ms);
        self.latency_matrix.insert((to, from), latency_ms);
    }

    /// Return estimated latency from `from` to `to`.
    ///
    /// Returns `u64::MAX` if the pair is not in the matrix.
    pub fn estimated_latency(&self, from: &str, to: &str) -> u64 {
        if from == to {
            return 0;
        }
        *self
            .latency_matrix
            .get(&(from.to_string(), to.to_string()))
            .unwrap_or(&u64::MAX)
    }

    /// Return the closest DC to `from` from the given `candidates`.
    ///
    /// Returns `None` if candidates is empty.
    pub fn closest_dc<'a>(&self, from: &str, candidates: &'a [String]) -> Option<&'a str> {
        candidates
            .iter()
            .min_by_key(|dc| self.estimated_latency(from, dc))
            .map(|s| s.as_str())
    }

    /// All DCs in this topology (primary + replicas)
    pub fn all_dcs(&self) -> Vec<&str> {
        let mut dcs: Vec<&str> = vec![self.primary_dc.as_str()];
        dcs.extend(self.replica_dcs.iter().map(|s| s.as_str()));
        dcs
    }

    /// Total number of DCs (1 primary + N replicas)
    pub fn dc_count(&self) -> usize {
        1 + self.replica_dcs.len()
    }
}

// ---------------------------------------------------------------------------
// Sync result types
// ---------------------------------------------------------------------------

/// Result of a cross-DC shard synchronisation attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    pub shard_id: u64,
    pub from_dc: String,
    pub to_dc: String,
    /// Bytes transferred during the sync
    pub bytes_transferred: u64,
    /// Duration the sync took
    pub duration: Duration,
    pub success: bool,
    /// Optional error message if sync failed
    pub error: Option<String>,
}

impl SyncResult {
    pub fn success(
        shard_id: u64,
        from_dc: &str,
        to_dc: &str,
        bytes: u64,
        duration: Duration,
    ) -> Self {
        Self {
            shard_id,
            from_dc: from_dc.to_string(),
            to_dc: to_dc.to_string(),
            bytes_transferred: bytes,
            duration,
            success: true,
            error: None,
        }
    }

    pub fn failure(shard_id: u64, from_dc: &str, to_dc: &str, error: impl Into<String>) -> Self {
        Self {
            shard_id,
            from_dc: from_dc.to_string(),
            to_dc: to_dc.to_string(),
            bytes_transferred: 0,
            duration: Duration::ZERO,
            success: false,
            error: Some(error.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// DcHealthStatus
// ---------------------------------------------------------------------------

/// Health status of a datacenter from the perspective of the sync manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcHealthStatus {
    pub dc_id: String,
    /// Current replication lag in milliseconds
    pub lag_ms: u64,
    /// Whether the DC is within healthy lag threshold
    pub is_healthy: bool,
    /// Number of operations behind the primary
    pub behind_by: u64,
}

// ---------------------------------------------------------------------------
// CrossDcSyncManager
// ---------------------------------------------------------------------------

/// Orchestrates cross-DC synchronisation.
///
/// Maintains per-DC lag tracking and simulates shard-level synchronisation.
pub struct CrossDcSyncManager {
    local_dc: String,
    topology: DcTopology,
    policy: CrossDcReplicationPolicy,
    /// Per-DC tracking: dc_id → (lag_ms, behind_ops, last_sync)
    dc_state: HashMap<String, DcState>,
}

struct DcState {
    lag_ms: u64,
    behind_by: u64,
    _last_sync: Instant,
}

impl CrossDcSyncManager {
    /// Create a new sync manager
    pub fn new(
        local_dc: impl Into<String>,
        topology: DcTopology,
        policy: CrossDcReplicationPolicy,
    ) -> Result<Self> {
        policy.validate()?;
        let local_dc = local_dc.into();
        let dc_state = topology
            .all_dcs()
            .iter()
            .filter(|dc| **dc != local_dc)
            .map(|dc| {
                (
                    dc.to_string(),
                    DcState {
                        lag_ms: 0,
                        behind_by: 0,
                        _last_sync: Instant::now(),
                    },
                )
            })
            .collect();
        Ok(Self {
            local_dc,
            topology,
            policy,
            dc_state,
        })
    }

    /// Simulate synchronisation of a shard from `from_dc` to `to_dc`.
    ///
    /// In a real system this would initiate data transfer; here it models
    /// the latency cost and records the outcome.
    pub fn sync_shard(
        &mut self,
        shard_id: u64,
        from_dc: &str,
        to_dc: &str,
        policy: &CrossDcReplicationPolicy,
    ) -> SyncResult {
        if from_dc == to_dc {
            return SyncResult::failure(shard_id, from_dc, to_dc, "Cannot sync DC to itself");
        }
        if !self.topology.all_dcs().contains(&from_dc) {
            return SyncResult::failure(
                shard_id,
                from_dc,
                to_dc,
                format!("Unknown source DC '{}'", from_dc),
            );
        }
        if !self.topology.all_dcs().contains(&to_dc) {
            return SyncResult::failure(
                shard_id,
                from_dc,
                to_dc,
                format!("Unknown target DC '{}'", to_dc),
            );
        }

        let latency = self.topology.estimated_latency(from_dc, to_dc);
        // Simulate: each shard contains ~1 MB of data
        let simulated_bytes: u64 = 1_048_576;
        let simulated_duration = Duration::from_millis(latency.min(u16::MAX as u64));

        // Update target DC state
        if let Some(state) = self.dc_state.get_mut(to_dc) {
            // After successful sync, lag decreases
            state.lag_ms = state.lag_ms.saturating_sub(latency / 2);
            state.behind_by = state.behind_by.saturating_sub(1);
            state._last_sync = Instant::now();
        }

        let _ = policy; // policy used in real impl for consistency level
        SyncResult::success(
            shard_id,
            from_dc,
            to_dc,
            simulated_bytes,
            simulated_duration,
        )
    }

    /// Simulate accumulation of lag on a target DC (for testing health checks)
    pub fn simulate_lag(&mut self, dc_id: &str, lag_ms: u64, behind_by: u64) {
        if let Some(state) = self.dc_state.get_mut(dc_id) {
            state.lag_ms = lag_ms;
            state.behind_by = behind_by;
        }
    }

    /// Check replication health for all remote DCs.
    ///
    /// Returns a health status per DC.
    pub fn check_replication_health(&self) -> Vec<DcHealthStatus> {
        self.dc_state
            .iter()
            .map(|(dc_id, state)| DcHealthStatus {
                dc_id: dc_id.clone(),
                lag_ms: state.lag_ms,
                is_healthy: state.lag_ms <= self.policy.max_lag_ms,
                behind_by: state.behind_by,
            })
            .collect()
    }

    /// Get health status for a specific DC
    pub fn dc_health(&self, dc_id: &str) -> Option<DcHealthStatus> {
        self.dc_state.get(dc_id).map(|state| DcHealthStatus {
            dc_id: dc_id.to_string(),
            lag_ms: state.lag_ms,
            is_healthy: state.lag_ms <= self.policy.max_lag_ms,
            behind_by: state.behind_by,
        })
    }

    /// Return the local DC identifier
    pub fn local_dc(&self) -> &str {
        &self.local_dc
    }

    /// Return a reference to the topology
    pub fn topology(&self) -> &DcTopology {
        &self.topology
    }

    /// Return the active replication policy
    pub fn policy(&self) -> &CrossDcReplicationPolicy {
        &self.policy
    }

    /// Number of remote DCs being tracked
    pub fn remote_dc_count(&self) -> usize {
        self.dc_state.len()
    }
}

// ---------------------------------------------------------------------------
// FlushStats
// ---------------------------------------------------------------------------

/// Statistics for a single flush operation in `BatchReplicator`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlushStats {
    /// Total bytes sent in this flush
    pub bytes_sent: u64,
    /// Number of write operations in the batch
    pub operations: u64,
    /// Duration of the flush in milliseconds
    pub duration_ms: u64,
    /// Target DC identifier
    pub target_dc: String,
}

// ---------------------------------------------------------------------------
// BatchReplicator
// ---------------------------------------------------------------------------

/// Efficient bulk replication via write batches.
///
/// Accumulates individual writes in an in-memory buffer and flushes them
/// to a target DC as a single batch. This reduces per-operation overhead
/// significantly for high-throughput workloads.
pub struct BatchReplicator {
    /// Buffer: (key, value) pairs pending replication
    buffer: Vec<(Vec<u8>, Vec<u8>)>,
    /// Maximum buffer size in bytes before an automatic flush is triggered
    max_buffer_bytes: u64,
    /// Current buffer size in bytes
    current_bytes: u64,
}

impl BatchReplicator {
    /// Create a new batch replicator with default 4 MB buffer
    pub fn new() -> Self {
        Self::with_buffer_limit(4 * 1024 * 1024)
    }

    /// Create a batch replicator with a custom buffer limit in bytes
    pub fn with_buffer_limit(max_buffer_bytes: u64) -> Self {
        Self {
            buffer: Vec::new(),
            max_buffer_bytes: max_buffer_bytes.max(1),
            current_bytes: 0,
        }
    }

    /// Buffer a write for deferred replication.
    ///
    /// Returns `Err` if the buffer would exceed `max_buffer_bytes`.
    pub fn buffer_write(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let write_bytes = (key.len() + value.len()) as u64;
        if self.current_bytes + write_bytes > self.max_buffer_bytes {
            return Err(ClusterError::ResourceLimit(format!(
                "Batch buffer full: {} / {} bytes",
                self.current_bytes, self.max_buffer_bytes
            )));
        }
        self.current_bytes += write_bytes;
        self.buffer.push((key, value));
        Ok(())
    }

    /// Flush all buffered writes to the given DC.
    ///
    /// Clears the buffer and returns statistics about the flush operation.
    pub fn flush_to_dc(&mut self, dc_id: &str) -> Result<FlushStats> {
        if self.buffer.is_empty() {
            return Ok(FlushStats {
                bytes_sent: 0,
                operations: 0,
                duration_ms: 0,
                target_dc: dc_id.to_string(),
            });
        }

        let start = Instant::now();
        let bytes_sent = self.current_bytes;
        let operations = self.buffer.len() as u64;

        // Simulate network flush (in real impl this would serialize and send)
        self.buffer.clear();
        self.current_bytes = 0;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(FlushStats {
            bytes_sent,
            operations,
            duration_ms,
            target_dc: dc_id.to_string(),
        })
    }

    /// Number of pending write operations in the buffer
    pub fn pending_ops(&self) -> usize {
        self.buffer.len()
    }

    /// Current buffer usage in bytes
    pub fn buffer_bytes(&self) -> u64 {
        self.current_bytes
    }

    /// Whether the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Whether the buffer is at or near capacity (>= 90%)
    pub fn is_near_full(&self) -> bool {
        self.current_bytes >= (self.max_buffer_bytes * 9 / 10)
    }

    /// Maximum buffer size in bytes
    pub fn max_buffer_bytes(&self) -> u64 {
        self.max_buffer_bytes
    }
}

impl Default for BatchReplicator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── CrossDcReplicationPolicy ─────────────────────────────────────────────

    #[test]
    fn test_policy_async_defaults() {
        let p = CrossDcReplicationPolicy::async_policy();
        assert!(p.async_replication);
        assert!(!p.quorum_required);
        assert!(p.max_lag_ms > 0);
    }

    #[test]
    fn test_policy_sync_quorum_defaults() {
        let p = CrossDcReplicationPolicy::sync_quorum_policy();
        assert!(!p.async_replication);
        assert!(p.quorum_required);
        assert!(p.max_lag_ms > 0);
    }

    #[test]
    fn test_policy_validate_ok() {
        let p = CrossDcReplicationPolicy::new(true, 5000, false);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_policy_validate_zero_lag_fails() {
        let p = CrossDcReplicationPolicy::new(false, 0, true);
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_policy_default() {
        let p = CrossDcReplicationPolicy::default();
        assert!(p.async_replication);
    }

    // ── DcTopology ───────────────────────────────────────────────────────────

    fn make_topology() -> DcTopology {
        let mut t = DcTopology::new("us-east-1");
        t.add_replica_dc("eu-west-1");
        t.add_replica_dc("ap-southeast-1");
        t.set_latency("us-east-1", "eu-west-1", 80);
        t.set_latency("us-east-1", "ap-southeast-1", 150);
        t.set_latency("eu-west-1", "ap-southeast-1", 200);
        t
    }

    #[test]
    fn test_topology_dc_count() {
        let t = make_topology();
        assert_eq!(t.dc_count(), 3);
        assert_eq!(t.all_dcs().len(), 3);
    }

    #[test]
    fn test_topology_estimated_latency_symmetric() {
        let t = make_topology();
        assert_eq!(t.estimated_latency("us-east-1", "eu-west-1"), 80);
        assert_eq!(t.estimated_latency("eu-west-1", "us-east-1"), 80);
    }

    #[test]
    fn test_topology_latency_self_is_zero() {
        let t = make_topology();
        assert_eq!(t.estimated_latency("us-east-1", "us-east-1"), 0);
    }

    #[test]
    fn test_topology_latency_unknown_pair() {
        let t = make_topology();
        assert_eq!(t.estimated_latency("us-west-2", "eu-central-1"), u64::MAX);
    }

    #[test]
    fn test_topology_closest_dc() {
        let t = make_topology();
        let candidates = vec!["eu-west-1".to_string(), "ap-southeast-1".to_string()];
        let closest = t.closest_dc("us-east-1", &candidates);
        assert_eq!(
            closest,
            Some("eu-west-1"),
            "eu-west-1 is closer (80ms vs 150ms)"
        );
    }

    #[test]
    fn test_topology_closest_dc_empty_candidates() {
        let t = make_topology();
        let closest = t.closest_dc("us-east-1", &[]);
        assert!(closest.is_none());
    }

    #[test]
    fn test_topology_add_replica_idempotent() {
        let mut t = DcTopology::new("primary");
        t.add_replica_dc("replica-1");
        t.add_replica_dc("replica-1"); // duplicate
        assert_eq!(t.replica_dcs.len(), 1);
    }

    // ── CrossDcSyncManager ───────────────────────────────────────────────────

    fn make_sync_manager() -> CrossDcSyncManager {
        let topology = make_topology();
        let policy = CrossDcReplicationPolicy::async_policy();
        CrossDcSyncManager::new("us-east-1", topology, policy).expect("sync manager creation")
    }

    #[test]
    fn test_sync_manager_creation() {
        let mgr = make_sync_manager();
        assert_eq!(mgr.local_dc(), "us-east-1");
        assert_eq!(mgr.remote_dc_count(), 2);
    }

    #[test]
    fn test_sync_manager_sync_shard_success() {
        let mut mgr = make_sync_manager();
        let policy = CrossDcReplicationPolicy::async_policy();
        let result = mgr.sync_shard(1, "us-east-1", "eu-west-1", &policy);
        assert!(result.success);
        assert_eq!(result.shard_id, 1);
        assert_eq!(result.from_dc, "us-east-1");
        assert_eq!(result.to_dc, "eu-west-1");
        assert!(result.bytes_transferred > 0);
    }

    #[test]
    fn test_sync_manager_sync_shard_self_fails() {
        let mut mgr = make_sync_manager();
        let policy = CrossDcReplicationPolicy::async_policy();
        let result = mgr.sync_shard(1, "us-east-1", "us-east-1", &policy);
        assert!(!result.success);
    }

    #[test]
    fn test_sync_manager_sync_unknown_dc_fails() {
        let mut mgr = make_sync_manager();
        let policy = CrossDcReplicationPolicy::async_policy();
        let result = mgr.sync_shard(1, "us-east-1", "unknown-dc", &policy);
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_sync_manager_health_check_all_healthy() {
        let mgr = make_sync_manager();
        let health = mgr.check_replication_health();
        assert_eq!(health.len(), 2);
        for h in &health {
            assert!(h.is_healthy, "DC {} should be healthy with 0 lag", h.dc_id);
        }
    }

    #[test]
    fn test_sync_manager_health_check_with_lag() {
        let mut mgr = make_sync_manager();
        mgr.simulate_lag("eu-west-1", 20_000, 500); // 20 seconds lag
        let health = mgr.check_replication_health();
        let eu_health = health.iter().find(|h| h.dc_id == "eu-west-1");
        assert!(eu_health.is_some());
        let eu = eu_health.expect("eu-west-1 health");
        assert!(!eu.is_healthy, "eu-west-1 should be unhealthy with 20s lag");
        assert_eq!(eu.behind_by, 500);
    }

    #[test]
    fn test_sync_manager_dc_health_specific() {
        let mut mgr = make_sync_manager();
        mgr.simulate_lag("ap-southeast-1", 500, 10);
        let h = mgr.dc_health("ap-southeast-1").expect("dc health");
        assert_eq!(h.lag_ms, 500);
        assert!(h.is_healthy, "500ms is within 10s default max");
        assert_eq!(h.behind_by, 10);
    }

    #[test]
    fn test_sync_manager_multiple_shards() {
        let mut mgr = make_sync_manager();
        let policy = CrossDcReplicationPolicy::async_policy();
        for shard_id in 0..10 {
            let result = mgr.sync_shard(shard_id, "us-east-1", "eu-west-1", &policy);
            assert!(
                result.success,
                "Shard {} should sync successfully",
                shard_id
            );
        }
    }

    // ── BatchReplicator ──────────────────────────────────────────────────────

    #[test]
    fn test_batch_replicator_buffer_write() {
        let mut rep = BatchReplicator::new();
        rep.buffer_write(b"key1".to_vec(), b"value1".to_vec())
            .expect("buffer write");
        assert_eq!(rep.pending_ops(), 1);
        assert_eq!(rep.buffer_bytes(), 10); // 4 + 6
    }

    #[test]
    fn test_batch_replicator_flush_empty() {
        let mut rep = BatchReplicator::new();
        let stats = rep.flush_to_dc("eu-west-1").expect("flush");
        assert_eq!(stats.operations, 0);
        assert_eq!(stats.bytes_sent, 0);
    }

    #[test]
    fn test_batch_replicator_flush_clears_buffer() {
        let mut rep = BatchReplicator::new();
        rep.buffer_write(b"key".to_vec(), b"value".to_vec())
            .expect("write");
        rep.buffer_write(b"key2".to_vec(), b"value2".to_vec())
            .expect("write");
        assert_eq!(rep.pending_ops(), 2);

        let stats = rep.flush_to_dc("eu-west-1").expect("flush");
        assert_eq!(stats.operations, 2);
        assert!(stats.bytes_sent > 0);
        assert_eq!(stats.target_dc, "eu-west-1");
        assert!(rep.is_empty());
        assert_eq!(rep.pending_ops(), 0);
        assert_eq!(rep.buffer_bytes(), 0);
    }

    #[test]
    fn test_batch_replicator_buffer_overflow() {
        let mut rep = BatchReplicator::with_buffer_limit(20); // tiny limit
        rep.buffer_write(b"key-1".to_vec(), b"value-1".to_vec())
            .expect("write 1");
        // This should overflow the 20-byte limit
        let result = rep.buffer_write(b"key-long-enough-to-overflow".to_vec(), b"val".to_vec());
        assert!(result.is_err(), "Should fail on buffer overflow");
    }

    #[test]
    fn test_batch_replicator_near_full() {
        let mut rep = BatchReplicator::with_buffer_limit(100);
        // Write ~92 bytes (>= 90% of 100)
        rep.buffer_write(vec![0u8; 46], vec![0u8; 46])
            .expect("write");
        assert!(rep.is_near_full(), "Buffer at 92% should be near full");
    }

    #[test]
    fn test_batch_replicator_large_batch() {
        let mut rep = BatchReplicator::with_buffer_limit(10 * 1024 * 1024); // 10 MB
        for i in 0..1000_u64 {
            let key = format!("key-{}", i).into_bytes();
            let value = format!("value-{}", i).into_bytes();
            rep.buffer_write(key, value).expect("write");
        }
        assert_eq!(rep.pending_ops(), 1000);

        let stats = rep.flush_to_dc("ap-southeast-1").expect("flush");
        assert_eq!(stats.operations, 1000);
        assert!(stats.bytes_sent > 0);
        assert!(rep.is_empty());
    }

    #[test]
    fn test_batch_replicator_multiple_flushes() {
        let mut rep = BatchReplicator::new();
        for round in 0..5 {
            for i in 0..100_u64 {
                let key = format!("round-{}-key-{}", round, i).into_bytes();
                rep.buffer_write(key, b"v".to_vec()).expect("write");
            }
            let stats = rep.flush_to_dc("eu-west-1").expect("flush");
            assert_eq!(stats.operations, 100, "Round {} flush count", round);
            assert!(rep.is_empty());
        }
    }

    #[test]
    fn test_batch_replicator_default_is_4mb() {
        let rep = BatchReplicator::new();
        assert_eq!(rep.max_buffer_bytes(), 4 * 1024 * 1024);
    }

    // ── Integration ──────────────────────────────────────────────────────────

    #[test]
    fn test_sync_manager_with_strict_policy() {
        let topology = make_topology();
        let policy = CrossDcReplicationPolicy::sync_quorum_policy();
        let mut mgr = CrossDcSyncManager::new("us-east-1", topology, policy).expect("sync mgr");

        let strict_policy = CrossDcReplicationPolicy::sync_quorum_policy();
        let result = mgr.sync_shard(99, "us-east-1", "ap-southeast-1", &strict_policy);
        assert!(result.success);
    }

    #[test]
    fn test_topology_and_batch_replicator_integration() {
        let t = make_topology();
        let candidates: Vec<String> = t.replica_dcs.clone();
        let closest = t.closest_dc("us-east-1", &candidates).expect("closest dc");

        let mut rep = BatchReplicator::new();
        for i in 0..50_u64 {
            rep.buffer_write(
                format!("k-{}", i).into_bytes(),
                format!("v-{}", i).into_bytes(),
            )
            .expect("write");
        }
        let stats = rep.flush_to_dc(closest).expect("flush");
        assert_eq!(stats.operations, 50);
        assert_eq!(stats.target_dc, closest);
    }
}
