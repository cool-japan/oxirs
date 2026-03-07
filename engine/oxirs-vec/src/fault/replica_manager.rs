//! Replica Manager – fault-tolerant shard replication for vector index shards.
//!
//! The `ReplicaManager` tracks which nodes hold replicas of each shard, monitors
//! health, and provides promotion/failure-marking operations needed for automatic
//! failover in a distributed vector search cluster.
//!
//! # Design
//!
//! - Each shard has exactly one **Primary** and zero or more **Replica** nodes.
//! - A replica in the `CatchingUp` state is receiving a delta stream from the primary
//!   and will be promoted to `Replica` once its progress reaches 1.0.
//! - When a node is marked `Failed`, the manager can elect a new primary from the
//!   remaining healthy replicas.
//! - `needs_rebalancing()` returns `true` when any shard is under- or over-replicated.

use crate::VectorError;
use std::collections::HashMap;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// State types
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of a single shard replica.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplicaState {
    /// This replica is the authoritative writer for the shard.
    Primary,
    /// This replica is fully synchronized and serves read requests.
    Replica,
    /// This replica is receiving a delta stream; `progress` is in `[0.0, 1.0]`.
    CatchingUp {
        /// Fraction of the sync work completed (0.0 = just started, 1.0 = done).
        progress: f64,
    },
    /// This replica has experienced a failure and is no longer serving requests.
    Failed,
}

impl ReplicaState {
    /// Returns `true` for states that can serve read requests.
    pub fn is_healthy(&self) -> bool {
        matches!(self, ReplicaState::Primary | ReplicaState::Replica)
    }

    /// Returns `true` if this replica is the primary.
    pub fn is_primary(&self) -> bool {
        matches!(self, ReplicaState::Primary)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ShardReplica
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata for a single replica of a shard.
#[derive(Debug, Clone)]
pub struct ShardReplica {
    /// Shard this replica belongs to.
    pub shard_id: u64,
    /// Unique identifier for this replica instance.
    pub replica_id: String,
    /// Cluster node that hosts this replica.
    pub node_id: String,
    /// Current lifecycle state.
    pub state: ReplicaState,
    /// Time of the last successful synchronization heartbeat.
    pub last_sync: Instant,
    /// Number of vectors currently indexed in this replica.
    pub vector_count: usize,
}

impl ShardReplica {
    /// Create a new replica description.
    pub fn new(
        shard_id: u64,
        replica_id: impl Into<String>,
        node_id: impl Into<String>,
        state: ReplicaState,
        vector_count: usize,
    ) -> Self {
        Self {
            shard_id,
            replica_id: replica_id.into(),
            node_id: node_id.into(),
            state,
            last_sync: Instant::now(),
            vector_count,
        }
    }

    /// Refresh the last-sync timestamp to now.
    pub fn touch(&mut self) {
        self.last_sync = Instant::now();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReplicationStatus
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate health report produced by [`ReplicaManager::replication_status`].
#[derive(Debug, Clone)]
pub struct ReplicationStatus {
    /// Total number of shards tracked.
    pub total_shards: usize,
    /// Shards with fewer healthy replicas than the target replication factor.
    pub under_replicated: usize,
    /// Shards with more healthy replicas than the target replication factor.
    pub over_replicated: usize,
    /// Total number of replicas in the `Failed` state across all shards.
    pub failed_replicas: usize,
    /// `true` when all shards are exactly at the target replication factor and
    /// no replicas are in the `Failed` state.
    pub healthy: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// ReplicaManager
// ─────────────────────────────────────────────────────────────────────────────

/// Central manager for shard-level replica tracking and failover coordination.
pub struct ReplicaManager {
    /// Map from `shard_id` to the list of its replicas.
    shards: HashMap<u64, Vec<ShardReplica>>,
    /// Desired number of healthy replicas per shard (including the primary).
    replication_factor: usize,
}

impl ReplicaManager {
    // ──────────────────────────────────────────────────────────────────────────
    // Construction
    // ──────────────────────────────────────────────────────────────────────────

    /// Create a new manager with the given target `replication_factor`.
    ///
    /// `replication_factor` must be at least 1 (1 = primary only, no replicas).
    pub fn new(replication_factor: usize) -> Self {
        let factor = replication_factor.max(1);
        Self {
            shards: HashMap::new(),
            replication_factor: factor,
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Registration
    // ──────────────────────────────────────────────────────────────────────────

    /// Register a new replica.
    ///
    /// # Errors
    ///
    /// Returns `VectorError::InvalidData` when:
    /// - A replica with the same `(shard_id, replica_id)` already exists.
    /// - `replica.state` is `Primary` but a primary is already registered for that shard.
    pub fn register_replica(&mut self, replica: ShardReplica) -> Result<(), VectorError> {
        let shard_id = replica.shard_id;
        let replica_id = replica.replica_id.clone();
        let is_primary = replica.state.is_primary();

        let entry = self.shards.entry(shard_id).or_default();

        // Reject duplicate replica IDs
        if entry.iter().any(|r| r.replica_id == replica_id) {
            return Err(VectorError::InvalidData(format!(
                "Replica '{}' for shard {} is already registered",
                replica_id, shard_id
            )));
        }

        // Reject duplicate primaries
        if is_primary && entry.iter().any(|r| r.state.is_primary()) {
            return Err(VectorError::InvalidData(format!(
                "Shard {} already has a primary; cannot register another",
                shard_id
            )));
        }

        entry.push(replica);
        Ok(())
    }

    /// Unregister a replica by `(shard_id, replica_id)`.
    ///
    /// Returns `true` if the replica was found and removed.
    pub fn unregister_replica(&mut self, shard_id: u64, replica_id: &str) -> bool {
        let Some(replicas) = self.shards.get_mut(&shard_id) else {
            return false;
        };
        let before = replicas.len();
        replicas.retain(|r| r.replica_id != replica_id);
        replicas.len() < before
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Promotion / failure
    // ──────────────────────────────────────────────────────────────────────────

    /// Promote `replica_id` to `Primary` for `shard_id`.
    ///
    /// Any existing primary is demoted to `Replica` first.
    ///
    /// # Errors
    ///
    /// Returns `VectorError::InvalidData` when the target replica is not found or
    /// is in the `Failed` state.
    pub fn promote_to_primary(
        &mut self,
        shard_id: u64,
        replica_id: &str,
    ) -> Result<(), VectorError> {
        let replicas = self
            .shards
            .get_mut(&shard_id)
            .ok_or_else(|| VectorError::InvalidData(format!("Shard {} not found", shard_id)))?;

        // Verify target exists and is healthy
        let target_exists = replicas.iter().any(|r| r.replica_id == replica_id);
        if !target_exists {
            return Err(VectorError::InvalidData(format!(
                "Replica '{}' not found in shard {}",
                replica_id, shard_id
            )));
        }

        let target_failed = replicas
            .iter()
            .find(|r| r.replica_id == replica_id)
            .map(|r| matches!(r.state, ReplicaState::Failed))
            .unwrap_or(false);
        if target_failed {
            return Err(VectorError::InvalidData(format!(
                "Cannot promote failed replica '{}' in shard {}",
                replica_id, shard_id
            )));
        }

        // Demote existing primary(s)
        for r in replicas.iter_mut() {
            if r.replica_id != replica_id && matches!(r.state, ReplicaState::Primary) {
                r.state = ReplicaState::Replica;
            }
        }

        // Promote target
        for r in replicas.iter_mut() {
            if r.replica_id == replica_id {
                r.state = ReplicaState::Primary;
                r.touch();
            }
        }

        Ok(())
    }

    /// Mark a replica as `Failed`.
    ///
    /// If the failed replica was the primary, no automatic failover occurs here —
    /// call `promote_to_primary` on a healthy replica to elect a new one.
    ///
    /// A no-op when the `(shard_id, replica_id)` pair does not exist.
    pub fn mark_failed(&mut self, shard_id: u64, replica_id: &str) {
        if let Some(replicas) = self.shards.get_mut(&shard_id) {
            for r in replicas.iter_mut() {
                if r.replica_id == replica_id {
                    r.state = ReplicaState::Failed;
                }
            }
        }
    }

    /// Attempt automatic failover for `shard_id` when the current primary fails.
    ///
    /// Selects the healthy replica with the largest `vector_count` (most up-to-date)
    /// and promotes it.  Returns the promoted replica's ID on success.
    ///
    /// # Errors
    ///
    /// Returns `VectorError::InvalidData` when no healthy replica exists for the shard.
    pub fn auto_failover(&mut self, shard_id: u64) -> Result<String, VectorError> {
        let best_id = {
            let replicas = self
                .shards
                .get(&shard_id)
                .ok_or_else(|| VectorError::InvalidData(format!("Shard {} not found", shard_id)))?;

            replicas
                .iter()
                .filter(|r| r.state.is_healthy() && !r.state.is_primary())
                .max_by_key(|r| r.vector_count)
                .map(|r| r.replica_id.clone())
                .ok_or_else(|| {
                    VectorError::InvalidData(format!(
                        "No healthy replica available to promote for shard {}",
                        shard_id
                    ))
                })?
        };

        self.promote_to_primary(shard_id, &best_id)?;
        Ok(best_id)
    }

    /// Update the `progress` of a `CatchingUp` replica.
    ///
    /// When `progress` reaches `1.0`, the replica is automatically promoted to `Replica`.
    pub fn update_sync_progress(&mut self, shard_id: u64, replica_id: &str, progress: f64) {
        let Some(replicas) = self.shards.get_mut(&shard_id) else {
            return;
        };
        for r in replicas.iter_mut() {
            if r.replica_id == replica_id {
                if progress >= 1.0 {
                    r.state = ReplicaState::Replica;
                } else {
                    r.state = ReplicaState::CatchingUp { progress };
                }
                r.touch();
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Queries
    // ──────────────────────────────────────────────────────────────────────────

    /// Return the primary replica for `shard_id`, if any.
    pub fn get_primary(&self, shard_id: u64) -> Option<&ShardReplica> {
        self.shards
            .get(&shard_id)?
            .iter()
            .find(|r| r.state.is_primary())
    }

    /// Return all replicas for `shard_id` (including primary).
    pub fn get_replicas(&self, shard_id: u64) -> Vec<&ShardReplica> {
        self.shards
            .get(&shard_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Return all healthy (non-failed) replicas for `shard_id`.
    pub fn get_healthy_replicas(&self, shard_id: u64) -> Vec<&ShardReplica> {
        self.shards
            .get(&shard_id)
            .map(|v| v.iter().filter(|r| r.state.is_healthy()).collect())
            .unwrap_or_default()
    }

    /// Return all tracked shard IDs.
    pub fn shard_ids(&self) -> Vec<u64> {
        self.shards.keys().cloned().collect()
    }

    /// The configured replication factor.
    pub fn replication_factor(&self) -> usize {
        self.replication_factor
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Rebalancing / status
    // ──────────────────────────────────────────────────────────────────────────

    /// Returns `true` when any shard is under- or over-replicated relative to
    /// the configured `replication_factor`.
    pub fn needs_rebalancing(&self) -> bool {
        self.shards.values().any(|replicas| {
            let healthy = replicas.iter().filter(|r| r.state.is_healthy()).count();
            healthy != self.replication_factor
        })
    }

    /// Produce a [`ReplicationStatus`] summary of the cluster's health.
    pub fn replication_status(&self) -> ReplicationStatus {
        let total_shards = self.shards.len();
        let mut under_replicated = 0usize;
        let mut over_replicated = 0usize;
        let mut failed_replicas = 0usize;

        for replicas in self.shards.values() {
            let healthy = replicas.iter().filter(|r| r.state.is_healthy()).count();
            let failed = replicas
                .iter()
                .filter(|r| matches!(r.state, ReplicaState::Failed))
                .count();

            failed_replicas += failed;

            match healthy.cmp(&self.replication_factor) {
                std::cmp::Ordering::Less => under_replicated += 1,
                std::cmp::Ordering::Greater => over_replicated += 1,
                std::cmp::Ordering::Equal => {}
            }
        }

        let healthy = under_replicated == 0 && over_replicated == 0 && failed_replicas == 0;

        ReplicationStatus {
            total_shards,
            under_replicated,
            over_replicated,
            failed_replicas,
            healthy,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn primary(shard: u64, rid: &str, node: &str) -> ShardReplica {
        ShardReplica::new(shard, rid, node, ReplicaState::Primary, 1000)
    }

    fn replica(shard: u64, rid: &str, node: &str) -> ShardReplica {
        ShardReplica::new(shard, rid, node, ReplicaState::Replica, 1000)
    }

    fn catching_up(shard: u64, rid: &str, node: &str, progress: f64) -> ShardReplica {
        ShardReplica::new(shard, rid, node, ReplicaState::CatchingUp { progress }, 500)
    }

    // ── Registration ──────────────────────────────────────────────────────────

    #[test]
    fn test_register_primary_and_replicas() {
        let mut mgr = ReplicaManager::new(3);

        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("primary");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("replica 1");
        mgr.register_replica(replica(1, "r2", "node-c"))
            .expect("replica 2");

        assert_eq!(mgr.get_replicas(1).len(), 3);
        assert!(mgr.get_primary(1).is_some());
    }

    #[test]
    fn test_duplicate_primary_rejected() {
        let mut mgr = ReplicaManager::new(2);

        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("first primary");
        let err = mgr.register_replica(primary(1, "r1", "node-b"));
        assert!(err.is_err(), "duplicate primary must be rejected");
    }

    #[test]
    fn test_duplicate_replica_id_rejected() {
        let mut mgr = ReplicaManager::new(2);

        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("first");
        let err = mgr.register_replica(replica(1, "r0", "node-b"));
        assert!(err.is_err(), "duplicate replica_id must be rejected");
    }

    // ── Promotion ─────────────────────────────────────────────────────────────

    #[test]
    fn test_promote_to_primary() {
        let mut mgr = ReplicaManager::new(2);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");

        mgr.promote_to_primary(1, "r1").expect("promote failed");

        let new_primary = mgr.get_primary(1).expect("primary should exist");
        assert_eq!(new_primary.replica_id, "r1");

        // Old primary should now be a replica
        let replicas = mgr.get_replicas(1);
        let old = replicas
            .iter()
            .find(|r| r.replica_id == "r0")
            .expect("r0 should still exist");
        assert!(matches!(old.state, ReplicaState::Replica));
    }

    #[test]
    fn test_promote_failed_replica_rejected() {
        let mut mgr = ReplicaManager::new(2);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");

        mgr.mark_failed(1, "r1");
        let err = mgr.promote_to_primary(1, "r1");
        assert!(err.is_err(), "promoting a failed replica must fail");
    }

    #[test]
    fn test_promote_nonexistent_replica_rejected() {
        let mut mgr = ReplicaManager::new(1);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");

        let err = mgr.promote_to_primary(1, "ghost");
        assert!(err.is_err());
    }

    // ── Failure marking ───────────────────────────────────────────────────────

    #[test]
    fn test_mark_failed() {
        let mut mgr = ReplicaManager::new(2);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");

        mgr.mark_failed(1, "r1");

        let replicas = mgr.get_replicas(1);
        let r1 = replicas
            .iter()
            .find(|r| r.replica_id == "r1")
            .expect("r1 exists");
        assert!(matches!(r1.state, ReplicaState::Failed));
    }

    #[test]
    fn test_mark_failed_noop_unknown() {
        // Should not panic when shard/replica doesn't exist
        let mut mgr = ReplicaManager::new(1);
        mgr.mark_failed(99, "ghost"); // no panic
    }

    // ── Auto failover ─────────────────────────────────────────────────────────

    #[test]
    fn test_auto_failover_selects_best_replica() {
        let mut mgr = ReplicaManager::new(3);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");

        // Register two replicas with different vector counts
        let mut r1 = replica(1, "r1", "node-b");
        r1.vector_count = 2000;
        let mut r2 = replica(1, "r2", "node-c");
        r2.vector_count = 1500;

        mgr.register_replica(r1).expect("ok");
        mgr.register_replica(r2).expect("ok");

        mgr.mark_failed(1, "r0");

        let promoted = mgr.auto_failover(1).expect("auto_failover failed");
        // r1 has the highest vector_count, should win
        assert_eq!(promoted, "r1");
    }

    #[test]
    fn test_auto_failover_fails_when_no_healthy_replica() {
        let mut mgr = ReplicaManager::new(2);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");

        mgr.mark_failed(1, "r0");
        mgr.mark_failed(1, "r1");

        let err = mgr.auto_failover(1);
        assert!(err.is_err(), "no healthy replica → should fail");
    }

    // ── Sync progress ─────────────────────────────────────────────────────────

    #[test]
    fn test_sync_progress_promotes_when_complete() {
        let mut mgr = ReplicaManager::new(2);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(catching_up(1, "r1", "node-b", 0.3))
            .expect("ok");

        mgr.update_sync_progress(1, "r1", 1.0);

        let replicas = mgr.get_replicas(1);
        let r1 = replicas.iter().find(|r| r.replica_id == "r1").expect("r1");
        assert!(matches!(r1.state, ReplicaState::Replica));
    }

    #[test]
    fn test_sync_progress_partial() {
        let mut mgr = ReplicaManager::new(2);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(catching_up(1, "r1", "node-b", 0.1))
            .expect("ok");

        mgr.update_sync_progress(1, "r1", 0.7);

        let replicas = mgr.get_replicas(1);
        let r1 = replicas.iter().find(|r| r.replica_id == "r1").expect("r1");
        if let ReplicaState::CatchingUp { progress } = r1.state {
            assert!((progress - 0.7).abs() < 1e-10);
        } else {
            panic!("Expected CatchingUp state");
        }
    }

    // ── Rebalancing / status ──────────────────────────────────────────────────

    #[test]
    fn test_needs_rebalancing_false_when_healthy() {
        let mut mgr = ReplicaManager::new(2);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");

        assert!(!mgr.needs_rebalancing());
    }

    #[test]
    fn test_needs_rebalancing_true_when_under_replicated() {
        let mut mgr = ReplicaManager::new(3);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        // Only 1 of 3 required replicas registered

        assert!(mgr.needs_rebalancing());
    }

    #[test]
    fn test_needs_rebalancing_true_when_over_replicated() {
        let mut mgr = ReplicaManager::new(1);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");

        // 2 healthy replicas but factor=1
        assert!(mgr.needs_rebalancing());
    }

    #[test]
    fn test_replication_status_healthy() {
        let mut mgr = ReplicaManager::new(2);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");

        let status = mgr.replication_status();
        assert_eq!(status.total_shards, 1);
        assert_eq!(status.under_replicated, 0);
        assert_eq!(status.over_replicated, 0);
        assert_eq!(status.failed_replicas, 0);
        assert!(status.healthy);
    }

    #[test]
    fn test_replication_status_with_failures() {
        let mut mgr = ReplicaManager::new(3);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");
        mgr.register_replica(replica(1, "r2", "node-c"))
            .expect("ok");
        mgr.mark_failed(1, "r2");

        let status = mgr.replication_status();
        assert!(!status.healthy);
        assert_eq!(status.failed_replicas, 1);
        assert_eq!(status.under_replicated, 1); // 2 healthy < 3 required
    }

    #[test]
    fn test_replication_status_multiple_shards() {
        let mut mgr = ReplicaManager::new(2);

        // Shard 1: healthy
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");

        // Shard 2: under-replicated
        mgr.register_replica(primary(2, "r0", "node-c"))
            .expect("ok");

        let status = mgr.replication_status();
        assert_eq!(status.total_shards, 2);
        assert_eq!(status.under_replicated, 1);
        assert!(!status.healthy);
    }

    #[test]
    fn test_unregister_replica() {
        let mut mgr = ReplicaManager::new(2);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");

        let removed = mgr.unregister_replica(1, "r1");
        assert!(removed);
        assert_eq!(mgr.get_replicas(1).len(), 1);
    }

    #[test]
    fn test_get_healthy_replicas() {
        let mut mgr = ReplicaManager::new(3);
        mgr.register_replica(primary(1, "r0", "node-a"))
            .expect("ok");
        mgr.register_replica(replica(1, "r1", "node-b"))
            .expect("ok");
        mgr.register_replica(replica(1, "r2", "node-c"))
            .expect("ok");
        mgr.mark_failed(1, "r2");

        let healthy = mgr.get_healthy_replicas(1);
        assert_eq!(healthy.len(), 2);
        assert!(healthy.iter().all(|r| r.state.is_healthy()));
    }
}
