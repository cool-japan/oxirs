//! MVCC (Multi-Version Concurrency Control) snapshot isolation
//!
//! This module implements snapshot isolation using MVCC to provide high concurrency
//! without blocking reads.

use super::TransactionId;
use crate::model::Quad;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// MVCC snapshot for a transaction
#[derive(Debug, Clone)]
pub struct MvccSnapshot {
    /// Transaction ID of this snapshot
    tx_id: TransactionId,
    /// Timestamp when snapshot was created
    timestamp: u64,
    /// Active transaction IDs at snapshot time
    active_txs: Vec<TransactionId>,
    /// Visible quad versions
    visible_versions: Arc<HashMap<Quad, VersionedQuad>>,
}

impl MvccSnapshot {
    /// Create a new MVCC snapshot
    pub fn new(tx_id: TransactionId, timestamp: u64, active_txs: Vec<TransactionId>) -> Self {
        Self {
            tx_id,
            timestamp,
            active_txs,
            visible_versions: Arc::new(HashMap::new()),
        }
    }

    /// Check if a transaction is visible in this snapshot
    pub fn is_visible(&self, version_tx_id: TransactionId) -> bool {
        // A version is visible if:
        // 1. It was created by this transaction, OR
        // 2. It was committed before this snapshot AND the creating tx is not in active_txs
        if version_tx_id == self.tx_id {
            return true;
        }

        // Check if the version's transaction was active when snapshot was taken
        !self.active_txs.contains(&version_tx_id)
    }

    /// Get the snapshot's transaction ID
    pub fn tx_id(&self) -> TransactionId {
        self.tx_id
    }

    /// Get the snapshot timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Get visible version of a quad
    pub fn get_visible_version(&self, quad: &Quad) -> Option<&VersionedQuad> {
        self.visible_versions.get(quad)
    }
}

/// Versioned quad for MVCC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedQuad {
    /// The quad data
    pub quad: Quad,
    /// Version number (transaction ID)
    pub version: u64,
    /// Transaction that created this version
    pub created_by: TransactionId,
    /// Transaction that deleted this version (if any)
    pub deleted_by: Option<TransactionId>,
    /// Whether this version is committed
    pub committed: bool,
}

impl VersionedQuad {
    /// Create a new versioned quad
    pub fn new(quad: Quad, version: u64, created_by: TransactionId) -> Self {
        Self {
            quad,
            version,
            created_by,
            deleted_by: None,
            committed: false,
        }
    }

    /// Check if this version is visible to a snapshot
    pub fn is_visible_to(&self, snapshot: &MvccSnapshot) -> bool {
        // Not visible if deleted by a transaction visible to the snapshot
        if let Some(deleted_by) = self.deleted_by {
            if snapshot.is_visible(deleted_by) {
                return false;
            }
        }

        // Visible if created by a transaction visible to the snapshot
        snapshot.is_visible(self.created_by)
    }

    /// Mark this version as committed
    pub fn mark_committed(&mut self) {
        self.committed = true;
    }

    /// Mark this version as deleted by a transaction
    pub fn mark_deleted(&mut self, tx_id: TransactionId) {
        self.deleted_by = Some(tx_id);
    }
}

/// Snapshot manager for MVCC
pub struct SnapshotManager {
    /// Next timestamp
    next_timestamp: u64,
    /// Active snapshots
    active_snapshots: HashMap<TransactionId, MvccSnapshot>,
    /// All quad versions
    quad_versions: HashMap<Quad, Vec<VersionedQuad>>,
}

impl SnapshotManager {
    /// Create a new snapshot manager
    pub fn new() -> Self {
        Self {
            next_timestamp: 0,
            active_snapshots: HashMap::new(),
            quad_versions: HashMap::new(),
        }
    }

    /// Create a snapshot for a transaction
    pub fn create_snapshot(&mut self, tx_id: TransactionId) -> MvccSnapshot {
        let timestamp = self.next_timestamp;
        self.next_timestamp += 1;

        // Get currently active transaction IDs (excluding this one)
        let active_txs: Vec<TransactionId> = self
            .active_snapshots
            .keys()
            .filter(|&id| *id != tx_id)
            .copied()
            .collect();

        let active_count = active_txs.len();
        let snapshot = MvccSnapshot::new(tx_id, timestamp, active_txs);

        self.active_snapshots.insert(tx_id, snapshot.clone());

        tracing::debug!(
            "Created snapshot for tx {} at timestamp {} with {} active transactions",
            tx_id.0,
            timestamp,
            active_count
        );

        snapshot
    }

    /// Release a snapshot
    pub fn release_snapshot(&mut self, tx_id: TransactionId) {
        self.active_snapshots.remove(&tx_id);
        tracing::debug!("Released snapshot for tx {}", tx_id.0);
    }

    /// Add a quad version
    pub fn add_version(&mut self, versioned_quad: VersionedQuad) {
        let quad = versioned_quad.quad.clone();
        self.quad_versions
            .entry(quad)
            .or_default()
            .push(versioned_quad);
    }

    /// Get all versions of a quad
    pub fn get_versions(&self, quad: &Quad) -> Option<&Vec<VersionedQuad>> {
        self.quad_versions.get(quad)
    }

    /// Get the visible version of a quad for a snapshot
    pub fn get_visible_version(
        &self,
        quad: &Quad,
        snapshot: &MvccSnapshot,
    ) -> Option<&VersionedQuad> {
        let versions = self.quad_versions.get(quad)?;

        // Find the newest version visible to this snapshot
        versions
            .iter()
            .filter(|v| v.is_visible_to(snapshot))
            .max_by_key(|v| v.version)
    }

    /// Garbage collect old versions
    pub fn gc_old_versions(&mut self, min_active_timestamp: u64) {
        let mut removed_count = 0;

        for versions in self.quad_versions.values_mut() {
            // Keep only versions that might still be visible
            let original_len = versions.len();
            versions.retain(|v| {
                // Keep if:
                // 1. Version is newer than or equal to min active timestamp
                // 2. Version is not committed yet
                // 3. Version is the only one (keep at least one version)
                v.version >= min_active_timestamp || !v.committed || original_len == 1
            });

            removed_count += original_len - versions.len();
        }

        if removed_count > 0 {
            tracing::info!(
                "Garbage collected {} old quad versions (min timestamp: {})",
                removed_count,
                min_active_timestamp
            );
        }
    }

    /// Get statistics about version storage
    pub fn version_stats(&self) -> VersionStats {
        let total_versions: usize = self.quad_versions.values().map(|v| v.len()).sum();
        let unique_quads = self.quad_versions.len();
        let avg_versions_per_quad = if unique_quads > 0 {
            total_versions as f64 / unique_quads as f64
        } else {
            0.0
        };

        VersionStats {
            total_versions,
            unique_quads,
            avg_versions_per_quad,
            active_snapshots: self.active_snapshots.len(),
        }
    }
}

impl Default for SnapshotManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Version storage statistics
#[derive(Debug, Clone)]
pub struct VersionStats {
    /// Total number of versions stored
    pub total_versions: usize,
    /// Number of unique quads
    pub unique_quads: usize,
    /// Average versions per quad
    pub avg_versions_per_quad: f64,
    /// Number of active snapshots
    pub active_snapshots: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GraphName, Literal, NamedNode, Object, Predicate, Subject};

    fn create_test_quad(id: usize) -> Quad {
        Quad::new(
            Subject::NamedNode(NamedNode::new(format!("http://s{}", id)).unwrap()),
            Predicate::NamedNode(NamedNode::new(format!("http://p{}", id)).unwrap()),
            Object::Literal(Literal::new(format!("value{}", id))),
            GraphName::DefaultGraph,
        )
    }

    #[test]
    fn test_snapshot_creation() {
        let mut mgr = SnapshotManager::new();

        let snapshot = mgr.create_snapshot(TransactionId(1));
        assert_eq!(snapshot.tx_id(), TransactionId(1));
        assert_eq!(snapshot.timestamp(), 0);
    }

    #[test]
    fn test_visibility() {
        let snapshot = MvccSnapshot::new(
            TransactionId(10),
            100,
            vec![TransactionId(5), TransactionId(8)],
        );

        // Own transaction is visible
        assert!(snapshot.is_visible(TransactionId(10)));

        // Active transactions are not visible
        assert!(!snapshot.is_visible(TransactionId(5)));
        assert!(!snapshot.is_visible(TransactionId(8)));

        // Other transactions are visible (assumed committed)
        assert!(snapshot.is_visible(TransactionId(1)));
        assert!(snapshot.is_visible(TransactionId(15)));
    }

    #[test]
    fn test_versioned_quad() {
        let quad = create_test_quad(1);
        let mut v_quad = VersionedQuad::new(quad, 10, TransactionId(5));

        assert!(!v_quad.committed);
        v_quad.mark_committed();
        assert!(v_quad.committed);

        assert!(v_quad.deleted_by.is_none());
        v_quad.mark_deleted(TransactionId(20));
        assert_eq!(v_quad.deleted_by, Some(TransactionId(20)));
    }

    #[test]
    fn test_snapshot_manager_versions() {
        let mut mgr = SnapshotManager::new();

        let quad = create_test_quad(1);
        let v_quad1 = VersionedQuad::new(quad.clone(), 1, TransactionId(1));
        let v_quad2 = VersionedQuad::new(quad.clone(), 2, TransactionId(2));

        mgr.add_version(v_quad1);
        mgr.add_version(v_quad2);

        let versions = mgr.get_versions(&quad).unwrap();
        assert_eq!(versions.len(), 2);
    }

    #[test]
    fn test_garbage_collection() {
        let mut mgr = SnapshotManager::new();

        let quad = create_test_quad(1);

        // Add old committed version
        let mut v_quad1 = VersionedQuad::new(quad.clone(), 1, TransactionId(1));
        v_quad1.mark_committed();
        mgr.add_version(v_quad1);

        // Add newer version
        let v_quad2 = VersionedQuad::new(quad.clone(), 100, TransactionId(100));
        mgr.add_version(v_quad2);

        // GC with min timestamp 50 should remove version 1
        mgr.gc_old_versions(50);

        let versions = mgr.get_versions(&quad).unwrap();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].version, 100);
    }

    #[test]
    fn test_version_stats() {
        let mut mgr = SnapshotManager::new();

        let quad1 = create_test_quad(1);
        let quad2 = create_test_quad(2);

        mgr.add_version(VersionedQuad::new(quad1.clone(), 1, TransactionId(1)));
        mgr.add_version(VersionedQuad::new(quad1, 2, TransactionId(2)));
        mgr.add_version(VersionedQuad::new(quad2, 1, TransactionId(1)));

        let stats = mgr.version_stats();
        assert_eq!(stats.total_versions, 3);
        assert_eq!(stats.unique_quads, 2);
        assert_eq!(stats.avg_versions_per_quad, 1.5);
    }
}
