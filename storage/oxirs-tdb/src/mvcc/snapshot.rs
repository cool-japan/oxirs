//! Snapshot read and statistics types for the MVCC manager.
//!
//! Provides `MvccStats` and the public `snapshot_read` / `stats` / `low_water_mark`
//! methods that are free of write-path logic.

use super::transaction::{TxId, TX_ID_COMMITTED};

/// Statistics snapshot for the MVCC manager
#[derive(Debug, Clone, Default)]
pub struct MvccStats {
    /// Total transactions ever begun
    pub total_begun: u64,
    /// Total transactions committed
    pub total_committed: u64,
    /// Total transactions rolled back
    pub total_rolled_back: u64,
    /// Total transactions aborted (system-initiated)
    pub total_aborted: u64,
    /// Number of currently active transactions
    pub active_count: usize,
    /// Total version entries currently in the store
    pub total_versions: usize,
    /// Versions removed by vacuum
    pub versions_vacuumed: u64,
    /// Serialization conflicts detected
    pub serialization_conflicts: u64,
    /// Write-write conflicts detected
    pub write_conflicts: u64,
    /// Deadlocks detected
    pub deadlocks_detected: u64,
    /// Current low-water mark (oldest active snapshot TxId)
    pub watermark: TxId,
}

/// Compute the highest committed TxId from a set of committed IDs.
///
/// Returns `TX_ID_COMMITTED` (0) when the set is empty.
pub fn committed_watermark(committed: &std::collections::HashSet<TxId>) -> TxId {
    committed.iter().copied().max().unwrap_or(TX_ID_COMMITTED)
}
