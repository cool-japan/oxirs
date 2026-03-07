//! Database checkpoint management for TDB storage.
//!
//! Provides periodic and on-demand checkpoint creation, WAL truncation after
//! successful checkpoints, recovery from the latest checkpoint, scheduling
//! policies (time-based and write-count-based), incremental checkpointing,
//! metadata tracking, integrity verification, and checkpoint history.

use std::collections::HashMap;

// ── CheckpointTrigger ────────────────────────────────────────────────────────

/// Policy that determines when a checkpoint should be created.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointTrigger {
    /// Trigger after a fixed number of seconds since the last checkpoint.
    TimeBased {
        /// Interval in seconds between checkpoints.
        interval_secs: u64,
    },
    /// Trigger after a certain number of writes since the last checkpoint.
    WriteCountBased {
        /// Number of writes before a checkpoint is triggered.
        threshold: u64,
    },
    /// Trigger on either time or write count, whichever comes first.
    Combined {
        /// Interval in seconds.
        interval_secs: u64,
        /// Write-count threshold.
        write_threshold: u64,
    },
    /// Never trigger automatically; checkpoints are created manually.
    Manual,
}

impl Default for CheckpointTrigger {
    fn default() -> Self {
        Self::Combined {
            interval_secs: 300,
            write_threshold: 10_000,
        }
    }
}

// ── CheckpointConfig ─────────────────────────────────────────────────────────

/// Configuration for the checkpoint manager.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Scheduling trigger policy.
    pub trigger: CheckpointTrigger,
    /// Maximum number of historical checkpoints to retain.
    pub max_history: usize,
    /// Whether to enable incremental checkpoints (only changed pages).
    pub incremental: bool,
    /// Whether to verify checkpoints after creation.
    pub verify_after_create: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            trigger: CheckpointTrigger::default(),
            max_history: 10,
            incremental: false,
            verify_after_create: true,
        }
    }
}

// ── CheckpointStatus ─────────────────────────────────────────────────────────

/// Status of a checkpoint operation.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointStatus {
    /// The checkpoint completed successfully.
    Success,
    /// The checkpoint is currently in progress.
    InProgress,
    /// The checkpoint failed with an error description.
    Failed(String),
    /// The checkpoint was verified and found to be consistent.
    Verified,
    /// The checkpoint was verified and found to be corrupt.
    Corrupt(String),
}

// ── CheckpointMetadata ───────────────────────────────────────────────────────

/// Metadata describing a single checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Unique identifier for this checkpoint.
    pub id: u64,
    /// Unix-epoch timestamp (seconds) when this checkpoint was created.
    pub created_at_secs: u64,
    /// WAL position (log sequence number) at the time of the checkpoint.
    pub wal_position: u64,
    /// Total size of the checkpoint data (bytes).
    pub data_size_bytes: u64,
    /// Number of pages included in this checkpoint.
    pub page_count: u64,
    /// Whether this is an incremental checkpoint.
    pub is_incremental: bool,
    /// Status of the checkpoint.
    pub status: CheckpointStatus,
    /// CRC32-style checksum over the checkpoint data.
    pub checksum: u32,
    /// IDs of dirty pages that were flushed (only for incremental).
    pub dirty_page_ids: Vec<u64>,
}

// ── PageData ─────────────────────────────────────────────────────────────────

/// A page included in a checkpoint snapshot.
#[derive(Debug, Clone)]
pub struct PageData {
    /// The page identifier.
    pub page_id: u64,
    /// Raw bytes of the page.
    pub data: Vec<u8>,
    /// Whether this page has been modified since the previous checkpoint.
    pub dirty: bool,
}

impl PageData {
    /// Create a new page snapshot.
    pub fn new(page_id: u64, data: Vec<u8>) -> Self {
        Self {
            page_id,
            data,
            dirty: false,
        }
    }

    /// Create a dirty page snapshot.
    pub fn dirty(page_id: u64, data: Vec<u8>) -> Self {
        Self {
            page_id,
            data,
            dirty: true,
        }
    }
}

// ── WalEntry ─────────────────────────────────────────────────────────────────

/// A write-ahead log entry that can be replayed during recovery.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Log sequence number.
    pub lsn: u64,
    /// Page id this write affects.
    pub page_id: u64,
    /// The data written.
    pub data: Vec<u8>,
    /// Whether this entry represents a committed transaction.
    pub committed: bool,
}

// ── CheckpointStats ──────────────────────────────────────────────────────────

/// Cumulative statistics tracked by the checkpoint manager.
#[derive(Debug, Clone, Default)]
pub struct CheckpointStats {
    /// Total number of checkpoints created.
    pub total_checkpoints: u64,
    /// Number of successful checkpoints.
    pub successful_checkpoints: u64,
    /// Number of failed checkpoints.
    pub failed_checkpoints: u64,
    /// Number of incremental checkpoints.
    pub incremental_checkpoints: u64,
    /// Number of full (non-incremental) checkpoints.
    pub full_checkpoints: u64,
    /// Total pages flushed across all checkpoints.
    pub total_pages_flushed: u64,
    /// Total bytes written across all checkpoints.
    pub total_bytes_written: u64,
    /// Number of WAL truncations performed.
    pub wal_truncations: u64,
    /// Number of recovery operations.
    pub recoveries: u64,
    /// Number of verification operations.
    pub verifications: u64,
}

// ── RecoveryResult ───────────────────────────────────────────────────────────

/// Result of a checkpoint-based recovery.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// The checkpoint id from which recovery started.
    pub checkpoint_id: u64,
    /// Number of WAL entries replayed after the checkpoint.
    pub wal_entries_replayed: u64,
    /// Number of pages restored from the checkpoint.
    pub pages_restored: u64,
    /// Whether the recovery completed successfully.
    pub success: bool,
    /// Error description if the recovery failed.
    pub error: Option<String>,
}

// ── CheckpointManager ────────────────────────────────────────────────────────

/// Manages database checkpoints for crash recovery and WAL truncation.
///
/// The checkpoint manager periodically (or on-demand) snapshots the current
/// database state, allowing the WAL to be truncated and speeding up recovery.
pub struct CheckpointManager {
    config: CheckpointConfig,
    /// Historical checkpoint metadata, keyed by checkpoint id.
    history: Vec<CheckpointMetadata>,
    /// Stored page data for the latest full checkpoint.
    full_snapshot_pages: HashMap<u64, PageData>,
    /// WAL entries that have not yet been checkpointed.
    pending_wal: Vec<WalEntry>,
    /// Next checkpoint id to assign.
    next_id: u64,
    /// Current simulated time (seconds since epoch).
    current_time_secs: u64,
    /// Number of writes since the last checkpoint.
    writes_since_checkpoint: u64,
    /// Cumulative statistics.
    stats: CheckpointStats,
    /// Pages that were modified since the last checkpoint (for incremental).
    dirty_pages: HashMap<u64, PageData>,
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new(CheckpointConfig::default())
    }
}

impl CheckpointManager {
    /// Create a new checkpoint manager with the given configuration.
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            full_snapshot_pages: HashMap::new(),
            pending_wal: Vec::new(),
            next_id: 1,
            current_time_secs: 0,
            writes_since_checkpoint: 0,
            stats: CheckpointStats::default(),
            dirty_pages: HashMap::new(),
        }
    }

    /// Advance the simulated clock.
    pub fn set_time(&mut self, secs: u64) {
        self.current_time_secs = secs;
    }

    /// Return the current configuration.
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Return cumulative statistics.
    pub fn stats(&self) -> &CheckpointStats {
        &self.stats
    }

    /// Return checkpoint history.
    pub fn history(&self) -> &[CheckpointMetadata] {
        &self.history
    }

    /// Return the most recent successful checkpoint metadata, if any.
    pub fn latest_checkpoint(&self) -> Option<&CheckpointMetadata> {
        self.history.iter().rev().find(|cp| {
            cp.status == CheckpointStatus::Success || cp.status == CheckpointStatus::Verified
        })
    }

    /// Record a page modification (used for incremental checkpointing).
    pub fn mark_page_dirty(&mut self, page: PageData) {
        self.dirty_pages.insert(page.page_id, page);
        self.writes_since_checkpoint += 1;
    }

    /// Append a WAL entry to the pending queue.
    pub fn append_wal(&mut self, entry: WalEntry) {
        self.pending_wal.push(entry);
        self.writes_since_checkpoint += 1;
    }

    /// Load a full set of pages as the baseline snapshot.
    pub fn load_pages(&mut self, pages: Vec<PageData>) {
        self.full_snapshot_pages.clear();
        for page in pages {
            self.full_snapshot_pages.insert(page.page_id, page);
        }
    }

    /// Check whether a checkpoint should be triggered based on the current
    /// trigger policy and state.
    pub fn should_checkpoint(&self) -> bool {
        let last_time = self
            .latest_checkpoint()
            .map(|cp| cp.created_at_secs)
            .unwrap_or(0);

        match &self.config.trigger {
            CheckpointTrigger::TimeBased { interval_secs } => {
                self.current_time_secs.saturating_sub(last_time) >= *interval_secs
            }
            CheckpointTrigger::WriteCountBased { threshold } => {
                self.writes_since_checkpoint >= *threshold
            }
            CheckpointTrigger::Combined {
                interval_secs,
                write_threshold,
            } => {
                self.current_time_secs.saturating_sub(last_time) >= *interval_secs
                    || self.writes_since_checkpoint >= *write_threshold
            }
            CheckpointTrigger::Manual => false,
        }
    }

    /// Create a full checkpoint: snapshot all pages and record metadata.
    ///
    /// Returns the checkpoint id on success.
    pub fn create_full_checkpoint(&mut self) -> Result<u64, String> {
        let id = self.next_id;
        self.next_id += 1;
        self.stats.total_checkpoints += 1;

        // Merge dirty pages into full snapshot
        for (pid, page) in self.dirty_pages.drain() {
            self.full_snapshot_pages.insert(pid, page);
        }

        let page_count = self.full_snapshot_pages.len() as u64;
        let data_size: u64 = self
            .full_snapshot_pages
            .values()
            .map(|p| p.data.len() as u64)
            .sum();
        let wal_pos = self.pending_wal.last().map(|e| e.lsn).unwrap_or(0);
        let checksum = self.compute_full_checksum();

        let meta = CheckpointMetadata {
            id,
            created_at_secs: self.current_time_secs,
            wal_position: wal_pos,
            data_size_bytes: data_size,
            page_count,
            is_incremental: false,
            status: CheckpointStatus::Success,
            checksum,
            dirty_page_ids: Vec::new(),
        };

        self.history.push(meta);
        self.stats.successful_checkpoints += 1;
        self.stats.full_checkpoints += 1;
        self.stats.total_pages_flushed += page_count;
        self.stats.total_bytes_written += data_size;
        self.writes_since_checkpoint = 0;

        // Auto-verify if configured
        if self.config.verify_after_create {
            self.verify_checkpoint(id)?;
        }

        // Trim history
        self.trim_history();

        Ok(id)
    }

    /// Create an incremental checkpoint: snapshot only dirty pages.
    ///
    /// Returns the checkpoint id on success, or an error if there is no prior
    /// full checkpoint to build upon.
    pub fn create_incremental_checkpoint(&mut self) -> Result<u64, String> {
        if self.history.is_empty() {
            return Err(
                "Cannot create incremental checkpoint without a prior full checkpoint".into(),
            );
        }

        let id = self.next_id;
        self.next_id += 1;
        self.stats.total_checkpoints += 1;

        let dirty_ids: Vec<u64> = self.dirty_pages.keys().copied().collect();
        let page_count = dirty_ids.len() as u64;
        let data_size: u64 = self.dirty_pages.values().map(|p| p.data.len() as u64).sum();
        let wal_pos = self.pending_wal.last().map(|e| e.lsn).unwrap_or(0);

        // Merge dirty pages into full snapshot
        for (pid, page) in self.dirty_pages.drain() {
            self.full_snapshot_pages.insert(pid, page);
        }

        let checksum = self.compute_incremental_checksum(&dirty_ids);

        let meta = CheckpointMetadata {
            id,
            created_at_secs: self.current_time_secs,
            wal_position: wal_pos,
            data_size_bytes: data_size,
            page_count,
            is_incremental: true,
            status: CheckpointStatus::Success,
            checksum,
            dirty_page_ids: dirty_ids,
        };

        self.history.push(meta);
        self.stats.successful_checkpoints += 1;
        self.stats.incremental_checkpoints += 1;
        self.stats.total_pages_flushed += page_count;
        self.stats.total_bytes_written += data_size;
        self.writes_since_checkpoint = 0;

        if self.config.verify_after_create {
            self.verify_checkpoint(id)?;
        }

        self.trim_history();

        Ok(id)
    }

    /// Create a checkpoint (full or incremental) according to configuration.
    pub fn create_checkpoint(&mut self) -> Result<u64, String> {
        if self.config.incremental && !self.history.is_empty() {
            self.create_incremental_checkpoint()
        } else {
            self.create_full_checkpoint()
        }
    }

    /// Checkpoint on shutdown: flush all pending data and create a full checkpoint.
    pub fn checkpoint_on_shutdown(&mut self) -> Result<u64, String> {
        // Apply all committed WAL entries to the page state
        let committed: Vec<WalEntry> = self
            .pending_wal
            .iter()
            .filter(|e| e.committed)
            .cloned()
            .collect();
        for entry in &committed {
            self.full_snapshot_pages.insert(
                entry.page_id,
                PageData::new(entry.page_id, entry.data.clone()),
            );
        }
        // Clear dirty pages as we're doing a full flush
        self.dirty_pages.clear();
        self.create_full_checkpoint()
    }

    /// Truncate WAL entries that are covered by the given checkpoint.
    ///
    /// Removes all pending WAL entries with LSN <= the checkpoint's WAL position.
    pub fn truncate_wal(&mut self, checkpoint_id: u64) -> Result<u64, String> {
        let wal_pos = self
            .history
            .iter()
            .find(|cp| cp.id == checkpoint_id)
            .map(|cp| cp.wal_position)
            .ok_or_else(|| format!("Checkpoint {} not found", checkpoint_id))?;

        let before = self.pending_wal.len() as u64;
        self.pending_wal.retain(|e| e.lsn > wal_pos);
        let removed = before - self.pending_wal.len() as u64;
        self.stats.wal_truncations += 1;
        Ok(removed)
    }

    /// Recover the database state from the latest checkpoint plus WAL replay.
    ///
    /// Returns a `RecoveryResult` describing what was recovered.
    pub fn recover(&mut self) -> Result<RecoveryResult, String> {
        self.stats.recoveries += 1;

        let checkpoint = self
            .latest_checkpoint()
            .cloned()
            .ok_or_else(|| "No checkpoint available for recovery".to_string())?;

        // Replay WAL entries after the checkpoint's position
        let to_replay: Vec<WalEntry> = self
            .pending_wal
            .iter()
            .filter(|e| e.lsn > checkpoint.wal_position && e.committed)
            .cloned()
            .collect();

        let replayed_count = to_replay.len() as u64;

        for entry in &to_replay {
            self.full_snapshot_pages.insert(
                entry.page_id,
                PageData::new(entry.page_id, entry.data.clone()),
            );
        }

        Ok(RecoveryResult {
            checkpoint_id: checkpoint.id,
            wal_entries_replayed: replayed_count,
            pages_restored: checkpoint.page_count,
            success: true,
            error: None,
        })
    }

    /// Recover from a specific checkpoint by id.
    pub fn recover_from(&mut self, checkpoint_id: u64) -> Result<RecoveryResult, String> {
        self.stats.recoveries += 1;

        let checkpoint = self
            .history
            .iter()
            .find(|cp| cp.id == checkpoint_id)
            .cloned()
            .ok_or_else(|| format!("Checkpoint {} not found", checkpoint_id))?;

        if checkpoint.status == CheckpointStatus::Failed("".to_string()) {
            return Err(format!("Checkpoint {} is in failed state", checkpoint_id));
        }

        let to_replay: Vec<WalEntry> = self
            .pending_wal
            .iter()
            .filter(|e| e.lsn > checkpoint.wal_position && e.committed)
            .cloned()
            .collect();

        let replayed_count = to_replay.len() as u64;

        for entry in &to_replay {
            self.full_snapshot_pages.insert(
                entry.page_id,
                PageData::new(entry.page_id, entry.data.clone()),
            );
        }

        Ok(RecoveryResult {
            checkpoint_id: checkpoint.id,
            wal_entries_replayed: replayed_count,
            pages_restored: checkpoint.page_count,
            success: true,
            error: None,
        })
    }

    /// Verify a checkpoint's integrity by checking its checksum.
    pub fn verify_checkpoint(&mut self, checkpoint_id: u64) -> Result<bool, String> {
        self.stats.verifications += 1;

        let idx = self
            .history
            .iter()
            .position(|cp| cp.id == checkpoint_id)
            .ok_or_else(|| format!("Checkpoint {} not found", checkpoint_id))?;

        let meta = &self.history[idx];
        let expected = meta.checksum;

        let actual = if meta.is_incremental {
            self.compute_incremental_checksum(&meta.dirty_page_ids.clone())
        } else {
            self.compute_full_checksum()
        };

        if actual == expected {
            self.history[idx].status = CheckpointStatus::Verified;
            Ok(true)
        } else {
            self.history[idx].status = CheckpointStatus::Corrupt(format!(
                "Checksum mismatch: expected {}, got {}",
                expected, actual
            ));
            Ok(false)
        }
    }

    /// Return a page's current data from the full snapshot, if present.
    pub fn get_page(&self, page_id: u64) -> Option<&PageData> {
        self.full_snapshot_pages.get(&page_id)
    }

    /// Return all page IDs currently in the snapshot.
    pub fn snapshot_page_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.full_snapshot_pages.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Return the number of pending WAL entries.
    pub fn pending_wal_count(&self) -> usize {
        self.pending_wal.len()
    }

    /// Return the number of dirty pages.
    pub fn dirty_page_count(&self) -> usize {
        self.dirty_pages.len()
    }

    /// Get checkpoint metadata by id.
    pub fn get_checkpoint(&self, id: u64) -> Option<&CheckpointMetadata> {
        self.history.iter().find(|cp| cp.id == id)
    }

    /// Record a failed checkpoint attempt.
    pub fn record_failure(&mut self, reason: &str) {
        let id = self.next_id;
        self.next_id += 1;
        self.stats.total_checkpoints += 1;
        self.stats.failed_checkpoints += 1;

        let meta = CheckpointMetadata {
            id,
            created_at_secs: self.current_time_secs,
            wal_position: 0,
            data_size_bytes: 0,
            page_count: 0,
            is_incremental: false,
            status: CheckpointStatus::Failed(reason.to_string()),
            checksum: 0,
            dirty_page_ids: Vec::new(),
        };
        self.history.push(meta);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Compute a simple additive checksum over all full-snapshot pages.
    fn compute_full_checksum(&self) -> u32 {
        let mut hash: u32 = 0;
        let mut ids: Vec<u64> = self.full_snapshot_pages.keys().copied().collect();
        ids.sort();
        for id in ids {
            if let Some(page) = self.full_snapshot_pages.get(&id) {
                hash = hash.wrapping_add(id as u32);
                for &b in &page.data {
                    hash = hash.wrapping_mul(31).wrapping_add(u32::from(b));
                }
            }
        }
        hash
    }

    /// Compute a checksum over only the given page ids.
    fn compute_incremental_checksum(&self, page_ids: &[u64]) -> u32 {
        let mut hash: u32 = 0;
        let mut sorted_ids = page_ids.to_vec();
        sorted_ids.sort();
        for id in sorted_ids {
            if let Some(page) = self.full_snapshot_pages.get(&id) {
                hash = hash.wrapping_add(id as u32);
                for &b in &page.data {
                    hash = hash.wrapping_mul(31).wrapping_add(u32::from(b));
                }
            }
        }
        hash
    }

    /// Remove old checkpoints beyond `max_history`.
    fn trim_history(&mut self) {
        while self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_page(id: u64, data: &[u8]) -> PageData {
        PageData::new(id, data.to_vec())
    }

    fn make_dirty_page(id: u64, data: &[u8]) -> PageData {
        PageData::dirty(id, data.to_vec())
    }

    fn make_wal(lsn: u64, page_id: u64, data: &[u8], committed: bool) -> WalEntry {
        WalEntry {
            lsn,
            page_id,
            data: data.to_vec(),
            committed,
        }
    }

    // ── Basic construction ───────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = CheckpointConfig::default();
        assert_eq!(cfg.max_history, 10);
        assert!(!cfg.incremental);
        assert!(cfg.verify_after_create);
    }

    #[test]
    fn test_default_manager() {
        let mgr = CheckpointManager::default();
        assert!(mgr.history().is_empty());
        assert_eq!(mgr.stats().total_checkpoints, 0);
    }

    #[test]
    fn test_new_manager_with_config() {
        let cfg = CheckpointConfig {
            trigger: CheckpointTrigger::Manual,
            max_history: 5,
            incremental: true,
            verify_after_create: false,
        };
        let mgr = CheckpointManager::new(cfg.clone());
        assert_eq!(mgr.config().max_history, 5);
        assert!(mgr.config().incremental);
    }

    // ── Trigger policies ─────────────────────────────────────────────────

    #[test]
    fn test_manual_trigger_never_fires() {
        let cfg = CheckpointConfig {
            trigger: CheckpointTrigger::Manual,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.set_time(999_999);
        mgr.writes_since_checkpoint = 999_999;
        assert!(!mgr.should_checkpoint());
    }

    #[test]
    fn test_time_based_trigger() {
        let cfg = CheckpointConfig {
            trigger: CheckpointTrigger::TimeBased { interval_secs: 60 },
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.set_time(100);
        assert!(mgr.should_checkpoint()); // no prior checkpoint => fires

        mgr.load_pages(vec![make_page(1, b"data")]);
        let _ = mgr.create_full_checkpoint();

        mgr.set_time(150);
        assert!(!mgr.should_checkpoint()); // 50s < 60s

        mgr.set_time(161);
        assert!(mgr.should_checkpoint()); // 61s >= 60s
    }

    #[test]
    fn test_write_count_trigger() {
        let cfg = CheckpointConfig {
            trigger: CheckpointTrigger::WriteCountBased { threshold: 5 },
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        assert!(!mgr.should_checkpoint()); // 0 writes

        for i in 0..4 {
            mgr.append_wal(make_wal(i + 1, 1, b"x", true));
        }
        assert!(!mgr.should_checkpoint()); // 4 < 5

        mgr.append_wal(make_wal(5, 1, b"x", true));
        assert!(mgr.should_checkpoint()); // 5 >= 5
    }

    #[test]
    fn test_combined_trigger_time() {
        let cfg = CheckpointConfig {
            trigger: CheckpointTrigger::Combined {
                interval_secs: 30,
                write_threshold: 100,
            },
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.set_time(31);
        assert!(mgr.should_checkpoint()); // time alone fires
    }

    #[test]
    fn test_combined_trigger_writes() {
        let cfg = CheckpointConfig {
            trigger: CheckpointTrigger::Combined {
                interval_secs: 9999,
                write_threshold: 3,
            },
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.set_time(1);
        for i in 0..3 {
            mgr.append_wal(make_wal(i + 1, 1, b"x", true));
        }
        assert!(mgr.should_checkpoint()); // write-count fires
    }

    // ── Full checkpoint ──────────────────────────────────────────────────

    #[test]
    fn test_full_checkpoint_basic() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"aaa"), make_page(2, b"bbb")]);
        mgr.set_time(100);

        let id = mgr.create_full_checkpoint().expect("should succeed");
        assert_eq!(id, 1);
        assert_eq!(mgr.stats().successful_checkpoints, 1);
        assert_eq!(mgr.stats().full_checkpoints, 1);
        assert_eq!(mgr.history().len(), 1);

        let meta = mgr.get_checkpoint(id).expect("exists");
        assert_eq!(meta.created_at_secs, 100);
        assert_eq!(meta.page_count, 2);
        assert!(!meta.is_incremental);
    }

    #[test]
    fn test_full_checkpoint_with_verify() {
        let mut mgr = CheckpointManager::new(CheckpointConfig::default());
        mgr.load_pages(vec![make_page(1, b"hello")]);
        let id = mgr.create_full_checkpoint().expect("should succeed");

        let meta = mgr.get_checkpoint(id).expect("exists");
        assert_eq!(meta.status, CheckpointStatus::Verified);
        assert_eq!(mgr.stats().verifications, 1);
    }

    #[test]
    fn test_full_checkpoint_resets_write_counter() {
        let cfg = CheckpointConfig {
            trigger: CheckpointTrigger::WriteCountBased { threshold: 5 },
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        for i in 0..5 {
            mgr.append_wal(make_wal(i + 1, 1, b"x", true));
        }
        assert!(mgr.should_checkpoint());

        mgr.load_pages(vec![make_page(1, b"x")]);
        let _ = mgr.create_full_checkpoint();
        assert!(!mgr.should_checkpoint());
    }

    #[test]
    fn test_full_checkpoint_records_wal_position() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.append_wal(make_wal(10, 1, b"x", true));
        mgr.append_wal(make_wal(20, 2, b"y", true));
        mgr.load_pages(vec![make_page(1, b"x")]);

        let id = mgr.create_full_checkpoint().expect("ok");
        let meta = mgr.get_checkpoint(id).expect("exists");
        assert_eq!(meta.wal_position, 20);
    }

    // ── Incremental checkpoint ───────────────────────────────────────────

    #[test]
    fn test_incremental_requires_prior_full() {
        let cfg = CheckpointConfig {
            incremental: true,
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        let result = mgr.create_incremental_checkpoint();
        assert!(result.is_err());
    }

    #[test]
    fn test_incremental_checkpoint_only_dirty() {
        let cfg = CheckpointConfig {
            incremental: true,
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"a"), make_page(2, b"b")]);
        let _ = mgr.create_full_checkpoint();

        // Modify page 2 only
        mgr.mark_page_dirty(make_dirty_page(2, b"b_updated"));
        let id = mgr.create_incremental_checkpoint().expect("ok");

        let meta = mgr.get_checkpoint(id).expect("exists");
        assert!(meta.is_incremental);
        assert_eq!(meta.page_count, 1);
        assert_eq!(meta.dirty_page_ids, vec![2]);
    }

    #[test]
    fn test_incremental_stats_tracking() {
        let cfg = CheckpointConfig {
            incremental: true,
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"a")]);
        let _ = mgr.create_full_checkpoint();

        mgr.mark_page_dirty(make_dirty_page(1, b"a2"));
        let _ = mgr.create_incremental_checkpoint();

        assert_eq!(mgr.stats().full_checkpoints, 1);
        assert_eq!(mgr.stats().incremental_checkpoints, 1);
        assert_eq!(mgr.stats().total_checkpoints, 2);
    }

    #[test]
    fn test_create_checkpoint_auto_selects_mode() {
        let cfg = CheckpointConfig {
            incremental: true,
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"a")]);

        // First call: no history => full
        let _ = mgr.create_checkpoint();
        assert_eq!(mgr.stats().full_checkpoints, 1);

        // Second call: has history => incremental
        mgr.mark_page_dirty(make_dirty_page(1, b"a2"));
        let _ = mgr.create_checkpoint();
        assert_eq!(mgr.stats().incremental_checkpoints, 1);
    }

    // ── WAL truncation ───────────────────────────────────────────────────

    #[test]
    fn test_wal_truncation() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        for i in 1..=10 {
            mgr.append_wal(make_wal(i, 1, b"x", true));
        }
        mgr.load_pages(vec![make_page(1, b"x")]);
        let cp_id = mgr.create_full_checkpoint().expect("ok");

        let removed = mgr.truncate_wal(cp_id).expect("ok");
        assert_eq!(removed, 10);
        assert_eq!(mgr.pending_wal_count(), 0);
        assert_eq!(mgr.stats().wal_truncations, 1);
    }

    #[test]
    fn test_wal_truncation_partial() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        for i in 1..=5 {
            mgr.append_wal(make_wal(i, 1, b"x", true));
        }
        mgr.load_pages(vec![make_page(1, b"x")]);
        let cp_id = mgr.create_full_checkpoint().expect("ok");

        // Add more WAL entries after checkpoint
        for i in 6..=8 {
            mgr.append_wal(make_wal(i, 2, b"y", true));
        }

        let removed = mgr.truncate_wal(cp_id).expect("ok");
        assert_eq!(removed, 5);
        assert_eq!(mgr.pending_wal_count(), 3);
    }

    #[test]
    fn test_wal_truncation_unknown_checkpoint() {
        let mut mgr = CheckpointManager::default();
        let result = mgr.truncate_wal(999);
        assert!(result.is_err());
    }

    // ── Checkpoint-on-shutdown ───────────────────────────────────────────

    #[test]
    fn test_checkpoint_on_shutdown() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.append_wal(make_wal(1, 10, b"shutdown_data", true));
        mgr.append_wal(make_wal(2, 11, b"more_data", true));

        let id = mgr.checkpoint_on_shutdown().expect("ok");
        assert!(mgr.get_checkpoint(id).is_some());

        // Committed WAL entries should be in the snapshot
        assert!(mgr.get_page(10).is_some());
        assert!(mgr.get_page(11).is_some());
    }

    #[test]
    fn test_shutdown_ignores_uncommitted() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.append_wal(make_wal(1, 10, b"committed", true));
        mgr.append_wal(make_wal(2, 11, b"uncommitted", false));

        let _ = mgr.checkpoint_on_shutdown().expect("ok");
        assert!(mgr.get_page(10).is_some());
        assert!(mgr.get_page(11).is_none());
    }

    // ── Recovery ─────────────────────────────────────────────────────────

    #[test]
    fn test_recover_from_latest() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"original")]);
        mgr.append_wal(make_wal(1, 1, b"original", true));
        let _ = mgr.create_full_checkpoint();

        // Add post-checkpoint WAL
        mgr.append_wal(make_wal(2, 2, b"new_page", true));

        let result = mgr.recover().expect("ok");
        assert!(result.success);
        assert_eq!(result.wal_entries_replayed, 1);
        assert!(mgr.get_page(2).is_some());
    }

    #[test]
    fn test_recover_no_checkpoint() {
        let mut mgr = CheckpointManager::default();
        let result = mgr.recover();
        assert!(result.is_err());
    }

    #[test]
    fn test_recover_from_specific() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"v1")]);
        let id1 = mgr.create_full_checkpoint().expect("ok");

        mgr.append_wal(make_wal(1, 5, b"after_cp1", true));

        let result = mgr.recover_from(id1).expect("ok");
        assert!(result.success);
        assert_eq!(result.checkpoint_id, id1);
    }

    #[test]
    fn test_recover_from_unknown() {
        let mut mgr = CheckpointManager::default();
        let result = mgr.recover_from(42);
        assert!(result.is_err());
    }

    #[test]
    fn test_recover_skips_uncommitted_wal() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"base")]);
        let _ = mgr.create_full_checkpoint();

        mgr.append_wal(make_wal(10, 99, b"committed", true));
        mgr.append_wal(make_wal(11, 100, b"uncommitted", false));

        let result = mgr.recover().expect("ok");
        assert_eq!(result.wal_entries_replayed, 1);
        assert!(mgr.get_page(99).is_some());
        assert!(mgr.get_page(100).is_none());
    }

    // ── Verification ─────────────────────────────────────────────────────

    #[test]
    fn test_verify_valid_checkpoint() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"data")]);
        let id = mgr.create_full_checkpoint().expect("ok");

        let valid = mgr.verify_checkpoint(id).expect("ok");
        assert!(valid);
        let meta = mgr.get_checkpoint(id).expect("exists");
        assert_eq!(meta.status, CheckpointStatus::Verified);
    }

    #[test]
    fn test_verify_unknown_checkpoint() {
        let mut mgr = CheckpointManager::default();
        let result = mgr.verify_checkpoint(42);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_detects_corruption() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"original")]);
        let id = mgr.create_full_checkpoint().expect("ok");

        // Corrupt the data
        mgr.full_snapshot_pages
            .insert(1, make_page(1, b"corrupted"));

        let valid = mgr.verify_checkpoint(id).expect("ok");
        assert!(!valid);
        let meta = mgr.get_checkpoint(id).expect("exists");
        matches!(meta.status, CheckpointStatus::Corrupt(_));
    }

    // ── History management ───────────────────────────────────────────────

    #[test]
    fn test_history_trimming() {
        let cfg = CheckpointConfig {
            max_history: 3,
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"x")]);

        for _ in 0..5 {
            let _ = mgr.create_full_checkpoint();
        }
        assert_eq!(mgr.history().len(), 3);
    }

    #[test]
    fn test_latest_checkpoint() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"x")]);

        let id1 = mgr.create_full_checkpoint().expect("ok");
        mgr.set_time(200);
        let id2 = mgr.create_full_checkpoint().expect("ok");

        let latest = mgr.latest_checkpoint().expect("exists");
        assert_eq!(latest.id, id2);
        assert_ne!(id1, id2);
    }

    // ── Page and WAL accessors ───────────────────────────────────────────

    #[test]
    fn test_get_page() {
        let mut mgr = CheckpointManager::default();
        mgr.load_pages(vec![make_page(42, b"hello")]);
        let page = mgr.get_page(42).expect("exists");
        assert_eq!(page.data, b"hello");
    }

    #[test]
    fn test_get_page_missing() {
        let mgr = CheckpointManager::default();
        assert!(mgr.get_page(999).is_none());
    }

    #[test]
    fn test_snapshot_page_ids() {
        let mut mgr = CheckpointManager::default();
        mgr.load_pages(vec![
            make_page(3, b"c"),
            make_page(1, b"a"),
            make_page(2, b"b"),
        ]);
        assert_eq!(mgr.snapshot_page_ids(), vec![1, 2, 3]);
    }

    #[test]
    fn test_pending_wal_count() {
        let mut mgr = CheckpointManager::default();
        assert_eq!(mgr.pending_wal_count(), 0);
        mgr.append_wal(make_wal(1, 1, b"x", true));
        assert_eq!(mgr.pending_wal_count(), 1);
    }

    #[test]
    fn test_dirty_page_count() {
        let mut mgr = CheckpointManager::default();
        assert_eq!(mgr.dirty_page_count(), 0);
        mgr.mark_page_dirty(make_dirty_page(1, b"x"));
        assert_eq!(mgr.dirty_page_count(), 1);
    }

    // ── Failure recording ────────────────────────────────────────────────

    #[test]
    fn test_record_failure() {
        let mut mgr = CheckpointManager::default();
        mgr.record_failure("disk full");
        assert_eq!(mgr.stats().failed_checkpoints, 1);
        assert_eq!(mgr.stats().total_checkpoints, 1);

        let meta = mgr.history().last().expect("exists");
        assert_eq!(
            meta.status,
            CheckpointStatus::Failed("disk full".to_string())
        );
    }

    // ── Full workflow ────────────────────────────────────────────────────

    #[test]
    fn test_full_workflow_checkpoint_truncate_recover() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);

        // Phase 1: Load initial data and WAL
        mgr.load_pages(vec![make_page(1, b"page1"), make_page(2, b"page2")]);
        mgr.append_wal(make_wal(1, 1, b"page1", true));
        mgr.append_wal(make_wal(2, 2, b"page2", true));

        // Phase 2: Checkpoint
        let cp1 = mgr.create_full_checkpoint().expect("ok");
        assert_eq!(mgr.pending_wal_count(), 2);

        // Phase 3: Truncate WAL
        let removed = mgr.truncate_wal(cp1).expect("ok");
        assert_eq!(removed, 2);
        assert_eq!(mgr.pending_wal_count(), 0);

        // Phase 4: More writes after checkpoint
        mgr.append_wal(make_wal(3, 3, b"page3", true));

        // Phase 5: Recover
        let result = mgr.recover().expect("ok");
        assert!(result.success);
        assert_eq!(result.wal_entries_replayed, 1);
        assert!(mgr.get_page(3).is_some());
    }

    #[test]
    fn test_multiple_checkpoints_with_incremental() {
        let cfg = CheckpointConfig {
            incremental: true,
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);

        // Full checkpoint
        mgr.load_pages(vec![
            make_page(1, b"a"),
            make_page(2, b"b"),
            make_page(3, b"c"),
        ]);
        let _ = mgr.create_full_checkpoint();

        // Incremental 1
        mgr.mark_page_dirty(make_dirty_page(2, b"b2"));
        let _ = mgr.create_incremental_checkpoint();

        // Incremental 2
        mgr.mark_page_dirty(make_dirty_page(3, b"c2"));
        let _ = mgr.create_incremental_checkpoint();

        assert_eq!(mgr.stats().full_checkpoints, 1);
        assert_eq!(mgr.stats().incremental_checkpoints, 2);

        // Verify the merged snapshot has the updated pages
        assert_eq!(mgr.get_page(2).expect("exists").data, b"b2");
        assert_eq!(mgr.get_page(3).expect("exists").data, b"c2");
        assert_eq!(mgr.get_page(1).expect("exists").data, b"a");
    }

    #[test]
    fn test_checkpoint_metadata_fields() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.set_time(12345);
        mgr.load_pages(vec![make_page(1, b"data")]);
        mgr.append_wal(make_wal(42, 1, b"data", true));

        let id = mgr.create_full_checkpoint().expect("ok");
        let meta = mgr.get_checkpoint(id).expect("exists");

        assert_eq!(meta.id, 1);
        assert_eq!(meta.created_at_secs, 12345);
        assert_eq!(meta.wal_position, 42);
        assert_eq!(meta.page_count, 1);
        assert_eq!(meta.data_size_bytes, 4); // b"data"
        assert!(!meta.is_incremental);
        assert_eq!(meta.status, CheckpointStatus::Success);
    }

    #[test]
    fn test_data_size_accumulation() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, &[0u8; 100]), make_page(2, &[0u8; 200])]);
        let _ = mgr.create_full_checkpoint();

        assert_eq!(mgr.stats().total_bytes_written, 300);
        assert_eq!(mgr.stats().total_pages_flushed, 2);
    }

    #[test]
    fn test_recovery_stats() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"x")]);
        let _ = mgr.create_full_checkpoint();

        let _ = mgr.recover();
        let _ = mgr.recover();
        assert_eq!(mgr.stats().recoveries, 2);
    }

    #[test]
    fn test_checkpoint_trigger_default() {
        let trigger = CheckpointTrigger::default();
        match trigger {
            CheckpointTrigger::Combined {
                interval_secs,
                write_threshold,
            } => {
                assert_eq!(interval_secs, 300);
                assert_eq!(write_threshold, 10_000);
            }
            _ => panic!("unexpected default trigger"),
        }
    }

    #[test]
    fn test_dirty_pages_merged_into_full_snapshot() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        mgr.load_pages(vec![make_page(1, b"original")]);
        mgr.mark_page_dirty(make_dirty_page(2, b"new_page"));

        let _ = mgr.create_full_checkpoint();

        assert_eq!(mgr.get_page(1).expect("exists").data, b"original");
        assert_eq!(mgr.get_page(2).expect("exists").data, b"new_page");
        assert_eq!(mgr.dirty_page_count(), 0);
    }

    #[test]
    fn test_empty_checkpoint() {
        let cfg = CheckpointConfig {
            verify_after_create: false,
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(cfg);
        let id = mgr.create_full_checkpoint().expect("ok");
        let meta = mgr.get_checkpoint(id).expect("exists");
        assert_eq!(meta.page_count, 0);
        assert_eq!(meta.data_size_bytes, 0);
    }

    #[test]
    fn test_wal_entry_construction() {
        let entry = make_wal(5, 10, b"payload", true);
        assert_eq!(entry.lsn, 5);
        assert_eq!(entry.page_id, 10);
        assert_eq!(entry.data, b"payload");
        assert!(entry.committed);
    }

    #[test]
    fn test_page_data_constructors() {
        let clean = make_page(1, b"clean");
        assert!(!clean.dirty);

        let dirty = make_dirty_page(2, b"dirty");
        assert!(dirty.dirty);
    }
}
