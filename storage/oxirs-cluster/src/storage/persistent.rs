//! Persistent storage implementation
//!
//! Large implementation is split across sibling modules:
//! - `persistent_wal.rs`         — Write-Ahead Log and its recovery
//! - `persistent_files.rs`       — Crash-safe state file writes
//! - `persistent_snapshot.rs`    — Snapshot files
//! - `persistent_integrity.rs`   — Integrity verification and crash recovery
//! - `persistent_tests.rs`       — Unit tests (declared from mod.rs)

use super::config::StorageConfig;
use super::persistent_files::{
    read_checksummed, write_checksummed, APP_STATE_FILE, RAFT_STATE_FILE,
};
use super::persistent_snapshot;
use super::persistent_wal::{self, WalOperationRef, WalWriter, WAL_FILE};
use super::recovery::*;
use super::stats::StorageStats;
use super::types::*;

use crate::network::LogEntry;
use crate::raft::{OxirsNodeId, RdfApp, RdfCommand};
use crate::shard::ShardId;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use oxirs_core::model::Triple;
use parking_lot::{Mutex, RwLock};
use std::fs;
use std::path::{Path, PathBuf};

/// Storage backend trait for sharding support
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Create a new shard
    async fn create_shard(&self, shard_id: ShardId) -> Result<()>;

    /// Delete a shard
    async fn delete_shard(&self, shard_id: ShardId) -> Result<()>;

    /// Insert a triple into a specific shard
    async fn insert_triple_to_shard(&self, shard_id: ShardId, triple: Triple) -> Result<()>;

    /// Delete a triple from a specific shard
    async fn delete_triple_from_shard(&self, shard_id: ShardId, triple: &Triple) -> Result<()>;

    /// Query triples from a specific shard
    async fn query_shard(
        &self,
        shard_id: ShardId,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<Triple>>;

    /// Get shard size in bytes
    async fn get_shard_size(&self, shard_id: ShardId) -> Result<u64>;

    /// Get shard triple count
    async fn get_shard_triple_count(&self, shard_id: ShardId) -> Result<usize>;

    /// Export shard data for migration
    async fn export_shard(&self, shard_id: ShardId) -> Result<Vec<Triple>>;

    /// Import shard data during migration
    async fn import_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()>;

    /// Get all triples from a specific shard
    async fn get_shard_triples(&self, shard_id: ShardId) -> Result<Vec<Triple>>;

    /// Insert multiple triples into a specific shard
    async fn insert_triples_to_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()>;

    /// Mark a shard for deletion (logical deletion before physical cleanup)
    async fn mark_shard_for_deletion(&self, shard_id: ShardId) -> Result<()>;
}

/// StorageBackend implementation for PersistentStorage
///
/// Every change is persisted before it becomes visible: the new state is
/// written (journaled in the WAL when enabled) while the state's write lock
/// is held, and only then replaces the state in memory. A change whose save
/// fails is therefore not applied at all. All critical sections are
/// synchronous, so the storage also works when it is driven by a blocking
/// executor such as `futures::executor::block_on`.
pub struct PersistentStorage {
    /// Data directory
    pub(crate) data_dir: PathBuf,
    /// Node ID
    pub(crate) node_id: OxirsNodeId,
    /// In-memory Raft state (cached)
    pub(crate) raft_state: RwLock<RaftState>,
    /// In-memory application state (cached)
    pub(crate) app_state: RwLock<RdfApp>,
    /// Configuration
    pub(crate) config: StorageConfig,
    /// WAL writer (`None` when the WAL is disabled), held for a whole save
    wal: Mutex<Option<WalWriter>>,
    /// Serializes snapshot writes, reads and backups
    snapshot_lock: Mutex<()>,
    /// Whether startup recovery redid a journaled write
    wal_recovered: bool,
}

impl PersistentStorage {
    /// Create a new persistent storage instance
    ///
    /// With the WAL enabled, a torn tail of the WAL is set aside, and with
    /// crash recovery enabled, a journaled write that did not reach its
    /// state file is redone, all before the state is loaded.
    ///
    /// # Errors
    ///
    /// Fails when the data directory cannot be created, when the WAL is
    /// damaged anywhere but at its end, or when a state file is corrupted
    /// while corruption detection is enabled.
    pub async fn new(node_id: OxirsNodeId, config: StorageConfig) -> Result<Self> {
        let data_dir = PathBuf::from(&config.data_dir).join(format!("node-{node_id}"));
        fs::create_dir_all(&data_dir)?;

        let (wal, wal_recovered) = if config.enable_wal {
            let recovery =
                persistent_wal::recover_wal(&data_dir, &config, config.enable_crash_recovery)?;
            let writer = WalWriter::open(
                data_dir.join(WAL_FILE),
                recovery.next_sequence,
                config.sync_writes,
            )?;
            tracing::info!(
                "Initialized WAL for node {} at sequence {}",
                node_id,
                recovery.next_sequence
            );
            (Some(writer), recovery.redone)
        } else {
            (None, false)
        };

        let storage = Self {
            data_dir,
            node_id,
            raft_state: RwLock::new(RaftState::default()),
            app_state: RwLock::new(RdfApp::default()),
            config,
            wal: Mutex::new(wal),
            snapshot_lock: Mutex::new(()),
            wal_recovered,
        };

        if storage.config.enable_crash_recovery {
            storage.recover_from_crash().await?;
        }

        storage.load_state()?;

        Ok(storage)
    }

    /// Get current term
    pub async fn get_current_term(&self) -> u64 {
        self.raft_state.read().current_term
    }

    /// Set current term
    pub async fn set_current_term(&self, term: u64) -> Result<()> {
        self.update_raft_state(|state| {
            state.current_term = term;
            state.voted_for = None;
        })
    }

    /// Get voted for
    pub async fn get_voted_for(&self) -> Option<OxirsNodeId> {
        self.raft_state.read().voted_for
    }

    /// Set voted for
    pub async fn set_voted_for(&self, candidate_id: Option<OxirsNodeId>) -> Result<()> {
        self.update_raft_state(|state| state.voted_for = candidate_id)
    }

    /// Append log entries
    pub async fn append_entries(&self, entries: Vec<LogEntry>) -> Result<()> {
        self.update_raft_state(|state| state.log.extend(entries))
    }

    /// Get log entries in range
    pub async fn get_log_entries(&self, start: u64, end: u64) -> Vec<LogEntry> {
        let state = self.raft_state.read();
        state
            .log
            .iter()
            .filter(|entry| entry.index >= start && entry.index < end)
            .cloned()
            .collect()
    }

    /// Get log entry at index
    pub async fn get_log_entry(&self, index: u64) -> Option<LogEntry> {
        let state = self.raft_state.read();
        state.log.iter().find(|entry| entry.index == index).cloned()
    }

    /// Get last log index
    pub async fn get_last_log_index(&self) -> u64 {
        let state = self.raft_state.read();
        state.log.last().map(|entry| entry.index).unwrap_or(0)
    }

    /// Get last log term
    pub async fn get_last_log_term(&self) -> u64 {
        let state = self.raft_state.read();
        state.log.last().map(|entry| entry.term).unwrap_or(0)
    }

    /// Truncate log from index
    pub async fn truncate_log(&self, from_index: u64) -> Result<()> {
        self.update_raft_state(|state| state.log.retain(|entry| entry.index < from_index))
    }

    /// Get commit index
    pub async fn get_commit_index(&self) -> u64 {
        self.raft_state.read().commit_index
    }

    /// Set commit index
    pub async fn set_commit_index(&self, index: u64) -> Result<()> {
        self.update_raft_state(|state| state.commit_index = index)
    }

    /// Get last applied index
    pub async fn get_last_applied(&self) -> u64 {
        self.raft_state.read().last_applied
    }

    /// Set last applied index
    pub async fn set_last_applied(&self, index: u64) -> Result<()> {
        self.update_raft_state(|state| state.last_applied = index)
    }

    /// Apply command to state machine
    pub async fn apply_command(&self, command: &RdfCommand) -> Result<()> {
        self.update_app_state(|app| {
            app.apply_command(command);
        })
    }

    /// Get application state
    pub async fn get_app_state(&self) -> RdfApp {
        self.app_state.read().clone()
    }

    /// Create snapshot
    pub async fn create_snapshot(&self) -> Result<SnapshotMetadata> {
        let _snapshot = self.snapshot_lock.lock();
        let raft_state = self.raft_state.read();
        let app_state = self.app_state.read();
        persistent_snapshot::write_snapshot(
            &self.data_dir,
            self.node_id,
            &raft_state,
            &app_state,
            &self.config,
        )
    }

    /// Load snapshot
    ///
    /// # Errors
    ///
    /// Fails when the snapshot data is missing or, with corruption detection
    /// enabled, does not match the checksum in its metadata.
    pub async fn load_snapshot(&self) -> Result<Option<(SnapshotMetadata, RdfApp)>> {
        let _snapshot = self.snapshot_lock.lock();
        persistent_snapshot::read_snapshot(&self.data_dir, &self.config)
    }

    /// Compact log (remove entries before last snapshot)
    pub async fn compact_log(&self, until_index: u64) -> Result<()> {
        self.update_raft_state(|state| state.log.retain(|entry| entry.index > until_index))
    }

    /// Check if compaction is needed
    pub async fn needs_compaction(&self) -> bool {
        self.raft_state.read().log.len() > self.config.max_log_entries
    }

    /// Change the Raft state: the changed copy is saved first and replaces
    /// the state in memory only once it is on disk.
    fn update_raft_state(&self, change: impl FnOnce(&mut RaftState)) -> Result<()> {
        let mut state = self.raft_state.write();
        let mut next = state.clone();
        change(&mut next);
        let write = || write_checksummed(&self.data_dir, RAFT_STATE_FILE, &next, &self.config);
        match self.wal.lock().as_mut() {
            Some(wal) => wal.journal(WalOperationRef::WriteRaftState(&next), write)?,
            None => write()?,
        }
        *state = next;
        Ok(())
    }

    /// Change the application state like [`Self::update_raft_state`].
    fn update_app_state<R>(&self, change: impl FnOnce(&mut RdfApp) -> R) -> Result<R> {
        let mut app_state = self.app_state.write();
        let mut next = app_state.clone();
        let outcome = change(&mut next);
        let write = || write_checksummed(&self.data_dir, APP_STATE_FILE, &next, &self.config);
        match self.wal.lock().as_mut() {
            Some(wal) => wal.journal(WalOperationRef::WriteAppState(&next), write)?,
            None => write()?,
        }
        *app_state = next;
        Ok(outcome)
    }

    /// Load Raft state from disk with corruption detection
    fn load_state(&self) -> Result<()> {
        let binary_path = self.data_dir.join(RAFT_STATE_FILE);
        if binary_path.exists() {
            match read_checksummed::<RaftState>(&binary_path, &self.config) {
                Ok(state) => {
                    *self.raft_state.write() = state;
                    tracing::info!("Loaded Raft state (binary) for node {}", self.node_id);
                }
                Err(e) => {
                    tracing::error!("Failed to load binary Raft state: {}", e);
                    if self.config.enable_corruption_detection {
                        return Err(anyhow!("Corrupted Raft state file"));
                    }
                }
            }
        } else {
            let json_path = self.data_dir.join("raft_state.json");
            if json_path.exists() {
                let data = fs::read(&json_path)?;
                let state: RaftState = serde_json::from_slice(&data)?;
                *self.raft_state.write() = state;
                tracing::info!("Loaded Raft state (legacy JSON) for node {}", self.node_id);
            }
        }

        let app_binary_path = self.data_dir.join(APP_STATE_FILE);
        if app_binary_path.exists() {
            match read_checksummed::<RdfApp>(&app_binary_path, &self.config) {
                Ok(app_state) => {
                    *self.app_state.write() = app_state;
                    tracing::info!(
                        "Loaded application state (binary) for node {}",
                        self.node_id
                    );
                }
                Err(e) => {
                    tracing::error!("Failed to load binary application state: {}", e);
                    if self.config.enable_corruption_detection {
                        return Err(anyhow!("Corrupted application state file"));
                    }
                }
            }
        } else {
            let app_json_path = self.data_dir.join("app_state.json");
            if app_json_path.exists() {
                let data = fs::read(&app_json_path)?;
                let app_state: RdfApp = serde_json::from_slice(&data)?;
                *self.app_state.write() = app_state;
                tracing::info!(
                    "Loaded application state (legacy JSON) for node {}",
                    self.node_id
                );
            }
        }

        Ok(())
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> StorageStats {
        let (log_entries, current_term, commit_index, last_applied) = {
            let raft_state = self.raft_state.read();
            (
                raft_state.log.len(),
                raft_state.current_term,
                raft_state.commit_index,
                raft_state.last_applied,
            )
        };
        let triple_count = self.app_state.read().len();
        let data_dir_size = Self::calculate_directory_size(&self.data_dir).unwrap_or(0);

        StorageStats {
            node_id: self.node_id,
            data_dir: self.data_dir.clone(),
            log_entries,
            current_term,
            commit_index,
            last_applied,
            triple_count,
            disk_usage_bytes: data_dir_size,
        }
    }

    /// Calculate directory size
    fn calculate_directory_size(dir: &Path) -> Result<u64> {
        let mut size = 0;
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    size += fs::metadata(&path)?.len();
                } else if path.is_dir() {
                    size += Self::calculate_directory_size(&path)?;
                }
            }
        }
        Ok(size)
    }

    /// Directory that holds the data directories and backups of all nodes.
    fn parent_dir(&self) -> Result<&Path> {
        self.data_dir
            .parent()
            .ok_or_else(|| anyhow!("{} has no parent directory", self.data_dir.display()))
    }

    /// Backup current state
    ///
    /// No save, snapshot or WAL maintenance runs while the files are copied,
    /// so the backup is a consistent picture of the data directory.
    pub async fn backup(&self) -> Result<PathBuf> {
        let timestamp = super::persistent_files::unix_secs();
        let backup_dir = self
            .parent_dir()?
            .join(format!("backup-{}-{}", self.node_id, timestamp));

        let _snapshot = self.snapshot_lock.lock();
        let _wal = self.wal.lock();
        fs::create_dir_all(&backup_dir)?;
        for entry in fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let source = entry.path();
            let dest = backup_dir.join(entry.file_name());
            fs::copy(&source, &dest)?;
        }

        tracing::info!("Created backup at {:?}", backup_dir);
        Ok(backup_dir)
    }

    /// Rotate WAL file when it gets too large
    pub async fn rotate_wal(&self) -> Result<()> {
        if let Some(wal) = self.wal.lock().as_mut() {
            if let Some(archive) = wal.rotate()? {
                tracing::info!(
                    "Rotated WAL for node {}, archived to {:?}",
                    self.node_id,
                    archive
                );
            }
        }
        Ok(())
    }

    /// Compact WAL by removing committed entries
    pub async fn compact_wal(&self) -> Result<()> {
        if let Some(wal) = self.wal.lock().as_mut() {
            let (removed, kept) = wal.compact()?;
            tracing::info!(
                "Compacted WAL for node {}, removed {} committed entries, kept {} uncommitted",
                self.node_id,
                removed,
                kept
            );
        }
        Ok(())
    }

    /// Verify data integrity across all files
    pub async fn verify_integrity(&self) -> Result<bool> {
        super::persistent_integrity::verify_integrity(&self.config, &self.data_dir, &self.wal)
    }

    /// Clean old backups
    pub async fn cleanup_old_backups(&self) -> Result<()> {
        let parent_dir = self.parent_dir()?;
        let backup_prefix = format!("backup-{}-", self.node_id);

        let mut backups = Vec::new();
        for entry in fs::read_dir(parent_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if let Some(name_str) = name.to_str() {
                if name_str.starts_with(&backup_prefix) {
                    backups.push((entry.path(), name_str.to_string()));
                }
            }
        }

        backups.sort_by(|a, b| b.1.cmp(&a.1));

        for (path, _) in backups.iter().skip(self.config.backup_retention) {
            fs::remove_dir_all(path)?;
            tracing::info!("Removed old backup: {:?}", path);
        }

        Ok(())
    }

    /// Perform crash recovery check and repair if needed
    pub async fn recover_from_crash(&self) -> Result<RecoveryReport> {
        super::persistent_integrity::recover_from_crash(
            &self.config,
            &self.data_dir,
            self.node_id,
            &self.raft_state,
            self.wal_recovered,
        )
    }
}

#[async_trait]
impl StorageBackend for PersistentStorage {
    async fn create_shard(&self, shard_id: ShardId) -> Result<()> {
        self.update_app_state(|app| app.create_shard(shard_id))
    }

    async fn delete_shard(&self, shard_id: ShardId) -> Result<()> {
        self.update_app_state(|app| app.delete_shard(shard_id))
    }

    async fn insert_triple_to_shard(&self, shard_id: ShardId, triple: Triple) -> Result<()> {
        self.update_app_state(|app| app.insert_triple_to_shard(shard_id, triple))
    }

    async fn delete_triple_from_shard(&self, shard_id: ShardId, triple: &Triple) -> Result<()> {
        self.update_app_state(|app| app.delete_triple_from_shard(shard_id, triple))
    }

    async fn query_shard(
        &self,
        shard_id: ShardId,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<Triple>> {
        Ok(self
            .app_state
            .read()
            .query_shard(shard_id, subject, predicate, object))
    }

    async fn get_shard_size(&self, shard_id: ShardId) -> Result<u64> {
        Ok(self.app_state.read().get_shard_size(shard_id))
    }

    async fn get_shard_triple_count(&self, shard_id: ShardId) -> Result<usize> {
        Ok(self.app_state.read().get_shard_triple_count(shard_id))
    }

    async fn export_shard(&self, shard_id: ShardId) -> Result<Vec<Triple>> {
        Ok(self.app_state.read().export_shard(shard_id))
    }

    async fn import_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()> {
        self.update_app_state(|app| app.import_shard(shard_id, triples))
    }

    async fn get_shard_triples(&self, shard_id: ShardId) -> Result<Vec<Triple>> {
        Ok(self.app_state.read().get_shard_triples(shard_id))
    }

    async fn insert_triples_to_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()> {
        self.update_app_state(|app| app.insert_triples_to_shard(shard_id, triples))
    }

    async fn mark_shard_for_deletion(&self, shard_id: ShardId) -> Result<()> {
        self.update_app_state(|app| app.mark_shard_for_deletion(shard_id))
    }
}
