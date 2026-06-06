//! Persistent storage implementation
//!
//! Large implementation is split across sibling modules:
//! - `persistent_wal.rs`         — Write-Ahead Log operations
//! - `persistent_integrity.rs`   — Integrity verification and crash recovery
//! - `persistent_tests.rs`       — Unit tests (declared from mod.rs)

use super::config::StorageConfig;
use super::recovery::*;
use super::stats::StorageStats;
use super::types::*;

use crate::network::LogEntry;
use crate::raft::{OxirsNodeId, RdfApp, RdfCommand};
use crate::shard::ShardId;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use oxirs_core::model::Triple;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

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
pub struct PersistentStorage {
    /// Data directory
    pub(crate) data_dir: PathBuf,
    /// Node ID
    pub(crate) node_id: OxirsNodeId,
    /// In-memory Raft state (cached)
    pub(crate) raft_state: Arc<RwLock<RaftState>>,
    /// In-memory application state (cached)
    pub(crate) app_state: Arc<RwLock<RdfApp>>,
    /// Configuration
    pub(crate) config: StorageConfig,
    /// WAL sequence counter
    pub(crate) wal_sequence: Arc<RwLock<u64>>,
    /// WAL file writer
    pub(crate) wal_writer: Arc<RwLock<Option<BufWriter<File>>>>,
}

impl PersistentStorage {
    /// Create a new persistent storage instance
    pub async fn new(node_id: OxirsNodeId, config: StorageConfig) -> Result<Self> {
        let data_dir = PathBuf::from(&config.data_dir).join(format!("node-{node_id}"));

        if !data_dir.exists() {
            std::fs::create_dir_all(&data_dir)?;
        }

        let storage = Self {
            data_dir,
            node_id,
            raft_state: Arc::new(RwLock::new(RaftState::default())),
            app_state: Arc::new(RwLock::new(RdfApp::default())),
            config,
            wal_sequence: Arc::new(RwLock::new(0)),
            wal_writer: Arc::new(RwLock::new(None)),
        };

        if storage.config.enable_wal {
            storage.init_wal().await?;
        }

        if storage.config.enable_crash_recovery {
            storage.recover_from_crash().await?;
        }

        storage.load_state().await?;

        Ok(storage)
    }

    /// Initialize Write-Ahead Log
    async fn init_wal(&self) -> Result<()> {
        let ctx = super::persistent_wal::WalContext {
            data_dir: &self.data_dir,
            node_id: self.node_id,
            config: &self.config,
            wal_sequence: &self.wal_sequence,
            wal_writer: &self.wal_writer,
        };
        super::persistent_wal::init_wal(ctx).await
    }

    /// Write entry to WAL
    async fn write_wal_entry(&self, operation: WalOperation) -> Result<()> {
        super::persistent_wal::write_wal_entry(
            &self.config,
            &self.wal_sequence,
            &self.wal_writer,
            operation,
        )
        .await
    }

    /// Get current term
    pub async fn get_current_term(&self) -> u64 {
        self.raft_state.read().await.current_term
    }

    /// Set current term
    pub async fn set_current_term(&self, term: u64) -> Result<()> {
        {
            let mut state = self.raft_state.write().await;
            state.current_term = term;
            state.voted_for = None;
        }
        self.save_state().await
    }

    /// Get voted for
    pub async fn get_voted_for(&self) -> Option<OxirsNodeId> {
        self.raft_state.read().await.voted_for
    }

    /// Set voted for
    pub async fn set_voted_for(&self, candidate_id: Option<OxirsNodeId>) -> Result<()> {
        {
            let mut state = self.raft_state.write().await;
            state.voted_for = candidate_id;
        }
        self.save_state().await
    }

    /// Append log entries
    pub async fn append_entries(&self, entries: Vec<LogEntry>) -> Result<()> {
        {
            let mut state = self.raft_state.write().await;
            state.log.extend(entries);
        }
        self.save_state().await
    }

    /// Get log entries in range
    pub async fn get_log_entries(&self, start: u64, end: u64) -> Vec<LogEntry> {
        let state = self.raft_state.read().await;
        state
            .log
            .iter()
            .filter(|entry| entry.index >= start && entry.index < end)
            .cloned()
            .collect()
    }

    /// Get log entry at index
    pub async fn get_log_entry(&self, index: u64) -> Option<LogEntry> {
        let state = self.raft_state.read().await;
        state.log.iter().find(|entry| entry.index == index).cloned()
    }

    /// Get last log index
    pub async fn get_last_log_index(&self) -> u64 {
        let state = self.raft_state.read().await;
        state.log.last().map(|entry| entry.index).unwrap_or(0)
    }

    /// Get last log term
    pub async fn get_last_log_term(&self) -> u64 {
        let state = self.raft_state.read().await;
        state.log.last().map(|entry| entry.term).unwrap_or(0)
    }

    /// Truncate log from index
    pub async fn truncate_log(&self, from_index: u64) -> Result<()> {
        {
            let mut state = self.raft_state.write().await;
            state.log.retain(|entry| entry.index < from_index);
        }
        self.save_state().await
    }

    /// Get commit index
    pub async fn get_commit_index(&self) -> u64 {
        self.raft_state.read().await.commit_index
    }

    /// Set commit index
    pub async fn set_commit_index(&self, index: u64) -> Result<()> {
        {
            let mut state = self.raft_state.write().await;
            state.commit_index = index;
        }
        self.save_state().await
    }

    /// Get last applied index
    pub async fn get_last_applied(&self) -> u64 {
        self.raft_state.read().await.last_applied
    }

    /// Set last applied index
    pub async fn set_last_applied(&self, index: u64) -> Result<()> {
        {
            let mut state = self.raft_state.write().await;
            state.last_applied = index;
        }
        self.save_state().await
    }

    /// Apply command to state machine
    pub async fn apply_command(&self, command: &RdfCommand) -> Result<()> {
        let mut app_state = self.app_state.write().await;
        app_state.apply_command(command);
        self.save_app_state().await
    }

    /// Get application state
    pub async fn get_app_state(&self) -> RdfApp {
        self.app_state.read().await.clone()
    }

    /// Create snapshot
    pub async fn create_snapshot(&self) -> Result<SnapshotMetadata> {
        let raft_state = self.raft_state.read().await;
        let app_state = self.app_state.read().await;

        let last_log_entry = raft_state.log.last();
        let snapshot_path = self.data_dir.join("snapshot.json");
        let snapshot_data = serde_json::to_vec(&*app_state)?;
        fs::write(&snapshot_path, &snapshot_data)?;

        let mut hasher = Sha256::new();
        hasher.update(&snapshot_data);
        let checksum = hex::encode(hasher.finalize());

        let metadata = SnapshotMetadata {
            last_included_index: last_log_entry.map(|e| e.index).unwrap_or(0),
            last_included_term: last_log_entry.map(|e| e.term).unwrap_or(0),
            configuration: vec![self.node_id],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            size: snapshot_data.len() as u64,
            checksum,
        };

        let metadata_path = self.data_dir.join("snapshot_metadata.json");
        let metadata_data = serde_json::to_vec(&metadata)?;
        fs::write(&metadata_path, &metadata_data)?;

        tracing::info!(
            "Created snapshot for node {} at index {} with {} bytes",
            self.node_id,
            metadata.last_included_index,
            metadata.size
        );

        Ok(metadata)
    }

    /// Load snapshot
    pub async fn load_snapshot(&self) -> Result<Option<(SnapshotMetadata, RdfApp)>> {
        let snapshot_path = self.data_dir.join("snapshot.json");
        let metadata_path = self.data_dir.join("snapshot_metadata.json");

        if !snapshot_path.exists() || !metadata_path.exists() {
            return Ok(None);
        }

        let metadata_data = fs::read(&metadata_path)?;
        let metadata: SnapshotMetadata = serde_json::from_slice(&metadata_data)?;

        let snapshot_data = fs::read(&snapshot_path)?;
        let app_state: RdfApp = serde_json::from_slice(&snapshot_data)?;

        Ok(Some((metadata, app_state)))
    }

    /// Compact log (remove entries before last snapshot)
    pub async fn compact_log(&self, until_index: u64) -> Result<()> {
        {
            let mut state = self.raft_state.write().await;
            state.log.retain(|entry| entry.index > until_index);
        }
        self.save_state().await
    }

    /// Check if compaction is needed
    pub async fn needs_compaction(&self) -> bool {
        let state = self.raft_state.read().await;
        state.log.len() > self.config.max_log_entries
    }

    /// Save Raft state to disk with WAL and atomic writes
    async fn save_state(&self) -> Result<()> {
        let state = self.raft_state.read().await;

        if self.config.enable_wal {
            self.write_wal_entry(WalOperation::WriteRaftState(state.clone()))
                .await?;
        }

        self.atomic_write_with_checksum("raft_state.dat", &*state)
            .await?;

        if self.config.enable_wal {
            let sequence = *self.wal_sequence.read().await;
            self.write_wal_entry(WalOperation::Commit(sequence)).await?;
        }

        Ok(())
    }

    /// Atomic write with corruption detection
    async fn atomic_write_with_checksum<T>(&self, filename: &str, data: &T) -> Result<()>
    where
        T: Serialize,
    {
        let path = self.data_dir.join(filename);
        let temp_path = self.data_dir.join(format!("{filename}.tmp"));

        let checksummed_data = if self.config.enable_corruption_detection {
            ChecksummedData::new(data)?
        } else {
            ChecksummedData {
                data,
                checksum: String::new(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("SystemTime should be after UNIX_EPOCH")
                    .as_secs(),
            }
        };

        let serialized =
            oxicode::serde::encode_to_vec(&checksummed_data, oxicode::config::standard())?;

        {
            let temp_file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)?;

            let mut writer = BufWriter::new(temp_file);
            writer.write_all(&serialized)?;
            writer.flush()?;

            if self.config.sync_writes {
                writer.get_mut().sync_all()?;
            }
        }

        std::fs::rename(&temp_path, &path)?;

        Ok(())
    }

    /// Load Raft state from disk with corruption detection
    async fn load_state(&self) -> Result<()> {
        let binary_path = self.data_dir.join("raft_state.dat");
        if binary_path.exists() {
            match self.load_with_checksum::<RaftState>(&binary_path).await {
                Ok(state) => {
                    *self.raft_state.write().await = state;
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
                let data = std::fs::read(&json_path)?;
                let state: RaftState = serde_json::from_slice(&data)?;
                *self.raft_state.write().await = state;
                tracing::info!("Loaded Raft state (legacy JSON) for node {}", self.node_id);
            }
        }

        let app_binary_path = self.data_dir.join("app_state.dat");
        if app_binary_path.exists() {
            match self.load_with_checksum::<RdfApp>(&app_binary_path).await {
                Ok(app_state) => {
                    *self.app_state.write().await = app_state;
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
                let data = std::fs::read(&app_json_path)?;
                let app_state: RdfApp = serde_json::from_slice(&data)?;
                *self.app_state.write().await = app_state;
                tracing::info!(
                    "Loaded application state (legacy JSON) for node {}",
                    self.node_id
                );
            }
        }

        Ok(())
    }

    /// Load data with checksum verification
    async fn load_with_checksum<T>(&self, path: &Path) -> Result<T>
    where
        T: for<'de> Deserialize<'de> + Serialize,
    {
        let data = std::fs::read(path)?;
        let (checksummed_data, _): (ChecksummedData<T>, _) =
            oxicode::serde::decode_from_slice(&data, oxicode::config::standard())?;

        if self.config.enable_corruption_detection && !checksummed_data.verify()? {
            return Err(anyhow!("Checksum verification failed for {:?}", path));
        }

        Ok(checksummed_data.data)
    }

    /// Save application state to disk with WAL and atomic writes
    async fn save_app_state(&self) -> Result<()> {
        let app_state = self.app_state.read().await;

        if self.config.enable_wal {
            self.write_wal_entry(WalOperation::WriteAppState(app_state.clone()))
                .await?;
        }

        self.atomic_write_with_checksum("app_state.dat", &*app_state)
            .await?;

        if self.config.enable_wal {
            let sequence = *self.wal_sequence.read().await;
            self.write_wal_entry(WalOperation::Commit(sequence)).await?;
        }

        Ok(())
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> StorageStats {
        let raft_state = self.raft_state.read().await;
        let app_state = self.app_state.read().await;

        let data_dir_size = self.calculate_directory_size(&self.data_dir).unwrap_or(0);

        StorageStats {
            node_id: self.node_id,
            data_dir: self.data_dir.clone(),
            log_entries: raft_state.log.len(),
            current_term: raft_state.current_term,
            commit_index: raft_state.commit_index,
            last_applied: raft_state.last_applied,
            triple_count: app_state.len(),
            disk_usage_bytes: data_dir_size,
        }
    }

    /// Calculate directory size
    #[allow(clippy::only_used_in_recursion)]
    fn calculate_directory_size(&self, dir: &Path) -> Result<u64> {
        let mut size = 0;
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    size += fs::metadata(&path)?.len();
                } else if path.is_dir() {
                    size += self.calculate_directory_size(&path)?;
                }
            }
        }
        Ok(size)
    }

    /// Backup current state
    pub async fn backup(&self) -> Result<PathBuf> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();
        let backup_dir = self
            .data_dir
            .parent()
            .expect("data_dir should have a parent directory")
            .join(format!("backup-{}-{}", self.node_id, timestamp));

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
        super::persistent_wal::rotate_wal(
            &self.config,
            &self.data_dir,
            self.node_id,
            &self.wal_sequence,
            &self.wal_writer,
        )
        .await
    }

    /// Compact WAL by removing committed entries
    pub async fn compact_wal(&self) -> Result<()> {
        super::persistent_wal::compact_wal(
            &self.config,
            &self.data_dir,
            self.node_id,
            &self.wal_sequence,
            &self.wal_writer,
        )
        .await
    }

    /// Verify data integrity across all files
    pub async fn verify_integrity(&self) -> Result<bool> {
        super::persistent_integrity::verify_integrity(
            &self.config,
            &self.data_dir,
            &self.raft_state,
            &self.app_state,
        )
        .await
    }

    /// Clean old backups
    pub async fn cleanup_old_backups(&self) -> Result<()> {
        let parent_dir = self
            .data_dir
            .parent()
            .expect("data_dir should have a parent directory");
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
        )
        .await
    }
}

#[async_trait]
impl StorageBackend for PersistentStorage {
    async fn create_shard(&self, shard_id: ShardId) -> Result<()> {
        let mut app_state = self.app_state.write().await;
        app_state.create_shard(shard_id);
        self.save_app_state().await
    }

    async fn delete_shard(&self, shard_id: ShardId) -> Result<()> {
        let mut app_state = self.app_state.write().await;
        app_state.delete_shard(shard_id);
        self.save_app_state().await
    }

    async fn insert_triple_to_shard(&self, shard_id: ShardId, triple: Triple) -> Result<()> {
        let mut app_state = self.app_state.write().await;
        app_state.insert_triple_to_shard(shard_id, triple);
        self.save_app_state().await
    }

    async fn delete_triple_from_shard(&self, shard_id: ShardId, triple: &Triple) -> Result<()> {
        let mut app_state = self.app_state.write().await;
        app_state.delete_triple_from_shard(shard_id, triple);
        self.save_app_state().await
    }

    async fn query_shard(
        &self,
        shard_id: ShardId,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<Triple>> {
        let app_state = self.app_state.read().await;
        Ok(app_state.query_shard(shard_id, subject, predicate, object))
    }

    async fn get_shard_size(&self, shard_id: ShardId) -> Result<u64> {
        let app_state = self.app_state.read().await;
        Ok(app_state.get_shard_size(shard_id))
    }

    async fn get_shard_triple_count(&self, shard_id: ShardId) -> Result<usize> {
        let app_state = self.app_state.read().await;
        Ok(app_state.get_shard_triple_count(shard_id))
    }

    async fn export_shard(&self, shard_id: ShardId) -> Result<Vec<Triple>> {
        let app_state = self.app_state.read().await;
        Ok(app_state.export_shard(shard_id))
    }

    async fn import_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()> {
        let mut app_state = self.app_state.write().await;
        app_state.import_shard(shard_id, triples);
        self.save_app_state().await
    }

    async fn get_shard_triples(&self, shard_id: ShardId) -> Result<Vec<Triple>> {
        let app_state = self.app_state.read().await;
        Ok(app_state.get_shard_triples(shard_id))
    }

    async fn insert_triples_to_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()> {
        let mut app_state = self.app_state.write().await;
        app_state.insert_triples_to_shard(shard_id, triples);
        self.save_app_state().await
    }

    async fn mark_shard_for_deletion(&self, shard_id: ShardId) -> Result<()> {
        let mut app_state = self.app_state.write().await;
        app_state.mark_shard_for_deletion(shard_id);
        self.save_app_state().await
    }
}
