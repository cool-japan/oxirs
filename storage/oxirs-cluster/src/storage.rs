//! # Persistent Storage
//!
//! Persistent storage backend for Raft consensus state and RDF data.
//! Provides durability guarantees required by the Raft protocol.

use crate::network::LogEntry;
use crate::raft::{OxirsNodeId, RdfApp, RdfCommand};
use crate::shard::ShardId;
use anyhow::Result;
use async_trait::async_trait;
use oxirs_core::model::Triple;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Persistent state required by Raft
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftState {
    /// Current term
    pub current_term: u64,
    /// Candidate ID that received vote in current term
    pub voted_for: Option<OxirsNodeId>,
    /// Log entries
    pub log: Vec<LogEntry>,
    /// Index of highest log entry known to be committed
    pub commit_index: u64,
    /// Index of highest log entry applied to state machine
    pub last_applied: u64,
}

impl Default for RaftState {
    fn default() -> Self {
        Self {
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
        }
    }
}

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Last included index
    pub last_included_index: u64,
    /// Last included term
    pub last_included_term: u64,
    /// Cluster configuration at the time of snapshot
    pub configuration: Vec<OxirsNodeId>,
    /// Timestamp when snapshot was created
    pub timestamp: u64,
    /// Size of the snapshot data in bytes
    pub size: u64,
}

/// Persistent storage backend
#[derive(Debug)]
pub struct PersistentStorage {
    /// Data directory
    data_dir: PathBuf,
    /// Node ID
    node_id: OxirsNodeId,
    /// In-memory Raft state (cached)
    raft_state: Arc<RwLock<RaftState>>,
    /// In-memory application state (cached)
    app_state: Arc<RwLock<RdfApp>>,
    /// Configuration
    config: StorageConfig,
}

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Data directory
    pub data_dir: String,
    /// Sync writes to disk immediately
    pub sync_writes: bool,
    /// Maximum log entries before forcing a snapshot
    pub max_log_entries: usize,
    /// Snapshot compression
    pub compress_snapshots: bool,
    /// Backup retention count
    pub backup_retention: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: "./data".to_string(),
            sync_writes: true,
            max_log_entries: 10000,
            compress_snapshots: true,
            backup_retention: 3,
        }
    }
}

impl PersistentStorage {
    /// Create a new persistent storage instance
    pub async fn new(node_id: OxirsNodeId, config: StorageConfig) -> Result<Self> {
        let data_dir = PathBuf::from(&config.data_dir).join(format!("node-{}", node_id));

        // Create data directory if it doesn't exist
        if !data_dir.exists() {
            fs::create_dir_all(&data_dir)?;
        }

        let mut storage = Self {
            data_dir,
            node_id,
            raft_state: Arc::new(RwLock::new(RaftState::default())),
            app_state: Arc::new(RwLock::new(RdfApp::default())),
            config,
        };

        // Load existing state if available
        storage.load_state().await?;

        Ok(storage)
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
            state.voted_for = None; // Reset vote when term changes
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
        let metadata = SnapshotMetadata {
            last_included_index: last_log_entry.map(|e| e.index).unwrap_or(0),
            last_included_term: last_log_entry.map(|e| e.term).unwrap_or(0),
            configuration: vec![self.node_id], // Simplified - would include all cluster members
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            size: 0, // Will be set after serialization
        };

        // Save snapshot
        let snapshot_path = self.data_dir.join("snapshot.json");
        let snapshot_data = serde_json::to_vec(&*app_state)?;
        fs::write(&snapshot_path, &snapshot_data)?;

        // Save metadata
        let mut metadata = metadata;
        metadata.size = snapshot_data.len() as u64;
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

    /// Save Raft state to disk
    async fn save_state(&self) -> Result<()> {
        let state = self.raft_state.read().await;
        let data = serde_json::to_vec(&*state)?;
        let path = self.data_dir.join("raft_state.json");
        fs::write(&path, &data)?;

        if self.config.sync_writes {
            // Force sync to disk
            let file = fs::File::open(&path)?;
            file.sync_all()?;
        }

        Ok(())
    }

    /// Load Raft state from disk
    async fn load_state(&self) -> Result<()> {
        let path = self.data_dir.join("raft_state.json");
        if path.exists() {
            let data = fs::read(&path)?;
            let state: RaftState = serde_json::from_slice(&data)?;
            *self.raft_state.write().await = state;
            tracing::info!("Loaded Raft state for node {}", self.node_id);
        }

        let app_path = self.data_dir.join("app_state.json");
        if app_path.exists() {
            let data = fs::read(&app_path)?;
            let app_state: RdfApp = serde_json::from_slice(&data)?;
            *self.app_state.write().await = app_state;
            tracing::info!("Loaded application state for node {}", self.node_id);
        }

        Ok(())
    }

    /// Save application state to disk
    async fn save_app_state(&self) -> Result<()> {
        let app_state = self.app_state.read().await;
        let data = serde_json::to_vec(&*app_state)?;
        let path = self.data_dir.join("app_state.json");
        fs::write(&path, &data)?;

        if self.config.sync_writes {
            let file = fs::File::open(&path)?;
            file.sync_all()?;
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
            .unwrap()
            .as_secs();
        let backup_dir = self
            .data_dir
            .parent()
            .unwrap()
            .join(format!("backup-{}-{}", self.node_id, timestamp));

        fs::create_dir_all(&backup_dir)?;

        // Copy all files from data directory
        for entry in fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let source = entry.path();
            let dest = backup_dir.join(entry.file_name());
            fs::copy(&source, &dest)?;
        }

        tracing::info!("Created backup at {:?}", backup_dir);
        Ok(backup_dir)
    }

    /// Clean old backups
    pub async fn cleanup_old_backups(&self) -> Result<()> {
        let parent_dir = self.data_dir.parent().unwrap();
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

        // Sort by name (which includes timestamp)
        backups.sort_by(|a, b| b.1.cmp(&a.1));

        // Remove old backups beyond retention limit
        for (path, _) in backups.iter().skip(self.config.backup_retention) {
            fs::remove_dir_all(path)?;
            tracing::info!("Removed old backup: {:?}", path);
        }

        Ok(())
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub node_id: OxirsNodeId,
    pub data_dir: PathBuf,
    pub log_entries: usize,
    pub current_term: u64,
    pub commit_index: u64,
    pub last_applied: u64,
    pub triple_count: usize,
    pub disk_usage_bytes: u64,
}

/// Storage-related errors
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Corruption detected in {file}: {message}")]
    Corruption { file: String, message: String },

    #[error("Snapshot not found")]
    SnapshotNotFound,

    #[error("Log entry not found at index {index}")]
    LogEntryNotFound { index: u64 },

    #[error("Invalid log range: {start} to {end}")]
    InvalidRange { start: u64, end: u64 },
}

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
}

/// Mock storage backend for testing
pub mod mock {
    use super::*;
    use std::collections::HashMap;
    
    #[derive(Debug, Default)]
    pub struct MockStorageBackend {
        shards: Arc<RwLock<HashMap<ShardId, Vec<Triple>>>>,
    }
    
    impl MockStorageBackend {
        pub fn new() -> Self {
            Self::default()
        }
    }
    
    #[async_trait]
    impl StorageBackend for MockStorageBackend {
        async fn create_shard(&self, shard_id: ShardId) -> Result<()> {
            self.shards.write().await.insert(shard_id, Vec::new());
            Ok(())
        }
        
        async fn delete_shard(&self, shard_id: ShardId) -> Result<()> {
            self.shards.write().await.remove(&shard_id);
            Ok(())
        }
        
        async fn insert_triple_to_shard(&self, shard_id: ShardId, triple: Triple) -> Result<()> {
            let mut shards = self.shards.write().await;
            if let Some(shard) = shards.get_mut(&shard_id) {
                shard.push(triple);
            }
            Ok(())
        }
        
        async fn delete_triple_from_shard(&self, shard_id: ShardId, triple: &Triple) -> Result<()> {
            let mut shards = self.shards.write().await;
            if let Some(shard) = shards.get_mut(&shard_id) {
                shard.retain(|t| t != triple);
            }
            Ok(())
        }
        
        async fn query_shard(
            &self,
            shard_id: ShardId,
            subject: Option<&str>,
            predicate: Option<&str>,
            object: Option<&str>,
        ) -> Result<Vec<Triple>> {
            let shards = self.shards.read().await;
            if let Some(shard) = shards.get(&shard_id) {
                let results: Vec<Triple> = shard.iter()
                    .filter(|triple| {
                        subject.map_or(true, |s| triple.subject.to_string() == s) &&
                        predicate.map_or(true, |p| triple.predicate.to_string() == p) &&
                        object.map_or(true, |o| triple.object.to_string() == o)
                    })
                    .cloned()
                    .collect();
                Ok(results)
            } else {
                Ok(Vec::new())
            }
        }
        
        async fn get_shard_size(&self, shard_id: ShardId) -> Result<u64> {
            let shards = self.shards.read().await;
            if let Some(shard) = shards.get(&shard_id) {
                // Estimate size as 100 bytes per triple
                Ok((shard.len() * 100) as u64)
            } else {
                Ok(0)
            }
        }
        
        async fn get_shard_triple_count(&self, shard_id: ShardId) -> Result<usize> {
            let shards = self.shards.read().await;
            Ok(shards.get(&shard_id).map_or(0, |s| s.len()))
        }
        
        async fn export_shard(&self, shard_id: ShardId) -> Result<Vec<Triple>> {
            let shards = self.shards.read().await;
            Ok(shards.get(&shard_id).cloned().unwrap_or_default())
        }
        
        async fn import_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()> {
            self.shards.write().await.insert(shard_id, triples);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raft::RdfCommand;
    use tempfile::TempDir;

    async fn create_test_storage() -> (PersistentStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            sync_writes: false, // Faster for tests
            max_log_entries: 100,
            compress_snapshots: false,
            backup_retention: 2,
        };
        let storage = PersistentStorage::new(1, config).await.unwrap();
        (storage, temp_dir)
    }

    #[tokio::test]
    async fn test_storage_creation() {
        let (storage, _temp_dir) = create_test_storage().await;
        assert_eq!(storage.node_id, 1);
        assert_eq!(storage.get_current_term().await, 0);
        assert_eq!(storage.get_voted_for().await, None);
    }

    #[tokio::test]
    async fn test_term_operations() {
        let (storage, _temp_dir) = create_test_storage().await;

        // Set term
        storage.set_current_term(5).await.unwrap();
        assert_eq!(storage.get_current_term().await, 5);

        // Vote should be reset when term changes
        assert_eq!(storage.get_voted_for().await, None);
    }

    #[tokio::test]
    async fn test_vote_operations() {
        let (storage, _temp_dir) = create_test_storage().await;

        // Set vote
        storage.set_voted_for(Some(2)).await.unwrap();
        assert_eq!(storage.get_voted_for().await, Some(2));

        // Clear vote
        storage.set_voted_for(None).await.unwrap();
        assert_eq!(storage.get_voted_for().await, None);
    }

    #[tokio::test]
    async fn test_log_operations() {
        let (storage, _temp_dir) = create_test_storage().await;

        let command = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        let entry = LogEntry::new(1, 1, command);

        // Append entry
        storage.append_entries(vec![entry.clone()]).await.unwrap();
        assert_eq!(storage.get_last_log_index().await, 1);
        assert_eq!(storage.get_last_log_term().await, 1);

        // Get entry
        let retrieved = storage.get_log_entry(1).await.unwrap();
        assert_eq!(retrieved.index, 1);
        assert_eq!(retrieved.term, 1);

        // Get entries in range
        let entries = storage.get_log_entries(1, 2).await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].index, 1);
    }

    #[tokio::test]
    async fn test_commit_operations() {
        let (storage, _temp_dir) = create_test_storage().await;

        storage.set_commit_index(5).await.unwrap();
        assert_eq!(storage.get_commit_index().await, 5);

        storage.set_last_applied(3).await.unwrap();
        assert_eq!(storage.get_last_applied().await, 3);
    }

    #[tokio::test]
    async fn test_application_state() {
        let (storage, _temp_dir) = create_test_storage().await;

        let command = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };

        storage.apply_command(&command).await.unwrap();

        let app_state = storage.get_app_state().await;
        assert_eq!(app_state.len(), 1);
    }

    #[tokio::test]
    async fn test_log_truncation() {
        let (storage, _temp_dir) = create_test_storage().await;

        // Add multiple entries
        for i in 1..=5 {
            let command = RdfCommand::Insert {
                subject: format!("s{}", i),
                predicate: "p".to_string(),
                object: "o".to_string(),
            };
            let entry = LogEntry::new(i, 1, command);
            storage.append_entries(vec![entry]).await.unwrap();
        }

        assert_eq!(storage.get_last_log_index().await, 5);

        // Truncate from index 3
        storage.truncate_log(3).await.unwrap();
        assert_eq!(storage.get_last_log_index().await, 2);

        // Entries 3, 4, 5 should be gone
        assert!(storage.get_log_entry(3).await.is_none());
        assert!(storage.get_log_entry(4).await.is_none());
        assert!(storage.get_log_entry(5).await.is_none());

        // Entries 1, 2 should still exist
        assert!(storage.get_log_entry(1).await.is_some());
        assert!(storage.get_log_entry(2).await.is_some());
    }

    #[tokio::test]
    async fn test_snapshot_operations() {
        let (storage, _temp_dir) = create_test_storage().await;

        // Add some state
        let command = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        storage.apply_command(&command).await.unwrap();

        // Create snapshot
        let metadata = storage.create_snapshot().await.unwrap();
        assert!(metadata.size > 0);

        // Load snapshot
        let loaded = storage.load_snapshot().await.unwrap();
        assert!(loaded.is_some());

        let (loaded_metadata, loaded_state) = loaded.unwrap();
        assert_eq!(loaded_metadata.size, metadata.size);
        assert_eq!(loaded_state.len(), 1);
    }

    #[tokio::test]
    async fn test_compaction_check() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            max_log_entries: 3, // Small limit for testing
            ..Default::default()
        };
        let storage = PersistentStorage::new(1, config).await.unwrap();

        // Should not need compaction initially
        assert!(!storage.needs_compaction().await);

        // Add entries beyond limit
        for i in 1..=5 {
            let command = RdfCommand::Insert {
                subject: format!("s{}", i),
                predicate: "p".to_string(),
                object: "o".to_string(),
            };
            let entry = LogEntry::new(i, 1, command);
            storage.append_entries(vec![entry]).await.unwrap();
        }

        // Should need compaction now
        assert!(storage.needs_compaction().await);
    }

    #[tokio::test]
    async fn test_storage_stats() {
        let (storage, _temp_dir) = create_test_storage().await;

        let stats = storage.get_stats().await;
        assert_eq!(stats.node_id, 1);
        assert_eq!(stats.log_entries, 0);
        assert_eq!(stats.current_term, 0);
        assert_eq!(stats.triple_count, 0);
    }

    #[tokio::test]
    async fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            sync_writes: false,
            ..Default::default()
        };

        // Create storage and add some state
        {
            let storage = PersistentStorage::new(1, config.clone()).await.unwrap();
            storage.set_current_term(5).await.unwrap();
            storage.set_voted_for(Some(2)).await.unwrap();

            let command = RdfCommand::Insert {
                subject: "s".to_string(),
                predicate: "p".to_string(),
                object: "o".to_string(),
            };
            let entry = LogEntry::new(1, 1, command.clone());
            storage.append_entries(vec![entry]).await.unwrap();
            storage.apply_command(&command).await.unwrap();
        }

        // Create new storage instance and verify state is loaded
        {
            let storage = PersistentStorage::new(1, config).await.unwrap();
            assert_eq!(storage.get_current_term().await, 5);
            assert_eq!(storage.get_voted_for().await, Some(2));
            assert_eq!(storage.get_last_log_index().await, 1);
            assert_eq!(storage.get_app_state().await.len(), 1);
        }
    }

    #[test]
    fn test_storage_error_display() {
        let err = StorageError::Corruption {
            file: "log.dat".to_string(),
            message: "checksum mismatch".to_string(),
        };
        assert!(err
            .to_string()
            .contains("Corruption detected in log.dat: checksum mismatch"));

        let err = StorageError::LogEntryNotFound { index: 42 };
        assert!(err.to_string().contains("Log entry not found at index 42"));

        let err = StorageError::InvalidRange { start: 10, end: 5 };
        assert!(err.to_string().contains("Invalid log range: 10 to 5"));
    }
}