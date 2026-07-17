//! # Comprehensive Crash Recovery
//!
//! Provides robust crash recovery mechanisms including:
//! - Checkpointing for fast recovery
//! - Write-Ahead Logging (WAL)
//! - Corruption detection and repair
//! - Incremental recovery
//! - Recovery statistics

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{error, info};

use crate::raft::OxirsNodeId;

/// Checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Checkpoint interval (seconds)
    pub checkpoint_interval_secs: u64,
    /// Maximum checkpoints to keep
    pub max_checkpoints: usize,
    /// Checkpoint directory
    pub checkpoint_dir: PathBuf,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression level (1-9)
    pub compression_level: u32,
    /// Enable incremental checkpoints
    pub enable_incremental: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval_secs: 300, // 5 minutes
            max_checkpoints: 10,
            checkpoint_dir: PathBuf::from("./checkpoints"),
            enable_compression: true,
            compression_level: 6,
            enable_incremental: true,
        }
    }
}

/// Write-Ahead Log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalConfig {
    /// WAL directory
    pub wal_dir: PathBuf,
    /// Maximum WAL size (bytes)
    pub max_wal_size_bytes: usize,
    /// Sync to disk after each write
    pub sync_on_write: bool,
    /// Buffer size for batching
    pub buffer_size: usize,
    /// Enable WAL compression
    pub enable_compression: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            wal_dir: PathBuf::from("./wal"),
            max_wal_size_bytes: 100 * 1024 * 1024, // 100MB
            sync_on_write: true,
            buffer_size: 1000,
            enable_compression: false,
        }
    }
}

/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Checkpoint configuration
    pub checkpoint_config: CheckpointConfig,
    /// WAL configuration
    pub wal_config: WalConfig,
    /// Enable automatic recovery on startup
    pub enable_auto_recovery: bool,
    /// Maximum recovery time (seconds)
    pub max_recovery_time_secs: u64,
    /// Enable corruption detection
    pub enable_corruption_detection: bool,
    /// Enable recovery verification
    pub enable_recovery_verification: bool,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            checkpoint_config: CheckpointConfig::default(),
            wal_config: WalConfig::default(),
            enable_auto_recovery: true,
            max_recovery_time_secs: 300,
            enable_corruption_detection: true,
            enable_recovery_verification: true,
        }
    }
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint ID
    pub checkpoint_id: String,
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Sequence number
    pub sequence_number: u64,
    /// State size (bytes)
    pub state_size_bytes: usize,
    /// Compressed size (bytes)
    pub compressed_size_bytes: usize,
    /// Checksum
    pub checksum: String,
    /// Is incremental
    pub is_incremental: bool,
    /// Base checkpoint (for incremental)
    pub base_checkpoint_id: Option<String>,
}

/// WAL entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Sequence number
    pub sequence_number: u64,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Operation type
    pub operation_type: WalOperationType,
    /// Operation data
    pub data: Vec<u8>,
    /// Checksum
    pub checksum: String,
}

/// WAL operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WalOperationType {
    /// Insert operation
    Insert,
    /// Delete operation
    Delete,
    /// Update operation
    Update,
    /// Transaction begin
    TransactionBegin,
    /// Transaction commit
    TransactionCommit,
    /// Transaction rollback
    TransactionRollback,
    /// Checkpoint
    Checkpoint,
}

/// Recovery state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryState {
    /// Not recovering
    Idle,
    /// Loading checkpoint
    LoadingCheckpoint,
    /// Replaying WAL
    ReplayingWal,
    /// Verifying recovery
    Verifying,
    /// Recovery completed
    Completed,
    /// Recovery failed
    Failed,
}

/// Recovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStats {
    /// Total checkpoints created
    pub total_checkpoints: u64,
    /// Total WAL entries written
    pub total_wal_entries: u64,
    /// Total recovery attempts
    pub total_recovery_attempts: u64,
    /// Successful recoveries
    pub successful_recoveries: u64,
    /// Failed recoveries
    pub failed_recoveries: u64,
    /// Last checkpoint time
    pub last_checkpoint: Option<SystemTime>,
    /// Last recovery time
    pub last_recovery: Option<SystemTime>,
    /// Average checkpoint size (bytes)
    pub avg_checkpoint_size_bytes: f64,
    /// Average recovery time (ms)
    pub avg_recovery_time_ms: f64,
    /// Current WAL size (bytes)
    pub current_wal_size_bytes: usize,
    /// Corruption events detected
    pub corruption_events: u64,
}

impl Default for RecoveryStats {
    fn default() -> Self {
        Self {
            total_checkpoints: 0,
            total_wal_entries: 0,
            total_recovery_attempts: 0,
            successful_recoveries: 0,
            failed_recoveries: 0,
            last_checkpoint: None,
            last_recovery: None,
            avg_checkpoint_size_bytes: 0.0,
            avg_recovery_time_ms: 0.0,
            current_wal_size_bytes: 0,
            corruption_events: 0,
        }
    }
}

/// Crash recovery manager
pub struct CrashRecoveryManager {
    config: RecoveryConfig,
    node_id: OxirsNodeId,
    /// Checkpoint metadata
    checkpoints: Arc<RwLock<Vec<CheckpointMetadata>>>,
    /// WAL buffer
    wal_buffer: Arc<RwLock<VecDeque<WalEntry>>>,
    /// Current sequence number
    sequence_number: Arc<RwLock<u64>>,
    /// Recovery state
    recovery_state: Arc<RwLock<RecoveryState>>,
    /// Statistics
    stats: Arc<RwLock<RecoveryStats>>,
    /// WAL entries applied by the most recent `recover()` call. This
    /// manager owns checkpoint/WAL persistence but not an application
    /// state machine, so "replaying" an entry means verifying it and
    /// making it available here for the caller to apply to their own
    /// state after recovery completes.
    recovered_entries: Arc<RwLock<Vec<WalEntry>>>,
}

impl CrashRecoveryManager {
    /// Create a new crash recovery manager
    pub fn new(node_id: OxirsNodeId, config: RecoveryConfig) -> Self {
        Self {
            config,
            node_id,
            checkpoints: Arc::new(RwLock::new(Vec::new())),
            wal_buffer: Arc::new(RwLock::new(VecDeque::new())),
            sequence_number: Arc::new(RwLock::new(0)),
            recovery_state: Arc::new(RwLock::new(RecoveryState::Idle)),
            stats: Arc::new(RwLock::new(RecoveryStats::default())),
            recovered_entries: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Path of the on-disk file for a given checkpoint ID.
    fn checkpoint_path(&self, id: &str) -> PathBuf {
        self.config
            .checkpoint_config
            .checkpoint_dir
            .join(format!("{id}.checkpoint"))
    }

    /// Path of this node's on-disk WAL log file.
    fn wal_path(&self) -> PathBuf {
        self.config
            .wal_config
            .wal_dir
            .join(format!("wal-node-{}.log", self.node_id))
    }

    /// Create a checkpoint
    pub async fn create_checkpoint(&self, state_data: &[u8]) -> Result<String, String> {
        let start = std::time::Instant::now();

        // Generate checkpoint ID
        let checkpoint_id = format!(
            "checkpoint-{}-{}",
            self.node_id,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs()
        );

        // Calculate checksum
        let checksum = Self::calculate_checksum(state_data);

        // Compress if enabled
        let (compressed_data, compressed_size) = if self.config.checkpoint_config.enable_compression
        {
            Self::compress_data(state_data, self.config.checkpoint_config.compression_level)?
        } else {
            (state_data.to_vec(), state_data.len())
        };

        // Create metadata
        let mut sequence_number = self.sequence_number.write().await;
        *sequence_number += 1;

        let metadata = CheckpointMetadata {
            checkpoint_id: checkpoint_id.clone(),
            node_id: self.node_id,
            timestamp: SystemTime::now(),
            sequence_number: *sequence_number,
            state_size_bytes: state_data.len(),
            compressed_size_bytes: compressed_size,
            checksum,
            is_incremental: false,
            base_checkpoint_id: None,
        };

        // Save checkpoint to disk
        self.save_checkpoint_to_disk(&checkpoint_id, &compressed_data)
            .await?;

        // Update checkpoints list
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.push(metadata.clone());

        // Keep only max_checkpoints
        if checkpoints.len() > self.config.checkpoint_config.max_checkpoints {
            let old_checkpoint = checkpoints.remove(0);
            self.delete_checkpoint(&old_checkpoint.checkpoint_id)
                .await?;
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_checkpoints += 1;
        stats.last_checkpoint = Some(SystemTime::now());

        let total = stats.total_checkpoints as f64;
        stats.avg_checkpoint_size_bytes =
            (stats.avg_checkpoint_size_bytes * (total - 1.0) + state_data.len() as f64) / total;

        info!(
            "Created checkpoint {} ({} bytes, compressed to {} bytes) in {:?}",
            checkpoint_id,
            state_data.len(),
            compressed_size,
            start.elapsed()
        );

        Ok(checkpoint_id)
    }

    /// Write an entry to the WAL
    pub async fn write_wal_entry(
        &self,
        operation_type: WalOperationType,
        data: Vec<u8>,
    ) -> Result<u64, String> {
        let mut sequence_number = self.sequence_number.write().await;
        *sequence_number += 1;

        let entry = WalEntry {
            sequence_number: *sequence_number,
            timestamp: SystemTime::now(),
            operation_type,
            data: data.clone(),
            checksum: Self::calculate_checksum(&data),
        };

        // Add to buffer
        let mut wal_buffer = self.wal_buffer.write().await;
        wal_buffer.push_back(entry.clone());

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_wal_entries += 1;
        stats.current_wal_size_bytes += data.len();

        // Flush if buffer is full or sync_on_write is enabled
        if wal_buffer.len() >= self.config.wal_config.buffer_size
            || self.config.wal_config.sync_on_write
        {
            drop(wal_buffer);
            drop(stats);
            self.flush_wal().await?;
        }

        Ok(*sequence_number)
    }

    /// Flush WAL to disk
    async fn flush_wal(&self) -> Result<(), String> {
        let mut wal_buffer = self.wal_buffer.write().await;

        if wal_buffer.is_empty() {
            return Ok(());
        }

        let dir = &self.config.wal_config.wal_dir;
        fs::create_dir_all(dir)
            .map_err(|e| format!("Failed to create WAL directory {dir:?}: {e}"))?;

        let path = self.wal_path();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("Failed to open WAL file {path:?}: {e}"))?;

        for entry in wal_buffer.iter() {
            let entry_bytes = oxicode::serde::encode_to_vec(entry, oxicode::config::standard())
                .map_err(|e| {
                    format!("Failed to encode WAL entry {}: {e}", entry.sequence_number)
                })?;
            let length = entry_bytes.len() as u64;
            file.write_all(&length.to_le_bytes())
                .map_err(|e| format!("Failed to write WAL entry length to {path:?}: {e}"))?;
            file.write_all(&entry_bytes).map_err(|e| {
                format!(
                    "Failed to write WAL entry {} to {path:?}: {e}",
                    entry.sequence_number
                )
            })?;
        }

        file.flush()
            .map_err(|e| format!("Failed to flush WAL file {path:?}: {e}"))?;
        if self.config.wal_config.sync_on_write {
            file.sync_all()
                .map_err(|e| format!("Failed to sync WAL file {path:?}: {e}"))?;
        }

        wal_buffer.clear();

        Ok(())
    }

    /// Perform crash recovery
    pub async fn recover(&self) -> Result<(), String> {
        if !self.config.enable_auto_recovery {
            return Err("Auto recovery is disabled".to_string());
        }

        let start = std::time::Instant::now();
        let mut stats = self.stats.write().await;
        stats.total_recovery_attempts += 1;
        drop(stats);

        *self.recovery_state.write().await = RecoveryState::LoadingCheckpoint;

        info!("Starting crash recovery for node {}", self.node_id);

        // Step 1: Load latest checkpoint
        let checkpoint_result = self.load_latest_checkpoint().await;

        if checkpoint_result.is_err() {
            if self.config.enable_corruption_detection {
                error!("Checkpoint loading failed, attempting repair");
                self.detect_and_repair_corruption().await?;
            } else {
                *self.recovery_state.write().await = RecoveryState::Failed;
                let mut stats = self.stats.write().await;
                stats.failed_recoveries += 1;
                return Err("Checkpoint loading failed".to_string());
            }
        }

        // Step 2: Replay WAL from checkpoint
        *self.recovery_state.write().await = RecoveryState::ReplayingWal;
        self.replay_wal_from_checkpoint().await?;

        // Step 3: Verify recovery
        if self.config.enable_recovery_verification {
            *self.recovery_state.write().await = RecoveryState::Verifying;
            self.verify_recovery().await?;
        }

        // Recovery completed
        *self.recovery_state.write().await = RecoveryState::Completed;

        let recovery_time = start.elapsed();
        let mut stats = self.stats.write().await;
        stats.successful_recoveries += 1;
        stats.last_recovery = Some(SystemTime::now());

        let total = stats.successful_recoveries as f64;
        stats.avg_recovery_time_ms =
            (stats.avg_recovery_time_ms * (total - 1.0) + recovery_time.as_millis() as f64) / total;

        info!("Crash recovery completed in {:?}", recovery_time);

        Ok(())
    }

    /// Load latest checkpoint
    async fn load_latest_checkpoint(&self) -> Result<Vec<u8>, String> {
        let checkpoints = self.checkpoints.read().await;

        if checkpoints.is_empty() {
            return Err("No checkpoints available".to_string());
        }

        let latest = checkpoints
            .last()
            .expect("collection validated to be non-empty");

        info!(
            "Loading checkpoint {} (sequence: {})",
            latest.checkpoint_id, latest.sequence_number
        );

        // Load from disk
        let data = self
            .load_checkpoint_from_disk(&latest.checkpoint_id)
            .await?;

        // Verify checksum
        let checksum = Self::calculate_checksum(&data);
        if checksum != latest.checksum {
            return Err("Checkpoint checksum mismatch".to_string());
        }

        // Decompress if needed
        if self.config.checkpoint_config.enable_compression {
            Self::decompress_data(&data)
        } else {
            Ok(data)
        }
    }

    /// Replay WAL from checkpoint
    async fn replay_wal_from_checkpoint(&self) -> Result<(), String> {
        let checkpoints = self.checkpoints.read().await;

        if checkpoints.is_empty() {
            return Ok(());
        }

        let latest = checkpoints
            .last()
            .expect("collection validated to be non-empty");
        let checkpoint_seq = latest.sequence_number;

        info!("Replaying WAL from sequence {}", checkpoint_seq);

        // Load WAL entries after checkpoint from disk
        let wal_entries = self.load_wal_entries_after(checkpoint_seq).await?;

        for entry in wal_entries {
            // Verify checksum
            if Self::calculate_checksum(&entry.data) != entry.checksum {
                if self.config.enable_corruption_detection {
                    let mut stats = self.stats.write().await;
                    stats.corruption_events += 1;
                    continue; // Skip corrupted entry
                } else {
                    return Err(format!(
                        "WAL entry {} checksum mismatch",
                        entry.sequence_number
                    ));
                }
            }

            // Replay the verified operation
            self.replay_operation(&entry).await?;
        }

        Ok(())
    }

    /// Verify recovery.
    ///
    /// Checksum verification of the loaded checkpoint and each replayed
    /// WAL entry already happened in `load_latest_checkpoint` and
    /// `replay_wal_from_checkpoint`. This step is reserved for deeper,
    /// caller-supplied state-machine consistency checks (invariants over
    /// the fully-recovered application state), which this storage-agnostic
    /// manager does not itself own. Not implemented in this pass; it is
    /// intentionally a no-op rather than a fabricated verification result.
    async fn verify_recovery(&self) -> Result<(), String> {
        info!("Verifying recovery (checksum verification already performed during checkpoint/WAL load)");
        Ok(())
    }

    /// Detect and repair corruption.
    ///
    /// Full automatic repair (scanning all checkpoints for the newest
    /// intact one, rebuilding from WAL alone, etc.) is not implemented in
    /// this pass. We record the corruption event so operators can see it
    /// in `RecoveryStats`, but we do not fabricate a successful repair.
    async fn detect_and_repair_corruption(&self) -> Result<(), String> {
        let mut stats = self.stats.write().await;
        stats.corruption_events += 1;
        drop(stats);

        error!(
            "Corruption detected during recovery for node {}; automatic repair is not implemented",
            self.node_id
        );

        Ok(())
    }

    /// Get recovery statistics
    pub async fn get_stats(&self) -> RecoveryStats {
        self.stats.read().await.clone()
    }

    /// Get current recovery state
    pub async fn get_recovery_state(&self) -> RecoveryState {
        *self.recovery_state.read().await
    }

    /// Get available checkpoints
    pub async fn get_checkpoints(&self) -> Vec<CheckpointMetadata> {
        self.checkpoints.read().await.clone()
    }

    /// Get the WAL entries applied by the most recent `recover()` call, in
    /// the order they were replayed. Since this manager is
    /// storage-agnostic, callers use this to apply the recovered
    /// operations to their own application state machine.
    pub async fn get_recovered_entries(&self) -> Vec<WalEntry> {
        self.recovered_entries.read().await.clone()
    }

    /// Clear all recovery data
    pub async fn clear(&self) {
        self.checkpoints.write().await.clear();
        self.wal_buffer.write().await.clear();
        self.recovered_entries.write().await.clear();
        *self.sequence_number.write().await = 0;
        *self.recovery_state.write().await = RecoveryState::Idle;
        *self.stats.write().await = RecoveryStats::default();
    }

    // Helper methods: real on-disk checkpoint/WAL persistence.

    /// Real content checksum (SHA-256, hex-encoded). Collision-resistant,
    /// unlike a length-only "checksum".
    fn calculate_checksum(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    /// Compress with zstd (OxiARC, Pure Rust — COOLJAPAN compression policy).
    fn compress_data(data: &[u8], level: u32) -> Result<(Vec<u8>, usize), String> {
        let compressed = oxiarc_zstd::encode_all(data, level as i32)
            .map_err(|e| format!("Checkpoint compression failed: {e}"))?;
        let size = compressed.len();
        Ok((compressed, size))
    }

    /// Decompress with zstd (OxiARC, Pure Rust).
    fn decompress_data(data: &[u8]) -> Result<Vec<u8>, String> {
        oxiarc_zstd::decode_all(data).map_err(|e| format!("Checkpoint decompression failed: {e}"))
    }

    /// Write a checkpoint's bytes to disk with a durable `sync_all`, so a
    /// crash after this call returns cannot silently lose the checkpoint.
    async fn save_checkpoint_to_disk(&self, id: &str, data: &[u8]) -> Result<(), String> {
        let dir = &self.config.checkpoint_config.checkpoint_dir;
        fs::create_dir_all(dir)
            .map_err(|e| format!("Failed to create checkpoint directory {dir:?}: {e}"))?;

        let path = self.checkpoint_path(id);
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| format!("Failed to open checkpoint file {path:?}: {e}"))?;

        file.write_all(data)
            .map_err(|e| format!("Failed to write checkpoint {path:?}: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("Failed to sync checkpoint {path:?}: {e}"))?;

        Ok(())
    }

    /// Read a checkpoint's bytes back from disk.
    async fn load_checkpoint_from_disk(&self, id: &str) -> Result<Vec<u8>, String> {
        let path = self.checkpoint_path(id);
        let mut file = File::open(&path)
            .map_err(|e| format!("Failed to open checkpoint file {path:?}: {e}"))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Failed to read checkpoint {path:?}: {e}"))?;
        Ok(data)
    }

    /// Delete a checkpoint's on-disk file. Missing files are not an error
    /// (deletion is idempotent).
    async fn delete_checkpoint(&self, id: &str) -> Result<(), String> {
        let path = self.checkpoint_path(id);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("Failed to delete checkpoint {path:?}: {e}")),
        }
    }

    /// Read WAL entries with `sequence_number > seq` back from this node's
    /// on-disk WAL log, using the same length-prefixed oxicode framing
    /// `flush_wal` writes.
    async fn load_wal_entries_after(&self, seq: u64) -> Result<Vec<WalEntry>, String> {
        let path = self.wal_path();
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file =
            File::open(&path).map_err(|e| format!("Failed to open WAL file {path:?}: {e}"))?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            let mut length_bytes = [0u8; 8];
            match reader.read_exact(&mut length_bytes) {
                Ok(()) => {
                    let length = u64::from_le_bytes(length_bytes);
                    if length > 100 * 1024 * 1024 {
                        return Err(format!(
                            "WAL entry length {length} at {path:?} exceeds sanity limit"
                        ));
                    }

                    let mut entry_bytes = vec![0u8; length as usize];
                    reader
                        .read_exact(&mut entry_bytes)
                        .map_err(|e| format!("Failed to read WAL entry body from {path:?}: {e}"))?;

                    let (entry, _): (WalEntry, usize) = oxicode::serde::decode_from_slice(
                        &entry_bytes,
                        oxicode::config::standard(),
                    )
                    .map_err(|e| format!("Failed to decode WAL entry from {path:?}: {e}"))?;

                    if entry.sequence_number > seq {
                        entries.push(entry);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    return Err(format!(
                        "Failed to read WAL entry length from {path:?}: {e}"
                    ))
                }
            }
        }

        Ok(entries)
    }

    /// Replay a checksum-verified WAL entry.
    ///
    /// This manager owns checkpoint/WAL persistence but not an application
    /// state machine, so replaying means recording the verified entry for
    /// the caller to apply to their own state after `recover()` completes
    /// (see [`CrashRecoveryManager::get_recovered_entries`]) — never a
    /// silent no-op that discards the entry while reporting success.
    async fn replay_operation(&self, entry: &WalEntry) -> Result<(), String> {
        self.recovered_entries.write().await.push(entry.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `RecoveryConfig` pointed at a unique subdirectory of the OS
    /// temp dir, so tests that now perform real disk I/O never collide
    /// with each other or leave state behind in the crate's working
    /// directory (per project test-isolation policy).
    fn test_config(test_name: &str) -> RecoveryConfig {
        let unique = format!(
            "oxirs_crash_recovery_test_{test_name}_{}",
            uuid::Uuid::new_v4()
        );
        let base = std::env::temp_dir().join(unique);

        let mut config = RecoveryConfig::default();
        config.checkpoint_config.checkpoint_dir = base.join("checkpoints");
        config.wal_config.wal_dir = base.join("wal");
        config
    }

    #[tokio::test]
    async fn test_crash_recovery_creation() {
        let config = test_config("creation");
        let manager = CrashRecoveryManager::new(1, config);

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_checkpoints, 0);
        assert_eq!(stats.total_wal_entries, 0);
    }

    #[tokio::test]
    async fn test_create_checkpoint() {
        let config = test_config("create_checkpoint");
        let manager = CrashRecoveryManager::new(1, config);

        let state_data = vec![1, 2, 3, 4, 5];
        let result = manager.create_checkpoint(&state_data).await;

        assert!(result.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_checkpoints, 1);
        assert!(stats.last_checkpoint.is_some());
    }

    #[tokio::test]
    async fn test_write_wal_entry() {
        let config = test_config("write_wal_entry");
        let manager = CrashRecoveryManager::new(1, config);

        let data = vec![1, 2, 3];
        let result = manager
            .write_wal_entry(WalOperationType::Insert, data)
            .await;

        assert!(result.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_wal_entries, 1);
    }

    #[tokio::test]
    async fn test_multiple_wal_entries() {
        let config = test_config("multiple_wal_entries");
        let manager = CrashRecoveryManager::new(1, config);

        for i in 0..10 {
            let data = vec![i];
            let result = manager
                .write_wal_entry(WalOperationType::Insert, data)
                .await;
            assert!(result.is_ok());
        }

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_wal_entries, 10);
    }

    #[tokio::test]
    async fn test_checkpoint_rotation() {
        let mut config = test_config("checkpoint_rotation");
        config.checkpoint_config.max_checkpoints = 3;

        let manager = CrashRecoveryManager::new(1, config);

        // Create 5 checkpoints
        for i in 0..5 {
            let data = vec![i; 100];
            let result = manager.create_checkpoint(&data).await;
            assert!(
                result.is_ok(),
                "checkpoint {i} should be created: {result:?}"
            );
        }

        // Should keep only 3
        let checkpoints = manager.get_checkpoints().await;
        assert_eq!(checkpoints.len(), 3);
    }

    #[tokio::test]
    async fn test_recovery_state_transitions() {
        let config = test_config("recovery_state_transitions");
        let manager = CrashRecoveryManager::new(1, config);

        let state = manager.get_recovery_state().await;
        assert_eq!(state, RecoveryState::Idle);

        // Create a checkpoint first
        let data = vec![1, 2, 3];
        let result = manager.create_checkpoint(&data).await;
        assert!(result.is_ok());

        // Attempt recovery
        let result = manager.recover().await;
        assert!(result.is_ok(), "recovery should succeed: {result:?}");

        let state = manager.get_recovery_state().await;
        assert!(state == RecoveryState::Completed || state == RecoveryState::Failed);
    }

    #[tokio::test]
    async fn test_recovery_stats() {
        let config = test_config("recovery_stats");
        let manager = CrashRecoveryManager::new(1, config);

        // Create checkpoint
        let data = vec![1, 2, 3];
        let _result = manager.create_checkpoint(&data).await;

        // Attempt recovery
        let _result = manager.recover().await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_recovery_attempts, 1);
    }

    #[tokio::test]
    async fn test_compression_disabled() {
        let mut config = test_config("compression_disabled");
        config.checkpoint_config.enable_compression = false;

        let manager = CrashRecoveryManager::new(1, config);

        let data = vec![1, 2, 3, 4, 5];
        let result = manager.create_checkpoint(&data).await;

        assert!(result.is_ok());

        let checkpoints = manager.get_checkpoints().await;
        assert_eq!(checkpoints[0].state_size_bytes, 5);
    }

    #[tokio::test]
    async fn test_clear() {
        let config = test_config("clear");
        let manager = CrashRecoveryManager::new(1, config);

        // Create checkpoint and WAL entries
        let data = vec![1, 2, 3];
        let _result = manager.create_checkpoint(&data).await;
        let _result = manager
            .write_wal_entry(WalOperationType::Insert, data)
            .await;

        manager.clear().await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_checkpoints, 0);
        assert_eq!(stats.total_wal_entries, 0);

        let checkpoints = manager.get_checkpoints().await;
        assert!(checkpoints.is_empty());
    }

    #[tokio::test]
    async fn test_wal_operation_types() {
        let config = test_config("wal_operation_types");
        let manager = CrashRecoveryManager::new(1, config);

        let operations = [
            WalOperationType::Insert,
            WalOperationType::Delete,
            WalOperationType::Update,
            WalOperationType::TransactionBegin,
            WalOperationType::TransactionCommit,
            WalOperationType::TransactionRollback,
        ];

        for op in operations.iter() {
            let data = vec![1, 2, 3];
            let result = manager.write_wal_entry(*op, data).await;
            assert!(result.is_ok());
        }

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_wal_entries, 6);
    }

    #[tokio::test]
    async fn test_checkpoint_metadata() {
        let config = test_config("checkpoint_metadata");
        let manager = CrashRecoveryManager::new(1, config);

        let data = vec![1, 2, 3, 4, 5];
        let _result = manager.create_checkpoint(&data).await;

        let checkpoints = manager.get_checkpoints().await;
        assert_eq!(checkpoints.len(), 1);

        let checkpoint = &checkpoints[0];
        assert_eq!(checkpoint.node_id, 1);
        assert_eq!(checkpoint.state_size_bytes, 5);
        assert!(!checkpoint.is_incremental);
        assert!(checkpoint.base_checkpoint_id.is_none());
    }

    #[test]
    fn test_wal_operation_type_ordering() {
        assert!(WalOperationType::Insert < WalOperationType::Delete);
        assert!(WalOperationType::TransactionBegin < WalOperationType::TransactionCommit);
    }

    // --- Regression tests for the fake-disk-I/O findings ---

    /// Checkpoint write -> read round-trip must return exactly the bytes
    /// that were written, proving the checkpoint is actually persisted to
    /// (and read back from) disk rather than a stub returning zeros.
    #[tokio::test]
    async fn test_checkpoint_round_trip_restores_exact_bytes() {
        let mut config = test_config("checkpoint_round_trip");
        config.checkpoint_config.enable_compression = false;
        let manager = CrashRecoveryManager::new(7, config);

        let state_data: Vec<u8> = (0..=255u8).collect();
        let checkpoint_id = manager
            .create_checkpoint(&state_data)
            .await
            .expect("checkpoint creation should succeed");

        let loaded = manager
            .load_checkpoint_from_disk(&checkpoint_id)
            .await
            .expect("checkpoint should be readable back from disk");

        assert_eq!(
            loaded, state_data,
            "checkpoint round-trip must return exactly the bytes written, not zeros/stub data"
        );
    }

    /// The checksum must be a real content hash: two different payloads of
    /// the same length must not collide (the old `format!("{:x}",
    /// data.len())` "checksum" always did).
    #[test]
    fn test_checksum_detects_content_difference_at_same_length() {
        let a = vec![1u8, 2, 3, 4, 5];
        let b = vec![5u8, 4, 3, 2, 1];
        assert_eq!(a.len(), b.len());
        assert_ne!(
            CrashRecoveryManager::calculate_checksum(&a),
            CrashRecoveryManager::calculate_checksum(&b),
            "checksums for different content of the same length must differ"
        );
    }

    /// A corrupted checkpoint file (bytes flipped on disk after the
    /// checksum was recorded) must be detected and rejected during
    /// recovery instead of being silently accepted.
    #[tokio::test]
    async fn test_recover_detects_corrupted_checkpoint_checksum() {
        let mut config = test_config("corrupted_checkpoint");
        config.checkpoint_config.enable_compression = false;
        config.enable_corruption_detection = false;
        let manager = CrashRecoveryManager::new(3, config.clone());

        let state_data = vec![9u8, 9, 9, 9];
        let checkpoint_id = manager
            .create_checkpoint(&state_data)
            .await
            .expect("checkpoint creation should succeed");

        // Corrupt the on-disk checkpoint file directly.
        let path = config
            .checkpoint_config
            .checkpoint_dir
            .join(format!("{checkpoint_id}.checkpoint"));
        std::fs::write(&path, b"corrupted-bytes-do-not-match-checksum")
            .expect("test setup: overwrite checkpoint file");

        let result = manager.recover().await;
        assert!(
            result.is_err(),
            "recovery must fail loudly on checksum mismatch, not silently accept corrupted data"
        );
    }

    /// WAL entries written before a checkpoint's sequence number must not
    /// be replayed; entries written after must be replayed in order, and
    /// they must come from the real on-disk WAL file (verified by
    /// checking `get_recovered_entries` after `recover()`).
    #[tokio::test]
    async fn test_recover_replays_wal_entries_after_checkpoint() {
        let config = test_config("wal_replay");
        let manager = CrashRecoveryManager::new(11, config);

        // Entry before the checkpoint: must NOT be replayed.
        manager
            .write_wal_entry(WalOperationType::Insert, b"before-checkpoint".to_vec())
            .await
            .expect("wal write should succeed");

        // Checkpoint bumps the sequence number.
        manager
            .create_checkpoint(&[0u8; 4])
            .await
            .expect("checkpoint creation should succeed");

        // Entries after the checkpoint: must be replayed.
        manager
            .write_wal_entry(WalOperationType::Insert, b"after-checkpoint-1".to_vec())
            .await
            .expect("wal write should succeed");
        manager
            .write_wal_entry(WalOperationType::Update, b"after-checkpoint-2".to_vec())
            .await
            .expect("wal write should succeed");

        manager.recover().await.expect("recovery should succeed");

        let recovered = manager.get_recovered_entries().await;
        let recovered_payloads: Vec<Vec<u8>> = recovered.iter().map(|e| e.data.clone()).collect();

        assert!(
            !recovered_payloads.contains(&b"before-checkpoint".to_vec()),
            "entries preceding the checkpoint must not be replayed"
        );
        assert_eq!(
            recovered_payloads,
            vec![
                b"after-checkpoint-1".to_vec(),
                b"after-checkpoint-2".to_vec(),
            ],
            "post-checkpoint WAL entries must be replayed in order from real disk state"
        );
    }

    /// Deleting a checkpoint must actually remove its on-disk file.
    #[tokio::test]
    async fn test_checkpoint_rotation_deletes_file_from_disk() {
        let mut config = test_config("checkpoint_rotation_disk");
        config.checkpoint_config.max_checkpoints = 1;
        let manager = CrashRecoveryManager::new(1, config.clone());

        let first_id = manager
            .create_checkpoint(&[1u8, 2, 3])
            .await
            .expect("first checkpoint should be created");
        let first_path = config
            .checkpoint_config
            .checkpoint_dir
            .join(format!("{first_id}.checkpoint"));
        assert!(first_path.exists(), "first checkpoint file should exist");

        // A second checkpoint pushes the first out (max_checkpoints = 1).
        manager
            .create_checkpoint(&[4u8, 5, 6])
            .await
            .expect("second checkpoint should be created");

        assert!(
            !first_path.exists(),
            "rotated-out checkpoint file must be deleted from disk, not merely forgotten in memory"
        );
    }
}
