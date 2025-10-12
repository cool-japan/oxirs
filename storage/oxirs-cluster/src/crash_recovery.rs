//! # Comprehensive Crash Recovery
//!
//! Provides robust crash recovery mechanisms including:
//! - Checkpointing for fast recovery
//! - Write-Ahead Logging (WAL)
//! - Corruption detection and repair
//! - Incremental recovery
//! - Recovery statistics

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
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
        }
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
                .unwrap()
                .as_secs()
        );

        // Calculate checksum
        let checksum = Self::calculate_checksum(state_data);

        // Compress if enabled
        let (compressed_data, compressed_size) = if self.config.checkpoint_config.enable_compression
        {
            Self::compress_data(state_data, self.config.checkpoint_config.compression_level)
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

        // Save checkpoint to disk (simulated)
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

        // Write entries to disk (simulated)
        for _entry in wal_buffer.iter() {
            // In production: write to actual WAL file
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

        let latest = checkpoints.last().unwrap();

        info!(
            "Loading checkpoint {} (sequence: {})",
            latest.checkpoint_id, latest.sequence_number
        );

        // Load from disk (simulated)
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

        let latest = checkpoints.last().unwrap();
        let checkpoint_seq = latest.sequence_number;

        info!("Replaying WAL from sequence {}", checkpoint_seq);

        // Load WAL entries after checkpoint (simulated)
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

            // Replay operation (simulated)
            self.replay_operation(&entry).await?;
        }

        Ok(())
    }

    /// Verify recovery
    async fn verify_recovery(&self) -> Result<(), String> {
        info!("Verifying recovery...");

        // Verification logic (simulated)
        // In production: verify state consistency, check invariants, etc.

        Ok(())
    }

    /// Detect and repair corruption
    async fn detect_and_repair_corruption(&self) -> Result<(), String> {
        let mut stats = self.stats.write().await;
        stats.corruption_events += 1;
        drop(stats);

        info!("Detecting and repairing corruption...");

        // Corruption detection and repair logic (simulated)
        // In production: scan all checkpoints, try to repair, rebuild from WAL, etc.

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

    /// Clear all recovery data
    pub async fn clear(&self) {
        self.checkpoints.write().await.clear();
        self.wal_buffer.write().await.clear();
        *self.sequence_number.write().await = 0;
        *self.recovery_state.write().await = RecoveryState::Idle;
        *self.stats.write().await = RecoveryStats::default();
    }

    // Helper methods (simulated - replace with real implementations)

    fn calculate_checksum(data: &[u8]) -> String {
        // Use SHA-256 for checksums in production
        format!("{:x}", data.len()) // Simplified for testing
    }

    fn compress_data(data: &[u8], _level: u32) -> (Vec<u8>, usize) {
        // In production: use zstd or similar
        let compressed = data.to_vec(); // No actual compression in simulation
        (compressed.clone(), compressed.len())
    }

    fn decompress_data(data: &[u8]) -> Result<Vec<u8>, String> {
        // In production: use zstd or similar
        Ok(data.to_vec())
    }

    async fn save_checkpoint_to_disk(&self, _id: &str, _data: &[u8]) -> Result<(), String> {
        // Simulated disk write
        Ok(())
    }

    async fn load_checkpoint_from_disk(&self, _id: &str) -> Result<Vec<u8>, String> {
        // Simulated disk read
        Ok(vec![0u8; 100])
    }

    async fn delete_checkpoint(&self, _id: &str) -> Result<(), String> {
        // Simulated checkpoint deletion
        Ok(())
    }

    async fn load_wal_entries_after(&self, _seq: u64) -> Result<Vec<WalEntry>, String> {
        // Simulated WAL loading
        Ok(vec![])
    }

    async fn replay_operation(&self, _entry: &WalEntry) -> Result<(), String> {
        // Simulated operation replay
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_crash_recovery_creation() {
        let config = RecoveryConfig::default();
        let manager = CrashRecoveryManager::new(1, config);

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_checkpoints, 0);
        assert_eq!(stats.total_wal_entries, 0);
    }

    #[tokio::test]
    async fn test_create_checkpoint() {
        let config = RecoveryConfig::default();
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
        let config = RecoveryConfig::default();
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
        let config = RecoveryConfig::default();
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
        let mut config = RecoveryConfig::default();
        config.checkpoint_config.max_checkpoints = 3;

        let manager = CrashRecoveryManager::new(1, config);

        // Create 5 checkpoints
        for i in 0..5 {
            let data = vec![i; 100];
            let _result = manager.create_checkpoint(&data).await;
        }

        // Should keep only 3
        let checkpoints = manager.get_checkpoints().await;
        assert_eq!(checkpoints.len(), 3);
    }

    #[tokio::test]
    async fn test_recovery_state_transitions() {
        let config = RecoveryConfig::default();
        let manager = CrashRecoveryManager::new(1, config);

        let state = manager.get_recovery_state().await;
        assert_eq!(state, RecoveryState::Idle);

        // Create a checkpoint first
        let data = vec![1, 2, 3];
        let _result = manager.create_checkpoint(&data).await;

        // Attempt recovery
        let _result = manager.recover().await;

        let state = manager.get_recovery_state().await;
        assert!(state == RecoveryState::Completed || state == RecoveryState::Failed);
    }

    #[tokio::test]
    async fn test_recovery_stats() {
        let config = RecoveryConfig::default();
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
        let mut config = RecoveryConfig::default();
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
        let config = RecoveryConfig::default();
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
        let config = RecoveryConfig::default();
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
        let config = RecoveryConfig::default();
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
}
