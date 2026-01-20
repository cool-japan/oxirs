//! Storage configuration

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
    /// Enable corruption detection via checksums
    pub enable_corruption_detection: bool,
    /// Enable automatic crash recovery
    pub enable_crash_recovery: bool,
    /// Write ahead log for atomic writes
    pub enable_wal: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: "./data".to_string(),
            sync_writes: true,
            max_log_entries: 10000,
            compress_snapshots: true,
            backup_retention: 3,
            enable_corruption_detection: true,
            enable_crash_recovery: true,
            enable_wal: true,
        }
    }
}
