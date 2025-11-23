//! Types for memory-mapped store
//!
//! This module contains the data types used by MmapStore for serialization
//! and statistics tracking.

use std::path::PathBuf;

/// Default page size for memory mapping (4KB)
pub(crate) const PAGE_SIZE: usize = 4096;

/// Magic number for file format identification
pub(crate) const MAGIC: &[u8; 8] = b"OXIRSMM\0";

/// Current file format version
pub(crate) const VERSION: u32 = 1;

/// Maximum size for a single memory map (1GB)
#[allow(dead_code)]
pub(crate) const MAX_MMAP_SIZE: usize = 1 << 30;

/// Header size (must be page-aligned)
pub(crate) const HEADER_SIZE: usize = PAGE_SIZE;

/// File header structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct FileHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub flags: u32,
    pub quad_count: u64,
    pub term_count: u64,
    pub data_offset: u64,
    pub index_offset: u64,
    pub term_offset: u64,
    pub checksum: [u8; 32],
    pub reserved: [u8; 3968], // Pad to PAGE_SIZE
}

impl FileHeader {
    pub fn new() -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            flags: 0,
            quad_count: 0,
            term_count: 0,
            data_offset: HEADER_SIZE as u64,
            index_offset: 0,
            term_offset: 0,
            checksum: [0; 32],
            reserved: [0; 3968],
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        use anyhow::bail;
        if self.magic != *MAGIC {
            bail!("Invalid magic number");
        }
        if self.version != VERSION {
            bail!("Unsupported version: {}", self.version);
        }
        Ok(())
    }

    pub fn compute_checksum(&mut self) {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(&self.magic);
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.flags.to_le_bytes());
        hasher.update(&self.quad_count.to_le_bytes());
        hasher.update(&self.term_count.to_le_bytes());
        hasher.update(&self.data_offset.to_le_bytes());
        hasher.update(&self.index_offset.to_le_bytes());
        hasher.update(&self.term_offset.to_le_bytes());
        self.checksum = *hasher.finalize().as_bytes();
    }
}

impl Default for FileHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// On-disk representation of a quad
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiskQuad {
    pub subject_id: u64,
    pub predicate_id: u64,
    pub object_id: u64,
    pub graph_id: u64,
}

/// Performance statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub add_operations: u64,
    pub batch_operations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_flush_time_ms: u64,
    pub average_batch_size: f64,
}

/// Access statistics for query optimization
#[derive(Debug, Clone, Default)]
pub struct AccessStats {
    /// Number of queries by index type
    pub spo_queries: u64,
    pub pos_queries: u64,
    pub osp_queries: u64,
    pub gspo_queries: u64,
    pub full_scans: u64,
    /// Average query latency in microseconds
    pub avg_query_latency_us: f64,
    /// Number of queries executed
    pub total_queries: u64,
    /// Hot subjects (frequently accessed)
    pub hot_subjects: Vec<(u64, u64)>, // (subject_id, access_count)
    /// Hot predicates (frequently accessed)
    pub hot_predicates: Vec<(u64, u64)>, // (predicate_id, access_count)
}

/// Backup metadata for incremental backups
#[derive(Debug, Clone)]
pub struct BackupMetadata {
    /// Timestamp of the backup
    pub timestamp: std::time::SystemTime,
    /// Number of quads in the backup
    pub quad_count: u64,
    /// Checkpoint marker (quad offset after which this backup was taken)
    pub checkpoint_offset: u64,
    /// Whether this is a full backup
    pub is_full_backup: bool,
    /// Backup file path
    pub backup_path: PathBuf,
}

/// Incremental backup configuration
#[derive(Debug, Clone)]
pub struct BackupConfig {
    /// Maximum number of incremental backups before forcing full backup
    pub max_incremental_chain: usize,
    /// Minimum number of changed quads to trigger incremental backup
    pub min_changes_for_backup: u64,
    /// Directory to store backups
    pub backup_dir: PathBuf,
}

/// Store statistics
#[derive(Debug, Clone)]
pub struct StoreStats {
    pub quad_count: u64,
    pub term_count: u64,
    pub data_size: u64,
    pub index_size: u64,
    pub term_size: u64,
}
