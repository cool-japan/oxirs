//! Write-Ahead Log Archiving for Point-in-Time Recovery
//!
//! This module provides WAL archiving capabilities for continuous backup and
//! point-in-time recovery (PITR). Archived WAL files can be used to restore
//! the database to any point in time.
//!
//! Features:
//! - Automatic WAL file archiving on rotation
//! - Configurable archive location
//! - Archive compression support
//! - Archive verification and validation
//! - Cleanup of old archives based on retention policy
//! - Archive metadata tracking

use crate::error::{Result, TdbError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// WAL archive configuration
#[derive(Debug, Clone)]
pub struct WalArchiveConfig {
    /// Archive directory path
    pub archive_dir: PathBuf,
    /// Enable WAL archiving
    pub enable_archiving: bool,
    /// Compress archived WAL files
    pub compress_archives: bool,
    /// Archive retention period (0 = keep forever)
    pub retention_period: Duration,
    /// Maximum archive size in bytes (0 = unlimited)
    pub max_archive_size: u64,
    /// Verify archives after creation
    pub verify_archives: bool,
    /// Archive file prefix
    pub archive_prefix: String,
}

impl Default for WalArchiveConfig {
    fn default() -> Self {
        Self {
            archive_dir: PathBuf::from("wal_archive"),
            enable_archiving: false,
            compress_archives: true,
            retention_period: Duration::from_secs(30 * 24 * 3600), // 30 days
            max_archive_size: 10 * 1024 * 1024 * 1024,             // 10 GB
            verify_archives: true,
            archive_prefix: "wal".to_string(),
        }
    }
}

/// WAL archive metadata
#[derive(Debug, Clone)]
pub struct WalArchiveMetadata {
    /// Archive file name
    pub file_name: String,
    /// Original WAL file path
    pub original_path: PathBuf,
    /// Archive creation time
    pub archived_at: SystemTime,
    /// Archive file size
    pub file_size: u64,
    /// Whether archive is compressed
    pub compressed: bool,
    /// Checksum of archive content
    pub checksum: u32,
    /// LSN range covered by this archive [start, end]
    pub lsn_range: (u64, u64),
}

/// WAL archiver
pub struct WalArchiver {
    /// Configuration
    config: WalArchiveConfig,
    /// Archive metadata index
    metadata: RwLock<HashMap<String, WalArchiveMetadata>>,
    /// Total archives created
    total_archives: AtomicU64,
    /// Total bytes archived
    total_bytes_archived: AtomicU64,
    /// Total archives deleted
    total_deleted: AtomicU64,
    /// Whether archiver is active
    active: AtomicBool,
    /// Statistics
    stats: WalArchiverStats,
}

impl WalArchiver {
    /// Create a new WAL archiver
    pub fn new(config: WalArchiveConfig) -> Result<Self> {
        // Create archive directory if it doesn't exist
        if config.enable_archiving {
            fs::create_dir_all(&config.archive_dir).map_err(|e| {
                TdbError::Other(format!("Failed to create archive directory: {}", e))
            })?;
        }

        Ok(Self {
            config,
            metadata: RwLock::new(HashMap::new()),
            total_archives: AtomicU64::new(0),
            total_bytes_archived: AtomicU64::new(0),
            total_deleted: AtomicU64::new(0),
            active: AtomicBool::new(true),
            stats: WalArchiverStats::default(),
        })
    }

    /// Archive a WAL file
    pub fn archive_wal_file(
        &self,
        wal_file_path: &Path,
        lsn_start: u64,
        lsn_end: u64,
    ) -> Result<String> {
        if !self.config.enable_archiving {
            return Err(TdbError::Other("WAL archiving is disabled".to_string()));
        }

        if !self.active.load(Ordering::Acquire) {
            return Err(TdbError::Other("WAL archiver is not active".to_string()));
        }

        // Generate archive file name
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        let archive_name = if self.config.compress_archives {
            format!(
                "{}_{:016x}_{:016x}_{}.wal.gz",
                self.config.archive_prefix, lsn_start, lsn_end, timestamp
            )
        } else {
            format!(
                "{}_{:016x}_{:016x}_{}.wal",
                self.config.archive_prefix, lsn_start, lsn_end, timestamp
            )
        };

        let archive_path = self.config.archive_dir.join(&archive_name);

        // Read WAL file
        let mut wal_data = Vec::new();
        File::open(wal_file_path)
            .and_then(|mut f| f.read_to_end(&mut wal_data))
            .map_err(|e| TdbError::Other(format!("Failed to read WAL file: {}", e)))?;

        // Calculate checksum
        let checksum = calculate_crc32(&wal_data);

        // Write archive (compress if enabled)
        let archive_data = if self.config.compress_archives {
            compress_data(&wal_data)?
        } else {
            wal_data.clone()
        };

        File::create(&archive_path)
            .and_then(|mut f| f.write_all(&archive_data))
            .map_err(|e| TdbError::Other(format!("Failed to write archive: {}", e)))?;

        // Verify if enabled
        if self.config.verify_archives {
            self.verify_archive(&archive_path, checksum)?;
        }

        // Store metadata
        let metadata = WalArchiveMetadata {
            file_name: archive_name.clone(),
            original_path: wal_file_path.to_path_buf(),
            archived_at: SystemTime::now(),
            file_size: archive_data.len() as u64,
            compressed: self.config.compress_archives,
            checksum,
            lsn_range: (lsn_start, lsn_end),
        };

        self.metadata.write().insert(archive_name.clone(), metadata);

        // Update statistics
        self.total_archives.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_archived
            .fetch_add(archive_data.len() as u64, Ordering::Relaxed);
        self.stats
            .successful_archives
            .fetch_add(1, Ordering::Relaxed);

        Ok(archive_name)
    }

    /// Verify an archive file
    fn verify_archive(&self, archive_path: &Path, expected_checksum: u32) -> Result<()> {
        let mut archive_data = Vec::new();
        File::open(archive_path)
            .and_then(|mut f| f.read_to_end(&mut archive_data))
            .map_err(|e| {
                TdbError::Other(format!("Failed to read archive for verification: {}", e))
            })?;

        // Decompress if needed
        let data = if self.config.compress_archives {
            decompress_data(&archive_data)?
        } else {
            archive_data
        };

        // Verify checksum
        let actual_checksum = calculate_crc32(&data);
        if actual_checksum != expected_checksum {
            return Err(TdbError::Other(format!(
                "Archive verification failed: checksum mismatch (expected {}, got {})",
                expected_checksum, actual_checksum
            )));
        }

        Ok(())
    }

    /// Restore a WAL file from archive
    pub fn restore_wal_file(&self, archive_name: &str, output_path: &Path) -> Result<()> {
        let metadata = self
            .metadata
            .read()
            .get(archive_name)
            .cloned()
            .ok_or_else(|| TdbError::Other(format!("Archive {} not found", archive_name)))?;

        let archive_path = self.config.archive_dir.join(&metadata.file_name);

        // Read archive
        let mut archive_data = Vec::new();
        File::open(&archive_path)
            .and_then(|mut f| f.read_to_end(&mut archive_data))
            .map_err(|e| TdbError::Other(format!("Failed to read archive: {}", e)))?;

        // Decompress if needed
        let wal_data = if metadata.compressed {
            decompress_data(&archive_data)?
        } else {
            archive_data
        };

        // Verify checksum
        let checksum = calculate_crc32(&wal_data);
        if checksum != metadata.checksum {
            return Err(TdbError::Other(
                "Archive restoration failed: checksum mismatch".to_string(),
            ));
        }

        // Write WAL file
        File::create(output_path)
            .and_then(|mut f| f.write_all(&wal_data))
            .map_err(|e| TdbError::Other(format!("Failed to write WAL file: {}", e)))?;

        self.stats
            .successful_restores
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// List all archived WAL files
    pub fn list_archives(&self) -> Vec<WalArchiveMetadata> {
        self.metadata.read().values().cloned().collect()
    }

    /// Get archive metadata
    pub fn get_metadata(&self, archive_name: &str) -> Option<WalArchiveMetadata> {
        self.metadata.read().get(archive_name).cloned()
    }

    /// Find archives covering a specific LSN range
    pub fn find_archives_for_lsn_range(
        &self,
        start_lsn: u64,
        end_lsn: u64,
    ) -> Vec<WalArchiveMetadata> {
        self.metadata
            .read()
            .values()
            .filter(|metadata| {
                // Check if archive's LSN range overlaps with requested range
                metadata.lsn_range.0 <= end_lsn && metadata.lsn_range.1 >= start_lsn
            })
            .cloned()
            .collect()
    }

    /// Cleanup old archives based on retention policy
    pub fn cleanup_old_archives(&self) -> Result<usize> {
        if self.config.retention_period.is_zero() {
            return Ok(0);
        }

        let cutoff_time = SystemTime::now()
            .checked_sub(self.config.retention_period)
            .ok_or_else(|| TdbError::Other("Invalid retention period".to_string()))?;

        let mut deleted_count = 0;
        let mut to_delete = Vec::new();

        // Find archives older than retention period
        for (name, metadata) in self.metadata.read().iter() {
            if metadata.archived_at < cutoff_time {
                to_delete.push((name.clone(), metadata.file_name.clone()));
            }
        }

        // Delete old archives
        for (name, file_name) in to_delete {
            let archive_path = self.config.archive_dir.join(&file_name);
            if let Err(e) = fs::remove_file(&archive_path) {
                eprintln!("Failed to delete archive {}: {}", file_name, e);
                continue;
            }

            self.metadata.write().remove(&name);
            deleted_count += 1;
            self.total_deleted.fetch_add(1, Ordering::Relaxed);
        }

        Ok(deleted_count)
    }

    /// Get total archive size
    pub fn total_archive_size(&self) -> u64 {
        self.metadata.read().values().map(|m| m.file_size).sum()
    }

    /// Check if archive size limit is exceeded
    pub fn is_size_limit_exceeded(&self) -> bool {
        if self.config.max_archive_size == 0 {
            return false;
        }

        self.total_archive_size() > self.config.max_archive_size
    }

    /// Get archiver statistics
    pub fn stats(&self) -> WalArchiverStatsSnapshot {
        WalArchiverStatsSnapshot {
            total_archives: self.total_archives.load(Ordering::Relaxed),
            total_bytes_archived: self.total_bytes_archived.load(Ordering::Relaxed),
            total_deleted: self.total_deleted.load(Ordering::Relaxed),
            successful_archives: self.stats.successful_archives.load(Ordering::Relaxed),
            failed_archives: self.stats.failed_archives.load(Ordering::Relaxed),
            successful_restores: self.stats.successful_restores.load(Ordering::Relaxed),
            failed_restores: self.stats.failed_restores.load(Ordering::Relaxed),
            current_archive_size: self.total_archive_size(),
        }
    }

    /// Stop the archiver
    pub fn stop(&self) {
        self.active.store(false, Ordering::Release);
    }

    /// Check if archiver is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }
}

/// WAL archiver statistics
#[derive(Debug, Default)]
struct WalArchiverStats {
    /// Successful archive operations
    successful_archives: AtomicU64,
    /// Failed archive operations
    failed_archives: AtomicU64,
    /// Successful restore operations
    successful_restores: AtomicU64,
    /// Failed restore operations
    failed_restores: AtomicU64,
}

/// Snapshot of WAL archiver statistics
#[derive(Debug, Clone)]
pub struct WalArchiverStatsSnapshot {
    /// Total archives created
    pub total_archives: u64,
    /// Total bytes archived
    pub total_bytes_archived: u64,
    /// Total archives deleted
    pub total_deleted: u64,
    /// Successful archive operations
    pub successful_archives: u64,
    /// Failed archive operations
    pub failed_archives: u64,
    /// Successful restore operations
    pub successful_restores: u64,
    /// Failed restore operations
    pub failed_restores: u64,
    /// Current total archive size
    pub current_archive_size: u64,
}

impl WalArchiverStatsSnapshot {
    /// Calculate archive success rate
    pub fn archive_success_rate(&self) -> f64 {
        let total = self.successful_archives + self.failed_archives;
        if total == 0 {
            0.0
        } else {
            (self.successful_archives as f64 / total as f64) * 100.0
        }
    }

    /// Calculate restore success rate
    pub fn restore_success_rate(&self) -> f64 {
        let total = self.successful_restores + self.failed_restores;
        if total == 0 {
            0.0
        } else {
            (self.successful_restores as f64 / total as f64) * 100.0
        }
    }
}

/// Simple CRC32 checksum calculation
fn calculate_crc32(data: &[u8]) -> u32 {
    const CRC32_POLY: u32 = 0xEDB88320;
    let mut crc = 0xFFFFFFFF_u32;

    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32_POLY;
            } else {
                crc >>= 1;
            }
        }
    }

    !crc
}

/// Simple compression (in real implementation, use a compression library)
fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
    // Placeholder: In production, use flate2, zstd, or lz4
    // For now, just return the data as-is
    Ok(data.to_vec())
}

/// Simple decompression
fn decompress_data(data: &[u8]) -> Result<Vec<u8>> {
    // Placeholder: In production, use flate2, zstd, or lz4
    // For now, just return the data as-is
    Ok(data.to_vec())
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use std::env;

    fn temp_archive_dir(name: &str) -> PathBuf {
        let mut path = env::temp_dir();
        path.push(format!("oxirs_tdb_wal_archive_test_{}", name));
        path
    }

    fn temp_wal_file(name: &str, content: &[u8]) -> PathBuf {
        let mut path = env::temp_dir();
        path.push(format!("oxirs_tdb_wal_test_{}.wal", name));

        let mut file = File::create(&path).unwrap();
        file.write_all(content).unwrap();

        path
    }

    #[test]
    fn test_wal_archive_config_default() {
        let config = WalArchiveConfig::default();
        assert!(!config.enable_archiving);
        assert!(config.compress_archives);
        assert_eq!(config.archive_prefix, "wal");
    }

    #[test]
    fn test_wal_archiver_creation() {
        let archive_dir = temp_archive_dir("creation");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = true;

        let archiver = WalArchiver::new(config).unwrap();
        assert!(archiver.is_active());
        assert!(archive_dir.exists());

        fs::remove_dir_all(archive_dir).ok();
    }

    #[test]
    fn test_archive_wal_file() {
        let archive_dir = temp_archive_dir("archive");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = true;
        config.verify_archives = true;

        let archiver = WalArchiver::new(config).unwrap();

        // Create a test WAL file
        let wal_content = b"Test WAL data for archiving";
        let wal_path = temp_wal_file("archive_test", wal_content);

        // Archive the WAL file
        let archive_name = archiver.archive_wal_file(&wal_path, 1000, 2000).unwrap();

        assert!(!archive_name.is_empty());

        let stats = archiver.stats();
        assert_eq!(stats.total_archives, 1);
        assert_eq!(stats.successful_archives, 1);

        // Cleanup
        fs::remove_file(wal_path).ok();
        fs::remove_dir_all(archive_dir).ok();
    }

    #[test]
    fn test_restore_wal_file() {
        let archive_dir = temp_archive_dir("restore");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = true;

        let archiver = WalArchiver::new(config).unwrap();

        // Create and archive a WAL file
        let wal_content = b"Test WAL data for restoration";
        let wal_path = temp_wal_file("restore_test", wal_content);

        let archive_name = archiver.archive_wal_file(&wal_path, 1000, 2000).unwrap();

        // Restore to a new location
        let restore_path = temp_wal_file("restore_output", &[]);
        archiver
            .restore_wal_file(&archive_name, &restore_path)
            .unwrap();

        // Verify restored content
        let mut restored_content = Vec::new();
        File::open(&restore_path)
            .unwrap()
            .read_to_end(&mut restored_content)
            .unwrap();

        assert_eq!(&restored_content, wal_content);

        let stats = archiver.stats();
        assert_eq!(stats.successful_restores, 1);

        // Cleanup
        fs::remove_file(wal_path).ok();
        fs::remove_file(restore_path).ok();
        fs::remove_dir_all(archive_dir).ok();
    }

    #[test]
    fn test_list_archives() {
        let archive_dir = temp_archive_dir("list");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = true;

        let archiver = WalArchiver::new(config).unwrap();

        // Create multiple archives
        for i in 0..3 {
            let wal_content = format!("WAL data {}", i);
            let wal_path = temp_wal_file(&format!("list_{}", i), wal_content.as_bytes());

            archiver
                .archive_wal_file(&wal_path, i * 1000, (i + 1) * 1000)
                .unwrap();

            fs::remove_file(wal_path).ok();
        }

        let archives = archiver.list_archives();
        assert_eq!(archives.len(), 3);

        // Cleanup
        fs::remove_dir_all(archive_dir).ok();
    }

    #[test]
    fn test_find_archives_for_lsn_range() {
        let archive_dir = temp_archive_dir("lsn_range");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = true;

        let archiver = WalArchiver::new(config).unwrap();

        // Create archives with different LSN ranges
        let wal_path1 = temp_wal_file("lsn1", b"WAL 1");
        let wal_path2 = temp_wal_file("lsn2", b"WAL 2");
        let wal_path3 = temp_wal_file("lsn3", b"WAL 3");

        archiver.archive_wal_file(&wal_path1, 1000, 2000).unwrap();
        archiver.archive_wal_file(&wal_path2, 2000, 3000).unwrap();
        archiver.archive_wal_file(&wal_path3, 3000, 4000).unwrap();

        // Find archives for LSN range [1500, 2500]
        let archives = archiver.find_archives_for_lsn_range(1500, 2500);
        assert_eq!(archives.len(), 2); // Should find archives 1 and 2

        // Cleanup
        fs::remove_file(wal_path1).ok();
        fs::remove_file(wal_path2).ok();
        fs::remove_file(wal_path3).ok();
        fs::remove_dir_all(archive_dir).ok();
    }

    #[test]
    fn test_cleanup_old_archives() {
        let archive_dir = temp_archive_dir("cleanup");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = true;
        config.retention_period = Duration::from_secs(1); // Very short retention for testing

        let archiver = WalArchiver::new(config).unwrap();

        // Create an archive
        let wal_path = temp_wal_file("cleanup_test", b"Old WAL data");
        archiver.archive_wal_file(&wal_path, 1000, 2000).unwrap();

        assert_eq!(archiver.list_archives().len(), 1);

        // Wait for retention period to pass
        std::thread::sleep(Duration::from_secs(2));

        // Cleanup old archives
        let deleted = archiver.cleanup_old_archives().unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(archiver.list_archives().len(), 0);

        // Cleanup
        fs::remove_file(wal_path).ok();
        fs::remove_dir_all(archive_dir).ok();
    }

    #[test]
    fn test_total_archive_size() {
        let archive_dir = temp_archive_dir("size");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = true;

        let archiver = WalArchiver::new(config).unwrap();

        // Create archives
        let wal_path = temp_wal_file("size_test", b"Test data for size calculation");
        archiver.archive_wal_file(&wal_path, 1000, 2000).unwrap();

        let total_size = archiver.total_archive_size();
        assert!(total_size > 0);

        // Cleanup
        fs::remove_file(wal_path).ok();
        fs::remove_dir_all(archive_dir).ok();
    }

    #[test]
    fn test_archiver_stop() {
        let archive_dir = temp_archive_dir("stop");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = true;

        let archiver = WalArchiver::new(config).unwrap();

        assert!(archiver.is_active());

        archiver.stop();
        assert!(!archiver.is_active());

        // Archiving should fail after stop
        let wal_path = temp_wal_file("stop_test", b"Test");
        let result = archiver.archive_wal_file(&wal_path, 1000, 2000);
        assert!(result.is_err());

        // Cleanup
        fs::remove_file(wal_path).ok();
        fs::remove_dir_all(archive_dir).ok();
    }

    #[test]
    fn test_crc32_calculation() {
        let data = b"Hello, CRC32!";
        let checksum = calculate_crc32(data);

        // CRC32 should be deterministic
        assert_eq!(checksum, calculate_crc32(data));

        // Different data should have different checksum
        let checksum2 = calculate_crc32(b"Different data");
        assert_ne!(checksum, checksum2);
    }

    #[test]
    fn test_stats_calculations() {
        let stats = WalArchiverStatsSnapshot {
            total_archives: 100,
            total_bytes_archived: 1_000_000,
            total_deleted: 20,
            successful_archives: 95,
            failed_archives: 5,
            successful_restores: 80,
            failed_restores: 20,
            current_archive_size: 500_000,
        };

        assert_eq!(stats.archive_success_rate(), 95.0);
        assert_eq!(stats.restore_success_rate(), 80.0);
    }

    #[test]
    fn test_disabled_archiving() {
        let archive_dir = temp_archive_dir("disabled");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = false; // Disabled

        let archiver = WalArchiver::new(config).unwrap();

        let wal_path = temp_wal_file("disabled_test", b"Test");
        let result = archiver.archive_wal_file(&wal_path, 1000, 2000);

        assert!(result.is_err());

        // Cleanup
        fs::remove_file(wal_path).ok();
        fs::remove_dir_all(archive_dir).ok();
    }

    #[test]
    fn test_get_metadata() {
        let archive_dir = temp_archive_dir("metadata");
        let mut config = WalArchiveConfig::default();
        config.archive_dir = archive_dir.clone();
        config.enable_archiving = true;

        let archiver = WalArchiver::new(config).unwrap();

        let wal_path = temp_wal_file("metadata_test", b"Metadata test");
        let archive_name = archiver.archive_wal_file(&wal_path, 5000, 6000).unwrap();

        let metadata = archiver.get_metadata(&archive_name);
        assert!(metadata.is_some());

        let meta = metadata.unwrap();
        assert_eq!(meta.lsn_range, (5000, 6000));
        assert_eq!(meta.file_name, archive_name);

        // Cleanup
        fs::remove_file(wal_path).ok();
        fs::remove_dir_all(archive_dir).ok();
    }
}
