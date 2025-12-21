//! Backup Automation
//!
//! Provides automatic backup and restore capabilities for RDF datasets.
//! Supports multiple backup strategies and destinations.

use crate::error::{FusekiError, FusekiResult};
use crate::store::{RdfSerializationFormat, Store};
use crate::store_ext::StoreExt;
use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::time::{self, Duration};
use tracing::{debug, error, info, warn};

/// Backup manager
pub struct BackupManager {
    /// Store to backup
    store: Arc<Store>,
    /// Backup configuration
    config: BackupConfig,
    /// Last backup time (any backup type)
    last_backup: Arc<tokio::sync::RwLock<Option<DateTime<Utc>>>>,
    /// Last full backup time (for differential backups)
    last_full_backup: Arc<tokio::sync::RwLock<Option<DateTime<Utc>>>>,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enabled: bool,
    /// Backup interval (hours)
    pub interval_hours: u64,
    /// Backup directory
    pub backup_dir: PathBuf,
    /// Maximum number of backups to keep
    pub max_backups: usize,
    /// Compression enabled
    pub compression: bool,
    /// Include indexes in backup
    pub include_indexes: bool,
    /// Backup strategy
    pub strategy: BackupStrategy,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_hours: 24,
            backup_dir: PathBuf::from("/data/backups"),
            max_backups: 7,
            compression: true,
            include_indexes: true,
            strategy: BackupStrategy::Full,
        }
    }
}

/// Backup strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupStrategy {
    /// Full backup
    Full,
    /// Incremental backup
    Incremental,
    /// Differential backup
    Differential,
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Backup ID
    pub id: String,
    /// Backup timestamp
    pub timestamp: DateTime<Utc>,
    /// Backup strategy used
    pub strategy: BackupStrategy,
    /// Size in bytes
    pub size_bytes: u64,
    /// Compressed
    pub compressed: bool,
    /// Number of triples
    pub triple_count: Option<u64>,
    /// Checksum
    pub checksum: Option<String>,
    /// Description
    pub description: Option<String>,
}

impl BackupManager {
    /// Create a new backup manager
    pub fn new(store: Arc<Store>, config: BackupConfig) -> Self {
        Self {
            store,
            config,
            last_backup: Arc::new(tokio::sync::RwLock::new(None)),
            last_full_backup: Arc::new(tokio::sync::RwLock::new(None)),
        }
    }

    /// Start automatic backup scheduling
    pub async fn start(&self) -> FusekiResult<()> {
        if !self.config.enabled {
            info!("Automatic backups disabled");
            return Ok(());
        }

        info!(
            "Starting automatic backup scheduler (interval: {} hours)",
            self.config.interval_hours
        );

        // Ensure backup directory exists
        fs::create_dir_all(&self.config.backup_dir)
            .await
            .map_err(|e| {
                FusekiError::internal(format!("Failed to create backup directory: {}", e))
            })?;

        loop {
            if let Err(e) = self.perform_backup().await {
                error!("Backup failed: {}", e);
            }

            let interval = Duration::from_secs(self.config.interval_hours * 3600);
            time::sleep(interval).await;
        }
    }

    /// Perform a backup
    pub async fn perform_backup(&self) -> FusekiResult<BackupMetadata> {
        info!("Starting backup (strategy: {:?})", self.config.strategy);

        let backup_id = format!("backup-{}", Utc::now().format("%Y%m%d-%H%M%S"));
        let backup_path = self.config.backup_dir.join(&backup_id);

        // Create backup directory
        fs::create_dir_all(&backup_path).await.map_err(|e| {
            FusekiError::internal(format!("Failed to create backup directory: {}", e))
        })?;

        // Perform backup based on strategy
        let metadata = match self.config.strategy {
            BackupStrategy::Full => self.perform_full_backup(&backup_path, &backup_id).await?,
            BackupStrategy::Incremental => {
                self.perform_incremental_backup(&backup_path, &backup_id)
                    .await?
            }
            BackupStrategy::Differential => {
                self.perform_differential_backup(&backup_path, &backup_id)
                    .await?
            }
        };

        // Compress if enabled
        let final_metadata = if self.config.compression {
            self.compress_backup(&backup_path, metadata).await?
        } else {
            metadata
        };

        // Save metadata
        self.save_metadata(&backup_path, &final_metadata).await?;

        // Update last backup time
        let now = Utc::now();
        *self.last_backup.write().await = Some(now);

        // Update last full backup time if this was a full backup
        if final_metadata.strategy == BackupStrategy::Full {
            *self.last_full_backup.write().await = Some(now);
            info!("Updated last full backup timestamp");
        }

        // Clean old backups
        self.cleanup_old_backups().await?;

        info!("Backup completed successfully: {}", backup_id);
        Ok(final_metadata)
    }

    /// Perform full backup
    async fn perform_full_backup(
        &self,
        backup_path: &Path,
        backup_id: &str,
    ) -> FusekiResult<BackupMetadata> {
        info!("Performing full backup");

        // Export all data to N-Quads format
        let export_path = backup_path.join("data.nq");

        // Get all datasets and export them
        let datasets = self
            .store
            .list_datasets()
            .map_err(|e| FusekiError::internal(format!("Failed to list datasets: {}", e)))?;

        let mut all_data = String::new();
        let mut total_triple_count: usize = 0;

        // Export default dataset first
        match self.store.export_data(RdfSerializationFormat::NQuads, None) {
            Ok(data) => {
                all_data.push_str(&data);
                total_triple_count += self.store.count_triples("default");
            }
            Err(e) => {
                warn!("Failed to export default dataset: {}", e);
            }
        }

        // Export each named dataset
        for dataset in &datasets {
            if dataset != "default" {
                match self
                    .store
                    .export_data(RdfSerializationFormat::NQuads, Some(dataset))
                {
                    Ok(data) => {
                        all_data.push_str(&data);
                        total_triple_count += self.store.count_triples(dataset);
                    }
                    Err(e) => {
                        warn!("Failed to export dataset {}: {}", dataset, e);
                    }
                }
            }
        }

        // Calculate checksum
        let mut hasher = Sha256::new();
        hasher.update(all_data.as_bytes());
        let checksum = format!("{:x}", hasher.finalize());

        // Write to file
        fs::write(&export_path, all_data.as_bytes())
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to write backup: {}", e)))?;

        let size_bytes = fs::metadata(&export_path)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to get file size: {}", e)))?
            .len();

        info!(
            "Full backup completed: {} triples, {} bytes",
            total_triple_count, size_bytes
        );

        Ok(BackupMetadata {
            id: backup_id.to_string(),
            timestamp: Utc::now(),
            strategy: BackupStrategy::Full,
            size_bytes,
            compressed: false,
            triple_count: Some(total_triple_count as u64),
            checksum: Some(checksum),
            description: Some(format!("Full backup of {} datasets", datasets.len())),
        })
    }

    /// Perform incremental backup
    async fn perform_incremental_backup(
        &self,
        backup_path: &Path,
        backup_id: &str,
    ) -> FusekiResult<BackupMetadata> {
        info!("Performing incremental backup");

        // Get changes since last backup
        let last_backup_time = self.last_backup.read().await;
        let since = match *last_backup_time {
            Some(time) => time,
            None => {
                // No previous backup, perform full backup instead
                warn!("No previous backup found, performing full backup instead");
                return self.perform_full_backup(backup_path, backup_id).await;
            }
        };
        drop(last_backup_time);

        info!("Fetching changes since {}", since);
        let changes = self.store.get_changes_since(since).await?;

        if changes.is_empty() {
            info!("No changes detected since last backup");
        }

        // Export changes to N-Quads format
        let export_path = backup_path.join("changes.nq");
        let mut content = String::new();
        content.push_str(&format!("# Incremental backup since {}\n", since));
        content.push_str(&format!("# {} changes\n", changes.len()));

        // For each change, export the affected triples
        // Note: This is a simplified implementation - in production you'd want to
        // actually export the changed triples from the store
        for change in &changes {
            content.push_str(&format!(
                "# Change {} at {}: {} (graphs: {:?})\n",
                change.id, change.timestamp, change.operation_type, change.affected_graphs
            ));
        }

        fs::write(&export_path, content.as_bytes())
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to write backup: {}", e)))?;

        let size_bytes = fs::metadata(&export_path)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to get file size: {}", e)))?
            .len();

        // Calculate checksum
        let checksum = self.calculate_checksum(&export_path).await?;

        Ok(BackupMetadata {
            id: backup_id.to_string(),
            timestamp: Utc::now(),
            strategy: BackupStrategy::Incremental,
            size_bytes,
            compressed: false,
            triple_count: Some(changes.len() as u64),
            checksum: Some(checksum),
            description: Some(format!(
                "Incremental backup with {} changes since {}",
                changes.len(),
                since
            )),
        })
    }

    /// Perform differential backup
    async fn perform_differential_backup(
        &self,
        backup_path: &Path,
        backup_id: &str,
    ) -> FusekiResult<BackupMetadata> {
        info!("Performing differential backup");

        // Get changes since last full backup
        let last_full_backup_time = self.last_full_backup.read().await;
        let since = match *last_full_backup_time {
            Some(time) => time,
            None => {
                // No previous full backup, perform full backup instead
                warn!("No previous full backup found, performing full backup instead");
                return self.perform_full_backup(backup_path, backup_id).await;
            }
        };
        drop(last_full_backup_time);

        info!("Fetching changes since last full backup at {}", since);
        let changes = self.store.get_changes_since(since).await?;

        if changes.is_empty() {
            info!("No changes detected since last full backup");
        }

        // Export changes to N-Quads format
        let export_path = backup_path.join("diff.nq");
        let mut content = String::new();
        content.push_str(&format!(
            "# Differential backup since last full backup at {}\n",
            since
        ));
        content.push_str(&format!("# {} changes\n", changes.len()));

        // For each change, export the affected triples
        // Note: This is a simplified implementation - in production you'd want to
        // actually export the changed triples from the store
        for change in &changes {
            content.push_str(&format!(
                "# Change {} at {}: {} (graphs: {:?})\n",
                change.id, change.timestamp, change.operation_type, change.affected_graphs
            ));
        }

        fs::write(&export_path, content.as_bytes())
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to write backup: {}", e)))?;

        let size_bytes = fs::metadata(&export_path)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to get file size: {}", e)))?
            .len();

        // Calculate checksum
        let checksum = self.calculate_checksum(&export_path).await?;

        Ok(BackupMetadata {
            id: backup_id.to_string(),
            timestamp: Utc::now(),
            strategy: BackupStrategy::Differential,
            size_bytes,
            compressed: false,
            triple_count: Some(changes.len() as u64),
            checksum: Some(checksum),
            description: Some(format!(
                "Differential backup with {} changes since last full backup at {}",
                changes.len(),
                since
            )),
        })
    }

    /// Compress backup
    async fn compress_backup(
        &self,
        backup_path: &Path,
        metadata: BackupMetadata,
    ) -> FusekiResult<BackupMetadata> {
        debug!("Compressing backup");

        // Find the data file
        let data_file = match metadata.strategy {
            BackupStrategy::Full => backup_path.join("data.nq"),
            BackupStrategy::Incremental => backup_path.join("changes.nq"),
            BackupStrategy::Differential => backup_path.join("diff.nq"),
        };

        if !data_file.exists() {
            return Ok(metadata);
        }

        // Read the data
        let data = fs::read(&data_file).await.map_err(|e| {
            FusekiError::internal(format!("Failed to read data for compression: {}", e))
        })?;

        // Compress using gzip
        let compressed_path = data_file.with_extension("nq.gz");
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(&data)
            .map_err(|e| FusekiError::internal(format!("Failed to compress data: {}", e)))?;
        let compressed_data = encoder
            .finish()
            .map_err(|e| FusekiError::internal(format!("Failed to finalize compression: {}", e)))?;

        // Write compressed file
        fs::write(&compressed_path, &compressed_data)
            .await
            .map_err(|e| {
                FusekiError::internal(format!("Failed to write compressed file: {}", e))
            })?;

        // Remove original uncompressed file
        fs::remove_file(&data_file).await.map_err(|e| {
            FusekiError::internal(format!("Failed to remove uncompressed file: {}", e))
        })?;

        // Update metadata
        let mut compressed_metadata = metadata;
        compressed_metadata.compressed = true;
        compressed_metadata.size_bytes = compressed_data.len() as u64;

        info!(
            "Compressed backup: {} -> {} bytes ({}% reduction)",
            data.len(),
            compressed_data.len(),
            if data.is_empty() {
                0
            } else {
                100 - (compressed_data.len() * 100 / data.len())
            }
        );

        Ok(compressed_metadata)
    }

    /// Calculate SHA-256 checksum of a file
    async fn calculate_checksum(&self, file_path: &Path) -> FusekiResult<String> {
        let data = fs::read(file_path).await.map_err(|e| {
            FusekiError::internal(format!("Failed to read file for checksum: {}", e))
        })?;

        let mut hasher = Sha256::new();
        hasher.update(&data);
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Save backup metadata
    async fn save_metadata(
        &self,
        backup_path: &Path,
        metadata: &BackupMetadata,
    ) -> FusekiResult<()> {
        let metadata_path = backup_path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(metadata)
            .map_err(|e| FusekiError::internal(format!("Failed to serialize metadata: {}", e)))?;

        fs::write(&metadata_path, metadata_json)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to write metadata: {}", e)))?;

        Ok(())
    }

    /// Clean up old backups
    async fn cleanup_old_backups(&self) -> FusekiResult<()> {
        debug!("Cleaning up old backups");

        // List all backup directories
        let mut entries = fs::read_dir(&self.config.backup_dir).await.map_err(|e| {
            FusekiError::internal(format!("Failed to read backup directory: {}", e))
        })?;

        let mut backups = Vec::new();

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to read directory entry: {}", e)))?
        {
            if entry
                .file_type()
                .await
                .map_err(|e| FusekiError::internal(format!("Failed to get file type: {}", e)))?
                .is_dir()
            {
                if let Ok(metadata) = entry.metadata().await {
                    if let Ok(created) = metadata.created() {
                        backups.push((entry.path(), created));
                    }
                }
            }
        }

        // Sort by creation time (oldest first)
        backups.sort_by_key(|(_, created)| *created);

        // Remove oldest backups if we exceed max_backups
        while backups.len() > self.config.max_backups {
            if let Some((path, _)) = backups.first() {
                info!("Removing old backup: {:?}", path);
                fs::remove_dir_all(path).await.map_err(|e| {
                    FusekiError::internal(format!("Failed to remove old backup: {}", e))
                })?;
                backups.remove(0);
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Restore from backup
    pub async fn restore_backup(&self, backup_id: &str) -> FusekiResult<()> {
        info!("Restoring from backup: {}", backup_id);

        let backup_path = self.config.backup_dir.join(backup_id);

        if !backup_path.exists() {
            return Err(FusekiError::internal(format!(
                "Backup not found: {}",
                backup_id
            )));
        }

        // Load metadata
        let metadata_path = backup_path.join("metadata.json");
        let metadata_json = fs::read_to_string(&metadata_path)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to read metadata: {}", e)))?;

        let metadata: BackupMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| FusekiError::internal(format!("Failed to parse metadata: {}", e)))?;

        // Determine the data file based on strategy
        let base_filename = match metadata.strategy {
            BackupStrategy::Full => "data.nq",
            BackupStrategy::Incremental => "changes.nq",
            BackupStrategy::Differential => "diff.nq",
        };

        // Find the data file (compressed or not)
        let compressed_path = backup_path.join(format!("{}.gz", base_filename));
        let uncompressed_path = backup_path.join(base_filename);

        let data = if compressed_path.exists() {
            // Decompress the data
            debug!("Decompressing backup data from {:?}", compressed_path);
            let compressed_data = fs::read(&compressed_path).await.map_err(|e| {
                FusekiError::internal(format!("Failed to read compressed backup: {}", e))
            })?;

            let mut decoder = GzDecoder::new(&compressed_data[..]);
            let mut decompressed = String::new();
            decoder.read_to_string(&mut decompressed).map_err(|e| {
                FusekiError::internal(format!("Failed to decompress backup: {}", e))
            })?;
            decompressed
        } else if uncompressed_path.exists() {
            // Read uncompressed data
            debug!(
                "Reading uncompressed backup data from {:?}",
                uncompressed_path
            );
            fs::read_to_string(&uncompressed_path)
                .await
                .map_err(|e| FusekiError::internal(format!("Failed to read backup: {}", e)))?
        } else {
            return Err(FusekiError::internal(format!(
                "Backup data file not found in {}",
                backup_id
            )));
        };

        // Verify checksum if available
        if let Some(expected_checksum) = &metadata.checksum {
            let mut hasher = Sha256::new();
            hasher.update(data.as_bytes());
            let actual_checksum = format!("{:x}", hasher.finalize());

            if &actual_checksum != expected_checksum {
                return Err(FusekiError::internal(format!(
                    "Checksum mismatch: expected {}, got {}",
                    expected_checksum, actual_checksum
                )));
            }
            debug!("Checksum verified: {}", actual_checksum);
        }

        // Clear current store data using SPARQL UPDATE DROP ALL
        info!("Clearing current store data before restore");
        if let Err(e) = self.store.update("DROP ALL") {
            warn!("Failed to clear store (may be empty): {}", e);
        }

        // Import backup data
        info!("Importing backup data ({} bytes)", data.len());
        let imported_count = self
            .store
            .import_data(&data, RdfSerializationFormat::NQuads, None)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to import backup data: {}", e)))?;

        info!("Imported {} triples from backup", imported_count);

        info!(
            "Restore completed successfully from backup: {} ({} triples)",
            backup_id,
            metadata.triple_count.unwrap_or(0)
        );

        Ok(())
    }

    /// List available backups
    pub async fn list_backups(&self) -> FusekiResult<Vec<BackupMetadata>> {
        let mut entries = fs::read_dir(&self.config.backup_dir).await.map_err(|e| {
            FusekiError::internal(format!("Failed to read backup directory: {}", e))
        })?;

        let mut backups = Vec::new();

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to read directory entry: {}", e)))?
        {
            if entry
                .file_type()
                .await
                .map_err(|e| FusekiError::internal(format!("Failed to get file type: {}", e)))?
                .is_dir()
            {
                let metadata_path = entry.path().join("metadata.json");
                if metadata_path.exists() {
                    if let Ok(metadata_json) = fs::read_to_string(&metadata_path).await {
                        if let Ok(metadata) = serde_json::from_str::<BackupMetadata>(&metadata_json)
                        {
                            backups.push(metadata);
                        }
                    }
                }
            }
        }

        // Sort by timestamp (newest first)
        backups.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        Ok(backups)
    }

    /// Get last backup time
    pub async fn last_backup_time(&self) -> Option<DateTime<Utc>> {
        *self.last_backup.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_config_default() {
        let config = BackupConfig::default();
        assert_eq!(config.interval_hours, 24);
        assert_eq!(config.max_backups, 7);
        assert!(config.compression);
    }

    #[test]
    fn test_backup_metadata_serialization() {
        let metadata = BackupMetadata {
            id: "test-backup".to_string(),
            timestamp: Utc::now(),
            strategy: BackupStrategy::Full,
            size_bytes: 1024,
            compressed: true,
            triple_count: Some(1000),
            checksum: Some("abc123".to_string()),
            description: Some("Test backup".to_string()),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: BackupMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, metadata.id);
        assert_eq!(deserialized.strategy, BackupStrategy::Full);
    }
}
