//! # Backup and Restore
//!
//! Provides comprehensive backup and restore capabilities for distributed RDF storage.
//!
//! ## Overview
//!
//! This module supports:
//! - Full backups of entire cluster state
//! - Incremental backups for efficiency
//! - Point-in-time recovery
//! - Snapshot-based backups
//! - Compressed backups
//! - Backup verification
//! - Automated backup scheduling
//!
//! ## Features
//!
//! - Multiple backup formats (JSON, Binary, Compressed)
//! - Backup encryption support
//! - Backup metadata tracking
//! - Parallel backup/restore operations
//! - Progress tracking and metrics
//! - SciRS2-optimized compression

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::at_rest::{self, AtRestKey};
use crate::error::{ClusterError, Result};
use crate::raft::OxirsNodeId;

/// Suffix of the metadata file kept next to each backup file.
const METADATA_SUFFIX: &str = ".meta.json";

/// Context bound into an encrypted backup file (see [`crate::at_rest`]): the
/// node, id, type and parent of the backup, so that a sealed file only opens
/// as the backup it was created as.
fn backup_context(
    node_id: OxirsNodeId,
    backup_id: &str,
    backup_type: BackupType,
    parent_backup_id: Option<&str>,
) -> Vec<u8> {
    let type_tag: &[u8] = match backup_type {
        BackupType::Full => b"full",
        BackupType::Incremental => b"incremental",
        BackupType::Snapshot => b"snapshot",
    };
    let mut context = b"oxirs-cluster/backup".to_vec();
    for field in [
        node_id.to_be_bytes().as_slice(),
        backup_id.as_bytes(),
        type_tag,
        parent_backup_id.unwrap_or_default().as_bytes(),
    ] {
        context.extend_from_slice(&(field.len() as u64).to_be_bytes());
        context.extend_from_slice(field);
    }
    context
}

/// The file name this manager gives backup `backup_id` of `backup_type`
/// created by `node_id`; `None` for types it never creates.
fn backup_file_name(
    backup_type: BackupType,
    node_id: OxirsNodeId,
    backup_id: &str,
) -> Option<String> {
    let kind = match backup_type {
        BackupType::Full => "full",
        BackupType::Incremental => "incr",
        BackupType::Snapshot => return None,
    };
    Some(format!("backup_{kind}_{node_id}_{backup_id}.dat"))
}

/// Replace `path` with `contents` atomically (temporary file, flush, rename),
/// so a crash never leaves a half-written file behind.
async fn write_file_atomically(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let mut temp_name = path.file_name().unwrap_or_default().to_os_string();
    temp_name.push(format!(".{}.tmp", uuid::Uuid::new_v4().simple()));
    let temp_path = path.with_file_name(temp_name);
    let written = async {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .await?;
        file.write_all(contents).await?;
        file.sync_all().await?;
        drop(file);
        fs::rename(&temp_path, path).await
    }
    .await;
    if written.is_err() {
        let _ = fs::remove_file(&temp_path).await;
    }
    written
}

/// Backup format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupFormat {
    /// JSON format (human-readable)
    Json,
    /// Binary format (compact)
    Binary,
    /// Compressed binary (smallest)
    CompressedBinary,
}

/// Backup type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupType {
    /// Full backup of all data
    Full,
    /// Incremental backup (changes since last backup)
    Incremental,
    /// Snapshot-based backup
    Snapshot,
}

/// Backup status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupStatus {
    /// Backup in progress
    InProgress,
    /// Backup completed successfully
    Completed,
    /// Backup failed
    Failed,
    /// Backup verified
    Verified,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Backup directory
    pub backup_dir: PathBuf,
    /// Backup format
    pub format: BackupFormat,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression level (1-9)
    pub compression_level: i32,
    /// Encrypt backup files at rest with XChaCha20-Poly1305 (see
    /// [`crate::at_rest`]). Requires `encryption_key_file`; without it the
    /// manager refuses to start instead of writing unencrypted backups.
    pub enable_encryption: bool,
    /// File holding the 256-bit backup key, as 32 raw bytes or 64 hex digits.
    /// Only valid together with `enable_encryption`.
    #[serde(default)]
    pub encryption_key_file: Option<PathBuf>,
    /// Verify backups (full and incremental) after creation
    pub verify_after_backup: bool,
    /// Maximum backup age (days). An older backup is still kept while a
    /// younger incremental backup that is kept is built on it.
    pub max_backup_age_days: u32,
    /// Maximum number of backups to keep, counting the newest first. The
    /// backups that a kept incremental backup is built on are kept as well,
    /// so more backups than this can remain on disk.
    pub max_backups: u32,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            backup_dir: PathBuf::from("./backups"),
            format: BackupFormat::CompressedBinary,
            enable_compression: true,
            compression_level: 6,
            enable_encryption: false,
            encryption_key_file: None,
            verify_after_backup: true,
            max_backup_age_days: 30,
            max_backups: 10,
        }
    }
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Backup ID
    pub id: String,
    /// Backup type
    pub backup_type: BackupType,
    /// Backup format
    pub format: BackupFormat,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Node ID that created the backup
    pub node_id: OxirsNodeId,
    /// Backup file path
    pub file_path: PathBuf,
    /// Backup size in bytes
    pub size_bytes: u64,
    /// Number of triples backed up
    pub triple_count: u64,
    /// Backup status
    pub status: BackupStatus,
    /// Checksum (for verification)
    pub checksum: Option<String>,
    /// Parent backup ID (for incremental backups)
    pub parent_backup_id: Option<String>,
    /// Completion timestamp
    pub completed_at: Option<SystemTime>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Whether the backup file is encrypted; restore insists on it matching.
    #[serde(default)]
    pub encrypted: bool,
    /// Whether the backup data was compressed. Restore follows this (and
    /// `format`) rather than the current configuration.
    #[serde(default)]
    pub compressed: bool,
}

/// Restore options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreOptions {
    /// Backup ID to restore
    pub backup_id: String,
    /// Skip verification
    pub skip_verification: bool,
    /// Overwrite existing data
    pub overwrite_existing: bool,
    /// Restore to specific point in time
    pub point_in_time: Option<SystemTime>,
}

/// Backup statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackupStatistics {
    /// Total backups created
    pub total_backups: u64,
    /// Total bytes backed up
    pub total_bytes_backed_up: u64,
    /// Total restores performed
    pub total_restores: u64,
    /// Failed backups
    pub failed_backups: u64,
    /// Failed restores
    pub failed_restores: u64,
    /// Last backup timestamp
    pub last_backup: Option<SystemTime>,
    /// Last restore timestamp
    pub last_restore: Option<SystemTime>,
    /// Average backup time (milliseconds)
    pub avg_backup_time_ms: f64,
    /// Average restore time (milliseconds)
    pub avg_restore_time_ms: f64,
}

/// Backup and restore manager
pub struct BackupRestoreManager {
    config: BackupConfig,
    /// Backup metadata registry
    backups: Arc<RwLock<HashMap<String, BackupMetadata>>>,
    /// Statistics
    stats: Arc<RwLock<BackupStatistics>>,
    /// Node ID
    node_id: OxirsNodeId,
    /// Key that seals backup files when encryption is enabled
    encryption_key: Option<AtRestKey>,
}

impl BackupRestoreManager {
    /// Create a new backup/restore manager
    ///
    /// # Errors
    ///
    /// Fails when the backup directory cannot be created, when
    /// `enable_encryption` is set without `encryption_key_file` (or the other
    /// way round), or when the key file cannot be loaded.
    pub async fn new(config: BackupConfig, node_id: OxirsNodeId) -> Result<Self> {
        let encryption_key = match (config.enable_encryption, &config.encryption_key_file) {
            (true, Some(path)) => Some(AtRestKey::from_file(path)?),
            (true, None) => {
                return Err(ClusterError::Encryption(
                    "enable_encryption is set but encryption_key_file is not; \
                     refusing to write unencrypted backups"
                        .to_string(),
                ))
            }
            (false, Some(_)) => {
                return Err(ClusterError::Encryption(
                    "encryption_key_file is set but enable_encryption is false; \
                     enable encryption or remove the key file setting"
                        .to_string(),
                ))
            }
            (false, None) => None,
        };

        // Create backup directory if it doesn't exist
        fs::create_dir_all(&config.backup_dir).await.map_err(|e| {
            ClusterError::Other(format!("Failed to create backup directory: {}", e))
        })?;

        let manager = Self {
            config,
            backups: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(BackupStatistics::default())),
            node_id,
            encryption_key,
        };

        // Load existing backup metadata
        manager.load_backup_metadata().await?;

        Ok(manager)
    }

    /// Create a full backup
    pub async fn create_full_backup(&self, data: Vec<(String, String, String)>) -> Result<String> {
        let backup_id = uuid::Uuid::new_v4().to_string();
        let timestamp = SystemTime::now();

        info!("Creating full backup: {}", backup_id);

        let file_path = self.backup_path(BackupType::Full, &backup_id)?;

        let start_time = std::time::Instant::now();

        // Serialize data based on format
        let serialized_data = self.serialize_data(&data)?;

        // Compress if enabled, then encrypt if enabled
        let compressed_data = if self.config.enable_compression {
            self.compress_data(&serialized_data)?
        } else {
            serialized_data
        };
        let context = backup_context(self.node_id, &backup_id, BackupType::Full, None);
        let final_data = self.seal_backup(&context, compressed_data)?;

        // Write to file
        write_file_atomically(&file_path, &final_data)
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to write backup: {}", e)))?;

        let size_bytes = final_data.len() as u64;
        let elapsed = start_time.elapsed().as_millis() as f64;

        // Calculate checksum
        let checksum = self.calculate_checksum(&final_data);

        // Create metadata
        let metadata = BackupMetadata {
            id: backup_id.clone(),
            backup_type: BackupType::Full,
            format: self.config.format,
            created_at: timestamp,
            node_id: self.node_id,
            file_path: file_path.clone(),
            size_bytes,
            triple_count: data.len() as u64,
            status: BackupStatus::Completed,
            checksum: Some(checksum),
            parent_backup_id: None,
            completed_at: Some(SystemTime::now()),
            error: None,
            encrypted: self.encryption_key.is_some(),
            compressed: self.config.enable_compression,
        };

        // Record the metadata next to the file, then register the backup
        self.persist_new_backup(&metadata).await?;
        {
            let mut backups = self.backups.write().await;
            backups.insert(backup_id.clone(), metadata.clone());
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_backups += 1;
            stats.total_bytes_backed_up += size_bytes;
            stats.last_backup = Some(timestamp);

            // Update average backup time
            if stats.total_backups > 1 {
                stats.avg_backup_time_ms =
                    (stats.avg_backup_time_ms * (stats.total_backups - 1) as f64 + elapsed)
                        / stats.total_backups as f64;
            } else {
                stats.avg_backup_time_ms = elapsed;
            }
        }

        // Verify if configured
        if self.config.verify_after_backup {
            self.verify_backup(&backup_id).await?;
        }

        // Cleanup old backups
        self.cleanup_old_backups().await?;

        info!(
            "Full backup completed: {} ({} bytes, {} triples, {:.2}ms)",
            backup_id,
            size_bytes,
            data.len(),
            elapsed
        );

        Ok(backup_id)
    }

    /// Create an incremental backup
    pub async fn create_incremental_backup(
        &self,
        parent_backup_id: &str,
        changed_data: Vec<(String, String, String)>,
    ) -> Result<String> {
        // Verify parent backup exists
        {
            let backups = self.backups.read().await;
            if !backups.contains_key(parent_backup_id) {
                return Err(ClusterError::Other(format!(
                    "Parent backup not found: {}",
                    parent_backup_id
                )));
            }
        }

        let backup_id = uuid::Uuid::new_v4().to_string();
        let timestamp = SystemTime::now();

        info!(
            "Creating incremental backup: {} (parent: {})",
            backup_id, parent_backup_id
        );

        let file_path = self.backup_path(BackupType::Incremental, &backup_id)?;

        let start_time = std::time::Instant::now();

        // Serialize incremental data
        let serialized_data = self.serialize_data(&changed_data)?;
        let compressed_data = if self.config.enable_compression {
            self.compress_data(&serialized_data)?
        } else {
            serialized_data
        };
        let context = backup_context(
            self.node_id,
            &backup_id,
            BackupType::Incremental,
            Some(parent_backup_id),
        );
        let final_data = self.seal_backup(&context, compressed_data)?;

        // Write to file
        write_file_atomically(&file_path, &final_data)
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to write backup: {}", e)))?;

        let size_bytes = final_data.len() as u64;
        let elapsed = start_time.elapsed().as_millis() as f64;

        // Calculate checksum
        let checksum = self.calculate_checksum(&final_data);

        // Create metadata
        let metadata = BackupMetadata {
            id: backup_id.clone(),
            backup_type: BackupType::Incremental,
            format: self.config.format,
            created_at: timestamp,
            node_id: self.node_id,
            file_path,
            size_bytes,
            triple_count: changed_data.len() as u64,
            status: BackupStatus::Completed,
            checksum: Some(checksum),
            parent_backup_id: Some(parent_backup_id.to_string()),
            completed_at: Some(SystemTime::now()),
            error: None,
            encrypted: self.encryption_key.is_some(),
            compressed: self.config.enable_compression,
        };

        // Record the metadata next to the file, then register the backup
        self.persist_new_backup(&metadata).await?;
        {
            let mut backups = self.backups.write().await;
            backups.insert(backup_id.clone(), metadata);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_backups += 1;
            stats.total_bytes_backed_up += size_bytes;
            stats.last_backup = Some(timestamp);

            if stats.total_backups > 1 {
                stats.avg_backup_time_ms =
                    (stats.avg_backup_time_ms * (stats.total_backups - 1) as f64 + elapsed)
                        / stats.total_backups as f64;
            } else {
                stats.avg_backup_time_ms = elapsed;
            }
        }

        if self.config.verify_after_backup {
            self.verify_backup(&backup_id).await?;
        }
        self.cleanup_old_backups().await?;

        info!(
            "Incremental backup completed: {} ({} bytes, {} triples, {:.2}ms)",
            backup_id,
            size_bytes,
            changed_data.len(),
            elapsed
        );

        Ok(backup_id)
    }

    /// Restore from backup
    ///
    /// An incremental backup is restored on top of its parents, oldest
    /// first. Each file is decoded the way its metadata records (format,
    /// compression, encryption), whatever the current configuration says.
    ///
    /// # Errors
    ///
    /// Fails when a backup in the chain is missing, does not verify, cannot
    /// be decrypted, or when the parent chain loops.
    pub async fn restore_backup(
        &self,
        options: RestoreOptions,
    ) -> Result<Vec<(String, String, String)>> {
        let backup_id = &options.backup_id;

        info!("Restoring from backup: {}", backup_id);

        // Collect the chain from the requested backup back to its full backup
        let chain = {
            let backups = self.backups.read().await;
            let mut chain = Vec::new();
            let mut seen = HashSet::new();
            let mut next = Some(backup_id.clone());
            while let Some(id) = next {
                if !seen.insert(id.clone()) {
                    return Err(ClusterError::Other(format!(
                        "backup {backup_id} has a parent chain that loops back to {id}"
                    )));
                }
                let metadata = backups
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| ClusterError::Other(format!("Backup not found: {}", id)))?;
                next = match metadata.backup_type {
                    BackupType::Incremental => metadata.parent_backup_id.clone(),
                    BackupType::Full | BackupType::Snapshot => None,
                };
                chain.push(metadata);
            }
            chain
        };

        let start_time = std::time::Instant::now();

        let mut restored_data = Vec::new();
        for metadata in chain.iter().rev() {
            // Verify backup if not skipped
            if !options.skip_verification {
                self.verify_backup(&metadata.id).await?;
            }

            // Read backup file
            let file_data = fs::read(&metadata.file_path)
                .await
                .map_err(|e| ClusterError::Other(format!("Failed to read backup file: {}", e)))?;

            // Decrypt, then decompress if needed
            let file_data = self.open_backup(metadata, file_data)?;
            let decompressed_data = if metadata.compressed {
                self.decompress_data(&file_data)?
            } else {
                file_data
            };

            restored_data.extend(self.deserialize_data(metadata.format, &decompressed_data)?);
        }

        let elapsed = start_time.elapsed().as_millis() as f64;

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_restores += 1;
            stats.last_restore = Some(SystemTime::now());

            if stats.total_restores > 1 {
                stats.avg_restore_time_ms =
                    (stats.avg_restore_time_ms * (stats.total_restores - 1) as f64 + elapsed)
                        / stats.total_restores as f64;
            } else {
                stats.avg_restore_time_ms = elapsed;
            }
        }

        info!(
            "Restore completed: {} ({} triples, {:.2}ms)",
            backup_id,
            restored_data.len(),
            elapsed
        );

        Ok(restored_data)
    }

    /// Verify backup integrity
    pub async fn verify_backup(&self, backup_id: &str) -> Result<()> {
        debug!("Verifying backup: {}", backup_id);

        let metadata = {
            let backups = self.backups.read().await;
            backups
                .get(backup_id)
                .cloned()
                .ok_or_else(|| ClusterError::Other(format!("Backup not found: {}", backup_id)))?
        };

        // Read backup file
        let file_data = fs::read(&metadata.file_path)
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to read backup file: {}", e)))?;

        // Verify checksum
        if let Some(expected_checksum) = &metadata.checksum {
            let actual_checksum = self.calculate_checksum(&file_data);
            if &actual_checksum != expected_checksum {
                return Err(ClusterError::Other(format!(
                    "Backup verification failed: checksum mismatch (expected: {}, actual: {})",
                    expected_checksum, actual_checksum
                )));
            }
        }

        // Update status, in memory and on disk
        let verified = {
            let mut backups = self.backups.write().await;
            backups.get_mut(backup_id).and_then(|backup| {
                (backup.status != BackupStatus::Verified).then(|| {
                    backup.status = BackupStatus::Verified;
                    backup.clone()
                })
            })
        };
        if let Some(metadata) = verified {
            self.persist_metadata(&metadata).await?;
        }

        debug!("Backup verified successfully: {}", backup_id);
        Ok(())
    }

    /// List all backups
    pub async fn list_backups(&self) -> Vec<BackupMetadata> {
        let backups = self.backups.read().await;
        backups.values().cloned().collect()
    }

    /// Get backup metadata
    pub async fn get_backup_metadata(&self, backup_id: &str) -> Option<BackupMetadata> {
        let backups = self.backups.read().await;
        backups.get(backup_id).cloned()
    }

    /// Delete a backup
    ///
    /// # Errors
    ///
    /// Fails when the backup does not exist, or when an incremental backup is
    /// built on it: that backup could no longer be restored, so it has to be
    /// deleted first.
    pub async fn delete_backup(&self, backup_id: &str) -> Result<()> {
        info!("Deleting backup: {}", backup_id);

        let metadata = {
            let mut backups = self.backups.write().await;
            let mut children: Vec<&str> = backups
                .values()
                .filter(|backup| backup.parent_backup_id.as_deref() == Some(backup_id))
                .map(|backup| backup.id.as_str())
                .collect();
            if !children.is_empty() {
                children.sort_unstable();
                return Err(ClusterError::Other(format!(
                    "backup {backup_id} is the parent of incremental backup(s) {}; delete them first",
                    children.join(", ")
                )));
            }
            backups
                .remove(backup_id)
                .ok_or_else(|| ClusterError::Other(format!("Backup not found: {}", backup_id)))?
        };

        // Delete the metadata first, so that no entry outlives its file
        self.remove_backup_files(&metadata).await?;

        info!("Backup deleted: {}", backup_id);
        Ok(())
    }

    /// Get statistics
    pub async fn get_statistics(&self) -> BackupStatistics {
        self.stats.read().await.clone()
    }

    /// Serialize data based on format
    fn serialize_data(&self, data: &[(String, String, String)]) -> Result<Vec<u8>> {
        match self.config.format {
            BackupFormat::Json => serde_json::to_vec(data)
                .map_err(|e| ClusterError::Serialize(format!("JSON serialization failed: {}", e))),
            BackupFormat::Binary | BackupFormat::CompressedBinary => {
                oxicode::serde::encode_to_vec(&data, oxicode::config::standard()).map_err(|e| {
                    ClusterError::Serialize(format!("Binary serialization failed: {}", e))
                })
            }
        }
    }

    /// Deserialize data written in `format`
    fn deserialize_data(
        &self,
        format: BackupFormat,
        data: &[u8],
    ) -> Result<Vec<(String, String, String)>> {
        match format {
            BackupFormat::Json => serde_json::from_slice(data).map_err(|e| {
                ClusterError::Serialize(format!("JSON deserialization failed: {}", e))
            }),
            BackupFormat::Binary | BackupFormat::CompressedBinary => {
                oxicode::serde::decode_from_slice(data, oxicode::config::standard())
                    .map(|(v, _)| v)
                    .map_err(|e| {
                        ClusterError::Serialize(format!("Binary deserialization failed: {}", e))
                    })
            }
        }
    }

    /// Compress data using zstd
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        oxiarc_zstd::encode_all(data, self.config.compression_level)
            .map_err(|e| ClusterError::Other(format!("Compression failed: {}", e)))
    }

    /// Decompress data
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        oxiarc_zstd::decode_all(data)
            .map_err(|e| ClusterError::Other(format!("Decompression failed: {}", e)))
    }

    /// Encrypt backup bytes under `context` when encryption is enabled.
    fn seal_backup(&self, context: &[u8], data: Vec<u8>) -> Result<Vec<u8>> {
        match &self.encryption_key {
            Some(key) => at_rest::seal(key, context, &data),
            None => Ok(data),
        }
    }

    /// Decrypt backup bytes according to how the backup was recorded. An
    /// encrypted backup must authenticate under the configured key and its
    /// own identity; with a key configured, an unencrypted backup (which
    /// anyone with write access could have planted) is refused; and a backup
    /// recorded as unencrypted must not be a sealed file.
    fn open_backup(&self, metadata: &BackupMetadata, data: Vec<u8>) -> Result<Vec<u8>> {
        match (metadata.encrypted, &self.encryption_key) {
            (true, Some(key)) => {
                let context = backup_context(
                    metadata.node_id,
                    &metadata.id,
                    metadata.backup_type,
                    metadata.parent_backup_id.as_deref(),
                );
                at_rest::open(key, &context, &data)
            }
            (true, None) => Err(ClusterError::Encryption(format!(
                "backup {} is encrypted but no encryption key is configured",
                metadata.id
            ))),
            (false, Some(_)) => Err(ClusterError::Encryption(format!(
                "backup {} is not encrypted but encryption is enabled; refusing to restore \
                 unauthenticated data",
                metadata.id
            ))),
            (false, None) if at_rest::is_sealed(&data) => Err(ClusterError::Encryption(format!(
                "backup {} is recorded as unencrypted but its file is encrypted",
                metadata.id
            ))),
            (false, None) => Ok(data),
        }
    }

    /// Calculate checksum (SHA256)
    fn calculate_checksum(&self, data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    /// Path of the file this manager writes backup `backup_id` to.
    fn backup_path(&self, backup_type: BackupType, backup_id: &str) -> Result<PathBuf> {
        backup_file_name(backup_type, self.node_id, backup_id)
            .map(|name| self.config.backup_dir.join(name))
            .ok_or_else(|| {
                ClusterError::Other(format!("{backup_type:?} backups are not created here"))
            })
    }

    /// Path of the metadata file kept next to the file of `metadata`.
    fn metadata_path(&self, metadata: &BackupMetadata) -> Result<PathBuf> {
        let mut path = self
            .backup_path(metadata.backup_type, &metadata.id)?
            .into_os_string();
        path.push(METADATA_SUFFIX);
        Ok(PathBuf::from(path))
    }

    /// Write `metadata` next to its backup file, atomically. The file path is
    /// stored relative to the backup directory and derived again on load.
    async fn persist_metadata(&self, metadata: &BackupMetadata) -> Result<()> {
        let mut stored = metadata.clone();
        stored.file_path = backup_file_name(metadata.backup_type, self.node_id, &metadata.id)
            .map(PathBuf::from)
            .ok_or_else(|| ClusterError::Other("unsupported backup type".to_string()))?;
        let json = serde_json::to_vec_pretty(&stored).map_err(|e| {
            ClusterError::Serialize(format!("Failed to encode backup metadata: {e}"))
        })?;
        let path = self.metadata_path(metadata)?;
        write_file_atomically(&path, &json).await.map_err(|e| {
            ClusterError::Other(format!(
                "Failed to write backup metadata {}: {e}",
                path.display()
            ))
        })
    }

    /// Persist the metadata of a backup whose file was just written; on
    /// failure the file is removed again, so no unlisted backup is left.
    async fn persist_new_backup(&self, metadata: &BackupMetadata) -> Result<()> {
        if let Err(e) = self.persist_metadata(metadata).await {
            let _ = fs::remove_file(&metadata.file_path).await;
            return Err(e);
        }
        Ok(())
    }

    /// Remove the metadata file, then the data file, of a backup.
    async fn remove_backup_files(&self, metadata: &BackupMetadata) -> Result<()> {
        for path in [self.metadata_path(metadata)?, metadata.file_path.clone()] {
            match fs::remove_file(&path).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(ClusterError::Other(format!(
                        "Failed to delete {}: {e}",
                        path.display()
                    )))
                }
            }
        }
        Ok(())
    }

    /// Load the metadata of this node's backups from the backup directory.
    ///
    /// Metadata that cannot be parsed, names any file other than the one
    /// this manager would have written, or whose backup file is missing is
    /// skipped with a warning, and nothing it points at is ever touched.
    async fn load_backup_metadata(&self) -> Result<()> {
        let prefixes = [
            format!("backup_full_{}_", self.node_id),
            format!("backup_incr_{}_", self.node_id),
        ];
        let mut entries = fs::read_dir(&self.config.backup_dir).await.map_err(|e| {
            ClusterError::Other(format!(
                "Failed to list backup directory {}: {e}",
                self.config.backup_dir.display()
            ))
        })?;

        let mut loaded = HashMap::new();
        loop {
            let entry = match entries.next_entry().await {
                Ok(Some(entry)) => entry,
                Ok(None) => break,
                Err(e) => {
                    return Err(ClusterError::Other(format!(
                        "Failed to list backup directory {}: {e}",
                        self.config.backup_dir.display()
                    )))
                }
            };
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(data_name) = name.strip_suffix(METADATA_SUFFIX) else {
                continue;
            };
            if !prefixes.iter().any(|prefix| data_name.starts_with(prefix)) {
                continue;
            }
            match self.read_metadata_file(&entry.path(), data_name).await {
                Ok(metadata) => {
                    loaded.insert(metadata.id.clone(), metadata);
                }
                Err(reason) => warn!(
                    "Ignoring backup metadata {}: {reason}",
                    entry.path().display()
                ),
            }
        }

        info!(
            "Loaded {} backups from {}",
            loaded.len(),
            self.config.backup_dir.display()
        );
        self.backups.write().await.extend(loaded);
        Ok(())
    }

    /// Read and check one metadata file, whose name says it describes
    /// `data_name`.
    async fn read_metadata_file(
        &self,
        path: &Path,
        data_name: &str,
    ) -> std::result::Result<BackupMetadata, String> {
        let bytes = fs::read(path).await.map_err(|e| e.to_string())?;
        let mut metadata: BackupMetadata =
            serde_json::from_slice(&bytes).map_err(|e| format!("cannot parse it: {e}"))?;
        if metadata.node_id != self.node_id {
            return Err(format!("it belongs to node {}", metadata.node_id));
        }
        for id in std::iter::once(&metadata.id).chain(&metadata.parent_backup_id) {
            uuid::Uuid::parse_str(id).map_err(|_| format!("backup id `{id}` is not a UUID"))?;
        }
        let expected = backup_file_name(metadata.backup_type, self.node_id, &metadata.id)
            .ok_or_else(|| format!("{:?} backups are not created here", metadata.backup_type))?;
        if expected != data_name || metadata.file_path != Path::new(&expected) {
            return Err(format!(
                "it describes `{}`, expected `{expected}`",
                metadata.file_path.display()
            ));
        }
        let file_path = self.config.backup_dir.join(&expected);
        if !file_path.is_file() {
            return Err(format!("backup file {} is missing", file_path.display()));
        }
        metadata.file_path = file_path;
        Ok(metadata)
    }

    /// Apply the retention policy.
    ///
    /// The newest `max_backups` backups that are younger than
    /// `max_backup_age_days` are kept, together with every backup they are
    /// built on, so that each kept backup can still be restored. All other
    /// backups are deleted.
    async fn cleanup_old_backups(&self) -> Result<()> {
        let removed = {
            let mut backups = self.backups.write().await;

            let mut newest_first: Vec<&BackupMetadata> = backups.values().collect();
            newest_first.sort_by_key(|b| std::cmp::Reverse(b.created_at));

            let now = SystemTime::now();
            let max_age = std::time::Duration::from_secs(
                u64::from(self.config.max_backup_age_days) * 24 * 60 * 60,
            );
            let mut keep = HashSet::new();
            for backup in newest_first.iter().take(self.config.max_backups as usize) {
                let expired = now
                    .duration_since(backup.created_at)
                    .is_ok_and(|age| age > max_age);
                if expired {
                    continue;
                }
                // Keep the backup and the chain it is restored on top of.
                let mut next = Some(backup.id.clone());
                while let Some(id) = next {
                    if !keep.insert(id.clone()) {
                        break;
                    }
                    next = backups
                        .get(&id)
                        .filter(|b| b.backup_type == BackupType::Incremental)
                        .and_then(|b| b.parent_backup_id.clone());
                }
            }

            let doomed: Vec<String> = backups
                .keys()
                .filter(|id| !keep.contains(*id))
                .cloned()
                .collect();
            doomed
                .iter()
                .filter_map(|id| backups.remove(id))
                .collect::<Vec<_>>()
        };

        for backup in &removed {
            info!("Removing old backup (retention policy): {}", backup.id);
            if let Err(e) = self.remove_backup_files(backup).await {
                warn!("Failed to delete backup {}: {}", backup.id, e);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests;
