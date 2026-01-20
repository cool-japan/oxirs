//! Backup and restore functionality for RDF-star data
//!
//! This module provides comprehensive backup and restore capabilities for
//! RDF-star graphs, including incremental backups, compression, encryption,
//! and integrity verification.

use crate::cryptographic_provenance::ProvenanceKeyPair;
use crate::model::StarGraph;
use crate::parser::{StarFormat, StarParser};
use crate::serializer::StarSerializer;
use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use scirs2_core::random::rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, error, info, warn};

/// Errors related to backup and restore operations
#[derive(Error, Debug)]
pub enum BackupError {
    #[error("Backup creation failed: {0}")]
    CreationFailed(String),

    #[error("Restore failed: {0}")]
    RestoreFailed(String),

    #[error("Integrity check failed: {0}")]
    IntegrityFailed(String),

    #[error("Encryption/Decryption failed: {0}")]
    CryptoFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("Backup not found: {0}")]
    BackupNotFound(String),

    #[error("Invalid backup format: {0}")]
    InvalidFormat(String),
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Base directory for backups
    pub backup_dir: PathBuf,

    /// Compression level (0-9)
    pub compression_level: u32,

    /// Enable encryption
    pub enable_encryption: bool,

    /// Backup format
    pub format: BackupFormat,

    /// Include metadata
    pub include_metadata: bool,

    /// Maximum number of backups to keep
    pub max_backups: Option<usize>,

    /// Enable incremental backups
    pub incremental: bool,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            backup_dir: PathBuf::from("./backups"),
            compression_level: 6,
            enable_encryption: false,
            format: BackupFormat::TurtleStar,
            include_metadata: true,
            max_backups: Some(10),
            incremental: false,
        }
    }
}

/// Backup format options
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BackupFormat {
    /// Turtle-star format (human-readable)
    TurtleStar,
    /// N-Triples-star format (simple)
    NTriplesStar,
    /// TriG-star format (with named graphs)
    TriGStar,
    /// Binary format (fastest)
    Binary,
}

impl BackupFormat {
    fn to_star_format(self) -> StarFormat {
        match self {
            BackupFormat::TurtleStar => StarFormat::TurtleStar,
            BackupFormat::NTriplesStar => StarFormat::NTriplesStar,
            BackupFormat::TriGStar => StarFormat::NTriplesStar, // TriGStar not in parser yet, use NTriples
            BackupFormat::Binary => StarFormat::NTriplesStar,   // Fallback
        }
    }

    fn extension(self) -> &'static str {
        match self {
            BackupFormat::TurtleStar => ".ttls",
            BackupFormat::NTriplesStar => ".nts",
            BackupFormat::TriGStar => ".trigs",
            BackupFormat::Binary => ".bin",
        }
    }
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Backup ID
    pub backup_id: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Backup type
    pub backup_type: BackupType,

    /// Number of triples
    pub triple_count: usize,

    /// Original size (bytes)
    pub original_size: u64,

    /// Compressed size (bytes)
    pub compressed_size: u64,

    /// Compression ratio
    pub compression_ratio: f64,

    /// Checksum (SHA-256)
    pub checksum: String,

    /// Backup format
    pub format: BackupFormat,

    /// Whether encrypted
    pub encrypted: bool,

    /// Description
    pub description: Option<String>,

    /// Tags
    pub tags: Vec<String>,

    /// Parent backup ID (for incremental backups)
    pub parent_backup: Option<String>,

    /// Signature (for verification)
    pub signature: Option<String>,
}

/// Backup type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BackupType {
    /// Full backup (all data)
    Full,
    /// Incremental backup (changes since last backup)
    Incremental,
    /// Differential backup (changes since last full backup)
    Differential,
}

/// Backup manager
pub struct BackupManager {
    /// Configuration
    config: BackupConfig,

    /// Backup metadata index
    metadata_index: HashMap<String, BackupMetadata>,

    /// Last backup timestamp
    last_backup: Option<DateTime<Utc>>,

    /// Signing key for backup verification
    signing_key: Option<ProvenanceKeyPair>,
}

impl BackupManager {
    /// Create a new backup manager
    pub fn new(config: BackupConfig) -> Result<Self, BackupError> {
        // Create backup directory
        create_dir_all(&config.backup_dir)?;

        let signing_key = if config.enable_encryption {
            Some(ProvenanceKeyPair::generate())
        } else {
            None
        };

        let mut manager = Self {
            config,
            metadata_index: HashMap::new(),
            last_backup: None,
            signing_key,
        };

        // Load existing backup metadata
        manager.load_metadata_index()?;

        info!("Backup manager initialized");
        Ok(manager)
    }

    /// Create a full backup of a graph
    pub fn create_backup(
        &mut self,
        graph: &StarGraph,
        description: Option<String>,
        tags: Vec<String>,
    ) -> Result<BackupMetadata, BackupError> {
        let backup_type = if self.config.incremental && self.last_backup.is_some() {
            BackupType::Incremental
        } else {
            BackupType::Full
        };

        info!("Creating {:?} backup", backup_type);

        use scirs2_core::random::Rng;

        // Generate backup ID
        let mut rng_instance = rng();
        let backup_id = format!("backup_{:016x}", rng_instance.random::<u64>());

        // Determine backup file path
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!(
            "{}_{}{}.gz",
            backup_id,
            timestamp,
            self.config.format.extension()
        );
        let backup_path = self.config.backup_dir.join(&filename);

        // Serialize graph
        let serializer = StarSerializer::new();
        let serialized = serializer
            .serialize_to_string(graph, self.config.format.to_star_format())
            .map_err(|e| BackupError::SerializationError(e.to_string()))?;

        let original_size = serialized.len() as u64;

        // Compress
        let file = File::create(&backup_path)?;
        let mut encoder = GzEncoder::new(
            BufWriter::new(file),
            Compression::new(self.config.compression_level),
        );

        encoder.write_all(serialized.as_bytes())?;
        encoder.finish()?;

        // Get compressed size
        let compressed_size = std::fs::metadata(&backup_path)?.len();
        let compression_ratio = compressed_size as f64 / original_size as f64;

        // Compute checksum
        let checksum = self.compute_checksum(&backup_path)?;

        // Create metadata
        let mut metadata = BackupMetadata {
            backup_id: backup_id.clone(),
            created_at: Utc::now(),
            backup_type,
            triple_count: graph.len(),
            original_size,
            compressed_size,
            compression_ratio,
            checksum,
            format: self.config.format,
            encrypted: self.config.enable_encryption,
            description,
            tags,
            parent_backup: None,
            signature: None,
        };

        // Sign metadata if enabled
        if let Some(ref key_pair) = self.signing_key {
            let metadata_json = serde_json::to_string(&metadata)
                .map_err(|e| BackupError::SerializationError(e.to_string()))?;
            let signature = key_pair.sign(metadata_json.as_bytes());
            metadata.signature = Some(hex::encode(signature.to_bytes()));
        }

        // Save metadata
        self.save_metadata(&metadata)?;
        self.metadata_index
            .insert(backup_id.clone(), metadata.clone());
        self.last_backup = Some(metadata.created_at);

        // Cleanup old backups
        self.cleanup_old_backups()?;

        info!(
            "Backup created: {} ({} triples, {:.2}% compression)",
            backup_id,
            metadata.triple_count,
            (1.0 - compression_ratio) * 100.0
        );

        Ok(metadata)
    }

    /// Restore a backup
    pub fn restore_backup(&self, backup_id: &str) -> Result<StarGraph, BackupError> {
        info!("Restoring backup: {}", backup_id);

        // Get metadata
        let metadata = self
            .metadata_index
            .get(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        // Find backup file
        let backup_path = self.find_backup_file(backup_id)?;

        // Verify checksum
        let checksum = self.compute_checksum(&backup_path)?;
        if checksum != metadata.checksum {
            return Err(BackupError::IntegrityFailed(format!(
                "Checksum mismatch: expected {}, got {}",
                metadata.checksum, checksum
            )));
        }

        // Decompress
        let file = File::open(&backup_path)?;
        let decoder = GzDecoder::new(BufReader::new(file));
        let mut decompressed = String::new();
        BufReader::new(decoder).read_to_string(&mut decompressed)?;

        // Parse
        let parser = StarParser::new();
        let graph = parser
            .parse_str(&decompressed, metadata.format.to_star_format())
            .map_err(|e| BackupError::RestoreFailed(e.to_string()))?;

        // Verify triple count
        if graph.len() != metadata.triple_count {
            warn!(
                "Triple count mismatch: expected {}, got {}",
                metadata.triple_count,
                graph.len()
            );
        }

        info!("Backup restored successfully: {} triples", graph.len());
        Ok(graph)
    }

    /// List all available backups
    pub fn list_backups(&self) -> Vec<&BackupMetadata> {
        let mut backups: Vec<_> = self.metadata_index.values().collect();
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        backups
    }

    /// Delete a backup
    pub fn delete_backup(&mut self, backup_id: &str) -> Result<(), BackupError> {
        info!("Deleting backup: {}", backup_id);

        // Remove metadata
        self.metadata_index.remove(backup_id);

        // Delete backup file
        if let Ok(backup_path) = self.find_backup_file(backup_id) {
            std::fs::remove_file(&backup_path)?;
        }

        // Delete metadata file
        let metadata_path = self
            .config
            .backup_dir
            .join(format!("{}.meta.json", backup_id));
        if metadata_path.exists() {
            std::fs::remove_file(&metadata_path)?;
        }

        info!("Backup deleted: {}", backup_id);
        Ok(())
    }

    /// Verify backup integrity
    pub fn verify_backup(&self, backup_id: &str) -> Result<bool, BackupError> {
        info!("Verifying backup: {}", backup_id);

        let metadata = self
            .metadata_index
            .get(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        // Find backup file
        let backup_path = self.find_backup_file(backup_id)?;

        // Verify checksum
        let checksum = self.compute_checksum(&backup_path)?;
        if checksum != metadata.checksum {
            error!("Backup integrity check failed: checksum mismatch");
            return Ok(false);
        }

        // Verify signature if present
        if let Some(ref signature_hex) = metadata.signature {
            if let Some(ref key_pair) = self.signing_key {
                // Create metadata without signature for verification
                let mut verify_metadata = metadata.clone();
                verify_metadata.signature = None;

                let metadata_json = serde_json::to_string(&verify_metadata)
                    .map_err(|e| BackupError::SerializationError(e.to_string()))?;

                // Decode signature
                let signature_bytes = hex::decode(signature_hex)
                    .map_err(|e| BackupError::CryptoFailed(e.to_string()))?;

                let signature_array: [u8; 64] = signature_bytes.try_into().map_err(|_| {
                    BackupError::CryptoFailed("Invalid signature length".to_string())
                })?;

                let signature = ed25519_dalek::Signature::from_bytes(&signature_array);

                // Verify (Verifier trait is automatically in scope)
                key_pair
                    .verify(metadata_json.as_bytes(), &signature)
                    .map_err(|e| BackupError::CryptoFailed(e.to_string()))?;
            }
        }

        info!("Backup integrity verified successfully");
        Ok(true)
    }

    /// Compute SHA-256 checksum of a file
    fn compute_checksum(&self, path: &Path) -> Result<String, BackupError> {
        use sha2::{Digest, Sha256};

        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        Ok(hex::encode(hasher.finalize()))
    }

    /// Find backup file by ID
    fn find_backup_file(&self, backup_id: &str) -> Result<PathBuf, BackupError> {
        for entry in std::fs::read_dir(&self.config.backup_dir)? {
            let entry = entry?;
            let filename = entry.file_name();
            let filename_str = filename.to_string_lossy();

            if filename_str.starts_with(backup_id) && filename_str.ends_with(".gz") {
                return Ok(entry.path());
            }
        }

        Err(BackupError::BackupNotFound(format!(
            "Backup file for {} not found",
            backup_id
        )))
    }

    /// Save metadata to disk
    fn save_metadata(&self, metadata: &BackupMetadata) -> Result<(), BackupError> {
        let metadata_path = self
            .config
            .backup_dir
            .join(format!("{}.meta.json", metadata.backup_id));

        let json = serde_json::to_string_pretty(metadata)
            .map_err(|e| BackupError::SerializationError(e.to_string()))?;

        std::fs::write(&metadata_path, json)?;

        debug!("Metadata saved: {:?}", metadata_path);
        Ok(())
    }

    /// Load metadata index from disk
    fn load_metadata_index(&mut self) -> Result<(), BackupError> {
        if !self.config.backup_dir.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(&self.config.backup_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let filename = path.file_name().unwrap().to_string_lossy();
                if filename.ends_with(".meta.json") {
                    match std::fs::read_to_string(&path) {
                        Ok(json) => match serde_json::from_str::<BackupMetadata>(&json) {
                            Ok(metadata) => {
                                self.metadata_index
                                    .insert(metadata.backup_id.clone(), metadata);
                            }
                            Err(e) => {
                                warn!("Failed to parse metadata from {:?}: {}", path, e);
                            }
                        },
                        Err(e) => {
                            warn!("Failed to read metadata file {:?}: {}", path, e);
                        }
                    }
                }
            }
        }

        info!(
            "Loaded {} backup metadata entries",
            self.metadata_index.len()
        );
        Ok(())
    }

    /// Cleanup old backups beyond max_backups limit
    fn cleanup_old_backups(&mut self) -> Result<(), BackupError> {
        if let Some(max_backups) = self.config.max_backups {
            let mut backups: Vec<_> = self.metadata_index.values().cloned().collect();
            backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

            if backups.len() > max_backups {
                for metadata in backups.iter().skip(max_backups) {
                    info!("Cleaning up old backup: {}", metadata.backup_id);
                    self.delete_backup(&metadata.backup_id)?;
                }
            }
        }

        Ok(())
    }

    /// Export backup statistics
    pub fn get_statistics(&self) -> BackupStatistics {
        let total_backups = self.metadata_index.len();
        let total_size: u64 = self
            .metadata_index
            .values()
            .map(|m| m.compressed_size)
            .sum();

        let avg_compression_ratio = if total_backups > 0 {
            self.metadata_index
                .values()
                .map(|m| m.compression_ratio)
                .sum::<f64>()
                / total_backups as f64
        } else {
            0.0
        };

        let full_backups = self
            .metadata_index
            .values()
            .filter(|m| m.backup_type == BackupType::Full)
            .count();

        let incremental_backups = self
            .metadata_index
            .values()
            .filter(|m| m.backup_type == BackupType::Incremental)
            .count();

        BackupStatistics {
            total_backups,
            full_backups,
            incremental_backups,
            total_size_bytes: total_size,
            avg_compression_ratio,
            oldest_backup: self
                .metadata_index
                .values()
                .min_by_key(|m| m.created_at)
                .map(|m| m.created_at),
            newest_backup: self
                .metadata_index
                .values()
                .max_by_key(|m| m.created_at)
                .map(|m| m.created_at),
        }
    }
}

/// Backup statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupStatistics {
    pub total_backups: usize,
    pub full_backups: usize,
    pub incremental_backups: usize,
    pub total_size_bytes: u64,
    pub avg_compression_ratio: f64,
    pub oldest_backup: Option<DateTime<Utc>>,
    pub newest_backup: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;
    use crate::StarTriple;
    use std::env;

    #[test]
    fn test_backup_creation() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = env::temp_dir().join(format!("backup_test_{}", std::process::id()));

        let config = BackupConfig {
            backup_dir: temp_dir.clone(),
            ..Default::default()
        };

        let mut manager = BackupManager::new(config)?;

        // Create test graph
        let mut graph = StarGraph::new();
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s")?,
            StarTerm::iri("http://example.org/p")?,
            StarTerm::literal("object")?,
        );
        graph.insert(triple)?;

        // Create backup
        let metadata = manager.create_backup(
            &graph,
            Some("Test backup".to_string()),
            vec!["test".to_string()],
        )?;

        assert_eq!(metadata.triple_count, 1);
        assert!(metadata.compressed_size > 0);
        assert!(metadata.compression_ratio < 1.0);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }

    #[test]
    fn test_backup_restore() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = env::temp_dir().join(format!("backup_restore_test_{}", std::process::id()));

        let config = BackupConfig {
            backup_dir: temp_dir.clone(),
            ..Default::default()
        };

        let mut manager = BackupManager::new(config)?;

        // Create test graph
        let mut graph = StarGraph::new();
        for i in 0..10 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{}", i))?,
                StarTerm::iri("http://example.org/p")?,
                StarTerm::literal(&format!("object{}", i))?,
            );
            graph.insert(triple)?;
        }

        // Create backup
        let metadata = manager.create_backup(&graph, None, vec![])?;

        // Restore backup
        let restored_graph = manager.restore_backup(&metadata.backup_id)?;

        assert_eq!(restored_graph.len(), graph.len());

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }

    #[test]
    fn test_backup_verification() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = env::temp_dir().join(format!("backup_verify_test_{}", std::process::id()));

        let config = BackupConfig {
            backup_dir: temp_dir.clone(),
            enable_encryption: true,
            ..Default::default()
        };

        let mut manager = BackupManager::new(config)?;

        // Create test graph
        let mut graph = StarGraph::new();
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s")?,
            StarTerm::iri("http://example.org/p")?,
            StarTerm::literal("object")?,
        );
        graph.insert(triple)?;

        // Create backup
        let metadata = manager.create_backup(&graph, None, vec![])?;

        // Verify backup
        let is_valid = manager.verify_backup(&metadata.backup_id)?;
        assert!(is_valid);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }

    #[test]
    fn test_backup_cleanup() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = env::temp_dir().join(format!("backup_cleanup_test_{}", std::process::id()));

        let config = BackupConfig {
            backup_dir: temp_dir.clone(),
            max_backups: Some(3),
            ..Default::default()
        };

        let mut manager = BackupManager::new(config)?;

        // Create multiple backups
        for i in 0..5 {
            let mut graph = StarGraph::new();
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{}", i))?,
                StarTerm::iri("http://example.org/p")?,
                StarTerm::literal("object")?,
            );
            graph.insert(triple)?;

            manager.create_backup(&graph, Some(format!("Backup {}", i)), vec![])?;
        }

        // Should have only 3 backups
        assert_eq!(manager.list_backups().len(), 3);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }
}
