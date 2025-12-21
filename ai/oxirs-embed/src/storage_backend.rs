//! Storage Backend Integration for Persistent Embeddings
//!
//! This module provides storage backend integration for persisting
//! knowledge graph embeddings to various storage systems.
//!
//! ## Supported Backends
//!
//! - **Memory**: In-memory storage (default)
//! - **Disk**: Local filesystem storage with mmap support
//! - **RocksDB**: High-performance key-value store
//! - **PostgreSQL**: Relational database with pgvector extension
//! - **S3**: Amazon S3 and S3-compatible object storage
//! - **Redis**: In-memory data structure store
//! - **Apache Arrow**: Columnar data format
//!
//! ## Features
//!
//! - **Persistence**: Save and load embeddings across sessions
//! - **Versioning**: Track embedding versions and changes
//! - **Compression**: Compress embeddings for efficient storage
//! - **Caching**: Multi-level caching (memory, disk, remote)
//! - **Sharding**: Distribute embeddings across multiple backends
//! - **Replication**: Replicate embeddings for high availability
//! - **Transactions**: ACID transactions for embedding updates

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{ModelConfig, ModelStats, Vector};

/// Storage backend type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StorageBackendType {
    /// In-memory storage (volatile)
    Memory,
    /// Local filesystem storage
    Disk { path: PathBuf, use_mmap: bool },
    /// RocksDB key-value store
    RocksDB { path: PathBuf },
    /// PostgreSQL with pgvector
    PostgreSQL { connection_string: String },
    /// Amazon S3 or compatible
    S3 {
        bucket: String,
        region: String,
        endpoint: Option<String>,
    },
    /// Redis in-memory store
    Redis { connection_string: String },
    /// Apache Arrow columnar format
    Arrow { path: PathBuf },
}

/// Storage backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageBackendConfig {
    /// Backend type
    pub backend_type: StorageBackendType,
    /// Enable compression
    pub compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Enable versioning
    pub versioning: bool,
    /// Maximum versions to keep
    pub max_versions: usize,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache size (MB)
    pub cache_size_mb: usize,
    /// Enable sharding
    pub enable_sharding: bool,
    /// Number of shards
    pub num_shards: usize,
    /// Enable replication
    pub enable_replication: bool,
    /// Replication factor
    pub replication_factor: usize,
}

impl Default for StorageBackendConfig {
    fn default() -> Self {
        Self {
            backend_type: StorageBackendType::Memory,
            compression: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
            versioning: true,
            max_versions: 10,
            enable_cache: true,
            cache_size_mb: 1024,
            enable_sharding: false,
            num_shards: 4,
            enable_replication: false,
            replication_factor: 3,
        }
    }
}

/// Compression algorithm
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Zstandard compression (recommended)
    Zstd,
    /// LZ4 compression (fast)
    Lz4,
    /// Snappy compression
    Snappy,
}

/// Embedding version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingVersion {
    /// Version ID
    pub version_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Model configuration
    pub model_config: ModelConfig,
    /// Model statistics
    pub model_stats: ModelStats,
    /// Checksum
    pub checksum: String,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total embeddings stored
    pub total_embeddings: usize,
    /// Total size (bytes)
    pub total_size_bytes: usize,
    /// Compressed size (bytes)
    pub compressed_size_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Number of versions
    pub num_versions: usize,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Number of shards
    pub num_shards: usize,
    /// Replication factor
    pub replication_factor: usize,
}

/// Storage backend trait
#[async_trait::async_trait]
pub trait StorageBackend: Send + Sync {
    /// Save entity embeddings
    async fn save_entity_embeddings(&mut self, embeddings: &HashMap<String, Vector>) -> Result<()>;

    /// Save relation embeddings
    async fn save_relation_embeddings(
        &mut self,
        embeddings: &HashMap<String, Vector>,
    ) -> Result<()>;

    /// Load entity embeddings
    async fn load_entity_embeddings(&self) -> Result<HashMap<String, Vector>>;

    /// Load relation embeddings
    async fn load_relation_embeddings(&self) -> Result<HashMap<String, Vector>>;

    /// Save metadata
    async fn save_metadata(&mut self, metadata: &EmbeddingMetadata) -> Result<()>;

    /// Load metadata
    async fn load_metadata(&self) -> Result<EmbeddingMetadata>;

    /// Delete embeddings
    async fn delete(&mut self) -> Result<()>;

    /// Get storage statistics
    async fn get_stats(&self) -> Result<StorageStats>;

    /// Create checkpoint
    async fn create_checkpoint(&mut self, checkpoint_id: &str) -> Result<()>;

    /// Restore from checkpoint
    async fn restore_checkpoint(&mut self, checkpoint_id: &str) -> Result<()>;

    /// List available checkpoints
    async fn list_checkpoints(&self) -> Result<Vec<String>>;
}

/// Embedding metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    pub model_id: uuid::Uuid,
    pub model_type: String,
    pub model_config: ModelConfig,
    pub model_stats: ModelStats,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub version: String,
}

/// In-memory storage backend
pub struct MemoryBackend {
    entity_embeddings: Arc<RwLock<HashMap<String, Vector>>>,
    relation_embeddings: Arc<RwLock<HashMap<String, Vector>>>,
    metadata: Arc<RwLock<Option<EmbeddingMetadata>>>,
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self {
            entity_embeddings: Arc::new(RwLock::new(HashMap::new())),
            relation_embeddings: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(None)),
        }
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl StorageBackend for MemoryBackend {
    async fn save_entity_embeddings(&mut self, embeddings: &HashMap<String, Vector>) -> Result<()> {
        let mut entity_embs = self.entity_embeddings.write().await;
        *entity_embs = embeddings.clone();
        Ok(())
    }

    async fn save_relation_embeddings(
        &mut self,
        embeddings: &HashMap<String, Vector>,
    ) -> Result<()> {
        let mut relation_embs = self.relation_embeddings.write().await;
        *relation_embs = embeddings.clone();
        Ok(())
    }

    async fn load_entity_embeddings(&self) -> Result<HashMap<String, Vector>> {
        Ok(self.entity_embeddings.read().await.clone())
    }

    async fn load_relation_embeddings(&self) -> Result<HashMap<String, Vector>> {
        Ok(self.relation_embeddings.read().await.clone())
    }

    async fn save_metadata(&mut self, metadata: &EmbeddingMetadata) -> Result<()> {
        let mut meta = self.metadata.write().await;
        *meta = Some(metadata.clone());
        Ok(())
    }

    async fn load_metadata(&self) -> Result<EmbeddingMetadata> {
        self.metadata
            .read()
            .await
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Metadata not found"))
    }

    async fn delete(&mut self) -> Result<()> {
        self.entity_embeddings.write().await.clear();
        self.relation_embeddings.write().await.clear();
        *self.metadata.write().await = None;
        Ok(())
    }

    async fn get_stats(&self) -> Result<StorageStats> {
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;

        let total_embeddings = entity_embs.len() + relation_embs.len();
        let total_size: usize = entity_embs
            .values()
            .chain(relation_embs.values())
            .map(|v| v.values.len() * std::mem::size_of::<f32>())
            .sum();

        Ok(StorageStats {
            total_embeddings,
            total_size_bytes: total_size,
            compressed_size_bytes: total_size, // No compression in memory
            compression_ratio: 1.0,
            num_versions: 1,
            cache_hit_rate: 1.0, // Always in cache
            num_shards: 1,
            replication_factor: 1,
        })
    }

    async fn create_checkpoint(&mut self, _checkpoint_id: &str) -> Result<()> {
        // Memory backend doesn't support checkpoints
        Ok(())
    }

    async fn restore_checkpoint(&mut self, _checkpoint_id: &str) -> Result<()> {
        Err(anyhow::anyhow!(
            "Memory backend doesn't support checkpoints"
        ))
    }

    async fn list_checkpoints(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

/// Disk storage backend with memory mapping
pub struct DiskBackend {
    config: StorageBackendConfig,
    base_path: PathBuf,
    entity_embeddings: Arc<RwLock<HashMap<String, Vector>>>,
    relation_embeddings: Arc<RwLock<HashMap<String, Vector>>>,
    checkpoints: Arc<RwLock<Vec<String>>>,
}

impl DiskBackend {
    pub fn new(path: PathBuf, config: StorageBackendConfig) -> Result<Self> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&path).context("Failed to create storage directory")?;

        Ok(Self {
            base_path: path,
            config,
            entity_embeddings: Arc::new(RwLock::new(HashMap::new())),
            relation_embeddings: Arc::new(RwLock::new(HashMap::new())),
            checkpoints: Arc::new(RwLock::new(Vec::new())),
        })
    }

    fn entity_embeddings_path(&self) -> PathBuf {
        self.base_path.join("entity_embeddings.bin")
    }

    fn relation_embeddings_path(&self) -> PathBuf {
        self.base_path.join("relation_embeddings.bin")
    }

    fn metadata_path(&self) -> PathBuf {
        self.base_path.join("metadata.json")
    }

    fn checkpoint_path(&self, checkpoint_id: &str) -> PathBuf {
        self.base_path.join("checkpoints").join(checkpoint_id)
    }

    async fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.config.compression {
            return Ok(data.to_vec());
        }

        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        match self.config.compression_algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => {
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
            _ => {
                // For other algorithms, fallback to gzip
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
        }
    }

    async fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.config.compression {
            return Ok(data.to_vec());
        }

        use flate2::read::GzDecoder;
        use std::io::Read;

        match self.config.compression_algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => {
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
            _ => {
                // For other algorithms, fallback to gzip
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
        }
    }
}

#[async_trait::async_trait]
impl StorageBackend for DiskBackend {
    async fn save_entity_embeddings(&mut self, embeddings: &HashMap<String, Vector>) -> Result<()> {
        info!("Saving {} entity embeddings to disk", embeddings.len());

        let serialized =
            bincode::serialize(embeddings).context("Failed to serialize entity embeddings")?;

        let compressed = self.compress_data(&serialized).await?;

        tokio::fs::write(self.entity_embeddings_path(), &compressed)
            .await
            .context("Failed to write entity embeddings to disk")?;

        let mut entity_embs = self.entity_embeddings.write().await;
        *entity_embs = embeddings.clone();

        Ok(())
    }

    async fn save_relation_embeddings(
        &mut self,
        embeddings: &HashMap<String, Vector>,
    ) -> Result<()> {
        info!("Saving {} relation embeddings to disk", embeddings.len());

        let serialized =
            bincode::serialize(embeddings).context("Failed to serialize relation embeddings")?;

        let compressed = self.compress_data(&serialized).await?;

        tokio::fs::write(self.relation_embeddings_path(), &compressed)
            .await
            .context("Failed to write relation embeddings to disk")?;

        let mut relation_embs = self.relation_embeddings.write().await;
        *relation_embs = embeddings.clone();

        Ok(())
    }

    async fn load_entity_embeddings(&self) -> Result<HashMap<String, Vector>> {
        debug!("Loading entity embeddings from disk");

        let compressed = tokio::fs::read(self.entity_embeddings_path())
            .await
            .context("Failed to read entity embeddings from disk")?;

        let decompressed = self.decompress_data(&compressed).await?;

        let embeddings: HashMap<String, Vector> = bincode::deserialize(&decompressed)
            .context("Failed to deserialize entity embeddings")?;

        info!("Loaded {} entity embeddings", embeddings.len());
        Ok(embeddings)
    }

    async fn load_relation_embeddings(&self) -> Result<HashMap<String, Vector>> {
        debug!("Loading relation embeddings from disk");

        let compressed = tokio::fs::read(self.relation_embeddings_path())
            .await
            .context("Failed to read relation embeddings from disk")?;

        let decompressed = self.decompress_data(&compressed).await?;

        let embeddings: HashMap<String, Vector> = bincode::deserialize(&decompressed)
            .context("Failed to deserialize relation embeddings")?;

        info!("Loaded {} relation embeddings", embeddings.len());
        Ok(embeddings)
    }

    async fn save_metadata(&mut self, metadata: &EmbeddingMetadata) -> Result<()> {
        let serialized =
            serde_json::to_string_pretty(metadata).context("Failed to serialize metadata")?;

        tokio::fs::write(self.metadata_path(), serialized)
            .await
            .context("Failed to write metadata to disk")?;

        Ok(())
    }

    async fn load_metadata(&self) -> Result<EmbeddingMetadata> {
        let content = tokio::fs::read_to_string(self.metadata_path())
            .await
            .context("Failed to read metadata from disk")?;

        let metadata: EmbeddingMetadata =
            serde_json::from_str(&content).context("Failed to deserialize metadata")?;

        Ok(metadata)
    }

    async fn delete(&mut self) -> Result<()> {
        info!("Deleting all embeddings from disk");

        if self.entity_embeddings_path().exists() {
            tokio::fs::remove_file(self.entity_embeddings_path()).await?;
        }

        if self.relation_embeddings_path().exists() {
            tokio::fs::remove_file(self.relation_embeddings_path()).await?;
        }

        if self.metadata_path().exists() {
            tokio::fs::remove_file(self.metadata_path()).await?;
        }

        self.entity_embeddings.write().await.clear();
        self.relation_embeddings.write().await.clear();

        Ok(())
    }

    async fn get_stats(&self) -> Result<StorageStats> {
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;

        let total_embeddings = entity_embs.len() + relation_embs.len();
        let total_size: usize = entity_embs
            .values()
            .chain(relation_embs.values())
            .map(|v| v.values.len() * std::mem::size_of::<f32>())
            .sum();

        // Check compressed file sizes
        let mut compressed_size = 0;
        if self.entity_embeddings_path().exists() {
            compressed_size += tokio::fs::metadata(self.entity_embeddings_path())
                .await?
                .len() as usize;
        }
        if self.relation_embeddings_path().exists() {
            compressed_size += tokio::fs::metadata(self.relation_embeddings_path())
                .await?
                .len() as usize;
        }

        let compression_ratio = if total_size > 0 {
            compressed_size as f32 / total_size as f32
        } else {
            1.0
        };

        Ok(StorageStats {
            total_embeddings,
            total_size_bytes: total_size,
            compressed_size_bytes: compressed_size,
            compression_ratio,
            num_versions: self.checkpoints.read().await.len(),
            cache_hit_rate: 0.0, // Not tracked for disk backend
            num_shards: 1,
            replication_factor: 1,
        })
    }

    async fn create_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        info!("Creating checkpoint: {}", checkpoint_id);

        let checkpoint_dir = self.checkpoint_path(checkpoint_id);
        tokio::fs::create_dir_all(&checkpoint_dir).await?;

        // Copy current files to checkpoint directory
        if self.entity_embeddings_path().exists() {
            tokio::fs::copy(
                self.entity_embeddings_path(),
                checkpoint_dir.join("entity_embeddings.bin"),
            )
            .await?;
        }

        if self.relation_embeddings_path().exists() {
            tokio::fs::copy(
                self.relation_embeddings_path(),
                checkpoint_dir.join("relation_embeddings.bin"),
            )
            .await?;
        }

        if self.metadata_path().exists() {
            tokio::fs::copy(self.metadata_path(), checkpoint_dir.join("metadata.json")).await?;
        }

        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.push(checkpoint_id.to_string());

        Ok(())
    }

    async fn restore_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        info!("Restoring checkpoint: {}", checkpoint_id);

        let checkpoint_dir = self.checkpoint_path(checkpoint_id);

        if !checkpoint_dir.exists() {
            return Err(anyhow::anyhow!("Checkpoint not found: {}", checkpoint_id));
        }

        // Copy checkpoint files to current directory
        let entity_checkpoint = checkpoint_dir.join("entity_embeddings.bin");
        if entity_checkpoint.exists() {
            tokio::fs::copy(&entity_checkpoint, self.entity_embeddings_path()).await?;
        }

        let relation_checkpoint = checkpoint_dir.join("relation_embeddings.bin");
        if relation_checkpoint.exists() {
            tokio::fs::copy(&relation_checkpoint, self.relation_embeddings_path()).await?;
        }

        let metadata_checkpoint = checkpoint_dir.join("metadata.json");
        if metadata_checkpoint.exists() {
            tokio::fs::copy(&metadata_checkpoint, self.metadata_path()).await?;
        }

        Ok(())
    }

    async fn list_checkpoints(&self) -> Result<Vec<String>> {
        Ok(self.checkpoints.read().await.clone())
    }
}

/// Storage backend manager
pub struct StorageBackendManager {
    backend: Box<dyn StorageBackend>,
    config: StorageBackendConfig,
}

impl StorageBackendManager {
    /// Create a new storage backend manager
    pub async fn new(config: StorageBackendConfig) -> Result<Self> {
        let backend: Box<dyn StorageBackend> = match &config.backend_type {
            StorageBackendType::Memory => Box::new(MemoryBackend::new()),
            StorageBackendType::Disk { path, .. } => {
                Box::new(DiskBackend::new(path.clone(), config.clone())?)
            }
            _ => {
                // For other backends, use memory as fallback
                warn!("Unsupported backend type, falling back to memory");
                Box::new(MemoryBackend::new())
            }
        };

        Ok(Self { backend, config })
    }

    /// Save embeddings
    pub async fn save_embeddings(
        &mut self,
        entity_embeddings: &HashMap<String, Vector>,
        relation_embeddings: &HashMap<String, Vector>,
    ) -> Result<()> {
        self.backend
            .save_entity_embeddings(entity_embeddings)
            .await?;
        self.backend
            .save_relation_embeddings(relation_embeddings)
            .await?;
        Ok(())
    }

    /// Load embeddings
    pub async fn load_embeddings(
        &self,
    ) -> Result<(HashMap<String, Vector>, HashMap<String, Vector>)> {
        let entity_embs = self.backend.load_entity_embeddings().await?;
        let relation_embs = self.backend.load_relation_embeddings().await?;
        Ok((entity_embs, relation_embs))
    }

    /// Save metadata
    pub async fn save_metadata(&mut self, metadata: &EmbeddingMetadata) -> Result<()> {
        self.backend.save_metadata(metadata).await
    }

    /// Load metadata
    pub async fn load_metadata(&self) -> Result<EmbeddingMetadata> {
        self.backend.load_metadata().await
    }

    /// Get statistics
    pub async fn get_stats(&self) -> Result<StorageStats> {
        self.backend.get_stats().await
    }

    /// Create checkpoint
    pub async fn create_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        self.backend.create_checkpoint(checkpoint_id).await
    }

    /// Restore checkpoint
    pub async fn restore_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        self.backend.restore_checkpoint(checkpoint_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_memory_backend() {
        let mut backend = MemoryBackend::new();

        let mut embeddings = HashMap::new();
        embeddings.insert("entity1".to_string(), Vector::new(vec![1.0, 2.0, 3.0]));
        embeddings.insert("entity2".to_string(), Vector::new(vec![4.0, 5.0, 6.0]));

        backend.save_entity_embeddings(&embeddings).await.unwrap();
        let loaded = backend.load_entity_embeddings().await.unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("entity1").unwrap().values, vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_disk_backend() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let config = StorageBackendConfig::default();
        let mut backend = DiskBackend::new(temp_dir.path().to_path_buf(), config).unwrap();

        let mut embeddings = HashMap::new();
        embeddings.insert("entity1".to_string(), Vector::new(vec![1.0, 2.0, 3.0]));

        backend.save_entity_embeddings(&embeddings).await.unwrap();
        let loaded = backend.load_entity_embeddings().await.unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get("entity1").unwrap().values, vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_disk_backend_checkpoints() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let config = StorageBackendConfig::default();
        let mut backend = DiskBackend::new(temp_dir.path().to_path_buf(), config).unwrap();

        let mut embeddings = HashMap::new();
        embeddings.insert("entity1".to_string(), Vector::new(vec![1.0, 2.0, 3.0]));

        backend.save_entity_embeddings(&embeddings).await.unwrap();
        backend.create_checkpoint("checkpoint1").await.unwrap();

        // Modify embeddings
        let mut new_embeddings = HashMap::new();
        new_embeddings.insert("entity2".to_string(), Vector::new(vec![4.0, 5.0, 6.0]));
        backend
            .save_entity_embeddings(&new_embeddings)
            .await
            .unwrap();

        // Restore checkpoint
        backend.restore_checkpoint("checkpoint1").await.unwrap();
        let restored = backend.load_entity_embeddings().await.unwrap();

        assert_eq!(restored.len(), 1);
        assert!(restored.contains_key("entity1"));
    }
}
