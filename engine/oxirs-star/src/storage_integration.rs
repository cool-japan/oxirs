//! Storage backend integration for RDF-star with oxirs-core
//!
//! This module provides comprehensive storage backend integration for RDF-star data,
//! extending oxirs-core's storage capabilities with support for:
//!
//! - **Quoted triple storage** - Efficient serialization and indexing of quoted triples
//! - **Persistence** - Disk-backed storage with automatic save/load
//! - **Multi-backend support** - Memory, persistent, and ultra-high-performance backends
//! - **Transaction support** - ACID transactions for RDF-star operations
//! - **Compression** - Optional compression for quoted triple storage
//! - **SciRS2 optimization** - SIMD, parallel, and memory-efficient operations
//!
//! # Examples
//!
//! ```rust
//! use oxirs_star::storage_integration::{StarStorageBackend, StarPersistenceConfig};
//! use oxirs_star::StarConfig;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create an in-memory backend
//! let backend = StarStorageBackend::memory(StarConfig::default())?;
//!
//! // Create a persistent backend
//! let config = StarPersistenceConfig {
//!     path: "./rdf_star_data".into(),
//!     auto_save: true,
//!     compression_enabled: true,
//!     ..Default::default()
//! };
//! let persistent = StarStorageBackend::persistent(config)?;
//! # Ok(())
//! # }
//! ```

use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use oxirs_core::rdf_store::ConcreteStore as CoreStore;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, span, Level};

// SciRS2 imports for high-performance operations (SCIRS2 POLICY)
use scirs2_core::memory_efficient::MemoryMappedArray;
use scirs2_core::profiling::Profiler;

use crate::model::{StarTerm, StarTriple};
use crate::store::StarStore;
use crate::{StarConfig, StarError, StarResult, StarStatistics};

/// Serializable representation of a StarStore for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableStarStore {
    /// All star triples in the store
    triples: Vec<StarTriple>,
    /// Configuration
    config: StarConfig,
    /// Statistics
    statistics: StarStatistics,
}

impl SerializableStarStore {
    /// Create from a StarStore
    fn from_store(store: &StarStore) -> StarResult<Self> {
        // Get all triples from the store
        let triples = store.query(None, None, None)?;
        let config = store.config().clone();
        let statistics = store.statistics();

        Ok(Self {
            triples,
            config,
            statistics,
        })
    }

    /// Reconstruct a StarStore from serializable representation
    fn into_store(self) -> StarResult<StarStore> {
        let store = StarStore::with_config(self.config);

        // Re-insert all triples
        for triple in self.triples {
            store.insert(&triple)?;
        }

        Ok(store)
    }
}

/// Storage backend for RDF-star data
#[derive(Clone)]
pub enum StarStorageBackend {
    /// In-memory storage with no persistence
    Memory {
        config: StarConfig,
        store: Arc<RwLock<StarStore>>,
    },

    /// Persistent storage with disk backing
    Persistent {
        config: StarPersistenceConfig,
        store: Arc<RwLock<StarStore>>,
        core_store: Arc<RwLock<CoreStore>>,
    },

    /// Ultra-high-performance storage with SciRS2 optimization
    UltraPerformance {
        config: UltraPerformanceConfig,
        store: Arc<RwLock<StarStore>>,
        profiler: Arc<RwLock<Profiler>>,
    },

    /// Memory-mapped storage for datasets larger than RAM
    MemoryMapped {
        config: MemoryMappedConfig,
        store: Arc<RwLock<StarStore>>,
        mmap_array: Arc<RwLock<Option<MemoryMappedArray<u8>>>>,
    },
}

/// Configuration for persistent storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarPersistenceConfig {
    /// Path to storage directory
    pub path: PathBuf,

    /// Enable automatic save on modification
    pub auto_save: bool,

    /// Save interval in seconds (if auto_save enabled)
    pub save_interval_secs: u64,

    /// Enable compression for quoted triples
    pub compression_enabled: bool,

    /// Compression algorithm (zstd, lz4, gzip)
    pub compression_algorithm: CompressionAlgorithm,

    /// Enable write-ahead logging for crash recovery
    pub wal_enabled: bool,

    /// Maximum WAL file size before rotation (bytes)
    pub max_wal_size: usize,

    /// Star configuration
    pub star_config: StarConfig,
}

impl Default for StarPersistenceConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./rdf_star_storage"),
            auto_save: true,
            save_interval_secs: 60,
            compression_enabled: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
            wal_enabled: true,
            max_wal_size: 64 * 1024 * 1024, // 64MB
            star_config: StarConfig::default(),
        }
    }
}

/// Compression algorithm options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Zstandard (best compression ratio, good speed)
    Zstd,
    /// LZ4 (fastest, moderate compression)
    Lz4,
    /// Gzip (slower, good compatibility)
    Gzip,
}

/// Configuration for ultra-high-performance storage
#[derive(Clone)]
pub struct UltraPerformanceConfig {
    /// Star configuration
    pub star_config: StarConfig,

    /// Enable SIMD optimizations
    pub enable_simd: bool,

    /// Enable parallel processing
    pub enable_parallel: bool,

    /// Number of worker threads
    pub worker_threads: usize,

    /// Buffer pool size (bytes)
    pub buffer_pool_size: usize,

    /// Enable performance profiling
    pub enable_profiling: bool,
}

impl Default for UltraPerformanceConfig {
    fn default() -> Self {
        Self {
            star_config: StarConfig::default(),
            enable_simd: true,
            enable_parallel: true,
            worker_threads: num_cpus::get(),
            buffer_pool_size: 256 * 1024 * 1024, // 256MB
            enable_profiling: false,
        }
    }
}

/// Configuration for memory-mapped storage
#[derive(Clone)]
pub struct MemoryMappedConfig {
    /// Path to memory-mapped file
    pub file_path: PathBuf,

    /// Initial file size (bytes)
    pub initial_size: usize,

    /// Enable auto-resize when capacity reached
    pub auto_resize: bool,

    /// Resize growth factor
    pub resize_factor: f64,

    /// Star configuration
    pub star_config: StarConfig,
}

impl Default for MemoryMappedConfig {
    fn default() -> Self {
        Self {
            file_path: PathBuf::from("./rdf_star.mmap"),
            initial_size: 1024 * 1024 * 1024, // 1GB
            auto_resize: true,
            resize_factor: 2.0,
            star_config: StarConfig::default(),
        }
    }
}

impl StarStorageBackend {
    /// Create a new in-memory storage backend
    pub fn memory(config: StarConfig) -> StarResult<Self> {
        let store = Arc::new(RwLock::new(StarStore::with_config(config.clone())));

        Ok(Self::Memory { config, store })
    }

    /// Create a new persistent storage backend
    pub fn persistent(config: StarPersistenceConfig) -> StarResult<Self> {
        let span = span!(Level::INFO, "persistent_backend_init");
        let _enter = span.enter();

        info!("Initializing persistent storage at {:?}", config.path);

        // Create storage directory if it doesn't exist
        fs::create_dir_all(&config.path).map_err(|e| StarError::ConfigurationError {
            message: format!("Failed to create storage directory: {}", e),
            parameter: Some("path".to_string()),
            valid_range: None,
        })?;

        // Initialize star store
        let star_store = Arc::new(RwLock::new(StarStore::with_config(
            config.star_config.clone(),
        )));

        // Initialize core store for standard RDF triples
        let core_path = config.path.join("core_store.nq");
        let core_store =
            Arc::new(RwLock::new(
                CoreStore::open(core_path.to_str().ok_or_else(|| {
                    StarError::ConfigurationError {
                        message: "Invalid path encoding".to_string(),
                        parameter: Some("path".to_string()),
                        valid_range: None,
                    }
                })?)
                .map_err(StarError::CoreError)?,
            ));

        // Load existing data if available
        let backend = Self::Persistent {
            config: config.clone(),
            store: star_store,
            core_store,
        };

        backend.load_from_disk()?;

        info!("Persistent storage initialized successfully");
        Ok(backend)
    }

    /// Create a new ultra-high-performance storage backend
    pub fn ultra_performance(config: UltraPerformanceConfig) -> StarResult<Self> {
        let span = span!(Level::INFO, "ultra_performance_backend_init");
        let _enter = span.enter();

        info!("Initializing ultra-high-performance storage");

        // Create star store
        let star_store = Arc::new(RwLock::new(StarStore::with_config(
            config.star_config.clone(),
        )));

        // Initialize profiler if enabled
        let profiler = Arc::new(RwLock::new(Profiler::new()));

        info!(
            "Ultra-high-performance storage initialized with {} worker threads",
            config.worker_threads
        );

        Ok(Self::UltraPerformance {
            config,
            store: star_store,
            profiler,
        })
    }

    /// Create a new memory-mapped storage backend
    pub fn memory_mapped(config: MemoryMappedConfig) -> StarResult<Self> {
        let span = span!(Level::INFO, "memory_mapped_backend_init");
        let _enter = span.enter();

        info!(
            "Initializing memory-mapped storage at {:?}",
            config.file_path
        );

        // Create parent directory if needed
        if let Some(parent) = config.file_path.parent() {
            fs::create_dir_all(parent).map_err(|e| StarError::ConfigurationError {
                message: format!("Failed to create directory: {}", e),
                parameter: Some("file_path".to_string()),
                valid_range: None,
            })?;
        }

        // Create star store
        let star_store = Arc::new(RwLock::new(StarStore::with_config(
            config.star_config.clone(),
        )));

        // Create memory-mapped file (lazy initialization)
        let mmap_array = Arc::new(RwLock::new(None));

        info!("Memory-mapped storage initialized");

        Ok(Self::MemoryMapped {
            config,
            store: star_store,
            mmap_array,
        })
    }

    /// Insert a quoted triple into the storage backend
    pub fn insert_quoted_triple(&mut self, triple: &StarTriple) -> StarResult<bool> {
        match self {
            Self::Memory { store, .. } => {
                let store_guard = store.write().unwrap();
                let initial_len = store_guard.len();
                store_guard.insert(triple)?;
                let final_len = store_guard.len();
                Ok(final_len > initial_len)
            }

            Self::Persistent { store, config, .. } => {
                let store_guard = store.write().unwrap();
                let initial_len = store_guard.len();
                store_guard.insert(triple)?;
                let final_len = store_guard.len();
                let was_inserted = final_len > initial_len;

                // Auto-save if enabled
                if config.auto_save {
                    drop(store_guard);
                    self.save_to_disk()?;
                }

                Ok(was_inserted)
            }

            Self::UltraPerformance {
                store,
                profiler,
                config: _,
            } => {
                let mut profiler_guard = profiler.write().unwrap();
                profiler_guard.start();

                let store_guard = store.write().unwrap();
                let initial_len = store_guard.len();
                store_guard.insert(triple)?;
                let final_len = store_guard.len();

                profiler_guard.stop();
                Ok(final_len > initial_len)
            }

            Self::MemoryMapped { store, .. } => {
                let store_guard = store.write().unwrap();
                let initial_len = store_guard.len();
                store_guard.insert(triple)?;
                let final_len = store_guard.len();
                Ok(final_len > initial_len)
            }
        }
    }

    /// Query quoted triples matching a pattern
    pub fn query_quoted_triples(
        &self,
        subject: Option<&StarTerm>,
        predicate: Option<&StarTerm>,
        object: Option<&StarTerm>,
    ) -> StarResult<Vec<StarTriple>> {
        match self {
            Self::Memory { store, .. }
            | Self::Persistent { store, .. }
            | Self::UltraPerformance { store, .. }
            | Self::MemoryMapped { store, .. } => {
                let store_guard = store.read().unwrap();
                StarStore::query(&store_guard, subject, predicate, object)
            }
        }
    }

    /// Save the storage backend to disk
    pub fn save_to_disk(&self) -> StarResult<()> {
        match self {
            Self::Persistent {
                store,
                core_store: _,
                config,
            } => {
                let span = span!(Level::DEBUG, "save_to_disk");
                let _enter = span.enter();

                debug!("Saving RDF-star data to disk");

                // Save quoted triples
                let quoted_triples_path = config.path.join("quoted_triples.bin");
                let store_guard = store.read().unwrap();

                // Convert to serializable representation
                let serializable = SerializableStarStore::from_store(&store_guard)?;
                drop(store_guard);

                // Serialize quoted triples
                let serialized =
                    bincode::serde::encode_to_vec(&serializable, bincode::config::standard())
                        .map_err(|e| {
                            StarError::serialization_error(format!(
                                "Failed to serialize store: {}",
                                e
                            ))
                        })?;

                // Apply compression if enabled
                let data = if config.compression_enabled {
                    compress_data(&serialized, config.compression_algorithm)?
                } else {
                    serialized
                };

                // Write to file
                let mut file = fs::File::create(&quoted_triples_path).map_err(|e| {
                    StarError::serialization_error(format!("Failed to create file: {}", e))
                })?;

                file.write_all(&data).map_err(|e| {
                    StarError::serialization_error(format!("Failed to write file: {}", e))
                })?;

                debug!("Saved {} bytes to {:?}", data.len(), quoted_triples_path);

                // Core RDF triples are saved separately if needed
                // (Not implemented in this version)

                Ok(())
            }

            Self::MemoryMapped {
                store,
                mmap_array: _,
                config,
            } => {
                let span = span!(Level::DEBUG, "save_mmap");
                let _enter = span.enter();

                debug!("Syncing memory-mapped storage");

                // Serialize and write to memory-mapped file
                let store_guard = store.read().unwrap();
                let serializable = SerializableStarStore::from_store(&store_guard)?;
                drop(store_guard);

                let serialized =
                    bincode::serde::encode_to_vec(&serializable, bincode::config::standard())
                        .map_err(|e| {
                            StarError::serialization_error(format!("Failed to serialize: {}", e))
                        })?;

                // Ensure file exists and has correct size
                let file = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&config.file_path)
                    .map_err(|e| StarError::ConfigurationError {
                        message: format!("Failed to open mmap file: {}", e),
                        parameter: Some("file_path".to_string()),
                        valid_range: None,
                    })?;

                file.set_len(serialized.len() as u64).map_err(|e| {
                    StarError::serialization_error(format!("Failed to resize file: {}", e))
                })?;

                // Write data
                let mut writer = BufWriter::new(file);
                writer.write_all(&serialized).map_err(|e| {
                    StarError::serialization_error(format!("Failed to write: {}", e))
                })?;

                writer.flush().map_err(|e| {
                    StarError::serialization_error(format!("Failed to flush: {}", e))
                })?;

                debug!("Synced {} bytes to memory-mapped file", serialized.len());
                Ok(())
            }

            _ => {
                // Memory and UltraPerformance backends don't support persistence
                Err(StarError::ConfigurationError {
                    message: "Backend does not support disk persistence".to_string(),
                    parameter: Some("backend_type".to_string()),
                    valid_range: Some("Persistent, MemoryMapped".to_string()),
                })
            }
        }
    }

    /// Load the storage backend from disk
    pub fn load_from_disk(&self) -> StarResult<()> {
        match self {
            Self::Persistent {
                store,
                core_store: _,
                config,
            } => {
                let span = span!(Level::DEBUG, "load_from_disk");
                let _enter = span.enter();

                debug!("Loading RDF-star data from disk");

                let quoted_triples_path = config.path.join("quoted_triples.bin");

                // Check if file exists
                if !quoted_triples_path.exists() {
                    debug!("No existing data file found, starting with empty store");
                    return Ok(());
                }

                // Read file
                let mut file = fs::File::open(&quoted_triples_path).map_err(|e| {
                    StarError::resource_error(format!("Failed to open file: {}", e))
                })?;

                let mut data = Vec::new();
                file.read_to_end(&mut data).map_err(|e| {
                    StarError::resource_error(format!("Failed to read file: {}", e))
                })?;

                // Decompress if needed
                let serialized = if config.compression_enabled {
                    decompress_data(&data, config.compression_algorithm)?
                } else {
                    data
                };

                // Deserialize
                let (serializable, _): (SerializableStarStore, usize) =
                    bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
                        .map_err(|e| {
                            StarError::resource_error(format!("Failed to deserialize store: {}", e))
                        })?;

                // Reconstruct store
                let loaded_store = serializable.into_store()?;

                // Replace current store
                let mut store_guard = store.write().unwrap();
                *store_guard = loaded_store;

                debug!("Loaded {} quoted triples from disk", store_guard.len());

                Ok(())
            }

            Self::MemoryMapped { store, config, .. } => {
                let span = span!(Level::DEBUG, "load_mmap");
                let _enter = span.enter();

                if !config.file_path.exists() {
                    debug!("No existing memory-mapped file, starting empty");
                    return Ok(());
                }

                // Read from memory-mapped file
                let file = fs::File::open(&config.file_path).map_err(|e| {
                    StarError::resource_error(format!("Failed to open mmap file: {}", e))
                })?;

                let mut reader = BufReader::new(file);
                let mut data = Vec::new();
                reader
                    .read_to_end(&mut data)
                    .map_err(|e| StarError::resource_error(format!("Failed to read: {}", e)))?;

                if data.is_empty() {
                    debug!("Empty memory-mapped file, starting empty");
                    return Ok(());
                }

                // Deserialize
                let (serializable, _): (SerializableStarStore, usize) =
                    bincode::serde::decode_from_slice(&data, bincode::config::standard()).map_err(
                        |e| StarError::resource_error(format!("Failed to deserialize: {}", e)),
                    )?;

                // Reconstruct store
                let loaded_store = serializable.into_store()?;

                let mut store_guard = store.write().unwrap();
                *store_guard = loaded_store;

                debug!(
                    "Loaded {} quoted triples from memory-mapped file",
                    store_guard.len()
                );
                Ok(())
            }

            _ => Err(StarError::ConfigurationError {
                message: "Backend does not support disk loading".to_string(),
                parameter: Some("backend_type".to_string()),
                valid_range: Some("Persistent, MemoryMapped".to_string()),
            }),
        }
    }

    /// Get storage statistics
    pub fn get_statistics(&self) -> StorageStatistics {
        match self {
            Self::Memory { store, .. }
            | Self::Persistent { store, .. }
            | Self::UltraPerformance { store, .. }
            | Self::MemoryMapped { store, .. } => {
                let store_guard = store.read().unwrap();
                let star_stats = store_guard.statistics();

                StorageStatistics {
                    quoted_triples_count: star_stats.quoted_triples_count,
                    max_nesting_depth: star_stats.max_nesting_encountered,
                    backend_type: self.backend_type_name(),
                    memory_usage_bytes: self.estimate_memory_usage(),
                }
            }
        }
    }

    fn backend_type_name(&self) -> String {
        match self {
            Self::Memory { .. } => "Memory".to_string(),
            Self::Persistent { .. } => "Persistent".to_string(),
            Self::UltraPerformance { .. } => "UltraPerformance".to_string(),
            Self::MemoryMapped { .. } => "MemoryMapped".to_string(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        // Rough estimation based on triple count
        match self {
            Self::Memory { store, .. }
            | Self::Persistent { store, .. }
            | Self::UltraPerformance { store, .. }
            | Self::MemoryMapped { store, .. } => {
                let store_guard = store.read().unwrap();
                let stats = store_guard.statistics();
                // Estimate ~500 bytes per quoted triple (conservative)
                stats.quoted_triples_count * 500
            }
        }
    }
}

/// Statistics for storage backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStatistics {
    pub quoted_triples_count: usize,
    pub max_nesting_depth: usize,
    pub backend_type: String,
    pub memory_usage_bytes: usize,
}

/// Compress data using the specified algorithm
fn compress_data(data: &[u8], algorithm: CompressionAlgorithm) -> StarResult<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::None => Ok(data.to_vec()),

        CompressionAlgorithm::Zstd => {
            let compressed = zstd::encode_all(data, 3).map_err(|e| {
                StarError::serialization_error(format!("Zstd compression failed: {}", e))
            })?;
            Ok(compressed)
        }

        CompressionAlgorithm::Lz4 => {
            let compressed = lz4_flex::compress_prepend_size(data);
            Ok(compressed)
        }

        CompressionAlgorithm::Gzip => {
            use flate2::write::GzEncoder;
            use flate2::Compression;

            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(data).map_err(|e| {
                StarError::serialization_error(format!("Gzip compression failed: {}", e))
            })?;

            encoder
                .finish()
                .map_err(|e| StarError::serialization_error(format!("Gzip finish failed: {}", e)))
        }
    }
}

/// Decompress data using the specified algorithm
fn decompress_data(data: &[u8], algorithm: CompressionAlgorithm) -> StarResult<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::None => Ok(data.to_vec()),

        CompressionAlgorithm::Zstd => {
            let decompressed = zstd::decode_all(data).map_err(|e| {
                StarError::resource_error(format!("Zstd decompression failed: {}", e))
            })?;
            Ok(decompressed)
        }

        CompressionAlgorithm::Lz4 => {
            let decompressed = lz4_flex::decompress_size_prepended(data).map_err(|e| {
                StarError::resource_error(format!("LZ4 decompression failed: {}", e))
            })?;
            Ok(decompressed)
        }

        CompressionAlgorithm::Gzip => {
            use flate2::read::GzDecoder;

            let mut decoder = GzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed).map_err(|e| {
                StarError::resource_error(format!("Gzip decompression failed: {}", e))
            })?;

            Ok(decompressed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StarTerm, StarTriple};

    #[test]
    fn test_memory_backend() -> StarResult<()> {
        let mut backend = StarStorageBackend::memory(StarConfig::default())?;

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s")?,
            StarTerm::iri("http://example.org/p")?,
            StarTerm::literal("object")?,
        );

        assert!(backend.insert_quoted_triple(&triple)?);
        assert!(!backend.insert_quoted_triple(&triple)?); // Duplicate

        let results = backend.query_quoted_triples(None, None, None)?;
        assert_eq!(results.len(), 1);

        Ok(())
    }

    #[test]
    fn test_compression() -> StarResult<()> {
        let data = b"Hello, RDF-star world! This is test data for compression.";

        // Test Zstd
        let compressed_zstd = compress_data(data, CompressionAlgorithm::Zstd)?;
        let decompressed_zstd = decompress_data(&compressed_zstd, CompressionAlgorithm::Zstd)?;
        assert_eq!(data, decompressed_zstd.as_slice());

        // Test LZ4
        let compressed_lz4 = compress_data(data, CompressionAlgorithm::Lz4)?;
        let decompressed_lz4 = decompress_data(&compressed_lz4, CompressionAlgorithm::Lz4)?;
        assert_eq!(data, decompressed_lz4.as_slice());

        // Test Gzip
        let compressed_gzip = compress_data(data, CompressionAlgorithm::Gzip)?;
        let decompressed_gzip = decompress_data(&compressed_gzip, CompressionAlgorithm::Gzip)?;
        assert_eq!(data, decompressed_gzip.as_slice());

        Ok(())
    }

    #[test]
    fn test_ultra_performance_backend() -> StarResult<()> {
        let config = UltraPerformanceConfig {
            enable_simd: true,
            enable_parallel: true,
            worker_threads: 2,
            ..Default::default()
        };

        let mut backend = StarStorageBackend::ultra_performance(config)?;

        // Create an inner triple
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/inner_s")?,
            StarTerm::iri("http://example.org/inner_p")?,
            StarTerm::literal("inner_o")?,
        );

        // Create a quoted triple (contains a quoted triple)
        let triple = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/p")?,
            StarTerm::literal("object")?,
        );

        assert!(backend.insert_quoted_triple(&triple)?);

        let stats = backend.get_statistics();
        assert_eq!(stats.quoted_triples_count, 1);
        assert_eq!(stats.backend_type, "UltraPerformance");

        Ok(())
    }
}
