//! Storage backends for DiskANN
//!
//! Provides abstractions for storing vectors and graph structures on disk
//! with support for memory-mapped I/O and buffered access.
//!
//! ## Storage Layout
//! - Vectors: Raw f32 arrays or PQ-compressed codes
//! - Graph: Adjacency lists with neighbor IDs
//! - Metadata: Index configuration and statistics
//!
//! ## Backends
//! - **DiskStorage**: Standard file I/O with buffering
//! - **MemoryMappedStorage**: Memory-mapped files for fast access
//! - **CachedStorage**: Hybrid with LRU caching

use crate::diskann::config::DiskAnnConfig;
use crate::diskann::graph::VamanaGraph;
use crate::diskann::types::{DiskAnnError, DiskAnnResult, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Storage backend trait
pub trait StorageBackend: Send + Sync {
    /// Write a vector to storage
    fn write_vector(&mut self, vector_id: &VectorId, vector: &[f32]) -> DiskAnnResult<()>;

    /// Read a vector from storage
    fn read_vector(&self, vector_id: &VectorId) -> DiskAnnResult<Vec<f32>>;

    /// Write graph structure
    fn write_graph(&mut self, graph: &VamanaGraph) -> DiskAnnResult<()>;

    /// Read graph structure
    fn read_graph(&self) -> DiskAnnResult<VamanaGraph>;

    /// Write metadata
    fn write_metadata(&mut self, metadata: &StorageMetadata) -> DiskAnnResult<()>;

    /// Read metadata
    fn read_metadata(&self) -> DiskAnnResult<StorageMetadata>;

    /// Delete all data
    fn clear(&mut self) -> DiskAnnResult<()>;

    /// Flush any pending writes
    fn flush(&mut self) -> DiskAnnResult<()>;

    /// Get storage size in bytes
    fn size(&self) -> DiskAnnResult<u64>;
}

/// Storage metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetadata {
    pub version: String,
    pub config: DiskAnnConfig,
    pub num_vectors: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl StorageMetadata {
    pub fn new(config: DiskAnnConfig) -> Self {
        let now = chrono::Utc::now();
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            config,
            num_vectors: 0,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn update_timestamp(&mut self) {
        self.updated_at = chrono::Utc::now();
    }
}

/// Standard disk storage with buffered I/O
#[derive(Debug)]
pub struct DiskStorage {
    base_path: PathBuf,
    vector_file: Option<PathBuf>,
    graph_file: Option<PathBuf>,
    metadata_file: Option<PathBuf>,
    dimension: usize,
    vector_cache: HashMap<VectorId, Vec<f32>>,
    cache_limit: usize,
}

impl DiskStorage {
    /// Create new disk storage at given path
    pub fn new<P: AsRef<Path>>(base_path: P, dimension: usize) -> DiskAnnResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path).map_err(|e| DiskAnnError::IoError {
                message: format!("Failed to create directory: {}", e),
            })?;
        }

        let vector_file = Some(base_path.join("vectors.bin"));
        let graph_file = Some(base_path.join("graph.bin"));
        let metadata_file = Some(base_path.join("metadata.json"));

        Ok(Self {
            base_path,
            vector_file,
            graph_file,
            metadata_file,
            dimension,
            vector_cache: HashMap::new(),
            cache_limit: 1000,
        })
    }

    /// Set cache limit (number of vectors to keep in memory)
    pub fn with_cache_limit(mut self, limit: usize) -> Self {
        self.cache_limit = limit;
        self
    }

    /// Get vector file path
    pub fn vector_file_path(&self) -> &Option<PathBuf> {
        &self.vector_file
    }

    /// Get graph file path
    pub fn graph_file_path(&self) -> &Option<PathBuf> {
        &self.graph_file
    }

    /// Evict old entries from cache if needed
    fn evict_cache_if_needed(&mut self) {
        if self.vector_cache.len() > self.cache_limit {
            // Simple eviction: remove first entry
            if let Some(key) = self.vector_cache.keys().next().cloned() {
                self.vector_cache.remove(&key);
            }
        }
    }
}

impl Clone for DiskStorage {
    fn clone(&self) -> Self {
        Self {
            base_path: self.base_path.clone(),
            vector_file: self.vector_file.clone(),
            graph_file: self.graph_file.clone(),
            metadata_file: self.metadata_file.clone(),
            dimension: self.dimension,
            vector_cache: HashMap::new(), // Don't clone cache
            cache_limit: self.cache_limit,
        }
    }
}

impl StorageBackend for DiskStorage {
    fn write_vector(&mut self, vector_id: &VectorId, vector: &[f32]) -> DiskAnnResult<()> {
        if vector.len() != self.dimension {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        // Add to cache
        self.vector_cache.insert(vector_id.clone(), vector.to_vec());
        self.evict_cache_if_needed();

        // Append to vector file
        if let Some(path) = &self.vector_file {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| DiskAnnError::IoError {
                    message: format!("Failed to open vector file: {}", e),
                })?;

            let mut writer = BufWriter::new(file);

            // Write vector ID length and ID
            let id_bytes = vector_id.as_bytes();
            writer
                .write_all(&(id_bytes.len() as u32).to_le_bytes())
                .map_err(|e| DiskAnnError::IoError {
                    message: format!("Failed to write vector ID length: {}", e),
                })?;
            writer
                .write_all(id_bytes)
                .map_err(|e| DiskAnnError::IoError {
                    message: format!("Failed to write vector ID: {}", e),
                })?;

            // Write vector data
            for &value in vector {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| DiskAnnError::IoError {
                        message: format!("Failed to write vector data: {}", e),
                    })?;
            }

            writer.flush().map_err(|e| DiskAnnError::IoError {
                message: format!("Failed to flush vector file: {}", e),
            })?;
        }

        Ok(())
    }

    fn read_vector(&self, vector_id: &VectorId) -> DiskAnnResult<Vec<f32>> {
        // Check cache first
        if let Some(vector) = self.vector_cache.get(vector_id) {
            return Ok(vector.clone());
        }

        // Read from disk
        if let Some(path) = &self.vector_file {
            if !path.exists() {
                return Err(DiskAnnError::VectorNotFound {
                    id: vector_id.clone(),
                });
            }

            let file = File::open(path).map_err(|e| DiskAnnError::IoError {
                message: format!("Failed to open vector file: {}", e),
            })?;
            let mut reader = BufReader::new(file);

            // Sequential scan (inefficient, but simple for now)
            loop {
                // Read ID length
                let mut id_len_bytes = [0u8; 4];
                if reader.read_exact(&mut id_len_bytes).is_err() {
                    break; // End of file
                }
                let id_len = u32::from_le_bytes(id_len_bytes) as usize;

                // Read ID
                let mut id_bytes = vec![0u8; id_len];
                reader
                    .read_exact(&mut id_bytes)
                    .map_err(|e| DiskAnnError::IoError {
                        message: format!("Failed to read vector ID: {}", e),
                    })?;
                let id = String::from_utf8(id_bytes).map_err(|e| DiskAnnError::IoError {
                    message: format!("Invalid UTF-8 in vector ID: {}", e),
                })?;

                // Read vector data
                let mut vector = vec![0.0f32; self.dimension];
                for value in &mut vector {
                    let mut bytes = [0u8; 4];
                    reader
                        .read_exact(&mut bytes)
                        .map_err(|e| DiskAnnError::IoError {
                            message: format!("Failed to read vector data: {}", e),
                        })?;
                    *value = f32::from_le_bytes(bytes);
                }

                if &id == vector_id {
                    return Ok(vector);
                }
            }

            Err(DiskAnnError::VectorNotFound {
                id: vector_id.clone(),
            })
        } else {
            Err(DiskAnnError::VectorNotFound {
                id: vector_id.clone(),
            })
        }
    }

    fn write_graph(&mut self, graph: &VamanaGraph) -> DiskAnnResult<()> {
        if let Some(path) = &self.graph_file {
            let file = File::create(path).map_err(|e| DiskAnnError::IoError {
                message: format!("Failed to create graph file: {}", e),
            })?;

            let mut writer = BufWriter::new(file);
            oxicode::serde::encode_into_std_write(graph, &mut writer, oxicode::config::standard())?;
        }
        Ok(())
    }

    fn read_graph(&self) -> DiskAnnResult<VamanaGraph> {
        if let Some(path) = &self.graph_file {
            if !path.exists() {
                return Err(DiskAnnError::StorageError {
                    message: "Graph file does not exist".to_string(),
                });
            }

            let file = File::open(path).map_err(|e| DiskAnnError::IoError {
                message: format!("Failed to open graph file: {}", e),
            })?;

            let mut reader = BufReader::new(file);
            let (graph, _) =
                oxicode::serde::decode_from_std_read(&mut reader, oxicode::config::standard())?;
            Ok(graph)
        } else {
            Err(DiskAnnError::StorageError {
                message: "Graph file path not set".to_string(),
            })
        }
    }

    fn write_metadata(&mut self, metadata: &StorageMetadata) -> DiskAnnResult<()> {
        if let Some(path) = &self.metadata_file {
            let mut file = File::create(path).map_err(|e| DiskAnnError::IoError {
                message: format!("Failed to create metadata file: {}", e),
            })?;

            serde_json::to_writer_pretty(&mut file, metadata).map_err(|e| {
                DiskAnnError::SerializationError {
                    message: format!("Failed to serialize metadata: {}", e),
                }
            })?;

            // Explicitly sync to disk
            file.sync_all().map_err(|e| DiskAnnError::IoError {
                message: format!("Failed to sync metadata file: {}", e),
            })?;
        }
        Ok(())
    }

    fn read_metadata(&self) -> DiskAnnResult<StorageMetadata> {
        if let Some(path) = &self.metadata_file {
            if !path.exists() {
                return Err(DiskAnnError::StorageError {
                    message: "Metadata file does not exist".to_string(),
                });
            }

            let file = File::open(path).map_err(|e| DiskAnnError::IoError {
                message: format!("Failed to open metadata file: {}", e),
            })?;

            let metadata =
                serde_json::from_reader(file).map_err(|e| DiskAnnError::SerializationError {
                    message: format!("Failed to deserialize metadata: {}", e),
                })?;

            Ok(metadata)
        } else {
            Err(DiskAnnError::StorageError {
                message: "Metadata file path not set".to_string(),
            })
        }
    }

    fn clear(&mut self) -> DiskAnnResult<()> {
        self.vector_cache.clear();

        if let Some(path) = &self.vector_file {
            if path.exists() {
                std::fs::remove_file(path).map_err(|e| DiskAnnError::IoError {
                    message: format!("Failed to remove vector file: {}", e),
                })?;
            }
        }

        if let Some(path) = &self.graph_file {
            if path.exists() {
                std::fs::remove_file(path).map_err(|e| DiskAnnError::IoError {
                    message: format!("Failed to remove graph file: {}", e),
                })?;
            }
        }

        if let Some(path) = &self.metadata_file {
            if path.exists() {
                std::fs::remove_file(path).map_err(|e| DiskAnnError::IoError {
                    message: format!("Failed to remove metadata file: {}", e),
                })?;
            }
        }

        Ok(())
    }

    fn flush(&mut self) -> DiskAnnResult<()> {
        // All writes are immediately flushed in this implementation
        Ok(())
    }

    fn size(&self) -> DiskAnnResult<u64> {
        let mut total_size = 0u64;

        if let Some(path) = &self.vector_file {
            if path.exists() {
                total_size += std::fs::metadata(path)
                    .map_err(|e| DiskAnnError::IoError {
                        message: format!("Failed to get vector file size: {}", e),
                    })?
                    .len();
            }
        }

        if let Some(path) = &self.graph_file {
            if path.exists() {
                total_size += std::fs::metadata(path)
                    .map_err(|e| DiskAnnError::IoError {
                        message: format!("Failed to get graph file size: {}", e),
                    })?
                    .len();
            }
        }

        if let Some(path) = &self.metadata_file {
            if path.exists() {
                total_size += std::fs::metadata(path)
                    .map_err(|e| DiskAnnError::IoError {
                        message: format!("Failed to get metadata file size: {}", e),
                    })?
                    .len();
            }
        }

        Ok(total_size)
    }
}

/// Memory-mapped storage (stub for future implementation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMappedStorage {
    base_path: PathBuf,
    dimension: usize,
}

impl MemoryMappedStorage {
    pub fn new<P: AsRef<Path>>(base_path: P, dimension: usize) -> DiskAnnResult<Self> {
        Ok(Self {
            base_path: base_path.as_ref().to_path_buf(),
            dimension,
        })
    }
}

impl StorageBackend for MemoryMappedStorage {
    fn write_vector(&mut self, _vector_id: &VectorId, _vector: &[f32]) -> DiskAnnResult<()> {
        Err(DiskAnnError::StorageError {
            message: "MemoryMappedStorage not yet implemented".to_string(),
        })
    }

    fn read_vector(&self, _vector_id: &VectorId) -> DiskAnnResult<Vec<f32>> {
        Err(DiskAnnError::StorageError {
            message: "MemoryMappedStorage not yet implemented".to_string(),
        })
    }

    fn write_graph(&mut self, _graph: &VamanaGraph) -> DiskAnnResult<()> {
        Err(DiskAnnError::StorageError {
            message: "MemoryMappedStorage not yet implemented".to_string(),
        })
    }

    fn read_graph(&self) -> DiskAnnResult<VamanaGraph> {
        Err(DiskAnnError::StorageError {
            message: "MemoryMappedStorage not yet implemented".to_string(),
        })
    }

    fn write_metadata(&mut self, _metadata: &StorageMetadata) -> DiskAnnResult<()> {
        Err(DiskAnnError::StorageError {
            message: "MemoryMappedStorage not yet implemented".to_string(),
        })
    }

    fn read_metadata(&self) -> DiskAnnResult<StorageMetadata> {
        Err(DiskAnnError::StorageError {
            message: "MemoryMappedStorage not yet implemented".to_string(),
        })
    }

    fn clear(&mut self) -> DiskAnnResult<()> {
        Ok(())
    }

    fn flush(&mut self) -> DiskAnnResult<()> {
        Ok(())
    }

    fn size(&self) -> DiskAnnResult<u64> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::config::PruningStrategy;
    use std::env;

    fn temp_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        env::temp_dir().join(format!(
            "diskann_storage_test_{}_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            id
        ))
    }

    #[test]
    fn test_disk_storage_vector_write_read() {
        let dir = temp_dir();
        let mut storage = DiskStorage::new(&dir, 3).unwrap();

        let vector = vec![1.0, 2.0, 3.0];
        storage.write_vector(&"vec1".to_string(), &vector).unwrap();

        let read_vector = storage.read_vector(&"vec1".to_string()).unwrap();
        assert_eq!(read_vector, vector);

        storage.clear().unwrap();
    }

    #[test]
    fn test_disk_storage_dimension_mismatch() {
        let dir = temp_dir();
        let mut storage = DiskStorage::new(&dir, 3).unwrap();

        let vector = vec![1.0, 2.0]; // Wrong dimension
        let result = storage.write_vector(&"vec1".to_string(), &vector);

        assert!(result.is_err());
        storage.clear().unwrap();
    }

    #[test]
    fn test_disk_storage_graph() {
        let dir = temp_dir();
        std::fs::remove_dir_all(&dir).ok(); // Clean up if exists
        let mut storage = DiskStorage::new(&dir, 3).unwrap();

        let mut graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.2);
        graph.add_node("v1".to_string()).unwrap();
        graph.add_node("v2".to_string()).unwrap();

        storage.write_graph(&graph).unwrap();
        let read_graph = storage.read_graph().unwrap();

        assert_eq!(read_graph.num_nodes(), 2);
        storage.clear().unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_disk_storage_metadata() {
        let dir = temp_dir();
        std::fs::remove_dir_all(&dir).ok(); // Clean up if exists
        let mut storage = DiskStorage::new(&dir, 128).unwrap();

        let config = DiskAnnConfig::default_config(128);
        let metadata = StorageMetadata::new(config);

        storage.write_metadata(&metadata).unwrap();
        let read_metadata = storage.read_metadata().unwrap();

        assert_eq!(read_metadata.config.dimension, 128);
        storage.clear().unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_disk_storage_size() {
        let dir = temp_dir();
        let mut storage = DiskStorage::new(&dir, 3).unwrap();

        let initial_size = storage.size().unwrap();
        assert_eq!(initial_size, 0);

        let vector = vec![1.0, 2.0, 3.0];
        storage.write_vector(&"vec1".to_string(), &vector).unwrap();

        let after_write = storage.size().unwrap();
        assert!(after_write > initial_size);

        storage.clear().unwrap();
    }

    #[test]
    fn test_disk_storage_cache() {
        let dir = temp_dir();
        std::fs::remove_dir_all(&dir).ok(); // Clean up if exists
        let mut storage = DiskStorage::new(&dir, 3).unwrap().with_cache_limit(2);

        storage
            .write_vector(&"v1".to_string(), &[1.0, 2.0, 3.0])
            .unwrap();
        storage
            .write_vector(&"v2".to_string(), &[4.0, 5.0, 6.0])
            .unwrap();
        storage
            .write_vector(&"v3".to_string(), &[7.0, 8.0, 9.0])
            .unwrap();

        // Cache should have at most 2 entries
        assert!(storage.vector_cache.len() <= 2);

        storage.clear().unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_vector_not_found() {
        let dir = temp_dir();
        let storage = DiskStorage::new(&dir, 3).unwrap();

        let result = storage.read_vector(&"nonexistent".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_storage_clear() {
        let dir = temp_dir();
        std::fs::remove_dir_all(&dir).ok(); // Clean up if exists
        let mut storage = DiskStorage::new(&dir, 3).unwrap();

        storage
            .write_vector(&"v1".to_string(), &[1.0, 2.0, 3.0])
            .unwrap();

        // Verify file was created
        let vector_file = storage.vector_file.as_ref().unwrap().clone();
        assert!(
            vector_file.exists(),
            "Vector file should exist after write: {:?}",
            vector_file
        );

        storage.clear().unwrap();
        assert!(
            !vector_file.exists(),
            "Vector file should not exist after clear: {:?}",
            vector_file
        );

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }
}
