//! Index persistence with compression
//!
//! This module provides serialization and deserialization of vector indices
//! with support for compression, versioning, and incremental updates.

use crate::hnsw::{HnswConfig, HnswIndex};
use crate::Vector;
use anyhow::{anyhow, Result};
use oxicode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Index persistence format version
const PERSISTENCE_VERSION: u32 = 1;

/// Magic number for index files (OxVe = OxiRS Vector)
const MAGIC_NUMBER: &[u8; 4] = b"OxVe";

/// Compression algorithm for index persistence
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum CompressionAlgorithm {
    /// No compression (fastest, largest)
    None,
    /// Zstd compression (balanced)
    Zstd { level: i32 },
    /// High compression (slowest, smallest)
    ZstdMax,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::Zstd { level: 3 } // Fast compression by default
    }
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct PersistenceConfig {
    /// Compression algorithm to use
    pub compression: CompressionAlgorithm,
    /// Include metadata in persistence
    pub include_metadata: bool,
    /// Validate data integrity on load
    pub validate_on_load: bool,
    /// Enable incremental persistence
    pub incremental: bool,
    /// Checkpoint interval for incremental persistence (in operations)
    pub checkpoint_interval: usize,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            compression: CompressionAlgorithm::default(),
            include_metadata: true,
            validate_on_load: true,
            incremental: false,
            checkpoint_interval: 10000,
        }
    }
}

/// Serializable index header
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
struct IndexHeader {
    version: u32,
    compression: CompressionAlgorithm,
    node_count: usize,
    dimension: usize,
    config: HnswConfig,
    timestamp: u64,
    checksum: u64,
}

/// Serializable node data
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
struct SerializableNode {
    uri: String,
    vector_data: Vec<f32>,
    connections: Vec<Vec<usize>>,
    level: usize,
}

/// Persistence manager for HNSW indices
pub struct PersistenceManager {
    config: PersistenceConfig,
}

impl PersistenceManager {
    /// Create a new persistence manager
    pub fn new(config: PersistenceConfig) -> Self {
        Self { config }
    }

    /// Save HNSW index to disk
    pub fn save_index<P: AsRef<Path>>(&self, index: &HnswIndex, path: P) -> Result<()> {
        let path = path.as_ref();
        tracing::info!("Saving HNSW index to {:?}", path);

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::new(file);

        // Write magic number
        writer.write_all(MAGIC_NUMBER)?;

        // Create header
        let header = IndexHeader {
            version: PERSISTENCE_VERSION,
            compression: self.config.compression,
            node_count: index.len(),
            dimension: if let Some(node) = index.nodes().first() {
                node.vector.dimensions
            } else {
                0
            },
            config: index.config().clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            checksum: 0, // Will be calculated
        };

        // Serialize header
        let header_bytes = oxicode::serde::encode_to_vec(&header, oxicode::config::standard())
            .map_err(|e| anyhow!("Failed to serialize header: {}", e))?;
        let header_len = header_bytes.len() as u32;
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(&header_bytes)?;

        // Serialize nodes
        let nodes = self.serialize_nodes(index)?;

        // Compress if needed
        let data = match self.config.compression {
            CompressionAlgorithm::None => nodes,
            CompressionAlgorithm::Zstd { level } => zstd::encode_all(&nodes[..], level)?,
            CompressionAlgorithm::ZstdMax => zstd::encode_all(&nodes[..], 21)?,
        };

        // Write data length and data
        let data_len = data.len() as u64;
        writer.write_all(&data_len.to_le_bytes())?;
        writer.write_all(&data)?;

        // Write URI mapping
        let uri_mapping =
            oxicode::serde::encode_to_vec(index.uri_to_id(), oxicode::config::standard())
                .map_err(|e| anyhow!("Failed to serialize URI mapping: {}", e))?;
        let mapping_len = uri_mapping.len() as u32;
        writer.write_all(&mapping_len.to_le_bytes())?;
        writer.write_all(&uri_mapping)?;

        // Write entry point
        let entry_point =
            oxicode::serde::encode_to_vec(&index.entry_point(), oxicode::config::standard())
                .map_err(|e| anyhow!("Failed to serialize entry point: {}", e))?;
        writer.write_all(&entry_point)?;

        writer.flush()?;

        tracing::info!(
            "Successfully saved HNSW index with {} nodes (compression: {:?})",
            index.len(),
            self.config.compression
        );

        Ok(())
    }

    /// Load HNSW index from disk
    pub fn load_index<P: AsRef<Path>>(&self, path: P) -> Result<HnswIndex> {
        let path = path.as_ref();
        tracing::info!("Loading HNSW index from {:?}", path);

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Verify magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC_NUMBER {
            return Err(anyhow!("Invalid index file format"));
        }

        // Read header
        let mut header_len_bytes = [0u8; 4];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u32::from_le_bytes(header_len_bytes) as usize;

        let mut header_bytes = vec![0u8; header_len];
        reader.read_exact(&mut header_bytes)?;
        let (header, _): (IndexHeader, _) =
            oxicode::serde::decode_from_slice(&header_bytes, oxicode::config::standard())
                .map_err(|e| anyhow!("Failed to deserialize header: {}", e))?;

        // Verify version
        if header.version != PERSISTENCE_VERSION {
            return Err(anyhow!(
                "Unsupported index version: {} (expected {})",
                header.version,
                PERSISTENCE_VERSION
            ));
        }

        // Read data length
        let mut data_len_bytes = [0u8; 8];
        reader.read_exact(&mut data_len_bytes)?;
        let data_len = u64::from_le_bytes(data_len_bytes) as usize;

        // Read and decompress data
        let mut compressed_data = vec![0u8; data_len];
        reader.read_exact(&mut compressed_data)?;

        let nodes_data = match header.compression {
            CompressionAlgorithm::None => compressed_data,
            CompressionAlgorithm::Zstd { .. } | CompressionAlgorithm::ZstdMax => {
                zstd::decode_all(&compressed_data[..])?
            }
        };

        // Read URI mapping
        let mut mapping_len_bytes = [0u8; 4];
        reader.read_exact(&mut mapping_len_bytes)?;
        let mapping_len = u32::from_le_bytes(mapping_len_bytes) as usize;

        let mut mapping_bytes = vec![0u8; mapping_len];
        reader.read_exact(&mut mapping_bytes)?;
        let (uri_mapping, _): (std::collections::HashMap<String, usize>, _) =
            oxicode::serde::decode_from_slice(&mapping_bytes, oxicode::config::standard())
                .map_err(|e| anyhow!("Failed to deserialize URI mapping: {}", e))?;

        // Read entry point
        let mut entry_point_bytes = Vec::new();
        reader.read_to_end(&mut entry_point_bytes)?;
        let (entry_point, _): (Option<usize>, _) =
            oxicode::serde::decode_from_slice(&entry_point_bytes, oxicode::config::standard())
                .map_err(|e| anyhow!("Failed to deserialize entry point: {}", e))?;

        // Reconstruct index
        let mut index = HnswIndex::new(header.config)?;
        self.deserialize_nodes(&nodes_data, &mut index)?;

        // Restore URI mapping
        *index.uri_to_id_mut() = uri_mapping;

        // Restore entry point
        index.set_entry_point(entry_point);

        // Validate if requested
        if self.config.validate_on_load {
            self.validate_index(&index)?;
        }

        tracing::info!("Successfully loaded HNSW index with {} nodes", index.len());

        Ok(index)
    }

    /// Serialize nodes to bytes
    fn serialize_nodes(&self, index: &HnswIndex) -> Result<Vec<u8>> {
        let serializable_nodes: Vec<SerializableNode> = index
            .nodes()
            .iter()
            .map(|node| SerializableNode {
                uri: node.uri.clone(),
                vector_data: node.vector.as_f32(),
                connections: node
                    .connections
                    .iter()
                    .map(|set| set.iter().copied().collect())
                    .collect(),
                level: node.level(),
            })
            .collect();

        oxicode::serde::encode_to_vec(&serializable_nodes, oxicode::config::standard())
            .map_err(|e| anyhow!("Failed to serialize nodes: {}", e))
    }

    /// Deserialize nodes from bytes
    fn deserialize_nodes(&self, data: &[u8], index: &mut HnswIndex) -> Result<()> {
        let (serializable_nodes, _): (Vec<SerializableNode>, _) =
            oxicode::serde::decode_from_slice(data, oxicode::config::standard())
                .map_err(|e| anyhow!("Failed to deserialize nodes: {}", e))?;

        for node_data in serializable_nodes {
            let vector = Vector::new(node_data.vector_data);
            let mut node = crate::hnsw::Node::new(node_data.uri, vector, node_data.level);

            // Restore connections
            for (level, connections) in node_data.connections.into_iter().enumerate() {
                for conn_id in connections {
                    node.add_connection(level, conn_id);
                }
            }

            index.nodes_mut().push(node);
        }

        Ok(())
    }

    /// Validate index integrity
    fn validate_index(&self, index: &HnswIndex) -> Result<()> {
        tracing::debug!("Validating index integrity");

        // Check that all connections are valid
        for (node_id, node) in index.nodes().iter().enumerate() {
            for level in 0..=node.level() {
                if let Some(connections) = node.get_connections(level) {
                    for &conn_id in connections {
                        if conn_id >= index.len() {
                            return Err(anyhow!(
                                "Invalid connection: node {} has connection to non-existent node {}",
                                node_id,
                                conn_id
                            ));
                        }
                    }
                }
            }
        }

        // Check URI mapping consistency
        for (uri, &node_id) in index.uri_to_id() {
            if node_id >= index.len() {
                return Err(anyhow!(
                    "Invalid URI mapping: {} points to non-existent node {}",
                    uri,
                    node_id
                ));
            }

            let actual_uri = &index.nodes()[node_id].uri;
            if uri != actual_uri {
                return Err(anyhow!(
                    "URI mapping mismatch: expected '{}', found '{}'",
                    uri,
                    actual_uri
                ));
            }
        }

        // Check entry point
        if let Some(entry_id) = index.entry_point() {
            if entry_id >= index.len() {
                return Err(anyhow!(
                    "Invalid entry point: {} (index has {} nodes)",
                    entry_id,
                    index.len()
                ));
            }
        }

        tracing::debug!("Index validation passed");
        Ok(())
    }

    /// Create a snapshot of the index
    pub fn create_snapshot<P: AsRef<Path>>(&self, index: &HnswIndex, path: P) -> Result<()> {
        let path = path.as_ref();
        let snapshot_path = path.with_extension(format!(
            "snapshot.{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ));

        self.save_index(index, snapshot_path)?;
        Ok(())
    }

    /// Estimate compressed size
    pub fn estimate_compressed_size(&self, index: &HnswIndex) -> Result<usize> {
        let nodes = self.serialize_nodes(index)?;

        let compressed_size = match self.config.compression {
            CompressionAlgorithm::None => nodes.len(),
            CompressionAlgorithm::Zstd { level } => zstd::encode_all(&nodes[..], level)?.len(),
            CompressionAlgorithm::ZstdMax => zstd::encode_all(&nodes[..], 21)?.len(),
        };

        Ok(compressed_size)
    }
}

/// Incremental persistence manager
pub struct IncrementalPersistence {
    config: PersistenceConfig,
    operation_count: usize,
    last_checkpoint: std::time::Instant,
}

impl IncrementalPersistence {
    pub fn new(config: PersistenceConfig) -> Self {
        Self {
            config,
            operation_count: 0,
            last_checkpoint: std::time::Instant::now(),
        }
    }

    /// Record an operation
    pub fn record_operation(&mut self) {
        self.operation_count += 1;
    }

    /// Check if checkpoint is needed
    pub fn needs_checkpoint(&self) -> bool {
        self.operation_count >= self.config.checkpoint_interval
    }

    /// Create checkpoint
    pub fn checkpoint<P: AsRef<Path>>(&mut self, index: &HnswIndex, base_path: P) -> Result<()> {
        if !self.needs_checkpoint() {
            return Ok(());
        }

        let manager = PersistenceManager::new(self.config.clone());
        manager.create_snapshot(index, base_path)?;

        self.operation_count = 0;
        self.last_checkpoint = std::time::Instant::now();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswConfig;
    use crate::Vector;
    use std::env::temp_dir;

    #[test]
    fn test_save_and_load_index() {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(config).unwrap();

        // Add some vectors
        for i in 0..10 {
            let vec = Vector::new(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
            index.add_vector(format!("vec_{}", i), vec).unwrap();
        }

        // Save index
        let mut temp_path = temp_dir();
        temp_path.push("test_hnsw_index.bin");

        let persistence_config = PersistenceConfig::default();
        let manager = PersistenceManager::new(persistence_config);

        manager.save_index(&index, &temp_path).unwrap();

        // Load index
        let loaded_index = manager.load_index(&temp_path).unwrap();

        assert_eq!(loaded_index.len(), 10);
        assert_eq!(loaded_index.uri_to_id().len(), 10);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_compression() {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(config).unwrap();

        // Add vectors
        for i in 0..50 {
            let vec = Vector::new(vec![i as f32; 128]);
            index.add_vector(format!("vec_{}", i), vec).unwrap();
        }

        let mut temp_path = temp_dir();
        temp_path.push("test_compressed_index.bin");

        // Test with compression
        let compressed_config = PersistenceConfig {
            compression: CompressionAlgorithm::Zstd { level: 3 },
            ..Default::default()
        };
        let compressed_manager = PersistenceManager::new(compressed_config);
        compressed_manager.save_index(&index, &temp_path).unwrap();

        let compressed_size = std::fs::metadata(&temp_path).unwrap().len();

        // Test without compression
        let uncompressed_config = PersistenceConfig {
            compression: CompressionAlgorithm::None,
            ..Default::default()
        };
        let uncompressed_manager = PersistenceManager::new(uncompressed_config);

        let mut temp_path2 = temp_dir();
        temp_path2.push("test_uncompressed_index.bin");
        uncompressed_manager
            .save_index(&index, &temp_path2)
            .unwrap();

        let uncompressed_size = std::fs::metadata(&temp_path2).unwrap().len();

        // Compressed should be smaller
        assert!(compressed_size < uncompressed_size);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
        std::fs::remove_file(temp_path2).ok();
    }

    #[test]
    fn test_validation() {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(config).unwrap();

        for i in 0..5 {
            let vec = Vector::new(vec![i as f32, 0.0, 0.0]);
            index.add_vector(format!("vec_{}", i), vec).unwrap();
        }

        let persistence_config = PersistenceConfig {
            validate_on_load: true,
            ..Default::default()
        };
        let manager = PersistenceManager::new(persistence_config);

        // Validation should pass
        manager.validate_index(&index).unwrap();
    }
}
