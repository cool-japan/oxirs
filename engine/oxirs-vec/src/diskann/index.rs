//! Main DiskANN index
//!
//! Provides the primary user-facing API for DiskANN, orchestrating all components:
//! - Graph structure (Vamana graph)
//! - Storage backend (disk I/O)
//! - Search algorithm (beam search)
//! - Index building (incremental construction)
//!
//! ## Example
//! ```rust,ignore
//! use oxirs_vec::diskann::{DiskAnnIndex, DiskAnnConfig};
//!
//! // Create index
//! let config = DiskAnnConfig::default_config(128);
//! let mut index = DiskAnnIndex::new(config, "/path/to/index")?;
//!
//! // Add vectors
//! index.add("vec1", vec![...])?;
//! index.add("vec2", vec![...])?;
//!
//! // Build and save
//! index.build()?;
//!
//! // Search
//! let results = index.search(&query, 10)?;
//! ```

use crate::diskann::builder::{DiskAnnBuildStats, DiskAnnBuilder};
use crate::diskann::config::DiskAnnConfig;
use crate::diskann::graph::VamanaGraph;
use crate::diskann::search::{BeamSearch, SearchResult};
use crate::diskann::storage::{DiskStorage, StorageBackend};
use crate::diskann::types::{DiskAnnError, DiskAnnResult, NodeId, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// DiskANN index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub version: String,
    pub num_vectors: usize,
    pub dimension: usize,
    pub config: DiskAnnConfig,
}

impl IndexMetadata {
    pub fn new(config: DiskAnnConfig, num_vectors: usize) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            num_vectors,
            dimension: config.dimension,
            config,
        }
    }
}

/// Main DiskANN index
pub struct DiskAnnIndex {
    config: DiskAnnConfig,
    graph: Arc<RwLock<Option<VamanaGraph>>>,
    vectors: Arc<RwLock<HashMap<VectorId, Vec<f32>>>>,
    storage: Arc<RwLock<Box<dyn StorageBackend>>>,
    metadata: Arc<RwLock<IndexMetadata>>,
    is_built: Arc<RwLock<bool>>,
}

impl DiskAnnIndex {
    /// Create a new DiskANN index with given configuration and storage path
    pub fn new<P: AsRef<Path>>(config: DiskAnnConfig, storage_path: P) -> DiskAnnResult<Self> {
        config
            .validate()
            .map_err(|msg| DiskAnnError::InvalidConfiguration { message: msg })?;

        let storage: Box<dyn StorageBackend> =
            Box::new(DiskStorage::new(storage_path, config.dimension)?);

        let metadata = IndexMetadata::new(config.clone(), 0);

        Ok(Self {
            config: config.clone(),
            graph: Arc::new(RwLock::new(None)),
            vectors: Arc::new(RwLock::new(HashMap::new())),
            storage: Arc::new(RwLock::new(storage)),
            metadata: Arc::new(RwLock::new(metadata)),
            is_built: Arc::new(RwLock::new(false)),
        })
    }

    /// Load existing index from storage
    pub fn load<P: AsRef<Path>>(storage_path: P) -> DiskAnnResult<Self> {
        let storage: Box<dyn StorageBackend> = Box::new(DiskStorage::new(&storage_path, 1)?); // Temp dimension

        let storage_lock = Arc::new(RwLock::new(storage));

        // Read metadata
        let storage_metadata = {
            let storage_guard = storage_lock
                .read()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            storage_guard.read_metadata()?
        };

        let config = storage_metadata.config.clone();

        // Recreate storage with correct dimension
        let storage: Box<dyn StorageBackend> =
            Box::new(DiskStorage::new(&storage_path, config.dimension)?);

        let storage_lock = Arc::new(RwLock::new(storage));

        // Read graph
        let graph = {
            let storage_guard = storage_lock
                .read()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            storage_guard.read_graph()?
        };

        let metadata = IndexMetadata::new(config.clone(), storage_metadata.num_vectors);

        Ok(Self {
            config,
            graph: Arc::new(RwLock::new(Some(graph))),
            vectors: Arc::new(RwLock::new(HashMap::new())),
            storage: storage_lock,
            metadata: Arc::new(RwLock::new(metadata)),
            is_built: Arc::new(RwLock::new(true)),
        })
    }

    /// Add a vector to the index (before building)
    pub fn add(&mut self, vector_id: VectorId, vector: Vec<f32>) -> DiskAnnResult<()> {
        if vector.len() != self.config.dimension {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dimension,
                actual: vector.len(),
            });
        }

        let is_built = *self
            .is_built
            .read()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;

        if is_built {
            return Err(DiskAnnError::InternalError {
                message: "Cannot add vectors after index is built".to_string(),
            });
        }

        let mut vectors = self
            .vectors
            .write()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;

        vectors.insert(vector_id, vector);

        Ok(())
    }

    /// Build the index from added vectors
    pub fn build(&mut self) -> DiskAnnResult<DiskAnnBuildStats> {
        let vectors = {
            let vectors_guard = self
                .vectors
                .read()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            vectors_guard.clone()
        };

        if vectors.is_empty() {
            return Err(DiskAnnError::InternalError {
                message: "No vectors to build index from".to_string(),
            });
        }

        // Create builder
        let storage = {
            let storage_guard = self
                .storage
                .read()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            let disk_storage = DiskStorage::new(
                storage_guard
                    .size()
                    .map(|_| PathBuf::from("."))
                    .unwrap_or_else(|_| PathBuf::from(".")),
                self.config.dimension,
            )?;
            Box::new(disk_storage) as Box<dyn StorageBackend>
        };

        let mut builder = DiskAnnBuilder::new(self.config.clone())?.with_storage(storage);

        // Add all vectors
        let vector_list: Vec<_> = vectors.into_iter().collect();
        builder.add_vectors_batch(vector_list)?;

        // Get stats before finalization
        let stats = builder.stats().clone();

        // Finalize and get graph
        let graph = builder.finalize()?;

        // Update index state
        {
            let mut graph_guard = self
                .graph
                .write()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            *graph_guard = Some(graph);
        }

        {
            let mut is_built_guard = self
                .is_built
                .write()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            *is_built_guard = true;
        }

        {
            let mut metadata_guard = self
                .metadata
                .write()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            metadata_guard.num_vectors = stats.num_vectors;
        }

        Ok(stats)
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> DiskAnnResult<SearchResult> {
        if query.len() != self.config.dimension {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        let is_built = *self
            .is_built
            .read()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;

        if !is_built {
            return Err(DiskAnnError::IndexNotBuilt);
        }

        let graph = self
            .graph
            .read()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;

        let graph_ref = graph.as_ref().ok_or(DiskAnnError::IndexNotBuilt)?;

        let beam_search = BeamSearch::new(self.config.search_beam_width);

        // Create distance function
        let storage_guard = self
            .storage
            .read()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;

        let distance_fn = |node_id: NodeId| {
            if let Some(node) = graph_ref.get_node(node_id) {
                if let Ok(vector) = storage_guard.read_vector(&node.vector_id) {
                    return Self::compute_distance(query, &vector);
                }
            }
            f32::MAX
        };

        beam_search.search(graph_ref, &distance_fn, k)
    }

    /// Get vector by ID
    pub fn get(&self, vector_id: &VectorId) -> DiskAnnResult<Vec<f32>> {
        let storage_guard = self
            .storage
            .read()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;

        storage_guard.read_vector(vector_id)
    }

    /// Get index metadata
    pub fn metadata(&self) -> DiskAnnResult<IndexMetadata> {
        let metadata_guard = self
            .metadata
            .read()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;

        Ok(metadata_guard.clone())
    }

    /// Get number of vectors in index
    pub fn num_vectors(&self) -> DiskAnnResult<usize> {
        let metadata_guard = self
            .metadata
            .read()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;

        Ok(metadata_guard.num_vectors)
    }

    /// Check if index is built
    pub fn is_built(&self) -> bool {
        self.is_built.read().map(|guard| *guard).unwrap_or(false)
    }

    /// Clear the index
    pub fn clear(&mut self) -> DiskAnnResult<()> {
        {
            let mut graph_guard = self
                .graph
                .write()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            *graph_guard = None;
        }

        {
            let mut vectors_guard = self
                .vectors
                .write()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            vectors_guard.clear();
        }

        {
            let mut storage_guard = self
                .storage
                .write()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            storage_guard.clear()?;
        }

        {
            let mut is_built_guard = self
                .is_built
                .write()
                .map_err(|_| DiskAnnError::ConcurrentModification)?;
            *is_built_guard = false;
        }

        Ok(())
    }

    /// Compute L2 distance between two vectors
    fn compute_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl Default for DiskAnnIndex {
    fn default() -> Self {
        Self::new(
            DiskAnnConfig::default(),
            std::env::temp_dir().join("diskann_default"),
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_dir() -> PathBuf {
        env::temp_dir().join(format!(
            "diskann_index_test_{}",
            chrono::Utc::now().timestamp()
        ))
    }

    #[test]
    fn test_index_create() {
        let dir = temp_dir();
        let config = DiskAnnConfig::default_config(3);
        let index = DiskAnnIndex::new(config, &dir).unwrap();

        assert_eq!(index.num_vectors().unwrap(), 0);
        assert!(!index.is_built());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_index_add_and_build() {
        let dir = temp_dir();
        let config = DiskAnnConfig::default_config(3);
        let mut index = DiskAnnIndex::new(config, &dir).unwrap();

        index.add("v1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        index.add("v2".to_string(), vec![0.0, 1.0, 0.0]).unwrap();
        index.add("v3".to_string(), vec![0.0, 0.0, 1.0]).unwrap();

        let stats = index.build().unwrap();

        assert_eq!(stats.num_vectors, 3);
        assert!(index.is_built());
        assert_eq!(index.num_vectors().unwrap(), 3);

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_index_search() {
        let dir = temp_dir();
        let config = DiskAnnConfig::default_config(3);
        let mut index = DiskAnnIndex::new(config, &dir).unwrap();

        index.add("v1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        index.add("v2".to_string(), vec![0.0, 1.0, 0.0]).unwrap();
        index.add("v3".to_string(), vec![0.0, 0.0, 1.0]).unwrap();

        index.build().unwrap();

        let query = vec![1.0, 0.1, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert!(!results.neighbors.is_empty());
        assert!(results.neighbors.len() <= 2);

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_index_dimension_mismatch() {
        let dir = temp_dir();
        std::fs::remove_dir_all(&dir).ok(); // Clean up if exists
        let config = DiskAnnConfig::default_config(3);
        let mut index = DiskAnnIndex::new(config, &dir).unwrap();

        let result = index.add("v1".to_string(), vec![1.0, 2.0]); // Wrong dimension
        assert!(result.is_err());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_search_before_build() {
        let dir = temp_dir();
        let config = DiskAnnConfig::default_config(3);
        let index = DiskAnnIndex::new(config, &dir).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let result = index.search(&query, 1);

        assert!(result.is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_add_after_build() {
        let dir = temp_dir();
        let config = DiskAnnConfig::default_config(3);
        let mut index = DiskAnnIndex::new(config, &dir).unwrap();

        index.add("v1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        index.build().unwrap();

        let result = index.add("v2".to_string(), vec![0.0, 1.0, 0.0]);
        assert!(result.is_err());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_index_metadata() {
        let dir = temp_dir();
        let config = DiskAnnConfig::default_config(3);
        let mut index = DiskAnnIndex::new(config.clone(), &dir).unwrap();

        index.add("v1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        index.build().unwrap();

        let metadata = index.metadata().unwrap();
        assert_eq!(metadata.num_vectors, 1);
        assert_eq!(metadata.dimension, 3);

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_index_clear() {
        let dir = temp_dir();
        std::fs::remove_dir_all(&dir).ok(); // Clean up if exists
        let config = DiskAnnConfig::default_config(3);
        let mut index = DiskAnnIndex::new(config, &dir).unwrap();

        index.add("v1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        index.build().unwrap();

        assert!(index.is_built());

        index.clear().unwrap();

        assert!(!index.is_built());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_distance_computation() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let distance = DiskAnnIndex::compute_distance(&a, &b);
        assert!((distance - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_empty_build() {
        let dir = temp_dir();
        let config = DiskAnnConfig::default_config(3);
        let mut index = DiskAnnIndex::new(config, &dir).unwrap();

        let result = index.build();
        assert!(result.is_err());

        std::fs::remove_dir_all(dir).ok();
    }
}
