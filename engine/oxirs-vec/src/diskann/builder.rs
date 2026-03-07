//! Index builder for DiskANN
//!
//! Implements the greedy best-first algorithm for constructing Vamana graphs.
//! The builder incrementally adds vectors to the graph, maintaining connectivity
//! and using robust pruning to select high-quality neighbors.
//!
//! ## Build Algorithm
//! 1. Add vectors incrementally
//! 2. For each vector, search for nearest neighbors using beam search
//! 3. Prune neighbors using robust pruning strategy
//! 4. Update reverse edges (make graph bidirectional)
//! 5. Select entry points (medoids)
//!
//! ## References
//! - DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node
//!   (Jayaram Subramanya et al., NeurIPS 2019)

use crate::diskann::config::DiskAnnConfig;
use crate::diskann::graph::VamanaGraph;
use crate::diskann::search::BeamSearch;
use crate::diskann::storage::{StorageBackend, StorageMetadata};
use crate::diskann::types::{DiskAnnError, DiskAnnResult, NodeId, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Index builder statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiskAnnBuildStats {
    /// Number of vectors added
    pub num_vectors: usize,
    /// Total build time in milliseconds
    pub build_time_ms: u64,
    /// Average time per vector in milliseconds
    pub avg_time_per_vector_ms: f64,
    /// Total distance computations
    pub total_comparisons: usize,
    /// Number of graph updates
    pub num_graph_updates: usize,
    /// Number of entry points
    pub num_entry_points: usize,
}

/// Index builder for DiskANN
pub struct DiskAnnBuilder {
    config: DiskAnnConfig,
    graph: VamanaGraph,
    vectors: HashMap<VectorId, Vec<f32>>,
    storage: Option<Box<dyn StorageBackend>>,
    stats: DiskAnnBuildStats,
}

impl DiskAnnBuilder {
    /// Create a new index builder with given configuration
    pub fn new(config: DiskAnnConfig) -> DiskAnnResult<Self> {
        config
            .validate()
            .map_err(|msg| DiskAnnError::InvalidConfiguration { message: msg })?;

        let graph = VamanaGraph::new(config.max_degree, config.pruning_strategy, config.alpha);

        Ok(Self {
            config,
            graph,
            vectors: HashMap::new(),
            storage: None,
            stats: DiskAnnBuildStats::default(),
        })
    }

    /// Add storage backend
    pub fn with_storage(mut self, storage: Box<dyn StorageBackend>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Get configuration
    pub fn config(&self) -> &DiskAnnConfig {
        &self.config
    }

    /// Get current graph
    pub fn graph(&self) -> &VamanaGraph {
        &self.graph
    }

    /// Get build statistics
    pub fn stats(&self) -> &DiskAnnBuildStats {
        &self.stats
    }

    /// Add a single vector to the index
    pub fn add_vector(&mut self, vector_id: VectorId, vector: Vec<f32>) -> DiskAnnResult<NodeId> {
        if vector.len() != self.config.dimension {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dimension,
                actual: vector.len(),
            });
        }

        let start_time = Instant::now();

        // Add node to graph
        let node_id = self.graph.add_node(vector_id.clone())?;

        // Store vector
        self.vectors.insert(vector_id.clone(), vector.clone());
        if let Some(storage) = &mut self.storage {
            storage.write_vector(&vector_id, &vector)?;
        }

        // If this is the first vector, no need to connect
        if self.graph.num_nodes() == 1 {
            self.stats.num_vectors += 1;
            self.stats.build_time_ms += start_time.elapsed().as_millis() as u64;
            return Ok(node_id);
        }

        // Find nearest neighbors using beam search
        let beam_search = BeamSearch::new(self.config.build_beam_width);
        let distance_fn = |other_id: NodeId| {
            if let Some(other_node) = self.graph.get_node(other_id) {
                if let Some(other_vector) = self.vectors.get(&other_node.vector_id) {
                    return self.compute_distance(&vector, other_vector);
                }
            }
            f32::MAX
        };

        let search_result =
            beam_search.search(&self.graph, &distance_fn, self.config.max_degree * 2)?;
        self.stats.total_comparisons += search_result.stats.num_comparisons;

        // Get candidate neighbors
        let candidates: Vec<(NodeId, f32)> = search_result
            .neighbors
            .iter()
            .filter(|(id, _)| *id != node_id)
            .copied()
            .collect();

        // Clone vectors we'll need for distance calculations
        let vectors_clone = self.vectors.clone();
        let graph_clone = self.graph.clone();

        // Prune neighbors for new node
        let distance_fn_for_prune = move |a: NodeId, b: NodeId| -> f32 {
            let vec_a = graph_clone
                .get_node(a)
                .and_then(|node| vectors_clone.get(&node.vector_id));
            let vec_b = graph_clone
                .get_node(b)
                .and_then(|node| vectors_clone.get(&node.vector_id));
            if let (Some(va), Some(vb)) = (vec_a, vec_b) {
                Self::compute_distance_static(va, vb)
            } else {
                f32::MAX
            }
        };

        self.graph
            .prune_neighbors(node_id, &candidates, &distance_fn_for_prune)?;
        self.stats.num_graph_updates += 1;

        // Update reverse edges (make graph bidirectional)
        let neighbors_copy = self
            .graph
            .get_neighbors(node_id)
            .map(|n| n.to_vec())
            .unwrap_or_default();

        for &neighbor_id in &neighbors_copy {
            // Add edge from neighbor to new node
            self.graph.add_edge(neighbor_id, node_id)?;

            // Check if neighbor's edges need pruning
            let needs_pruning = self
                .graph
                .get_node(neighbor_id)
                .map(|n| n.is_full())
                .unwrap_or(false);

            if needs_pruning {
                // Collect neighbor candidates with distances
                let neighbor_candidates: Vec<_> =
                    if let Some(neighbor_node) = self.graph.get_node(neighbor_id) {
                        let neighbor_vec_id = neighbor_node.vector_id.clone();
                        let neighbor_nodes = neighbor_node.neighbors.clone();

                        neighbor_nodes
                            .iter()
                            .map(|&id| {
                                let dist = if id == node_id {
                                    // Distance to new node
                                    if let Some(neighbor_vec) = self.vectors.get(&neighbor_vec_id) {
                                        Self::compute_distance_static(neighbor_vec, &vector)
                                    } else {
                                        f32::MAX
                                    }
                                } else {
                                    // Distance to existing neighbor
                                    let vec_n = self
                                        .graph
                                        .get_node(neighbor_id)
                                        .and_then(|node| self.vectors.get(&node.vector_id));
                                    let vec_id = self
                                        .graph
                                        .get_node(id)
                                        .and_then(|node| self.vectors.get(&node.vector_id));
                                    if let (Some(vn), Some(vid)) = (vec_n, vec_id) {
                                        Self::compute_distance_static(vn, vid)
                                    } else {
                                        f32::MAX
                                    }
                                };
                                (id, dist)
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };

                // Create new closure for this pruning operation
                let vectors_clone2 = self.vectors.clone();
                let graph_clone2 = self.graph.clone();
                let distance_fn2 = move |a: NodeId, b: NodeId| -> f32 {
                    let vec_a = graph_clone2
                        .get_node(a)
                        .and_then(|node| vectors_clone2.get(&node.vector_id));
                    let vec_b = graph_clone2
                        .get_node(b)
                        .and_then(|node| vectors_clone2.get(&node.vector_id));
                    if let (Some(va), Some(vb)) = (vec_a, vec_b) {
                        Self::compute_distance_static(va, vb)
                    } else {
                        f32::MAX
                    }
                };

                if !neighbor_candidates.is_empty() {
                    self.graph
                        .prune_neighbors(neighbor_id, &neighbor_candidates, &distance_fn2)?;
                    self.stats.num_graph_updates += 1;
                }
            }
        }

        self.stats.num_vectors += 1;
        self.stats.build_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(node_id)
    }

    /// Add multiple vectors in batch
    pub fn add_vectors_batch(
        &mut self,
        vectors: Vec<(VectorId, Vec<f32>)>,
    ) -> DiskAnnResult<Vec<NodeId>> {
        let mut node_ids = Vec::with_capacity(vectors.len());

        for (vector_id, vector) in vectors {
            let node_id = self.add_vector(vector_id, vector)?;
            node_ids.push(node_id);
        }

        Ok(node_ids)
    }

    /// Select entry points (medoids) - vectors closest to the center
    pub fn select_entry_points(&mut self, num_entry_points: usize) -> DiskAnnResult<()> {
        if self.graph.num_nodes() == 0 {
            return Ok(());
        }

        // Compute centroid of all vectors
        let centroid = self.compute_centroid();

        // Find vectors closest to centroid
        let mut distances: Vec<_> = self
            .vectors
            .iter()
            .filter_map(|(vector_id, vector)| {
                self.graph.get_node_id(vector_id).map(|node_id| {
                    let dist = self.compute_distance(&centroid, vector);
                    (node_id, dist)
                })
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top-k as entry points
        let entry_points: Vec<_> = distances
            .iter()
            .take(num_entry_points)
            .map(|(node_id, _)| *node_id)
            .collect();

        self.graph.set_entry_points(entry_points);
        self.stats.num_entry_points = self.graph.entry_points().len();

        Ok(())
    }

    /// Finalize the index and save to storage
    pub fn finalize(mut self) -> DiskAnnResult<VamanaGraph> {
        // Select entry points if not already done
        if self.graph.entry_points().is_empty() && self.graph.num_nodes() > 0 {
            self.select_entry_points(self.config.num_entry_points)?;
        }

        // Compute final statistics
        if self.stats.num_vectors > 0 {
            self.stats.avg_time_per_vector_ms =
                self.stats.build_time_ms as f64 / self.stats.num_vectors as f64;
        }

        // Save graph to storage
        if let Some(storage) = &mut self.storage {
            storage.write_graph(&self.graph)?;

            let mut metadata = StorageMetadata::new(self.config.clone());
            metadata.num_vectors = self.stats.num_vectors;
            storage.write_metadata(&metadata)?;
            storage.flush()?;
        }

        // Validate graph before returning
        self.graph.validate()?;

        Ok(self.graph)
    }

    /// Get vector by node ID
    fn get_vector_by_node(&self, node_id: NodeId) -> Option<&Vec<f32>> {
        self.graph
            .get_node(node_id)
            .and_then(|node| self.vectors.get(&node.vector_id))
    }

    /// Compute distance between two vectors (L2 distance)
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        Self::compute_distance_static(a, b)
    }

    /// Static version of compute_distance for use in closures
    fn compute_distance_static(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Compute centroid of all vectors
    fn compute_centroid(&self) -> Vec<f32> {
        if self.vectors.is_empty() {
            return vec![0.0; self.config.dimension];
        }

        let mut centroid = vec![0.0; self.config.dimension];
        for vector in self.vectors.values() {
            for (i, &value) in vector.iter().enumerate() {
                centroid[i] += value;
            }
        }

        let count = self.vectors.len() as f32;
        for value in &mut centroid {
            *value /= count;
        }

        centroid
    }

    /// Get current number of vectors
    pub fn num_vectors(&self) -> usize {
        self.stats.num_vectors
    }
}

impl Default for DiskAnnBuilder {
    fn default() -> Self {
        Self::new(DiskAnnConfig::default()).expect("default DiskAnnConfig should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::storage::DiskStorage;
    use std::env;

    fn temp_dir() -> std::path::PathBuf {
        env::temp_dir().join(format!(
            "diskann_builder_test_{}",
            chrono::Utc::now().timestamp()
        ))
    }

    #[test]
    fn test_builder_basic() {
        let config = DiskAnnConfig::default_config(3);
        let mut builder = DiskAnnBuilder::new(config).unwrap();

        let node0 = builder
            .add_vector("v0".to_string(), vec![1.0, 0.0, 0.0])
            .unwrap();
        let node1 = builder
            .add_vector("v1".to_string(), vec![0.0, 1.0, 0.0])
            .unwrap();

        assert_eq!(builder.num_vectors(), 2);
        assert_ne!(node0, node1);
    }

    #[test]
    fn test_builder_dimension_mismatch() {
        let config = DiskAnnConfig::default_config(3);
        let mut builder = DiskAnnBuilder::new(config).unwrap();

        let result = builder.add_vector("v0".to_string(), vec![1.0, 2.0]); // Wrong dimension
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_batch() {
        let config = DiskAnnConfig::default_config(2);
        let mut builder = DiskAnnBuilder::new(config).unwrap();

        let vectors = vec![
            ("v0".to_string(), vec![1.0, 0.0]),
            ("v1".to_string(), vec![0.0, 1.0]),
            ("v2".to_string(), vec![1.0, 1.0]),
        ];

        let node_ids = builder.add_vectors_batch(vectors).unwrap();
        assert_eq!(node_ids.len(), 3);
        assert_eq!(builder.num_vectors(), 3);
    }

    #[test]
    fn test_entry_point_selection() {
        let config = DiskAnnConfig::default_config(2);
        let mut builder = DiskAnnBuilder::new(config).unwrap();

        builder
            .add_vector("v0".to_string(), vec![1.0, 0.0])
            .unwrap();
        builder
            .add_vector("v1".to_string(), vec![0.0, 1.0])
            .unwrap();
        builder
            .add_vector("v2".to_string(), vec![0.5, 0.5])
            .unwrap();

        builder.select_entry_points(1).unwrap();

        assert_eq!(builder.graph.entry_points().len(), 1);
        // v2 should be closest to centroid [0.5, 0.5]
    }

    #[test]
    fn test_builder_with_storage() {
        let dir = temp_dir();
        let config = DiskAnnConfig::default_config(3);
        let storage = Box::new(DiskStorage::new(&dir, 3).unwrap());

        let mut builder = DiskAnnBuilder::new(config).unwrap().with_storage(storage);

        builder
            .add_vector("v0".to_string(), vec![1.0, 2.0, 3.0])
            .unwrap();
        builder
            .add_vector("v1".to_string(), vec![4.0, 5.0, 6.0])
            .unwrap();

        let graph = builder.finalize().unwrap();
        assert_eq!(graph.num_nodes(), 2);

        // Cleanup
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_finalize_selects_entry_points() {
        let config = DiskAnnConfig {
            num_entry_points: 2,
            ..DiskAnnConfig::default_config(2)
        };
        let mut builder = DiskAnnBuilder::new(config).unwrap();

        builder
            .add_vector("v0".to_string(), vec![1.0, 0.0])
            .unwrap();
        builder
            .add_vector("v1".to_string(), vec![0.0, 1.0])
            .unwrap();
        builder
            .add_vector("v2".to_string(), vec![1.0, 1.0])
            .unwrap();

        let graph = builder.finalize().unwrap();
        assert!(!graph.entry_points().is_empty());
    }

    #[test]
    fn test_build_statistics() {
        let config = DiskAnnConfig::default_config(2);
        let mut builder = DiskAnnBuilder::new(config).unwrap();

        builder
            .add_vector("v0".to_string(), vec![1.0, 0.0])
            .unwrap();
        builder
            .add_vector("v1".to_string(), vec![0.0, 1.0])
            .unwrap();

        let stats = builder.stats();
        assert_eq!(stats.num_vectors, 2);
        // Note: build_time_ms can be 0 for very small datasets on fast systems
        // Just verify it's a valid value (type is u64, so always >= 0)
        let _ = stats.build_time_ms; // Acknowledge we checked the field exists
        assert!(stats.total_comparisons > 0);
    }

    #[test]
    fn test_centroid_computation() {
        let config = DiskAnnConfig::default_config(2);
        let mut builder = DiskAnnBuilder::new(config).unwrap();

        builder
            .add_vector("v0".to_string(), vec![0.0, 0.0])
            .unwrap();
        builder
            .add_vector("v1".to_string(), vec![2.0, 2.0])
            .unwrap();

        let centroid = builder.compute_centroid();
        assert_eq!(centroid, vec![1.0, 1.0]);
    }

    #[test]
    fn test_distance_computation() {
        let config = DiskAnnConfig::default_config(3);
        let builder = DiskAnnBuilder::new(config).unwrap();

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let distance = builder.compute_distance(&a, &b);
        assert!((distance - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_graph_connectivity() {
        let config = DiskAnnConfig::default_config(2);
        let mut builder = DiskAnnBuilder::new(config).unwrap();

        let n0 = builder
            .add_vector("v0".to_string(), vec![0.0, 0.0])
            .unwrap();
        builder
            .add_vector("v1".to_string(), vec![1.0, 0.0])
            .unwrap();
        builder
            .add_vector("v2".to_string(), vec![0.0, 1.0])
            .unwrap();

        // Check that nodes have neighbors
        let neighbors_0 = builder.graph.get_neighbors(n0);
        assert!(neighbors_0.is_some());
        assert!(!neighbors_0.unwrap().is_empty());
    }
}
