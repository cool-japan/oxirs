//! Index management for the OxiRS vector store.
//!
//! Contains the two primary built-in ANN index implementations:
//!
//! - [`FlatIndex`]  — brute-force exact search
//! - [`HNSWIndex`]  — Hierarchical Navigable Small World approximate search
//!
//! IVF, LSH and PQ backends live in the dedicated sibling modules
//! (`ivf_index`, `lsh_index`, `pq_index`) and are re-exported from the
//! `vector_store` facade.

use crate::ai::vector_store_search::compute_similarity;
use crate::ai::vector_store_types::{
    IndexStats, SimilarityItem, SimilarityMetric, VectorData, VectorIndex,
};
use anyhow::Result;
use dashmap::DashMap;
use scirs2_core::random::Random;
use scirs2_core::rngs::StdRng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// FlatIndex — brute-force search
// ---------------------------------------------------------------------------

/// Flat (brute-force) vector index.
///
/// Stores every vector in a `HashMap` and computes pairwise similarities at
/// query time.  Produces exact nearest-neighbours at the cost of O(n)
/// query complexity.
pub struct FlatIndex {
    vectors: HashMap<String, Vec<f32>>,
    stats: IndexStats,
}

impl Default for FlatIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl FlatIndex {
    /// Create an empty flat index.
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            stats: IndexStats {
                index_type: "Flat".to_string(),
                num_vectors: 0,
                build_time: std::time::Duration::from_secs(0),
                memory_usage: 0,
            },
        }
    }
}

#[async_trait::async_trait]
impl VectorIndex for FlatIndex {
    async fn build(&mut self, vectors: &DashMap<String, VectorData>) -> Result<()> {
        let start = std::time::Instant::now();

        self.vectors.clear();
        for entry in vectors.iter() {
            self.vectors
                .insert(entry.key().clone(), entry.value().vector.clone());
        }

        self.stats.num_vectors = self.vectors.len();
        self.stats.build_time = start.elapsed();
        self.stats.memory_usage = self.vectors.len()
            * self
                .vectors
                .values()
                .next()
                .map(|v| v.len() * 4)
                .unwrap_or(0);

        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        let mut similarities = Vec::with_capacity(self.vectors.len());

        for (id, vector) in &self.vectors {
            let similarity = compute_similarity(query, vector, metric)?;
            similarities.push((id.clone(), similarity));
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        similarities.truncate(k);
        Ok(similarities)
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        self.vectors.insert(id, vector);
        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    async fn remove(&mut self, id: &str) -> Result<()> {
        self.vectors.remove(id);
        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    fn get_stats(&self) -> IndexStats {
        self.stats.clone()
    }
}

// ---------------------------------------------------------------------------
// HNSWIndex — Hierarchical Navigable Small World
// ---------------------------------------------------------------------------

/// HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.
///
/// Organises vectors into a multi-layer probabilistic graph.  During search a
/// greedy top-down navigation narrows candidates down to the bottom layer where
/// an `ef_search`-wide beam search produces the final results.
pub struct HNSWIndex {
    #[allow(dead_code)]
    max_connections: usize,
    #[allow(dead_code)]
    ef_construction: usize,
    #[allow(dead_code)]
    ef_search: usize,

    /// Multi-layer graph: `layers[l][node_id] = [neighbour_id, …]`.
    layers: Vec<HashMap<String, Vec<String>>>,

    /// Flat vector storage (source of truth for similarity computation).
    vectors: HashMap<String, Vec<f32>>,

    /// Current entry point (topmost / highest-layer node).
    entry_point: Option<String>,

    stats: IndexStats,

    /// Deterministic PRNG for layer assignment.
    rng: Random<StdRng>,
}

impl HNSWIndex {
    /// Construct a new HNSW index with the given hyper-parameters.
    ///
    /// - `max_connections`   — `M` in the paper (neighbours per layer ≥ 1)
    /// - `ef_construction`   — candidate list size during build
    /// - `ef_search`         — candidate list size during search
    pub fn new(max_connections: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            max_connections,
            ef_construction,
            ef_search,
            layers: Vec::new(),
            vectors: HashMap::new(),
            entry_point: None,
            stats: IndexStats {
                index_type: "HNSW".to_string(),
                num_vectors: 0,
                build_time: std::time::Duration::from_secs(0),
                memory_usage: 0,
            },
            rng: Random::seed(42),
        }
    }
}

#[async_trait::async_trait]
impl VectorIndex for HNSWIndex {
    async fn build(&mut self, vectors: &DashMap<String, VectorData>) -> Result<()> {
        let start = std::time::Instant::now();

        self.layers.clear();
        self.vectors.clear();

        for entry in vectors.iter() {
            let id = entry.key().clone();
            let vector = entry.value().vector.clone();

            self.vectors.insert(id.clone(), vector);

            let layer = self.get_random_layer();

            while self.layers.len() <= layer {
                self.layers.push(HashMap::new());
            }

            for l in 0..=layer {
                if l >= self.layers.len() {
                    self.layers.push(HashMap::new());
                }
                self.layers[l].insert(id.clone(), Vec::new());
            }

            if self.entry_point.is_none() || layer >= self.layers.len() - 1 {
                self.entry_point = Some(id.clone());
            }
        }

        self.build_connections().await?;

        self.stats.num_vectors = self.vectors.len();
        self.stats.build_time = start.elapsed();

        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        self.beam_search(query, k, metric)
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        self.vectors.insert(id.clone(), vector);

        if self.layers.is_empty() {
            self.layers.push(HashMap::new());
        }
        self.layers[0].insert(id.clone(), Vec::new());

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
        }

        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    async fn remove(&mut self, id: &str) -> Result<()> {
        self.vectors.remove(id);

        for layer in &mut self.layers {
            layer.remove(id);
        }

        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    fn get_stats(&self) -> IndexStats {
        self.stats.clone()
    }
}

impl HNSWIndex {
    /// Assign a random layer to a new node using an exponential distribution.
    fn get_random_layer(&mut self) -> usize {
        let mut layer = 0usize;
        while (self.rng.random_f64() as f32) < 0.5 && layer < 16 {
            layer += 1;
        }
        layer
    }

    /// Build bidirectional connections in every layer after all nodes are inserted.
    async fn build_connections(&mut self) -> Result<()> {
        let ids: Vec<String> = self.vectors.keys().cloned().collect();

        if ids.is_empty() {
            return Ok(());
        }

        for id in &ids {
            let vector = match self.vectors.get(id) {
                Some(v) => v.clone(),
                None => continue,
            };

            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                if !layer.contains_key(id) {
                    continue;
                }

                let mut candidates: Vec<(String, f32)> = Vec::new();
                for (other_id, _) in layer.iter() {
                    if other_id == id {
                        continue;
                    }
                    if let Some(other_vector) = self.vectors.get(other_id) {
                        let similarity =
                            compute_similarity(&vector, other_vector, SimilarityMetric::Cosine)
                                .unwrap_or(0.0);
                        candidates.push((other_id.clone(), similarity));
                    }
                }

                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                let max_conn = if layer_idx == 0 {
                    self.max_connections * 2
                } else {
                    self.max_connections
                };
                candidates.truncate(max_conn);

                let connections: Vec<String> = candidates.into_iter().map(|(cid, _)| cid).collect();
                layer.insert(id.clone(), connections.clone());

                for neighbor_id in connections {
                    if let Some(neighbor_connections) = layer.get_mut(&neighbor_id) {
                        if !neighbor_connections.contains(id)
                            && neighbor_connections.len() < max_conn
                        {
                            neighbor_connections.push(id.clone());
                        }
                    }
                }
            }
        }

        // Update memory-usage estimate.
        let mut memory = 0usize;
        for (id, vec) in &self.vectors {
            memory += id.len() + vec.len() * 4;
        }
        for layer in &self.layers {
            for (id, connections) in layer {
                memory += id.len() + connections.len() * 8;
            }
        }
        self.stats.memory_usage = memory;

        Ok(())
    }

    /// Top-down greedy search followed by ef-wide beam search on layer 0.
    fn beam_search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let entry = match &self.entry_point {
            Some(e) => e.clone(),
            None => return Ok(Vec::new()),
        };

        // Greedy descent from top layer to layer 1.
        let mut current_best = entry;

        for layer_idx in (1..self.layers.len()).rev() {
            let layer = &self.layers[layer_idx];

            #[allow(clippy::while_let_loop)]
            loop {
                let current_vector = match self.vectors.get(&current_best) {
                    Some(v) => v,
                    None => break,
                };
                let current_sim = compute_similarity(query, current_vector, metric)?;

                let mut improved = false;
                if let Some(neighbors) = layer.get(&current_best) {
                    for neighbor in neighbors {
                        if let Some(neighbor_vector) = self.vectors.get(neighbor) {
                            let neighbor_sim = compute_similarity(query, neighbor_vector, metric)?;
                            if neighbor_sim > current_sim {
                                current_best = neighbor.clone();
                                improved = true;
                                break;
                            }
                        }
                    }
                }

                if !improved {
                    break;
                }
            }
        }

        if self.layers.is_empty() {
            return Ok(Vec::new());
        }

        // Beam search on layer 0 with candidate list size `ef`.
        let bottom_layer = &self.layers[0];
        let ef = k.max(self.ef_search);

        let mut candidates: BinaryHeap<SimilarityItem> = BinaryHeap::new();
        let mut visited: HashSet<String> = HashSet::new();

        if let Some(entry_vector) = self.vectors.get(&current_best) {
            let sim = compute_similarity(query, entry_vector, metric)?;
            candidates.push(SimilarityItem {
                id: current_best.clone(),
                similarity: sim,
            });
            visited.insert(current_best);
        }

        let mut results: Vec<(String, f32)> = Vec::new();

        while let Some(current) = candidates.pop() {
            if results.len() < ef {
                results.push((current.id.clone(), current.similarity));
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            } else if current.similarity > results.last().map(|r| r.1).unwrap_or(f32::NEG_INFINITY)
            {
                results.pop();
                results.push((current.id.clone(), current.similarity));
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            }

            if let Some(neighbors) = bottom_layer.get(&current.id) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        if let Some(neighbor_vector) = self.vectors.get(neighbor) {
                            let sim = compute_similarity(query, neighbor_vector, metric)?;
                            let worst = results.last().map(|r| r.1).unwrap_or(f32::NEG_INFINITY);
                            if results.len() < ef || sim > worst {
                                candidates.push(SimilarityItem {
                                    id: neighbor.clone(),
                                    similarity: sim,
                                });
                            }
                        }
                    }
                }
            }
        }

        results.truncate(k);
        Ok(results)
    }
}
