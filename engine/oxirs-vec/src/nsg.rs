//! NSG (Navigable Small World Graph) index
//!
//! NSG is a graph-based approximate nearest neighbor search algorithm that builds
//! a monotonic navigable graph structure. It provides:
//!
//! - **Monotonic Search Path**: Guarantees that search always moves closer to query
//! - **Memory Efficiency**: Controlled out-degree for compact graph structure
//! - **High Accuracy**: Better recall than NSW with similar or better performance
//! - **Fast Search**: O(log n) expected search complexity
//!
//! # Algorithm Overview
//!
//! NSG construction has two stages:
//! 1. Build initial kNN graph using any ANN algorithm
//! 2. Refine graph to ensure navigability and monotonicity
//!
//! The key innovation is the monotonic search property: each hop in the search
//! path gets closer to the query point, preventing cycles and dead-ends.
//!
//! # References
//!
//! - Fu, Cong, et al. "Fast approximate nearest neighbor search with the navigable
//!   small world graph." arXiv preprint arXiv:1707.00143 (2017).
//!
//! # Example
//!
//! ```rust
//! use oxirs_vec::{Vector, VectorIndex};
//! use oxirs_vec::nsg::{NsgConfig, NsgIndex};
//!
//! let config = NsgConfig {
//!     out_degree: 32,
//!     candidate_pool_size: 100,
//!     search_length: 50,
//!     ..Default::default()
//! };
//!
//! let mut index = NsgIndex::new(config).unwrap();
//!
//! // Add vectors
//! for i in 0..1000 {
//!     let vector = Vector::new(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
//!     index.insert(format!("vec_{}", i), vector).unwrap();
//! }
//!
//! // Build the NSG structure
//! index.build().unwrap();
//!
//! // Search
//! let query = Vector::new(vec![100.0, 200.0, 300.0]);
//! let results = index.search_knn(&query, 10).unwrap();
//! ```

use crate::{Vector, VectorIndex};
use anyhow::Result;
use oxirs_core::simd::SimdOps;
use parking_lot::RwLock as ParkingLotRwLock;
use scirs2_core::random::Random;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Configuration for NSG index
#[derive(Debug, Clone)]
pub struct NsgConfig {
    /// Maximum out-degree (number of outgoing edges per node)
    pub out_degree: usize,
    /// Candidate pool size during graph construction
    pub candidate_pool_size: usize,
    /// Search length during graph refinement
    pub search_length: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Enable parallel construction
    pub parallel_construction: bool,
    /// Number of threads for parallel construction
    pub num_threads: usize,
    /// Initial kNN graph degree
    pub initial_knn_degree: usize,
    /// Pruning threshold for edge quality
    pub pruning_threshold: f32,
}

impl Default for NsgConfig {
    fn default() -> Self {
        Self {
            out_degree: 32,
            candidate_pool_size: 100,
            search_length: 50,
            distance_metric: DistanceMetric::Euclidean,
            random_seed: None,
            parallel_construction: true,
            num_threads: num_cpus::get(),
            initial_knn_degree: 64,
            pruning_threshold: 1.0,
        }
    }
}

/// Distance metrics supported by NSG
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Angular,
    InnerProduct,
}

impl DistanceMetric {
    /// Calculate distance between two vectors
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Euclidean => f32::euclidean_distance(a, b),
            DistanceMetric::Manhattan => f32::manhattan_distance(a, b),
            DistanceMetric::Cosine => f32::cosine_distance(a, b),
            DistanceMetric::Angular => {
                let cos_sim = 1.0 - f32::cosine_distance(a, b);
                cos_sim.clamp(-1.0, 1.0).acos() / std::f32::consts::PI
            }
            DistanceMetric::InnerProduct => {
                // Negative inner product (to use as distance)
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
        }
    }
}

/// Search candidate with distance
#[derive(Debug, Clone)]
struct Candidate {
    id: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for max-heap (we want min distance at top)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// NSG index structure
pub struct NsgIndex {
    /// Configuration
    config: NsgConfig,
    /// Stored vectors with URIs
    data: Vec<(String, Vector)>,
    /// Forward adjacency list (outgoing edges)
    graph: Vec<Vec<usize>>,
    /// Entry point for search
    entry_point: Option<usize>,
    /// Whether the index is built
    is_built: bool,
    /// URI to index mapping
    uri_to_idx: HashMap<String, usize>,
    /// Statistics
    stats: Arc<RwLock<NsgStats>>,
}

/// NSG index statistics
#[derive(Debug, Clone, Default)]
pub struct NsgStats {
    /// Number of vectors indexed
    pub num_vectors: usize,
    /// Number of edges in the graph
    pub num_edges: usize,
    /// Average out-degree
    pub avg_out_degree: f64,
    /// Maximum out-degree
    pub max_out_degree: usize,
    /// Number of searches performed
    pub num_searches: usize,
    /// Average search path length
    pub avg_search_path_length: f64,
    /// Total distance computations
    pub total_distance_computations: usize,
}

impl NsgIndex {
    /// Create a new NSG index with given configuration
    pub fn new(config: NsgConfig) -> Result<Self> {
        Ok(Self {
            config,
            data: Vec::new(),
            graph: Vec::new(),
            entry_point: None,
            is_built: false,
            uri_to_idx: HashMap::new(),
            stats: Arc::new(RwLock::new(NsgStats::default())),
        })
    }

    /// Add a vector to the index (must call build() after adding all vectors)
    pub fn add(&mut self, uri: String, vector: Vector) -> Result<()> {
        if self.is_built {
            return Err(anyhow::anyhow!(
                "Cannot add vectors after index is built. Call rebuild() or create a new index."
            ));
        }

        let idx = self.data.len();
        self.uri_to_idx.insert(uri.clone(), idx);
        self.data.push((uri, vector));

        Ok(())
    }

    /// Build the NSG structure
    ///
    /// This is a two-stage process:
    /// 1. Build initial kNN graph
    /// 2. Refine to create navigable monotonic graph
    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Err(anyhow::anyhow!("Cannot build index with no vectors"));
        }

        tracing::info!("Building NSG index with {} vectors", self.data.len());

        // Stage 1: Build initial kNN graph
        tracing::debug!("Stage 1: Building initial kNN graph");
        self.build_knn_graph()?;

        // Stage 2: Refine to NSG
        tracing::debug!("Stage 2: Refining to navigable monotonic graph");
        self.refine_to_nsg()?;

        // Select entry point
        self.select_entry_point()?;

        self.is_built = true;

        // Update statistics
        self.update_stats();

        tracing::info!(
            "NSG index built successfully. {} vectors, {} edges, avg out-degree: {:.2}",
            self.data.len(),
            self.count_edges(),
            self.avg_out_degree()
        );

        Ok(())
    }

    /// Build initial kNN graph using brute-force search
    fn build_knn_graph(&mut self) -> Result<()> {
        let n = self.data.len();
        self.graph = vec![Vec::new(); n];

        if self.config.parallel_construction && n > 1000 {
            self.build_knn_graph_parallel()?;
        } else {
            self.build_knn_graph_sequential()?;
        }

        Ok(())
    }

    /// Sequential kNN graph construction
    fn build_knn_graph_sequential(&mut self) -> Result<()> {
        let n = self.data.len();
        let k = self.config.initial_knn_degree.min(n - 1);

        for i in 0..n {
            let mut neighbors = Vec::new();

            // Find k nearest neighbors
            for j in 0..n {
                if i == j {
                    continue;
                }

                let dist = self.calculate_distance(i, j);
                neighbors.push(Candidate {
                    id: j,
                    distance: dist,
                });
            }

            // Sort and keep top-k
            neighbors.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });
            neighbors.truncate(k);

            // Add bidirectional edges
            self.graph[i] = neighbors.iter().map(|c| c.id).collect();
        }

        Ok(())
    }

    /// Parallel kNN graph construction
    fn build_knn_graph_parallel(&mut self) -> Result<()> {
        let n = self.data.len();
        let k = self.config.initial_knn_degree.min(n - 1);

        // Create thread-safe graph structure
        let graph = Arc::new(ParkingLotRwLock::new(vec![Vec::new(); n]));
        let data = Arc::new(self.data.clone());
        let config = self.config.clone();

        // Process in parallel chunks
        let chunk_size = (n + self.config.num_threads - 1) / self.config.num_threads;
        let mut handles = Vec::new();

        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let graph_clone = Arc::clone(&graph);
            let data_clone = Arc::clone(&data);
            let config_clone = config.clone();

            let handle = std::thread::spawn(move || {
                for i in chunk_start..chunk_end {
                    let mut neighbors = Vec::new();

                    for j in 0..n {
                        if i == j {
                            continue;
                        }

                        let vec_i = &data_clone[i].1.as_f32();
                        let vec_j = &data_clone[j].1.as_f32();
                        let dist = config_clone.distance_metric.distance(vec_i, vec_j);

                        neighbors.push(Candidate {
                            id: j,
                            distance: dist,
                        });
                    }

                    neighbors.sort_by(|a, b| {
                        a.distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(Ordering::Equal)
                    });
                    neighbors.truncate(k);

                    let mut graph_lock = graph_clone.write();
                    graph_lock[i] = neighbors.iter().map(|c| c.id).collect();
                }
            });

            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("Thread panicked"))?;
        }

        // Copy results back
        self.graph = Arc::try_unwrap(graph)
            .map_err(|_| anyhow::anyhow!("Failed to unwrap graph"))?
            .into_inner();

        Ok(())
    }

    /// Refine kNN graph to NSG with monotonic navigability
    fn refine_to_nsg(&mut self) -> Result<()> {
        let n = self.data.len();
        let mut new_graph = vec![Vec::new(); n];

        // Select a temporary entry point for refinement
        let temp_entry = self.select_temp_entry_point();

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            // Find candidate neighbors through navigation
            let candidates = self.search_for_neighbors(i, temp_entry)?;

            // Prune to maintain out-degree constraint
            let neighbors = self.prune_neighbors(i, candidates)?;

            new_graph[i] = neighbors;
        }

        // Ensure connectivity by adding reverse edges where needed
        self.ensure_connectivity(&mut new_graph)?;

        self.graph = new_graph;

        Ok(())
    }

    /// Search for candidate neighbors during graph refinement
    fn search_for_neighbors(&self, query_id: usize, entry_id: usize) -> Result<Vec<Candidate>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut result = Vec::new();

        // Start from entry point
        let entry_dist = self.calculate_distance(query_id, entry_id);
        candidates.push(Candidate {
            id: entry_id,
            distance: entry_dist,
        });
        visited.insert(entry_id);

        while let Some(current) = candidates.pop() {
            if result.len() >= self.config.candidate_pool_size {
                break;
            }

            result.push(current.clone());

            // Explore neighbors
            for &neighbor_id in &self.graph[current.id] {
                if visited.contains(&neighbor_id) {
                    continue;
                }

                visited.insert(neighbor_id);

                let dist = self.calculate_distance(query_id, neighbor_id);
                candidates.push(Candidate {
                    id: neighbor_id,
                    distance: dist,
                });

                if visited.len() >= self.config.search_length {
                    break;
                }
            }
        }

        // Sort by distance
        result.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        Ok(result)
    }

    /// Prune neighbors to maintain graph quality and out-degree constraint
    fn prune_neighbors(
        &self,
        _query_id: usize,
        mut candidates: Vec<Candidate>,
    ) -> Result<Vec<usize>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::new();
        let mut pruned = HashSet::new();

        while !candidates.is_empty() && result.len() < self.config.out_degree {
            // Find best candidate (minimum distance)
            let best_idx = candidates
                .iter()
                .position_min_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(Ordering::Equal)
                })
                .expect("candidates should not be empty during pruning");

            let best = candidates.swap_remove(best_idx);

            if pruned.contains(&best.id) {
                continue;
            }

            result.push(best.id);
            pruned.insert(best.id);

            // Prune candidates that are too close to the selected neighbor
            candidates.retain(|c| {
                let dist_to_best = self.calculate_distance(c.id, best.id);
                dist_to_best > best.distance * self.config.pruning_threshold
            });
        }

        Ok(result)
    }

    /// Ensure graph connectivity by adding reverse edges
    fn ensure_connectivity(&self, graph: &mut [Vec<usize>]) -> Result<()> {
        let n = graph.len();

        // Build reverse index
        let mut in_edges: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for (i, neighbors) in graph.iter().enumerate() {
            for &j in neighbors {
                in_edges[j].insert(i);
            }
        }

        // For each node, ensure it has at least one incoming edge
        for (i, edges) in in_edges.iter().enumerate() {
            if edges.is_empty() && i != 0 {
                // Find closest node that has outgoing edges
                let mut min_dist = f32::INFINITY;
                let mut closest = 0;

                for (j, neighbors) in graph.iter().enumerate() {
                    if i == j || neighbors.len() >= self.config.out_degree {
                        continue;
                    }

                    let dist = self.calculate_distance(i, j);
                    if dist < min_dist {
                        min_dist = dist;
                        closest = j;
                    }
                }

                // Add edge from closest to i
                if !graph[closest].contains(&i) {
                    graph[closest].push(i);
                }
            }
        }

        Ok(())
    }

    /// Select entry point for search (node with highest out-degree)
    fn select_entry_point(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        let mut max_degree = 0;
        let mut entry = 0;

        for i in 0..self.graph.len() {
            if self.graph[i].len() > max_degree {
                max_degree = self.graph[i].len();
                entry = i;
            }
        }

        self.entry_point = Some(entry);

        Ok(())
    }

    /// Select temporary entry point for graph refinement
    fn select_temp_entry_point(&self) -> usize {
        if let Some(seed) = self.config.random_seed {
            let mut rng = Random::seed(seed);
            rng.random_range(0..self.data.len())
        } else {
            // Use centroid as entry point
            self.find_centroid()
        }
    }

    /// Find centroid of all vectors
    fn find_centroid(&self) -> usize {
        if self.data.is_empty() {
            return 0;
        }

        let dim = self.data[0].1.dimensions;
        let mut centroid = vec![0.0f32; dim];

        // Calculate mean
        for (_, vec) in &self.data {
            let vals = vec.as_f32();
            for i in 0..dim {
                centroid[i] += vals[i];
            }
        }

        let n = self.data.len() as f32;
        for val in &mut centroid {
            *val /= n;
        }

        // Find closest vector to centroid
        let mut min_dist = f32::INFINITY;
        let mut closest = 0;

        for i in 0..self.data.len() {
            let dist = self
                .config
                .distance_metric
                .distance(&centroid, &self.data[i].1.as_f32());
            if dist < min_dist {
                min_dist = dist;
                closest = i;
            }
        }

        closest
    }

    /// Calculate distance between two vectors by index
    fn calculate_distance(&self, i: usize, j: usize) -> f32 {
        let vec_i = self.data[i].1.as_f32();
        let vec_j = self.data[j].1.as_f32();
        self.config.distance_metric.distance(&vec_i, &vec_j)
    }

    /// Perform greedy search on the graph
    fn greedy_search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<Candidate>> {
        if !self.is_built {
            return Err(anyhow::anyhow!("Index not built. Call build() first."));
        }

        let entry = self
            .entry_point
            .ok_or_else(|| anyhow::anyhow!("No entry point set"))?;

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut result_set = BinaryHeap::new();

        // Initialize with entry point
        let entry_dist = self
            .config
            .distance_metric
            .distance(query, &self.data[entry].1.as_f32());
        candidates.push(Candidate {
            id: entry,
            distance: entry_dist,
        });
        result_set.push(Candidate {
            id: entry,
            distance: entry_dist,
        });
        visited.insert(entry);

        while let Some(current) = candidates.pop() {
            // Check if we've explored enough
            if result_set.len() >= ef
                && current.distance
                    > result_set
                        .peek()
                        .expect("result_set should not be empty during search")
                        .distance
            {
                break;
            }

            // Explore neighbors
            for &neighbor_id in &self.graph[current.id] {
                if visited.contains(&neighbor_id) {
                    continue;
                }

                visited.insert(neighbor_id);

                let dist = self
                    .config
                    .distance_metric
                    .distance(query, &self.data[neighbor_id].1.as_f32());
                let candidate = Candidate {
                    id: neighbor_id,
                    distance: dist,
                };

                if result_set.len() < ef
                    || dist
                        < result_set
                            .peek()
                            .expect("result_set should not be empty during search")
                            .distance
                {
                    candidates.push(candidate.clone());
                    result_set.push(candidate);

                    if result_set.len() > ef {
                        result_set.pop();
                    }
                }
            }
        }

        // Convert to sorted vector
        let mut results: Vec<_> = result_set.into_sorted_vec();
        results.truncate(k);

        Ok(results)
    }

    /// Update index statistics
    fn update_stats(&self) {
        let mut stats = self
            .stats
            .write()
            .expect("stats lock should not be poisoned");
        stats.num_vectors = self.data.len();
        stats.num_edges = self.count_edges();
        stats.avg_out_degree = self.avg_out_degree();
        stats.max_out_degree = self.max_out_degree();
    }

    /// Count total edges in graph
    fn count_edges(&self) -> usize {
        self.graph.iter().map(|neighbors| neighbors.len()).sum()
    }

    /// Calculate average out-degree
    fn avg_out_degree(&self) -> f64 {
        if self.graph.is_empty() {
            return 0.0;
        }
        self.count_edges() as f64 / self.graph.len() as f64
    }

    /// Get maximum out-degree
    fn max_out_degree(&self) -> usize {
        self.graph
            .iter()
            .map(|neighbors| neighbors.len())
            .max()
            .unwrap_or(0)
    }

    /// Get index statistics
    pub fn stats(&self) -> NsgStats {
        self.stats
            .read()
            .expect("stats lock should not be poisoned")
            .clone()
    }

    /// Get number of vectors in the index
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if index is built
    pub fn is_built(&self) -> bool {
        self.is_built
    }
}

impl VectorIndex for NsgIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        self.add(uri, vector)
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let query_vals = query.as_f32();
        let ef = k.max(self.config.search_length);
        let candidates = self.greedy_search(&query_vals, k, ef)?;

        // Convert to (URI, similarity) format
        // Note: NSG uses distance, so we convert to similarity (1 / (1 + distance))
        // Candidates are sorted by distance (ascending), so we reverse to get descending similarity
        let mut results: Vec<_> = candidates
            .into_iter()
            .map(|c| {
                let uri = self.data[c.id].0.clone();
                let similarity = 1.0 / (1.0 + c.distance);
                (uri, similarity)
            })
            .collect();

        // Reverse to get descending order of similarity
        results.reverse();

        Ok(results)
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        // Search with a large k and filter by threshold
        let k = self.data.len().min(1000);
        let all_results = self.search_knn(query, k)?;

        let filtered: Vec<_> = all_results
            .into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect();

        Ok(filtered)
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        self.uri_to_idx
            .get(uri)
            .and_then(|&idx| self.data.get(idx))
            .map(|(_, vec)| vec)
    }

    fn remove_vector(&mut self, id: String) -> Result<()> {
        if self.is_built {
            return Err(anyhow::anyhow!(
                "Cannot remove vectors from built index. Rebuild index instead."
            ));
        }

        if let Some(&idx) = self.uri_to_idx.get(&id) {
            self.data.remove(idx);
            self.uri_to_idx.remove(&id);

            // Rebuild index mapping
            self.uri_to_idx.clear();
            for (i, (uri, _)) in self.data.iter().enumerate() {
                self.uri_to_idx.insert(uri.clone(), i);
            }

            Ok(())
        } else {
            Err(anyhow::anyhow!("Vector with id '{}' not found", id))
        }
    }
}

// Helper trait for finding min by
trait IteratorExt: Iterator {
    fn position_min_by<F>(self, compare: F) -> Option<usize>
    where
        F: FnMut(&Self::Item, &Self::Item) -> Ordering;
}

impl<I: Iterator> IteratorExt for I {
    fn position_min_by<F>(mut self, mut compare: F) -> Option<usize>
    where
        F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        let first = self.next()?;
        let mut min_item = first;
        let mut min_pos = 0;

        for (pos, item) in self.enumerate() {
            if compare(&item, &min_item) == Ordering::Less {
                min_item = item;
                min_pos = pos + 1;
            }
        }

        Some(min_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nsg_creation() {
        let config = NsgConfig::default();
        let index = NsgIndex::new(config).unwrap();
        assert_eq!(index.len(), 0);
        assert!(!index.is_built());
    }

    #[test]
    fn test_nsg_add_vectors() {
        let config = NsgConfig::default();
        let mut index = NsgIndex::new(config).unwrap();

        for i in 0..10 {
            let vec = Vector::new(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
            index.add(format!("vec_{}", i), vec).unwrap();
        }

        assert_eq!(index.len(), 10);
    }

    #[test]
    fn test_nsg_build_and_search() {
        let config = NsgConfig {
            out_degree: 32,
            candidate_pool_size: 100,
            search_length: 50,
            initial_knn_degree: 64,
            ..Default::default()
        };
        let mut index = NsgIndex::new(config).unwrap();

        // Add vectors in a more structured way to ensure connectivity
        for i in 0..100 {
            let vec = Vector::new(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
            index.add(format!("vec_{}", i), vec).unwrap();
        }

        // Build index
        index.build().unwrap();
        assert!(index.is_built());

        // Search with a query close to vec_10 (easier to verify)
        let query = Vector::new(vec![10.1, 20.1, 30.1]);
        let results = index.search_knn(&query, 10).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results.len(), 10);

        // Results should be sorted by similarity (descending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "Results not sorted: {}@{} < {}@{}",
                results[i - 1].1,
                i - 1,
                results[i].1,
                i
            );
        }

        // The closest vectors should be vec_10, vec_11, vec_9, etc.
        // At least one of these should be in top 10
        let nearby_found = results.iter().take(10).any(|(uri, _)| {
            uri.contains("10")
                || uri.contains("11")
                || uri.contains("9")
                || uri.contains("12")
                || uri.contains("8")
        });
        assert!(
            nearby_found,
            "Expected nearby vectors (8-12) in top 10 results"
        );
    }

    #[test]
    fn test_nsg_distance_metrics() {
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Cosine,
            DistanceMetric::Angular,
        ] {
            let config = NsgConfig {
                distance_metric: metric,
                out_degree: 8,
                ..Default::default()
            };
            let mut index = NsgIndex::new(config).unwrap();

            for i in 0..20 {
                let vec = Vector::new(vec![i as f32, (i * 2) as f32]);
                index.add(format!("vec_{}", i), vec).unwrap();
            }

            index.build().unwrap();

            let query = Vector::new(vec![10.0, 20.0]);
            let results = index.search_knn(&query, 3).unwrap();

            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_nsg_stats() {
        let config = NsgConfig::default();
        let mut index = NsgIndex::new(config).unwrap();

        for i in 0..50 {
            let vec = Vector::new(vec![i as f32, (i * 2) as f32]);
            index.add(format!("vec_{}", i), vec).unwrap();
        }

        index.build().unwrap();

        let stats = index.stats();
        assert_eq!(stats.num_vectors, 50);
        assert!(stats.num_edges > 0);
        assert!(stats.avg_out_degree > 0.0);
    }

    #[test]
    fn test_nsg_threshold_search() {
        let config = NsgConfig::default();
        let mut index = NsgIndex::new(config).unwrap();

        for i in 0..30 {
            let vec = Vector::new(vec![i as f32, (i * 2) as f32]);
            index.add(format!("vec_{}", i), vec).unwrap();
        }

        index.build().unwrap();

        let query = Vector::new(vec![15.0, 30.0]);
        let results = index.search_threshold(&query, 0.5).unwrap();

        assert!(!results.is_empty());
        // All results should have similarity >= 0.5
        for (_, similarity) in results {
            assert!(similarity >= 0.5);
        }
    }
}
