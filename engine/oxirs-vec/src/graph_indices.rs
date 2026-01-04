//! Graph-based indices for efficient nearest neighbor search
//!
//! This module implements various graph-based data structures optimized for
//! nearest neighbor search:
//! - NSW: Navigable Small World
//! - ONNG: Optimized Nearest Neighbor Graph
//! - PANNG: Pruned Approximate Nearest Neighbor Graph
//! - Delaunay Graph: Approximation for high-dimensional space
//! - RNG: Relative Neighborhood Graph

use crate::{Vector, VectorIndex};
use anyhow::Result;
use oxirs_core::parallel::*;
use oxirs_core::simd::SimdOps;
use petgraph::graph::{Graph, NodeIndex};
#[allow(unused_imports)]
use scirs2_core::random::{Random, Rng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Configuration for graph-based indices
#[derive(Debug, Clone)]
pub struct GraphIndexConfig {
    /// Type of graph to use
    pub graph_type: GraphType,
    /// Number of neighbors per node
    pub num_neighbors: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Enable parallel construction
    pub parallel_construction: bool,
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Enable pruning for better quality
    pub enable_pruning: bool,
    /// Search depth multiplier
    pub search_expansion: f32,
}

impl Default for GraphIndexConfig {
    fn default() -> Self {
        Self {
            graph_type: GraphType::NSW,
            num_neighbors: 32,
            random_seed: None,
            parallel_construction: true,
            distance_metric: DistanceMetric::Euclidean,
            enable_pruning: true,
            search_expansion: 1.5,
        }
    }
}

/// Available graph types
#[derive(Debug, Clone, Copy)]
pub enum GraphType {
    NSW,      // Navigable Small World
    ONNG,     // Optimized Nearest Neighbor Graph
    PANNG,    // Pruned Approximate Nearest Neighbor Graph
    Delaunay, // Delaunay Graph approximation
    RNG,      // Relative Neighborhood Graph
}

/// Distance metrics
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Angular,
}

impl DistanceMetric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Euclidean => f32::euclidean_distance(a, b),
            DistanceMetric::Manhattan => f32::manhattan_distance(a, b),
            DistanceMetric::Cosine => f32::cosine_distance(a, b),
            DistanceMetric::Angular => {
                // Angular distance = arccos(cosine_similarity) / pi
                let cos_sim: f32 = 1.0 - f32::cosine_distance(a, b);
                cos_sim.clamp(-1.0, 1.0).acos() / std::f32::consts::PI
            }
        }
    }
}

/// Search result with distance
#[derive(Debug, Clone)]
struct SearchResult {
    index: usize,
    distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Navigable Small World (NSW) implementation
pub struct NSWGraph {
    /// Graph structure
    graph: Graph<usize, f32>,
    /// Node index mapping
    node_map: HashMap<usize, NodeIndex>,
    /// Data storage
    data: Vec<(String, Vector)>,
    /// Configuration
    config: GraphIndexConfig,
    /// Entry points for search
    entry_points: Vec<NodeIndex>,
}

impl NSWGraph {
    pub fn new(config: GraphIndexConfig) -> Self {
        Self {
            graph: Graph::new(),
            node_map: HashMap::new(),
            data: Vec::new(),
            config,
            entry_points: Vec::new(),
        }
    }

    /// Build the graph from data
    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        // Create nodes
        for (idx, _) in self.data.iter().enumerate() {
            let node = self.graph.add_node(idx);
            self.node_map.insert(idx, node);
        }

        // Select random entry points
        let num_entry_points = (self.data.len() as f32).sqrt() as usize;
        let mut rng = if let Some(seed) = self.config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        // Note: Using manual random selection instead of SliceRandom
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        // Manually shuffle using Fisher-Yates algorithm
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..i + 1);
            indices.swap(i, j);
        }

        self.entry_points = indices[..num_entry_points.min(self.data.len())]
            .iter()
            .map(|&idx| self.node_map[&idx])
            .collect();

        // Build graph structure
        if self.config.parallel_construction && self.data.len() > 1000 {
            self.build_parallel()?;
        } else {
            self.build_sequential()?;
        }

        Ok(())
    }

    fn build_sequential(&mut self) -> Result<()> {
        for idx in 0..self.data.len() {
            let neighbors = self.find_neighbors(idx, self.config.num_neighbors)?;
            let node = self.node_map[&idx];

            for (neighbor_idx, distance) in neighbors {
                let neighbor_node = self.node_map[&neighbor_idx];
                if !self.graph.contains_edge(node, neighbor_node) {
                    self.graph.add_edge(node, neighbor_node, distance);
                }
            }
        }

        Ok(())
    }

    fn build_parallel(&mut self) -> Result<()> {
        let _chunk_size = (self.data.len() / num_threads()).max(100);

        // Pre-compute all edges that need to be added
        let mut all_edges = Vec::new();
        for idx in 0..self.data.len() {
            let neighbors = self.find_neighbors(idx, self.config.num_neighbors)?;
            let node = self.node_map[&idx];

            for (neighbor_idx, distance) in neighbors {
                let neighbor_node = self.node_map[&neighbor_idx];
                all_edges.push((node, neighbor_node, distance));
            }
        }

        // Now add all edges to the graph
        for (from, to, weight) in all_edges {
            if !self.graph.contains_edge(from, to) {
                self.graph.add_edge(from, to, weight);
            }
        }

        Ok(())
    }

    fn find_neighbors(&self, idx: usize, k: usize) -> Result<Vec<(usize, f32)>> {
        let query = &self.data[idx].1.as_f32();
        let mut heap = BinaryHeap::new();

        for (other_idx, (_, vector)) in self.data.iter().enumerate() {
            if other_idx == idx {
                continue;
            }

            let other = vector.as_f32();
            let distance = self.config.distance_metric.distance(query, &other);

            if heap.len() < k {
                heap.push(SearchResult {
                    index: other_idx,
                    distance,
                });
            } else if distance < heap.peek().expect("heap should have k elements").distance {
                heap.pop();
                heap.push(SearchResult {
                    index: other_idx,
                    distance,
                });
            }
        }

        Ok(heap.into_iter().map(|r| (r.index, r.distance)).collect())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.entry_points.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results: BinaryHeap<SearchResult> = BinaryHeap::new();

        // Initialize with entry points
        for &entry in &self.entry_points {
            let idx = self.graph[entry];
            let distance = self
                .config
                .distance_metric
                .distance(query, &self.data[idx].1.as_f32());
            candidates.push(std::cmp::Reverse(SearchResult {
                index: idx,
                distance,
            }));
            visited.insert(idx);
        }

        // Search expansion
        let max_candidates = (k as f32 * self.config.search_expansion) as usize;

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            // Only apply early termination if we have k results
            if results.len() >= k
                && current.distance
                    > results
                        .peek()
                        .expect("results should have k elements")
                        .distance
            {
                break;
            }

            // Update results
            if results.len() < k {
                results.push(current.clone());
            } else if current.distance
                < results
                    .peek()
                    .expect("results should have k elements")
                    .distance
            {
                results.pop();
                results.push(current.clone());
            }

            // Explore neighbors
            let node = self.node_map[&current.index];
            for neighbor in self.graph.neighbors(node) {
                let neighbor_idx = self.graph[neighbor];

                if visited.contains(&neighbor_idx) {
                    continue;
                }

                visited.insert(neighbor_idx);
                let distance = self
                    .config
                    .distance_metric
                    .distance(query, &self.data[neighbor_idx].1.as_f32());

                if candidates.len() < max_candidates
                    || distance
                        < candidates
                            .peek()
                            .expect("candidates should have elements")
                            .0
                            .distance
                {
                    candidates.push(std::cmp::Reverse(SearchResult {
                        index: neighbor_idx,
                        distance,
                    }));
                }
            }
        }

        let mut results: Vec<(usize, f32)> =
            results.into_iter().map(|r| (r.index, r.distance)).collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }
}

/// Optimized Nearest Neighbor Graph (ONNG) implementation
pub struct ONNGGraph {
    /// Adjacency list representation
    adjacency: Vec<Vec<(usize, f32)>>,
    /// Data storage
    data: Vec<(String, Vector)>,
    /// Configuration
    config: GraphIndexConfig,
}

impl ONNGGraph {
    pub fn new(config: GraphIndexConfig) -> Self {
        Self {
            adjacency: Vec::new(),
            data: Vec::new(),
            config,
        }
    }

    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        // Initialize adjacency lists
        self.adjacency = vec![Vec::new(); self.data.len()];

        // Build initial k-NN graph
        self.build_knn_graph()?;

        // Optimize graph structure
        self.optimize_graph()?;

        Ok(())
    }

    fn build_knn_graph(&mut self) -> Result<()> {
        for idx in 0..self.data.len() {
            let neighbors = self.find_k_nearest(idx, self.config.num_neighbors)?;
            self.adjacency[idx] = neighbors;
        }

        Ok(())
    }

    fn find_k_nearest(&self, idx: usize, k: usize) -> Result<Vec<(usize, f32)>> {
        let query = &self.data[idx].1.as_f32();
        let mut neighbors = Vec::new();

        for (other_idx, (_, vector)) in self.data.iter().enumerate() {
            if other_idx == idx {
                continue;
            }

            let distance = self
                .config
                .distance_metric
                .distance(query, &vector.as_f32());
            neighbors.push((other_idx, distance));
        }

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        neighbors.truncate(k);

        Ok(neighbors)
    }

    fn optimize_graph(&mut self) -> Result<()> {
        // Add reverse edges for better connectivity
        let mut reverse_edges = vec![Vec::new(); self.data.len()];

        for (idx, neighbors) in self.adjacency.iter().enumerate() {
            for &(neighbor_idx, distance) in neighbors {
                reverse_edges[neighbor_idx].push((idx, distance));
            }
        }

        // Merge and optimize
        for (idx, reverse) in reverse_edges.into_iter().enumerate() {
            let mut all_neighbors = self.adjacency[idx].clone();
            all_neighbors.extend(reverse);

            // Remove duplicates and sort
            all_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            all_neighbors.dedup_by_key(|&mut (idx, _)| idx);
            all_neighbors.truncate(self.config.num_neighbors);

            self.adjacency[idx] = all_neighbors;
        }

        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.data.is_empty() {
            return Vec::new();
        }

        // Start from multiple random points
        let start_points = self.select_start_points();
        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();

        // Initialize with start points
        for start in start_points {
            let distance = self
                .config
                .distance_metric
                .distance(query, &self.data[start].1.as_f32());
            heap.push(std::cmp::Reverse(SearchResult {
                index: start,
                distance,
            }));
            visited.insert(start);
        }

        let mut results = Vec::new();

        while let Some(std::cmp::Reverse(current)) = heap.pop() {
            results.push((current.index, current.distance));

            if results.len() >= k {
                break;
            }

            // Explore neighbors
            for &(neighbor_idx, _) in &self.adjacency[current.index] {
                if visited.contains(&neighbor_idx) {
                    continue;
                }

                visited.insert(neighbor_idx);
                let distance = self
                    .config
                    .distance_metric
                    .distance(query, &self.data[neighbor_idx].1.as_f32());
                heap.push(std::cmp::Reverse(SearchResult {
                    index: neighbor_idx,
                    distance,
                }));
            }
        }

        results.truncate(k);
        results
    }

    fn select_start_points(&self) -> Vec<usize> {
        // Simple strategy: select sqrt(n) random points
        let num_points = (self.data.len() as f32).sqrt() as usize;
        let mut indices: Vec<usize> = (0..self.data.len()).collect();

        let mut rng = if let Some(seed) = self.config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        // Note: Using manual random selection instead of SliceRandom
        // Manually shuffle using Fisher-Yates algorithm
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..i + 1);
            indices.swap(i, j);
        }
        indices.truncate(num_points.max(1));

        indices
    }
}

/// Pruned Approximate Nearest Neighbor Graph (PANNG) implementation
pub struct PANNGGraph {
    /// Pruned adjacency list
    adjacency: Vec<Vec<(usize, f32)>>,
    /// Data storage
    data: Vec<(String, Vector)>,
    /// Configuration
    config: GraphIndexConfig,
    /// Pruning threshold
    pruning_threshold: f32,
}

impl PANNGGraph {
    pub fn new(config: GraphIndexConfig) -> Self {
        Self {
            adjacency: Vec::new(),
            data: Vec::new(),
            config,
            pruning_threshold: 0.9, // Angle-based pruning threshold
        }
    }

    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        // Build initial k-NN graph
        self.adjacency = vec![Vec::new(); self.data.len()];
        self.build_initial_graph()?;

        // Apply pruning
        if self.config.enable_pruning {
            self.prune_graph()?;
        }

        Ok(())
    }

    fn build_initial_graph(&mut self) -> Result<()> {
        // Build with more neighbors initially for pruning
        let initial_neighbors = self.config.num_neighbors * 2;

        for idx in 0..self.data.len() {
            let neighbors = self.find_k_nearest(idx, initial_neighbors)?;
            self.adjacency[idx] = neighbors;
        }

        Ok(())
    }

    fn find_k_nearest(&self, idx: usize, k: usize) -> Result<Vec<(usize, f32)>> {
        let query = &self.data[idx].1.as_f32();
        let mut heap = BinaryHeap::new();

        for (other_idx, (_, vector)) in self.data.iter().enumerate() {
            if other_idx == idx {
                continue;
            }

            let distance = self
                .config
                .distance_metric
                .distance(query, &vector.as_f32());

            if heap.len() < k {
                heap.push(SearchResult {
                    index: other_idx,
                    distance,
                });
            } else if distance < heap.peek().expect("heap should have k elements").distance {
                heap.pop();
                heap.push(SearchResult {
                    index: other_idx,
                    distance,
                });
            }
        }

        Ok(heap
            .into_sorted_vec()
            .into_iter()
            .map(|r| (r.index, r.distance))
            .collect())
    }

    fn prune_graph(&mut self) -> Result<()> {
        for idx in 0..self.data.len() {
            let pruned = self.prune_neighbors(idx)?;
            self.adjacency[idx] = pruned;
        }

        Ok(())
    }

    fn prune_neighbors(&self, idx: usize) -> Result<Vec<(usize, f32)>> {
        let neighbors = &self.adjacency[idx];
        if neighbors.len() <= self.config.num_neighbors {
            return Ok(neighbors.clone());
        }

        let mut pruned = Vec::new();
        let (_, vector) = &self.data[idx];
        let query = vector.as_f32();

        for &(neighbor_idx, distance) in neighbors {
            let (_, vector) = &self.data[neighbor_idx];
            let neighbor = vector.as_f32();
            let mut keep = true;

            // Check angle with already selected neighbors
            for &(selected_idx, _) in &pruned {
                let (_id, vector): &(String, Vector) = &self.data[selected_idx];
                let selected = vector.as_f32();

                // Calculate angle between neighbor and selected
                let angle = self.calculate_angle(&query, &neighbor, &selected);

                if angle < self.pruning_threshold {
                    keep = false;
                    break;
                }
            }

            if keep {
                pruned.push((neighbor_idx, distance));

                if pruned.len() >= self.config.num_neighbors {
                    break;
                }
            }
        }

        Ok(pruned)
    }

    fn calculate_angle(&self, origin: &[f32], a: &[f32], b: &[f32]) -> f32 {
        // Calculate vectors from origin
        let va: Vec<f32> = a
            .iter()
            .zip(origin.iter())
            .map(|(ai, oi)| ai - oi)
            .collect();
        let vb: Vec<f32> = b
            .iter()
            .zip(origin.iter())
            .map(|(bi, oi)| bi - oi)
            .collect();

        // Calculate cosine of angle
        let dot = f32::dot(&va, &vb);
        let norm_a = f32::norm(&va);
        let norm_b = f32::norm(&vb);

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0).acos()
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.data.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut candidates = VecDeque::new();
        let mut results = Vec::new();

        // Start from closest point
        let start = self.find_closest_point(query);
        candidates.push_back(start);
        visited.insert(start);

        while let Some(current) = candidates.pop_front() {
            let distance = self
                .config
                .distance_metric
                .distance(query, &self.data[current].1.as_f32());
            results.push((current, distance));

            // Explore neighbors
            for &(neighbor_idx, _) in &self.adjacency[current] {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    candidates.push_back(neighbor_idx);
                }
            }

            if results.len() >= k * 2 {
                break;
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    fn find_closest_point(&self, query: &[f32]) -> usize {
        let mut min_dist = f32::INFINITY;
        let mut closest = 0;

        // Sample a few random points
        let sample_size = (self.data.len() as f32).sqrt() as usize;
        let step = self.data.len() / sample_size.max(1);

        for idx in (0..self.data.len()).step_by(step.max(1)) {
            let distance = self
                .config
                .distance_metric
                .distance(query, &self.data[idx].1.as_f32());
            if distance < min_dist {
                min_dist = distance;
                closest = idx;
            }
        }

        closest
    }
}

/// Delaunay Graph approximation for high dimensions
pub struct DelaunayGraph {
    /// Approximate Delaunay edges
    edges: Vec<Vec<(usize, f32)>>,
    /// Data storage
    data: Vec<(String, Vector)>,
    /// Configuration
    config: GraphIndexConfig,
}

impl DelaunayGraph {
    pub fn new(config: GraphIndexConfig) -> Self {
        Self {
            edges: Vec::new(),
            data: Vec::new(),
            config,
        }
    }

    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        self.edges = vec![Vec::new(); self.data.len()];

        // For high dimensions, we approximate Delaunay by local criteria
        for idx in 0..self.data.len() {
            let neighbors = self.find_delaunay_neighbors(idx)?;
            self.edges[idx] = neighbors;
        }

        // Make edges bidirectional
        self.symmetrize_edges();

        Ok(())
    }

    fn find_delaunay_neighbors(&self, idx: usize) -> Result<Vec<(usize, f32)>> {
        let point = &self.data[idx].1.as_f32();
        let mut candidates = Vec::new();

        // Find potential neighbors
        for (other_idx, (_, other_vec)) in self.data.iter().enumerate() {
            if other_idx == idx {
                continue;
            }

            let other = other_vec.as_f32();
            let distance = self.config.distance_metric.distance(point, &other);
            candidates.push((other_idx, distance));
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Apply Delaunay criterion approximation
        let mut neighbors = Vec::new();

        for &(candidate_idx, distance) in &candidates {
            if neighbors.len() >= self.config.num_neighbors {
                break;
            }

            let candidate = &self.data[candidate_idx].1.as_f32();
            let mut is_neighbor = true;

            // Check if any existing neighbor violates the empty circumsphere property
            for &(neighbor_idx, _) in &neighbors {
                let (_id, vector): &(String, Vector) = &self.data[neighbor_idx];
                let neighbor = vector.as_f32();

                // Approximate check: if candidate is closer to neighbor than to point
                let dist_to_neighbor = self.config.distance_metric.distance(candidate, &neighbor);
                if dist_to_neighbor < distance * 0.9 {
                    is_neighbor = false;
                    break;
                }
            }

            if is_neighbor {
                neighbors.push((candidate_idx, distance));
            }
        }

        Ok(neighbors)
    }

    fn symmetrize_edges(&mut self) {
        let mut symmetric_edges = vec![Vec::new(); self.data.len()];

        // Collect all edges
        for (idx, neighbors) in self.edges.iter().enumerate() {
            for &(neighbor_idx, distance) in neighbors {
                symmetric_edges[idx].push((neighbor_idx, distance));
                symmetric_edges[neighbor_idx].push((idx, distance));
            }
        }

        // Remove duplicates and sort
        for edges in &mut symmetric_edges {
            edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            edges.dedup_by_key(|&mut (idx, _)| idx);
            edges.truncate(self.config.num_neighbors);
        }

        self.edges = symmetric_edges;
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.data.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();
        let mut results = Vec::new();

        // Start from a random point
        let start = 0;
        let distance = self
            .config
            .distance_metric
            .distance(query, &self.data[start].1.as_f32());
        heap.push(std::cmp::Reverse(SearchResult {
            index: start,
            distance,
        }));
        visited.insert(start);

        while let Some(std::cmp::Reverse(current)) = heap.pop() {
            results.push((current.index, current.distance));

            if results.len() >= k {
                break;
            }

            // Explore neighbors
            for &(neighbor_idx, _) in &self.edges[current.index] {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    let distance = self
                        .config
                        .distance_metric
                        .distance(query, &self.data[neighbor_idx].1.as_f32());
                    heap.push(std::cmp::Reverse(SearchResult {
                        index: neighbor_idx,
                        distance,
                    }));
                }
            }
        }

        results
    }
}

/// Relative Neighborhood Graph (RNG) implementation
pub struct RNGGraph {
    /// RNG edges
    edges: Vec<Vec<(usize, f32)>>,
    /// Data storage
    data: Vec<(String, Vector)>,
    /// Configuration
    config: GraphIndexConfig,
}

impl RNGGraph {
    pub fn new(config: GraphIndexConfig) -> Self {
        Self {
            edges: Vec::new(),
            data: Vec::new(),
            config,
        }
    }

    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        self.edges = vec![Vec::new(); self.data.len()];

        // Build RNG by checking the RNG criterion for each pair
        for i in 0..self.data.len() {
            for j in i + 1..self.data.len() {
                if self.is_rng_edge(i, j)? {
                    let distance = self
                        .config
                        .distance_metric
                        .distance(&self.data[i].1.as_f32(), &self.data[j].1.as_f32());

                    self.edges[i].push((j, distance));
                    self.edges[j].push((i, distance));
                }
            }
        }

        // Sort edges by distance
        for edges in &mut self.edges {
            edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        }

        Ok(())
    }

    fn is_rng_edge(&self, i: usize, j: usize) -> Result<bool> {
        let pi = &self.data[i].1.as_f32();
        let pj = &self.data[j].1.as_f32();
        let dist_ij = self.config.distance_metric.distance(pi, pj);

        // Check RNG criterion: no other point k exists such that
        // max(dist(i,k), dist(j,k)) < dist(i,j)
        for k in 0..self.data.len() {
            if k == i || k == j {
                continue;
            }

            let pk = &self.data[k].1.as_f32();
            let dist_ik = self.config.distance_metric.distance(pi, pk);
            let dist_jk = self.config.distance_metric.distance(pj, pk);

            if dist_ik.max(dist_jk) < dist_ij {
                return Ok(false);
            }
        }

        Ok(true)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.data.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = Vec::new();

        // Start from the closest sampled point
        let start = self.find_start_point(query);
        let distance = self
            .config
            .distance_metric
            .distance(query, &self.data[start].1.as_f32());
        candidates.push(std::cmp::Reverse(SearchResult {
            index: start,
            distance,
        }));
        visited.insert(start);

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            results.push((current.index, current.distance));

            if results.len() >= k {
                break;
            }

            // Explore neighbors
            for &(neighbor_idx, _) in &self.edges[current.index] {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    let distance = self
                        .config
                        .distance_metric
                        .distance(query, &self.data[neighbor_idx].1.as_f32());
                    candidates.push(std::cmp::Reverse(SearchResult {
                        index: neighbor_idx,
                        distance,
                    }));
                }
            }
        }

        results
    }

    fn find_start_point(&self, query: &[f32]) -> usize {
        // Sample a subset of points
        let sample_size = (self.data.len() as f32).sqrt() as usize;
        let mut min_dist = f32::INFINITY;
        let mut best = 0;

        for i in 0..sample_size.min(self.data.len()) {
            let idx = (i * self.data.len()) / sample_size;
            let distance = self
                .config
                .distance_metric
                .distance(query, &self.data[idx].1.as_f32());

            if distance < min_dist {
                min_dist = distance;
                best = idx;
            }
        }

        best
    }
}

/// Unified graph index interface
pub struct GraphIndex {
    graph_type: GraphType,
    nsw: Option<NSWGraph>,
    onng: Option<ONNGGraph>,
    panng: Option<PANNGGraph>,
    delaunay: Option<DelaunayGraph>,
    rng: Option<RNGGraph>,
}

impl GraphIndex {
    pub fn new(config: GraphIndexConfig) -> Self {
        let graph_type = config.graph_type;

        let (nsw, onng, panng, delaunay, rng) = match graph_type {
            GraphType::NSW => (Some(NSWGraph::new(config)), None, None, None, None),
            GraphType::ONNG => (None, Some(ONNGGraph::new(config)), None, None, None),
            GraphType::PANNG => (None, None, Some(PANNGGraph::new(config)), None, None),
            GraphType::Delaunay => (None, None, None, Some(DelaunayGraph::new(config)), None),
            GraphType::RNG => (None, None, None, None, Some(RNGGraph::new(config))),
        };

        Self {
            graph_type,
            nsw,
            onng,
            panng,
            delaunay,
            rng,
        }
    }

    fn build(&mut self) -> Result<()> {
        match self.graph_type {
            GraphType::NSW => self
                .nsw
                .as_mut()
                .expect("nsw should be initialized for NSW type")
                .build(),
            GraphType::ONNG => self
                .onng
                .as_mut()
                .expect("onng should be initialized for ONNG type")
                .build(),
            GraphType::PANNG => self
                .panng
                .as_mut()
                .expect("panng should be initialized for PANNG type")
                .build(),
            GraphType::Delaunay => self
                .delaunay
                .as_mut()
                .expect("delaunay should be initialized for Delaunay type")
                .build(),
            GraphType::RNG => self
                .rng
                .as_mut()
                .expect("rng should be initialized for RNG type")
                .build(),
        }
    }

    fn search_internal(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        match self.graph_type {
            GraphType::NSW => self
                .nsw
                .as_ref()
                .expect("nsw should be initialized for NSW type")
                .search(query, k),
            GraphType::ONNG => self
                .onng
                .as_ref()
                .expect("onng should be initialized for ONNG type")
                .search(query, k),
            GraphType::PANNG => self
                .panng
                .as_ref()
                .expect("panng should be initialized for PANNG type")
                .search(query, k),
            GraphType::Delaunay => self
                .delaunay
                .as_ref()
                .expect("delaunay should be initialized for Delaunay type")
                .search(query, k),
            GraphType::RNG => self
                .rng
                .as_ref()
                .expect("rng should be initialized for RNG type")
                .search(query, k),
        }
    }
}

impl VectorIndex for GraphIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        let data = match self.graph_type {
            GraphType::NSW => {
                &mut self
                    .nsw
                    .as_mut()
                    .expect("nsw should be initialized for NSW type")
                    .data
            }
            GraphType::ONNG => {
                &mut self
                    .onng
                    .as_mut()
                    .expect("onng should be initialized for ONNG type")
                    .data
            }
            GraphType::PANNG => {
                &mut self
                    .panng
                    .as_mut()
                    .expect("panng should be initialized for PANNG type")
                    .data
            }
            GraphType::Delaunay => {
                &mut self
                    .delaunay
                    .as_mut()
                    .expect("delaunay should be initialized for Delaunay type")
                    .data
            }
            GraphType::RNG => {
                &mut self
                    .rng
                    .as_mut()
                    .expect("rng should be initialized for RNG type")
                    .data
            }
        };

        data.push((uri, vector));
        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let query_f32 = query.as_f32();
        let results = self.search_internal(&query_f32, k);

        let data = match self.graph_type {
            GraphType::NSW => {
                &self
                    .nsw
                    .as_ref()
                    .expect("nsw should be initialized for NSW type")
                    .data
            }
            GraphType::ONNG => {
                &self
                    .onng
                    .as_ref()
                    .expect("onng should be initialized for ONNG type")
                    .data
            }
            GraphType::PANNG => {
                &self
                    .panng
                    .as_ref()
                    .expect("panng should be initialized for PANNG type")
                    .data
            }
            GraphType::Delaunay => {
                &self
                    .delaunay
                    .as_ref()
                    .expect("delaunay should be initialized for Delaunay type")
                    .data
            }
            GraphType::RNG => {
                &self
                    .rng
                    .as_ref()
                    .expect("rng should be initialized for RNG type")
                    .data
            }
        };

        Ok(results
            .into_iter()
            .map(|(idx, dist)| (data[idx].0.clone(), dist))
            .collect())
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let query_f32 = query.as_f32();
        let all_results = self.search_internal(&query_f32, 1000);

        let data = match self.graph_type {
            GraphType::NSW => {
                &self
                    .nsw
                    .as_ref()
                    .expect("nsw should be initialized for NSW type")
                    .data
            }
            GraphType::ONNG => {
                &self
                    .onng
                    .as_ref()
                    .expect("onng should be initialized for ONNG type")
                    .data
            }
            GraphType::PANNG => {
                &self
                    .panng
                    .as_ref()
                    .expect("panng should be initialized for PANNG type")
                    .data
            }
            GraphType::Delaunay => {
                &self
                    .delaunay
                    .as_ref()
                    .expect("delaunay should be initialized for Delaunay type")
                    .data
            }
            GraphType::RNG => {
                &self
                    .rng
                    .as_ref()
                    .expect("rng should be initialized for RNG type")
                    .data
            }
        };

        Ok(all_results
            .into_iter()
            .filter(|(_, dist)| *dist <= threshold)
            .map(|(idx, dist)| (data[idx].0.clone(), dist))
            .collect())
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        let data = match self.graph_type {
            GraphType::NSW => {
                &self
                    .nsw
                    .as_ref()
                    .expect("nsw should be initialized for NSW type")
                    .data
            }
            GraphType::ONNG => {
                &self
                    .onng
                    .as_ref()
                    .expect("onng should be initialized for ONNG type")
                    .data
            }
            GraphType::PANNG => {
                &self
                    .panng
                    .as_ref()
                    .expect("panng should be initialized for PANNG type")
                    .data
            }
            GraphType::Delaunay => {
                &self
                    .delaunay
                    .as_ref()
                    .expect("delaunay should be initialized for Delaunay type")
                    .data
            }
            GraphType::RNG => {
                &self
                    .rng
                    .as_ref()
                    .expect("rng should be initialized for RNG type")
                    .data
            }
        };

        data.iter().find(|(u, _)| u == uri).map(|(_, v)| v)
    }
}

// Add dependencies
use petgraph;
// Note: Replaced with scirs2_core::random

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nsw_graph() {
        let config = GraphIndexConfig {
            graph_type: GraphType::NSW,
            num_neighbors: 10,
            ..Default::default()
        };

        let mut index = GraphIndex::new(config);

        // Insert test vectors
        for i in 0..50 {
            let vector = Vector::new(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
            index.insert(format!("vec_{i}"), vector).unwrap();
        }

        index.build().unwrap();

        // Search for nearest neighbors
        let query = Vector::new(vec![25.0, 50.0, 75.0]);
        let results = index.search_knn(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, "vec_25"); // Exact match
    }

    #[test]
    fn test_onng_graph() {
        let config = GraphIndexConfig {
            graph_type: GraphType::ONNG,
            num_neighbors: 8,
            ..Default::default()
        };

        let mut index = GraphIndex::new(config);

        // Insert test vectors in a circle
        for i in 0..20 {
            let angle = (i as f32) * 2.0 * std::f32::consts::PI / 20.0;
            let vector = Vector::new(vec![angle.cos(), angle.sin()]);
            index.insert(format!("vec_{i}"), vector).unwrap();
        }

        index.build().unwrap();

        // Search for nearest neighbors
        let query = Vector::new(vec![1.0, 0.0]);
        let results = index.search_knn(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_panng_graph() {
        let config = GraphIndexConfig {
            graph_type: GraphType::PANNG,
            num_neighbors: 5,
            enable_pruning: true,
            ..Default::default()
        };

        let mut index = GraphIndex::new(config);

        // Insert test vectors
        for i in 0..30 {
            let vector = Vector::new(vec![(i as f32).sin(), (i as f32).cos(), (i as f32) / 10.0]);
            index.insert(format!("vec_{i}"), vector).unwrap();
        }

        index.build().unwrap();

        // Search for nearest neighbors
        let query = Vector::new(vec![0.0, 1.0, 0.0]);
        let results = index.search_knn(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
    }
}
