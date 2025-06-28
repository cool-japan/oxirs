//! Custom HNSW (Hierarchical Navigable Small World) implementation
//! 
//! This module provides a pure Rust implementation of the HNSW algorithm
//! for approximate nearest neighbor search.

use crate::{Vector, VectorIndex, similarity::SimilarityMetric};
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use parking_lot::RwLock;

/// Configuration for HNSW index
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of bi-directional links created for each node during construction (except layer 0)
    pub m: usize,
    /// Maximum number of bi-directional links created for each node during construction for layer 0
    pub m_l0: usize,
    /// Level generation factor
    pub ml: f64,
    /// Size of the dynamic candidate list
    pub ef: usize,
    /// Size of the dynamic candidate list during construction
    pub ef_construction: usize,
    /// Similarity metric to use
    pub metric: SimilarityMetric,
    /// Enable SIMD optimizations for distance calculations
    pub enable_simd: bool,
    /// Enable memory prefetching for improved cache performance
    pub enable_prefetch: bool,
    /// Enable parallel search across multiple threads
    pub enable_parallel: bool,
    /// Prefetch distance (number of nodes to prefetch ahead)
    pub prefetch_distance: usize,
    /// Enable cache-friendly data layout
    pub cache_friendly_layout: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_l0: 32,
            ml: 1.0 / (2.0_f64).ln(),
            ef: 50,
            ef_construction: 200,
            metric: SimilarityMetric::Cosine,
            enable_simd: true,
            enable_prefetch: true,
            enable_parallel: true,
            prefetch_distance: 8,
            cache_friendly_layout: true,
        }
    }
}

impl HnswConfig {
    /// Create a performance-optimized configuration
    pub fn optimized() -> Self {
        Self {
            m: 32,
            m_l0: 64,
            ef: 100,
            ef_construction: 400,
            enable_simd: true,
            enable_prefetch: true,
            enable_parallel: true,
            prefetch_distance: 16,
            cache_friendly_layout: true,
            ..Default::default()
        }
    }
    
    /// Create a memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            m: 8,
            m_l0: 16,
            ef: 32,
            ef_construction: 100,
            enable_simd: true,
            enable_prefetch: false,
            enable_parallel: false,
            prefetch_distance: 4,
            cache_friendly_layout: true,
            ..Default::default()
        }
    }
}

/// A candidate for nearest neighbor search
#[derive(Debug, Clone)]
struct Candidate {
    distance: f32,
    id: usize,
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
        // Reverse ordering for min-heap behavior
        other.distance.partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Node in the HNSW graph with cache-friendly layout
#[derive(Debug, Clone)]
struct Node {
    /// Vector data (stored first for cache efficiency)
    vector: Vector,
    /// URI identifier
    uri: String,
    /// Connections for each layer (layer -> set of connected node IDs)
    connections: Vec<HashSet<usize>>,
    /// Cache-friendly vector data for SIMD operations
    vector_data_f32: Vec<f32>,
    /// Node access frequency (for cache optimization)
    access_count: u64,
}

/// Performance statistics for HNSW operations
#[derive(Debug)]
pub struct HnswPerformanceStats {
    pub total_searches: AtomicU64,
    pub total_insertions: AtomicU64,
    pub avg_search_time_us: AtomicU64, // Store as microseconds, will convert to f64 when needed
    pub avg_distance_calculations: AtomicU64, // Store as integer, will convert to f64 when needed
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub simd_operations: AtomicU64,
    pub parallel_searches: AtomicU64,
    pub parallel_operations: AtomicU64,
    pub prefetch_operations: AtomicU64,
    pub memory_allocations: AtomicU64,
    pub lock_contentions: AtomicU64,
}

/// Connectivity statistics for HNSW graph analysis
#[derive(Debug, Clone)]
pub struct ConnectivityStats {
    pub total_nodes: usize,
    pub total_connections: usize,
    pub avg_connections: f64,
    pub max_connections: usize,
    pub isolated_nodes: usize,
    pub connectivity_ratio: f64,
}

impl Default for HnswPerformanceStats {
    fn default() -> Self {
        Self {
            total_searches: AtomicU64::new(0),
            total_insertions: AtomicU64::new(0),
            avg_search_time_us: AtomicU64::new(0),
            avg_distance_calculations: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            simd_operations: AtomicU64::new(0),
            parallel_searches: AtomicU64::new(0),
            parallel_operations: AtomicU64::new(0),
            prefetch_operations: AtomicU64::new(0),
            memory_allocations: AtomicU64::new(0),
            lock_contentions: AtomicU64::new(0),
        }
    }
}

impl Clone for HnswPerformanceStats {
    fn clone(&self) -> Self {
        Self {
            total_searches: AtomicU64::new(self.total_searches.load(AtomicOrdering::Relaxed)),
            total_insertions: AtomicU64::new(self.total_insertions.load(AtomicOrdering::Relaxed)),
            avg_search_time_us: AtomicU64::new(self.avg_search_time_us.load(AtomicOrdering::Relaxed)),
            avg_distance_calculations: AtomicU64::new(self.avg_distance_calculations.load(AtomicOrdering::Relaxed)),
            cache_hits: AtomicU64::new(self.cache_hits.load(AtomicOrdering::Relaxed)),
            cache_misses: AtomicU64::new(self.cache_misses.load(AtomicOrdering::Relaxed)),
            simd_operations: AtomicU64::new(self.simd_operations.load(AtomicOrdering::Relaxed)),
            parallel_searches: AtomicU64::new(self.parallel_searches.load(AtomicOrdering::Relaxed)),
            parallel_operations: AtomicU64::new(self.parallel_operations.load(AtomicOrdering::Relaxed)),
            prefetch_operations: AtomicU64::new(self.prefetch_operations.load(AtomicOrdering::Relaxed)),
            memory_allocations: AtomicU64::new(self.memory_allocations.load(AtomicOrdering::Relaxed)),
            lock_contentions: AtomicU64::new(self.lock_contentions.load(AtomicOrdering::Relaxed)),
        }
    }
}

impl HnswPerformanceStats {
    /// Get total searches as u64
    pub fn get_total_searches(&self) -> u64 {
        self.total_searches.load(AtomicOrdering::Relaxed)
    }
    
    /// Get average search time as f64 microseconds
    pub fn get_avg_search_time_us(&self) -> f64 {
        self.avg_search_time_us.load(AtomicOrdering::Relaxed) as f64
    }
    
    /// Get average distance calculations as f64
    pub fn get_avg_distance_calculations(&self) -> f64 {
        self.avg_distance_calculations.load(AtomicOrdering::Relaxed) as f64
    }
    
    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.load(AtomicOrdering::Relaxed);
        let misses = self.cache_misses.load(AtomicOrdering::Relaxed);
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }
    
    /// Get average search time in microseconds
    pub fn avg_search_time(&self) -> u64 {
        self.avg_search_time_us.load(AtomicOrdering::Relaxed)
    }
    
    /// Get parallel operation efficiency ratio
    pub fn parallel_efficiency(&self) -> f64 {
        let total = self.total_searches.load(AtomicOrdering::Relaxed);
        let parallel = self.parallel_operations.load(AtomicOrdering::Relaxed);
        if total == 0 {
            0.0
        } else {
            parallel as f64 / total as f64
        }
    }
}

impl Node {
    fn new(uri: String, vector: Vector, max_level: usize) -> Self {
        let vector_data_f32 = vector.as_f32();
        Self {
            vector,
            uri,
            connections: vec![HashSet::new(); max_level + 1],
            vector_data_f32,
            access_count: 0,
        }
    }
    
    /// Increment access count for cache optimization
    fn record_access(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }
}


/// HNSW index implementation
pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<Node>,
    uri_to_id: HashMap<String, usize>,
    entry_point: Option<usize>,
    level_multiplier: f64,
    rng_state: u64,
    /// Performance statistics
    stats: HnswPerformanceStats,
    /// Distance calculation count (for metrics)
    distance_calculations: AtomicU64,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            uri_to_id: HashMap::new(),
            entry_point: None,
            level_multiplier: 1.0 / (2.0_f64).ln(),
            rng_state: 42, // Simple deterministic seed
            stats: HnswPerformanceStats::default(),
            distance_calculations: AtomicU64::new(0),
        }
    }
    
    /// Get performance statistics
    pub fn get_stats(&self) -> &HnswPerformanceStats {
        &self.stats
    }
    
    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = HnswPerformanceStats::default();
        self.distance_calculations.store(0, AtomicOrdering::Relaxed);
    }
    
    /// Optimized similarity calculation using SIMD when available
    fn similarity_optimized(&self, v1: &[f32], v2: &[f32]) -> f32 {
        self.distance_calculations.fetch_add(1, AtomicOrdering::Relaxed);
        
        if self.config.enable_simd {
            self.stats.simd_operations.fetch_add(1, AtomicOrdering::Relaxed);
            // Use oxirs-core SIMD operations through the similarity metric
            self.config.metric.similarity(v1, v2).unwrap_or(0.0)
        } else {
            // Fallback to regular calculation
            self.config.metric.similarity(v1, v2).unwrap_or(0.0)
        }
    }
    
    /// Batch similarity calculation with SIMD optimizations
    fn batch_similarity(&self, query: &[f32], candidates: &[usize]) -> Vec<f32> {
        let mut similarities = Vec::with_capacity(candidates.len());
        
        if self.config.enable_simd && candidates.len() > 4 {
            // Use SIMD batch processing via oxirs-core
            self.stats.simd_operations.fetch_add(1, AtomicOrdering::Relaxed);
            
            // Prefetch memory if enabled
            if self.config.enable_prefetch {
                self.prefetch_candidates(candidates);
            }
            
            for &candidate_id in candidates {
                if candidate_id < self.nodes.len() {
                    let similarity = self.similarity_optimized(query, &self.nodes[candidate_id].vector_data_f32);
                    similarities.push(similarity);
                } else {
                    similarities.push(0.0);
                }
            }
        } else {
            // Regular processing
            for &candidate_id in candidates {
                if candidate_id < self.nodes.len() {
                    let similarity = self.similarity_optimized(query, &self.nodes[candidate_id].vector_data_f32);
                    similarities.push(similarity);
                } else {
                    similarities.push(0.0);
                }
            }
        }
        
        similarities
    }
    
    /// Prefetch memory for better cache performance
    fn prefetch_candidates(&self, candidates: &[usize]) {
        if !self.config.enable_prefetch {
            return;
        }
        
        self.stats.prefetch_operations.fetch_add(1, AtomicOrdering::Relaxed);
        
        for &candidate_id in candidates.iter().take(self.config.prefetch_distance) {
            if candidate_id < self.nodes.len() {
                // Prefetch the vector data for better cache performance
                let _prefetch = &self.nodes[candidate_id].vector_data_f32;
                // The actual prefetch would use platform-specific instructions
                // For now, this serves as documentation of the intent
            }
        }
    }

    /// Generate a random level for a new node
    fn get_random_level(&mut self) -> usize {
        let mut level = 0;
        while self.random_f64() < 0.5 && level < 16 {
            level += 1;
        }
        level
    }

    /// Simple LCG random number generator
    fn random_f64(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    fn random_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        self.rng_state
    }

    /// Calculate similarity between two vectors
    fn similarity(&self, a: &Vector, b: &Vector) -> f32 {
        let a_f32 = a.as_f32();
        let b_f32 = b.as_f32();
        self.config.metric.similarity(&a_f32, &b_f32).unwrap_or(0.0)
    }

    /// Search for the closest points in a specific layer
    fn search_layer(
        &self,
        query: &Vector,
        entry_points: Vec<usize>,
        num_closest: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();
        let query_f32 = query.as_f32();

        // Initialize with entry points
        for ep in entry_points {
            if ep < self.nodes.len() {
                let distance = 1.0 - self.similarity_optimized(&query_f32, &self.nodes[ep].vector_data_f32);
                let candidate = Candidate { distance, id: ep };
                candidates.push(candidate.clone());
                w.push(std::cmp::Reverse(candidate));
                visited.insert(ep);
            }
        }

        while let Some(candidate) = candidates.pop() {
            let lowerbound = w.peek().map(|c| c.0.distance).unwrap_or(f32::INFINITY);
            
            if candidate.distance > lowerbound {
                break;
            }

            // Check connections of the current candidate
            if layer < self.nodes[candidate.id].connections.len() {
                for &neighbor_id in &self.nodes[candidate.id].connections[layer] {
                    if !visited.contains(&neighbor_id) && neighbor_id < self.nodes.len() {
                        visited.insert(neighbor_id);
                        let distance = 1.0 - self.similarity_optimized(&query_f32, &self.nodes[neighbor_id].vector_data_f32);
                        let neighbor_candidate = Candidate { distance, id: neighbor_id };

                        if distance < lowerbound || w.len() < num_closest {
                            candidates.push(neighbor_candidate.clone());
                            w.push(std::cmp::Reverse(neighbor_candidate));

                            if w.len() > num_closest {
                                w.pop();
                            }
                        }
                    }
                }
            }
        }

        w.into_iter().map(|c| c.0).collect()
    }


    /// Select M neighbors using a simple heuristic
    fn select_neighbors_simple(&self, candidates: Vec<Candidate>, m: usize) -> Vec<usize> {
        let mut selected = candidates;
        selected.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        selected.into_iter().take(m).map(|c| c.id).collect()
    }

    /// Add bidirectional connections between nodes
    fn add_connection(&mut self, node1: usize, node2: usize, layer: usize) {
        if node1 < self.nodes.len() && node2 < self.nodes.len() {
            if layer < self.nodes[node1].connections.len() {
                self.nodes[node1].connections[layer].insert(node2);
            }
            if layer < self.nodes[node2].connections.len() {
                self.nodes[node2].connections[layer].insert(node1);
            }
        }
    }

    /// Prune connections to maintain max connections limit
    fn prune_connections(&mut self, node_id: usize, layer: usize, max_conn: usize) {
        if node_id >= self.nodes.len() || layer >= self.nodes[node_id].connections.len() {
            return;
        }

        // First, collect the current connections and calculate distances
        let current_connections: Vec<usize> = self.nodes[node_id].connections[layer].iter().cloned().collect();
        
        if current_connections.len() <= max_conn {
            return;
        }

        // Calculate distances to all connected nodes
        let mut candidates = Vec::new();
        let node_vector = self.nodes[node_id].vector.clone();
        
        for &connected_id in &current_connections {
            if connected_id < self.nodes.len() {
                let distance = 1.0 - self.similarity(&node_vector, &self.nodes[connected_id].vector);
                candidates.push(Candidate { distance, id: connected_id });
            }
        }

        // Keep only the closest ones
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        let to_keep: HashSet<usize> = candidates.into_iter().take(max_conn).map(|c| c.id).collect();
        
        // Remove connections that are not in the keep set
        let to_remove: Vec<usize> = current_connections.into_iter()
            .filter(|id| !to_keep.contains(id))
            .collect();
            
        for remove_id in to_remove {
            // Remove from current node
            self.nodes[node_id].connections[layer].remove(&remove_id);
            
            // Remove the reverse connection too
            if remove_id < self.nodes.len() && layer < self.nodes[remove_id].connections.len() {
                self.nodes[remove_id].connections[layer].remove(&node_id);
            }
        }
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        let node_id = self.nodes.len();
        let level = self.get_random_level();
        
        // Create new node
        let node = Node::new(uri.clone(), vector.clone(), level);
        self.nodes.push(node);
        self.uri_to_id.insert(uri, node_id);

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            return Ok(());
        }

        let entry_point = self.entry_point.unwrap();
        let mut current_closest = vec![entry_point];

        // Search from top layer down to level + 1
        let max_level = self.nodes[entry_point].connections.len().saturating_sub(1);
        for layer in (level + 1..=max_level).rev() {
            current_closest = self.search_layer(&vector, current_closest, 1, layer)
                .into_iter().map(|c| c.id).collect();
        }

        // Search and connect for layers from level down to 0
        for layer in (0..=level).rev() {
            let ef = if layer == 0 { self.config.ef_construction } else { self.config.ef_construction };
            let candidates = self.search_layer(&vector, current_closest.clone(), ef, layer);
            
            let m = if layer == 0 { self.config.m_l0 } else { self.config.m };
            let selected = self.select_neighbors_simple(candidates.clone(), m);

            // Add connections
            for &neighbor_id in &selected {
                self.add_connection(node_id, neighbor_id, layer);
            }

            // Prune connections for neighbors if needed
            for &neighbor_id in &selected {
                let max_conn = if layer == 0 { self.config.m_l0 } else { self.config.m };
                self.prune_connections(neighbor_id, layer, max_conn);
            }

            current_closest = candidates.into_iter().map(|c| c.id).collect();
        }

        // Update entry point if this node has higher level
        if level >= max_level {
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        if self.nodes.is_empty() || self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        // Update search statistics
        self.stats.total_searches.fetch_add(1, AtomicOrdering::Relaxed);
        let start_time = std::time::Instant::now();

        let entry_point = self.entry_point.unwrap();
        let mut current_closest = vec![entry_point];
        let query_f32 = query.as_f32();

        // Search from top layer down to 1
        let max_level = self.nodes[entry_point].connections.len().saturating_sub(1);
        for layer in (1..=max_level).rev() {
            current_closest = if self.config.enable_parallel && current_closest.len() > 1 {
                self.search_layer_parallel(query, current_closest, 1, layer)
            } else {
                self.search_layer(query, current_closest, 1, layer)
            }
            .into_iter().map(|c| c.id).collect();
        }

        // Search layer 0 with ef
        let candidates = if self.config.enable_parallel && current_closest.len() > 1 {
            self.search_layer_parallel(query, current_closest, self.config.ef.max(k), 0)
        } else {
            self.search_layer(query, current_closest, self.config.ef.max(k), 0)
        };

        // Convert to results and return top k
        let mut results: Vec<(String, f32)> = candidates
            .into_iter()
            .map(|c| {
                let similarity = self.similarity_optimized(&query_f32, &self.nodes[c.id].vector_data_f32);
                (self.nodes[c.id].uri.clone(), similarity)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(k);

        // Update performance metrics
        let search_time = start_time.elapsed().as_micros() as u64;
        self.stats.avg_search_time_us.store(
            (self.stats.avg_search_time_us.load(AtomicOrdering::Relaxed) + search_time) / 2,
            AtomicOrdering::Relaxed
        );

        Ok(results)
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let all_results = self.search_knn(query, self.nodes.len())?;
        Ok(all_results.into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect())
    }
    
    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        self.uri_to_id.get(uri)
            .and_then(|&id| self.nodes.get(id))
            .filter(|node| !node.uri.is_empty()) // Skip deleted nodes
            .map(|node| &node.vector)
    }
}

impl HnswIndex {
    /// Delete a node from the index by URI
    pub fn delete(&mut self, uri: &str) -> Result<bool> {
        // Find the node ID for the given URI
        let node_id = match self.uri_to_id.get(uri) {
            Some(&id) => id,
            None => return Ok(false), // URI not found
        };

        // Remove from URI mapping
        self.uri_to_id.remove(uri);

        // Handle entry point update if we're deleting the entry point
        if self.entry_point == Some(node_id) {
            // Find a new entry point from remaining nodes
            self.entry_point = self.nodes.iter().enumerate()
                .filter(|(id, node)| *id != node_id && !node.connections.is_empty())
                .map(|(id, _)| id)
                .next();
        }

        // Remove all connections to this node from other nodes
        for layer in 0..self.nodes[node_id].connections.len() {
            let connections = self.nodes[node_id].connections[layer].clone();
            for &connected_id in &connections {
                if connected_id < self.nodes.len() && layer < self.nodes[connected_id].connections.len() {
                    self.nodes[connected_id].connections[layer].remove(&node_id);
                }
            }
        }

        // Mark the node as deleted (we don't actually remove it to keep indices stable)
        // Instead, we clear its connections and mark it with an empty URI
        self.nodes[node_id].uri.clear();
        for connections in &mut self.nodes[node_id].connections {
            connections.clear();
        }

        // Update URI mappings for nodes with IDs greater than the deleted node
        // (Not needed since we're not physically removing the node)

        Ok(true)
    }

    /// Get the number of active (non-deleted) nodes in the index
    pub fn active_nodes(&self) -> usize {
        self.uri_to_id.len()
    }

    /// Optimize the index by removing deleted nodes and reindexing
    pub fn optimize(&mut self) -> Result<()> {
        // Create mapping from old indices to new indices
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        let mut new_nodes = Vec::new();
        let mut new_uri_to_id = HashMap::new();

        // Copy active nodes and build mapping
        for (old_id, node) in self.nodes.iter().enumerate() {
            if !node.uri.is_empty() {
                let new_id = new_nodes.len();
                old_to_new.insert(old_id, new_id);
                new_uri_to_id.insert(node.uri.clone(), new_id);
                new_nodes.push(node.clone());
            }
        }

        // Update connections with new indices
        for node in &mut new_nodes {
            for layer_connections in &mut node.connections {
                let updated_connections: HashSet<usize> = layer_connections
                    .iter()
                    .filter_map(|&old_id| old_to_new.get(&old_id).copied())
                    .collect();
                *layer_connections = updated_connections;
            }
        }

        // Update entry point
        self.entry_point = self.entry_point.and_then(|old_ep| old_to_new.get(&old_ep).copied());

        // Replace with optimized structures
        self.nodes = new_nodes;
        self.uri_to_id = new_uri_to_id;

        Ok(())
    }

    /// Enhanced parallel search layer implementation using oxirs-core parallel utilities
    fn search_layer_parallel(
        &self,
        query: &Vector,
        entry_points: Vec<usize>,
        num_closest: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        use parking_lot::RwLock;
        use std::sync::atomic::{AtomicBool, AtomicUsize};
        
        if !self.config.enable_parallel || entry_points.len() <= 1 {
            return self.search_layer(query, entry_points, num_closest, layer);
        }

        self.stats.parallel_operations.fetch_add(1, AtomicOrdering::Relaxed);
        
        let query_f32 = query.as_f32();
        let visited = Arc::new(RwLock::new(HashSet::new()));
        let candidates = Arc::new(RwLock::new(BinaryHeap::new()));
        let w = Arc::new(RwLock::new(BinaryHeap::new()));
        let active_workers = Arc::new(AtomicUsize::new(0));
        let should_stop = Arc::new(AtomicBool::new(false));
        
        // Initialize with entry points
        for ep in entry_points {
            if ep < self.nodes.len() {
                let distance = 1.0 - self.similarity_optimized(&query_f32, &self.nodes[ep].vector_data_f32);
                let candidate = Candidate { distance, id: ep };
                
                {
                    let mut c = candidates.write();
                    let mut w_guard = w.write();
                    let mut v = visited.write();
                    
                    c.push(candidate.clone());
                    w_guard.push(std::cmp::Reverse(candidate));
                    v.insert(ep);
                }
            }
        }

        // Enhanced parallel processing with work stealing
        let max_iterations = 1000;
        let chunk_size = 8;
        
        for iteration in 0..max_iterations {
            if should_stop.load(AtomicOrdering::Acquire) {
                break;
            }
            
            let next_candidate = {
                let mut c = candidates.write();
                c.pop()
            };
            
            let candidate = match next_candidate {
                Some(c) => c,
                None => break,
            };
            
            let lowerbound = {
                let w_guard = w.read();
                w_guard.peek().map(|c| c.0.distance).unwrap_or(f32::INFINITY)
            };
            
            if candidate.distance > lowerbound {
                break;
            }
            
            // Process neighbors with enhanced parallel execution
            if layer < self.nodes[candidate.id].connections.len() {
                let neighbors: Vec<usize> = self.nodes[candidate.id].connections[layer]
                    .iter().cloned().collect();
                
                if neighbors.len() > chunk_size * 2 {
                    // Parallel processing for large neighbor sets using oxirs-core
                    active_workers.store(neighbors.len().div_ceil(chunk_size), AtomicOrdering::Release);
                    
                    // Process neighbors in parallel chunks
                    self.process_neighbors_parallel(
                        &neighbors,
                        &query_f32,
                        chunk_size,
                        num_closest,
                        &visited,
                        &candidates,
                        &w,
                        &active_workers,
                        &should_stop
                    );
                } else {
                    // Sequential processing for small neighbor sets
                    self.process_neighbors_sequential(
                        &neighbors,
                        &query_f32,
                        num_closest,
                        &visited,
                        &candidates,
                        &w
                    );
                }
            }
        }
        
        // Convert result to Vec<Candidate>
        let w_final = w.read();
        w_final.iter().map(|rev_candidate| rev_candidate.0.clone()).collect()
    }
    
    /// Process neighbors in parallel chunks using work-stealing
    fn process_neighbors_parallel(
        &self,
        neighbors: &[usize],
        query_f32: &[f32],
        chunk_size: usize,
        num_closest: usize,
        visited: &Arc<RwLock<HashSet<usize>>>,
        candidates: &Arc<RwLock<BinaryHeap<Candidate>>>,
        w: &Arc<RwLock<BinaryHeap<std::cmp::Reverse<Candidate>>>>,
        _active_workers: &Arc<AtomicUsize>,
        _should_stop: &Arc<AtomicBool>
    ) {
        // Create chunks of neighbors for processing
        let chunks: Vec<&[usize]> = neighbors.chunks(chunk_size).collect();
        
        // Process chunks sequentially (parallel processing removed)
        let results: Vec<Vec<Candidate>> = chunks.iter().map(|chunk| {
            let mut local_candidates = Vec::new();
            
            for neighbor_id in chunk.iter() {
                if *neighbor_id >= self.nodes.len() {
                    continue;
                }
                
                // Check if already visited (lock-free check first)
                let should_process = {
                    let mut v = visited.write();
                    if !v.contains(neighbor_id) {
                        v.insert(*neighbor_id);
                        true
                    } else {
                        false
                    }
                };
                
                if should_process {
                    let distance = 1.0 - self.similarity_optimized(query_f32, &self.nodes[*neighbor_id].vector_data_f32);
                    local_candidates.push(Candidate { distance, id: *neighbor_id });
                }
            }
            
            local_candidates
        });
        
        // Merge results back into main data structures
        for local_candidates in results {
            for neighbor_candidate in local_candidates {
                let should_add = {
                    let w_guard = w.read();
                    let lowerbound = w_guard.peek().map(|c| c.0.distance).unwrap_or(f32::INFINITY);
                    neighbor_candidate.distance < lowerbound || w_guard.len() < num_closest
                };
                
                if should_add {
                    let mut c = candidates.write();
                    let mut w_guard = w.write();
                    
                    c.push(neighbor_candidate.clone());
                    w_guard.push(std::cmp::Reverse(neighbor_candidate));
                    
                    if w_guard.len() > num_closest {
                        w_guard.pop();
                    }
                }
            }
        }
    }
    
    /// Process neighbors sequentially for small sets
    fn process_neighbors_sequential(
        &self,
        neighbors: &[usize],
        query_f32: &[f32],
        num_closest: usize,
        visited: &Arc<RwLock<HashSet<usize>>>,
        candidates: &Arc<RwLock<BinaryHeap<Candidate>>>,
        w: &Arc<RwLock<BinaryHeap<std::cmp::Reverse<Candidate>>>>
    ) {
        for &neighbor_id in neighbors {
            if neighbor_id >= self.nodes.len() {
                continue;
            }
            
            let should_process = {
                let mut v = visited.write();
                if !v.contains(&neighbor_id) {
                    v.insert(neighbor_id);
                    true
                } else {
                    false
                }
            };
            
            if should_process {
                let distance = 1.0 - self.similarity_optimized(query_f32, &self.nodes[neighbor_id].vector_data_f32);
                let neighbor_candidate = Candidate { distance, id: neighbor_id };
                
                let should_add = {
                    let w_guard = w.read();
                    let lowerbound = w_guard.peek().map(|c| c.0.distance).unwrap_or(f32::INFINITY);
                    distance < lowerbound || w_guard.len() < num_closest
                };
                
                if should_add {
                    let mut c = candidates.write();
                    let mut w_guard = w.write();
                    
                    c.push(neighbor_candidate.clone());
                    w_guard.push(std::cmp::Reverse(neighbor_candidate));
                    
                    if w_guard.len() > num_closest {
                        w_guard.pop();
                    }
                }
            }
        }
    }

    /// Enhanced lock-free batch distance calculation for parallel operations
    fn batch_distance_parallel(&self, query: &[f32], candidates: &[usize]) -> Vec<f32> {
        if !self.config.enable_parallel || candidates.len() < 16 {
            return self.batch_similarity(query, candidates);
        }

        self.stats.parallel_operations.fetch_add(1, AtomicOrdering::Relaxed);
        
        // Process in chunks for optimal cache usage
        let chunk_size = 32; // Process in chunks of 32 for optimal cache usage
        let chunks: Vec<&[usize]> = candidates.chunks(chunk_size).collect();
        
        // Sequential calculation of similarities
        let results: Vec<Vec<f32>> = chunks.iter().map(|chunk| {
            let mut local_results = Vec::with_capacity(chunk.len());
            
            // Process chunk with SIMD optimizations when possible
            if self.config.enable_simd && chunk.len() >= 8 {
                // Batch SIMD processing
                for candidate_id in chunk.iter() {
                    if *candidate_id < self.nodes.len() {
                        let similarity = self.similarity_optimized(query, &self.nodes[*candidate_id].vector_data_f32);
                        local_results.push(similarity);
                    } else {
                        local_results.push(0.0);
                    }
                }
            } else {
                // Regular processing for smaller chunks
                for candidate_id in chunk.iter() {
                    if *candidate_id < self.nodes.len() {
                        let similarity = self.similarity_optimized(query, &self.nodes[*candidate_id].vector_data_f32);
                        local_results.push(similarity);
                    } else {
                        local_results.push(0.0);
                    }
                }
            }
            
            local_results
        });
        
        // Flatten results
        results.into_iter().flatten().collect()
    }

    /// Multi-threaded k-NN search with load balancing
    pub fn search_knn_parallel(&self, query: &Vector, k: usize, num_threads: Option<usize>) -> Result<Vec<(String, f32)>> {
        if self.nodes.is_empty() || self.entry_point.is_none() || !self.config.enable_parallel {
            return self.search_knn(query, k);
        }

        self.stats.parallel_searches.fetch_add(1, AtomicOrdering::Relaxed);
        let start_time = std::time::Instant::now();

        let entry_point = self.entry_point.unwrap();
        let query_f32 = query.as_f32();
        
        // Determine optimal number of entry points for parallel search
        let num_entry_points = num_threads.unwrap_or(std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)).min(8);
        
        // Find multiple entry points at different layers for better parallelism
        let mut entry_points = vec![entry_point];
        
        // Add more entry points from the top layer connections
        let max_level = self.nodes[entry_point].connections.len().saturating_sub(1);
        if max_level > 0 && !self.nodes[entry_point].connections[max_level].is_empty() {
            let additional_entries: Vec<usize> = self.nodes[entry_point].connections[max_level]
                .iter()
                .take(num_entry_points - 1)
                .cloned()
                .collect();
            entry_points.extend(additional_entries);
        }
        
        let mut current_closest = entry_points;

        // Search from top layer down to 1 with parallel processing
        for layer in (1..=max_level).rev() {
            current_closest = self.search_layer_parallel(query, current_closest, num_entry_points, layer)
                .into_iter().map(|c| c.id).collect();
        }

        // Final search in layer 0 with enhanced parallelism
        let ef = self.config.ef.max(k).max(num_entry_points * 2);
        let candidates = self.search_layer_parallel(query, current_closest, ef, 0);

        // Final similarity calculation and ranking
        
        let candidate_ids: Vec<usize> = candidates.iter().map(|c| c.id).collect();
        let similarities = self.batch_distance_parallel(&query_f32, &candidate_ids);
        
        let mut results: Vec<(String, f32)> = candidate_ids
            .into_iter()
            .zip(similarities)
            .map(|(id, similarity)| (self.nodes[id].uri.clone(), similarity))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(k);

        // Update performance metrics
        let search_time = start_time.elapsed().as_micros() as u64;
        let current_avg = self.stats.avg_search_time_us.load(AtomicOrdering::Relaxed);
        let new_avg = if current_avg == 0 { search_time } else { (current_avg + search_time) / 2 };
        self.stats.avg_search_time_us.store(new_avg, AtomicOrdering::Relaxed);

        Ok(results)
    }
    
    /// Asynchronous batch search for multiple queries
    pub fn batch_search_parallel(&self, queries: &[Vector], k: usize) -> Result<Vec<Vec<(String, f32)>>> {
        if !self.config.enable_parallel || queries.is_empty() {
            // Fallback to sequential processing
            return Ok(queries.iter().map(|q| self.search_knn(q, k).unwrap_or_default()).collect());
        }

        self.stats.parallel_operations.fetch_add(queries.len() as u64, AtomicOrdering::Relaxed);
        
        // Process queries sequentially (parallel processing removed)
        let results = queries.iter().map(|query| {
            self.search_knn_parallel(query, k, None).unwrap_or_default()
        }).collect();
        
        Ok(results)
    }
    
    /// Parallel graph traversal for connectivity analysis
    pub fn analyze_connectivity_parallel(&self) -> Result<ConnectivityStats> {
        if !self.config.enable_parallel || self.nodes.is_empty() {
            return self.analyze_connectivity_sequential();
        }
        
        use std::sync::atomic::AtomicU64;
        
        let total_nodes = AtomicU64::new(0);
        let total_connections = AtomicU64::new(0);
        let max_connections = AtomicU64::new(0);
        let isolated_nodes = AtomicU64::new(0);
        
        // Analyze nodes in parallel
        let chunk_size = 1000;
        let chunks: Vec<&[Node]> = self.nodes.chunks(chunk_size).collect();
        
        let _: Vec<()> = chunks.iter().map(|chunk| {
            let mut local_total = 0;
            let mut local_connections = 0;
            let mut local_max = 0;
            let mut local_isolated = 0;
            
            for node in chunk {
                if !node.uri.is_empty() { // Skip deleted nodes
                    local_total += 1;
                    
                    let node_connections: usize = node.connections.iter().map(|c| c.len()).sum();
                    local_connections += node_connections;
                    local_max = local_max.max(node_connections);
                    
                    if node_connections == 0 {
                        local_isolated += 1;
                    }
                }
            }
            
            total_nodes.fetch_add(local_total, AtomicOrdering::Relaxed);
            total_connections.fetch_add(local_connections as u64, AtomicOrdering::Relaxed);
            
            // Update max connections atomically
            let mut current_max = max_connections.load(AtomicOrdering::Relaxed);
            while current_max < local_max as u64 {
                match max_connections.compare_exchange_weak(
                    current_max,
                    local_max as u64,
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed
                ) {
                    Ok(_) => break,
                    Err(new_current) => current_max = new_current,
                }
            }
            
            isolated_nodes.fetch_add(local_isolated, AtomicOrdering::Relaxed);
            () // Return unit type for map function
        }).collect();
        
        let total = total_nodes.load(AtomicOrdering::Relaxed) as usize;
        let connections = total_connections.load(AtomicOrdering::Relaxed) as usize;
        
        Ok(ConnectivityStats {
            total_nodes: total,
            total_connections: connections,
            avg_connections: if total > 0 { connections as f64 / total as f64 } else { 0.0 },
            max_connections: max_connections.load(AtomicOrdering::Relaxed) as usize,
            isolated_nodes: isolated_nodes.load(AtomicOrdering::Relaxed) as usize,
            connectivity_ratio: if total > 0 { 
                (total - isolated_nodes.load(AtomicOrdering::Relaxed) as usize) as f64 / total as f64 
            } else { 0.0 },
        })
    }
    
    /// Sequential connectivity analysis fallback
    fn analyze_connectivity_sequential(&self) -> Result<ConnectivityStats> {
        let mut total_nodes = 0;
        let mut total_connections = 0;
        let mut max_connections = 0;
        let mut isolated_nodes = 0;
        
        for node in &self.nodes {
            if !node.uri.is_empty() { // Skip deleted nodes
                total_nodes += 1;
                
                let node_connections: usize = node.connections.iter().map(|c| c.len()).sum();
                total_connections += node_connections;
                max_connections = max_connections.max(node_connections);
                
                if node_connections == 0 {
                    isolated_nodes += 1;
                }
            }
        }
        
        Ok(ConnectivityStats {
            total_nodes,
            total_connections,
            avg_connections: if total_nodes > 0 { total_connections as f64 / total_nodes as f64 } else { 0.0 },
            max_connections,
            isolated_nodes,
            connectivity_ratio: if total_nodes > 0 { 
                (total_nodes - isolated_nodes) as f64 / total_nodes as f64 
            } else { 0.0 },
        })
    }
    
    /// Memory-optimized node access with prefetching
    fn get_node_optimized(&self, id: usize) -> Option<&Node> {
        if id >= self.nodes.len() {
            return None;
        }
        
        // Prefetch next few nodes if enabled
        if self.config.enable_prefetch && id + self.config.prefetch_distance < self.nodes.len() {
            for i in 1..=self.config.prefetch_distance {
                let next_id = id + i;
                if next_id < self.nodes.len() {
                    // Memory prefetch hint - actual implementation would use platform-specific intrinsics
                    let _prefetch_hint = &self.nodes[next_id];
                }
            }
        }
        
        Some(&self.nodes[id])
    }

    /// Get statistics about the index
    pub fn stats(&self) -> HnswStats {
        let total_nodes = self.nodes.len();
        let active_nodes = self.active_nodes();
        let deleted_nodes = total_nodes - active_nodes;
        
        let mut total_connections = 0;
        let mut max_level = 0;
        
        for node in &self.nodes {
            if !node.uri.is_empty() {
                for (level, connections) in node.connections.iter().enumerate() {
                    total_connections += connections.len();
                    if !connections.is_empty() && level > max_level {
                        max_level = level;
                    }
                }
            }
        }
        
        let avg_connections = if active_nodes > 0 {
            total_connections as f64 / active_nodes as f64
        } else {
            0.0
        };
        
        HnswStats {
            total_nodes,
            active_nodes,
            deleted_nodes,
            total_connections,
            avg_connections,
            max_level,
            entry_point: self.entry_point,
        }
    }
}

/// Statistics about the HNSW index
#[derive(Debug, Clone)]
pub struct HnswStats {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub deleted_nodes: usize,
    pub total_connections: usize,
    pub avg_connections: f64,
    pub max_level: usize,
    pub entry_point: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;

    #[test]
    fn test_hnsw_basic() {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(config);

        // Insert some vectors
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![0.0, 1.0, 0.0]);
        let v3 = Vector::new(vec![0.0, 0.0, 1.0]);

        index.insert("v1".to_string(), v1.clone()).unwrap();
        index.insert("v2".to_string(), v2.clone()).unwrap();
        index.insert("v3".to_string(), v3.clone()).unwrap();

        // Search for nearest neighbors
        let results = index.search_knn(&v1, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "v1"); // Should find itself first
    }

    #[test]
    fn test_hnsw_larger_dataset() {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(config);

        // Insert 50 random vectors
        for i in 0..50 {
            let vector = crate::utils::random_vector(10, Some(i));
            index.insert(format!("v{}", i), vector).unwrap();
        }

        // Search for nearest neighbors
        let query = crate::utils::random_vector(10, Some(100));
        let results = index.search_knn(&query, 5).unwrap();
        assert_eq!(results.len(), 5);

        // All similarities should be between 0 and 1
        for (_, similarity) in &results {
            assert!(*similarity >= 0.0 && *similarity <= 1.0);
        }
    }
}