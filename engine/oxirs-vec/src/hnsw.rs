//! Custom HNSW (Hierarchical Navigable Small World) implementation
//! 
//! This module provides a pure Rust implementation of the HNSW algorithm
//! for approximate nearest neighbor search.

use crate::{Vector, VectorIndex, similarity::SimilarityMetric};
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

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

/// Node in the HNSW graph
#[derive(Debug, Clone)]
struct Node {
    /// Vector data
    vector: Vector,
    /// URI identifier
    uri: String,
    /// Connections for each layer (layer -> set of connected node IDs)
    connections: Vec<HashSet<usize>>,
}

impl Node {
    fn new(uri: String, vector: Vector, max_level: usize) -> Self {
        Self {
            vector,
            uri,
            connections: vec![HashSet::new(); max_level + 1],
        }
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

        // Initialize with entry points
        for ep in entry_points {
            if ep < self.nodes.len() {
                let distance = 1.0 - self.similarity(query, &self.nodes[ep].vector);
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
                        let distance = 1.0 - self.similarity(query, &self.nodes[neighbor_id].vector);
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

        let entry_point = self.entry_point.unwrap();
        let mut current_closest = vec![entry_point];

        // Search from top layer down to 1
        let max_level = self.nodes[entry_point].connections.len().saturating_sub(1);
        for layer in (1..=max_level).rev() {
            current_closest = self.search_layer(query, current_closest, 1, layer)
                .into_iter().map(|c| c.id).collect();
        }

        // Search layer 0 with ef
        let candidates = self.search_layer(query, current_closest, self.config.ef.max(k), 0);

        // Convert to results and return top k
        let mut results: Vec<(String, f32)> = candidates
            .into_iter()
            .map(|c| {
                let similarity = self.similarity(query, &self.nodes[c.id].vector);
                (self.nodes[c.id].uri.clone(), similarity)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let all_results = self.search_knn(query, self.nodes.len())?;
        Ok(all_results.into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect())
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