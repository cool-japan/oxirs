//! Basic types for HNSW implementation

use crate::Vector;
use std::cmp::Ordering;
use std::collections::HashSet;

/// A candidate for nearest neighbor search
#[derive(Debug, Clone)]
pub struct Candidate {
    pub distance: f32,
    pub id: usize,
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
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl Candidate {
    /// Create a new candidate
    pub fn new(id: usize, distance: f32) -> Self {
        Self { id, distance }
    }
}

/// Node in the HNSW graph with cache-friendly layout
#[derive(Debug, Clone)]
pub struct Node {
    /// Vector data (stored first for cache efficiency)
    pub vector: Vector,
    /// URI identifier
    pub uri: String,
    /// Connections for each layer (layer -> set of connected node IDs)
    pub connections: Vec<HashSet<usize>>,
    /// Cache-friendly vector data for SIMD operations
    pub vector_data_f32: Vec<f32>,
    /// Node access frequency (for cache optimization)
    pub access_count: u64,
}

impl Node {
    pub fn new(uri: String, vector: Vector, max_level: usize) -> Self {
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
    pub fn record_access(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Get the maximum level for this node
    pub fn level(&self) -> usize {
        self.connections.len().saturating_sub(1)
    }

    /// Get connections at a specific level
    pub fn get_connections(&self, level: usize) -> Option<&HashSet<usize>> {
        self.connections.get(level)
    }

    /// Get mutable connections at a specific level
    pub fn get_connections_mut(&mut self, level: usize) -> Option<&mut HashSet<usize>> {
        self.connections.get_mut(level)
    }

    /// Add a connection at a specific level
    pub fn add_connection(&mut self, level: usize, node_id: usize) {
        if let Some(connections) = self.connections.get_mut(level) {
            connections.insert(node_id);
        }
    }

    /// Remove a connection at a specific level
    pub fn remove_connection(&mut self, level: usize, node_id: usize) {
        if let Some(connections) = self.connections.get_mut(level) {
            connections.remove(&node_id);
        }
    }
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
