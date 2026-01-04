//! Vamana graph for DiskANN
//!
//! Implements the Vamana graph structure used in Microsoft's DiskANN.
//! Vamana is a navigable small-world graph optimized for approximate nearest
//! neighbor search on disk-based datasets.
//!
//! ## Algorithm
//! - Each node has at most R neighbors (max_degree)
//! - Neighbors are selected using robust pruning to ensure good coverage
//! - Graph supports incremental updates and entry point management
//!
//! ## References
//! - DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node
//!   (Jayaram Subramanya et al., NeurIPS 2019)

use crate::diskann::config::PruningStrategy;
use crate::diskann::types::{DiskAnnError, DiskAnnResult, NodeId, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// A node in the Vamana graph
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
pub struct VamanaNode {
    /// Node ID (internal)
    pub id: NodeId,
    /// Vector ID (external)
    pub vector_id: VectorId,
    /// List of neighbor node IDs
    pub neighbors: Vec<NodeId>,
    /// Maximum degree for this node
    pub max_degree: usize,
}

impl VamanaNode {
    /// Create a new Vamana node
    pub fn new(id: NodeId, vector_id: VectorId, max_degree: usize) -> Self {
        Self {
            id,
            vector_id,
            neighbors: Vec::with_capacity(max_degree),
            max_degree,
        }
    }

    /// Add a neighbor if not already present
    pub fn add_neighbor(&mut self, neighbor_id: NodeId) -> bool {
        if !self.neighbors.contains(&neighbor_id) && self.neighbors.len() < self.max_degree {
            self.neighbors.push(neighbor_id);
            true
        } else {
            false
        }
    }

    /// Remove a neighbor
    pub fn remove_neighbor(&mut self, neighbor_id: NodeId) -> bool {
        if let Some(pos) = self.neighbors.iter().position(|&id| id == neighbor_id) {
            self.neighbors.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if neighbor limit is reached
    pub fn is_full(&self) -> bool {
        self.neighbors.len() >= self.max_degree
    }

    /// Get number of neighbors
    pub fn degree(&self) -> usize {
        self.neighbors.len()
    }

    /// Replace neighbors with pruned set
    pub fn set_neighbors(&mut self, neighbors: Vec<NodeId>) {
        self.neighbors = neighbors;
        if self.neighbors.len() > self.max_degree {
            self.neighbors.truncate(self.max_degree);
        }
    }
}

/// Vamana graph structure
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
pub struct VamanaGraph {
    /// Graph nodes indexed by NodeId
    nodes: HashMap<NodeId, VamanaNode>,
    /// Mapping from VectorId to NodeId
    vector_to_node: HashMap<VectorId, NodeId>,
    /// Entry points for search (medoids)
    entry_points: Vec<NodeId>,
    /// Maximum degree for nodes
    max_degree: usize,
    /// Pruning strategy
    pruning_strategy: PruningStrategy,
    /// Alpha parameter for pruning
    alpha: f32,
    /// Next available node ID
    next_node_id: NodeId,
}

impl VamanaGraph {
    /// Create a new empty Vamana graph
    pub fn new(max_degree: usize, pruning_strategy: PruningStrategy, alpha: f32) -> Self {
        Self {
            nodes: HashMap::new(),
            vector_to_node: HashMap::new(),
            entry_points: Vec::new(),
            max_degree,
            pruning_strategy,
            alpha,
            next_node_id: 0,
        }
    }

    /// Get number of nodes in graph
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get maximum degree
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Get entry points
    pub fn entry_points(&self) -> &[NodeId] {
        &self.entry_points
    }

    /// Set entry points
    pub fn set_entry_points(&mut self, entry_points: Vec<NodeId>) {
        self.entry_points = entry_points;
    }

    /// Add entry point
    pub fn add_entry_point(&mut self, node_id: NodeId) -> DiskAnnResult<()> {
        if !self.nodes.contains_key(&node_id) {
            return Err(DiskAnnError::GraphError {
                message: format!("Node {} does not exist", node_id),
            });
        }
        if !self.entry_points.contains(&node_id) {
            self.entry_points.push(node_id);
        }
        Ok(())
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: NodeId) -> Option<&VamanaNode> {
        self.nodes.get(&node_id)
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut VamanaNode> {
        self.nodes.get_mut(&node_id)
    }

    /// Get node ID by vector ID
    pub fn get_node_id(&self, vector_id: &VectorId) -> Option<NodeId> {
        self.vector_to_node.get(vector_id).copied()
    }

    /// Add a new node to the graph
    pub fn add_node(&mut self, vector_id: VectorId) -> DiskAnnResult<NodeId> {
        if self.vector_to_node.contains_key(&vector_id) {
            return Err(DiskAnnError::GraphError {
                message: format!("Vector {} already exists", vector_id),
            });
        }

        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let node = VamanaNode::new(node_id, vector_id.clone(), self.max_degree);
        self.nodes.insert(node_id, node);
        self.vector_to_node.insert(vector_id, node_id);

        // If this is the first node, make it an entry point
        if self.entry_points.is_empty() {
            self.entry_points.push(node_id);
        }

        Ok(node_id)
    }

    /// Remove a node from the graph
    pub fn remove_node(&mut self, node_id: NodeId) -> DiskAnnResult<()> {
        let node = self
            .nodes
            .remove(&node_id)
            .ok_or_else(|| DiskAnnError::GraphError {
                message: format!("Node {} does not exist", node_id),
            })?;

        self.vector_to_node.remove(&node.vector_id);

        // Remove from entry points
        self.entry_points.retain(|&id| id != node_id);

        // Remove all edges pointing to this node
        for other_node in self.nodes.values_mut() {
            other_node.remove_neighbor(node_id);
        }

        Ok(())
    }

    /// Add a directed edge from source to target
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> DiskAnnResult<bool> {
        if source == target {
            return Ok(false); // Self-loops not allowed
        }

        // Check target exists first
        if !self.nodes.contains_key(&target) {
            return Err(DiskAnnError::GraphError {
                message: format!("Target node {} does not exist", target),
            });
        }

        // Then get mutable reference to source
        let source_node = self
            .get_node_mut(source)
            .ok_or_else(|| DiskAnnError::GraphError {
                message: format!("Source node {} does not exist", source),
            })?;

        Ok(source_node.add_neighbor(target))
    }

    /// Remove a directed edge from source to target
    pub fn remove_edge(&mut self, source: NodeId, target: NodeId) -> DiskAnnResult<bool> {
        let source_node = self
            .get_node_mut(source)
            .ok_or_else(|| DiskAnnError::GraphError {
                message: format!("Source node {} does not exist", source),
            })?;

        Ok(source_node.remove_neighbor(target))
    }

    /// Prune neighbors of a node using the configured strategy
    ///
    /// # Arguments
    /// * `node_id` - Node whose neighbors to prune
    /// * `candidates` - Candidate neighbors with their distances
    /// * `distance_fn` - Function to compute distance between node IDs
    pub fn prune_neighbors<F>(
        &mut self,
        node_id: NodeId,
        candidates: &[(NodeId, f32)],
        distance_fn: &F,
    ) -> DiskAnnResult<()>
    where
        F: Fn(NodeId, NodeId) -> f32,
    {
        if candidates.is_empty() {
            return Ok(());
        }

        let pruned = match self.pruning_strategy {
            PruningStrategy::Alpha => {
                self.alpha_prune(node_id, candidates, self.max_degree, self.alpha)
            }
            PruningStrategy::Robust => self.robust_prune(
                node_id,
                candidates,
                distance_fn,
                self.max_degree,
                self.alpha,
            ),
            PruningStrategy::Hybrid => {
                // Use robust pruning for first half, alpha for second half
                let mid = self.max_degree / 2;
                let mut robust =
                    self.robust_prune(node_id, candidates, distance_fn, mid, self.alpha);

                // Get remaining candidates
                let robust_set: HashSet<_> = robust.iter().copied().collect();
                let remaining: Vec<_> = candidates
                    .iter()
                    .filter(|(id, _)| !robust_set.contains(id))
                    .copied()
                    .collect();

                let mut alpha =
                    self.alpha_prune(node_id, &remaining, self.max_degree - mid, self.alpha);
                robust.append(&mut alpha);
                robust
            }
        };

        // Update node's neighbors
        if let Some(node) = self.get_node_mut(node_id) {
            node.set_neighbors(pruned);
        }

        Ok(())
    }

    /// Alpha pruning: select R closest neighbors within alpha * distance to closest
    fn alpha_prune(
        &self,
        _node_id: NodeId,
        candidates: &[(NodeId, f32)],
        max_neighbors: usize,
        alpha: f32,
    ) -> Vec<NodeId> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let threshold = sorted[0].1 * alpha;
        sorted
            .into_iter()
            .filter(|(_, dist)| *dist <= threshold)
            .take(max_neighbors)
            .map(|(id, _)| id)
            .collect()
    }

    /// Robust pruning: diversified neighbor selection for better graph connectivity
    fn robust_prune<F>(
        &self,
        node_id: NodeId,
        candidates: &[(NodeId, f32)],
        distance_fn: &F,
        max_neighbors: usize,
        alpha: f32,
    ) -> Vec<NodeId>
    where
        F: Fn(NodeId, NodeId) -> f32,
    {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected = Vec::new();
        let mut selected_set = HashSet::new();

        for (candidate_id, candidate_dist) in &sorted {
            if selected.len() >= max_neighbors {
                break;
            }

            if *candidate_id == node_id || selected_set.contains(candidate_id) {
                continue;
            }

            // Check if candidate is closer to query than to any selected neighbor
            let mut should_add = true;
            for &selected_id in &selected {
                let inter_distance = distance_fn(*candidate_id, selected_id);
                if inter_distance < alpha * candidate_dist {
                    should_add = false;
                    break;
                }
            }

            if should_add {
                selected.push(*candidate_id);
                selected_set.insert(*candidate_id);
            }
        }

        // If we don't have enough neighbors, add closest remaining ones
        if selected.len() < max_neighbors {
            for (candidate_id, _) in &sorted {
                if selected.len() >= max_neighbors {
                    break;
                }
                if *candidate_id != node_id && !selected_set.contains(candidate_id) {
                    selected.push(*candidate_id);
                    selected_set.insert(*candidate_id);
                }
            }
        }

        selected
    }

    /// Get all neighbors of a node
    pub fn get_neighbors(&self, node_id: NodeId) -> Option<&[NodeId]> {
        self.nodes
            .get(&node_id)
            .map(|node| node.neighbors.as_slice())
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let total_nodes = self.nodes.len();
        let total_edges: usize = self.nodes.values().map(|n| n.degree()).sum();
        let avg_degree = if total_nodes > 0 {
            total_edges as f64 / total_nodes as f64
        } else {
            0.0
        };

        let max_degree_actual = self.nodes.values().map(|n| n.degree()).max().unwrap_or(0);
        let min_degree_actual = self.nodes.values().map(|n| n.degree()).min().unwrap_or(0);

        GraphStats {
            num_nodes: total_nodes,
            num_edges: total_edges,
            avg_degree,
            max_degree_configured: self.max_degree,
            max_degree_actual,
            min_degree_actual,
            num_entry_points: self.entry_points.len(),
        }
    }

    /// Validate graph integrity
    pub fn validate(&self) -> DiskAnnResult<()> {
        // Check all edges point to existing nodes
        for (node_id, node) in &self.nodes {
            for &neighbor_id in &node.neighbors {
                if !self.nodes.contains_key(&neighbor_id) {
                    return Err(DiskAnnError::GraphError {
                        message: format!(
                            "Node {} has edge to non-existent node {}",
                            node_id, neighbor_id
                        ),
                    });
                }
            }

            // Check degree constraint
            if node.neighbors.len() > node.max_degree {
                return Err(DiskAnnError::GraphError {
                    message: format!(
                        "Node {} has {} neighbors, exceeding max degree {}",
                        node_id,
                        node.neighbors.len(),
                        node.max_degree
                    ),
                });
            }

            // Check for self-loops
            if node.neighbors.contains(node_id) {
                return Err(DiskAnnError::GraphError {
                    message: format!("Node {} has self-loop", node_id),
                });
            }

            // Check for duplicates
            let mut seen = HashSet::new();
            for &neighbor_id in &node.neighbors {
                if !seen.insert(neighbor_id) {
                    return Err(DiskAnnError::GraphError {
                        message: format!("Node {} has duplicate neighbor {}", node_id, neighbor_id),
                    });
                }
            }
        }

        // Check entry points exist
        for &entry_id in &self.entry_points {
            if !self.nodes.contains_key(&entry_id) {
                return Err(DiskAnnError::GraphError {
                    message: format!("Entry point {} does not exist", entry_id),
                });
            }
        }

        Ok(())
    }
}

impl Default for VamanaGraph {
    fn default() -> Self {
        Self::new(64, PruningStrategy::Robust, 1.2)
    }
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f64,
    pub max_degree_configured: usize,
    pub max_degree_actual: usize,
    pub min_degree_actual: usize,
    pub num_entry_points: usize,
}

/// Thread-safe wrapper for VamanaGraph
#[derive(Debug, Clone)]
pub struct VamanaGraphHandle {
    graph: Arc<RwLock<VamanaGraph>>,
}

impl VamanaGraphHandle {
    pub fn new(graph: VamanaGraph) -> Self {
        Self {
            graph: Arc::new(RwLock::new(graph)),
        }
    }

    pub fn read<F, R>(&self, f: F) -> DiskAnnResult<R>
    where
        F: FnOnce(&VamanaGraph) -> R,
    {
        let graph = self
            .graph
            .read()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;
        Ok(f(&graph))
    }

    pub fn write<F, R>(&self, f: F) -> DiskAnnResult<R>
    where
        F: FnOnce(&mut VamanaGraph) -> R,
    {
        let mut graph = self
            .graph
            .write()
            .map_err(|_| DiskAnnError::ConcurrentModification)?;
        Ok(f(&mut graph))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vamana_node() {
        let mut node = VamanaNode::new(0, "vec0".to_string(), 3);
        assert_eq!(node.id, 0);
        assert_eq!(node.degree(), 0);
        assert!(!node.is_full());

        assert!(node.add_neighbor(1));
        assert!(node.add_neighbor(2));
        assert!(node.add_neighbor(3));
        assert_eq!(node.degree(), 3);
        assert!(node.is_full());

        assert!(!node.add_neighbor(4)); // Full
        assert!(!node.add_neighbor(1)); // Duplicate

        assert!(node.remove_neighbor(2));
        assert_eq!(node.degree(), 2);
        assert!(!node.remove_neighbor(2)); // Not present
    }

    #[test]
    fn test_vamana_graph_basic() {
        let mut graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.2);
        assert_eq!(graph.num_nodes(), 0);

        let node0 = graph.add_node("vec0".to_string()).unwrap();
        let node1 = graph.add_node("vec1".to_string()).unwrap();
        assert_eq!(graph.num_nodes(), 2);

        assert!(graph.add_edge(node0, node1).unwrap());
        assert!(!graph.add_edge(node0, node0).unwrap()); // Self-loop

        let neighbors = graph.get_neighbors(node0).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], node1);
    }

    #[test]
    fn test_alpha_pruning() {
        let graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.5);

        let candidates = vec![(1, 1.0), (2, 1.2), (3, 1.4), (4, 2.0), (5, 3.0)];

        let pruned = graph.alpha_prune(0, &candidates, 3, 1.5);
        assert!(pruned.len() <= 3);
        assert!(pruned.contains(&1)); // Closest
    }

    #[test]
    fn test_robust_pruning() {
        let graph = VamanaGraph::new(3, PruningStrategy::Robust, 1.2);

        let candidates = vec![(1, 1.0), (2, 1.5), (3, 2.0)];

        let distance_fn = |a: NodeId, b: NodeId| (a as i32 - b as i32).abs() as f32;
        let pruned = graph.robust_prune(0, &candidates, &distance_fn, 3, 1.2);

        assert!(pruned.len() <= 3);
        assert!(pruned.contains(&1));
    }

    #[test]
    fn test_entry_points() {
        let mut graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.2);
        let _node0 = graph.add_node("vec0".to_string()).unwrap();
        let node1 = graph.add_node("vec1".to_string()).unwrap();

        assert_eq!(graph.entry_points().len(), 1); // First node is entry point

        graph.add_entry_point(node1).unwrap();
        assert_eq!(graph.entry_points().len(), 2);
    }

    #[test]
    fn test_graph_validation() {
        let mut graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.2);
        let node0 = graph.add_node("vec0".to_string()).unwrap();
        let node1 = graph.add_node("vec1".to_string()).unwrap();

        graph.add_edge(node0, node1).unwrap();
        assert!(graph.validate().is_ok());

        // Remove node1 but leave edge - should fail validation
        graph.nodes.remove(&node1);
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_graph_stats() {
        let mut graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.2);
        let node0 = graph.add_node("vec0".to_string()).unwrap();
        let node1 = graph.add_node("vec1".to_string()).unwrap();
        let node2 = graph.add_node("vec2".to_string()).unwrap();

        graph.add_edge(node0, node1).unwrap();
        graph.add_edge(node0, node2).unwrap();
        graph.add_edge(node1, node2).unwrap();

        let stats = graph.stats();
        assert_eq!(stats.num_nodes, 3);
        assert_eq!(stats.num_edges, 3);
        assert!(stats.avg_degree > 0.0);
    }

    #[test]
    fn test_remove_node() {
        let mut graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.2);
        let node0 = graph.add_node("vec0".to_string()).unwrap();
        let node1 = graph.add_node("vec1".to_string()).unwrap();

        graph.add_edge(node0, node1).unwrap();
        assert_eq!(graph.num_nodes(), 2);

        graph.remove_node(node1).unwrap();
        assert_eq!(graph.num_nodes(), 1);
        assert!(graph.get_neighbors(node0).unwrap().is_empty());
    }

    #[test]
    fn test_thread_safe_handle() {
        let graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.2);
        let handle = VamanaGraphHandle::new(graph);

        let node_id = handle
            .write(|g| g.add_node("vec0".to_string()))
            .unwrap()
            .unwrap();
        let count = handle.read(|g| g.num_nodes()).unwrap();

        assert_eq!(count, 1);
        assert_eq!(node_id, 0);
    }
}
