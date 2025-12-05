//! Beam search for DiskANN
//!
//! Implements the greedy best-first beam search algorithm used in DiskANN
//! for approximate nearest neighbor search on the Vamana graph.
//!
//! ## Algorithm
//! 1. Start from entry points
//! 2. Maintain a priority queue of top-L closest candidates
//! 3. Greedily expand the closest unvisited node
//! 4. Continue until no closer nodes are found
//!
//! ## References
//! - DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node
//!   (Jayaram Subramanya et al., NeurIPS 2019)

use crate::diskann::graph::VamanaGraph;
use crate::diskann::types::{DiskAnnError, DiskAnnResult, NodeId};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Search result containing neighbors and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Found neighbors with their distances
    pub neighbors: Vec<(NodeId, f32)>,
    /// Search statistics
    pub stats: SearchStats,
}

impl SearchResult {
    pub fn new(neighbors: Vec<(NodeId, f32)>, stats: SearchStats) -> Self {
        Self { neighbors, stats }
    }

    /// Get top-k results
    pub fn top_k(&self, k: usize) -> Vec<(NodeId, f32)> {
        self.neighbors.iter().take(k).copied().collect()
    }
}

/// Search statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchStats {
    /// Number of distance comparisons
    pub num_comparisons: usize,
    /// Number of graph hops
    pub num_hops: usize,
    /// Number of nodes visited
    pub num_visited: usize,
    /// Search beam width used
    pub beam_width: usize,
    /// Whether search converged
    pub converged: bool,
}

/// Candidate node in priority queue (min-heap by distance)
#[derive(Debug, Clone, Copy)]
struct Candidate {
    node_id: NodeId,
    distance: f32,
}

impl Candidate {
    fn new(node_id: NodeId, distance: f32) -> Self {
        Self { node_id, distance }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node_id == other.node_id
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
        // Reverse ordering for min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

/// Beam search implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamSearch {
    /// Beam width (L parameter)
    beam_width: usize,
    /// Maximum number of hops
    max_hops: Option<usize>,
}

impl BeamSearch {
    /// Create a new beam search with given beam width
    pub fn new(beam_width: usize) -> Self {
        Self {
            beam_width,
            max_hops: None,
        }
    }

    /// Set maximum number of hops
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = Some(max_hops);
        self
    }

    /// Get beam width
    pub fn beam_width(&self) -> usize {
        self.beam_width
    }

    /// Search for k nearest neighbors starting from entry points
    ///
    /// # Arguments
    /// * `graph` - Vamana graph to search
    /// * `query_distance_fn` - Function to compute distance from query to node
    /// * `k` - Number of neighbors to return
    pub fn search<F>(
        &self,
        graph: &VamanaGraph,
        query_distance_fn: &F,
        k: usize,
    ) -> DiskAnnResult<SearchResult>
    where
        F: Fn(NodeId) -> f32,
    {
        let entry_points = graph.entry_points();
        if entry_points.is_empty() {
            return Err(DiskAnnError::GraphError {
                message: "No entry points in graph".to_string(),
            });
        }

        // Initialize search from entry points
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut stats = SearchStats {
            beam_width: self.beam_width,
            ..Default::default()
        };

        // Add entry points to candidates
        for &entry_id in entry_points {
            let distance = query_distance_fn(entry_id);
            stats.num_comparisons += 1;
            candidates.push(Candidate::new(entry_id, distance));
            visited.insert(entry_id);
        }

        let mut best_candidates = Vec::new();

        // Greedy beam search
        loop {
            if stats.num_hops >= self.max_hops.unwrap_or(usize::MAX) {
                break;
            }

            // Get next closest unvisited candidate
            let current = match self.pop_next_candidate(&mut candidates, &visited) {
                Some(c) => c,
                None => {
                    stats.converged = true;
                    break;
                }
            };

            stats.num_hops += 1;

            // Mark as visited
            visited.insert(current.node_id);
            stats.num_visited += 1;

            // Add to best candidates
            best_candidates.push((current.node_id, current.distance));
            best_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            if best_candidates.len() > self.beam_width {
                best_candidates.truncate(self.beam_width);
            }

            // Explore neighbors
            if let Some(neighbors) = graph.get_neighbors(current.node_id) {
                for &neighbor_id in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }

                    let distance = query_distance_fn(neighbor_id);
                    stats.num_comparisons += 1;

                    // Add to candidates if within beam or better than worst in beam
                    if candidates.len() < self.beam_width
                        || distance < self.get_worst_distance(&candidates)
                    {
                        candidates.push(Candidate::new(neighbor_id, distance));
                        visited.insert(neighbor_id);

                        // Prune candidates to beam width
                        self.prune_candidates(&mut candidates);
                    }
                }
            }

            // Early termination: if current is worse than k-th best, and we have enough candidates
            if best_candidates.len() >= k {
                let kth_best = best_candidates
                    .get(k - 1)
                    .map(|(_, d)| *d)
                    .unwrap_or(f32::MAX);
                if current.distance > kth_best && candidates.is_empty() {
                    stats.converged = true;
                    break;
                }
            }
        }

        // Return top-k results
        best_candidates.truncate(k);

        Ok(SearchResult::new(best_candidates, stats))
    }

    /// Search from specific starting nodes (useful for incremental search)
    pub fn search_from<F>(
        &self,
        graph: &VamanaGraph,
        start_nodes: &[NodeId],
        query_distance_fn: &F,
        k: usize,
    ) -> DiskAnnResult<SearchResult>
    where
        F: Fn(NodeId) -> f32,
    {
        if start_nodes.is_empty() {
            return Err(DiskAnnError::GraphError {
                message: "No starting nodes provided".to_string(),
            });
        }

        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut stats = SearchStats {
            beam_width: self.beam_width,
            ..Default::default()
        };

        // Initialize from starting nodes
        for &node_id in start_nodes {
            let distance = query_distance_fn(node_id);
            stats.num_comparisons += 1;
            candidates.push(Candidate::new(node_id, distance));
            visited.insert(node_id);
        }

        self.continue_search(graph, candidates, visited, query_distance_fn, k, stats)
    }

    /// Continue search from current state (internal helper)
    fn continue_search<F>(
        &self,
        graph: &VamanaGraph,
        mut candidates: BinaryHeap<Candidate>,
        mut visited: HashSet<NodeId>,
        query_distance_fn: &F,
        k: usize,
        mut stats: SearchStats,
    ) -> DiskAnnResult<SearchResult>
    where
        F: Fn(NodeId) -> f32,
    {
        let mut best_candidates = Vec::new();

        loop {
            if stats.num_hops >= self.max_hops.unwrap_or(usize::MAX) {
                break;
            }

            let current = match self.pop_next_candidate(&mut candidates, &visited) {
                Some(c) => c,
                None => {
                    stats.converged = true;
                    break;
                }
            };

            stats.num_hops += 1;
            visited.insert(current.node_id);
            stats.num_visited += 1;

            best_candidates.push((current.node_id, current.distance));
            best_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            if best_candidates.len() > self.beam_width {
                best_candidates.truncate(self.beam_width);
            }

            if let Some(neighbors) = graph.get_neighbors(current.node_id) {
                for &neighbor_id in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }

                    let distance = query_distance_fn(neighbor_id);
                    stats.num_comparisons += 1;

                    if candidates.len() < self.beam_width
                        || distance < self.get_worst_distance(&candidates)
                    {
                        candidates.push(Candidate::new(neighbor_id, distance));
                        visited.insert(neighbor_id);
                        self.prune_candidates(&mut candidates);
                    }
                }
            }

            if best_candidates.len() >= k {
                let kth_best = best_candidates
                    .get(k - 1)
                    .map(|(_, d)| *d)
                    .unwrap_or(f32::MAX);
                if current.distance > kth_best && candidates.is_empty() {
                    stats.converged = true;
                    break;
                }
            }
        }

        best_candidates.truncate(k);
        Ok(SearchResult::new(best_candidates, stats))
    }

    /// Pop next candidate that hasn't been fully explored
    fn pop_next_candidate(
        &self,
        candidates: &mut BinaryHeap<Candidate>,
        _visited: &HashSet<NodeId>,
    ) -> Option<Candidate> {
        // Simply pop from the priority queue
        // (visited set tracks nodes that have been expanded)
        candidates.pop()
    }

    /// Get worst distance in candidates heap
    fn get_worst_distance(&self, candidates: &BinaryHeap<Candidate>) -> f32 {
        candidates
            .iter()
            .map(|c| c.distance)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(f32::MAX)
    }

    /// Prune candidates to beam width (keep top-L by distance)
    fn prune_candidates(&self, candidates: &mut BinaryHeap<Candidate>) {
        if candidates.len() <= self.beam_width {
            return;
        }

        // Convert to vec, sort, keep top-L, rebuild heap
        let mut vec: Vec<_> = candidates.drain().collect();
        vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        vec.truncate(self.beam_width);

        *candidates = vec.into_iter().collect();
    }
}

impl Default for BeamSearch {
    fn default() -> Self {
        Self::new(75)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::config::PruningStrategy;
    use crate::diskann::graph::VamanaGraph;

    fn build_test_graph() -> VamanaGraph {
        let mut graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.2);

        // Add nodes
        let n0 = graph.add_node("v0".to_string()).unwrap();
        let n1 = graph.add_node("v1".to_string()).unwrap();
        let n2 = graph.add_node("v2".to_string()).unwrap();
        let n3 = graph.add_node("v3".to_string()).unwrap();

        // Create connections: 0 -> 1 -> 2 -> 3
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n3).unwrap();
        graph.add_edge(n0, n2).unwrap(); // Shortcut

        graph
    }

    #[test]
    fn test_beam_search_basic() {
        let graph = build_test_graph();
        let beam_search = BeamSearch::new(10);

        // Distance function: distance to node 3
        let query_fn = |node_id: NodeId| (3 - node_id as i32).abs() as f32;

        let result = beam_search.search(&graph, &query_fn, 2).unwrap();

        assert!(!result.neighbors.is_empty());
        assert_eq!(result.neighbors[0].0, 3); // Closest should be node 3
        assert!(result.stats.num_comparisons > 0);
        assert!(result.stats.num_hops > 0);
    }

    #[test]
    fn test_search_with_max_hops() {
        let graph = build_test_graph();
        let beam_search = BeamSearch::new(10).with_max_hops(1);

        let query_fn = |node_id: NodeId| (3 - node_id as i32).abs() as f32;
        let result = beam_search.search(&graph, &query_fn, 2).unwrap();

        assert_eq!(result.stats.num_hops, 1);
    }

    #[test]
    fn test_search_from_specific_nodes() {
        let graph = build_test_graph();
        let beam_search = BeamSearch::new(10);

        let query_fn = |node_id: NodeId| (3 - node_id as i32).abs() as f32;
        let result = beam_search.search_from(&graph, &[2], &query_fn, 2).unwrap();

        assert!(!result.neighbors.is_empty());
        // Should find node 3 quickly since we start from node 2
        assert!(result.neighbors.iter().any(|(id, _)| *id == 3));
    }

    #[test]
    fn test_top_k_results() {
        let graph = build_test_graph();
        let beam_search = BeamSearch::new(10);

        let query_fn = |node_id: NodeId| node_id as f32;
        let result = beam_search.search(&graph, &query_fn, 4).unwrap();

        let top2 = result.top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 0); // Closest
    }

    #[test]
    fn test_candidate_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(Candidate::new(0, 3.0));
        heap.push(Candidate::new(1, 1.0));
        heap.push(Candidate::new(2, 2.0));

        // Min-heap: should pop in ascending order of distance
        assert_eq!(heap.pop().unwrap().node_id, 1); // distance 1.0
        assert_eq!(heap.pop().unwrap().node_id, 2); // distance 2.0
        assert_eq!(heap.pop().unwrap().node_id, 0); // distance 3.0
    }

    #[test]
    fn test_empty_graph_error() {
        let graph = VamanaGraph::new(3, PruningStrategy::Alpha, 1.2);
        let beam_search = BeamSearch::new(10);

        let query_fn = |_: NodeId| 1.0;
        let result = beam_search.search(&graph, &query_fn, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_search_stats() {
        let graph = build_test_graph();
        let beam_search = BeamSearch::new(10);

        let query_fn = |node_id: NodeId| node_id as f32;
        let result = beam_search.search(&graph, &query_fn, 2).unwrap();

        let stats = &result.stats;
        assert_eq!(stats.beam_width, 10);
        assert!(stats.num_comparisons > 0);
        assert!(stats.num_hops > 0);
        assert!(stats.num_visited > 0);
    }

    #[test]
    fn test_beam_width_constraint() {
        let graph = build_test_graph();
        let beam_search = BeamSearch::new(2); // Small beam

        let query_fn = |node_id: NodeId| node_id as f32;
        let result = beam_search.search(&graph, &query_fn, 3).unwrap();

        // Should still work with small beam, just fewer candidates explored
        assert!(!result.neighbors.is_empty());
        assert!(result.stats.num_visited <= 10); // Limited by beam width
    }
}
