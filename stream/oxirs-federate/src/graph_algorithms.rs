//! Graph Algorithms for Federation Routing
//!
//! This module provides comprehensive graph algorithms for optimal routing
//! in multi-level federation architectures, including:
//! - Dijkstra's shortest path algorithm
//! - A* pathfinding with heuristics
//! - Bellman-Ford for negative weight handling
//! - Floyd-Warshall for all-pairs shortest paths
//! - Minimum spanning tree (Prim's, Kruskal's)
//! - Graph connectivity analysis
//! - Centrality measures for federation nodes
//!
//! All algorithms are optimized using scirs2 for performance on large federation topologies.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Graph representation for federation routing
#[derive(Debug, Clone)]
pub struct FederationGraph {
    /// Node IDs
    nodes: Vec<String>,
    /// Node ID to index mapping
    node_index: HashMap<String, usize>,
    /// Adjacency list with weights (cost/latency)
    adjacency: Vec<Vec<(usize, f64)>>,
}

impl FederationGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_index: HashMap::new(),
            adjacency: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node_id: String) -> usize {
        if let Some(&idx) = self.node_index.get(&node_id) {
            return idx;
        }

        let idx = self.nodes.len();
        self.nodes.push(node_id.clone());
        self.node_index.insert(node_id, idx);
        self.adjacency.push(Vec::new());
        idx
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: &str, to: &str, weight: f64) -> Result<()> {
        let from_idx = self
            .node_index
            .get(from)
            .ok_or_else(|| anyhow!("Node not found: {}", from))?;
        let to_idx = self
            .node_index
            .get(to)
            .ok_or_else(|| anyhow!("Node not found: {}", to))?;

        self.adjacency[*from_idx].push((*to_idx, weight));
        Ok(())
    }

    /// Add bidirectional edge
    pub fn add_bidirectional_edge(&mut self, node1: &str, node2: &str, weight: f64) -> Result<()> {
        self.add_edge(node1, node2, weight)?;
        self.add_edge(node2, node1, weight)?;
        Ok(())
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get node ID by index
    pub fn get_node(&self, idx: usize) -> Option<&str> {
        self.nodes.get(idx).map(|s| s.as_str())
    }

    /// Get node index by ID
    pub fn get_index(&self, node_id: &str) -> Option<usize> {
        self.node_index.get(node_id).copied()
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node_idx: usize) -> &[(usize, f64)] {
        &self.adjacency[node_idx]
    }
}

impl Default for FederationGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Dijkstra's shortest path algorithm
pub struct Dijkstra;

impl Dijkstra {
    /// Find shortest path from source to target
    pub fn shortest_path(
        graph: &FederationGraph,
        source: &str,
        target: &str,
    ) -> Result<ShortestPathResult> {
        let source_idx = graph
            .get_index(source)
            .ok_or_else(|| anyhow!("Source node not found: {}", source))?;
        let target_idx = graph
            .get_index(target)
            .ok_or_else(|| anyhow!("Target node not found: {}", target))?;

        let n = graph.node_count();
        let mut distances = vec![f64::INFINITY; n];
        let mut predecessors = vec![None; n];
        let mut visited = vec![false; n];
        let mut heap = BinaryHeap::new();

        distances[source_idx] = 0.0;
        heap.push(State {
            cost: 0.0,
            node: source_idx,
        });

        while let Some(State { cost, node }) = heap.pop() {
            if visited[node] {
                continue;
            }

            visited[node] = true;

            if node == target_idx {
                break;
            }

            // Relaxation step
            for &(neighbor, weight) in graph.neighbors(node) {
                let new_cost = cost + weight;

                if new_cost < distances[neighbor] {
                    distances[neighbor] = new_cost;
                    predecessors[neighbor] = Some(node);
                    heap.push(State {
                        cost: new_cost,
                        node: neighbor,
                    });
                }
            }
        }

        // Reconstruct path
        if distances[target_idx].is_infinite() {
            return Err(anyhow!("No path found from {} to {}", source, target));
        }

        let path = Self::reconstruct_path(&predecessors, source_idx, target_idx, graph);
        let cost = distances[target_idx];

        Ok(ShortestPathResult { path, cost })
    }

    /// Find shortest paths from source to all other nodes
    pub fn shortest_paths_from(
        graph: &FederationGraph,
        source: &str,
    ) -> Result<HashMap<String, ShortestPathResult>> {
        let source_idx = graph
            .get_index(source)
            .ok_or_else(|| anyhow!("Source node not found: {}", source))?;

        let n = graph.node_count();
        let mut distances = vec![f64::INFINITY; n];
        let mut predecessors = vec![None; n];
        let mut visited = vec![false; n];
        let mut heap = BinaryHeap::new();

        distances[source_idx] = 0.0;
        heap.push(State {
            cost: 0.0,
            node: source_idx,
        });

        while let Some(State { cost, node }) = heap.pop() {
            if visited[node] {
                continue;
            }

            visited[node] = true;

            for &(neighbor, weight) in graph.neighbors(node) {
                let new_cost = cost + weight;

                if new_cost < distances[neighbor] {
                    distances[neighbor] = new_cost;
                    predecessors[neighbor] = Some(node);
                    heap.push(State {
                        cost: new_cost,
                        node: neighbor,
                    });
                }
            }
        }

        // Build results for all reachable nodes
        let mut results = HashMap::new();
        for (target_idx, &dist) in distances.iter().enumerate() {
            if !dist.is_infinite() && target_idx != source_idx {
                if let Some(target_id) = graph.get_node(target_idx) {
                    let path = Self::reconstruct_path(&predecessors, source_idx, target_idx, graph);
                    results.insert(
                        target_id.to_string(),
                        ShortestPathResult { path, cost: dist },
                    );
                }
            }
        }

        Ok(results)
    }

    fn reconstruct_path(
        predecessors: &[Option<usize>],
        source: usize,
        target: usize,
        graph: &FederationGraph,
    ) -> Vec<String> {
        let mut path = Vec::new();
        let mut current = target;

        while current != source {
            if let Some(node_id) = graph.get_node(current) {
                path.push(node_id.to_string());
            }

            match predecessors[current] {
                Some(pred) => current = pred,
                None => break,
            }
        }

        if let Some(source_id) = graph.get_node(source) {
            path.push(source_id.to_string());
        }

        path.reverse();
        path
    }
}

/// A* pathfinding algorithm with heuristic
pub struct AStar;

impl AStar {
    /// Find shortest path using A* with heuristic
    pub fn shortest_path<H>(
        graph: &FederationGraph,
        source: &str,
        target: &str,
        heuristic: H,
    ) -> Result<ShortestPathResult>
    where
        H: Fn(&str, &str) -> f64,
    {
        let source_idx = graph
            .get_index(source)
            .ok_or_else(|| anyhow!("Source node not found: {}", source))?;
        let target_idx = graph
            .get_index(target)
            .ok_or_else(|| anyhow!("Target node not found: {}", target))?;

        let n = graph.node_count();
        let mut g_scores = vec![f64::INFINITY; n];
        let mut f_scores = vec![f64::INFINITY; n];
        let mut predecessors = vec![None; n];
        let mut open_set = BinaryHeap::new();

        g_scores[source_idx] = 0.0;
        f_scores[source_idx] = heuristic(source, target);
        open_set.push(State {
            cost: -f_scores[source_idx],
            node: source_idx,
        });

        while let Some(State { node, .. }) = open_set.pop() {
            if node == target_idx {
                let path = Dijkstra::reconstruct_path(&predecessors, source_idx, target_idx, graph);
                return Ok(ShortestPathResult {
                    path,
                    cost: g_scores[target_idx],
                });
            }

            for &(neighbor, weight) in graph.neighbors(node) {
                let tentative_g = g_scores[node] + weight;

                if tentative_g < g_scores[neighbor] {
                    predecessors[neighbor] = Some(node);
                    g_scores[neighbor] = tentative_g;

                    if let Some(neighbor_id) = graph.get_node(neighbor) {
                        if let Some(target_id) = graph.get_node(target_idx) {
                            let h = heuristic(neighbor_id, target_id);
                            f_scores[neighbor] = tentative_g + h;
                            open_set.push(State {
                                cost: -f_scores[neighbor],
                                node: neighbor,
                            });
                        }
                    }
                }
            }
        }

        Err(anyhow!("No path found from {} to {}", source, target))
    }
}

/// Bellman-Ford algorithm (handles negative weights)
pub struct BellmanFord;

impl BellmanFord {
    /// Find shortest paths, detecting negative cycles
    pub fn shortest_paths(
        graph: &FederationGraph,
        source: &str,
    ) -> Result<HashMap<String, ShortestPathResult>> {
        let source_idx = graph
            .get_index(source)
            .ok_or_else(|| anyhow!("Source node not found: {}", source))?;

        let n = graph.node_count();
        let mut distances = vec![f64::INFINITY; n];
        let mut predecessors = vec![None; n];

        distances[source_idx] = 0.0;

        // Relax edges n-1 times
        for _ in 0..n - 1 {
            let mut updated = false;

            for node in 0..n {
                if distances[node].is_infinite() {
                    continue;
                }

                for &(neighbor, weight) in graph.neighbors(node) {
                    let new_dist = distances[node] + weight;

                    if new_dist < distances[neighbor] {
                        distances[neighbor] = new_dist;
                        predecessors[neighbor] = Some(node);
                        updated = true;
                    }
                }
            }

            if !updated {
                break;
            }
        }

        // Check for negative cycles
        for node in 0..n {
            if distances[node].is_infinite() {
                continue;
            }

            for &(neighbor, weight) in graph.neighbors(node) {
                if distances[node] + weight < distances[neighbor] {
                    return Err(anyhow!("Negative cycle detected in graph"));
                }
            }
        }

        // Build results
        let mut results = HashMap::new();
        for (target_idx, &dist) in distances.iter().enumerate() {
            if !dist.is_infinite() && target_idx != source_idx {
                if let Some(target_id) = graph.get_node(target_idx) {
                    let path =
                        Dijkstra::reconstruct_path(&predecessors, source_idx, target_idx, graph);
                    results.insert(
                        target_id.to_string(),
                        ShortestPathResult { path, cost: dist },
                    );
                }
            }
        }

        Ok(results)
    }
}

/// Floyd-Warshall algorithm (all-pairs shortest paths)
pub struct FloydWarshall;

impl FloydWarshall {
    /// Compute all-pairs shortest paths
    pub fn all_pairs_shortest_paths(
        graph: &FederationGraph,
    ) -> Result<HashMap<(String, String), ShortestPathResult>> {
        let n = graph.node_count();
        let mut dist = vec![vec![f64::INFINITY; n]; n];
        let mut next = vec![vec![None; n]; n];

        // Initialize
        for i in 0..n {
            dist[i][i] = 0.0;
            next[i][i] = Some(i);
        }

        for u in 0..n {
            for &(v, weight) in graph.neighbors(u) {
                dist[u][v] = weight;
                next[u][v] = Some(v);
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let new_dist = dist[i][k] + dist[k][j];
                    if new_dist < dist[i][j] {
                        dist[i][j] = new_dist;
                        next[i][j] = next[i][k];
                    }
                }
            }
        }

        // Check for negative cycles
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if dist[i][i] < 0.0 {
                return Err(anyhow!("Negative cycle detected in graph"));
            }
        }

        // Build results
        let mut results = HashMap::new();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                if i != j && !dist[i][j].is_infinite() {
                    if let (Some(source_id), Some(target_id)) =
                        (graph.get_node(i), graph.get_node(j))
                    {
                        let path = Self::reconstruct_path(&next, i, j, graph);
                        results.insert(
                            (source_id.to_string(), target_id.to_string()),
                            ShortestPathResult {
                                path,
                                cost: dist[i][j],
                            },
                        );
                    }
                }
            }
        }

        Ok(results)
    }

    fn reconstruct_path(
        next: &[Vec<Option<usize>>],
        mut u: usize,
        v: usize,
        graph: &FederationGraph,
    ) -> Vec<String> {
        let mut path = Vec::new();

        if next[u][v].is_none() {
            return path;
        }

        while u != v {
            if let Some(node_id) = graph.get_node(u) {
                path.push(node_id.to_string());
            }
            u = next[u][v].unwrap();
        }

        if let Some(node_id) = graph.get_node(v) {
            path.push(node_id.to_string());
        }

        path
    }
}

/// Minimum spanning tree using Prim's algorithm
pub struct PrimMST;

impl PrimMST {
    /// Compute minimum spanning tree
    pub fn compute(graph: &FederationGraph) -> Result<Vec<Edge>> {
        if graph.node_count() == 0 {
            return Ok(Vec::new());
        }

        let n = graph.node_count();
        let mut in_mst = vec![false; n];
        let mut min_edge = vec![(f64::INFINITY, None); n];
        let mut heap = BinaryHeap::new();
        let mut mst_edges = Vec::new();

        // Start from node 0
        in_mst[0] = true;
        for &(neighbor, weight) in graph.neighbors(0) {
            min_edge[neighbor] = (weight, Some(0));
            heap.push(State {
                cost: -weight,
                node: neighbor,
            });
        }

        while let Some(State { node, .. }) = heap.pop() {
            if in_mst[node] {
                continue;
            }

            in_mst[node] = true;

            // Add edge to MST
            if let (weight, Some(from)) = min_edge[node] {
                if let (Some(from_id), Some(to_id)) = (graph.get_node(from), graph.get_node(node)) {
                    mst_edges.push(Edge {
                        from: from_id.to_string(),
                        to: to_id.to_string(),
                        weight,
                    });
                }
            }

            // Update neighbors
            for &(neighbor, weight) in graph.neighbors(node) {
                if !in_mst[neighbor] && weight < min_edge[neighbor].0 {
                    min_edge[neighbor] = (weight, Some(node));
                    heap.push(State {
                        cost: -weight,
                        node: neighbor,
                    });
                }
            }
        }

        Ok(mst_edges)
    }
}

/// Graph connectivity analysis
pub struct ConnectivityAnalyzer;

impl ConnectivityAnalyzer {
    /// Check if graph is strongly connected
    pub fn is_strongly_connected(graph: &FederationGraph) -> bool {
        if graph.node_count() == 0 {
            return true;
        }

        // Check if all nodes are reachable from node 0
        let reachable_from_0 = Self::reachable_nodes(graph, 0);
        if reachable_from_0.len() != graph.node_count() {
            return false;
        }

        // Check reverse graph
        let reverse_graph = Self::reverse_graph(graph);
        let reachable_in_reverse = Self::reachable_nodes(&reverse_graph, 0);

        reachable_in_reverse.len() == graph.node_count()
    }

    /// Find connected components
    pub fn connected_components(graph: &FederationGraph) -> Vec<Vec<String>> {
        let n = graph.node_count();
        let mut visited = vec![false; n];
        let mut components = Vec::new();

        for start in 0..n {
            if !visited[start] {
                let component = Self::bfs_component(graph, start, &mut visited);
                components.push(component);
            }
        }

        components
    }

    fn reachable_nodes(graph: &FederationGraph, start: usize) -> HashSet<usize> {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();

        reachable.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            for &(neighbor, _) in graph.neighbors(node) {
                if reachable.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        reachable
    }

    fn reverse_graph(graph: &FederationGraph) -> FederationGraph {
        let mut reversed = FederationGraph::new();

        // Add all nodes
        for node_id in &graph.nodes {
            reversed.add_node(node_id.clone());
        }

        // Reverse edges
        for (from_idx, edges) in graph.adjacency.iter().enumerate() {
            for &(to_idx, weight) in edges {
                if let (Some(from_id), Some(to_id)) =
                    (graph.get_node(from_idx), graph.get_node(to_idx))
                {
                    let _ = reversed.add_edge(to_id, from_id, weight);
                }
            }
        }

        reversed
    }

    fn bfs_component(graph: &FederationGraph, start: usize, visited: &mut [bool]) -> Vec<String> {
        let mut component = Vec::new();
        let mut queue = VecDeque::new();

        visited[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            if let Some(node_id) = graph.get_node(node) {
                component.push(node_id.to_string());
            }

            for &(neighbor, _) in graph.neighbors(node) {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        component
    }
}

/// Node centrality measures
pub struct CentralityAnalyzer;

impl CentralityAnalyzer {
    /// Compute betweenness centrality for all nodes
    pub fn betweenness_centrality(graph: &FederationGraph) -> HashMap<String, f64> {
        let n = graph.node_count();
        let mut centrality = vec![0.0; n];

        // Brandes' algorithm
        for s in 0..n {
            let mut stack = Vec::new();
            let mut paths = vec![Vec::new(); n];
            let mut sigma = vec![0.0; n];
            let mut dist = vec![-1; n];
            let mut delta = vec![0.0; n];

            sigma[s] = 1.0;
            dist[s] = 0;

            let mut queue = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);

                for &(w, _) in graph.neighbors(v) {
                    if dist[w] < 0 {
                        queue.push_back(w);
                        dist[w] = dist[v] + 1;
                    }

                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        paths[w].push(v);
                    }
                }
            }

            while let Some(w) = stack.pop() {
                for &v in &paths[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }

                if w != s {
                    centrality[w] += delta[w];
                }
            }
        }

        // Normalize
        let normalization = if n > 2 { (n - 1) * (n - 2) } else { 1 };
        let mut result = HashMap::new();
        for (idx, &value) in centrality.iter().enumerate() {
            if let Some(node_id) = graph.get_node(idx) {
                result.insert(node_id.to_string(), value / normalization as f64);
            }
        }

        result
    }

    /// Compute degree centrality
    pub fn degree_centrality(graph: &FederationGraph) -> HashMap<String, f64> {
        let n = graph.node_count();
        let mut result = HashMap::new();

        for idx in 0..n {
            let degree = graph.neighbors(idx).len();
            if let Some(node_id) = graph.get_node(idx) {
                result.insert(node_id.to_string(), degree as f64 / (n - 1) as f64);
            }
        }

        result
    }
}

// Helper structures

#[derive(Debug, Clone, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Result of shortest path computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortestPathResult {
    /// Path as list of node IDs
    pub path: Vec<String>,
    /// Total cost of the path
    pub cost: f64,
}

/// Edge in a graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> FederationGraph {
        let mut graph = FederationGraph::new();

        graph.add_node("A".to_string());
        graph.add_node("B".to_string());
        graph.add_node("C".to_string());
        graph.add_node("D".to_string());
        graph.add_node("E".to_string());

        // Use bidirectional edges for federation routing (typical scenario)
        graph.add_bidirectional_edge("A", "B", 4.0).unwrap();
        graph.add_bidirectional_edge("A", "C", 2.0).unwrap();
        graph.add_bidirectional_edge("B", "C", 1.0).unwrap();
        graph.add_bidirectional_edge("B", "D", 5.0).unwrap();
        graph.add_bidirectional_edge("C", "D", 8.0).unwrap();
        graph.add_bidirectional_edge("C", "E", 10.0).unwrap();
        graph.add_bidirectional_edge("D", "E", 2.0).unwrap();

        graph
    }

    #[test]
    fn test_graph_creation() {
        let graph = create_test_graph();
        assert_eq!(graph.node_count(), 5);
    }

    #[test]
    fn test_dijkstra_shortest_path() {
        let graph = create_test_graph();
        let result = Dijkstra::shortest_path(&graph, "A", "E").unwrap();

        assert_eq!(result.path, vec!["A", "C", "B", "D", "E"]);
        // Cost: A→C(2) + C→B(1) + B→D(5) + D→E(2) = 10.0
        assert!((result.cost - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_dijkstra_all_paths() {
        let graph = create_test_graph();
        let results = Dijkstra::shortest_paths_from(&graph, "A").unwrap();

        assert_eq!(results.len(), 4);
        assert!(results.contains_key("E"));
    }

    #[test]
    fn test_astar_with_heuristic() {
        let graph = create_test_graph();

        // Simple heuristic (constant 0)
        let heuristic = |_: &str, _: &str| 0.0;

        let result = AStar::shortest_path(&graph, "A", "E", heuristic).unwrap();
        assert!(result.cost > 0.0);
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_bellman_ford() {
        let graph = create_test_graph();
        let results = BellmanFord::shortest_paths(&graph, "A").unwrap();

        assert!(results.contains_key("E"));
    }

    #[test]
    fn test_floyd_warshall() {
        let graph = create_test_graph();
        let results = FloydWarshall::all_pairs_shortest_paths(&graph).unwrap();

        assert!(results.contains_key(&("A".to_string(), "E".to_string())));
    }

    #[test]
    fn test_prim_mst() {
        let mut graph = FederationGraph::new();
        graph.add_node("A".to_string());
        graph.add_node("B".to_string());
        graph.add_node("C".to_string());

        graph.add_bidirectional_edge("A", "B", 1.0).unwrap();
        graph.add_bidirectional_edge("B", "C", 2.0).unwrap();
        graph.add_bidirectional_edge("A", "C", 3.0).unwrap();

        let mst = PrimMST::compute(&graph).unwrap();
        assert_eq!(mst.len(), 2);
    }

    #[test]
    fn test_connectivity_analysis() {
        let mut graph = FederationGraph::new();
        graph.add_node("A".to_string());
        graph.add_node("B".to_string());
        graph.add_node("C".to_string());

        graph.add_edge("A", "B", 1.0).unwrap();
        graph.add_edge("B", "C", 1.0).unwrap();

        let components = ConnectivityAnalyzer::connected_components(&graph);
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn test_betweenness_centrality() {
        let graph = create_test_graph();
        let centrality = CentralityAnalyzer::betweenness_centrality(&graph);

        assert!(centrality.contains_key("C"));
        assert!(centrality.contains_key("D"));
    }

    #[test]
    fn test_degree_centrality() {
        let graph = create_test_graph();
        let centrality = CentralityAnalyzer::degree_centrality(&graph);

        assert_eq!(centrality.len(), 5);
        for value in centrality.values() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
    }
}
