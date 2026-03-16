//! K-hop subgraph extraction from knowledge graphs.
//!
//! Provides BFS-based neighbourhood traversal, shortest-path computation,
//! connected-component discovery, and degree queries for in-memory RDF-like
//! knowledge graphs.

use std::collections::{HashMap, HashSet, VecDeque};

/// A directed, weighted edge in a knowledge graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// Source node identifier
    pub from: String,
    /// Target node identifier
    pub to: String,
    /// Edge predicate / relationship label
    pub predicate: String,
    /// Optional edge weight (default 1.0)
    pub weight: f64,
}

/// In-memory knowledge graph with directed, labelled edges.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraph {
    /// All node identifiers
    pub nodes: HashSet<String>,
    /// All edges
    pub edges: Vec<Edge>,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a node. No-op if already present.
    pub fn add_node(&mut self, id: impl Into<String>) {
        self.nodes.insert(id.into());
    }

    /// Insert a directed edge with default weight 1.0, auto-inserting nodes.
    pub fn add_edge(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        predicate: impl Into<String>,
    ) {
        self.add_edge_weighted(from, to, predicate, 1.0);
    }

    /// Insert a directed edge with an explicit weight, auto-inserting nodes.
    pub fn add_edge_weighted(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        predicate: impl Into<String>,
        weight: f64,
    ) {
        let from = from.into();
        let to = to.into();
        let predicate = predicate.into();
        self.nodes.insert(from.clone());
        self.nodes.insert(to.clone());
        self.edges.push(Edge { from, to, predicate, weight });
    }

    /// Total number of distinct nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of edges (including duplicates).
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// All outgoing edges from `node`.
    pub fn neighbors_out(&self, node: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.from == node).collect()
    }

    /// All incoming edges to `node`.
    pub fn neighbors_in(&self, node: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.to == node).collect()
    }
}

/// The result of a subgraph extraction.
#[derive(Debug, Clone, Default)]
pub struct SubgraphResult {
    /// Nodes included in the subgraph
    pub nodes: HashSet<String>,
    /// Edges included in the subgraph
    pub edges: Vec<Edge>,
    /// BFS depth at which each node was first discovered (start node = 0)
    pub depth_map: HashMap<String, usize>,
}

/// Direction to traverse edges during subgraph extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    /// Follow outgoing edges only
    Outgoing,
    /// Follow incoming edges only
    Incoming,
    /// Follow both outgoing and incoming edges
    Both,
}

/// Configuration for subgraph extraction.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Maximum number of hops from any start node
    pub max_hops: usize,
    /// Hard limit on the total number of nodes in the result (BFS order)
    pub max_nodes: usize,
    /// If `Some`, only traverse edges matching these predicates
    pub predicates: Option<Vec<String>>,
    /// Direction to follow edges
    pub direction: TraversalDirection,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            max_nodes: usize::MAX,
            predicates: None,
            direction: TraversalDirection::Both,
        }
    }
}

/// Stateless subgraph extraction engine.
pub struct SubgraphExtractor;

impl SubgraphExtractor {
    /// Extract a k-hop subgraph starting from a single node.
    ///
    /// Uses BFS to discover reachable nodes up to `config.max_hops` hops,
    /// respecting `max_nodes` and optional predicate filters.
    pub fn extract(
        graph: &KnowledgeGraph,
        start_node: &str,
        config: &ExtractionConfig,
    ) -> SubgraphResult {
        Self::extract_multi(graph, &[start_node], config)
    }

    /// Extract a k-hop subgraph starting from multiple seed nodes.
    ///
    /// All seed nodes are placed at depth 0. BFS proceeds jointly from all
    /// seeds. If a node is reachable from multiple seeds it is recorded at
    /// the smallest depth.
    pub fn extract_multi(
        graph: &KnowledgeGraph,
        start_nodes: &[&str],
        config: &ExtractionConfig,
    ) -> SubgraphResult {
        let mut result = SubgraphResult::default();

        // BFS queue: (node_id, current_depth)
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        for &start in start_nodes {
            if graph.nodes.contains(start) && !result.depth_map.contains_key(start) {
                result.nodes.insert(start.to_string());
                result.depth_map.insert(start.to_string(), 0);
                queue.push_back((start.to_string(), 0));
            }
        }

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= config.max_hops {
                continue;
            }
            if result.nodes.len() >= config.max_nodes {
                break;
            }

            // Collect candidate edges
            let candidates: Vec<&Edge> = match config.direction {
                TraversalDirection::Outgoing => graph.neighbors_out(&current),
                TraversalDirection::Incoming => graph.neighbors_in(&current),
                TraversalDirection::Both => {
                    let mut all = graph.neighbors_out(&current);
                    all.extend(graph.neighbors_in(&current));
                    all
                }
            };

            for edge in candidates {
                // Predicate filter
                if let Some(ref predicates) = config.predicates {
                    if !predicates.contains(&edge.predicate) {
                        continue;
                    }
                }

                // Determine the neighbour based on traversal direction
                let neighbour = match config.direction {
                    TraversalDirection::Outgoing => &edge.to,
                    TraversalDirection::Incoming => &edge.from,
                    TraversalDirection::Both => {
                        if edge.from == current {
                            &edge.to
                        } else {
                            &edge.from
                        }
                    }
                };

                // Include the edge if both endpoints are already in the subgraph
                // or will be added now.
                let new_node = !result.depth_map.contains_key(neighbour.as_str());

                if new_node {
                    if result.nodes.len() >= config.max_nodes {
                        continue;
                    }
                    result.nodes.insert(neighbour.clone());
                    result.depth_map.insert(neighbour.clone(), depth + 1);
                    queue.push_back((neighbour.clone(), depth + 1));
                }

                // Add edge if both endpoints are present
                if result.nodes.contains(edge.from.as_str())
                    && result.nodes.contains(edge.to.as_str())
                {
                    // Avoid duplicate edges
                    let already = result.edges.iter().any(|e| {
                        e.from == edge.from
                            && e.to == edge.to
                            && e.predicate == edge.predicate
                    });
                    if !already {
                        result.edges.push(edge.clone());
                    }
                }
            }
        }

        result
    }

    /// Find the shortest undirected path between `from` and `to` using BFS.
    ///
    /// Returns `None` if no path exists or if either node is absent.
    pub fn shortest_path(
        graph: &KnowledgeGraph,
        from: &str,
        to: &str,
    ) -> Option<Vec<String>> {
        if !graph.nodes.contains(from) || !graph.nodes.contains(to) {
            return None;
        }
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut predecessor: HashMap<String, String> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        visited.insert(from.to_string());
        queue.push_back(from.to_string());

        'bfs: while let Some(current) = queue.pop_front() {
            // Collect undirected neighbours (outgoing + incoming)
            let mut neighbours: Vec<String> = graph
                .neighbors_out(&current)
                .iter()
                .map(|e| e.to.clone())
                .collect();
            neighbours.extend(
                graph
                    .neighbors_in(&current)
                    .iter()
                    .map(|e| e.from.clone()),
            );

            for nb in neighbours {
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb.clone());
                predecessor.insert(nb.clone(), current.clone());

                if nb == to {
                    break 'bfs;
                }
                queue.push_back(nb);
            }
        }

        // Reconstruct path
        if !predecessor.contains_key(to) {
            return None;
        }

        let mut path = vec![to.to_string()];
        let mut current = to.to_string();
        while current != from {
            current = predecessor.get(&current)?.clone();
            path.push(current.clone());
        }
        path.reverse();
        Some(path)
    }

    /// Compute the undirected connected component containing `start`.
    ///
    /// Returns an empty set if `start` is not in the graph.
    pub fn connected_component(graph: &KnowledgeGraph, start: &str) -> HashSet<String> {
        if !graph.nodes.contains(start) {
            return HashSet::new();
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        visited.insert(start.to_string());
        queue.push_back(start.to_string());

        while let Some(current) = queue.pop_front() {
            for edge in graph.neighbors_out(&current) {
                if !visited.contains(&edge.to) {
                    visited.insert(edge.to.clone());
                    queue.push_back(edge.to.clone());
                }
            }
            for edge in graph.neighbors_in(&current) {
                if !visited.contains(&edge.from) {
                    visited.insert(edge.from.clone());
                    queue.push_back(edge.from.clone());
                }
            }
        }
        visited
    }

    /// Return the `(in_degree, out_degree)` of a node.
    ///
    /// Returns `(0, 0)` for nodes not present in the graph.
    pub fn node_degree(graph: &KnowledgeGraph, node: &str) -> (usize, usize) {
        let in_deg = graph.neighbors_in(node).len();
        let out_deg = graph.neighbors_out(node).len();
        (in_deg, out_deg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn chain_graph(n: usize) -> KnowledgeGraph {
        // A → B → C → … (n nodes, n-1 edges)
        let mut g = KnowledgeGraph::new();
        for i in 0..n {
            g.add_node(format!("n{}", i));
        }
        for i in 0..n - 1 {
            g.add_edge(format!("n{}", i), format!("n{}", i + 1), "next");
        }
        g
    }

    // ── KnowledgeGraph basics ─────────────────────────────────────────────────

    #[test]
    fn test_graph_empty() {
        let g = KnowledgeGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut g = KnowledgeGraph::new();
        g.add_node("A");
        g.add_node("A"); // idempotent
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_add_edge_auto_inserts_nodes() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "knows");
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_add_edge_weighted() {
        let mut g = KnowledgeGraph::new();
        g.add_edge_weighted("X", "Y", "likes", 0.75);
        assert!((g.edges[0].weight - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_neighbors_out() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p1");
        g.add_edge("A", "C", "p2");
        g.add_edge("B", "C", "p3");
        assert_eq!(g.neighbors_out("A").len(), 2);
        assert_eq!(g.neighbors_out("C").len(), 0);
    }

    #[test]
    fn test_neighbors_in() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "C", "p1");
        g.add_edge("B", "C", "p2");
        assert_eq!(g.neighbors_in("C").len(), 2);
        assert_eq!(g.neighbors_in("A").len(), 0);
    }

    // ── 1-hop extraction ──────────────────────────────────────────────────────

    #[test]
    fn test_extract_1hop_outgoing() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p");
        g.add_edge("A", "C", "p");
        g.add_edge("B", "D", "p");

        let cfg = ExtractionConfig {
            max_hops: 1,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "A", &cfg);
        assert!(r.nodes.contains("A"));
        assert!(r.nodes.contains("B"));
        assert!(r.nodes.contains("C"));
        // D is 2 hops away
        assert!(!r.nodes.contains("D"));
    }

    #[test]
    fn test_extract_1hop_start_depth_zero() {
        let g = chain_graph(3);
        let cfg = ExtractionConfig { max_hops: 1, ..Default::default() };
        let r = SubgraphExtractor::extract(&g, "n0", &cfg);
        assert_eq!(r.depth_map.get("n0"), Some(&0));
        assert_eq!(r.depth_map.get("n1"), Some(&1));
    }

    #[test]
    fn test_extract_node_not_in_graph_empty() {
        let g = chain_graph(3);
        let cfg = ExtractionConfig { max_hops: 2, ..Default::default() };
        let r = SubgraphExtractor::extract(&g, "ghost", &cfg);
        assert!(r.nodes.is_empty());
        assert!(r.edges.is_empty());
    }

    // ── 2-hop extraction ──────────────────────────────────────────────────────

    #[test]
    fn test_extract_2hop_reaches_far_nodes() {
        let g = chain_graph(5);
        let cfg = ExtractionConfig {
            max_hops: 2,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "n0", &cfg);
        assert!(r.nodes.contains("n2"));
        assert!(!r.nodes.contains("n3")); // 3 hops away
    }

    #[test]
    fn test_extract_2hop_depth_map() {
        let g = chain_graph(4);
        let cfg = ExtractionConfig {
            max_hops: 2,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "n0", &cfg);
        assert_eq!(r.depth_map.get("n0"), Some(&0));
        assert_eq!(r.depth_map.get("n1"), Some(&1));
        assert_eq!(r.depth_map.get("n2"), Some(&2));
    }

    // ── max_nodes limit ───────────────────────────────────────────────────────

    #[test]
    fn test_extract_max_nodes_respected() {
        let g = chain_graph(10);
        let cfg = ExtractionConfig {
            max_hops: 10,
            max_nodes: 3,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "n0", &cfg);
        assert!(r.nodes.len() <= 3);
    }

    // ── predicate filter ──────────────────────────────────────────────────────

    #[test]
    fn test_extract_predicate_filter_passes() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "knows");
        g.add_edge("A", "C", "hates");

        let cfg = ExtractionConfig {
            max_hops: 1,
            predicates: Some(vec!["knows".to_string()]),
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "A", &cfg);
        assert!(r.nodes.contains("B"));
        assert!(!r.nodes.contains("C"));
    }

    #[test]
    fn test_extract_predicate_filter_blocks_all() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "knows");

        let cfg = ExtractionConfig {
            max_hops: 2,
            predicates: Some(vec!["nonexistent".to_string()]),
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "A", &cfg);
        // Only the start node should be present
        assert_eq!(r.nodes.len(), 1);
    }

    // ── direction modes ───────────────────────────────────────────────────────

    #[test]
    fn test_direction_incoming_only() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p");
        g.add_edge("C", "B", "p");

        let cfg = ExtractionConfig {
            max_hops: 1,
            direction: TraversalDirection::Incoming,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "B", &cfg);
        assert!(r.nodes.contains("A"));
        assert!(r.nodes.contains("C"));
    }

    #[test]
    fn test_direction_outgoing_does_not_see_predecessors() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p");
        g.add_edge("C", "B", "p");

        let cfg = ExtractionConfig {
            max_hops: 1,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "B", &cfg);
        // B has no outgoing edges; no neighbours should be found
        assert!(!r.nodes.contains("A"));
        assert!(!r.nodes.contains("C"));
    }

    #[test]
    fn test_direction_both_follows_all() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p");
        g.add_edge("B", "C", "p");

        let cfg = ExtractionConfig {
            max_hops: 1,
            direction: TraversalDirection::Both,
            ..Default::default()
        };
        // Starting from B: should see A (via incoming) and C (via outgoing)
        let r = SubgraphExtractor::extract(&g, "B", &cfg);
        assert!(r.nodes.contains("A"));
        assert!(r.nodes.contains("C"));
    }

    // ── extract_multi ─────────────────────────────────────────────────────────

    #[test]
    fn test_extract_multi_merges_seeds() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "X", "p");
        g.add_edge("B", "Y", "p");

        let cfg = ExtractionConfig {
            max_hops: 1,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract_multi(&g, &["A", "B"], &cfg);
        assert!(r.nodes.contains("X"));
        assert!(r.nodes.contains("Y"));
    }

    #[test]
    fn test_extract_multi_both_seeds_at_depth_zero() {
        let mut g = KnowledgeGraph::new();
        g.add_node("P");
        g.add_node("Q");

        let cfg = ExtractionConfig { max_hops: 1, ..Default::default() };
        let r = SubgraphExtractor::extract_multi(&g, &["P", "Q"], &cfg);
        assert_eq!(r.depth_map.get("P"), Some(&0));
        assert_eq!(r.depth_map.get("Q"), Some(&0));
    }

    // ── shortest_path ─────────────────────────────────────────────────────────

    #[test]
    fn test_shortest_path_direct() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p");
        let path = SubgraphExtractor::shortest_path(&g, "A", "B");
        assert_eq!(path, Some(vec!["A".to_string(), "B".to_string()]));
    }

    #[test]
    fn test_shortest_path_two_hops() {
        let g = chain_graph(3);
        let path = SubgraphExtractor::shortest_path(&g, "n0", "n2").expect("path should exist");
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "n0");
        assert_eq!(path[1], "n1");
        assert_eq!(path[2], "n2");
    }

    #[test]
    fn test_shortest_path_no_path() {
        let mut g = KnowledgeGraph::new();
        g.add_node("A");
        g.add_node("B");
        let path = SubgraphExtractor::shortest_path(&g, "A", "B");
        assert!(path.is_none());
    }

    #[test]
    fn test_shortest_path_self() {
        let mut g = KnowledgeGraph::new();
        g.add_node("X");
        let path = SubgraphExtractor::shortest_path(&g, "X", "X");
        assert_eq!(path, Some(vec!["X".to_string()]));
    }

    #[test]
    fn test_shortest_path_missing_node() {
        let g = chain_graph(3);
        assert!(SubgraphExtractor::shortest_path(&g, "n0", "ghost").is_none());
    }

    #[test]
    fn test_shortest_path_prefers_shorter() {
        let mut g = KnowledgeGraph::new();
        // A → B → D (2 hops) and A → C → D (2 hops); no shorter path
        g.add_edge("A", "B", "p");
        g.add_edge("B", "D", "p");
        g.add_edge("A", "C", "p");
        g.add_edge("C", "D", "p");
        // Also A → D directly (1 hop)
        g.add_edge("A", "D", "p");
        let path = SubgraphExtractor::shortest_path(&g, "A", "D")
            .expect("should find path");
        assert_eq!(path.len(), 2); // [A, D]
    }

    // ── connected_component ───────────────────────────────────────────────────

    #[test]
    fn test_connected_component_single() {
        let mut g = KnowledgeGraph::new();
        g.add_node("X");
        let cc = SubgraphExtractor::connected_component(&g, "X");
        assert!(cc.contains("X"));
        assert_eq!(cc.len(), 1);
    }

    #[test]
    fn test_connected_component_chain() {
        let g = chain_graph(4);
        let cc = SubgraphExtractor::connected_component(&g, "n0");
        for i in 0..4 {
            assert!(cc.contains(&format!("n{}", i)));
        }
    }

    #[test]
    fn test_connected_component_isolated_node() {
        let mut g = KnowledgeGraph::new();
        g.add_node("isolated");
        g.add_edge("A", "B", "p");
        let cc = SubgraphExtractor::connected_component(&g, "isolated");
        assert!(!cc.contains("A"));
    }

    #[test]
    fn test_connected_component_missing_node_empty() {
        let g = chain_graph(3);
        let cc = SubgraphExtractor::connected_component(&g, "ghost");
        assert!(cc.is_empty());
    }

    // ── node_degree ───────────────────────────────────────────────────────────

    #[test]
    fn test_node_degree_no_edges() {
        let mut g = KnowledgeGraph::new();
        g.add_node("A");
        assert_eq!(SubgraphExtractor::node_degree(&g, "A"), (0, 0));
    }

    #[test]
    fn test_node_degree_out_only() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p");
        g.add_edge("A", "C", "p");
        assert_eq!(SubgraphExtractor::node_degree(&g, "A"), (0, 2));
    }

    #[test]
    fn test_node_degree_in_only() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("X", "Z", "p");
        g.add_edge("Y", "Z", "p");
        assert_eq!(SubgraphExtractor::node_degree(&g, "Z"), (2, 0));
    }

    #[test]
    fn test_node_degree_missing_node() {
        let g = chain_graph(3);
        assert_eq!(SubgraphExtractor::node_degree(&g, "ghost"), (0, 0));
    }

    // ── Additional edge-case & regression tests ────────────────────────────────

    #[test]
    fn test_extract_edges_included() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "knows");
        let cfg = ExtractionConfig {
            max_hops: 1,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "A", &cfg);
        assert_eq!(r.edges.len(), 1);
        assert_eq!(r.edges[0].predicate, "knows");
    }

    #[test]
    fn test_extract_no_duplicate_edges() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p");
        g.add_edge("A", "B", "p"); // duplicate
        let cfg = ExtractionConfig {
            max_hops: 1,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "A", &cfg);
        assert_eq!(r.edges.len(), 1, "duplicate edges should be deduplicated");
    }

    #[test]
    fn test_graph_default_edge_weight() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p");
        assert!((g.edges[0].weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_multi_empty_seeds() {
        let g = chain_graph(3);
        let cfg = ExtractionConfig { max_hops: 1, ..Default::default() };
        let r = SubgraphExtractor::extract_multi(&g, &[], &cfg);
        assert!(r.nodes.is_empty());
    }

    #[test]
    fn test_extract_0_hops_only_start_node() {
        let g = chain_graph(5);
        let cfg = ExtractionConfig {
            max_hops: 0,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "n0", &cfg);
        assert_eq!(r.nodes.len(), 1);
        assert!(r.nodes.contains("n0"));
    }

    #[test]
    fn test_connected_component_count() {
        let g = chain_graph(5);
        let cc = SubgraphExtractor::connected_component(&g, "n0");
        assert_eq!(cc.len(), 5);
    }

    #[test]
    fn test_node_degree_both_in_and_out() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "p");
        g.add_edge("C", "B", "p");
        g.add_edge("B", "D", "p");
        // B: 2 in (from A, C), 1 out (to D)
        assert_eq!(SubgraphExtractor::node_degree(&g, "B"), (2, 1));
    }

    #[test]
    fn test_extract_weight_preserved_in_result() {
        let mut g = KnowledgeGraph::new();
        g.add_edge_weighted("A", "B", "p", 2.5);
        let cfg = ExtractionConfig {
            max_hops: 1,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "A", &cfg);
        let edge = r.edges.iter().find(|e| e.from == "A" && e.to == "B");
        assert!(edge.is_some());
        assert!((edge.expect("should succeed").weight - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_shortest_path_undirected_follows_reverse_edge() {
        let mut g = KnowledgeGraph::new();
        // Edge goes B → A, but BFS should traverse it in reverse for A→B
        g.add_edge("B", "A", "p");
        let path = SubgraphExtractor::shortest_path(&g, "A", "B");
        assert!(path.is_some(), "should find path via reverse edge traversal");
    }

    #[test]
    fn test_extract_multi_missing_seeds_ignored() {
        let mut g = KnowledgeGraph::new();
        g.add_node("real");
        let cfg = ExtractionConfig { max_hops: 1, ..Default::default() };
        let r = SubgraphExtractor::extract_multi(&g, &["real", "ghost"], &cfg);
        assert!(r.nodes.contains("real"));
        assert!(!r.nodes.contains("ghost"));
    }

    #[test]
    fn test_both_directions_includes_incoming_and_outgoing_edges() {
        let mut g = KnowledgeGraph::new();
        g.add_edge("A", "B", "forward");
        g.add_edge("C", "B", "backward");

        let cfg = ExtractionConfig {
            max_hops: 1,
            direction: TraversalDirection::Both,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "B", &cfg);
        assert!(r.nodes.contains("A"), "incoming source A should be included");
        assert!(!r.nodes.contains("A") || r.edges.iter().any(|e| e.predicate == "backward" || e.predicate == "forward"));
    }

    #[test]
    fn test_depth_map_populated_for_all_reachable_nodes() {
        let g = chain_graph(4);
        let cfg = ExtractionConfig {
            max_hops: 3,
            direction: TraversalDirection::Outgoing,
            ..Default::default()
        };
        let r = SubgraphExtractor::extract(&g, "n0", &cfg);
        for node in &r.nodes {
            assert!(r.depth_map.contains_key(node.as_str()),
                "node {node} missing from depth_map");
        }
    }
}
