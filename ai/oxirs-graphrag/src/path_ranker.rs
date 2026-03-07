//! # Knowledge Graph Path Ranker
//!
//! Implements path finding and scoring over an in-memory knowledge graph for
//! GraphRAG retrieval.  Paths are discovered via DFS (cycle-free, bounded by
//! `max_hops`) and scored with a configurable weighting scheme.  Dijkstra is
//! also provided for shortest-path queries.
//!
//! # Example
//!
//! ```rust
//! use oxirs_graphrag::path_ranker::{
//!     KgNode, KgEdge, KnowledgeGraph, PathRanker, PathRankingConfig,
//! };
//!
//! let mut g = KnowledgeGraph::new();
//! g.add_node(KgNode { id: "A".into(), label: "A".into(), node_type: "Entity".into(), importance: 1.0 });
//! g.add_node(KgNode { id: "B".into(), label: "B".into(), node_type: "Entity".into(), importance: 0.5 });
//! g.add_edge(KgEdge { from_id: "A".into(), to_id: "B".into(), relation: "knows".into(), weight: 1.0, confidence: 0.9 });
//!
//! let paths = PathRanker::find_paths(&g, "A", "B", 2);
//! assert_eq!(paths.len(), 1);
//! ```

use std::collections::{BinaryHeap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A node in the knowledge graph
#[derive(Debug, Clone)]
pub struct KgNode {
    /// Unique node identifier (IRI or local name)
    pub id: String,
    /// Human-readable label
    pub label: String,
    /// Ontological or categorical type
    pub node_type: String,
    /// Domain-specific importance score (0..∞)
    pub importance: f64,
}

/// A directed edge between two nodes
#[derive(Debug, Clone)]
pub struct KgEdge {
    /// Source node id
    pub from_id: String,
    /// Target node id
    pub to_id: String,
    /// Relation / predicate label
    pub relation: String,
    /// Edge weight (higher = stronger association)
    pub weight: f64,
    /// Confidence in this edge (0–1)
    pub confidence: f64,
}

/// A path through the knowledge graph
#[derive(Debug, Clone)]
pub struct KgPath {
    /// Ordered list of node ids from start to end
    pub nodes: Vec<String>,
    /// Ordered list of relation labels along the path
    pub edges: Vec<String>,
    /// Sum of edge weights
    pub total_weight: f64,
    /// Number of hops (= edges.len())
    pub hop_count: usize,
}

impl KgPath {
    /// Returns the (start, end) node id pair.
    pub fn endpoint_pair(&self) -> (&str, &str) {
        let start = self.nodes.first().map(String::as_str).unwrap_or("");
        let end = self.nodes.last().map(String::as_str).unwrap_or("");
        (start, end)
    }
}

// ---------------------------------------------------------------------------
// Ranking configuration
// ---------------------------------------------------------------------------

/// Configuration knobs for the `PathRanker` scoring function
#[derive(Debug, Clone)]
pub struct PathRankingConfig {
    /// Weight applied to the raw edge-weight sum
    pub weight_factor: f64,
    /// Multiplier applied once per hop: score *= hop_penalty^hops
    pub hop_penalty: f64,
    /// Scales the product-of-confidences term
    pub confidence_factor: f64,
    /// Additive bonus per unit of node importance
    pub importance_bonus: f64,
}

impl Default for PathRankingConfig {
    fn default() -> Self {
        Self {
            weight_factor: 1.0,
            hop_penalty: 0.9,
            confidence_factor: 1.0,
            importance_bonus: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Knowledge graph (adjacency list)
// ---------------------------------------------------------------------------

/// In-memory directed knowledge graph backed by adjacency lists
pub struct KnowledgeGraph {
    nodes: HashMap<String, KgNode>,
    /// adjacency: source → Vec<(edge_index, target_id)>
    adj: HashMap<String, Vec<usize>>,
    edges: Vec<KgEdge>,
}

impl KnowledgeGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            adj: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Insert or replace a node.
    pub fn add_node(&mut self, node: KgNode) {
        self.adj.entry(node.id.clone()).or_default();
        self.nodes.insert(node.id.clone(), node);
    }

    /// Append a directed edge.  Both endpoint nodes are auto-created with
    /// default values if they do not yet exist.
    pub fn add_edge(&mut self, edge: KgEdge) {
        // Ensure nodes exist
        for id in [&edge.from_id, &edge.to_id] {
            if !self.nodes.contains_key(id.as_str()) {
                let n = KgNode {
                    id: id.to_string(),
                    label: id.to_string(),
                    node_type: "Unknown".to_string(),
                    importance: 0.0,
                };
                self.nodes.insert(id.to_string(), n);
                self.adj.entry(id.to_string()).or_default();
            }
        }

        let idx = self.edges.len();
        self.adj.entry(edge.from_id.clone()).or_default().push(idx);
        self.edges.push(edge);
    }

    /// Return outgoing (edge, target_node) pairs for `node_id`.
    pub fn neighbors<'a>(&'a self, node_id: &str) -> Vec<(&'a KgEdge, &'a KgNode)> {
        let Some(edge_indices) = self.adj.get(node_id) else {
            return Vec::new();
        };
        edge_indices
            .iter()
            .filter_map(|&idx| {
                let edge = &self.edges[idx];
                let node = self.nodes.get(&edge.to_id)?;
                Some((edge, node))
            })
            .collect()
    }

    /// Number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Look up a node by id.
    pub fn get_node(&self, id: &str) -> Option<&KgNode> {
        self.nodes.get(id)
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PathRanker
// ---------------------------------------------------------------------------

/// Stateless path-finding and scoring engine
pub struct PathRanker;

impl PathRanker {
    // -----------------------------------------------------------------------
    // Path finding (DFS, cycle-free)
    // -----------------------------------------------------------------------

    /// Find all simple paths from `start` to `end` with at most `max_hops` edges.
    ///
    /// Uses iterative DFS with an explicit stack to avoid stack overflow on
    /// large graphs.
    pub fn find_paths(
        graph: &KnowledgeGraph,
        start: &str,
        end: &str,
        max_hops: usize,
    ) -> Vec<KgPath> {
        if max_hops == 0 {
            return Vec::new();
        }

        let mut results: Vec<KgPath> = Vec::new();

        // Stack item: (current_node_id, path_nodes, path_edge_labels, acc_weight, visited)
        type StackItem = (String, Vec<String>, Vec<String>, f64, HashSet<String>);

        let mut stack: Vec<StackItem> = Vec::new();
        let mut initial_visited = HashSet::new();
        initial_visited.insert(start.to_string());
        stack.push((
            start.to_string(),
            vec![start.to_string()],
            Vec::new(),
            0.0,
            initial_visited,
        ));

        while let Some((current, nodes, edges_so_far, weight, visited)) = stack.pop() {
            if current == end && !edges_so_far.is_empty() {
                results.push(KgPath {
                    hop_count: edges_so_far.len(),
                    nodes: nodes.clone(),
                    edges: edges_so_far.clone(),
                    total_weight: weight,
                });
                // Continue — there may be longer paths through other routes
            }

            if edges_so_far.len() >= max_hops {
                continue;
            }

            for (edge, neighbor) in graph.neighbors(&current) {
                if visited.contains(&neighbor.id) {
                    continue; // avoid cycles
                }

                let mut new_nodes = nodes.clone();
                new_nodes.push(neighbor.id.clone());

                let mut new_edges = edges_so_far.clone();
                new_edges.push(edge.relation.clone());

                let mut new_visited = visited.clone();
                new_visited.insert(neighbor.id.clone());

                stack.push((
                    neighbor.id.clone(),
                    new_nodes,
                    new_edges,
                    weight + edge.weight,
                    new_visited,
                ));
            }
        }

        results
    }

    // -----------------------------------------------------------------------
    // Scoring
    // -----------------------------------------------------------------------

    /// Score a single path according to `config`.
    ///
    /// score = (Σ edge_weights) * weight_factor
    ///       * hop_penalty^hop_count
    ///       * (Π edge_confidences) * confidence_factor
    ///       + (Σ node_importance) * importance_bonus
    pub fn score_path(graph: &KnowledgeGraph, path: &KgPath, config: &PathRankingConfig) -> f64 {
        // Raw edge weight contribution
        let weight_score = path.total_weight * config.weight_factor;

        // Hop penalty (shorter paths preferred when penalty < 1)
        let hop_multiplier = config.hop_penalty.powi(path.hop_count as i32);

        // Confidence: product of all edge confidences along path
        let confidence_product: f64 = Self::edge_confidences_product(graph, path);
        let confidence_score = confidence_product * config.confidence_factor;

        // Importance bonus from all intermediate and endpoint nodes
        let importance_sum: f64 = path
            .nodes
            .iter()
            .filter_map(|id| graph.get_node(id))
            .map(|n| n.importance)
            .sum();
        let importance_score = importance_sum * config.importance_bonus;

        (weight_score * hop_multiplier * confidence_score) + importance_score
    }

    /// Sort paths by descending score and return (path, score) pairs.
    pub fn rank_paths(
        graph: &KnowledgeGraph,
        paths: Vec<KgPath>,
        config: &PathRankingConfig,
    ) -> Vec<(KgPath, f64)> {
        let mut scored: Vec<(KgPath, f64)> = paths
            .into_iter()
            .map(|p| {
                let s = Self::score_path(graph, &p, config);
                (p, s)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    // -----------------------------------------------------------------------
    // Dijkstra (shortest path by 1/weight)
    // -----------------------------------------------------------------------

    /// Find the shortest path from `start` to `end` using Dijkstra's algorithm.
    ///
    /// Distance metric: `d = 1/weight` (higher weight = shorter distance).
    /// Edges with weight ≤ 0 are treated as distance `f64::INFINITY`.
    pub fn shortest_path_dijkstra(
        graph: &KnowledgeGraph,
        start: &str,
        end: &str,
    ) -> Option<KgPath> {
        if !graph.nodes.contains_key(start) || !graph.nodes.contains_key(end) {
            return None;
        }
        if start == end {
            return Some(KgPath {
                nodes: vec![start.to_string()],
                edges: Vec::new(),
                total_weight: 0.0,
                hop_count: 0,
            });
        }

        // (neg_dist_so_far, node_id, path_nodes, path_edge_labels, acc_weight)
        // We negate distance because BinaryHeap is a max-heap.
        #[derive(PartialEq)]
        struct HeapItem {
            neg_dist: f64,
            node: String,
            nodes: Vec<String>,
            edges: Vec<String>,
            acc_weight: f64,
        }

        impl Eq for HeapItem {}

        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.neg_dist
                    .partial_cmp(&other.neg_dist)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let mut dist_map: HashMap<String, f64> = HashMap::new();
        dist_map.insert(start.to_string(), 0.0);

        let mut heap = BinaryHeap::new();
        heap.push(HeapItem {
            neg_dist: 0.0,
            node: start.to_string(),
            nodes: vec![start.to_string()],
            edges: Vec::new(),
            acc_weight: 0.0,
        });

        while let Some(item) = heap.pop() {
            let current_dist = -item.neg_dist;

            if item.node == end {
                return Some(KgPath {
                    hop_count: item.edges.len(),
                    nodes: item.nodes,
                    edges: item.edges,
                    total_weight: item.acc_weight,
                });
            }

            // Skip stale entries
            if let Some(&best) = dist_map.get(&item.node) {
                if current_dist > best + 1e-12 {
                    continue;
                }
            }

            for (edge, neighbor) in graph.neighbors(&item.node) {
                // Avoid revisiting nodes already on this path
                if item.nodes.contains(&neighbor.id) {
                    continue;
                }

                let step_dist = if edge.weight > 0.0 {
                    1.0 / edge.weight
                } else {
                    f64::INFINITY
                };
                let new_dist = current_dist + step_dist;

                let best = dist_map.entry(neighbor.id.clone()).or_insert(f64::INFINITY);
                if new_dist < *best - 1e-12 {
                    *best = new_dist;

                    let mut new_nodes = item.nodes.clone();
                    new_nodes.push(neighbor.id.clone());
                    let mut new_edges = item.edges.clone();
                    new_edges.push(edge.relation.clone());

                    heap.push(HeapItem {
                        neg_dist: -new_dist,
                        node: neighbor.id.clone(),
                        nodes: new_nodes,
                        edges: new_edges,
                        acc_weight: item.acc_weight + edge.weight,
                    });
                }
            }
        }

        None // no path found
    }

    // -----------------------------------------------------------------------
    // Combined: find + rank + top-k
    // -----------------------------------------------------------------------

    /// Find all paths up to `max_hops`, rank them, and return the top-`k`.
    pub fn most_relevant_paths(
        graph: &KnowledgeGraph,
        start: &str,
        end: &str,
        max_hops: usize,
        top_k: usize,
        config: &PathRankingConfig,
    ) -> Vec<(KgPath, f64)> {
        let paths = Self::find_paths(graph, start, end, max_hops);
        let mut ranked = Self::rank_paths(graph, paths, config);
        ranked.truncate(top_k);
        ranked
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Product of edge confidences along a path.
    ///
    /// We match edges by walking pairs of consecutive nodes in the path and
    /// finding an edge in the graph between them with the recorded relation.
    fn edge_confidences_product(graph: &KnowledgeGraph, path: &KgPath) -> f64 {
        if path.edges.is_empty() {
            return 1.0;
        }
        let mut product = 1.0;
        for (i, relation) in path.edges.iter().enumerate() {
            let from = &path.nodes[i];
            let confidence = graph
                .neighbors(from)
                .into_iter()
                .find(|(e, _)| &e.relation == relation)
                .map(|(e, _)| e.confidence)
                .unwrap_or(1.0);
            product *= confidence;
        }
        product
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple triangle graph: A → B → C, A → C directly
    fn triangle_graph() -> KnowledgeGraph {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode {
            id: "A".into(),
            label: "Alpha".into(),
            node_type: "Entity".into(),
            importance: 1.0,
        });
        g.add_node(KgNode {
            id: "B".into(),
            label: "Beta".into(),
            node_type: "Entity".into(),
            importance: 0.5,
        });
        g.add_node(KgNode {
            id: "C".into(),
            label: "Gamma".into(),
            node_type: "Entity".into(),
            importance: 0.8,
        });
        g.add_edge(KgEdge {
            from_id: "A".into(),
            to_id: "B".into(),
            relation: "knows".into(),
            weight: 1.0,
            confidence: 0.9,
        });
        g.add_edge(KgEdge {
            from_id: "B".into(),
            to_id: "C".into(),
            relation: "related".into(),
            weight: 2.0,
            confidence: 0.8,
        });
        g.add_edge(KgEdge {
            from_id: "A".into(),
            to_id: "C".into(),
            relation: "direct".into(),
            weight: 0.5,
            confidence: 0.95,
        });
        g
    }

    // --- Graph construction ---

    #[test]
    fn test_graph_node_count() {
        let g = triangle_graph();
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_graph_edge_count() {
        let g = triangle_graph();
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_graph_neighbors() {
        let g = triangle_graph();
        let nb = g.neighbors("A");
        assert_eq!(nb.len(), 2);
    }

    #[test]
    fn test_graph_add_node_idempotent() {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode {
            id: "X".into(),
            label: "X".into(),
            node_type: "T".into(),
            importance: 1.0,
        });
        g.add_node(KgNode {
            id: "X".into(),
            label: "X2".into(),
            node_type: "T2".into(),
            importance: 2.0,
        });
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_auto_create_nodes_on_add_edge() {
        let mut g = KnowledgeGraph::new();
        g.add_edge(KgEdge {
            from_id: "P".into(),
            to_id: "Q".into(),
            relation: "r".into(),
            weight: 1.0,
            confidence: 1.0,
        });
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_get_node() {
        let g = triangle_graph();
        let n = g.get_node("B");
        assert!(n.is_some());
        assert_eq!(n.unwrap().label, "Beta");
    }

    #[test]
    fn test_get_node_missing() {
        let g = triangle_graph();
        assert!(g.get_node("Z").is_none());
    }

    // --- Path finding ---

    #[test]
    fn test_find_paths_direct() {
        let g = triangle_graph();
        let paths = PathRanker::find_paths(&g, "A", "C", 1);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].edges[0], "direct");
    }

    #[test]
    fn test_find_paths_two_hops() {
        let g = triangle_graph();
        let paths = PathRanker::find_paths(&g, "A", "C", 2);
        // direct (1 hop) + via B (2 hops)
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_find_paths_no_path() {
        let g = triangle_graph();
        let paths = PathRanker::find_paths(&g, "C", "A", 5);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_find_paths_zero_max_hops() {
        let g = triangle_graph();
        let paths = PathRanker::find_paths(&g, "A", "C", 0);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_find_paths_cycle_avoidance() {
        let mut g = KnowledgeGraph::new();
        // A <-> B
        g.add_edge(KgEdge {
            from_id: "A".into(),
            to_id: "B".into(),
            relation: "ab".into(),
            weight: 1.0,
            confidence: 1.0,
        });
        g.add_edge(KgEdge {
            from_id: "B".into(),
            to_id: "A".into(),
            relation: "ba".into(),
            weight: 1.0,
            confidence: 1.0,
        });
        // No path from A to C
        let paths = PathRanker::find_paths(&g, "A", "C", 10);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_find_paths_self_loop_ignored() {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode {
            id: "A".into(),
            label: "A".into(),
            node_type: "E".into(),
            importance: 1.0,
        });
        g.add_node(KgNode {
            id: "B".into(),
            label: "B".into(),
            node_type: "E".into(),
            importance: 1.0,
        });
        g.add_edge(KgEdge {
            from_id: "A".into(),
            to_id: "A".into(),
            relation: "self".into(),
            weight: 1.0,
            confidence: 1.0,
        });
        g.add_edge(KgEdge {
            from_id: "A".into(),
            to_id: "B".into(),
            relation: "ab".into(),
            weight: 1.0,
            confidence: 1.0,
        });
        let paths = PathRanker::find_paths(&g, "A", "B", 2);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_find_paths_hop_count_correct() {
        let g = triangle_graph();
        let paths = PathRanker::find_paths(&g, "A", "C", 2);
        let hops: Vec<usize> = paths.iter().map(|p| p.hop_count).collect();
        assert!(hops.contains(&1));
        assert!(hops.contains(&2));
    }

    #[test]
    fn test_endpoint_pair() {
        let path = KgPath {
            nodes: vec!["A".into(), "B".into(), "C".into()],
            edges: vec!["r1".into(), "r2".into()],
            total_weight: 3.0,
            hop_count: 2,
        };
        let (s, e) = path.endpoint_pair();
        assert_eq!(s, "A");
        assert_eq!(e, "C");
    }

    // --- Scoring / ranking ---

    #[test]
    fn test_score_path_direct_higher_with_low_hop_penalty() {
        // Use a graph where the direct edge has high weight so direct beats 2-hop.
        // Direct A→C: weight=10, confidence=0.99
        // Via B: A→B weight=1, B→C weight=1 (total=2, two hops)
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode {
            id: "A".into(),
            label: "A".into(),
            node_type: "E".into(),
            importance: 0.0,
        });
        g.add_node(KgNode {
            id: "B".into(),
            label: "B".into(),
            node_type: "E".into(),
            importance: 0.0,
        });
        g.add_node(KgNode {
            id: "C".into(),
            label: "C".into(),
            node_type: "E".into(),
            importance: 0.0,
        });
        g.add_edge(KgEdge {
            from_id: "A".into(),
            to_id: "B".into(),
            relation: "ab".into(),
            weight: 1.0,
            confidence: 0.9,
        });
        g.add_edge(KgEdge {
            from_id: "B".into(),
            to_id: "C".into(),
            relation: "bc".into(),
            weight: 1.0,
            confidence: 0.9,
        });
        g.add_edge(KgEdge {
            from_id: "A".into(),
            to_id: "C".into(),
            relation: "direct".into(),
            weight: 10.0,
            confidence: 0.99,
        });

        let config = PathRankingConfig {
            weight_factor: 1.0,
            hop_penalty: 0.5, // strong penalty per hop
            confidence_factor: 1.0,
            importance_bonus: 0.0,
        };
        let paths = PathRanker::find_paths(&g, "A", "C", 2);
        assert_eq!(paths.len(), 2, "expected 2 paths");
        let scores: Vec<f64> = paths
            .iter()
            .map(|p| PathRanker::score_path(&g, p, &config))
            .collect();
        // Direct path (1 hop): 10 * 1.0 * 0.5^1 * 0.99 * 1.0 + 0 = 4.95
        // Via B (2 hops):  2 * 1.0 * 0.5^2 * (0.9*0.9) * 1.0 + 0 = 2*0.25*0.81 = 0.405
        let (direct, two_hop) = if paths[0].hop_count == 1 {
            (scores[0], scores[1])
        } else {
            (scores[1], scores[0])
        };
        assert!(direct > two_hop, "direct={direct}, two_hop={two_hop}");
    }

    #[test]
    fn test_rank_paths_sorted_descending() {
        let g = triangle_graph();
        let paths = PathRanker::find_paths(&g, "A", "C", 2);
        let config = PathRankingConfig::default();
        let ranked = PathRanker::rank_paths(&g, paths, &config);
        assert!(ranked.len() <= 2);
        if ranked.len() == 2 {
            assert!(ranked[0].1 >= ranked[1].1);
        }
    }

    #[test]
    fn test_rank_paths_empty() {
        let g = triangle_graph();
        let ranked = PathRanker::rank_paths(&g, vec![], &PathRankingConfig::default());
        assert!(ranked.is_empty());
    }

    // --- Dijkstra ---

    #[test]
    fn test_dijkstra_direct_path() {
        let g = triangle_graph();
        let path = PathRanker::shortest_path_dijkstra(&g, "A", "B");
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p.nodes, vec!["A", "B"]);
    }

    #[test]
    fn test_dijkstra_same_node() {
        let g = triangle_graph();
        let path = PathRanker::shortest_path_dijkstra(&g, "A", "A");
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p.nodes, vec!["A"]);
        assert_eq!(p.hop_count, 0);
    }

    #[test]
    fn test_dijkstra_no_path() {
        let g = triangle_graph();
        let path = PathRanker::shortest_path_dijkstra(&g, "C", "A");
        assert!(path.is_none());
    }

    #[test]
    fn test_dijkstra_missing_node() {
        let g = triangle_graph();
        let path = PathRanker::shortest_path_dijkstra(&g, "A", "Z");
        assert!(path.is_none());
    }

    #[test]
    fn test_dijkstra_prefers_high_weight_edge() {
        // A --weight=1--> B --weight=10--> C
        // A --weight=2--> C  (Dijkstra by 1/weight: 2-hop = 1/1+1/10=1.1; direct = 1/2=0.5)
        // Dijkstra picks direct (lower distance = higher weight)
        let mut g = KnowledgeGraph::new();
        g.add_edge(KgEdge {
            from_id: "A".into(),
            to_id: "B".into(),
            relation: "r1".into(),
            weight: 1.0,
            confidence: 1.0,
        });
        g.add_edge(KgEdge {
            from_id: "B".into(),
            to_id: "C".into(),
            relation: "r2".into(),
            weight: 10.0,
            confidence: 1.0,
        });
        g.add_edge(KgEdge {
            from_id: "A".into(),
            to_id: "C".into(),
            relation: "direct".into(),
            weight: 2.0,
            confidence: 1.0,
        });
        let path = PathRanker::shortest_path_dijkstra(&g, "A", "C").unwrap();
        // dist(direct) = 1/2 = 0.5
        // dist(via B)  = 1/1 + 1/10 = 1.1
        // shortest = direct
        assert_eq!(path.hop_count, 1);
        assert_eq!(path.edges[0], "direct");
    }

    // --- Most relevant paths ---

    #[test]
    fn test_most_relevant_paths_top_k() {
        let g = triangle_graph();
        let config = PathRankingConfig::default();
        let results = PathRanker::most_relevant_paths(&g, "A", "C", 3, 1, &config);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_most_relevant_paths_no_path() {
        let g = triangle_graph();
        let config = PathRankingConfig::default();
        let results = PathRanker::most_relevant_paths(&g, "C", "A", 5, 10, &config);
        assert!(results.is_empty());
    }

    #[test]
    fn test_most_relevant_paths_all_returned_when_k_large() {
        let g = triangle_graph();
        let config = PathRankingConfig::default();
        let results = PathRanker::most_relevant_paths(&g, "A", "C", 2, 100, &config);
        assert_eq!(results.len(), 2);
    }

    // --- Disconnected graph ---

    #[test]
    fn test_disconnected_graph() {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode {
            id: "X".into(),
            label: "X".into(),
            node_type: "E".into(),
            importance: 1.0,
        });
        g.add_node(KgNode {
            id: "Y".into(),
            label: "Y".into(),
            node_type: "E".into(),
            importance: 1.0,
        });
        // No edges
        let paths = PathRanker::find_paths(&g, "X", "Y", 5);
        assert!(paths.is_empty());
        let shortest = PathRanker::shortest_path_dijkstra(&g, "X", "Y");
        assert!(shortest.is_none());
    }

    // --- Additional scoring tests ---

    #[test]
    fn test_score_increases_with_importance_bonus() {
        let g = triangle_graph();
        let config_low = PathRankingConfig {
            importance_bonus: 0.0,
            ..Default::default()
        };
        let config_high = PathRankingConfig {
            importance_bonus: 10.0,
            ..Default::default()
        };
        let paths = PathRanker::find_paths(&g, "A", "C", 1);
        assert!(!paths.is_empty());
        let s_low = PathRanker::score_path(&g, &paths[0], &config_low);
        let s_high = PathRanker::score_path(&g, &paths[0], &config_high);
        assert!(s_high > s_low);
    }

    #[test]
    fn test_score_path_single_node_path() {
        let g = triangle_graph();
        let single = KgPath {
            nodes: vec!["A".into()],
            edges: Vec::new(),
            total_weight: 0.0,
            hop_count: 0,
        };
        let config = PathRankingConfig::default();
        let s = PathRanker::score_path(&g, &single, &config);
        // weight_score=0, hop_multiplier=1, confidence=1, importance=1.0*0.1=0.1
        assert!((s - 0.1).abs() < 1e-9, "score={s}");
    }
}
