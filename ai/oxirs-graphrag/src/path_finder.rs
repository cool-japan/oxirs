//! # Path Finder for Graph-RAG
//!
//! Graph path-finding algorithms optimised for knowledge graph retrieval:
//!
//! - **BFS paths** — all paths between two entity nodes up to a depth limit.
//! - **DFS paths** — depth-first enumeration with cycle detection.
//! - **Shortest path** — minimum hop count (BFS-based).
//! - **Predicate filtering** — only traverse edges whose predicate IRI is in an
//!   allow-list.
//! - **Path scoring** — weight edges by predicate relevance.
//! - **Path narrative** — convert a path into a human-readable sentence.
//! - **Multi-hop collection** — all paths up to depth N.
//! - **Cycle detection** — prevent infinite traversal on cyclic graphs.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_graphrag::path_finder::{PathFinder, KnowledgeEdge, PathFinderConfig};
//!
//! let edges = vec![
//!     KnowledgeEdge::new("Alice", "knows", "Bob"),
//!     KnowledgeEdge::new("Bob", "works_at", "ACME"),
//!     KnowledgeEdge::new("ACME", "located_in", "Berlin"),
//! ];
//!
//! let config = PathFinderConfig::default();
//! let finder = PathFinder::new(edges, config);
//!
//! let paths = finder.bfs_paths("Alice", "ACME", 3);
//! assert!(!paths.is_empty());
//! assert_eq!(paths[0].nodes[0], "Alice");
//! assert_eq!(*paths[0].nodes.last().unwrap(), "ACME");
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Private type aliases
// ─────────────────────────────────────────────────────────────────────────────

/// BFS queue entry: (current_node, path_nodes, path_predicates, visited_set).
type BfsQueueEntry = (String, Vec<String>, Vec<String>, HashSet<String>);

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A directed edge in the knowledge graph.
#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeEdge {
    /// Subject / source node IRI.
    pub subject: String,
    /// Predicate / relation IRI.
    pub predicate: String,
    /// Object / target node IRI.
    pub object: String,
}

impl KnowledgeEdge {
    /// Create a new edge.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// A path through the knowledge graph represented as an ordered list of nodes
/// and the edges traversed.
#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgePath {
    /// Ordered node sequence (subject → ... → object).
    pub nodes: Vec<String>,
    /// Predicates traversed at each step (length = `nodes.len() - 1`).
    pub predicates: Vec<String>,
    /// Number of hops.
    pub hop_count: usize,
    /// Aggregate path score (sum of edge weights).
    pub score: f64,
}

impl KnowledgePath {
    /// Build a human-readable narrative of the path.
    ///
    /// Example: `"Alice —[knows]→ Bob —[works_at]→ ACME"`
    pub fn narrative(&self) -> String {
        if self.nodes.is_empty() {
            return String::new();
        }
        let mut parts = vec![self.nodes[0].clone()];
        for (pred, node) in self.predicates.iter().zip(self.nodes.iter().skip(1)) {
            parts.push(format!("—[{pred}]→ {node}"));
        }
        parts.join(" ")
    }
}

/// Configuration for the [`PathFinder`].
#[derive(Debug, Clone)]
pub struct PathFinderConfig {
    /// Maximum hop depth for BFS / DFS searches.
    pub max_depth: usize,
    /// Maximum number of paths returned per query.
    pub max_paths: usize,
    /// Optional set of predicate IRIs to traverse.  When `None`, all
    /// predicates are traversed.
    pub allowed_predicates: Option<HashSet<String>>,
    /// Predicate relevance weights for path scoring.  Unlisted predicates
    /// receive weight `1.0`.
    pub predicate_weights: HashMap<String, f64>,
}

impl Default for PathFinderConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            max_paths: 50,
            allowed_predicates: None,
            predicate_weights: HashMap::new(),
        }
    }
}

/// Statistics produced by a path-finding run.
#[derive(Debug, Clone)]
pub struct PathFinderStats {
    /// Number of paths found.
    pub paths_found: usize,
    /// Minimum hop count across all paths.
    pub min_hops: Option<usize>,
    /// Maximum hop count across all paths.
    pub max_hops: Option<usize>,
    /// Nodes visited during the search.
    pub nodes_visited: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// PathFinder
// ─────────────────────────────────────────────────────────────────────────────

/// Knowledge-graph path finder.
pub struct PathFinder {
    /// Adjacency list: source → list of (predicate, target).
    adj: HashMap<String, Vec<(String, String)>>,
    config: PathFinderConfig,
}

impl PathFinder {
    /// Build a path finder from a set of edges.
    pub fn new(edges: Vec<KnowledgeEdge>, config: PathFinderConfig) -> Self {
        let mut adj: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for edge in edges {
            adj.entry(edge.subject.clone())
                .or_default()
                .push((edge.predicate.clone(), edge.object.clone()));
        }
        Self { adj, config }
    }

    /// Add a single edge to the graph.
    pub fn add_edge(&mut self, edge: KnowledgeEdge) {
        self.adj
            .entry(edge.subject)
            .or_default()
            .push((edge.predicate, edge.object));
    }

    /// Find all paths from `source` to `target` using **BFS** up to
    /// `max_depth` hops.
    ///
    /// Returns paths sorted by score (descending).
    pub fn bfs_paths(&self, source: &str, target: &str, max_depth: usize) -> Vec<KnowledgePath> {
        let depth = max_depth.min(self.config.max_depth);
        let mut results = Vec::new();

        // BFS queue: (current_node, path_nodes, path_predicates, visited_set)
        let mut queue: VecDeque<BfsQueueEntry> = VecDeque::new();
        let mut start_visited = HashSet::new();
        start_visited.insert(source.to_string());
        queue.push_back((
            source.to_string(),
            vec![source.to_string()],
            Vec::new(),
            start_visited,
        ));

        while let Some((current, nodes, preds, visited)) = queue.pop_front() {
            if nodes.len() > depth {
                continue;
            }
            if let Some(neighbours) = self.adj.get(&current) {
                for (pred, next) in neighbours {
                    if !self.is_allowed(pred) {
                        continue;
                    }
                    if visited.contains(next.as_str()) {
                        continue; // cycle detection
                    }
                    let mut new_nodes = nodes.clone();
                    let mut new_preds = preds.clone();
                    let mut new_visited = visited.clone();
                    new_nodes.push(next.clone());
                    new_preds.push(pred.clone());
                    new_visited.insert(next.clone());

                    if next == target {
                        let score = self.score_path(&new_preds);
                        results.push(KnowledgePath {
                            hop_count: new_preds.len(),
                            nodes: new_nodes,
                            predicates: new_preds,
                            score,
                        });
                        if results.len() >= self.config.max_paths {
                            results.sort_by(|a, b| {
                                b.score
                                    .partial_cmp(&a.score)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            return results;
                        }
                    } else {
                        queue.push_back((next.clone(), new_nodes, new_preds, new_visited));
                    }
                }
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Find all paths from `source` to `target` using **DFS** up to
    /// `max_depth` hops.
    pub fn dfs_paths(&self, source: &str, target: &str, max_depth: usize) -> Vec<KnowledgePath> {
        let depth = max_depth.min(self.config.max_depth);
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(source.to_string());
        self.dfs_recursive(
            source,
            target,
            &mut vec![source.to_string()],
            &mut Vec::new(),
            &mut visited,
            depth,
            &mut results,
        );
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Find the **shortest path** (minimum hops) from `source` to `target`.
    ///
    /// Returns `None` when no path exists within `max_depth`.
    pub fn shortest_path(&self, source: &str, target: &str) -> Option<KnowledgePath> {
        // BFS naturally finds the shortest path first
        let paths = self.bfs_paths(source, target, self.config.max_depth);
        paths.into_iter().min_by_key(|p| p.hop_count)
    }

    /// Collect **all paths** up to depth N (multi-hop collection).
    ///
    /// Returns paths to all reachable nodes that are different from `source`.
    /// Useful for building context windows.
    pub fn multi_hop_paths(&self, source: &str, max_depth: usize) -> Vec<KnowledgePath> {
        let depth = max_depth.min(self.config.max_depth);
        let mut results = Vec::new();

        // Collect all nodes reachable from source
        let reachable: HashSet<String> = self
            .adj
            .values()
            .flat_map(|nbrs| nbrs.iter().map(|(_, t)| t.clone()))
            .chain(self.adj.keys().cloned())
            .collect();

        for target in &reachable {
            if target == source {
                continue;
            }
            let mut sub_paths = self.bfs_paths(source, target, depth);
            results.append(&mut sub_paths);
            if results.len() >= self.config.max_paths {
                break;
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.config.max_paths);
        results
    }

    /// Compute search statistics for the paths returned by a previous search.
    pub fn stats(&self, paths: &[KnowledgePath]) -> PathFinderStats {
        let nodes_visited: HashSet<&str> = paths
            .iter()
            .flat_map(|p| p.nodes.iter().map(String::as_str))
            .collect();
        PathFinderStats {
            paths_found: paths.len(),
            min_hops: paths.iter().map(|p| p.hop_count).min(),
            max_hops: paths.iter().map(|p| p.hop_count).max(),
            nodes_visited: nodes_visited.len(),
        }
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        let subjects: HashSet<&str> = self.adj.keys().map(String::as_str).collect();
        let objects: HashSet<&str> = self
            .adj
            .values()
            .flat_map(|nbrs| nbrs.iter().map(|(_, t)| t.as_str()))
            .collect();
        subjects.union(&objects).count()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.adj.values().map(|nbrs| nbrs.len()).sum()
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn dfs_recursive(
        &self,
        current: &str,
        target: &str,
        nodes: &mut Vec<String>,
        predicates: &mut Vec<String>,
        visited: &mut HashSet<String>,
        remaining_depth: usize,
        results: &mut Vec<KnowledgePath>,
    ) {
        if remaining_depth == 0 {
            return;
        }
        if let Some(neighbours) = self.adj.get(current) {
            for (pred, next) in neighbours {
                if !self.is_allowed(pred) {
                    continue;
                }
                if visited.contains(next.as_str()) {
                    continue; // cycle detection
                }
                nodes.push(next.clone());
                predicates.push(pred.clone());
                visited.insert(next.clone());

                if next == target {
                    let score = self.score_path(predicates);
                    results.push(KnowledgePath {
                        hop_count: predicates.len(),
                        nodes: nodes.clone(),
                        predicates: predicates.clone(),
                        score,
                    });
                } else {
                    self.dfs_recursive(
                        next,
                        target,
                        nodes,
                        predicates,
                        visited,
                        remaining_depth - 1,
                        results,
                    );
                }

                nodes.pop();
                predicates.pop();
                visited.remove(next.as_str());

                if results.len() >= self.config.max_paths {
                    return;
                }
            }
        }
    }

    /// Return `true` if the predicate is allowed by the configuration.
    fn is_allowed(&self, predicate: &str) -> bool {
        match &self.config.allowed_predicates {
            None => true,
            Some(allowed) => allowed.contains(predicate),
        }
    }

    /// Compute the aggregate score for a path (sum of edge weights).
    fn score_path(&self, predicates: &[String]) -> f64 {
        predicates
            .iter()
            .map(|p| {
                *self
                    .config
                    .predicate_weights
                    .get(p.as_str())
                    .unwrap_or(&1.0)
            })
            .sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test helpers ─────────────────────────────────────────────────────────

    fn simple_edges() -> Vec<KnowledgeEdge> {
        vec![
            KnowledgeEdge::new("Alice", "knows", "Bob"),
            KnowledgeEdge::new("Bob", "works_at", "ACME"),
            KnowledgeEdge::new("ACME", "located_in", "Berlin"),
            KnowledgeEdge::new("Alice", "lives_in", "Berlin"),
            KnowledgeEdge::new("Bob", "knows", "Carol"),
        ]
    }

    fn default_finder() -> PathFinder {
        PathFinder::new(simple_edges(), PathFinderConfig::default())
    }

    // ── BFS paths ────────────────────────────────────────────────────────────

    #[test]
    fn test_bfs_direct_connection() {
        let finder = default_finder();
        let paths = finder.bfs_paths("Alice", "Bob", 1);
        assert!(!paths.is_empty());
        assert_eq!(paths[0].hop_count, 1);
        assert_eq!(paths[0].nodes[0], "Alice");
        assert_eq!(paths[0].nodes[1], "Bob");
    }

    #[test]
    fn test_bfs_two_hop_path() {
        let finder = default_finder();
        let paths = finder.bfs_paths("Alice", "ACME", 2);
        assert!(!paths.is_empty());
        assert!(paths.iter().any(|p| p.hop_count == 2));
    }

    #[test]
    fn test_bfs_no_path_beyond_depth() {
        let finder = default_finder();
        // Berlin is 3 hops from Alice via ACME, but we limit to 1
        let paths = finder.bfs_paths("Alice", "Berlin", 1);
        // Direct Alice→Berlin (lives_in) has 1 hop — should be found
        assert!(!paths.is_empty());
        // But Alice→Bob→ACME→Berlin requires 3 hops — not returned
        assert!(paths.iter().all(|p| p.hop_count <= 1));
    }

    #[test]
    fn test_bfs_no_such_path_returns_empty() {
        let finder = default_finder();
        let paths = finder.bfs_paths("Berlin", "Alice", 5);
        // Graph is directed — Berlin has no outgoing edges
        assert!(paths.is_empty());
    }

    #[test]
    fn test_bfs_paths_sorted_by_score_desc() {
        let finder = default_finder();
        let paths = finder.bfs_paths("Alice", "Berlin", 4);
        for w in paths.windows(2) {
            assert!(w[0].score >= w[1].score - 1e-10);
        }
    }

    #[test]
    fn test_bfs_respects_max_paths() {
        let config = PathFinderConfig {
            max_paths: 1,
            ..Default::default()
        };
        let finder = PathFinder::new(simple_edges(), config);
        let paths = finder.bfs_paths("Alice", "Berlin", 4);
        assert!(paths.len() <= 1);
    }

    // ── DFS paths ────────────────────────────────────────────────────────────

    #[test]
    fn test_dfs_finds_paths() {
        let finder = default_finder();
        let paths = finder.dfs_paths("Alice", "Bob", 2);
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_dfs_hop_count_within_depth() {
        let finder = default_finder();
        let paths = finder.dfs_paths("Alice", "ACME", 3);
        for p in &paths {
            assert!(p.hop_count <= 3);
        }
    }

    #[test]
    fn test_dfs_no_cycles_in_paths() {
        let finder = default_finder();
        let paths = finder.dfs_paths("Alice", "Berlin", 5);
        for p in &paths {
            let mut seen = HashSet::new();
            for node in &p.nodes {
                assert!(seen.insert(node), "cycle detected in path: {node}");
            }
        }
    }

    #[test]
    fn test_dfs_no_path_returns_empty() {
        let finder = default_finder();
        let paths = finder.dfs_paths("Berlin", "Alice", 5);
        assert!(paths.is_empty());
    }

    // ── Shortest path ─────────────────────────────────────────────────────────

    #[test]
    fn test_shortest_path_direct() {
        let finder = default_finder();
        let path = finder.shortest_path("Alice", "Bob").unwrap();
        assert_eq!(path.hop_count, 1);
    }

    #[test]
    fn test_shortest_path_two_hops() {
        let finder = default_finder();
        let path = finder.shortest_path("Alice", "ACME").unwrap();
        assert_eq!(path.hop_count, 2);
    }

    #[test]
    fn test_shortest_path_no_path_is_none() {
        let finder = default_finder();
        let path = finder.shortest_path("Berlin", "Alice");
        assert!(path.is_none());
    }

    #[test]
    fn test_shortest_path_self_loop_check() {
        let finder = default_finder();
        // source == target: BFS won't find it since we start with source visited
        let path = finder.shortest_path("Alice", "Alice");
        assert!(path.is_none());
    }

    // ── Predicate filtering ───────────────────────────────────────────────────

    #[test]
    fn test_predicate_filter_restricts_traversal() {
        let mut allowed = HashSet::new();
        allowed.insert("knows".to_string());
        let config = PathFinderConfig {
            allowed_predicates: Some(allowed),
            ..Default::default()
        };
        let finder = PathFinder::new(simple_edges(), config);
        // Alice→Bob (knows) is allowed; Alice→ACME (works_at) is not
        let paths = finder.bfs_paths("Alice", "ACME", 3);
        // ACME can only be reached via works_at — filtered out → no paths
        assert!(paths.is_empty());
    }

    #[test]
    fn test_predicate_filter_allows_direct_path() {
        let mut allowed = HashSet::new();
        allowed.insert("knows".to_string());
        allowed.insert("works_at".to_string());
        let config = PathFinderConfig {
            allowed_predicates: Some(allowed),
            ..Default::default()
        };
        let finder = PathFinder::new(simple_edges(), config);
        let paths = finder.bfs_paths("Alice", "ACME", 3);
        assert!(!paths.is_empty());
    }

    // ── Path scoring ──────────────────────────────────────────────────────────

    #[test]
    fn test_path_scoring_uses_weights() {
        let mut weights = HashMap::new();
        weights.insert("knows".to_string(), 2.0);
        weights.insert("works_at".to_string(), 1.0);
        let config = PathFinderConfig {
            predicate_weights: weights,
            ..Default::default()
        };
        let finder = PathFinder::new(simple_edges(), config);
        let paths = finder.bfs_paths("Alice", "ACME", 3);
        // Path Alice→Bob(knows,w=2)→ACME(works_at,w=1) → score=3
        assert!(!paths.is_empty());
        let best = &paths[0];
        assert!((best.score - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_path_scoring_default_weight_one() {
        let finder = default_finder();
        let paths = finder.bfs_paths("Alice", "Bob", 1);
        assert!(!paths.is_empty());
        // 1 hop, default weight 1.0 → score = 1.0
        assert!((paths[0].score - 1.0).abs() < 1e-10);
    }

    // ── Narrative ─────────────────────────────────────────────────────────────

    #[test]
    fn test_narrative_single_hop() {
        let path = KnowledgePath {
            nodes: vec!["Alice".into(), "Bob".into()],
            predicates: vec!["knows".into()],
            hop_count: 1,
            score: 1.0,
        };
        let narrative = path.narrative();
        assert!(narrative.contains("Alice"));
        assert!(narrative.contains("knows"));
        assert!(narrative.contains("Bob"));
    }

    #[test]
    fn test_narrative_multi_hop() {
        let path = KnowledgePath {
            nodes: vec!["Alice".into(), "Bob".into(), "ACME".into()],
            predicates: vec!["knows".into(), "works_at".into()],
            hop_count: 2,
            score: 2.0,
        };
        let narrative = path.narrative();
        assert!(narrative.contains("—[knows]→"));
        assert!(narrative.contains("—[works_at]→"));
    }

    #[test]
    fn test_narrative_empty_path() {
        let path = KnowledgePath {
            nodes: vec![],
            predicates: vec![],
            hop_count: 0,
            score: 0.0,
        };
        assert_eq!(path.narrative(), "");
    }

    // ── Multi-hop collection ──────────────────────────────────────────────────

    #[test]
    fn test_multi_hop_returns_paths() {
        let finder = default_finder();
        let paths = finder.multi_hop_paths("Alice", 3);
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_multi_hop_all_start_at_source() {
        let finder = default_finder();
        let paths = finder.multi_hop_paths("Alice", 2);
        for p in &paths {
            assert_eq!(p.nodes[0], "Alice");
        }
    }

    #[test]
    fn test_multi_hop_respects_depth() {
        let finder = default_finder();
        let paths = finder.multi_hop_paths("Alice", 1);
        for p in &paths {
            assert!(p.hop_count <= 1);
        }
    }

    // ── Cycle detection ───────────────────────────────────────────────────────

    #[test]
    fn test_bfs_no_cycles_even_in_cyclic_graph() {
        let edges = vec![
            KnowledgeEdge::new("A", "rel", "B"),
            KnowledgeEdge::new("B", "rel", "C"),
            KnowledgeEdge::new("C", "rel", "A"), // cycle
            KnowledgeEdge::new("B", "rel", "D"),
        ];
        let finder = PathFinder::new(edges, PathFinderConfig::default());
        let paths = finder.bfs_paths("A", "D", 4);
        // Should find A→B→D without looping
        assert!(!paths.is_empty());
        for p in &paths {
            let mut seen = HashSet::new();
            for node in &p.nodes {
                assert!(seen.insert(node));
            }
        }
    }

    #[test]
    fn test_dfs_no_cycles_in_cyclic_graph() {
        let edges = vec![
            KnowledgeEdge::new("X", "p", "Y"),
            KnowledgeEdge::new("Y", "p", "Z"),
            KnowledgeEdge::new("Z", "p", "X"), // cycle
        ];
        let finder = PathFinder::new(edges, PathFinderConfig::default());
        let paths = finder.dfs_paths("X", "Z", 5);
        for p in &paths {
            let mut seen = HashSet::new();
            for node in &p.nodes {
                assert!(seen.insert(node));
            }
        }
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_paths() {
        let finder = default_finder();
        let stats = finder.stats(&[]);
        assert_eq!(stats.paths_found, 0);
        assert!(stats.min_hops.is_none());
        assert!(stats.max_hops.is_none());
    }

    #[test]
    fn test_stats_populated() {
        let finder = default_finder();
        let paths = finder.bfs_paths("Alice", "Berlin", 4);
        let stats = finder.stats(&paths);
        assert_eq!(stats.paths_found, paths.len());
        assert!(stats.nodes_visited > 0);
    }

    #[test]
    fn test_stats_min_max_hops() {
        let finder = default_finder();
        let paths = finder.bfs_paths("Alice", "Berlin", 4);
        if !paths.is_empty() {
            let stats = finder.stats(&paths);
            assert!(stats.min_hops.unwrap() <= stats.max_hops.unwrap());
        }
    }

    // ── Graph mutation ─────────────────────────────────────────────────────────

    #[test]
    fn test_add_edge_extends_graph() {
        let mut finder = default_finder();
        let before = finder.edge_count();
        finder.add_edge(KnowledgeEdge::new("Carol", "works_at", "GlobalCo"));
        assert_eq!(finder.edge_count(), before + 1);
    }

    #[test]
    fn test_node_count() {
        let finder = default_finder();
        // Alice, Bob, ACME, Berlin, Carol
        assert!(finder.node_count() >= 5);
    }

    #[test]
    fn test_edge_count() {
        let finder = default_finder();
        assert_eq!(finder.edge_count(), simple_edges().len());
    }

    // ── KnowledgeEdge ─────────────────────────────────────────────────────────

    #[test]
    fn test_edge_fields() {
        let edge = KnowledgeEdge::new("S", "p", "O");
        assert_eq!(edge.subject, "S");
        assert_eq!(edge.predicate, "p");
        assert_eq!(edge.object, "O");
    }

    #[test]
    fn test_edge_clone() {
        let edge = KnowledgeEdge::new("A", "b", "C");
        let cloned = edge.clone();
        assert_eq!(cloned, edge);
    }

    // ── PathFinderConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = PathFinderConfig::default();
        assert_eq!(cfg.max_depth, 4);
        assert_eq!(cfg.max_paths, 50);
        assert!(cfg.allowed_predicates.is_none());
        assert!(cfg.predicate_weights.is_empty());
    }

    // ── KnowledgePath ─────────────────────────────────────────────────────────

    #[test]
    fn test_knowledge_path_fields() {
        let p = KnowledgePath {
            nodes: vec!["A".into(), "B".into()],
            predicates: vec!["rel".into()],
            hop_count: 1,
            score: 2.5,
        };
        assert_eq!(p.hop_count, 1);
        assert!((p.score - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_bfs_and_dfs_agree_on_shortest_path() {
        let finder = default_finder();
        let bfs = finder.bfs_paths("Alice", "Bob", 3);
        let dfs = finder.dfs_paths("Alice", "Bob", 3);
        // Both should find the direct 1-hop path
        assert!(bfs.iter().any(|p| p.hop_count == 1));
        assert!(dfs.iter().any(|p| p.hop_count == 1));
    }

    #[test]
    fn test_multi_hop_no_self_paths() {
        let finder = default_finder();
        let paths = finder.multi_hop_paths("Alice", 3);
        for p in &paths {
            assert_ne!(p.nodes.first(), p.nodes.last().or(Some(&String::new())));
            assert!(p.hop_count > 0);
        }
    }
}
