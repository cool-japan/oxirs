//! Centrality algorithms: PageRank, degree centrality, betweenness centrality

use super::adapter::{NodeId, RdfGraphAdapter};
use std::collections::{HashMap, VecDeque};

// ── PageRank ──────────────────────────────────────────────────────────────────

/// PageRank algorithm for RDF graphs.
///
/// Uses the power-iteration method with dangling-node handling.
pub struct PageRank {
    /// Damping factor (default `0.85`)
    pub damping: f64,
    /// Maximum number of power iterations (default `100`)
    pub max_iter: usize,
    /// Convergence tolerance – stops when the L1 change falls below this
    /// (default `1e-6`)
    pub tolerance: f64,
}

impl Default for PageRank {
    fn default() -> Self {
        Self::new()
    }
}

impl PageRank {
    /// Create with default parameters.
    pub fn new() -> Self {
        Self {
            damping: 0.85,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }

    /// Override the damping factor.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Override the maximum number of iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Compute PageRank scores for all nodes.
    ///
    /// Returns a map from `NodeId` to score.  Scores sum to approximately 1.
    pub fn compute(&self, graph: &RdfGraphAdapter) -> HashMap<NodeId, f64> {
        let n = graph.node_count();
        if n == 0 {
            return HashMap::new();
        }

        let initial = 1.0 / n as f64;
        let mut scores: Vec<f64> = vec![initial; n];
        let mut new_scores: Vec<f64> = vec![0.0; n];

        // Pre-compute out-degree for each node
        let out_degree: Vec<f64> = (0..n)
            .map(|u| graph.adjacency[u].iter().map(|(_, w)| w).sum::<f64>())
            .collect();

        for _ in 0..self.max_iter {
            // Dangling-node mass: sum of scores of nodes with out_degree == 0
            let dangling_sum: f64 = (0..n)
                .filter(|&u| out_degree[u] == 0.0)
                .map(|u| scores[u])
                .sum();

            let dangling_contrib = self.damping * dangling_sum / n as f64;
            let base = (1.0 - self.damping) / n as f64 + dangling_contrib;

            for score in new_scores.iter_mut().take(n) {
                *score = base;
            }

            // Distribute score along edges
            for u in 0..n {
                if out_degree[u] > 0.0 {
                    let contrib = self.damping * scores[u] / out_degree[u];
                    for &(v, w) in &graph.adjacency[u] {
                        new_scores[v] += contrib * w;
                    }
                }
            }

            // Check convergence (L1 norm of change)
            let delta: f64 = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            std::mem::swap(&mut scores, &mut new_scores);

            if delta < self.tolerance {
                break;
            }
        }

        scores.into_iter().enumerate().collect()
    }

    /// Return the top-`k` nodes sorted by descending PageRank score.
    pub fn top_k(&self, graph: &RdfGraphAdapter, k: usize) -> Vec<(NodeId, f64)> {
        let mut entries: Vec<(NodeId, f64)> = self.compute(graph).into_iter().collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(k);
        entries
    }
}

// ── Degree Centrality ─────────────────────────────────────────────────────────

/// Normalised degree centrality metrics.
pub struct DegreeCentrality;

impl DegreeCentrality {
    /// Normalised in-degree centrality.
    ///
    /// For each node, the score is `in_degree / (n - 1)` (or `0` when `n <= 1`).
    pub fn in_degree(graph: &RdfGraphAdapter) -> HashMap<NodeId, f64> {
        let n = graph.node_count();
        let norm = if n > 1 { (n - 1) as f64 } else { 1.0 };
        (0..n)
            .map(|v| (v, graph.reverse_adjacency[v].len() as f64 / norm))
            .collect()
    }

    /// Normalised out-degree centrality.
    pub fn out_degree(graph: &RdfGraphAdapter) -> HashMap<NodeId, f64> {
        let n = graph.node_count();
        let norm = if n > 1 { (n - 1) as f64 } else { 1.0 };
        (0..n)
            .map(|u| (u, graph.adjacency[u].len() as f64 / norm))
            .collect()
    }

    /// Normalised total-degree centrality (in + out, self-loops counted once).
    pub fn total_degree(graph: &RdfGraphAdapter) -> HashMap<NodeId, f64> {
        let n = graph.node_count();
        let norm = if n > 1 { 2.0 * (n - 1) as f64 } else { 1.0 };
        (0..n)
            .map(|u| {
                let d = graph.adjacency[u].len() + graph.reverse_adjacency[u].len();
                (u, d as f64 / norm)
            })
            .collect()
    }
}

// ── Betweenness Centrality ────────────────────────────────────────────────────

/// Betweenness centrality via Brandes' algorithm.
///
/// Treats the graph as **unweighted** and **directed**.
pub struct BetweennessCentrality {
    /// When `true` (default), scores are divided by `(n-1)(n-2)` (directed
    /// normalisation).
    pub normalized: bool,
}

impl Default for BetweennessCentrality {
    fn default() -> Self {
        Self::new()
    }
}

impl BetweennessCentrality {
    /// Create with normalisation enabled.
    pub fn new() -> Self {
        Self { normalized: true }
    }

    /// Compute betweenness centrality for all nodes using Brandes' O(VE) BFS.
    pub fn compute(&self, graph: &RdfGraphAdapter) -> HashMap<NodeId, f64> {
        let n = graph.node_count();
        let mut cb = vec![0.0f64; n];

        for s in 0..n {
            // ── BFS from source s ──────────────────────────────────────────
            let mut stack: Vec<NodeId> = Vec::with_capacity(n);
            let mut pred: Vec<Vec<NodeId>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0f64; n]; // number of shortest paths from s
            sigma[s] = 1.0;
            let mut dist = vec![-1i64; n];
            dist[s] = 0;

            let mut queue: VecDeque<NodeId> = VecDeque::with_capacity(n);
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                for &(w, _wgt) in &graph.adjacency[v] {
                    // First visit?
                    if dist[w] < 0 {
                        queue.push_back(w);
                        dist[w] = dist[v] + 1;
                    }
                    // Shortest path via v?
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        pred[w].push(v);
                    }
                }
            }

            // ── Back-propagation ───────────────────────────────────────────
            let mut delta = vec![0.0f64; n];
            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    if sigma[w] != 0.0 {
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                    }
                }
                if w != s {
                    cb[w] += delta[w];
                }
            }
        }

        // Normalise
        let norm = if self.normalized && n > 2 {
            1.0 / ((n - 1) as f64 * (n - 2) as f64)
        } else {
            1.0
        };

        cb.into_iter()
            .enumerate()
            .map(|(u, v)| (u, v * norm))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_graph(edges: &[(&str, &str)]) -> RdfGraphAdapter {
        let triples: Vec<(String, String, String)> = edges
            .iter()
            .map(|(s, o)| (s.to_string(), "ex:rel".to_string(), o.to_string()))
            .collect();
        RdfGraphAdapter::from_triples(&triples)
    }

    // ── PageRank tests ────────────────────────────────────────────────────

    #[test]
    fn test_pagerank_single_node() {
        // Single node with a self-loop treated as dangling
        let triples = vec![(
            "ex:A".to_string(),
            "ex:rel".to_string(),
            "\"lit\"".to_string(),
        )];
        let g = RdfGraphAdapter::from_triples(&triples);
        let pr = PageRank::new().compute(&g);
        assert!((pr[&0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_pagerank_scores_sum_to_one() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:C", "ex:A")]);
        let pr = PageRank::new().compute(&g);
        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-4, "sum={total}");
    }

    #[test]
    fn test_pagerank_dangling_node() {
        // ex:B is dangling (no outgoing edges)
        let g = build_graph(&[("ex:A", "ex:B")]);
        let pr = PageRank::new().compute(&g);
        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-4, "sum={total}");
    }

    #[test]
    fn test_pagerank_symmetric_ring() {
        // All nodes in a symmetric ring should have equal PageRank
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:C", "ex:A")]);
        let pr = PageRank::new().compute(&g);
        let scores: Vec<f64> = pr.values().copied().collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        for s in &scores {
            assert!(
                (s - mean).abs() < 1e-4,
                "unequal scores in ring: {s} vs mean {mean}"
            );
        }
    }

    #[test]
    fn test_pagerank_star_center_highest() {
        // Hub node should have higher PR than spokes
        let g = build_graph(&[
            ("ex:S1", "ex:Hub"),
            ("ex:S2", "ex:Hub"),
            ("ex:S3", "ex:Hub"),
            ("ex:Hub", "ex:S1"),
        ]);
        let pr = PageRank::new().compute(&g);
        let hub_id = g.get_node_id("ex:Hub").unwrap();
        let hub_pr = pr[&hub_id];
        let s2_id = g.get_node_id("ex:S2").unwrap();
        assert!(hub_pr > pr[&s2_id], "hub PR should dominate");
    }

    #[test]
    fn test_pagerank_with_custom_damping() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:C", "ex:A")]);
        let pr = PageRank::new().with_damping(0.5).compute(&g);
        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_pagerank_top_k() {
        let g = build_graph(&[
            ("ex:A", "ex:B"),
            ("ex:B", "ex:C"),
            ("ex:C", "ex:A"),
            ("ex:D", "ex:A"),
        ]);
        let top2 = PageRank::new().top_k(&g, 2);
        assert_eq!(top2.len(), 2);
        // Results must be in descending order
        assert!(top2[0].1 >= top2[1].1);
    }

    #[test]
    fn test_pagerank_top_k_larger_than_nodes() {
        let g = build_graph(&[("ex:A", "ex:B")]);
        let top = PageRank::new().top_k(&g, 100);
        assert_eq!(top.len(), g.node_count());
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let g = RdfGraphAdapter::from_triples(&[]);
        let pr = PageRank::new().compute(&g);
        assert!(pr.is_empty());
    }

    // ── Degree Centrality tests ───────────────────────────────────────────

    #[test]
    fn test_in_degree_centrality() {
        // A→B, C→B  –  B has in-degree 2, others 0
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:C", "ex:B")]);
        let dc = DegreeCentrality::in_degree(&g);
        let b_id = g.get_node_id("ex:B").unwrap();
        let a_id = g.get_node_id("ex:A").unwrap();
        assert!(dc[&b_id] > dc[&a_id]);
    }

    #[test]
    fn test_out_degree_centrality() {
        // A→B, A→C  –  A has out-degree 2, others 0
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:A", "ex:C")]);
        let dc = DegreeCentrality::out_degree(&g);
        let a_id = g.get_node_id("ex:A").unwrap();
        let b_id = g.get_node_id("ex:B").unwrap();
        assert!(dc[&a_id] > dc[&b_id]);
    }

    #[test]
    fn test_total_degree_centrality() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let dc = DegreeCentrality::total_degree(&g);
        // All scores should be in [0, 1]
        for &v in dc.values() {
            assert!((0.0_f64..=1.0).contains(&v), "score {v} out of range");
        }
    }

    #[test]
    fn test_degree_single_node_no_panic() {
        let triples = vec![(
            "ex:A".to_string(),
            "ex:rel".to_string(),
            "\"lit\"".to_string(),
        )];
        let g = RdfGraphAdapter::from_triples(&triples);
        let _dc = DegreeCentrality::in_degree(&g);
        let _dc = DegreeCentrality::out_degree(&g);
        let _dc = DegreeCentrality::total_degree(&g);
    }

    // ── Betweenness Centrality tests ──────────────────────────────────────

    #[test]
    fn test_betweenness_linear_graph() {
        // A → B → C  –  only B is on all paths A→C, so B has highest betweenness
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let bc = BetweennessCentrality::new().compute(&g);
        let a_id = g.get_node_id("ex:A").unwrap();
        let b_id = g.get_node_id("ex:B").unwrap();
        let c_id = g.get_node_id("ex:C").unwrap();
        assert!(bc[&b_id] > bc[&a_id]);
        assert!(bc[&b_id] > bc[&c_id]);
    }

    #[test]
    fn test_betweenness_star_graph() {
        // Hub connects all spokes – hub should have highest betweenness
        let g = build_graph(&[
            ("ex:Hub", "ex:S1"),
            ("ex:Hub", "ex:S2"),
            ("ex:Hub", "ex:S3"),
            ("ex:S1", "ex:Hub"),
            ("ex:S2", "ex:Hub"),
            ("ex:S3", "ex:Hub"),
        ]);
        let bc = BetweennessCentrality::new().compute(&g);
        let hub_id = g.get_node_id("ex:Hub").unwrap();
        let s1_id = g.get_node_id("ex:S1").unwrap();
        assert!(
            bc[&hub_id] >= bc[&s1_id],
            "hub should have highest betweenness"
        );
    }

    #[test]
    fn test_betweenness_complete_graph_equal() {
        // In a complete directed graph all nodes have equal betweenness
        let g = build_graph(&[
            ("ex:A", "ex:B"),
            ("ex:A", "ex:C"),
            ("ex:B", "ex:A"),
            ("ex:B", "ex:C"),
            ("ex:C", "ex:A"),
            ("ex:C", "ex:B"),
        ]);
        let bc = BetweennessCentrality::new().compute(&g);
        let vals: Vec<f64> = bc.values().copied().collect();
        let first = vals[0];
        for v in &vals {
            assert!((v - first).abs() < 1e-9, "not equal: {v} vs {first}");
        }
    }

    #[test]
    fn test_betweenness_unnormalized() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let bc = BetweennessCentrality { normalized: false }.compute(&g);
        let b_id = g.get_node_id("ex:B").unwrap();
        // Unnormalized: B appears on exactly 1 shortest path A→C
        assert!(bc[&b_id] > 0.0);
    }
}
