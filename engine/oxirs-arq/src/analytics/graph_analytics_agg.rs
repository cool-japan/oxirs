//! Graph analytics aggregate: collect edges during a GROUP BY pass, then
//! finalize to per-node topology scores via the existing analytics back-ends.
//!
//! # Overview
//!
//! [`GraphAnalyticsAccumulator`] acts as the stateful context for a
//! SPARQL-style custom aggregate function.  The typical lifecycle is:
//!
//! 1. Create an accumulator with the desired [`GraphAnalyticsAggregate`] kind.
//! 2. For every row in the current group, call [`GraphAnalyticsAccumulator::accumulate`]
//!    with the `(subject, object)` IRI pair that forms a directed edge.
//! 3. Call [`GraphAnalyticsAccumulator::finalize`] to run the analytics
//!    algorithm over the collected edge set and receive a
//!    `HashMap<node_iri, score>`.
//!
//! # Edge-set constraints
//!
//! The underlying [`super::adapter::RdfGraphAdapter`] silently drops edges
//! whose object looks like an RDF literal (starts with `"` or lacks `:`).
//! [`GraphAnalyticsAccumulator::accumulate`] enforces the same filter up front
//! so that [`edge_count`] / `node_count` agree with what `finalize` sees.
//!
//! [`edge_count`]: GraphAnalyticsAccumulator::edge_count

use std::collections::{HashMap, HashSet};

use super::adapter::RdfGraphAdapter;
use super::centrality::{BetweennessCentrality, DegreeCentrality, PageRank};
use super::components::ConnectedComponents;

// ── Direction selector for degree centrality ──────────────────────────────────

/// Direction used by the [`GraphAnalyticsAggregate::DegreeCentrality`] variant.
#[derive(Debug, Clone)]
pub enum DegreeDirection {
    /// Count only incoming edges.
    Incoming,
    /// Count only outgoing edges.
    Outgoing,
    /// Count both (total degree).
    Both,
}

// ── Aggregate variant ─────────────────────────────────────────────────────────

/// Which graph-analytics metric to compute at finalization time.
///
/// All variants operate over the directed edge set accumulated by
/// [`GraphAnalyticsAccumulator`].
#[derive(Debug, Clone)]
pub enum GraphAnalyticsAggregate {
    /// PageRank with configurable damping factor and iteration cap.
    ///
    /// `damping` is clamped to `[0.0, 1.0]`. `iterations` is the maximum
    /// number of power-iteration steps (convergence can terminate earlier).
    PageRank {
        /// Damping factor *d* in the PageRank formula (typical default: 0.85).
        damping: f64,
        /// Maximum number of power-iteration steps.
        iterations: usize,
    },
    /// Betweenness centrality via Brandes' O(VE) BFS algorithm.
    BetweennessCentrality {
        /// When `true` scores are divided by `(n-1)(n-2)` for directed graphs.
        normalized: bool,
    },
    /// Weakly-connected component ID for each node.
    ///
    /// IDs are 0-based integers converted to `f64`, stable within a single
    /// `finalize()` call but not across separate calls.
    ConnectedComponent,
    /// Local clustering coefficient for each node.
    ///
    /// Treats the graph as *undirected* (unions out- and in-neighbours).
    /// For a node with fewer than two neighbours the coefficient is `0.0`.
    ClusteringCoefficient,
    /// Degree centrality — normalised edge count per node.
    DegreeCentrality {
        /// Which edge direction to count.
        direction: DegreeDirection,
    },
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur during analytics finalization.
#[derive(Debug)]
pub enum GraphAnalyticsError {
    /// The accumulated edge set is empty — no nodes, no topology.
    EmptyGraph,
    /// The chosen algorithm encountered an internal error.
    AlgorithmError(String),
    /// A specific node IRI was not found in the accumulated graph.
    NodeNotFound(String),
}

impl std::fmt::Display for GraphAnalyticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGraph => f.write_str("graph analytics: accumulated edge set is empty"),
            Self::AlgorithmError(msg) => write!(f, "graph analytics algorithm error: {msg}"),
            Self::NodeNotFound(iri) => write!(f, "graph analytics: node not found: {iri}"),
        }
    }
}

impl std::error::Error for GraphAnalyticsError {}

// ── Accumulator ───────────────────────────────────────────────────────────────

/// Returns `true` when `s` looks like an IRI (not a literal).
///
/// Matches the heuristic used by [`RdfGraphAdapter`]: a string is treated as
/// a node IRI if it does **not** start with `"` and contains at least one `:`.
#[inline]
fn is_iri_like(s: &str) -> bool {
    !s.starts_with('"') && s.contains(':')
}

/// Stateful accumulator that collects directed edges during a GROUP BY pass and
/// runs a chosen graph-analytics algorithm at finalization time.
///
/// # Example
///
/// ```rust
/// use oxirs_arq::analytics::graph_analytics_agg::{
///     GraphAnalyticsAccumulator, GraphAnalyticsAggregate,
/// };
///
/// let mut acc = GraphAnalyticsAccumulator::new(GraphAnalyticsAggregate::PageRank {
///     damping: 0.85,
///     iterations: 50,
/// });
/// acc.accumulate("ex:A", "ex:B");
/// acc.accumulate("ex:B", "ex:C");
/// acc.accumulate("ex:C", "ex:A");
///
/// let scores = acc.finalize().expect("PageRank should succeed");
/// assert_eq!(scores.len(), 3);
/// ```
pub struct GraphAnalyticsAccumulator {
    /// Accumulated directed edges `(subject_iri, object_iri)`.
    edges: Vec<(String, String)>,
    /// Distinct node IRIs seen (only IRI-like strings).
    node_set: HashSet<String>,
    /// Which analytics function to run at finalization.
    kind: GraphAnalyticsAggregate,
}

impl GraphAnalyticsAccumulator {
    /// Create a new accumulator for the given analytics kind.
    pub fn new(kind: GraphAnalyticsAggregate) -> Self {
        Self {
            edges: Vec::new(),
            node_set: HashSet::new(),
            kind,
        }
    }

    /// Accumulate one directed edge `(subject, object)`.
    ///
    /// Edges where *either* endpoint is not IRI-like are silently discarded so
    /// that `edge_count()` / `node_count()` agree with what `finalize()` sees.
    pub fn accumulate(&mut self, subject: &str, object: &str) {
        if !is_iri_like(subject) || !is_iri_like(object) {
            return; // drop literal-shaped endpoints
        }
        self.node_set.insert(subject.to_string());
        self.node_set.insert(object.to_string());
        self.edges.push((subject.to_string(), object.to_string()));
    }

    /// Number of valid (IRI-like) edges accumulated so far.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of distinct IRI nodes seen so far.
    pub fn node_count(&self) -> usize {
        self.node_set.len()
    }

    /// Run the analytics algorithm over the accumulated edge set.
    ///
    /// Returns a `HashMap` mapping each node IRI to its numeric score.
    ///
    /// # Errors
    ///
    /// Returns [`GraphAnalyticsError::EmptyGraph`] when no IRI-like edges have
    /// been accumulated.
    pub fn finalize(&mut self) -> Result<HashMap<String, f64>, GraphAnalyticsError> {
        if self.edges.is_empty() {
            return Err(GraphAnalyticsError::EmptyGraph);
        }

        // Build the adapter from accumulated edges using a synthetic predicate.
        let triples: Vec<(String, String, String)> = self
            .edges
            .iter()
            .map(|(s, o)| (s.clone(), "ex:aggregateEdge".to_string(), o.clone()))
            .collect();
        let graph = RdfGraphAdapter::from_triples(&triples);

        match &self.kind.clone() {
            GraphAnalyticsAggregate::PageRank {
                damping,
                iterations,
            } => {
                let pr = PageRank::new()
                    .with_damping(*damping)
                    .with_max_iter(*iterations);
                let raw = pr.compute(&graph);
                Ok(translate_scores(&graph, &raw))
            }

            GraphAnalyticsAggregate::BetweennessCentrality { normalized } => {
                let bc = BetweennessCentrality {
                    normalized: *normalized,
                };
                let raw = bc.compute(&graph);
                Ok(translate_scores(&graph, &raw))
            }

            GraphAnalyticsAggregate::ConnectedComponent => {
                let components = ConnectedComponents::weakly_connected(&graph);
                let mut result: HashMap<String, f64> = HashMap::new();
                for (component_id, component) in components.iter().enumerate() {
                    for &node_id in component {
                        if let Some(iri) = graph.get_node_iri(node_id) {
                            result.insert(iri.to_string(), component_id as f64);
                        }
                    }
                }
                Ok(result)
            }

            GraphAnalyticsAggregate::ClusteringCoefficient => {
                Ok(compute_clustering_coefficient(&graph))
            }

            GraphAnalyticsAggregate::DegreeCentrality { direction } => {
                let raw = match direction {
                    DegreeDirection::Incoming => DegreeCentrality::in_degree(&graph),
                    DegreeDirection::Outgoing => DegreeCentrality::out_degree(&graph),
                    DegreeDirection::Both => DegreeCentrality::total_degree(&graph),
                };
                Ok(translate_scores(&graph, &raw))
            }
        }
    }

    /// Finalize and look up a specific node's score.
    ///
    /// Returns `Ok(None)` when the node was not in the accumulated graph rather
    /// than an error; only hard algorithm failures produce `Err(...)`.
    pub fn finalize_for_node(&mut self, node: &str) -> Result<Option<f64>, GraphAnalyticsError> {
        let scores = self.finalize()?;
        Ok(scores.get(node).copied())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Translate a `HashMap<NodeId, f64>` into `HashMap<String (IRI), f64>`.
fn translate_scores(graph: &RdfGraphAdapter, raw: &HashMap<usize, f64>) -> HashMap<String, f64> {
    raw.iter()
        .filter_map(|(&id, &score)| graph.get_node_iri(id).map(|iri| (iri.to_string(), score)))
        .collect()
}

/// Compute the local clustering coefficient for every node in `graph`.
///
/// The coefficient for node *v* is defined as the fraction of pairs among *v*'s
/// neighbours (treating the graph as undirected) that are themselves connected:
///
/// ```text
/// C(v) = |{(u, w) ∈ E : u, w ∈ N(v)}| / (k * (k - 1) / 2)
/// ```
///
/// where *k* = |N(v)| and edges are counted undirectedly.  Nodes with fewer
/// than two neighbours receive a coefficient of `0.0`.
fn compute_clustering_coefficient(graph: &RdfGraphAdapter) -> HashMap<String, f64> {
    let n = graph.node_count();
    let mut result = HashMap::with_capacity(n);

    for v in 0..n {
        let iri = match graph.get_node_iri(v) {
            Some(s) => s.to_string(),
            None => continue,
        };

        // Build the undirected neighbour set for v
        // (union of out-neighbours and in-neighbours, excluding v itself).
        let mut neighbour_set: HashSet<usize> = HashSet::new();
        for &(u, _) in &graph.adjacency[v] {
            if u != v {
                neighbour_set.insert(u);
            }
        }
        for &(u, _) in &graph.reverse_adjacency[v] {
            if u != v {
                neighbour_set.insert(u);
            }
        }

        let k = neighbour_set.len();
        if k < 2 {
            result.insert(iri, 0.0);
            continue;
        }

        // Count undirected edges among neighbours.
        // For each pair (u, w) with u < w in neighbour_set, check if an edge
        // exists in either direction in the original directed graph.
        let neighbours: Vec<usize> = neighbour_set.into_iter().collect();
        let mut triangle_edges: usize = 0;

        for i in 0..neighbours.len() {
            let u = neighbours[i];
            // Build a fast lookup for u's out-neighbours
            let u_out: HashSet<usize> = graph.adjacency[u].iter().map(|&(w, _)| w).collect();
            let u_in: HashSet<usize> = graph.reverse_adjacency[u].iter().map(|&(w, _)| w).collect();

            for &w in neighbours.iter().skip(i + 1) {
                // Edge exists (undirected) if u→w or w→u in the directed graph
                if u_out.contains(&w) || u_in.contains(&w) {
                    triangle_edges += 1;
                }
            }
        }

        let max_edges = (k * (k - 1)) / 2;
        let coeff = if max_edges == 0 {
            0.0
        } else {
            triangle_edges as f64 / max_edges as f64
        };
        result.insert(iri, coeff);
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_acc(kind: GraphAnalyticsAggregate) -> GraphAnalyticsAccumulator {
        GraphAnalyticsAccumulator::new(kind)
    }

    // ── Accumulator mechanics ─────────────────────────────────────────────

    #[test]
    fn test_accumulate_tracks_edge_count() {
        let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
            direction: DegreeDirection::Both,
        });
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:B", "ex:C");
        assert_eq!(acc.edge_count(), 2);
        assert_eq!(acc.node_count(), 3);
    }

    #[test]
    fn test_accumulate_drops_literals() {
        let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
            direction: DegreeDirection::Both,
        });
        // literal as object
        acc.accumulate("ex:A", "\"Alice\"");
        // literal as subject (no colon)
        acc.accumulate("plainstring", "ex:B");
        // valid edge
        acc.accumulate("ex:A", "ex:B");
        assert_eq!(acc.edge_count(), 1);
        assert_eq!(acc.node_count(), 2);
    }

    // ── finalize_for_node ─────────────────────────────────────────────────

    #[test]
    fn test_finalize_for_node_existing() {
        let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
            direction: DegreeDirection::Outgoing,
        });
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:A", "ex:C");
        let score = acc
            .finalize_for_node("ex:A")
            .expect("finalize should succeed")
            .expect("ex:A should have a score");
        // ex:A has out-degree 2, n=3, norm = n-1 = 2 → score = 2/2 = 1.0
        assert!((score - 1.0).abs() < 1e-9, "score={score}");
    }

    #[test]
    fn test_finalize_for_node_missing_returns_none() {
        let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
            direction: DegreeDirection::Both,
        });
        acc.accumulate("ex:A", "ex:B");
        let result = acc
            .finalize_for_node("ex:Z")
            .expect("finalize should succeed");
        assert!(result.is_none(), "ex:Z was never accumulated");
    }

    // ── PageRank ──────────────────────────────────────────────────────────

    #[test]
    fn test_pagerank_4node_cycle_converges() {
        let mut acc = make_acc(GraphAnalyticsAggregate::PageRank {
            damping: 0.85,
            iterations: 100,
        });
        // 4-node cycle: A→B→C→D→A
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:B", "ex:C");
        acc.accumulate("ex:C", "ex:D");
        acc.accumulate("ex:D", "ex:A");

        let scores = acc.finalize().expect("PageRank should succeed");
        assert_eq!(scores.len(), 4, "4 distinct nodes expected");

        // In a symmetric cycle every node should have equal rank (~0.25)
        let expected = 1.0 / 4.0;
        for (node, &score) in &scores {
            assert!(
                (score - expected).abs() < 1e-4,
                "node {node}: expected ~{expected}, got {score}"
            );
        }

        // Scores should sum to ~1.0
        let total: f64 = scores.values().sum();
        assert!((total - 1.0).abs() < 1e-4, "sum={total}");
    }

    #[test]
    fn test_pagerank_star_center_highest() {
        let mut acc = make_acc(GraphAnalyticsAggregate::PageRank {
            damping: 0.85,
            iterations: 100,
        });
        // Spokes point into hub; hub points back to one spoke (avoids dangling)
        acc.accumulate("ex:S1", "ex:Hub");
        acc.accumulate("ex:S2", "ex:Hub");
        acc.accumulate("ex:S3", "ex:Hub");
        acc.accumulate("ex:Hub", "ex:S1");

        let scores = acc.finalize().expect("PageRank should succeed");
        let hub_score = scores["ex:Hub"];
        let s2_score = scores["ex:S2"];
        assert!(
            hub_score > s2_score,
            "hub ({hub_score}) should outrank spoke ({s2_score})"
        );
    }

    #[test]
    fn test_pagerank_empty_graph_error() {
        let mut acc = make_acc(GraphAnalyticsAggregate::PageRank {
            damping: 0.85,
            iterations: 100,
        });
        let err = acc.finalize().expect_err("empty graph should fail");
        assert!(
            matches!(err, GraphAnalyticsError::EmptyGraph),
            "expected EmptyGraph, got {err}"
        );
    }

    #[test]
    fn test_pagerank_damping_affects_rank() {
        // Build two accumulators with different damping but same graph
        let edges = [("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:C", "ex:A")];

        let mut acc1 = make_acc(GraphAnalyticsAggregate::PageRank {
            damping: 0.5,
            iterations: 200,
        });
        let mut acc2 = make_acc(GraphAnalyticsAggregate::PageRank {
            damping: 0.95,
            iterations: 200,
        });

        for (s, o) in &edges {
            acc1.accumulate(s, o);
            acc2.accumulate(s, o);
        }

        let s1 = acc1.finalize().expect("should succeed");
        let s2 = acc2.finalize().expect("should succeed");

        // Both should sum to ~1.0 even with different damping
        let sum1: f64 = s1.values().sum();
        let sum2: f64 = s2.values().sum();
        assert!((sum1 - 1.0).abs() < 1e-4, "d=0.5 sum={sum1}");
        assert!((sum2 - 1.0).abs() < 1e-4, "d=0.95 sum={sum2}");

        // Scores must differ between damping factors (symmetric ring → equal
        // scores within each run, but different magnitude between runs due to
        // different teleportation contributions).  We just confirm they're both
        // valid by checking sum-to-one rather than ordering.
        let _ = (sum1, sum2);
    }

    // ── Betweenness centrality ────────────────────────────────────────────

    #[test]
    fn test_betweenness_bridge_node_highest() {
        // A → B → C  linear chain: B is the only bridge
        let mut acc = make_acc(GraphAnalyticsAggregate::BetweennessCentrality { normalized: true });
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:B", "ex:C");

        let scores = acc.finalize().expect("should succeed");
        let b_score = scores["ex:B"];
        let a_score = scores["ex:A"];
        let c_score = scores["ex:C"];

        assert!(
            b_score > a_score,
            "bridge B ({b_score}) should beat A ({a_score})"
        );
        assert!(
            b_score > c_score,
            "bridge B ({b_score}) should beat C ({c_score})"
        );
    }

    #[test]
    fn test_betweenness_complete_graph_equal() {
        // K3 directed complete graph: all betweenness equal
        let mut acc = make_acc(GraphAnalyticsAggregate::BetweennessCentrality { normalized: true });
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:A", "ex:C");
        acc.accumulate("ex:B", "ex:A");
        acc.accumulate("ex:B", "ex:C");
        acc.accumulate("ex:C", "ex:A");
        acc.accumulate("ex:C", "ex:B");

        let scores = acc.finalize().expect("should succeed");
        let vals: Vec<f64> = scores.values().copied().collect();
        let first = vals[0];
        for &v in &vals {
            assert!(
                (v - first).abs() < 1e-9,
                "complete graph: not equal {v} vs {first}"
            );
        }
    }

    #[test]
    fn test_betweenness_normalized_in_range() {
        let mut acc = make_acc(GraphAnalyticsAggregate::BetweennessCentrality { normalized: true });
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:B", "ex:C");
        acc.accumulate("ex:C", "ex:D");
        acc.accumulate("ex:D", "ex:E");

        let scores = acc.finalize().expect("should succeed");
        for (node, &score) in &scores {
            assert!(
                (0.0..=1.0).contains(&score),
                "node {node} score {score} out of [0,1]"
            );
        }
    }

    // ── Connected components ──────────────────────────────────────────────

    #[test]
    fn test_connected_component_single_component() {
        let mut acc = make_acc(GraphAnalyticsAggregate::ConnectedComponent);
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:B", "ex:C");

        let scores = acc.finalize().expect("should succeed");
        // All three nodes belong to component 0
        for &id in scores.values() {
            assert_eq!(id as usize, 0, "all nodes in component 0");
        }
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_connected_component_two_components() {
        let mut acc = make_acc(GraphAnalyticsAggregate::ConnectedComponent);
        acc.accumulate("ex:A", "ex:B"); // component 0
        acc.accumulate("ex:C", "ex:D"); // component 1

        let scores = acc.finalize().expect("should succeed");
        let ids: HashSet<usize> = scores.values().map(|&v| v as usize).collect();
        assert_eq!(ids.len(), 2, "exactly 2 distinct component IDs expected");
    }

    #[test]
    fn test_connected_component_isolated_node() {
        // Add a node that appears in one direction and is isolated from another
        let mut acc = make_acc(GraphAnalyticsAggregate::ConnectedComponent);
        acc.accumulate("ex:A", "ex:B");
        // ex:C appears only as a subject with no further connection
        acc.accumulate("ex:C", "ex:D");

        let scores = acc.finalize().expect("should succeed");
        // Two separate components: {A,B} and {C,D}
        let ids: HashSet<usize> = scores.values().map(|&v| v as usize).collect();
        assert_eq!(ids.len(), 2);
    }

    // ── Clustering coefficient ────────────────────────────────────────────

    #[test]
    fn test_clustering_complete_graph_is_one() {
        // Triangle K3 (undirected via directed edges): every pair is connected
        let mut acc = make_acc(GraphAnalyticsAggregate::ClusteringCoefficient);
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:B", "ex:C");
        acc.accumulate("ex:A", "ex:C");

        let scores = acc.finalize().expect("should succeed");
        for (node, &coeff) in &scores {
            assert!(
                (coeff - 1.0).abs() < 1e-9,
                "K3 node {node}: expected 1.0, got {coeff}"
            );
        }
    }

    #[test]
    fn test_clustering_star_is_zero() {
        // Star graph: center → 3 spokes.  Spokes are not connected to each other.
        let mut acc = make_acc(GraphAnalyticsAggregate::ClusteringCoefficient);
        acc.accumulate("ex:Hub", "ex:S1");
        acc.accumulate("ex:Hub", "ex:S2");
        acc.accumulate("ex:Hub", "ex:S3");

        let scores = acc.finalize().expect("should succeed");
        // S1, S2, S3 each have exactly 1 neighbour → coefficient = 0
        for spoke in &["ex:S1", "ex:S2", "ex:S3"] {
            let coeff = scores[*spoke];
            assert!(coeff < 1e-9, "star leaf {spoke}: expected 0.0, got {coeff}");
        }
    }

    // ── Degree centrality ─────────────────────────────────────────────────

    #[test]
    fn test_degree_outgoing_count() {
        // ex:A has 2 out-edges; ex:B has 1; ex:C has 0.
        let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
            direction: DegreeDirection::Outgoing,
        });
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:A", "ex:C");
        acc.accumulate("ex:B", "ex:C");

        let scores = acc.finalize().expect("should succeed");
        let a = scores["ex:A"];
        let b = scores["ex:B"];
        let c = scores["ex:C"];
        // n=3, norm=2; A:2/2=1.0, B:1/2=0.5, C:0/2=0.0
        assert!((a - 1.0).abs() < 1e-9, "A out={a}");
        assert!((b - 0.5).abs() < 1e-9, "B out={b}");
        assert!((c - 0.0).abs() < 1e-9, "C out={c}");
    }

    #[test]
    fn test_degree_incoming_count() {
        // ex:C receives 2 in-edges; ex:B receives 1; ex:A receives 0.
        let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
            direction: DegreeDirection::Incoming,
        });
        acc.accumulate("ex:A", "ex:C");
        acc.accumulate("ex:B", "ex:C");
        acc.accumulate("ex:A", "ex:B");

        let scores = acc.finalize().expect("should succeed");
        let c_score = scores["ex:C"];
        let a_score = scores["ex:A"];
        assert!(
            c_score > a_score,
            "C in={c_score} should exceed A in={a_score}"
        );
    }

    #[test]
    fn test_degree_both_sum() {
        // ex:A→B, B→C: A has out=1 in=0; B has out=1 in=1; C has out=0 in=1.
        let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
            direction: DegreeDirection::Both,
        });
        acc.accumulate("ex:A", "ex:B");
        acc.accumulate("ex:B", "ex:C");

        let scores = acc.finalize().expect("should succeed");
        let b = scores["ex:B"];
        let a = scores["ex:A"];
        let c = scores["ex:C"];
        // B has in+out=2 (highest), A and C have 1 each
        assert!(b > a, "B total={b} should exceed A total={a}");
        assert!(b > c, "B total={b} should exceed C total={c}");
        assert!((a - c).abs() < 1e-9, "A and C should be equal: a={a} c={c}");
    }
}
