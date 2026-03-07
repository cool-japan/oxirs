//! Graph statistics summary

use super::adapter::{NodeId, RdfGraphAdapter};
use super::components::ConnectedComponents;

/// Summary statistics for an RDF graph.
#[derive(Debug, Clone)]
pub struct GraphStatsSummary {
    /// Number of unique IRI nodes.
    pub node_count: usize,
    /// Number of unique directed edges.
    pub edge_count: usize,
    /// Average total degree (in + out) across all nodes.
    pub avg_degree: f64,
    /// Maximum total degree of any single node.
    pub max_degree: usize,
    /// Graph density: `edges / (nodes * (nodes - 1))` for a directed graph.
    pub density: f64,
    /// `true` when the graph is a directed acyclic graph (no strongly connected
    /// components of size > 1).
    pub is_dag: bool,
    /// Number of weakly connected components.
    pub component_count: usize,
}

/// Compute summary statistics for an RDF graph.
pub struct GraphStats;

impl GraphStats {
    /// Compute all summary statistics in one pass.
    pub fn compute(graph: &RdfGraphAdapter) -> GraphStatsSummary {
        let n = graph.node_count();
        let e = graph.edge_count();

        // Degree statistics
        let degrees: Vec<usize> = (0..n)
            .map(|u| graph.adjacency[u].len() + graph.reverse_adjacency[u].len())
            .collect();
        let max_degree = degrees.iter().copied().max().unwrap_or(0);
        let avg_degree = if n == 0 {
            0.0
        } else {
            degrees.iter().sum::<usize>() as f64 / n as f64
        };

        // Density
        let density = if n <= 1 {
            0.0
        } else {
            e as f64 / (n as f64 * (n - 1) as f64)
        };

        // is_dag: true iff every SCC has exactly one node
        let sccs = ConnectedComponents::strongly_connected(graph);
        let is_dag = sccs.iter().all(|c| c.len() == 1);

        // Component count (weakly connected)
        let components = ConnectedComponents::weakly_connected(graph);
        let component_count = components.len();

        GraphStatsSummary {
            node_count: n,
            edge_count: e,
            avg_degree,
            max_degree,
            density,
            is_dag,
            component_count,
        }
    }
}

/// Check whether a graph contains a cycle (quick wrapper).
pub fn is_dag(graph: &RdfGraphAdapter) -> bool {
    GraphStats::compute(graph).is_dag
}

/// Check whether a node is reachable from another using BFS.
pub fn is_reachable(graph: &RdfGraphAdapter, from: NodeId, to: NodeId) -> bool {
    use std::collections::VecDeque;
    let n = graph.node_count();
    if from >= n || to >= n {
        return false;
    }
    if from == to {
        return true;
    }
    let mut visited = vec![false; n];
    visited[from] = true;
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    queue.push_back(from);
    while let Some(u) = queue.pop_front() {
        for &(v, _) in &graph.adjacency[u] {
            if v == to {
                return true;
            }
            if !visited[v] {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }
    false
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

    #[test]
    fn test_stats_empty() {
        let g = RdfGraphAdapter::from_triples(&[]);
        let s = GraphStats::compute(&g);
        assert_eq!(s.node_count, 0);
        assert_eq!(s.edge_count, 0);
        assert_eq!(s.avg_degree, 0.0);
        assert_eq!(s.max_degree, 0);
        assert_eq!(s.density, 0.0);
        assert!(s.is_dag);
        assert_eq!(s.component_count, 0);
    }

    #[test]
    fn test_stats_single_edge() {
        let g = build_graph(&[("ex:A", "ex:B")]);
        let s = GraphStats::compute(&g);
        assert_eq!(s.node_count, 2);
        assert_eq!(s.edge_count, 1);
        assert!(s.is_dag);
        assert_eq!(s.component_count, 1);
    }

    #[test]
    fn test_stats_cycle_not_dag() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:A")]);
        let s = GraphStats::compute(&g);
        assert!(!s.is_dag, "graph with cycle should not be a DAG");
    }

    #[test]
    fn test_stats_dag() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:A", "ex:C")]);
        let s = GraphStats::compute(&g);
        assert!(s.is_dag);
    }

    #[test]
    fn test_stats_density() {
        // 3 nodes, 2 edges → density = 2 / (3*2) = 0.333...
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let s = GraphStats::compute(&g);
        assert!((s.density - 2.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_component_count() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:C", "ex:D"), ("ex:E", "ex:F")]);
        let s = GraphStats::compute(&g);
        assert_eq!(s.component_count, 3);
    }

    #[test]
    fn test_stats_avg_degree() {
        // A→B: A has out-deg 1, in-deg 0 → total 1
        //       B has out-deg 0, in-deg 1 → total 1
        // avg = 1.0
        let g = build_graph(&[("ex:A", "ex:B")]);
        let s = GraphStats::compute(&g);
        assert!((s.avg_degree - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_max_degree() {
        // Hub: A→B, A→C, A→D, B→A   → A has out 3 + in 1 = 4
        let g = build_graph(&[
            ("ex:A", "ex:B"),
            ("ex:A", "ex:C"),
            ("ex:A", "ex:D"),
            ("ex:B", "ex:A"),
        ]);
        let s = GraphStats::compute(&g);
        assert!(s.max_degree >= 4);
    }

    #[test]
    fn test_is_dag_helper() {
        let g_dag = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let g_cycle = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:A")]);
        assert!(is_dag(&g_dag));
        assert!(!is_dag(&g_cycle));
    }

    #[test]
    fn test_is_reachable() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let a = g.get_node_id("ex:A").unwrap();
        let b = g.get_node_id("ex:B").unwrap();
        let c = g.get_node_id("ex:C").unwrap();
        assert!(is_reachable(&g, a, c));
        assert!(!is_reachable(&g, c, a)); // directed: no path back
        assert!(is_reachable(&g, a, b));
        assert!(is_reachable(&g, a, a)); // self
    }

    #[test]
    fn test_is_reachable_out_of_range() {
        let g = build_graph(&[("ex:A", "ex:B")]);
        let a = g.get_node_id("ex:A").unwrap();
        assert!(!is_reachable(&g, a, 999));
        assert!(!is_reachable(&g, 999, a));
    }
}
