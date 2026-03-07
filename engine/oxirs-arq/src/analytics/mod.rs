//! # Graph Analytics for RDF Knowledge Graphs
//!
//! This module provides graph-analytics algorithms that treat an RDF graph as a
//! property graph and compute centrality measures, community structure, and
//! path-related metrics.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`adapter`] | [`RdfGraphAdapter`] – convert triples to adjacency form |
//! | [`centrality`] | [`PageRank`], [`DegreeCentrality`], [`BetweennessCentrality`] |
//! | [`community`] | [`LouvainCommunities`] – modularity-based community detection |
//! | [`components`] | [`ConnectedComponents`] – WCC and SCC |
//! | [`paths`] | [`ShortestPaths`] – Dijkstra, BFS, path reconstruction |
//! | [`stats`] | [`GraphStats`] – summary statistics |

pub mod adapter;
pub mod centrality;
pub mod community;
pub mod components;
pub mod paths;
pub mod stats;

// ── Flat re-exports ───────────────────────────────────────────────────────────

pub use adapter::{EdgeWeight, NodeId, RdfGraphAdapter};
pub use centrality::{BetweennessCentrality, DegreeCentrality, PageRank};
pub use community::LouvainCommunities;
pub use components::ConnectedComponents;
pub use paths::ShortestPaths;
pub use stats::{is_dag, is_reachable, GraphStats, GraphStatsSummary};

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn make_triples(edges: &[(&str, &str)]) -> Vec<(String, String, String)> {
        edges
            .iter()
            .map(|(s, o)| (s.to_string(), "ex:rel".to_string(), o.to_string()))
            .collect()
    }

    #[test]
    fn test_full_pipeline_small_graph() {
        let edges = [
            ("ex:A", "ex:B"),
            ("ex:B", "ex:C"),
            ("ex:C", "ex:A"),
            ("ex:D", "ex:E"),
            ("ex:E", "ex:D"),
        ];
        let triples = make_triples(&edges);
        let g = RdfGraphAdapter::from_triples(&triples);

        // Basic counts
        assert_eq!(g.node_count(), 5);
        assert_eq!(g.edge_count(), 5);

        // PageRank sums to 1
        let pr = PageRank::new().compute(&g);
        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-4, "PR total={total}");

        // Degree centrality
        let _in_dc = DegreeCentrality::in_degree(&g);
        let _out_dc = DegreeCentrality::out_degree(&g);

        // Betweenness centrality
        let _bc = BetweennessCentrality::new().compute(&g);

        // Communities
        let comms = LouvainCommunities::new().communities(&g);
        assert!(!comms.is_empty());

        // Connected components
        let wccs = ConnectedComponents::weakly_connected(&g);
        assert_eq!(wccs.len(), 2);

        let sccs = ConnectedComponents::strongly_connected(&g);
        // A-B-C form one SCC, D-E form another
        let large: Vec<&Vec<NodeId>> = sccs.iter().filter(|c| c.len() >= 2).collect();
        assert_eq!(large.len(), 2);

        // Shortest paths
        let a = g.get_node_id("ex:A").unwrap();
        let c = g.get_node_id("ex:C").unwrap();
        let d = ShortestPaths::dijkstra(&g, a);
        assert!(d.contains_key(&c));
        let p = ShortestPaths::path(&g, a, c).unwrap();
        assert_eq!(*p.first().unwrap(), a);
        assert_eq!(*p.last().unwrap(), c);

        // Graph stats
        let st = GraphStats::compute(&g);
        assert_eq!(st.node_count, 5);
        assert!(!st.is_dag); // has cycles
        assert_eq!(st.component_count, 2);
    }

    #[test]
    fn test_weighted_adapter_pipeline() {
        let triples = vec![
            (
                "ex:A".to_string(),
                "ex:heavy".to_string(),
                "ex:B".to_string(),
            ),
            (
                "ex:B".to_string(),
                "ex:light".to_string(),
                "ex:C".to_string(),
            ),
        ];
        let g = RdfGraphAdapter::from_triples_weighted(&triples, |pred, _| {
            if pred == "ex:heavy" {
                10.0
            } else {
                1.0
            }
        });
        assert_eq!(g.node_count(), 3);
        let pr = PageRank::new().compute(&g);
        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_top_k_pagerank_sorted() {
        let triples = make_triples(&[
            ("ex:A", "ex:B"),
            ("ex:A", "ex:C"),
            ("ex:A", "ex:D"),
            ("ex:B", "ex:A"),
            ("ex:C", "ex:A"),
        ]);
        let g = RdfGraphAdapter::from_triples(&triples);
        let top3 = PageRank::new().top_k(&g, 3);
        assert!(top3.len() <= 3);
        for i in 1..top3.len() {
            assert!(top3[i - 1].1 >= top3[i].1);
        }
    }

    #[test]
    fn test_modularity_louvain_partition() {
        let triples = make_triples(&[
            ("ex:A", "ex:B"),
            ("ex:B", "ex:A"),
            ("ex:C", "ex:D"),
            ("ex:D", "ex:C"),
        ]);
        let g = RdfGraphAdapter::from_triples(&triples);
        let lv = LouvainCommunities::new();
        let partition = lv.detect(&g);
        let q = lv.modularity(&g, &partition);
        assert!(q.is_finite());
    }

    #[test]
    fn test_is_dag_and_is_reachable_helpers() {
        let triples = make_triples(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let g = RdfGraphAdapter::from_triples(&triples);
        assert!(is_dag(&g));
        let a = g.get_node_id("ex:A").unwrap();
        let c = g.get_node_id("ex:C").unwrap();
        assert!(is_reachable(&g, a, c));
        assert!(!is_reachable(&g, c, a));
    }
}
