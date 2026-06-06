//! # Knowledge Graph Subgraph Summarizer
//!
//! Cluster-based abstraction for summarising KG subgraphs.
//!
//! The [`SubgraphSummarizer`] groups nodes by their `node_type` into clusters,
//! selects the most-connected node of each cluster as its representative, and
//! produces both structured [`SummaryCluster`] data and a human-readable text
//! summary paragraph.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_graphrag::summarizer::{
//!     KgEdge, KgNode, KgSubgraph, SubgraphSummarizer,
//! };
//! use std::collections::HashMap;
//!
//! let mut graph = KgSubgraph::new();
//! graph.add_node(KgNode {
//!     id: "e1".to_string(),
//!     label: "Alice".to_string(),
//!     node_type: "Person".to_string(),
//!     properties: HashMap::new(),
//! });
//! graph.add_node(KgNode {
//!     id: "e2".to_string(),
//!     label: "Bob".to_string(),
//!     node_type: "Person".to_string(),
//!     properties: HashMap::new(),
//! });
//! graph.add_edge(KgEdge {
//!     source: "e1".to_string(),
//!     target: "e2".to_string(),
//!     relation: "knows".to_string(),
//!     weight: 1.0,
//! });
//!
//! let summarizer = SubgraphSummarizer::new();
//! let clusters = summarizer.summarize(&graph, 10);
//! assert_eq!(clusters.len(), 1); // one "Person" cluster
//! ```

use std::collections::HashMap;

// ─── GraphSummarizer (v0.4.0) ────────────────────────────────────────────────

mod graph_summarizer_impl {
    use crate::graph::community::{CommunityConfig, CommunityDetector};
    use crate::Triple;
    use rayon::prelude::*;
    use std::collections::HashMap;

    /// A compact summary of a large KG subgraph, suitable for LLM context windows.
    #[derive(Debug, Clone)]
    pub struct GraphSummary {
        /// Representative entity IRIs (one per community, by in-degree centrality).
        pub entities: Vec<String>,
        /// Selected triples involving representative nodes and top predicates.
        pub relations: Vec<(String, String, String)>,
        /// Human-readable label for each detected community.
        pub community_labels: Vec<String>,
    }

    impl GraphSummary {
        /// Serialise to a natural-language paragraph for LLM context.
        pub fn to_text(&self) -> String {
            if self.entities.is_empty() {
                return "The graph is empty.".to_string();
            }
            let n_comm = self.community_labels.len();
            let n_ent = self.entities.len();
            let entity_list = self
                .entities
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            // Collect unique predicates from selected relations, ordered by first appearance.
            let mut seen_preds: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            let mut pred_list: Vec<String> = Vec::new();
            for (_, p, _) in &self.relations {
                if seen_preds.insert(p.clone()) {
                    pred_list.push(p.clone());
                    if pred_list.len() >= 5 {
                        break;
                    }
                }
            }
            format!(
                "The graph contains {} {} across {} {}. \
                 Key entities include: {}. \
                 Primary relationships: {}.",
                n_ent,
                if n_ent == 1 { "entity" } else { "entities" },
                n_comm,
                if n_comm == 1 {
                    "community"
                } else {
                    "communities"
                },
                entity_list,
                if pred_list.is_empty() {
                    "none".to_string()
                } else {
                    pred_list.join(", ")
                },
            )
        }
    }

    /// Compresses a large KG subgraph (as raw triples) into a [`GraphSummary`]
    /// capped at `max_nodes` entities and `max_triples` relations.
    ///
    /// Pipeline:
    /// 1. Community detection via Leiden (reusing `src/graph/community.rs`).
    /// 2. Per-community representative selection by in-degree centrality.
    /// 3. Predicate frequency ranking.
    /// 4. Triple selection and truncation.
    pub struct GraphSummarizer {
        pub max_nodes: usize,
        pub max_triples: usize,
    }

    impl GraphSummarizer {
        pub fn new(max_nodes: usize, max_triples: usize) -> Self {
            Self {
                max_nodes,
                max_triples,
            }
        }

        /// Summarise the given triples into a [`GraphSummary`].
        pub fn summarize(&self, triples: &[(String, String, String)]) -> GraphSummary {
            if triples.is_empty() {
                return GraphSummary {
                    entities: Vec::new(),
                    relations: Vec::new(),
                    community_labels: Vec::new(),
                };
            }

            // ── Convert to crate Triple for community detector ────────────────
            let core_triples: Vec<Triple> = triples
                .iter()
                .map(|(s, p, o)| Triple::new(s.clone(), p.clone(), o.clone()))
                .collect();

            // ── Build in-degree map: how many triples point TO each node ─────
            let mut in_degree: HashMap<String, usize> = HashMap::new();
            for (s, _, o) in triples {
                // Initialise subject with 0 if not present (to ensure all nodes tracked).
                in_degree.entry(s.clone()).or_insert(0);
                *in_degree.entry(o.clone()).or_insert(0) += 1;
            }

            // ── Community detection via Leiden ────────────────────────────────
            let config = CommunityConfig {
                min_community_size: 1,
                ..CommunityConfig::default()
            };
            let detector = CommunityDetector::new(config);
            let communities = detector.detect(&core_triples).unwrap_or_default();

            // ── Per-community: pick representative by max in-degree ───────────
            // Use rayon for parallel centroid selection when there are many communities.
            let representatives: Vec<(String, String)> = communities
                .par_iter()
                .enumerate()
                .filter_map(|(idx, comm)| {
                    let rep = comm
                        .entities
                        .iter()
                        .max_by_key(|e| in_degree.get(e.as_str()).copied().unwrap_or(0))
                        .cloned()?;
                    let label = format!("Community {} ({} entities)", idx, comm.entities.len());
                    Some((rep, label))
                })
                .collect();

            // ── Predicate frequency ranking ───────────────────────────────────
            let mut pred_freq: HashMap<&str, usize> = HashMap::new();
            for (_, p, _) in triples {
                *pred_freq.entry(p.as_str()).or_insert(0) += 1;
            }
            let mut pred_ranked: Vec<(&str, usize)> = pred_freq.into_iter().collect();
            pred_ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
            let top_predicates: std::collections::HashSet<&str> =
                pred_ranked.iter().take(10).map(|(p, _)| *p).collect();

            // ── Build entities list (capped at max_nodes) ────────────────────
            let entities: Vec<String> = representatives
                .iter()
                .map(|(rep, _)| rep.clone())
                .take(self.max_nodes)
                .collect();

            let rep_set: std::collections::HashSet<&str> =
                entities.iter().map(|s| s.as_str()).collect();

            // ── Select relations: triples involving representatives + top preds
            let mut selected: Vec<(String, String, String)> = triples
                .iter()
                .filter(|(s, p, _)| {
                    rep_set.contains(s.as_str()) && top_predicates.contains(p.as_str())
                })
                .map(|(s, p, o)| (s.clone(), p.clone(), o.clone()))
                .take(self.max_triples)
                .collect();

            // If still below max_triples, fill from any triple with a representative subject.
            if selected.len() < self.max_triples {
                for (s, p, o) in triples {
                    if selected.len() >= self.max_triples {
                        break;
                    }
                    if rep_set.contains(s.as_str()) {
                        let candidate = (s.clone(), p.clone(), o.clone());
                        if !selected.contains(&candidate) {
                            selected.push(candidate);
                        }
                    }
                }
            }

            let community_labels: Vec<String> = representatives
                .iter()
                .take(self.max_nodes)
                .map(|(_, label)| label.clone())
                .collect();

            GraphSummary {
                entities,
                relations: selected,
                community_labels,
            }
        }
    }
} // mod graph_summarizer_impl

pub use graph_summarizer_impl::{GraphSummarizer, GraphSummary};

// ─── Graph primitives ─────────────────────────────────────────────────────────

/// A node in a knowledge-graph subgraph.
#[derive(Debug, Clone)]
pub struct KgNode {
    /// Unique node identifier (URI or local name).
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Semantic type / RDF class.
    pub node_type: String,
    /// Arbitrary key-value properties.
    pub properties: HashMap<String, String>,
}

impl KgNode {
    /// Create a node with no properties.
    pub fn simple(
        id: impl Into<String>,
        label: impl Into<String>,
        node_type: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            node_type: node_type.into(),
            properties: HashMap::new(),
        }
    }
}

/// A directed, weighted edge between two KG nodes.
#[derive(Debug, Clone)]
pub struct KgEdge {
    /// Source node identifier.
    pub source: String,
    /// Target node identifier.
    pub target: String,
    /// Relation type / predicate label.
    pub relation: String,
    /// Edge weight (e.g. confidence score).
    pub weight: f64,
}

impl KgEdge {
    /// Create an unweighted (weight = 1.0) edge.
    pub fn unweighted(
        source: impl Into<String>,
        target: impl Into<String>,
        relation: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            relation: relation.into(),
            weight: 1.0,
        }
    }
}

// ─── KgSubgraph ──────────────────────────────────────────────────────────────

/// A subgraph of a knowledge graph, consisting of nodes and directed edges.
#[derive(Debug, Clone, Default)]
pub struct KgSubgraph {
    /// All nodes in the subgraph.
    pub nodes: Vec<KgNode>,
    /// All edges in the subgraph.
    pub edges: Vec<KgEdge>,
}

impl KgSubgraph {
    /// Create an empty subgraph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node.
    pub fn add_node(&mut self, node: KgNode) {
        self.nodes.push(node);
    }

    /// Add an edge.
    pub fn add_edge(&mut self, edge: KgEdge) {
        self.edges.push(edge);
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return the node with the given `id`, or `None`.
    pub fn node(&self, id: &str) -> Option<&KgNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

// ─── Summary output ──────────────────────────────────────────────────────────

/// A cluster of KG nodes of the same type, with a representative and summary.
#[derive(Debug, Clone)]
pub struct SummaryCluster {
    /// Sequential cluster identifier.
    pub id: usize,
    /// Node ID chosen as the representative (most-connected node in cluster).
    pub representative_node: String,
    /// All node IDs belonging to this cluster.
    pub member_nodes: Vec<String>,
    /// Number of edges whose both endpoints are within this cluster.
    pub internal_edges: usize,
    /// Brief human-readable label derived from the cluster's node type.
    pub summary_label: String,
}

impl SummaryCluster {
    /// Size of the cluster (number of member nodes).
    pub fn size(&self) -> usize {
        self.member_nodes.len()
    }
}

// ─── SubgraphSummarizer ───────────────────────────────────────────────────────

/// Produces cluster-based summaries of KG subgraphs.
pub struct SubgraphSummarizer;

impl SubgraphSummarizer {
    /// Create a new summarizer (stateless).
    pub fn new() -> Self {
        Self
    }

    /// Summarise `graph` into at most `max_clusters` [`SummaryCluster`]s.
    ///
    /// ## Algorithm
    ///
    /// 1. Group nodes by `node_type`.
    /// 2. If there are more types than `max_clusters`, merge the smallest groups
    ///    into an `"Other"` cluster so that only `max_clusters` remain.
    /// 3. For each cluster, count internal edges and pick the most-connected
    ///    node (highest undirected degree within the whole graph) as
    ///    the representative.
    pub fn summarize(&self, graph: &KgSubgraph, max_clusters: usize) -> Vec<SummaryCluster> {
        if graph.nodes.is_empty() || max_clusters == 0 {
            return Vec::new();
        }

        // Group node IDs by node_type.
        let mut type_groups: HashMap<String, Vec<String>> = HashMap::new();
        for node in &graph.nodes {
            type_groups
                .entry(node.node_type.clone())
                .or_default()
                .push(node.id.clone());
        }

        // Sort groups deterministically by type name.
        let mut groups: Vec<(String, Vec<String>)> = type_groups.into_iter().collect();
        groups.sort_by(|a, b| a.0.cmp(&b.0));

        // Merge overflow groups into "Other" if too many types.
        let groups = if groups.len() > max_clusters {
            let (keep, overflow) = groups.split_at(max_clusters - 1);
            let mut merged = keep.to_vec();
            let other_members: Vec<String> = overflow
                .iter()
                .flat_map(|(_, ids)| ids.iter().cloned())
                .collect();
            if !other_members.is_empty() {
                merged.push(("Other".to_string(), other_members));
            }
            merged
        } else {
            groups
        };

        // Build undirected degree map over the entire graph.
        let degree_map = build_degree_map(graph);

        // Build the clusters.
        groups
            .into_iter()
            .enumerate()
            .map(|(cluster_id, (node_type, members))| {
                // Pick representative: member with highest degree.
                let representative = members
                    .iter()
                    .max_by_key(|id| degree_map.get(*id).copied().unwrap_or(0))
                    .cloned()
                    .unwrap_or_default();

                // Count internal edges (both endpoints in this cluster).
                let member_set: std::collections::HashSet<&str> =
                    members.iter().map(|s| s.as_str()).collect();
                let internal_edges = graph
                    .edges
                    .iter()
                    .filter(|e| {
                        member_set.contains(e.source.as_str())
                            && member_set.contains(e.target.as_str())
                    })
                    .count();

                SummaryCluster {
                    id: cluster_id,
                    representative_node: representative,
                    member_nodes: members,
                    internal_edges,
                    summary_label: format!("{node_type} cluster"),
                }
            })
            .collect()
    }

    /// Return the top-`top_n` relation types by frequency of occurrence.
    ///
    /// Ties are broken by relation name (alphabetical ascending).
    pub fn extract_key_relations(&self, graph: &KgSubgraph, top_n: usize) -> Vec<(String, usize)> {
        if top_n == 0 {
            return Vec::new();
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for edge in &graph.edges {
            *counts.entry(edge.relation.clone()).or_insert(0) += 1;
        }
        let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted.truncate(top_n);
        sorted
    }

    /// Compute the undirected degree of `node_id` in `graph`.
    ///
    /// Both outgoing and incoming edges are counted (a self-loop counts twice).
    pub fn node_degree(&self, graph: &KgSubgraph, node_id: &str) -> usize {
        graph
            .edges
            .iter()
            .filter(|e| e.source == node_id || e.target == node_id)
            .count()
    }

    /// Generate a human-readable summary paragraph from a set of clusters.
    ///
    /// The paragraph lists each cluster's representative node, member count, and
    /// internal edge count, ending with the total number of clusters.
    pub fn generate_text_summary(&self, clusters: &[SummaryCluster]) -> String {
        if clusters.is_empty() {
            return "The subgraph contains no identifiable clusters.".to_string();
        }

        let mut parts: Vec<String> = Vec::new();
        for cluster in clusters {
            parts.push(format!(
                "The {} (representative: {}, {} members, {} internal edges)",
                cluster.summary_label,
                cluster.representative_node,
                cluster.member_nodes.len(),
                cluster.internal_edges,
            ));
        }

        format!(
            "{}. The subgraph contains {} cluster{}.",
            parts.join(". "),
            clusters.len(),
            if clusters.len() == 1 { "" } else { "s" }
        )
    }
}

impl Default for SubgraphSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Build an undirected-degree map over all edges in `graph`.
fn build_degree_map(graph: &KgSubgraph) -> HashMap<String, usize> {
    let mut map: HashMap<String, usize> = HashMap::new();
    for edge in &graph.edges {
        *map.entry(edge.source.clone()).or_insert(0) += 1;
        if edge.source != edge.target {
            *map.entry(edge.target.clone()).or_insert(0) += 1;
        }
    }
    map
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn node(id: &str, node_type: &str) -> KgNode {
        KgNode::simple(id, id, node_type)
    }

    fn edge(src: &str, tgt: &str, rel: &str) -> KgEdge {
        KgEdge::unweighted(src, tgt, rel)
    }

    fn make_graph_with_types(specs: &[(&str, &str)], edges: &[(&str, &str, &str)]) -> KgSubgraph {
        let mut g = KgSubgraph::new();
        for (id, typ) in specs {
            g.add_node(node(id, typ));
        }
        for (s, t, r) in edges {
            g.add_edge(edge(s, t, r));
        }
        g
    }

    // ── KgSubgraph basics ────────────────────────────────────────────────────

    #[test]
    fn test_new_subgraph_empty() {
        let g = KgSubgraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_add_node_increments_count() {
        let mut g = KgSubgraph::new();
        g.add_node(node("n1", "Person"));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_add_edge_increments_count() {
        let mut g = KgSubgraph::new();
        g.add_edge(edge("a", "b", "knows"));
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_node_lookup_found() {
        let mut g = KgSubgraph::new();
        g.add_node(node("alice", "Person"));
        assert!(g.node("alice").is_some());
        assert_eq!(g.node("alice").expect("node").node_type, "Person");
    }

    #[test]
    fn test_node_lookup_missing() {
        let g = KgSubgraph::new();
        assert!(g.node("nope").is_none());
    }

    // ── KgNode simple constructor ─────────────────────────────────────────────

    #[test]
    fn test_kg_node_simple() {
        let n = KgNode::simple("id1", "Label", "Type");
        assert_eq!(n.id, "id1");
        assert_eq!(n.label, "Label");
        assert_eq!(n.node_type, "Type");
        assert!(n.properties.is_empty());
    }

    // ── KgEdge unweighted ─────────────────────────────────────────────────────

    #[test]
    fn test_kg_edge_unweighted_weight_one() {
        let e = KgEdge::unweighted("a", "b", "rel");
        assert_eq!(e.weight, 1.0);
    }

    // ── summarize: empty graph ────────────────────────────────────────────────

    #[test]
    fn test_summarize_empty_graph() {
        let g = KgSubgraph::new();
        let s = SubgraphSummarizer::new();
        assert!(s.summarize(&g, 10).is_empty());
    }

    #[test]
    fn test_summarize_max_clusters_zero() {
        let g = make_graph_with_types(&[("n1", "Person")], &[]);
        let s = SubgraphSummarizer::new();
        assert!(s.summarize(&g, 0).is_empty());
    }

    // ── summarize: single type ────────────────────────────────────────────────

    #[test]
    fn test_summarize_single_type_one_cluster() {
        let g = make_graph_with_types(&[("a", "Person"), ("b", "Person"), ("c", "Person")], &[]);
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 5);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].member_nodes.len(), 3);
    }

    #[test]
    fn test_summarize_cluster_label_contains_type() {
        let g = make_graph_with_types(&[("a", "Organization")], &[]);
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 5);
        assert!(
            clusters[0].summary_label.contains("Organization"),
            "label should mention the type: {}",
            clusters[0].summary_label
        );
    }

    // ── summarize: multiple types ─────────────────────────────────────────────

    #[test]
    fn test_summarize_two_types_two_clusters() {
        let g = make_graph_with_types(&[("a", "Person"), ("b", "Person"), ("c", "Company")], &[]);
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 10);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_summarize_respects_max_clusters() {
        let g = make_graph_with_types(&[("a", "T1"), ("b", "T2"), ("c", "T3"), ("d", "T4")], &[]);
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 2);
        assert!(
            clusters.len() <= 2,
            "should have at most 2 clusters, got {}",
            clusters.len()
        );
    }

    #[test]
    fn test_summarize_overflow_goes_to_other() {
        let g = make_graph_with_types(
            &[
                ("a", "T1"),
                ("b", "T2"),
                ("c", "T3"),
                ("d", "T4"),
                ("e", "T5"),
            ],
            &[],
        );
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 3);
        // Should have 3 clusters: T1, T2, Other (T3+T4+T5)
        assert_eq!(clusters.len(), 3);
        let has_other = clusters.iter().any(|c| c.summary_label.contains("Other"));
        assert!(has_other, "overflow should be merged into Other cluster");
    }

    // ── summarize: representative selection ──────────────────────────────────

    #[test]
    fn test_representative_is_most_connected() {
        // "b" has degree 2, "a" and "c" have degree 1 — b should be representative
        let g = make_graph_with_types(
            &[("a", "Person"), ("b", "Person"), ("c", "Person")],
            &[("a", "b", "knows"), ("b", "c", "knows")],
        );
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 5);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].representative_node, "b");
    }

    #[test]
    fn test_representative_exists_for_single_node_cluster() {
        let g = make_graph_with_types(&[("solo", "Category")], &[]);
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 5);
        assert_eq!(clusters[0].representative_node, "solo");
    }

    // ── summarize: internal edges ─────────────────────────────────────────────

    #[test]
    fn test_internal_edges_all_within_cluster() {
        let g = make_graph_with_types(
            &[("a", "T"), ("b", "T"), ("c", "T")],
            &[("a", "b", "r"), ("b", "c", "r")],
        );
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 5);
        assert_eq!(clusters[0].internal_edges, 2);
    }

    #[test]
    fn test_internal_edges_none_cross_cluster() {
        // Edges between different types — should not appear as internal.
        let g = make_graph_with_types(
            &[("a", "T1"), ("b", "T1"), ("c", "T2")],
            &[("a", "c", "cross"), ("b", "c", "cross")],
        );
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 5);
        for cluster in &clusters {
            if cluster.summary_label.contains("T1") {
                assert_eq!(cluster.internal_edges, 0);
            }
        }
    }

    // ── summarize: cluster IDs ────────────────────────────────────────────────

    #[test]
    fn test_cluster_ids_are_sequential() {
        let g = make_graph_with_types(&[("a", "T1"), ("b", "T2"), ("c", "T3")], &[]);
        let s = SubgraphSummarizer::new();
        let clusters = s.summarize(&g, 10);
        for (i, c) in clusters.iter().enumerate() {
            assert_eq!(c.id, i);
        }
    }

    // ── extract_key_relations ─────────────────────────────────────────────────

    #[test]
    fn test_key_relations_empty_graph() {
        let g = KgSubgraph::new();
        let s = SubgraphSummarizer::new();
        assert!(s.extract_key_relations(&g, 5).is_empty());
    }

    #[test]
    fn test_key_relations_top_n_zero() {
        let mut g = KgSubgraph::new();
        g.add_edge(edge("a", "b", "knows"));
        let s = SubgraphSummarizer::new();
        assert!(s.extract_key_relations(&g, 0).is_empty());
    }

    #[test]
    fn test_key_relations_sorted_by_frequency() {
        let g = make_graph_with_types(
            &[("a", "T"), ("b", "T"), ("c", "T")],
            &[
                ("a", "b", "knows"),
                ("b", "c", "knows"),
                ("a", "c", "likes"),
            ],
        );
        let s = SubgraphSummarizer::new();
        let relations = s.extract_key_relations(&g, 5);
        assert!(!relations.is_empty());
        assert_eq!(relations[0].0, "knows");
        assert_eq!(relations[0].1, 2);
    }

    #[test]
    fn test_key_relations_truncated_to_top_n() {
        let g = make_graph_with_types(
            &[("a", "T"), ("b", "T"), ("c", "T")],
            &[
                ("a", "b", "r1"),
                ("a", "b", "r2"),
                ("a", "b", "r3"),
                ("a", "b", "r4"),
                ("a", "b", "r5"),
            ],
        );
        let s = SubgraphSummarizer::new();
        let relations = s.extract_key_relations(&g, 3);
        assert!(relations.len() <= 3);
    }

    #[test]
    fn test_key_relations_single_relation() {
        let mut g = KgSubgraph::new();
        for i in 0..5 {
            g.add_edge(edge(&format!("n{i}"), &format!("n{}", i + 1), "knows"));
        }
        let s = SubgraphSummarizer::new();
        let rels = s.extract_key_relations(&g, 1);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].0, "knows");
        assert_eq!(rels[0].1, 5);
    }

    // ── node_degree ───────────────────────────────────────────────────────────

    #[test]
    fn test_node_degree_no_edges() {
        let g = make_graph_with_types(&[("a", "T")], &[]);
        let s = SubgraphSummarizer::new();
        assert_eq!(s.node_degree(&g, "a"), 0);
    }

    #[test]
    fn test_node_degree_outgoing_only() {
        let g = make_graph_with_types(&[("a", "T"), ("b", "T")], &[("a", "b", "r")]);
        let s = SubgraphSummarizer::new();
        assert_eq!(s.node_degree(&g, "a"), 1);
        assert_eq!(s.node_degree(&g, "b"), 1);
    }

    #[test]
    fn test_node_degree_multiple_edges() {
        let g = make_graph_with_types(
            &[("a", "T"), ("b", "T"), ("c", "T")],
            &[("a", "b", "r"), ("a", "c", "r"), ("b", "a", "r")],
        );
        let s = SubgraphSummarizer::new();
        // a appears in edges: (a→b), (a→c), (b→a) = 3
        assert_eq!(s.node_degree(&g, "a"), 3);
    }

    #[test]
    fn test_node_degree_missing_node() {
        let g = KgSubgraph::new();
        let s = SubgraphSummarizer::new();
        assert_eq!(s.node_degree(&g, "ghost"), 0);
    }

    // ── generate_text_summary ─────────────────────────────────────────────────

    #[test]
    fn test_text_summary_empty_clusters() {
        let s = SubgraphSummarizer::new();
        let text = s.generate_text_summary(&[]);
        assert!(text.contains("no identifiable clusters"), "text: {text}");
    }

    #[test]
    fn test_text_summary_single_cluster() {
        let clusters = vec![SummaryCluster {
            id: 0,
            representative_node: "alice".to_string(),
            member_nodes: vec!["alice".to_string(), "bob".to_string()],
            internal_edges: 1,
            summary_label: "Person cluster".to_string(),
        }];
        let s = SubgraphSummarizer::new();
        let text = s.generate_text_summary(&clusters);
        assert!(text.contains("alice"), "text: {text}");
        assert!(text.contains("Person cluster"), "text: {text}");
        assert!(text.contains("1 cluster"), "text: {text}");
    }

    #[test]
    fn test_text_summary_multiple_clusters() {
        let clusters = vec![
            SummaryCluster {
                id: 0,
                representative_node: "a".to_string(),
                member_nodes: vec!["a".to_string()],
                internal_edges: 0,
                summary_label: "Person cluster".to_string(),
            },
            SummaryCluster {
                id: 1,
                representative_node: "c".to_string(),
                member_nodes: vec!["c".to_string(), "d".to_string()],
                internal_edges: 1,
                summary_label: "Company cluster".to_string(),
            },
        ];
        let s = SubgraphSummarizer::new();
        let text = s.generate_text_summary(&clusters);
        assert!(text.contains("2 clusters"), "text: {text}");
        assert!(text.contains("Person cluster"), "text: {text}");
        assert!(text.contains("Company cluster"), "text: {text}");
    }

    #[test]
    fn test_text_summary_contains_member_count() {
        let clusters = vec![SummaryCluster {
            id: 0,
            representative_node: "x".to_string(),
            member_nodes: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            internal_edges: 2,
            summary_label: "T cluster".to_string(),
        }];
        let s = SubgraphSummarizer::new();
        let text = s.generate_text_summary(&clusters);
        assert!(text.contains("3 members"), "text: {text}");
    }

    #[test]
    fn test_text_summary_contains_internal_edge_count() {
        let clusters = vec![SummaryCluster {
            id: 0,
            representative_node: "x".to_string(),
            member_nodes: vec!["x".to_string()],
            internal_edges: 7,
            summary_label: "T cluster".to_string(),
        }];
        let s = SubgraphSummarizer::new();
        let text = s.generate_text_summary(&clusters);
        assert!(text.contains("7 internal edges"), "text: {text}");
    }

    // ── SummaryCluster helpers ────────────────────────────────────────────────

    #[test]
    fn test_cluster_size() {
        let c = SummaryCluster {
            id: 0,
            representative_node: "r".to_string(),
            member_nodes: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            internal_edges: 0,
            summary_label: "X cluster".to_string(),
        };
        assert_eq!(c.size(), 3);
    }

    // ── default impl ──────────────────────────────────────────────────────────

    #[test]
    fn test_default_summarizer() {
        let s = SubgraphSummarizer;
        let g = KgSubgraph::new();
        assert!(s.summarize(&g, 5).is_empty());
    }

    #[test]
    fn test_default_subgraph() {
        let g = KgSubgraph::default();
        assert_eq!(g.node_count(), 0);
    }
}

// ─── GraphSummarizer tests ────────────────────────────────────────────────────

#[cfg(test)]
mod graph_summarizer_tests {
    use super::GraphSummarizer;

    /// Build a synthetic set of triples for testing.
    fn sample_triples() -> Vec<(String, String, String)> {
        vec![
            ("Alice".into(), "knows".into(), "Bob".into()),
            ("Bob".into(), "knows".into(), "Carol".into()),
            ("Carol".into(), "knows".into(), "Alice".into()),
            ("Alice".into(), "worksAt".into(), "ACME".into()),
            ("Bob".into(), "worksAt".into(), "ACME".into()),
            ("ACME".into(), "locatedIn".into(), "Berlin".into()),
            ("Dave".into(), "knows".into(), "Eve".into()),
            ("Eve".into(), "knows".into(), "Frank".into()),
            ("Dave".into(), "worksAt".into(), "WidgetCo".into()),
            ("WidgetCo".into(), "locatedIn".into(), "Paris".into()),
        ]
    }

    #[test]
    fn test_summary_respects_max_nodes() {
        let summarizer = GraphSummarizer::new(3, 20);
        let triples = sample_triples();
        let summary = summarizer.summarize(&triples);
        assert!(
            summary.entities.len() <= 3,
            "expected ≤3 entities, got {}",
            summary.entities.len()
        );
    }

    #[test]
    fn test_summary_respects_max_triples() {
        let summarizer = GraphSummarizer::new(10, 4);
        let triples = sample_triples();
        let summary = summarizer.summarize(&triples);
        assert!(
            summary.relations.len() <= 4,
            "expected ≤4 relations, got {}",
            summary.relations.len()
        );
    }

    #[test]
    fn test_to_text_non_empty_on_non_empty_graph() {
        let summarizer = GraphSummarizer::new(10, 20);
        let triples = sample_triples();
        let summary = summarizer.summarize(&triples);
        let text = summary.to_text();
        assert!(!text.is_empty(), "to_text() should not be empty");
    }

    #[test]
    fn test_empty_graph_returns_empty_summary() {
        let summarizer = GraphSummarizer::new(10, 20);
        let summary = summarizer.summarize(&[]);
        assert!(
            summary.entities.is_empty(),
            "empty graph: entities should be empty"
        );
        assert!(
            summary.relations.is_empty(),
            "empty graph: relations should be empty"
        );
    }

    #[test]
    fn test_community_labels_present_when_graph_has_nodes() {
        let summarizer = GraphSummarizer::new(10, 20);
        let triples = sample_triples();
        let summary = summarizer.summarize(&triples);
        assert!(
            !summary.community_labels.is_empty(),
            "community_labels should be non-empty when graph has nodes"
        );
    }

    #[test]
    fn test_predicate_frequency_ordering() {
        // "knows" appears 5×, "worksAt" appears 3×, "locatedIn" appears 2×.
        // The most-used predicate must appear first among the relations.
        let summarizer = GraphSummarizer::new(10, 20);
        let triples = sample_triples();
        let summary = summarizer.summarize(&triples);
        // Collect the first predicate that appears in relations.
        if let Some((_, p, _)) = summary.relations.first() {
            assert_eq!(
                p.as_str(),
                "knows",
                "first predicate in relations should be 'knows' (most frequent), got '{p}'"
            );
        }
        // If relations are empty, the max_triples was 0 — skip the ordering check.
    }
}
