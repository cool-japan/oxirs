//! RDF graph adapter: converts RDF triples into adjacency representation

use std::collections::HashMap;

/// Internal node identifier
pub type NodeId = usize;

/// Edge weight type
pub type EdgeWeight = f64;

/// Converts an RDF graph (set of triples) into an adjacency representation
/// suitable for graph-analytics algorithms.
///
/// Only IRI subjects and objects are treated as nodes; literal objects are
/// silently ignored.
pub struct RdfGraphAdapter {
    /// IRI string → internal node id
    pub node_to_id: HashMap<String, NodeId>,
    /// Internal node id → IRI string
    pub id_to_node: Vec<String>,
    /// Outgoing adjacency list: `adjacency[u]` = `[(v, w), …]`
    pub adjacency: Vec<Vec<(NodeId, EdgeWeight)>>,
    /// Incoming adjacency list: `reverse_adjacency[v]` = `[(u, w), …]`
    pub reverse_adjacency: Vec<Vec<(NodeId, EdgeWeight)>>,
    /// Total number of edges (including duplicates compressed to one entry with accumulated weight)
    edge_count_internal: usize,
}

impl RdfGraphAdapter {
    /// Build from a slice of `(subject, predicate, object)` string triples.
    ///
    /// Every distinct IRI subject/object becomes a node.  All edges receive a
    /// weight of `1.0`.  Duplicate `(s, o)` pairs with the same predicate are
    /// collapsed to a single edge whose weight equals the number of occurrences.
    pub fn from_triples(triples: &[(String, String, String)]) -> Self {
        Self::from_triples_weighted(triples, |_pred, _obj| 1.0)
    }

    /// Build with a custom per-edge weight function.
    ///
    /// The function receives `(predicate, object_iri)` and returns the weight
    /// for that edge.  When the same `(subject, object)` pair appears multiple
    /// times the weights are summed.
    pub fn from_triples_weighted(
        triples: &[(String, String, String)],
        weight_fn: impl Fn(&str, &str) -> f64,
    ) -> Self {
        let mut node_to_id: HashMap<String, NodeId> = HashMap::new();
        let mut id_to_node: Vec<String> = Vec::new();

        // ── pass 1: collect all IRI nodes ──────────────────────────────────
        for (subj, _pred, obj) in triples {
            if !node_to_id.contains_key(subj) {
                node_to_id.insert(subj.clone(), id_to_node.len());
                id_to_node.push(subj.clone());
            }
            // Skip literals (cheap heuristic: literals start with '"' or contain
            // no ':' – but IRIs are assumed to be non-empty strings that don't
            // start with '"').
            if !obj.starts_with('"') && obj.contains(':') && !node_to_id.contains_key(obj) {
                node_to_id.insert(obj.clone(), id_to_node.len());
                id_to_node.push(obj.clone());
            }
        }

        let n = id_to_node.len();
        let mut adjacency: Vec<Vec<(NodeId, EdgeWeight)>> = vec![Vec::new(); n];
        let mut reverse_adjacency: Vec<Vec<(NodeId, EdgeWeight)>> = vec![Vec::new(); n];
        let mut edge_count_internal = 0usize;

        // ── pass 2: build adjacency lists ──────────────────────────────────
        // We use a temp map to accumulate weights for parallel (s,o) edges.
        // Key: (src, dst), value: index into adjacency[src].
        let mut edge_index: HashMap<(NodeId, NodeId), usize> = HashMap::new();

        for (subj, pred, obj) in triples {
            let src = match node_to_id.get(subj) {
                Some(&id) => id,
                None => continue,
            };
            let dst = match node_to_id.get(obj) {
                Some(&id) => id,
                None => continue, // literal – skip
            };

            let w = weight_fn(pred, obj);

            let key = (src, dst);
            if let Some(&idx) = edge_index.get(&key) {
                // Accumulate weight on existing edge
                adjacency[src][idx].1 += w;
                // Find the reverse entry and update it too
                if let Some(rev_entry) = reverse_adjacency[dst].iter_mut().find(|(u, _)| *u == src)
                {
                    rev_entry.1 += w;
                }
            } else {
                edge_index.insert(key, adjacency[src].len());
                adjacency[src].push((dst, w));
                reverse_adjacency[dst].push((src, w));
                edge_count_internal += 1;
            }
        }

        Self {
            node_to_id,
            id_to_node,
            adjacency,
            reverse_adjacency,
            edge_count_internal,
        }
    }

    /// Number of unique IRI nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.id_to_node.len()
    }

    /// Number of unique directed edges.
    pub fn edge_count(&self) -> usize {
        self.edge_count_internal
    }

    /// IRI string for a node id, or `None` if out of range.
    pub fn get_node_iri(&self, id: NodeId) -> Option<&str> {
        self.id_to_node.get(id).map(|s| s.as_str())
    }

    /// Node id for an IRI string, or `None` if not present.
    pub fn get_node_id(&self, iri: &str) -> Option<NodeId> {
        self.node_to_id.get(iri).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(s: &str, p: &str, o: &str) -> (String, String, String) {
        (s.to_string(), p.to_string(), o.to_string())
    }

    #[test]
    fn test_empty_graph() {
        let g = RdfGraphAdapter::from_triples(&[]);
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_single_triple() {
        let triples = vec![triple("ex:A", "ex:rel", "ex:B")];
        let g = RdfGraphAdapter::from_triples(&triples);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_literal_object_ignored() {
        let triples = vec![
            triple("ex:A", "ex:name", "\"Alice\""),
            triple("ex:A", "ex:rel", "ex:B"),
        ];
        let g = RdfGraphAdapter::from_triples(&triples);
        // "Alice" is a literal – should not be a node
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_node_lookup() {
        let triples = vec![triple("ex:A", "ex:rel", "ex:B")];
        let g = RdfGraphAdapter::from_triples(&triples);
        let id_a = g.get_node_id("ex:A").expect("A must exist");
        let id_b = g.get_node_id("ex:B").expect("B must exist");
        assert_eq!(g.get_node_iri(id_a), Some("ex:A"));
        assert_eq!(g.get_node_iri(id_b), Some("ex:B"));
        assert!(g.get_node_id("ex:C").is_none());
        assert!(g.get_node_iri(999).is_none());
    }

    #[test]
    fn test_duplicate_edges_accumulated() {
        let triples = vec![
            triple("ex:A", "ex:rel", "ex:B"),
            triple("ex:A", "ex:rel2", "ex:B"),
        ];
        let g = RdfGraphAdapter::from_triples(&triples);
        // Same (A→B) pair – should be one edge with accumulated weight
        assert_eq!(g.edge_count(), 1);
        let id_a = g.get_node_id("ex:A").unwrap();
        assert_eq!(g.adjacency[id_a].len(), 1);
        assert!((g.adjacency[id_a][0].1 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_reverse_adjacency() {
        let triples = vec![
            triple("ex:A", "ex:rel", "ex:B"),
            triple("ex:C", "ex:rel", "ex:B"),
        ];
        let g = RdfGraphAdapter::from_triples(&triples);
        let id_b = g.get_node_id("ex:B").unwrap();
        assert_eq!(g.reverse_adjacency[id_b].len(), 2);
    }

    #[test]
    fn test_weighted_triples() {
        let triples = vec![triple("ex:A", "ex:heavy", "ex:B")];
        let g = RdfGraphAdapter::from_triples_weighted(&triples, |pred, _| {
            if pred == "ex:heavy" {
                5.0
            } else {
                1.0
            }
        });
        let id_a = g.get_node_id("ex:A").unwrap();
        assert!((g.adjacency[id_a][0].1 - 5.0).abs() < 1e-9);
    }
}
