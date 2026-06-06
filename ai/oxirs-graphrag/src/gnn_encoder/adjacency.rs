//! Adjacency graph representation for GNN-based entity encoding over RDF triples.
//!
//! Converts a list of RDF triples into an indexed adjacency structure suitable
//! for message-passing neural network computations.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A list of (subject, predicate, object) triples in string form.
pub type EdgeList = Vec<(String, String, String)>;

/// Adjacency graph built from RDF triples.
///
/// Each unique subject or object string is mapped to a dense integer index.
/// The adjacency list `adjacency[i]` stores the indices of all nodes reachable
/// from node `i` by following any predicate.  Edge features encode a simple
/// hash of the predicate string, normalised to [0, 1].
#[derive(Debug, Clone)]
pub struct AdjacencyGraph {
    /// Map from entity string to its integer index
    pub entity_to_idx: HashMap<String, usize>,
    /// Inverse: index → entity string
    pub idx_to_entity: Vec<String>,
    /// `adjacency[i]` = list of neighbour indices for node `i`
    pub adjacency: Vec<Vec<usize>>,
    /// `edge_features[i]` = list of scalar features for each edge leaving node `i`
    /// (one-to-one correspondence with `adjacency[i]`)
    pub edge_features: Vec<Vec<f64>>,
}

impl AdjacencyGraph {
    /// Build an `AdjacencyGraph` from a slice of (subject, predicate, object) triples.
    ///
    /// Duplicate entities are deduplicated; duplicate edges are preserved (they
    /// represent multiple predicates between the same entity pair).
    pub fn from_triples(triples: &[(String, String, String)]) -> Self {
        let mut entity_to_idx: HashMap<String, usize> = HashMap::new();
        let mut idx_to_entity: Vec<String> = Vec::new();

        // First pass: collect all unique entities
        let ensure_entity = |name: &str,
                             entity_to_idx: &mut HashMap<String, usize>,
                             idx_to_entity: &mut Vec<String>|
         -> usize {
            if let Some(&idx) = entity_to_idx.get(name) {
                return idx;
            }
            let idx = idx_to_entity.len();
            entity_to_idx.insert(name.to_string(), idx);
            idx_to_entity.push(name.to_string());
            idx
        };

        for (s, _p, o) in triples {
            ensure_entity(s, &mut entity_to_idx, &mut idx_to_entity);
            ensure_entity(o, &mut entity_to_idx, &mut idx_to_entity);
        }

        let n = idx_to_entity.len();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut edge_features: Vec<Vec<f64>> = vec![Vec::new(); n];

        // Second pass: populate edges
        for (s, p, o) in triples {
            let si = *entity_to_idx
                .get(s.as_str())
                .expect("entity must exist after first pass");
            let oi = *entity_to_idx
                .get(o.as_str())
                .expect("entity must exist after first pass");

            adjacency[si].push(oi);
            edge_features[si].push(predicate_feature(p));

            // Undirected: also add reverse edge
            if si != oi {
                adjacency[oi].push(si);
                edge_features[oi].push(predicate_feature(p));
            }
        }

        Self {
            entity_to_idx,
            idx_to_entity,
            adjacency,
            edge_features,
        }
    }

    /// Return the neighbour indices of node `idx`.
    /// Returns an empty slice for out-of-range indices.
    pub fn neighbors(&self, idx: usize) -> &[usize] {
        if idx < self.adjacency.len() {
            &self.adjacency[idx]
        } else {
            &[]
        }
    }

    /// Return the total number of unique entities (nodes) in the graph.
    pub fn entity_count(&self) -> usize {
        self.idx_to_entity.len()
    }

    /// Look up the string label for a given node index.
    /// Returns `None` if the index is out of range.
    pub fn entity_name(&self, idx: usize) -> Option<&str> {
        self.idx_to_entity.get(idx).map(|s| s.as_str())
    }

    /// Look up the integer index for a given entity string.
    /// Returns `None` if the entity is not in the graph.
    pub fn entity_index(&self, name: &str) -> Option<usize> {
        self.entity_to_idx.get(name).copied()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic scalar feature derived from a predicate string.
/// Uses a simple polynomial hash, then maps to [0, 1] via modular arithmetic.
fn predicate_feature(predicate: &str) -> f64 {
    let hash: u64 = predicate
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    // Map to [0, 1]
    (hash % 100_007) as f64 / 100_007.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: &str) -> String {
        x.to_string()
    }

    #[test]
    fn test_empty_graph() {
        let g = AdjacencyGraph::from_triples(&[]);
        assert_eq!(g.entity_count(), 0);
        assert!(g.neighbors(0).is_empty());
    }

    #[test]
    fn test_single_triple() {
        let triples = vec![(s("Alice"), s("knows"), s("Bob"))];
        let g = AdjacencyGraph::from_triples(&triples);
        assert_eq!(g.entity_count(), 2);
        // Each node should have one neighbour (undirected)
        let alice_idx = g.entity_index("Alice").expect("Alice must be present");
        let bob_idx = g.entity_index("Bob").expect("Bob must be present");
        assert_eq!(g.neighbors(alice_idx), &[bob_idx]);
        assert_eq!(g.neighbors(bob_idx), &[alice_idx]);
    }

    #[test]
    fn test_entity_deduplication() {
        let triples = vec![
            (s("Alice"), s("knows"), s("Bob")),
            (s("Alice"), s("worksAt"), s("Acme")),
            (s("Bob"), s("worksAt"), s("Acme")),
        ];
        let g = AdjacencyGraph::from_triples(&triples);
        // Alice, Bob, Acme — three unique entities
        assert_eq!(g.entity_count(), 3);
    }

    #[test]
    fn test_neighbor_lookup() {
        let triples = vec![(s("A"), s("r"), s("B")), (s("A"), s("r"), s("C"))];
        let g = AdjacencyGraph::from_triples(&triples);
        let a = g.entity_index("A").expect("A present");
        let neighbors = g.neighbors(a);
        assert_eq!(neighbors.len(), 2);
        let b = g.entity_index("B").expect("B present");
        let c = g.entity_index("C").expect("C present");
        assert!(neighbors.contains(&b));
        assert!(neighbors.contains(&c));
    }

    #[test]
    fn test_round_trip_entity_names() {
        let triples = vec![(s("X"), s("p"), s("Y"))];
        let g = AdjacencyGraph::from_triples(&triples);
        for (name, &idx) in &g.entity_to_idx {
            assert_eq!(g.entity_name(idx), Some(name.as_str()));
            assert_eq!(g.entity_index(name), Some(idx));
        }
    }
}
