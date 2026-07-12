//! RDFS subclass-closure helper shared across SHACL-AF modules.

use oxirs_core::{
    model::{NamedNode, Object, Predicate, Subject},
    Store,
};
use std::collections::{HashMap, HashSet};

use crate::ShaclError;

const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";

/// Build a subclass→superclasses adjacency map from every `rdfs:subClassOf` triple
/// in the store, then compute the Floyd–Warshall transitive closure.
///
/// The returned map has the property: for every pair (A, B) where A is a subclass
/// of B (directly or transitively), `result[A]` contains `B`.
pub(crate) fn build_rdfs_subclass_closure(
    store: &dyn Store,
) -> crate::Result<HashMap<NamedNode, HashSet<NamedNode>>> {
    let rdfs_subclass_of = NamedNode::new_unchecked(RDFS_SUBCLASS_OF);
    let graph = build_relation_graph(store, &rdfs_subclass_of)?;
    compute_transitive_closure(&graph)
}

/// Build a named-node adjacency graph for any transitive predicate.
pub(crate) fn build_relation_graph(
    store: &dyn Store,
    predicate: &NamedNode,
) -> crate::Result<HashMap<NamedNode, HashSet<NamedNode>>> {
    let mut graph: HashMap<NamedNode, HashSet<NamedNode>> = HashMap::new();
    let predicate_term = Predicate::from(predicate.clone());
    let quads = store
        .find_quads(None, Some(&predicate_term), None, None)
        .map_err(|e| ShaclError::TargetSelection(format!("Failed to query store: {e}")))?;
    for quad in quads {
        if let (Subject::NamedNode(subject), Object::NamedNode(object)) =
            (quad.subject().clone(), quad.object().clone())
        {
            graph.entry(subject).or_default().insert(object);
        }
    }
    Ok(graph)
}

/// Floyd–Warshall transitive closure over a named-node adjacency graph.
///
/// Each node is included in its own reachable set (reflexive closure).
pub(crate) fn compute_transitive_closure(
    graph: &HashMap<NamedNode, HashSet<NamedNode>>,
) -> crate::Result<HashMap<NamedNode, HashSet<NamedNode>>> {
    if graph.is_empty() {
        return Ok(HashMap::new());
    }

    // Collect all nodes (both sources and targets).
    let mut node_set: HashSet<&NamedNode> = HashSet::new();
    for (from_node, to_nodes) in graph {
        node_set.insert(from_node);
        for to in to_nodes {
            node_set.insert(to);
        }
    }
    let nodes: Vec<&NamedNode> = node_set.into_iter().collect();
    let node_to_idx: HashMap<&NamedNode, usize> =
        nodes.iter().enumerate().map(|(i, n)| (*n, i)).collect();

    let n = nodes.len();
    // Use Vec<Vec<bool>> for the adjacency/closure matrix — no ndarray dependency needed.
    let mut adj: Vec<Vec<bool>> = vec![vec![false; n]; n];

    for (from, tos) in graph {
        if let Some(&fi) = node_to_idx.get(from) {
            for to in tos {
                if let Some(&ti) = node_to_idx.get(to) {
                    adj[fi][ti] = true;
                }
            }
        }
    }

    // Floyd–Warshall: reflexive + transitive closure.
    // Phase 1: mark reflexive diagonal using enumerated iterator.
    for (diag, row) in adj.iter_mut().enumerate().take(n) {
        row[diag] = true;
    }
    // Phase 2: transitive closure.
    // For each intermediary k, snapshot row k *at this point in the iteration*
    // (Floyd-Warshall correctness), then propagate reachability.
    for k in 0..n {
        let row_k: Vec<bool> = adj[k].clone();
        for row_i in adj.iter_mut().take(n) {
            if row_i[k] {
                for (j, &reach) in row_k.iter().enumerate() {
                    if reach {
                        row_i[j] = true;
                    }
                }
            }
        }
    }

    let mut result: HashMap<NamedNode, HashSet<NamedNode>> = HashMap::new();
    for (i, from) in nodes.iter().enumerate() {
        let mut reachable = HashSet::new();
        for (j, to) in nodes.iter().enumerate() {
            if adj[i][j] {
                reachable.insert((*to).clone());
            }
        }
        if !reachable.is_empty() {
            result.insert((*from).clone(), reachable);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::ConcreteStore;

    #[test]
    fn test_build_rdfs_subclass_closure_empty_store() {
        let store = ConcreteStore::new().expect("store creation");
        let closure = build_rdfs_subclass_closure(&store).expect("closure should succeed");
        assert!(
            closure.is_empty(),
            "Empty store should produce empty closure"
        );
    }

    #[test]
    fn test_subclass_closure_transitive() {
        use oxirs_core::model::{GraphName, Object, Predicate, Quad, Subject};
        // A rdfs:subClassOf B, B rdfs:subClassOf C => A rdfs:subClassOf C (closure)
        let store = ConcreteStore::new().expect("store creation");
        let a = NamedNode::new_unchecked("http://example.org/A");
        let b = NamedNode::new_unchecked("http://example.org/B");
        let c = NamedNode::new_unchecked("http://example.org/C");
        let sco = NamedNode::new_unchecked(RDFS_SUBCLASS_OF);

        store
            .insert(&Quad::new(
                Subject::NamedNode(a.clone()),
                Predicate::NamedNode(sco.clone()),
                Object::NamedNode(b.clone()),
                GraphName::DefaultGraph,
            ))
            .expect("insert A subClassOf B");
        store
            .insert(&Quad::new(
                Subject::NamedNode(b.clone()),
                Predicate::NamedNode(sco.clone()),
                Object::NamedNode(c.clone()),
                GraphName::DefaultGraph,
            ))
            .expect("insert B subClassOf C");

        let closure = build_rdfs_subclass_closure(&store).expect("closure");
        // A should reach C transitively
        assert!(
            closure.get(&a).map(|s| s.contains(&c)).unwrap_or(false),
            "A should transitively reach C"
        );
        // B should reach C directly
        assert!(
            closure.get(&b).map(|s| s.contains(&c)).unwrap_or(false),
            "B should reach C"
        );
    }

    #[test]
    fn test_subclass_closure_reflexive() {
        use oxirs_core::model::{GraphName, Object, Predicate, Quad, Subject};
        let store = ConcreteStore::new().expect("store creation");
        let a = NamedNode::new_unchecked("http://example.org/A");
        let b = NamedNode::new_unchecked("http://example.org/B");
        let sco = NamedNode::new_unchecked(RDFS_SUBCLASS_OF);

        store
            .insert(&Quad::new(
                Subject::NamedNode(a.clone()),
                Predicate::NamedNode(sco),
                Object::NamedNode(b.clone()),
                GraphName::DefaultGraph,
            ))
            .expect("insert");

        let closure = build_rdfs_subclass_closure(&store).expect("closure");
        // Reflexive: A reaches A, B reaches B
        assert!(
            closure.get(&a).map(|s| s.contains(&a)).unwrap_or(false),
            "A should be reflexively reachable"
        );
        assert!(
            closure.get(&b).map(|s| s.contains(&b)).unwrap_or(false),
            "B should be reflexively reachable"
        );
    }
}
