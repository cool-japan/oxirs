//! Graph traversal for RDF-star nested triples.
//!
//! `StarGraph` stores `StarTriple` values where subjects/objects can themselves
//! be quoted triples, enabling arbitrary nesting. BFS and DFS walkers traverse
//! the graph using the IRI labels of non-triple nodes.

use std::collections::{HashSet, VecDeque};

/// A node in an RDF-star graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StarNode {
    /// An IRI node.
    Iri(String),
    /// A plain literal node.
    Literal(String),
    /// A blank node.
    Blank(String),
    /// A quoted (nested) triple.
    Triple(Box<StarTriple>),
}

/// A triple in an RDF-star graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StarTriple {
    /// Subject — may be a quoted triple.
    pub subject: StarNode,
    /// Predicate IRI.
    pub predicate: String,
    /// Object — may be a quoted triple.
    pub object: StarNode,
}

impl StarTriple {
    /// Construct a triple.
    pub fn new(subject: StarNode, predicate: impl Into<String>, object: StarNode) -> Self {
        Self { subject, predicate: predicate.into(), object }
    }
}

/// Result of a walk step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkResult {
    /// Sequence of node IRI labels along the path.
    pub path: Vec<String>,
    /// Depth at which the result node was discovered.
    pub depth: usize,
}

/// An in-memory RDF-star graph.
#[derive(Debug, Default)]
pub struct StarGraph {
    triples: Vec<StarTriple>,
}

impl StarGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the graph.
    pub fn add(&mut self, triple: StarTriple) {
        self.triples.push(triple);
    }

    /// Number of triples.
    pub fn count(&self) -> usize {
        self.triples.len()
    }

    /// All triples where the subject resolves to `node_iri`.
    pub fn outgoing(&self, node_iri: &str) -> Vec<&StarTriple> {
        self.triples
            .iter()
            .filter(|t| node_iri_of(&t.subject) == Some(node_iri))
            .collect()
    }

    /// All triples where the object resolves to `node_iri`.
    pub fn incoming(&self, node_iri: &str) -> Vec<&StarTriple> {
        self.triples
            .iter()
            .filter(|t| node_iri_of(&t.object) == Some(node_iri))
            .collect()
    }

    /// Breadth-first walk from `start`, up to `max_depth` hops.
    ///
    /// Each `WalkResult` records the path from `start` to a discovered node.
    pub fn walk_bfs(&self, start: &str, max_depth: usize) -> Vec<WalkResult> {
        let mut results = Vec::new();
        // Queue entries: (current_iri, path_so_far, depth)
        let mut queue: VecDeque<(String, Vec<String>, usize)> = VecDeque::new();
        let mut visited: HashSet<String> = HashSet::new();

        queue.push_back((start.to_owned(), vec![start.to_owned()], 0));
        visited.insert(start.to_owned());

        while let Some((current, path, depth)) = queue.pop_front() {
            if depth > 0 {
                results.push(WalkResult { path: path.clone(), depth });
            }
            if depth >= max_depth {
                continue;
            }
            for triple in self.outgoing(&current) {
                if let Some(next) = node_iri_of(&triple.object) {
                    if visited.insert(next.to_owned()) {
                        let mut new_path = path.clone();
                        new_path.push(next.to_owned());
                        queue.push_back((next.to_owned(), new_path, depth + 1));
                    }
                }
            }
        }
        results
    }

    /// Depth-first walk from `start`, up to `max_depth` hops.
    pub fn walk_dfs(&self, start: &str, max_depth: usize) -> Vec<WalkResult> {
        let mut results = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start.to_owned());
        self.dfs_recurse(start, &[start.to_owned()], 0, max_depth, &mut visited, &mut results);
        results
    }

    fn dfs_recurse(
        &self,
        current: &str,
        path: &[String],
        depth: usize,
        max_depth: usize,
        visited: &mut HashSet<String>,
        results: &mut Vec<WalkResult>,
    ) {
        for triple in self.outgoing(current) {
            if let Some(next) = node_iri_of(&triple.object) {
                if visited.insert(next.to_owned()) {
                    let mut new_path = path.to_vec();
                    new_path.push(next.to_owned());
                    let new_depth = depth + 1;
                    results.push(WalkResult { path: new_path.clone(), depth: new_depth });
                    if new_depth < max_depth {
                        self.dfs_recurse(next, &new_path, new_depth, max_depth, visited, results);
                    }
                    visited.remove(next);
                }
            }
        }
    }

    /// All distinct subject IRIs in the graph.
    pub fn all_subjects(&self) -> Vec<String> {
        let mut subjects: Vec<String> = self
            .triples
            .iter()
            .flat_map(|t| flatten_node(&t.subject))
            .collect();
        subjects.sort();
        subjects.dedup();
        subjects
    }

    /// All distinct predicate IRIs in the graph.
    pub fn all_predicates(&self) -> Vec<String> {
        let mut predicates: Vec<String> = self.triples.iter().map(|t| t.predicate.clone()).collect();
        predicates.sort();
        predicates.dedup();
        predicates
    }

    /// Triples that have a `StarNode::Triple` as subject or object.
    pub fn nested_triples(&self) -> Vec<&StarTriple> {
        self.triples
            .iter()
            .filter(|t| {
                matches!(t.subject, StarNode::Triple(_)) || matches!(t.object, StarNode::Triple(_))
            })
            .collect()
    }

    /// Maximum nesting depth across all triples (0 = no nesting).
    pub fn max_nesting_depth(&self) -> usize {
        self.triples
            .iter()
            .map(|t| {
                let sd = node_depth(&t.subject);
                let od = node_depth(&t.object);
                sd.max(od)
            })
            .max()
            .unwrap_or(0)
    }
}

// ── private helpers ───────────────────────────────────────────────────────────

/// Return the IRI label of a non-triple node, if it has one.
fn node_iri_of(node: &StarNode) -> Option<&str> {
    match node {
        StarNode::Iri(s) => Some(s.as_str()),
        StarNode::Blank(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Collect all IRI-like strings reachable from `node` (flattens nested triples).
fn flatten_node(node: &StarNode) -> Vec<String> {
    match node {
        StarNode::Iri(s) | StarNode::Blank(s) => vec![s.clone()],
        StarNode::Literal(_) => Vec::new(),
        StarNode::Triple(t) => {
            let mut v = flatten_node(&t.subject);
            v.push(t.predicate.clone());
            v.extend(flatten_node(&t.object));
            v
        }
    }
}

/// Nesting depth of a node: IRI/Literal/Blank = 0; Triple = 1 + max of children.
fn node_depth(node: &StarNode) -> usize {
    match node {
        StarNode::Iri(_) | StarNode::Literal(_) | StarNode::Blank(_) => 0,
        StarNode::Triple(t) => {
            1 + node_depth(&t.subject).max(node_depth(&t.object))
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn iri(s: &str) -> StarNode {
        StarNode::Iri(s.to_string())
    }

    fn lit(s: &str) -> StarNode {
        StarNode::Literal(s.to_string())
    }

    fn blank(s: &str) -> StarNode {
        StarNode::Blank(s.to_string())
    }

    fn t(s: StarNode, p: &str, o: StarNode) -> StarTriple {
        StarTriple::new(s, p, o)
    }

    // ── StarGraph::new / add / count ─────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = StarGraph::new();
        assert_eq!(g.count(), 0);
    }

    #[test]
    fn test_add_triple() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        assert_eq!(g.count(), 1);
    }

    #[test]
    fn test_add_multiple() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        g.add(t(iri("b"), "q", iri("c")));
        assert_eq!(g.count(), 2);
    }

    // ── outgoing / incoming ──────────────────────────────────────────────────

    #[test]
    fn test_outgoing_basic() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        let out = g.outgoing("a");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].predicate, "p");
    }

    #[test]
    fn test_outgoing_none() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        assert!(g.outgoing("b").is_empty());
    }

    #[test]
    fn test_incoming_basic() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        let inc = g.incoming("b");
        assert_eq!(inc.len(), 1);
    }

    #[test]
    fn test_incoming_none() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        assert!(g.incoming("a").is_empty());
    }

    // ── walk_bfs ─────────────────────────────────────────────────────────────

    #[test]
    fn test_bfs_single_hop() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        let results = g.walk_bfs("a", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].depth, 1);
        assert_eq!(results[0].path, vec!["a", "b"]);
    }

    #[test]
    fn test_bfs_two_hops() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        g.add(t(iri("b"), "q", iri("c")));
        let results = g.walk_bfs("a", 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bfs_max_depth_limit() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        g.add(t(iri("b"), "q", iri("c")));
        let results = g.walk_bfs("a", 1);
        // Only b reachable at depth 1
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path.last().unwrap(), "b");
    }

    #[test]
    fn test_bfs_empty_graph() {
        let g = StarGraph::new();
        let results = g.walk_bfs("a", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bfs_no_cycle_infinite() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        g.add(t(iri("b"), "p", iri("a"))); // cycle
        let results = g.walk_bfs("a", 10);
        // Should not loop forever; a and b both visited once.
        assert!(results.len() <= 2);
    }

    // ── walk_dfs ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dfs_single_hop() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        let results = g.walk_dfs("a", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].depth, 1);
    }

    #[test]
    fn test_dfs_two_hops() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        g.add(t(iri("b"), "q", iri("c")));
        let results = g.walk_dfs("a", 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_dfs_empty_graph() {
        let g = StarGraph::new();
        let results = g.walk_dfs("a", 5);
        assert!(results.is_empty());
    }

    // ── all_subjects ─────────────────────────────────────────────────────────

    #[test]
    fn test_all_subjects_basic() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        g.add(t(iri("b"), "q", iri("c")));
        let subs = g.all_subjects();
        assert!(subs.contains(&"a".to_string()));
        assert!(subs.contains(&"b".to_string()));
    }

    #[test]
    fn test_all_subjects_dedup() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        g.add(t(iri("a"), "q", iri("c")));
        let subs = g.all_subjects();
        assert_eq!(subs.iter().filter(|s| s.as_str() == "a").count(), 1);
    }

    // ── all_predicates ───────────────────────────────────────────────────────

    #[test]
    fn test_all_predicates() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        g.add(t(iri("b"), "q", iri("c")));
        let preds = g.all_predicates();
        assert!(preds.contains(&"p".to_string()));
        assert!(preds.contains(&"q".to_string()));
    }

    #[test]
    fn test_all_predicates_dedup() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        g.add(t(iri("c"), "p", iri("d")));
        let preds = g.all_predicates();
        assert_eq!(preds.len(), 1);
    }

    // ── nested_triples ───────────────────────────────────────────────────────

    #[test]
    fn test_nested_triples_detected() {
        let mut g = StarGraph::new();
        let inner = t(iri("s"), "p", iri("o"));
        g.add(t(StarNode::Triple(Box::new(inner)), "cert", iri("high")));
        assert_eq!(g.nested_triples().len(), 1);
    }

    #[test]
    fn test_nested_triples_none() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        assert!(g.nested_triples().is_empty());
    }

    #[test]
    fn test_nested_in_object() {
        let mut g = StarGraph::new();
        let inner = t(iri("x"), "y", iri("z"));
        g.add(t(iri("src"), "ref", StarNode::Triple(Box::new(inner))));
        assert_eq!(g.nested_triples().len(), 1);
    }

    // ── max_nesting_depth ────────────────────────────────────────────────────

    #[test]
    fn test_max_depth_no_nesting() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", iri("b")));
        assert_eq!(g.max_nesting_depth(), 0);
    }

    #[test]
    fn test_max_depth_one_level() {
        let mut g = StarGraph::new();
        let inner = t(iri("s"), "p", iri("o"));
        g.add(t(StarNode::Triple(Box::new(inner)), "q", iri("x")));
        assert_eq!(g.max_nesting_depth(), 1);
    }

    #[test]
    fn test_max_depth_two_levels() {
        let inner1 = t(iri("a"), "b", iri("c"));
        let inner2 = t(StarNode::Triple(Box::new(inner1)), "d", iri("e"));
        let mut g = StarGraph::new();
        g.add(t(StarNode::Triple(Box::new(inner2)), "f", iri("g")));
        assert_eq!(g.max_nesting_depth(), 2);
    }

    #[test]
    fn test_max_depth_empty_graph() {
        let g = StarGraph::new();
        assert_eq!(g.max_nesting_depth(), 0);
    }

    // ── literal / blank nodes ─────────────────────────────────────────────────

    #[test]
    fn test_literal_node_not_walkable() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "label", lit("hello")));
        let out = g.outgoing("a");
        assert_eq!(out.len(), 1); // triple recorded
        // Walking: literal has no IRI so won't be followed.
        let results = g.walk_bfs("a", 2);
        // a -> literal: literal has no IRI so walk stops.
        assert!(results.is_empty());
    }

    #[test]
    fn test_blank_node_walkable() {
        let mut g = StarGraph::new();
        g.add(t(iri("a"), "p", blank("_b1")));
        g.add(t(blank("_b1"), "q", iri("c")));
        let results = g.walk_bfs("a", 2);
        assert_eq!(results.len(), 2);
    }

    // ── WalkResult ───────────────────────────────────────────────────────────

    #[test]
    fn test_walk_result_equality() {
        let r1 = WalkResult { path: vec!["a".to_string(), "b".to_string()], depth: 1 };
        let r2 = WalkResult { path: vec!["a".to_string(), "b".to_string()], depth: 1 };
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_star_triple_equality() {
        let t1 = t(iri("s"), "p", iri("o"));
        let t2 = t(iri("s"), "p", iri("o"));
        assert_eq!(t1, t2);
    }
}
