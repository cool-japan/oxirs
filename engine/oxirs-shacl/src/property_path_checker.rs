/// SHACL property path constraint checking.
///
/// Implements evaluation of SPARQL/SHACL property paths over an in-memory
/// triple graph, supporting the full set of path operators defined in the
/// SPARQL 1.1 specification and reused by SHACL.
use std::collections::{HashSet, VecDeque};

// ── Data structures ───────────────────────────────────────────────────────────

/// A triple in the RDF graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub s: String,
    pub p: String,
    pub o: String,
}

/// A SHACL/SPARQL property path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyPath {
    /// A single predicate IRI.
    Predicate(String),
    /// `path1 / path2` — sequence of two paths.
    Sequence(Vec<PropertyPath>),
    /// `path1 | path2` — alternative paths (union).
    Alternative(Vec<PropertyPath>),
    /// `^path` — inverse path.
    InversePath(Box<PropertyPath>),
    /// `path*` — zero or more repetitions.
    ZeroOrMore(Box<PropertyPath>),
    /// `path+` — one or more repetitions.
    OneOrMore(Box<PropertyPath>),
    /// `path?` — zero or one repetition.
    ZeroOrOne(Box<PropertyPath>),
    /// `!pred` — negated predicate (matches any predicate except the given one).
    NegatedPredicate(String),
}

/// The result of evaluating a path from a start node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathResult {
    /// All nodes reachable from the start via the path.
    pub reachable: Vec<String>,
    /// Total number of path steps traversed (may count duplicates for repetition paths).
    pub path_count: usize,
}

// ── PropertyPathChecker ───────────────────────────────────────────────────────

/// Evaluates SHACL property paths over an in-memory RDF graph.
pub struct PropertyPathChecker;

impl PropertyPathChecker {
    /// Evaluate a `path` starting from `start` over `graph`.
    pub fn evaluate(path: &PropertyPath, start: &str, graph: &[Triple]) -> PathResult {
        let nodes = eval_path(path, start, graph);
        let path_count = nodes.len();
        PathResult {
            reachable: nodes.into_iter().collect(),
            path_count,
        }
    }

    /// Evaluate a path from multiple start nodes.
    pub fn evaluate_multi<'a>(
        path: &PropertyPath,
        starts: &[&'a str],
        graph: &[Triple],
    ) -> Vec<(&'a str, PathResult)> {
        starts
            .iter()
            .map(|&s| {
                let r = Self::evaluate(path, s, graph);
                (s, r)
            })
            .collect()
    }

    /// Maximum nesting depth of a path expression.
    pub fn path_depth(path: &PropertyPath) -> usize {
        match path {
            PropertyPath::Predicate(_) | PropertyPath::NegatedPredicate(_) => 0,
            PropertyPath::InversePath(inner)
            | PropertyPath::ZeroOrMore(inner)
            | PropertyPath::OneOrMore(inner)
            | PropertyPath::ZeroOrOne(inner) => 1 + Self::path_depth(inner),
            PropertyPath::Sequence(paths) | PropertyPath::Alternative(paths) => {
                paths.iter().map(Self::path_depth).max().unwrap_or(0) + 1
            }
        }
    }

    /// Count the number of predicate atoms (leaf predicates) in a path.
    pub fn count_predicates(path: &PropertyPath) -> usize {
        match path {
            PropertyPath::Predicate(_) | PropertyPath::NegatedPredicate(_) => 1,
            PropertyPath::InversePath(inner)
            | PropertyPath::ZeroOrMore(inner)
            | PropertyPath::OneOrMore(inner)
            | PropertyPath::ZeroOrOne(inner) => Self::count_predicates(inner),
            PropertyPath::Sequence(paths) | PropertyPath::Alternative(paths) => {
                paths.iter().map(Self::count_predicates).sum()
            }
        }
    }

    /// Returns `true` when the path contains no repetition operators
    /// (`*`, `+`, `?`).
    pub fn is_simple(path: &PropertyPath) -> bool {
        match path {
            PropertyPath::Predicate(_) | PropertyPath::NegatedPredicate(_) => true,
            PropertyPath::InversePath(inner) => Self::is_simple(inner),
            PropertyPath::ZeroOrMore(_)
            | PropertyPath::OneOrMore(_)
            | PropertyPath::ZeroOrOne(_) => false,
            PropertyPath::Sequence(paths) | PropertyPath::Alternative(paths) => {
                paths.iter().all(Self::is_simple)
            }
        }
    }
}

// ── Private evaluation helpers ────────────────────────────────────────────────

/// Evaluate a path from `start`, returning all reachable object nodes.
fn eval_path(path: &PropertyPath, start: &str, graph: &[Triple]) -> HashSet<String> {
    match path {
        PropertyPath::Predicate(pred) => predicate_step(pred, start, graph).into_iter().collect(),
        PropertyPath::NegatedPredicate(negated) => graph
            .iter()
            .filter(|t| t.s == start && t.p != *negated)
            .map(|t| t.o.clone())
            .collect(),
        PropertyPath::InversePath(inner) => {
            // Evaluate the inner path in reverse: treat `graph` edges backwards.
            let reversed = reverse_graph(graph);
            eval_path(inner, start, &reversed)
        }
        PropertyPath::Sequence(steps) => {
            let mut current: HashSet<String> = std::iter::once(start.to_string()).collect();
            for step in steps {
                let mut next: HashSet<String> = HashSet::new();
                for node in &current {
                    let reachable = eval_path(step, node, graph);
                    next.extend(reachable);
                }
                current = next;
            }
            current
        }
        PropertyPath::Alternative(alts) => {
            let mut result: HashSet<String> = HashSet::new();
            for alt in alts {
                result.extend(eval_path(alt, start, graph));
            }
            result
        }
        PropertyPath::ZeroOrMore(inner) => {
            // BFS: start node itself (zero steps) plus all reachable via inner*
            transitive_closure(inner, start, graph, true)
        }
        PropertyPath::OneOrMore(inner) => {
            // At least one step via inner, then transitive closure
            transitive_closure(inner, start, graph, false)
        }
        PropertyPath::ZeroOrOne(inner) => {
            let mut result: HashSet<String> = std::iter::once(start.to_string()).collect();
            result.extend(eval_path(inner, start, graph));
            result
        }
    }
}

/// BFS transitive closure over `path` from `start`.
/// If `include_start` is `true`, the start node itself is included (zero steps).
fn transitive_closure(
    path: &PropertyPath,
    start: &str,
    graph: &[Triple],
    include_start: bool,
) -> HashSet<String> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();

    if include_start {
        visited.insert(start.to_string());
    }

    // Seed the queue with the first step
    let first_step = eval_path(path, start, graph);
    for node in first_step {
        if visited.insert(node.clone()) {
            queue.push_back(node);
        }
    }

    while let Some(current) = queue.pop_front() {
        let next_nodes = eval_path(path, &current, graph);
        for node in next_nodes {
            if visited.insert(node.clone()) {
                queue.push_back(node);
            }
        }
    }

    visited
}

/// Follow a single predicate from `start` and return object nodes.
fn predicate_step(pred: &str, start: &str, graph: &[Triple]) -> Vec<String> {
    graph
        .iter()
        .filter(|t| t.s == start && t.p == pred)
        .map(|t| t.o.clone())
        .collect()
}

/// Follow a single predicate in reverse from `start` (i.e. `start` is the object).
pub fn inverse_step(pred: &str, start: &str, graph: &[Triple]) -> Vec<String> {
    graph
        .iter()
        .filter(|t| t.o == start && t.p == pred)
        .map(|t| t.s.clone())
        .collect()
}

/// Build a copy of `graph` with all edges reversed.
fn reverse_graph(graph: &[Triple]) -> Vec<Triple> {
    graph
        .iter()
        .map(|t| Triple {
            s: t.o.clone(),
            p: t.p.clone(),
            o: t.s.clone(),
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }

    fn pred(s: &str) -> PropertyPath {
        PropertyPath::Predicate(s.into())
    }

    fn seq(steps: Vec<PropertyPath>) -> PropertyPath {
        PropertyPath::Sequence(steps)
    }

    fn alt(alts: Vec<PropertyPath>) -> PropertyPath {
        PropertyPath::Alternative(alts)
    }

    fn inv(p: PropertyPath) -> PropertyPath {
        PropertyPath::InversePath(Box::new(p))
    }

    fn zero_or_more(p: PropertyPath) -> PropertyPath {
        PropertyPath::ZeroOrMore(Box::new(p))
    }

    fn one_or_more(p: PropertyPath) -> PropertyPath {
        PropertyPath::OneOrMore(Box::new(p))
    }

    fn zero_or_one(p: PropertyPath) -> PropertyPath {
        PropertyPath::ZeroOrOne(Box::new(p))
    }

    fn neg(s: &str) -> PropertyPath {
        PropertyPath::NegatedPredicate(s.into())
    }

    // Small helper: collect reachable as a sorted vec for deterministic asserts
    fn sorted_reachable(path: &PropertyPath, start: &str, graph: &[Triple]) -> Vec<String> {
        let mut r = PropertyPathChecker::evaluate(path, start, graph).reachable;
        r.sort();
        r
    }

    // ── Predicate step ────────────────────────────────────────────────────────

    #[test]
    fn test_predicate_single_hop() {
        let graph = vec![t("a", "p", "b")];
        let r = sorted_reachable(&pred("p"), "a", &graph);
        assert_eq!(r, vec!["b"]);
    }

    #[test]
    fn test_predicate_multiple_objects() {
        let graph = vec![t("a", "p", "b"), t("a", "p", "c")];
        let r = sorted_reachable(&pred("p"), "a", &graph);
        assert_eq!(r, vec!["b", "c"]);
    }

    #[test]
    fn test_predicate_no_match() {
        let graph = vec![t("a", "q", "b")];
        let r = sorted_reachable(&pred("p"), "a", &graph);
        assert!(r.is_empty());
    }

    #[test]
    fn test_predicate_different_start_no_match() {
        let graph = vec![t("a", "p", "b")];
        let r = sorted_reachable(&pred("p"), "x", &graph);
        assert!(r.is_empty());
    }

    // ── Sequence ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sequence_two_hops() {
        let graph = vec![t("a", "p1", "b"), t("b", "p2", "c")];
        let path = seq(vec![pred("p1"), pred("p2")]);
        let r = sorted_reachable(&path, "a", &graph);
        assert_eq!(r, vec!["c"]);
    }

    #[test]
    fn test_sequence_no_second_hop() {
        let graph = vec![t("a", "p1", "b")];
        let path = seq(vec![pred("p1"), pred("p2")]);
        let r = sorted_reachable(&path, "a", &graph);
        assert!(r.is_empty());
    }

    #[test]
    fn test_sequence_three_hops() {
        let graph = vec![t("a", "e", "b"), t("b", "e", "c"), t("c", "e", "d")];
        let path = seq(vec![pred("e"), pred("e"), pred("e")]);
        let r = sorted_reachable(&path, "a", &graph);
        assert_eq!(r, vec!["d"]);
    }

    // ── Alternative ───────────────────────────────────────────────────────────

    #[test]
    fn test_alternative_union() {
        let graph = vec![t("a", "p1", "b"), t("a", "p2", "c")];
        let path = alt(vec![pred("p1"), pred("p2")]);
        let r = sorted_reachable(&path, "a", &graph);
        assert_eq!(r, vec!["b", "c"]);
    }

    #[test]
    fn test_alternative_only_one_matches() {
        let graph = vec![t("a", "p1", "b")];
        let path = alt(vec![pred("p1"), pred("p2")]);
        let r = sorted_reachable(&path, "a", &graph);
        assert_eq!(r, vec!["b"]);
    }

    #[test]
    fn test_alternative_deduplicates() {
        // Both branches reach the same node
        let graph = vec![t("a", "p1", "b"), t("a", "p2", "b")];
        let path = alt(vec![pred("p1"), pred("p2")]);
        let r = sorted_reachable(&path, "a", &graph);
        assert_eq!(r, vec!["b"]); // deduplicated
    }

    // ── InversePath ───────────────────────────────────────────────────────────

    #[test]
    fn test_inverse_follows_reverse_edge() {
        let graph = vec![t("a", "p", "b")];
        let path = inv(pred("p"));
        // From "b" following ^p should reach "a"
        let r = sorted_reachable(&path, "b", &graph);
        assert_eq!(r, vec!["a"]);
    }

    #[test]
    fn test_inverse_no_match() {
        let graph = vec![t("a", "p", "b")];
        let path = inv(pred("p"));
        let r = sorted_reachable(&path, "a", &graph);
        assert!(r.is_empty());
    }

    #[test]
    fn test_inverse_multiple_subjects() {
        let graph = vec![t("a", "child", "c"), t("b", "child", "c")];
        let path = inv(pred("child"));
        let mut r = sorted_reachable(&path, "c", &graph);
        r.sort();
        assert_eq!(r, vec!["a", "b"]);
    }

    // ── ZeroOrMore ────────────────────────────────────────────────────────────

    #[test]
    fn test_zero_or_more_includes_start() {
        let graph = vec![t("a", "e", "b"), t("b", "e", "c")];
        let path = zero_or_more(pred("e"));
        let r = sorted_reachable(&path, "a", &graph);
        // Should include a (zero steps), b (one step), c (two steps)
        assert!(r.contains(&"a".to_string()));
        assert!(r.contains(&"b".to_string()));
        assert!(r.contains(&"c".to_string()));
    }

    #[test]
    fn test_zero_or_more_cycle_handling() {
        // Cycle: a→b→a
        let graph = vec![t("a", "e", "b"), t("b", "e", "a")];
        let path = zero_or_more(pred("e"));
        let r = sorted_reachable(&path, "a", &graph);
        // Should not loop forever; should contain a and b
        assert!(r.contains(&"a".to_string()));
        assert!(r.contains(&"b".to_string()));
    }

    #[test]
    fn test_zero_or_more_empty_graph() {
        let path = zero_or_more(pred("e"));
        let r = sorted_reachable(&path, "a", &[]);
        // Zero steps: only the start node
        assert_eq!(r, vec!["a"]);
    }

    // ── OneOrMore ─────────────────────────────────────────────────────────────

    #[test]
    fn test_one_or_more_requires_at_least_one_hop() {
        let graph = vec![t("a", "e", "b"), t("b", "e", "c")];
        let path = one_or_more(pred("e"));
        let r = sorted_reachable(&path, "a", &graph);
        // Should NOT include "a" (zero steps)
        assert!(!r.contains(&"a".to_string()));
        assert!(r.contains(&"b".to_string()));
        assert!(r.contains(&"c".to_string()));
    }

    #[test]
    fn test_one_or_more_no_edges() {
        let path = one_or_more(pred("e"));
        let r = sorted_reachable(&path, "a", &[]);
        assert!(r.is_empty());
    }

    #[test]
    fn test_one_or_more_cycle_does_not_loop() {
        let graph = vec![t("a", "e", "a")]; // self-loop
        let path = one_or_more(pred("e"));
        let r = sorted_reachable(&path, "a", &graph);
        assert_eq!(r, vec!["a"]);
    }

    // ── ZeroOrOne ─────────────────────────────────────────────────────────────

    #[test]
    fn test_zero_or_one_includes_start() {
        let graph = vec![t("a", "e", "b")];
        let path = zero_or_one(pred("e"));
        let r = sorted_reachable(&path, "a", &graph);
        assert!(r.contains(&"a".to_string()));
        assert!(r.contains(&"b".to_string()));
    }

    #[test]
    fn test_zero_or_one_no_edge_only_start() {
        let path = zero_or_one(pred("e"));
        let r = sorted_reachable(&path, "a", &[]);
        assert_eq!(r, vec!["a"]);
    }

    #[test]
    fn test_zero_or_one_at_most_one_hop() {
        // Two hops exist, but ZeroOrOne should only do zero or one
        let graph = vec![t("a", "e", "b"), t("b", "e", "c")];
        let path = zero_or_one(pred("e"));
        let r = sorted_reachable(&path, "a", &graph);
        assert!(r.contains(&"a".to_string()));
        assert!(r.contains(&"b".to_string()));
        assert!(!r.contains(&"c".to_string()));
    }

    // ── NegatedPredicate ──────────────────────────────────────────────────────

    #[test]
    fn test_negated_predicate_excludes_matching() {
        let graph = vec![t("a", "p", "b"), t("a", "q", "c")];
        let path = neg("p");
        let r = sorted_reachable(&path, "a", &graph);
        // "b" via "p" is excluded; "c" via "q" is included
        assert!(!r.contains(&"b".to_string()));
        assert!(r.contains(&"c".to_string()));
    }

    #[test]
    fn test_negated_predicate_all_excluded() {
        let graph = vec![t("a", "p", "b")];
        let path = neg("p");
        let r = sorted_reachable(&path, "a", &graph);
        assert!(r.is_empty());
    }

    #[test]
    fn test_negated_predicate_none_excluded() {
        let graph = vec![t("a", "q", "b")];
        let path = neg("p");
        let r = sorted_reachable(&path, "a", &graph);
        assert_eq!(r, vec!["b"]);
    }

    // ── evaluate_multi ────────────────────────────────────────────────────────

    #[test]
    fn test_evaluate_multi_two_starts() {
        let graph = vec![t("a", "p", "b"), t("x", "p", "y")];
        let path = pred("p");
        let results = PropertyPathChecker::evaluate_multi(&path, &["a", "x"], &graph);
        assert_eq!(results.len(), 2);
        let a_result = results
            .iter()
            .find(|(s, _)| *s == "a")
            .expect("should succeed");
        assert!(a_result.1.reachable.contains(&"b".to_string()));
        let x_result = results
            .iter()
            .find(|(s, _)| *s == "x")
            .expect("should succeed");
        assert!(x_result.1.reachable.contains(&"y".to_string()));
    }

    #[test]
    fn test_evaluate_multi_empty_starts() {
        let path = pred("p");
        let results = PropertyPathChecker::evaluate_multi(&path, &[], &[]);
        assert!(results.is_empty());
    }

    // ── path_depth ────────────────────────────────────────────────────────────

    #[test]
    fn test_path_depth_predicate() {
        assert_eq!(PropertyPathChecker::path_depth(&pred("p")), 0);
    }

    #[test]
    fn test_path_depth_inverse() {
        assert_eq!(PropertyPathChecker::path_depth(&inv(pred("p"))), 1);
    }

    #[test]
    fn test_path_depth_sequence() {
        let path = seq(vec![pred("p"), pred("q")]);
        assert_eq!(PropertyPathChecker::path_depth(&path), 1);
    }

    #[test]
    fn test_path_depth_nested() {
        let path = inv(zero_or_more(pred("p")));
        assert_eq!(PropertyPathChecker::path_depth(&path), 2);
    }

    // ── count_predicates ──────────────────────────────────────────────────────

    #[test]
    fn test_count_predicates_single() {
        assert_eq!(PropertyPathChecker::count_predicates(&pred("p")), 1);
    }

    #[test]
    fn test_count_predicates_sequence() {
        let path = seq(vec![pred("p"), pred("q"), pred("r")]);
        assert_eq!(PropertyPathChecker::count_predicates(&path), 3);
    }

    #[test]
    fn test_count_predicates_alternative() {
        let path = alt(vec![pred("p"), pred("q")]);
        assert_eq!(PropertyPathChecker::count_predicates(&path), 2);
    }

    #[test]
    fn test_count_predicates_zero_or_more() {
        assert_eq!(
            PropertyPathChecker::count_predicates(&zero_or_more(pred("p"))),
            1
        );
    }

    // ── is_simple ─────────────────────────────────────────────────────────────

    #[test]
    fn test_is_simple_predicate() {
        assert!(PropertyPathChecker::is_simple(&pred("p")));
    }

    #[test]
    fn test_is_simple_inverse() {
        assert!(PropertyPathChecker::is_simple(&inv(pred("p"))));
    }

    #[test]
    fn test_is_simple_sequence_of_predicates() {
        let path = seq(vec![pred("p"), pred("q")]);
        assert!(PropertyPathChecker::is_simple(&path));
    }

    #[test]
    fn test_is_simple_zero_or_more_not_simple() {
        assert!(!PropertyPathChecker::is_simple(&zero_or_more(pred("p"))));
    }

    #[test]
    fn test_is_simple_one_or_more_not_simple() {
        assert!(!PropertyPathChecker::is_simple(&one_or_more(pred("p"))));
    }

    #[test]
    fn test_is_simple_zero_or_one_not_simple() {
        assert!(!PropertyPathChecker::is_simple(&zero_or_one(pred("p"))));
    }

    #[test]
    fn test_is_simple_sequence_containing_star() {
        let path = seq(vec![pred("p"), zero_or_more(pred("q"))]);
        assert!(!PropertyPathChecker::is_simple(&path));
    }

    // ── PathResult fields ─────────────────────────────────────────────────────

    #[test]
    fn test_path_result_path_count_equals_reachable_len() {
        let graph = vec![t("a", "p", "b"), t("a", "p", "c")];
        let result = PropertyPathChecker::evaluate(&pred("p"), "a", &graph);
        assert_eq!(result.path_count, result.reachable.len());
    }

    // ── inverse_step helper (used internally) ─────────────────────────────────

    #[test]
    fn test_inverse_step_helper() {
        let graph = vec![t("parent", "child", "kid"), t("other", "child", "kid")];
        let result = inverse_step("child", "kid", &graph);
        let mut sorted = result;
        sorted.sort();
        assert_eq!(sorted, vec!["other", "parent"]);
    }

    // ── predicate_step helper ─────────────────────────────────────────────────

    #[test]
    fn test_predicate_step_helper() {
        let graph = vec![t("a", "knows", "b"), t("a", "knows", "c")];
        let mut result = predicate_step("knows", "a", &graph);
        result.sort();
        assert_eq!(result, vec!["b", "c"]);
    }
}
