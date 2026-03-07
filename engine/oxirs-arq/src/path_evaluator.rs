//! SPARQL 1.1 property path evaluation.
//!
//! Evaluates property paths defined by SPARQL 1.1 Section 9 against an in-memory
//! graph represented as an adjacency list: `node -> [(predicate, target)]`.

use std::collections::{HashMap, HashSet, VecDeque};

/// A SPARQL 1.1 property path expression.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyPath {
    /// A single IRI predicate.
    Iri(String),
    /// Inverse path `^p`.
    Inverse(Box<PropertyPath>),
    /// Sequence path `p/q`.
    Sequence(Box<PropertyPath>, Box<PropertyPath>),
    /// Alternative path `p|q`.
    Alternative(Box<PropertyPath>, Box<PropertyPath>),
    /// Zero-or-more path `p*`.
    ZeroOrMore(Box<PropertyPath>),
    /// One-or-more path `p+`.
    OneOrMore(Box<PropertyPath>),
    /// Zero-or-one path `p?`.
    ZeroOrOne(Box<PropertyPath>),
    /// Negated property set `!p`.
    Negation(Box<PropertyPath>),
}

/// A triple pattern whose predicate is a property path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathTriple {
    /// Subject node IRI.
    pub subject: String,
    /// Property path expression.
    pub path: PropertyPath,
    /// Object node IRI.
    pub object: String,
}

/// Stateless property path evaluator.
pub struct PathEvaluator;

impl PathEvaluator {
    /// Evaluate a property path starting from `start` and return all reachable
    /// end nodes according to the path expression.
    ///
    /// # Arguments
    /// * `graph`  - Adjacency list: `node -> [(predicate, target)]`.
    /// * `path`   - Property path to evaluate.
    /// * `start`  - Starting node IRI.
    pub fn evaluate(
        graph: &HashMap<String, Vec<(String, String)>>,
        path: &PropertyPath,
        start: &str,
    ) -> Vec<String> {
        let mut results: Vec<String> = match path {
            PropertyPath::Iri(iri) => Self::eval_iri(graph, iri, start),
            PropertyPath::Inverse(inner) => Self::eval_inverse(graph, inner, start),
            PropertyPath::Sequence(left, right) => Self::eval_sequence(graph, left, right, start),
            PropertyPath::Alternative(left, right) => {
                Self::eval_alternative(graph, left, right, start)
            }
            PropertyPath::ZeroOrMore(inner) => Self::eval_zero_or_more(graph, inner, start),
            PropertyPath::OneOrMore(inner) => Self::eval_one_or_more(graph, inner, start),
            PropertyPath::ZeroOrOne(inner) => Self::eval_zero_or_one(graph, inner, start),
            PropertyPath::Negation(inner) => Self::eval_negation(graph, inner, start),
        };
        results.sort();
        results.dedup();
        results
    }

    /// Evaluate a single IRI predicate step.
    pub fn eval_iri(
        graph: &HashMap<String, Vec<(String, String)>>,
        iri: &str,
        start: &str,
    ) -> Vec<String> {
        graph
            .get(start)
            .map(|edges| {
                edges
                    .iter()
                    .filter(|(pred, _)| pred == iri)
                    .map(|(_, target)| target.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Evaluate inverse path `^p`: find all nodes `x` such that `x -p-> start`.
    pub fn eval_inverse(
        graph: &HashMap<String, Vec<(String, String)>>,
        path: &PropertyPath,
        start: &str,
    ) -> Vec<String> {
        // Build reverse graph on-the-fly.
        let mut results = Vec::new();
        for (node, edges) in graph {
            let reachable = Self::evaluate(graph, path, node);
            if reachable.iter().any(|t| t == start) {
                results.push(node.clone());
            }
        }
        results
    }

    /// Evaluate sequence `p/q`: first evaluate `p` from `start`, then `q` from
    /// each intermediate node.
    pub fn eval_sequence(
        graph: &HashMap<String, Vec<(String, String)>>,
        left: &PropertyPath,
        right: &PropertyPath,
        start: &str,
    ) -> Vec<String> {
        let intermediates = Self::evaluate(graph, left, start);
        let mut results = Vec::new();
        for mid in &intermediates {
            results.extend(Self::evaluate(graph, right, mid));
        }
        results
    }

    /// Evaluate alternative `p|q`: union of results from both paths.
    pub fn eval_alternative(
        graph: &HashMap<String, Vec<(String, String)>>,
        left: &PropertyPath,
        right: &PropertyPath,
        start: &str,
    ) -> Vec<String> {
        let mut results = Self::evaluate(graph, left, start);
        results.extend(Self::evaluate(graph, right, start));
        results
    }

    /// Evaluate zero-or-more `p*`: BFS transitive closure including `start`.
    pub fn eval_zero_or_more(
        graph: &HashMap<String, Vec<(String, String)>>,
        path: &PropertyPath,
        start: &str,
    ) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(start.to_owned());
        visited.insert(start.to_owned());

        while let Some(current) = queue.pop_front() {
            for next in Self::evaluate(graph, path, &current) {
                if visited.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }
        visited.into_iter().collect()
    }

    /// Evaluate one-or-more `p+`: BFS transitive closure excluding `start`
    /// (unless it is re-reached via the path).
    pub fn eval_one_or_more(
        graph: &HashMap<String, Vec<(String, String)>>,
        path: &PropertyPath,
        start: &str,
    ) -> Vec<String> {
        let first_step: Vec<String> = Self::evaluate(graph, path, start);
        let mut visited: HashSet<String> = first_step.iter().cloned().collect();
        let mut queue: VecDeque<String> = first_step.into_iter().collect();

        while let Some(current) = queue.pop_front() {
            for next in Self::evaluate(graph, path, &current) {
                if visited.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }
        visited.into_iter().collect()
    }

    /// Evaluate zero-or-one `p?`: `start` plus at most one step via `p`.
    pub fn eval_zero_or_one(
        graph: &HashMap<String, Vec<(String, String)>>,
        path: &PropertyPath,
        start: &str,
    ) -> Vec<String> {
        let mut results = vec![start.to_owned()];
        results.extend(Self::evaluate(graph, path, start));
        results.sort();
        results.dedup();
        results
    }

    /// Evaluate negated property set `!p`: all direct neighbours reachable by
    /// ANY predicate EXCEPT those reachable by `p`.
    pub fn eval_negation(
        graph: &HashMap<String, Vec<(String, String)>>,
        path: &PropertyPath,
        start: &str,
    ) -> Vec<String> {
        let excluded: HashSet<String> = Self::evaluate(graph, path, start).into_iter().collect();
        // Collect all direct neighbours.
        let all_neighbours: HashSet<String> = graph
            .get(start)
            .map(|edges| edges.iter().map(|(_, t)| t.clone()).collect())
            .unwrap_or_default();
        all_neighbours
            .difference(&excluded)
            .cloned()
            .collect()
    }

    /// Check whether `end` is reachable from `start` via `path`.
    pub fn matches(
        graph: &HashMap<String, Vec<(String, String)>>,
        path: &PropertyPath,
        start: &str,
        end: &str,
    ) -> bool {
        Self::evaluate(graph, path, start)
            .iter()
            .any(|n| n == end)
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn build_graph(triples: &[(&str, &str, &str)]) -> HashMap<String, Vec<(String, String)>> {
    let mut graph: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (s, p, o) in triples {
        graph
            .entry(s.to_string())
            .or_default()
            .push((p.to_string(), o.to_string()));
    }
    graph
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // helper
    fn g(triples: &[(&str, &str, &str)]) -> HashMap<String, Vec<(String, String)>> {
        build_graph(triples)
    }

    // ── PropertyPath::Iri ────────────────────────────────────────────────────

    #[test]
    fn test_iri_single_match() {
        let graph = g(&[("a", "knows", "b")]);
        let path = PropertyPath::Iri("knows".to_string());
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert_eq!(result, vec!["b"]);
    }

    #[test]
    fn test_iri_no_match() {
        let graph = g(&[("a", "knows", "b")]);
        let path = PropertyPath::Iri("likes".to_string());
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert!(result.is_empty());
    }

    #[test]
    fn test_iri_multiple_targets() {
        let graph = g(&[("a", "knows", "b"), ("a", "knows", "c"), ("a", "knows", "d")]);
        let path = PropertyPath::Iri("knows".to_string());
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["b", "c", "d"]);
    }

    #[test]
    fn test_iri_non_existent_node() {
        let graph = g(&[]);
        let path = PropertyPath::Iri("p".to_string());
        let result = PathEvaluator::evaluate(&graph, &path, "missing");
        assert!(result.is_empty());
    }

    #[test]
    fn test_eval_iri_helper() {
        let graph = g(&[("x", "p", "y"), ("x", "q", "z")]);
        let result = PathEvaluator::eval_iri(&graph, "p", "x");
        assert_eq!(result, vec!["y"]);
    }

    // ── PropertyPath::Inverse ────────────────────────────────────────────────

    #[test]
    fn test_inverse_basic() {
        let graph = g(&[("a", "knows", "b")]);
        let path = PropertyPath::Inverse(Box::new(PropertyPath::Iri("knows".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "b");
        assert_eq!(result, vec!["a"]);
    }

    #[test]
    fn test_inverse_no_incoming() {
        let graph = g(&[("a", "knows", "b")]);
        let path = PropertyPath::Inverse(Box::new(PropertyPath::Iri("knows".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert!(result.is_empty());
    }

    #[test]
    fn test_inverse_multiple_sources() {
        let graph = g(&[("a", "p", "c"), ("b", "p", "c")]);
        let path = PropertyPath::Inverse(Box::new(PropertyPath::Iri("p".to_string())));
        let mut result = PathEvaluator::evaluate(&graph, &path, "c");
        result.sort();
        assert_eq!(result, vec!["a", "b"]);
    }

    // ── PropertyPath::Sequence ───────────────────────────────────────────────

    #[test]
    fn test_sequence_basic() {
        let graph = g(&[("a", "p", "b"), ("b", "q", "c")]);
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())),
        );
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert_eq!(result, vec!["c"]);
    }

    #[test]
    fn test_sequence_no_second_step() {
        let graph = g(&[("a", "p", "b")]);
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())),
        );
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert!(result.is_empty());
    }

    #[test]
    fn test_sequence_three_steps() {
        let graph = g(&[("a", "p", "b"), ("b", "q", "c"), ("c", "r", "d")]);
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Sequence(
                Box::new(PropertyPath::Iri("p".to_string())),
                Box::new(PropertyPath::Iri("q".to_string())),
            )),
            Box::new(PropertyPath::Iri("r".to_string())),
        );
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert_eq!(result, vec!["d"]);
    }

    // ── PropertyPath::Alternative ────────────────────────────────────────────

    #[test]
    fn test_alternative_both_match() {
        let graph = g(&[("a", "p", "b"), ("a", "q", "c")]);
        let path = PropertyPath::Alternative(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())),
        );
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["b", "c"]);
    }

    #[test]
    fn test_alternative_one_match() {
        let graph = g(&[("a", "p", "b")]);
        let path = PropertyPath::Alternative(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())),
        );
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert_eq!(result, vec!["b"]);
    }

    #[test]
    fn test_alternative_dedup() {
        let graph = g(&[("a", "p", "b"), ("a", "q", "b")]);
        let path = PropertyPath::Alternative(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())),
        );
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert_eq!(result, vec!["b"]);
    }

    // ── PropertyPath::ZeroOrMore ─────────────────────────────────────────────

    #[test]
    fn test_zero_or_more_includes_start() {
        let graph = g(&[("a", "p", "b"), ("b", "p", "c")]);
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert!(result.contains(&"a".to_string()));
        assert!(result.contains(&"b".to_string()));
        assert!(result.contains(&"c".to_string()));
    }

    #[test]
    fn test_zero_or_more_cycle() {
        let graph = g(&[("a", "p", "b"), ("b", "p", "a")]);
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["a", "b"]);
    }

    #[test]
    fn test_zero_or_more_no_outgoing() {
        let graph = g(&[]);
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert_eq!(result, vec!["a"]);
    }

    // ── PropertyPath::OneOrMore ──────────────────────────────────────────────

    #[test]
    fn test_one_or_more_basic() {
        let graph = g(&[("a", "p", "b"), ("b", "p", "c")]);
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["b", "c"]);
    }

    #[test]
    fn test_one_or_more_no_start() {
        let graph = g(&[("a", "p", "b"), ("b", "p", "c")]);
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert!(!result.contains(&"a".to_string()));
    }

    #[test]
    fn test_one_or_more_no_steps() {
        let graph = g(&[]);
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert!(result.is_empty());
    }

    // ── PropertyPath::ZeroOrOne ──────────────────────────────────────────────

    #[test]
    fn test_zero_or_one_with_match() {
        let graph = g(&[("a", "p", "b")]);
        let path = PropertyPath::ZeroOrOne(Box::new(PropertyPath::Iri("p".to_string())));
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["a", "b"]);
    }

    #[test]
    fn test_zero_or_one_no_match() {
        let graph = g(&[]);
        let path = PropertyPath::ZeroOrOne(Box::new(PropertyPath::Iri("p".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert_eq!(result, vec!["a"]);
    }

    // ── PropertyPath::Negation ───────────────────────────────────────────────

    #[test]
    fn test_negation_excludes_matched() {
        let graph = g(&[("a", "p", "b"), ("a", "q", "c")]);
        let path = PropertyPath::Negation(Box::new(PropertyPath::Iri("p".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert!(!result.contains(&"b".to_string()));
        assert!(result.contains(&"c".to_string()));
    }

    #[test]
    fn test_negation_empty_neighbours() {
        let graph = g(&[]);
        let path = PropertyPath::Negation(Box::new(PropertyPath::Iri("p".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert!(result.is_empty());
    }

    // ── PathEvaluator::matches ───────────────────────────────────────────────

    #[test]
    fn test_matches_true() {
        let graph = g(&[("a", "p", "b"), ("b", "p", "c")]);
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        assert!(PathEvaluator::matches(&graph, &path, "a", "c"));
    }

    #[test]
    fn test_matches_false() {
        let graph = g(&[("a", "p", "b")]);
        let path = PropertyPath::Iri("p".to_string());
        assert!(!PathEvaluator::matches(&graph, &path, "a", "a"));
    }

    #[test]
    fn test_matches_zero_or_more_self() {
        let graph = g(&[]);
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        assert!(PathEvaluator::matches(&graph, &path, "a", "a"));
    }

    // ── PathTriple ───────────────────────────────────────────────────────────

    #[test]
    fn test_path_triple_struct() {
        let pt = PathTriple {
            subject: "a".to_string(),
            path: PropertyPath::Iri("p".to_string()),
            object: "b".to_string(),
        };
        assert_eq!(pt.subject, "a");
        assert_eq!(pt.object, "b");
    }

    // ── complex paths ────────────────────────────────────────────────────────

    #[test]
    fn test_sequence_then_alternative() {
        let graph = g(&[("a", "p", "b"), ("b", "q", "c"), ("b", "r", "d")]);
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Alternative(
                Box::new(PropertyPath::Iri("q".to_string())),
                Box::new(PropertyPath::Iri("r".to_string())),
            )),
        );
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["c", "d"]);
    }

    #[test]
    fn test_inverse_after_sequence() {
        let graph = g(&[("a", "p", "b"), ("b", "q", "c")]);
        // (a/p/q)^-1 starting from c should give a
        let path = PropertyPath::Inverse(Box::new(PropertyPath::Sequence(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())),
        )));
        let result = PathEvaluator::evaluate(&graph, &path, "c");
        assert_eq!(result, vec!["a"]);
    }

    #[test]
    fn test_zero_or_more_long_chain() {
        let graph = g(&[
            ("a", "next", "b"),
            ("b", "next", "c"),
            ("c", "next", "d"),
            ("d", "next", "e"),
        ]);
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("next".to_string())));
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["a", "b", "c", "d", "e"]);
    }

    #[test]
    fn test_one_or_more_long_chain() {
        let graph = g(&[
            ("a", "next", "b"),
            ("b", "next", "c"),
            ("c", "next", "d"),
        ]);
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("next".to_string())));
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["b", "c", "d"]);
    }

    #[test]
    fn test_complex_alternative_sequence() {
        let graph = g(&[
            ("start", "a", "mid1"),
            ("start", "b", "mid2"),
            ("mid1", "c", "end1"),
            ("mid2", "c", "end2"),
        ]);
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Alternative(
                Box::new(PropertyPath::Iri("a".to_string())),
                Box::new(PropertyPath::Iri("b".to_string())),
            )),
            Box::new(PropertyPath::Iri("c".to_string())),
        );
        let mut result = PathEvaluator::evaluate(&graph, &path, "start");
        result.sort();
        assert_eq!(result, vec!["end1", "end2"]);
    }

    #[test]
    fn test_eval_iri_multiple_predicates() {
        let graph = g(&[("x", "p", "y"), ("x", "p", "z"), ("x", "q", "w")]);
        let mut result = PathEvaluator::eval_iri(&graph, "p", "x");
        result.sort();
        assert_eq!(result, vec!["y", "z"]);
    }

    #[test]
    fn test_eval_inverse_helper() {
        let graph = g(&[("a", "knows", "b"), ("c", "knows", "b")]);
        let path = PropertyPath::Iri("knows".to_string());
        let mut result = PathEvaluator::eval_inverse(&graph, &path, "b");
        result.sort();
        assert_eq!(result, vec!["a", "c"]);
    }

    #[test]
    fn test_eval_sequence_helper() {
        let graph = g(&[("a", "p", "b"), ("b", "q", "c"), ("b", "q", "d")]);
        let left = PropertyPath::Iri("p".to_string());
        let right = PropertyPath::Iri("q".to_string());
        let mut result = PathEvaluator::eval_sequence(&graph, &left, &right, "a");
        result.sort();
        assert_eq!(result, vec!["c", "d"]);
    }

    #[test]
    fn test_eval_alternative_helper() {
        let graph = g(&[("a", "p", "b"), ("a", "q", "c")]);
        let left = PropertyPath::Iri("p".to_string());
        let right = PropertyPath::Iri("q".to_string());
        let mut result = PathEvaluator::eval_alternative(&graph, &left, &right, "a");
        result.sort();
        assert_eq!(result, vec!["b", "c"]);
    }

    #[test]
    fn test_eval_zero_or_more_helper() {
        let graph = g(&[("a", "p", "b"), ("b", "p", "c")]);
        let path = PropertyPath::Iri("p".to_string());
        let mut result = PathEvaluator::eval_zero_or_more(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_eval_one_or_more_helper() {
        let graph = g(&[("a", "p", "b"), ("b", "p", "c")]);
        let path = PropertyPath::Iri("p".to_string());
        let mut result = PathEvaluator::eval_one_or_more(&graph, &path, "a");
        result.sort();
        assert_eq!(result, vec!["b", "c"]);
    }

    #[test]
    fn test_path_triple_equality() {
        let p1 = PathTriple {
            subject: "s".to_string(),
            path: PropertyPath::Iri("p".to_string()),
            object: "o".to_string(),
        };
        let p2 = PathTriple {
            subject: "s".to_string(),
            path: PropertyPath::Iri("p".to_string()),
            object: "o".to_string(),
        };
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_mixed_predicates_no_cross_contamination() {
        let graph = g(&[("a", "type", "Person"), ("a", "knows", "b")]);
        let path = PropertyPath::Iri("type".to_string());
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert_eq!(result, vec!["Person"]);
        assert!(!result.contains(&"b".to_string()));
    }

    #[test]
    fn test_zero_or_one_dedup() {
        // If start == the one-step result, should appear only once
        let graph = g(&[("a", "self", "a")]);
        let path = PropertyPath::ZeroOrOne(Box::new(PropertyPath::Iri("self".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "a");
    }

    #[test]
    fn test_negation_all_excluded() {
        // All outgoing predicates are negated
        let graph = g(&[("a", "p", "b")]);
        let path = PropertyPath::Negation(Box::new(PropertyPath::Iri("p".to_string())));
        let result = PathEvaluator::evaluate(&graph, &path, "a");
        assert!(result.is_empty());
    }

    #[test]
    fn test_nested_zero_or_more() {
        // Deeply nested: (p/q)*
        let graph = g(&[
            ("a", "p", "b"),
            ("b", "q", "c"),
            ("c", "p", "d"),
            ("d", "q", "e"),
        ]);
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Sequence(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())),
        )));
        let mut result = PathEvaluator::evaluate(&graph, &path, "a");
        result.sort();
        assert!(result.contains(&"a".to_string()));
        assert!(result.contains(&"c".to_string()));
        assert!(result.contains(&"e".to_string()));
    }
}
