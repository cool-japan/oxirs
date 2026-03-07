//! SPARQL 1.1 Property Path Evaluation
//!
//! This module implements the property path evaluation semantics as defined in
//! the SPARQL 1.1 specification (<https://www.w3.org/TR/sparql11-query/#propertypaths>).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// A SPARQL 1.1 property path expression.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyPath {
    /// A single IRI predicate: `<iri>`
    Iri(String),
    /// Sequence path: `path1/path2`
    Sequence(Box<PropertyPath>, Box<PropertyPath>),
    /// Alternative path: `path1|path2`
    Alternative(Box<PropertyPath>, Box<PropertyPath>),
    /// Zero-or-more: `path*`
    ZeroOrMore(Box<PropertyPath>),
    /// One-or-more: `path+`
    OneOrMore(Box<PropertyPath>),
    /// Zero-or-one: `path?`
    ZeroOrOne(Box<PropertyPath>),
    /// Inverse path: `^path`
    Inverse(Box<PropertyPath>),
    /// Negated property set: `!(iri1|iri2|...)`
    NegatedSet(Vec<String>),
}

/// Error type for property path parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum PathError {
    /// Unknown or unsupported syntax encountered.
    UnknownSyntax(String),
    /// An empty path was provided.
    EmptyPath,
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathError::UnknownSyntax(s) => write!(f, "Unknown path syntax: {s}"),
            PathError::EmptyPath => write!(f, "Empty path expression"),
        }
    }
}

impl std::error::Error for PathError {}

/// An in-memory RDF graph for property path evaluation.
///
/// Stores triples as `subject -> [(predicate, object)]` for efficient lookup.
#[derive(Debug, Default, Clone)]
pub struct PathGraph {
    /// Forward index: subject -> list of (predicate, object)
    forward: HashMap<String, Vec<(String, String)>>,
    /// Reverse index: object -> list of (predicate, subject)
    reverse: HashMap<String, Vec<(String, String)>>,
}

impl PathGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the graph.
    pub fn add_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        self.forward
            .entry(subject.to_string())
            .or_default()
            .push((predicate.to_string(), object.to_string()));
        self.reverse
            .entry(object.to_string())
            .or_default()
            .push((predicate.to_string(), subject.to_string()));
    }

    /// Return all distinct subjects in the graph.
    pub fn subjects(&self) -> Vec<String> {
        self.forward.keys().cloned().collect()
    }

    /// Return all objects reachable from `subject` via `predicate`.
    pub fn objects_of(&self, subject: &str, predicate: &str) -> Vec<String> {
        self.forward
            .get(subject)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(p, _)| p == predicate)
                    .map(|(_, o)| o.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Return all subjects that have `predicate` pointing to `object`.
    pub fn subjects_of(&self, object: &str, predicate: &str) -> Vec<String> {
        self.reverse
            .get(object)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(p, _)| p == predicate)
                    .map(|(_, s)| s.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Return all distinct predicates in the graph.
    pub fn all_predicates(&self) -> Vec<String> {
        let mut preds: HashSet<String> = HashSet::new();
        for pairs in self.forward.values() {
            for (p, _) in pairs {
                preds.insert(p.clone());
            }
        }
        preds.into_iter().collect()
    }

    /// Return all objects reachable from `subject` via any predicate not in `excluded`.
    pub fn objects_via_any_except(&self, subject: &str, excluded: &[String]) -> Vec<String> {
        let excluded_set: HashSet<&String> = excluded.iter().collect();
        self.forward
            .get(subject)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(p, _)| !excluded_set.contains(p))
                    .map(|(_, o)| o.clone())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Evaluator for SPARQL 1.1 property path expressions.
pub struct PropertyPathEvaluator;

impl PropertyPathEvaluator {
    /// Create a new evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate a property path starting from `start_node` in `graph`.
    ///
    /// Returns the list of distinct reachable nodes (no duplicates).
    pub fn evaluate(
        &self,
        graph: &PathGraph,
        path: &PropertyPath,
        start_node: &str,
    ) -> Vec<String> {
        let mut results = self.eval_inner(graph, path, start_node);
        results.sort();
        results.dedup();
        results
    }

    fn eval_inner(&self, graph: &PathGraph, path: &PropertyPath, start: &str) -> Vec<String> {
        match path {
            PropertyPath::Iri(iri) => graph.objects_of(start, iri),

            PropertyPath::Sequence(a, b) => {
                let mid_nodes = self.eval_inner(graph, a, start);
                let mut results = Vec::new();
                for mid in &mid_nodes {
                    let mut tail = self.eval_inner(graph, b, mid);
                    results.append(&mut tail);
                }
                results
            }

            PropertyPath::Alternative(a, b) => {
                let mut ra = self.eval_inner(graph, a, start);
                let mut rb = self.eval_inner(graph, b, start);
                ra.append(&mut rb);
                ra
            }

            PropertyPath::ZeroOrMore(inner) => {
                // BFS closure including start_node itself
                self.bfs_closure(graph, inner, start, true)
            }

            PropertyPath::OneOrMore(inner) => {
                // BFS closure excluding start_node (unless reachable via path)
                self.bfs_closure(graph, inner, start, false)
            }

            PropertyPath::ZeroOrOne(inner) => {
                // Direct objects + start_node itself
                let mut results = self.eval_inner(graph, inner, start);
                results.push(start.to_string());
                results
            }

            PropertyPath::Inverse(inner) => {
                // Evaluate with reversed graph
                let reversed = Self::build_reverse_graph(graph);
                self.eval_inner(&reversed, inner, start)
            }

            PropertyPath::NegatedSet(iris) => graph.objects_via_any_except(start, iris),
        }
    }

    /// BFS transitive closure.
    ///
    /// `include_start` = true means ZeroOrMore (start node is always in results).
    /// `include_start` = false means OneOrMore (start node only included if reachable).
    fn bfs_closure(
        &self,
        graph: &PathGraph,
        path: &PropertyPath,
        start: &str,
        include_start: bool,
    ) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut results: Vec<String> = Vec::new();

        if include_start {
            visited.insert(start.to_string());
            results.push(start.to_string());
        }

        // Seed the queue with direct successors
        let direct = self.eval_inner(graph, path, start);
        for node in direct {
            if visited.insert(node.clone()) {
                results.push(node.clone());
                queue.push_back(node);
            }
        }

        // BFS
        while let Some(current) = queue.pop_front() {
            let next_nodes = self.eval_inner(graph, path, &current);
            for node in next_nodes {
                if visited.insert(node.clone()) {
                    results.push(node.clone());
                    queue.push_back(node);
                }
            }
        }

        results
    }

    /// Build a graph with all edges reversed (used for Inverse paths).
    fn build_reverse_graph(graph: &PathGraph) -> PathGraph {
        let mut reversed = PathGraph::new();
        for (subject, pairs) in &graph.forward {
            for (predicate, object) in pairs {
                reversed.add_triple(object, predicate, subject);
            }
        }
        reversed
    }
}

impl Default for PropertyPathEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a simple property path string into a `PropertyPath`.
///
/// Handles the following forms:
/// - `<iri>` — single IRI
/// - `path/path` — sequence
/// - `path|path` — alternative
/// - `path*` — zero or more
/// - `path+` — one or more
/// - `path?` — zero or one
/// - `^path` — inverse
/// - `!(iri)` — negated set (single IRI)
/// - `!(<iri1>|<iri2>|...)` — negated set (multiple IRIs)
pub fn parse_path(s: &str) -> Result<PropertyPath, PathError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(PathError::EmptyPath);
    }
    parse_alternative(s)
}

/// Parse at the lowest precedence level: alternative (`|`).
fn parse_alternative(s: &str) -> Result<PropertyPath, PathError> {
    // Split on `|` but only at the top level (not inside angle brackets or parens)
    let parts = split_top_level(s, '|');
    if parts.len() == 1 {
        return parse_sequence(s);
    }
    // Fold: a|b|c => Alternative(a, Alternative(b, c))
    let mut iter = parts.iter().rev();
    let last = parse_sequence(iter.next().ok_or(PathError::EmptyPath)?.trim())?;
    let mut result = last;
    for part in iter {
        let left = parse_sequence(part.trim())?;
        result = PropertyPath::Alternative(Box::new(left), Box::new(result));
    }
    Ok(result)
}

/// Parse sequence (`/`) at higher precedence than alternative.
fn parse_sequence(s: &str) -> Result<PropertyPath, PathError> {
    let parts = split_top_level(s, '/');
    if parts.len() == 1 {
        return parse_postfix(s);
    }
    let mut iter = parts.iter();
    let first_str = iter.next().ok_or(PathError::EmptyPath)?.trim();
    let mut result = parse_postfix(first_str)?;
    for part in iter {
        let right = parse_postfix(part.trim())?;
        result = PropertyPath::Sequence(Box::new(result), Box::new(right));
    }
    Ok(result)
}

/// Parse postfix operators (`*`, `+`, `?`) and prefix `^` and `!`.
fn parse_postfix(s: &str) -> Result<PropertyPath, PathError> {
    if s.is_empty() {
        return Err(PathError::EmptyPath);
    }

    // Handle inverse: ^path
    if let Some(rest) = s.strip_prefix('^') {
        let inner = parse_postfix(rest.trim())?;
        return Ok(PropertyPath::Inverse(Box::new(inner)));
    }

    // Handle negated property set: !(iri) or !(<iri1>|<iri2>)
    if let Some(stripped) = s.strip_prefix('!') {
        let inner_str = stripped.trim();
        // Must be wrapped in parens
        if inner_str.starts_with('(') && inner_str.ends_with(')') {
            let content = &inner_str[1..inner_str.len() - 1];
            // Split by | to get individual IRIs
            let iri_parts: Vec<&str> = content.split('|').collect();
            let mut iris = Vec::new();
            for part in iri_parts {
                let p = part.trim();
                if p.starts_with('<') && p.ends_with('>') {
                    iris.push(p[1..p.len() - 1].to_string());
                } else if !p.is_empty() {
                    iris.push(p.to_string());
                }
            }
            if iris.is_empty() {
                return Err(PathError::UnknownSyntax(s.to_string()));
            }
            return Ok(PropertyPath::NegatedSet(iris));
        }
        return Err(PathError::UnknownSyntax(s.to_string()));
    }

    // Check for postfix suffix: *, +, ?
    // The suffix is the last character if the string is wrapped in parens or an IRI
    let (base_str, suffix) = extract_postfix_suffix(s);

    let base = parse_atom(base_str)?;
    match suffix {
        Some('*') => Ok(PropertyPath::ZeroOrMore(Box::new(base))),
        Some('+') => Ok(PropertyPath::OneOrMore(Box::new(base))),
        Some('?') => Ok(PropertyPath::ZeroOrOne(Box::new(base))),
        Some(c) => Err(PathError::UnknownSyntax(format!("Unknown postfix: {c}"))),
        None => Ok(base),
    }
}

/// Extract the postfix suffix character from a path string, returning (base, suffix).
fn extract_postfix_suffix(s: &str) -> (&str, Option<char>) {
    if s.is_empty() {
        return (s, None);
    }
    let last = s.chars().next_back();
    match last {
        Some('*') | Some('+') | Some('?') => {
            let suffix_char = last;
            let base = s[..s.len() - 1].trim_end();
            // Only strip the suffix if the base makes sense
            if !base.is_empty() {
                (base, suffix_char)
            } else {
                (s, None)
            }
        }
        _ => (s, None),
    }
}

/// Parse an atom: a parenthesized expression or a bare IRI.
fn parse_atom(s: &str) -> Result<PropertyPath, PathError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(PathError::EmptyPath);
    }

    // Parenthesized group: (expr)
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        return parse_alternative(inner);
    }

    // IRI in angle brackets: <iri>
    if s.starts_with('<') && s.ends_with('>') {
        let iri = &s[1..s.len() - 1];
        return Ok(PropertyPath::Iri(iri.to_string()));
    }

    // Bare name (prefix:local or just a name for testing)
    if s.chars()
        .all(|c| c.is_alphanumeric() || c == ':' || c == '_' || c == '-' || c == '.')
    {
        return Ok(PropertyPath::Iri(s.to_string()));
    }

    Err(PathError::UnknownSyntax(s.to_string()))
}

/// Split a string on `delimiter` only at the top level (not inside `<>` or `()`).
fn split_top_level(s: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth_angle = 0usize;
    let mut depth_paren = 0usize;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '<' => depth_angle += 1,
            '>' => depth_angle = depth_angle.saturating_sub(1),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            _ if c == delimiter && depth_angle == 0 && depth_paren == 0 => {
                parts.push(&s[start..i]);
                start = i + delimiter.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> PathGraph {
        let mut g = PathGraph::new();
        // Simple chain: a -p-> b -p-> c -p-> d
        g.add_triple("a", "p", "b");
        g.add_triple("b", "p", "c");
        g.add_triple("c", "p", "d");
        // Alternative predicate
        g.add_triple("a", "q", "e");
        // Inverse: e -r-> a (so ^r from a is nothing, but from e via ^r gives a's predecessors)
        g.add_triple("e", "r", "a");
        g
    }

    fn evaluator() -> PropertyPathEvaluator {
        PropertyPathEvaluator::new()
    }

    // ── IRI path ──────────────────────────────────────────────────────────────

    #[test]
    fn test_iri_path_simple() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::Iri("p".to_string());
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["b"]);
    }

    #[test]
    fn test_iri_path_no_match() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::Iri("p".to_string());
        let r = ev.evaluate(&g, &path, "d"); // d has no outgoing p
        assert!(r.is_empty());
    }

    #[test]
    fn test_iri_path_multiple_objects() {
        let mut g = PathGraph::new();
        g.add_triple("a", "p", "b");
        g.add_triple("a", "p", "c");
        let ev = evaluator();
        let path = PropertyPath::Iri("p".to_string());
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["b", "c"]);
    }

    // ── Sequence path ─────────────────────────────────────────────────────────

    #[test]
    fn test_sequence_two_hops() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("p".to_string())),
        );
        let r = ev.evaluate(&g, &path, "a");
        assert_eq!(r, vec!["c"]);
    }

    #[test]
    fn test_sequence_three_hops() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Sequence(
                Box::new(PropertyPath::Iri("p".to_string())),
                Box::new(PropertyPath::Iri("p".to_string())),
            )),
            Box::new(PropertyPath::Iri("p".to_string())),
        );
        let r = ev.evaluate(&g, &path, "a");
        assert_eq!(r, vec!["d"]);
    }

    #[test]
    fn test_sequence_no_match() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())), // no q after p
        );
        let r = ev.evaluate(&g, &path, "a");
        assert!(r.is_empty());
    }

    // ── Alternative path ──────────────────────────────────────────────────────

    #[test]
    fn test_alternative_both_branches() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::Alternative(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())),
        );
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["b", "e"]);
    }

    #[test]
    fn test_alternative_dedup() {
        let mut g = PathGraph::new();
        g.add_triple("a", "p", "b");
        g.add_triple("a", "q", "b"); // both reach b
        let ev = evaluator();
        let path = PropertyPath::Alternative(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("q".to_string())),
        );
        let r = ev.evaluate(&g, &path, "a");
        assert_eq!(r, vec!["b"]); // deduped
    }

    #[test]
    fn test_alternative_one_empty() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::Alternative(
            Box::new(PropertyPath::Iri("p".to_string())),
            Box::new(PropertyPath::Iri("z".to_string())), // z doesn't exist
        );
        let r = ev.evaluate(&g, &path, "a");
        assert_eq!(r, vec!["b"]);
    }

    // ── ZeroOrMore path ───────────────────────────────────────────────────────

    #[test]
    fn test_zero_or_more_includes_start() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_zero_or_more_no_outgoing() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let r = ev.evaluate(&g, &path, "d"); // d has no outgoing p
        assert_eq!(r, vec!["d"]); // just start
    }

    #[test]
    fn test_zero_or_more_cycle_safety() {
        let mut g = PathGraph::new();
        g.add_triple("a", "p", "b");
        g.add_triple("b", "p", "a"); // cycle
        let ev = evaluator();
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["a", "b"]); // no infinite loop
    }

    // ── OneOrMore path ────────────────────────────────────────────────────────

    #[test]
    fn test_one_or_more_excludes_start() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["b", "c", "d"]);
    }

    #[test]
    fn test_one_or_more_no_outgoing() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let r = ev.evaluate(&g, &path, "d");
        assert!(r.is_empty());
    }

    #[test]
    fn test_one_or_more_cycle_safety() {
        let mut g = PathGraph::new();
        g.add_triple("a", "p", "b");
        g.add_triple("b", "p", "a");
        let ev = evaluator();
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        // b is reachable; a is reachable via b->a but we only add if not visited
        // "a" is not in initial visited set (OneOrMore), so it will be added when b->a is found
        assert!(r.contains(&"b".to_string()));
    }

    // ── ZeroOrOne path ────────────────────────────────────────────────────────

    #[test]
    fn test_zero_or_one_includes_start() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::ZeroOrOne(Box::new(PropertyPath::Iri("p".to_string())));
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["a", "b"]);
    }

    #[test]
    fn test_zero_or_one_no_outgoing() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::ZeroOrOne(Box::new(PropertyPath::Iri("p".to_string())));
        let mut r = ev.evaluate(&g, &path, "d");
        r.sort();
        assert_eq!(r, vec!["d"]); // just start node
    }

    // ── Inverse path ──────────────────────────────────────────────────────────

    #[test]
    fn test_inverse_path() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::Inverse(Box::new(PropertyPath::Iri("p".to_string())));
        let r = ev.evaluate(&g, &path, "b"); // who points to b via p?
        assert_eq!(r, vec!["a"]);
    }

    #[test]
    fn test_inverse_chain() {
        let g = make_graph();
        let ev = evaluator();
        // From d, go backwards via p twice = b
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Inverse(Box::new(PropertyPath::Iri(
                "p".to_string(),
            )))),
            Box::new(PropertyPath::Inverse(Box::new(PropertyPath::Iri(
                "p".to_string(),
            )))),
        );
        let r = ev.evaluate(&g, &path, "d");
        assert_eq!(r, vec!["b"]);
    }

    #[test]
    fn test_inverse_no_match() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::Inverse(Box::new(PropertyPath::Iri("p".to_string())));
        let r = ev.evaluate(&g, &path, "a"); // nobody points to a via p
        assert!(r.is_empty());
    }

    // ── NegatedSet path ───────────────────────────────────────────────────────

    #[test]
    fn test_negated_set_excludes_predicate() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::NegatedSet(vec!["q".to_string()]);
        let mut r = ev.evaluate(&g, &path, "a"); // a has p->b and q->e; exclude q
        r.sort();
        assert_eq!(r, vec!["b"]);
    }

    #[test]
    fn test_negated_set_all_excluded() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::NegatedSet(vec!["p".to_string(), "q".to_string()]);
        let r = ev.evaluate(&g, &path, "a"); // all predicates excluded
        assert!(r.is_empty());
    }

    #[test]
    fn test_negated_set_empty_exclusion() {
        let g = make_graph();
        let ev = evaluator();
        let path = PropertyPath::NegatedSet(vec![]); // exclude nothing = all
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["b", "e"]);
    }

    // ── PathGraph methods ─────────────────────────────────────────────────────

    #[test]
    fn test_path_graph_subjects() {
        let g = make_graph();
        let mut s = g.subjects();
        s.sort();
        assert!(s.contains(&"a".to_string()));
        assert!(s.contains(&"e".to_string()));
    }

    #[test]
    fn test_path_graph_objects_of() {
        let g = make_graph();
        let r = g.objects_of("a", "p");
        assert_eq!(r, vec!["b"]);
    }

    #[test]
    fn test_path_graph_add_multiple() {
        let mut g = PathGraph::new();
        g.add_triple("x", "pred", "y1");
        g.add_triple("x", "pred", "y2");
        g.add_triple("x", "pred", "y3");
        let mut r = g.objects_of("x", "pred");
        r.sort();
        assert_eq!(r, vec!["y1", "y2", "y3"]);
    }

    // ── parse_path ────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_iri() {
        let p = parse_path("<http://example.org/p>").unwrap();
        assert_eq!(p, PropertyPath::Iri("http://example.org/p".to_string()));
    }

    #[test]
    fn test_parse_empty_error() {
        let r = parse_path("");
        assert_eq!(r, Err(PathError::EmptyPath));
    }

    #[test]
    fn test_parse_sequence() {
        let p = parse_path("<a>/<b>").unwrap();
        assert_eq!(
            p,
            PropertyPath::Sequence(
                Box::new(PropertyPath::Iri("a".to_string())),
                Box::new(PropertyPath::Iri("b".to_string())),
            )
        );
    }

    #[test]
    fn test_parse_alternative() {
        let p = parse_path("<a>|<b>").unwrap();
        assert_eq!(
            p,
            PropertyPath::Alternative(
                Box::new(PropertyPath::Iri("a".to_string())),
                Box::new(PropertyPath::Iri("b".to_string())),
            )
        );
    }

    #[test]
    fn test_parse_zero_or_more() {
        let p = parse_path("<a>*").unwrap();
        assert_eq!(
            p,
            PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("a".to_string())))
        );
    }

    #[test]
    fn test_parse_one_or_more() {
        let p = parse_path("<a>+").unwrap();
        assert_eq!(
            p,
            PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("a".to_string())))
        );
    }

    #[test]
    fn test_parse_zero_or_one() {
        let p = parse_path("<a>?").unwrap();
        assert_eq!(
            p,
            PropertyPath::ZeroOrOne(Box::new(PropertyPath::Iri("a".to_string())))
        );
    }

    #[test]
    fn test_parse_inverse() {
        let p = parse_path("^<a>").unwrap();
        assert_eq!(
            p,
            PropertyPath::Inverse(Box::new(PropertyPath::Iri("a".to_string())))
        );
    }

    #[test]
    fn test_parse_negated_set_single() {
        let p = parse_path("!(<http://example.org/p>)").unwrap();
        assert_eq!(
            p,
            PropertyPath::NegatedSet(vec!["http://example.org/p".to_string()])
        );
    }

    #[test]
    fn test_parse_negated_set_multiple() {
        let p = parse_path("!(<a>|<b>)").unwrap();
        assert_eq!(
            p,
            PropertyPath::NegatedSet(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn test_parse_bare_name() {
        let p = parse_path("rdf:type").unwrap();
        assert_eq!(p, PropertyPath::Iri("rdf:type".to_string()));
    }

    // ── BFS cycle detection ───────────────────────────────────────────────────

    #[test]
    fn test_zero_or_more_complex_cycle() {
        let mut g = PathGraph::new();
        g.add_triple("a", "p", "b");
        g.add_triple("b", "p", "c");
        g.add_triple("c", "p", "a"); // back to a
        let ev = evaluator();
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_one_or_more_self_loop() {
        let mut g = PathGraph::new();
        g.add_triple("a", "p", "a"); // self loop
        let ev = evaluator();
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("p".to_string())));
        let mut r = ev.evaluate(&g, &path, "a");
        r.sort();
        assert_eq!(r, vec!["a"]);
    }

    // ── PathError display ─────────────────────────────────────────────────────

    #[test]
    fn test_path_error_display() {
        let e = PathError::UnknownSyntax("???".to_string());
        assert!(e.to_string().contains("Unknown path syntax"));

        let e2 = PathError::EmptyPath;
        assert!(e2.to_string().contains("Empty path"));
    }

    // ── default impl ──────────────────────────────────────────────────────────

    #[test]
    fn test_evaluator_default() {
        let ev = PropertyPathEvaluator;
        let g = PathGraph::new();
        let path = PropertyPath::Iri("p".to_string());
        let r = ev.evaluate(&g, &path, "x");
        assert!(r.is_empty());
    }

    #[test]
    fn test_graph_default() {
        let g = PathGraph::default();
        assert!(g.subjects().is_empty());
    }
}
