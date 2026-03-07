//! SPARQL 1.1 Property Path evaluation
//!
//! Supports:
//! - Sequence: iri1 / iri2
//! - Alternative: iri1 | iri2
//! - ZeroOrMore: path*
//! - OneOrMore: path+
//! - ZeroOrOne: path?
//! - Inverse: ^path
//! - Negated property set: !(iri1 | iri2)
//! - Parenthesized: (path)

use crate::store::OxiRSStore;
use std::collections::HashSet;

/// A SPARQL property path expression
#[derive(Debug, Clone)]
pub(crate) enum PropertyPath {
    /// A single IRI predicate  (e.g. `<ex:knows>`)
    Iri(String),
    /// Inverse:  `^path`
    Inverse(Box<PropertyPath>),
    /// Sequence: `path1 / path2`
    Sequence(Box<PropertyPath>, Box<PropertyPath>),
    /// Alternative: `path1 | path2`
    Alternative(Box<PropertyPath>, Box<PropertyPath>),
    /// Zero or more: `path*`
    ZeroOrMore(Box<PropertyPath>),
    /// One or more: `path+`
    OneOrMore(Box<PropertyPath>),
    /// Zero or one: `path?`
    ZeroOrOne(Box<PropertyPath>),
    /// Negated property set: `!(iri1 | iri2)`
    NegatedSet(Vec<String>),
}

impl PropertyPath {
    /// Evaluate this property path from a given `subject` in the store.
    ///
    /// Returns all objects reachable via this path from `subject`.
    /// Cycle detection is applied for `*` and `+` paths.
    pub(crate) fn evaluate(&self, subject: &str, store: &OxiRSStore) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        self.eval_from(subject, store, &mut visited)
    }

    /// Evaluate this path from `subject`, tracking visited nodes for cycle detection.
    fn eval_from(
        &self,
        subject: &str,
        store: &OxiRSStore,
        visited: &mut HashSet<String>,
    ) -> Vec<String> {
        match self {
            PropertyPath::Iri(iri) => {
                // Match ?s <iri> ?o where ?s = subject
                store
                    .all_triples()
                    .filter(|t| t.subject == subject && t.predicate == *iri)
                    .map(|t| t.object.clone())
                    .collect()
            }

            PropertyPath::Inverse(inner) => {
                // Inverse: treat subject as object, find matching subjects
                store
                    .all_triples()
                    .filter(|t| t.object == subject && inner.matches_predicate(&t.predicate))
                    .map(|t| t.subject.clone())
                    .collect()
            }

            PropertyPath::Sequence(left, right) => {
                // left / right: find intermediates from left, then traverse right
                let intermediates = left.eval_from(subject, store, visited);
                let mut results = Vec::new();
                let mut result_set: HashSet<String> = HashSet::new();
                for mid in intermediates {
                    let mut mid_visited = visited.clone();
                    for obj in right.eval_from(&mid, store, &mut mid_visited) {
                        if result_set.insert(obj.clone()) {
                            results.push(obj);
                        }
                    }
                }
                results
            }

            PropertyPath::Alternative(left, right) => {
                let mut results = left.eval_from(subject, store, visited);
                let result_set: HashSet<String> = results.iter().cloned().collect();
                for obj in right.eval_from(subject, store, visited) {
                    if !result_set.contains(&obj) {
                        results.push(obj);
                    }
                }
                results
            }

            PropertyPath::ZeroOrMore(inner) => {
                // Include subject itself (zero steps), then expand transitively
                let mut results: Vec<String> = Vec::new();
                let mut result_set: HashSet<String> = HashSet::new();
                // Zero steps: include the subject itself
                result_set.insert(subject.to_string());
                results.push(subject.to_string());
                // Iterative expansion (BFS)
                let mut frontier: Vec<String> = vec![subject.to_string()];
                while !frontier.is_empty() {
                    let mut next_frontier = Vec::new();
                    for node in &frontier {
                        let mut node_visited = HashSet::new();
                        for obj in inner.eval_from(node, store, &mut node_visited) {
                            if result_set.insert(obj.clone()) {
                                results.push(obj.clone());
                                next_frontier.push(obj);
                            }
                        }
                    }
                    frontier = next_frontier;
                }
                results
            }

            PropertyPath::OneOrMore(inner) => {
                // At least one step
                let mut results: Vec<String> = Vec::new();
                let mut result_set: HashSet<String> = HashSet::new();
                // Avoid re-visiting subject
                visited.insert(subject.to_string());
                let mut frontier: Vec<String> = Vec::new();
                // First step
                let mut first_visited = HashSet::new();
                for obj in inner.eval_from(subject, store, &mut first_visited) {
                    if result_set.insert(obj.clone()) {
                        results.push(obj.clone());
                        frontier.push(obj);
                    }
                }
                // Subsequent steps (BFS)
                while !frontier.is_empty() {
                    let mut next_frontier = Vec::new();
                    for node in &frontier {
                        if visited.contains(node) {
                            continue;
                        }
                        visited.insert(node.clone());
                        let mut node_visited = HashSet::new();
                        for obj in inner.eval_from(node, store, &mut node_visited) {
                            if result_set.insert(obj.clone()) {
                                results.push(obj.clone());
                                next_frontier.push(obj);
                            }
                        }
                    }
                    frontier = next_frontier;
                }
                results
            }

            PropertyPath::ZeroOrOne(inner) => {
                // 0 or 1 steps
                let mut results: Vec<String> = vec![subject.to_string()];
                let mut result_set: HashSet<String> = HashSet::new();
                result_set.insert(subject.to_string());
                let mut pv = HashSet::new();
                for obj in inner.eval_from(subject, store, &mut pv) {
                    if result_set.insert(obj.clone()) {
                        results.push(obj);
                    }
                }
                results
            }

            PropertyPath::NegatedSet(excluded_iris) => {
                // Any predicate NOT in the excluded set
                store
                    .all_triples()
                    .filter(|t| t.subject == subject && !excluded_iris.contains(&t.predicate))
                    .map(|t| t.object.clone())
                    .collect()
            }
        }
    }

    /// Check if this path is a simple predicate match (used in inverse evaluation).
    fn matches_predicate(&self, predicate: &str) -> bool {
        match self {
            PropertyPath::Iri(iri) => iri == predicate,
            PropertyPath::Alternative(l, r) => {
                l.matches_predicate(predicate) || r.matches_predicate(predicate)
            }
            _ => false,
        }
    }

    /// Find all subjects that can reach `object` via this path.
    pub(crate) fn evaluate_reverse(&self, object: &str, store: &OxiRSStore) -> Vec<String> {
        let inverse = PropertyPath::Inverse(Box::new(self.clone()));
        inverse.evaluate(object, store)
    }
}

// -----------------------------------------------------------------------
// Property path parser
// -----------------------------------------------------------------------

/// Parse a property path string into a [`PropertyPath`].
///
/// Supports: `<iri>`, `path / path`, `path | path`, `path*`, `path+`, `path?`, `^path`
pub(crate) fn parse_property_path(s: &str) -> Option<PropertyPath> {
    let s = s.trim();
    parse_path_expr(s)
}

/// Parse at the lowest precedence: alternative `|`
fn parse_path_expr(s: &str) -> Option<PropertyPath> {
    // Find top-level `|`
    if let Some(pos) = find_top_level_char(s, '|') {
        let left = parse_path_sequence(&s[..pos])?;
        let right = parse_path_expr(&s[pos + 1..])?;
        return Some(PropertyPath::Alternative(Box::new(left), Box::new(right)));
    }
    parse_path_sequence(s)
}

/// Parse sequence `path / path`
fn parse_path_sequence(s: &str) -> Option<PropertyPath> {
    if let Some(pos) = find_top_level_char(s, '/') {
        let left = parse_path_unary(&s[..pos])?;
        let right = parse_path_sequence(&s[pos + 1..])?;
        return Some(PropertyPath::Sequence(Box::new(left), Box::new(right)));
    }
    parse_path_unary(s)
}

/// Parse unary path with quantifiers (`*`, `+`, `?`) and inverse (`^`)
fn parse_path_unary(s: &str) -> Option<PropertyPath> {
    let s = s.trim();

    // Inverse `^`
    if let Some(rest) = s.strip_prefix('^') {
        let inner = parse_path_primary(rest.trim())?;
        return Some(PropertyPath::Inverse(Box::new(inner)));
    }

    // Parenthesized with quantifier: (path)*  (path)+  (path)?
    if s.starts_with('(') {
        let close = find_matching_paren(s, 0)?;
        let inner_str = &s[1..close];
        let inner = parse_path_expr(inner_str)?;
        let suffix = s[close + 1..].trim();
        return Some(match suffix {
            "*" => PropertyPath::ZeroOrMore(Box::new(inner)),
            "+" => PropertyPath::OneOrMore(Box::new(inner)),
            "?" => PropertyPath::ZeroOrOne(Box::new(inner)),
            "" => inner,
            _ => return None,
        });
    }

    // Negated property set: !(iri | iri)
    if s.starts_with("!(") && s.ends_with(')') {
        let inner = &s[2..s.len() - 1];
        let iris: Vec<String> = inner
            .split('|')
            .map(|part| {
                let p = part.trim();
                if p.starts_with('<') && p.ends_with('>') {
                    p[1..p.len() - 1].to_string()
                } else {
                    p.to_string()
                }
            })
            .collect();
        return Some(PropertyPath::NegatedSet(iris));
    }

    // Primary with trailing quantifier
    let (primary_str, suffix) = split_path_quantifier(s);
    let primary = parse_path_primary(primary_str.trim())?;
    Some(match suffix {
        "*" => PropertyPath::ZeroOrMore(Box::new(primary)),
        "+" => PropertyPath::OneOrMore(Box::new(primary)),
        "?" => PropertyPath::ZeroOrOne(Box::new(primary)),
        "" => primary,
        _ => return None,
    })
}

/// Split a path string into (base, quantifier)  e.g. `<ex:knows>*` → (`<ex:knows>`, `*`)
fn split_path_quantifier(s: &str) -> (&str, &str) {
    if let Some(stripped) = s.strip_suffix('*') {
        (stripped, "*")
    } else if let Some(stripped) = s.strip_suffix('+') {
        (stripped, "+")
    } else if s.ends_with('?') && !s.ends_with("(?:") {
        (&s[..s.len() - 1], "?")
    } else {
        (s, "")
    }
}

/// Parse a primary path element: `<iri>` or `(path)`
fn parse_path_primary(s: &str) -> Option<PropertyPath> {
    let s = s.trim();
    if s.starts_with('<') && s.ends_with('>') {
        Some(PropertyPath::Iri(s[1..s.len() - 1].to_string()))
    } else if s.starts_with('(') && s.ends_with(')') {
        parse_path_expr(&s[1..s.len() - 1])
    } else if !s.is_empty() && !s.starts_with('?') && !s.starts_with('"') {
        // Bare IRI without angle brackets
        Some(PropertyPath::Iri(s.to_string()))
    } else {
        None
    }
}

/// Find the first occurrence of `ch` at the top level (not inside `<>` or `()`)
fn find_top_level_char(s: &str, ch: char) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_angle = false;
    for (i, c) in s.chars().enumerate() {
        if in_angle {
            if c == '>' {
                in_angle = false;
            }
            continue;
        }
        match c {
            '<' => in_angle = true,
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
            }
            _ if c == ch && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Find the matching closing paren for an opening paren at `start`
fn find_matching_paren(s: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in s[start..].chars().enumerate() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + i);
                }
            }
            _ => {}
        }
    }
    None
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::OxiRSStore;

    fn make_store() -> OxiRSStore {
        let mut store = OxiRSStore::new();
        // a → b → c → d (chain)
        store.insert("http://a", "http://knows", "http://b");
        store.insert("http://b", "http://knows", "http://c");
        store.insert("http://c", "http://knows", "http://d");
        // a → e via different predicate
        store.insert("http://a", "http://likes", "http://e");
        store
    }

    #[test]
    fn test_simple_iri_path() {
        let store = make_store();
        let path = PropertyPath::Iri("http://knows".into());
        let mut results = path.evaluate("http://a", &store);
        results.sort();
        assert_eq!(results, vec!["http://b"]);
    }

    #[test]
    fn test_sequence_path() {
        let store = make_store();
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Iri("http://knows".into())),
            Box::new(PropertyPath::Iri("http://knows".into())),
        );
        let results = path.evaluate("http://a", &store);
        assert_eq!(results, vec!["http://c"]);
    }

    #[test]
    fn test_alternative_path() {
        let store = make_store();
        let path = PropertyPath::Alternative(
            Box::new(PropertyPath::Iri("http://knows".into())),
            Box::new(PropertyPath::Iri("http://likes".into())),
        );
        let mut results = path.evaluate("http://a", &store);
        results.sort();
        assert_eq!(results, vec!["http://b", "http://e"]);
    }

    #[test]
    fn test_zero_or_more_path() {
        let store = make_store();
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("http://knows".into())));
        let mut results = path.evaluate("http://a", &store);
        results.sort();
        // a, b, c, d (0 steps = a itself, then b, c, d)
        assert!(results.contains(&"http://a".to_string()));
        assert!(results.contains(&"http://b".to_string()));
        assert!(results.contains(&"http://c".to_string()));
        assert!(results.contains(&"http://d".to_string()));
    }

    #[test]
    fn test_one_or_more_path() {
        let store = make_store();
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("http://knows".into())));
        let mut results = path.evaluate("http://a", &store);
        results.sort();
        // b, c, d (at least 1 step, not a itself)
        assert!(!results.contains(&"http://a".to_string()));
        assert!(results.contains(&"http://b".to_string()));
        assert!(results.contains(&"http://c".to_string()));
        assert!(results.contains(&"http://d".to_string()));
    }

    #[test]
    fn test_zero_or_one_path() {
        let store = make_store();
        let path = PropertyPath::ZeroOrOne(Box::new(PropertyPath::Iri("http://knows".into())));
        let mut results = path.evaluate("http://a", &store);
        results.sort();
        // a (zero steps) + b (one step)
        assert!(results.contains(&"http://a".to_string()));
        assert!(results.contains(&"http://b".to_string()));
        assert!(!results.contains(&"http://c".to_string()));
    }

    #[test]
    fn test_inverse_path() {
        let store = make_store();
        let path = PropertyPath::Inverse(Box::new(PropertyPath::Iri("http://knows".into())));
        // Who knows b? → a
        let results = path.evaluate("http://b", &store);
        assert_eq!(results, vec!["http://a"]);
    }

    #[test]
    fn test_negated_set() {
        let store = make_store();
        // All objects of a where predicate != http://knows
        let path = PropertyPath::NegatedSet(vec!["http://knows".into()]);
        let results = path.evaluate("http://a", &store);
        assert_eq!(results, vec!["http://e"]);
    }

    #[test]
    fn test_cycle_detection_zero_or_more() {
        // Build a cycle: a → b → a
        let mut store = OxiRSStore::new();
        store.insert("http://a", "http://p", "http://b");
        store.insert("http://b", "http://p", "http://a");

        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Iri("http://p".into())));
        let mut results = path.evaluate("http://a", &store);
        results.sort();
        // Should terminate and return {a, b}
        results.dedup();
        assert!(results.contains(&"http://a".to_string()));
        assert!(results.contains(&"http://b".to_string()));
    }

    #[test]
    fn test_cycle_detection_one_or_more() {
        let mut store = OxiRSStore::new();
        store.insert("http://a", "http://p", "http://b");
        store.insert("http://b", "http://p", "http://a");

        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Iri("http://p".into())));
        let mut results = path.evaluate("http://a", &store);
        results.sort();
        results.dedup();
        // Should terminate and return {b, a}
        assert!(results.contains(&"http://b".to_string()));
    }

    #[test]
    fn test_parse_simple_iri() {
        let path = parse_property_path("<http://knows>").unwrap();
        assert!(matches!(path, PropertyPath::Iri(_)));
    }

    #[test]
    fn test_parse_zero_or_more() {
        let path = parse_property_path("<http://knows>*").unwrap();
        assert!(matches!(path, PropertyPath::ZeroOrMore(_)));
    }

    #[test]
    fn test_parse_one_or_more() {
        let path = parse_property_path("<http://knows>+").unwrap();
        assert!(matches!(path, PropertyPath::OneOrMore(_)));
    }

    #[test]
    fn test_parse_zero_or_one() {
        let path = parse_property_path("<http://knows>?").unwrap();
        assert!(matches!(path, PropertyPath::ZeroOrOne(_)));
    }

    #[test]
    fn test_parse_alternative() {
        let path = parse_property_path("<http://a> | <http://b>").unwrap();
        assert!(matches!(path, PropertyPath::Alternative(_, _)));
    }

    #[test]
    fn test_parse_sequence() {
        let path = parse_property_path("<http://a> / <http://b>").unwrap();
        assert!(matches!(path, PropertyPath::Sequence(_, _)));
    }

    #[test]
    fn test_parse_inverse() {
        let path = parse_property_path("^<http://knows>").unwrap();
        assert!(matches!(path, PropertyPath::Inverse(_)));
    }

    #[test]
    fn test_sequence_three_hops() {
        let store = make_store();
        let path = PropertyPath::Sequence(
            Box::new(PropertyPath::Iri("http://knows".into())),
            Box::new(PropertyPath::Sequence(
                Box::new(PropertyPath::Iri("http://knows".into())),
                Box::new(PropertyPath::Iri("http://knows".into())),
            )),
        );
        let results = path.evaluate("http://a", &store);
        assert_eq!(results, vec!["http://d"]);
    }

    #[test]
    fn test_evaluate_reverse() {
        let store = make_store();
        let path = PropertyPath::Iri("http://knows".into());
        let mut subjects = path.evaluate_reverse("http://b", &store);
        subjects.sort();
        assert_eq!(subjects, vec!["http://a"]);
    }
}
