//! SPARQL 1.1 property path expression parser and evaluator.
//!
//! This module provides a representation for SPARQL property path expressions
//! as defined in the SPARQL 1.1 specification, along with utilities for
//! analysis and serialisation.

use std::collections::BTreeSet;

/// A SPARQL property path expression.
///
/// Represents the full set of SPARQL 1.1 property path operators,
/// enabling in-memory construction, analysis, and serialisation without
/// requiring a triple store.
#[derive(Debug, Clone, PartialEq)]
pub enum PathExpr {
    /// A single IRI step.
    Iri(String),
    /// Inverse path: `^p`.
    Inverse(Box<PathExpr>),
    /// Sequence: `p/q`.
    Sequence(Box<PathExpr>, Box<PathExpr>),
    /// Alternative: `p|q`.
    Alternative(Box<PathExpr>, Box<PathExpr>),
    /// Zero-or-more: `p*`.
    ZeroOrMore(Box<PathExpr>),
    /// One-or-more: `p+`.
    OneOrMore(Box<PathExpr>),
    /// Zero-or-one: `p?`.
    ZeroOrOne(Box<PathExpr>),
    /// Negated property set: `!(p|q)`.
    NegatedSet(Vec<String>),
}

impl PathExpr {
    /// Create a single IRI step path expression.
    pub fn iri(iri: &str) -> Self {
        PathExpr::Iri(iri.to_string())
    }

    /// Create an inverse path expression (`^p`).
    pub fn inverse(inner: PathExpr) -> Self {
        PathExpr::Inverse(Box::new(inner))
    }

    /// Create a sequence path expression (`p/q`).
    pub fn sequence(left: PathExpr, right: PathExpr) -> Self {
        PathExpr::Sequence(Box::new(left), Box::new(right))
    }

    /// Create an alternative path expression (`p|q`).
    pub fn alternative(left: PathExpr, right: PathExpr) -> Self {
        PathExpr::Alternative(Box::new(left), Box::new(right))
    }

    /// Create a zero-or-more repetition path expression (`p*`).
    pub fn zero_or_more(inner: PathExpr) -> Self {
        PathExpr::ZeroOrMore(Box::new(inner))
    }

    /// Create a one-or-more repetition path expression (`p+`).
    pub fn one_or_more(inner: PathExpr) -> Self {
        PathExpr::OneOrMore(Box::new(inner))
    }

    /// Create a zero-or-one path expression (`p?`).
    pub fn zero_or_one(inner: PathExpr) -> Self {
        PathExpr::ZeroOrOne(Box::new(inner))
    }

    /// Create a negated property set path expression (`!(p|q|...)`).
    ///
    /// Accepts a slice of IRI strings.
    pub fn negated_set(iris: &[&str]) -> Self {
        PathExpr::NegatedSet(iris.iter().map(|s| s.to_string()).collect())
    }

    /// Returns the nesting depth / complexity of the path expression.
    ///
    /// A single IRI or negated set has depth 1. Each wrapping operator adds 1.
    pub fn depth(&self) -> usize {
        match self {
            PathExpr::Iri(_) => 1,
            PathExpr::NegatedSet(_) => 1,
            PathExpr::Inverse(inner) => 1 + inner.depth(),
            PathExpr::ZeroOrMore(inner) => 1 + inner.depth(),
            PathExpr::OneOrMore(inner) => 1 + inner.depth(),
            PathExpr::ZeroOrOne(inner) => 1 + inner.depth(),
            PathExpr::Sequence(left, right) => 1 + left.depth().max(right.depth()),
            PathExpr::Alternative(left, right) => 1 + left.depth().max(right.depth()),
        }
    }

    /// Returns all unique IRIs referenced in the path expression, in sorted order.
    pub fn iris(&self) -> Vec<String> {
        let mut set = BTreeSet::new();
        self.collect_iris(&mut set);
        set.into_iter().collect()
    }

    /// Internal recursive IRI collector.
    fn collect_iris(&self, set: &mut BTreeSet<String>) {
        match self {
            PathExpr::Iri(iri) => {
                set.insert(iri.clone());
            }
            PathExpr::NegatedSet(iris) => {
                for iri in iris {
                    set.insert(iri.clone());
                }
            }
            PathExpr::Inverse(inner) => inner.collect_iris(set),
            PathExpr::ZeroOrMore(inner) => inner.collect_iris(set),
            PathExpr::OneOrMore(inner) => inner.collect_iris(set),
            PathExpr::ZeroOrOne(inner) => inner.collect_iris(set),
            PathExpr::Sequence(left, right) => {
                left.collect_iris(set);
                right.collect_iris(set);
            }
            PathExpr::Alternative(left, right) => {
                left.collect_iris(set);
                right.collect_iris(set);
            }
        }
    }

    /// Returns `true` if this path expression can match zero steps (i.e., is nullable).
    ///
    /// - `*` and `?` operators are always nullable.
    /// - `+` is nullable iff its inner expression is nullable (it cannot be nullable
    ///   unless inner is, but `p+` itself requires ≥1 match).
    /// - Sequences are nullable only if both arms are nullable.
    /// - Alternatives are nullable if either arm is nullable.
    /// - Plain IRIs and negated sets are not nullable.
    /// - Inverse is nullable iff its inner is nullable.
    pub fn can_match_zero(&self) -> bool {
        match self {
            PathExpr::Iri(_) => false,
            PathExpr::NegatedSet(_) => false,
            PathExpr::ZeroOrMore(_) => true,
            PathExpr::ZeroOrOne(_) => true,
            PathExpr::OneOrMore(inner) => inner.can_match_zero(),
            PathExpr::Inverse(inner) => inner.can_match_zero(),
            PathExpr::Sequence(left, right) => left.can_match_zero() && right.can_match_zero(),
            PathExpr::Alternative(left, right) => left.can_match_zero() || right.can_match_zero(),
        }
    }

    /// Converts the path expression to its SPARQL 1.1 string representation.
    ///
    /// Parentheses are inserted to make operator precedence explicit.
    pub fn to_sparql(&self) -> String {
        match self {
            PathExpr::Iri(iri) => iri.clone(),
            PathExpr::Inverse(inner) => format!("^({})", inner.to_sparql()),
            PathExpr::Sequence(left, right) => {
                format!("({}/{})", left.to_sparql(), right.to_sparql())
            }
            PathExpr::Alternative(left, right) => {
                format!("({}|{})", left.to_sparql(), right.to_sparql())
            }
            PathExpr::ZeroOrMore(inner) => format!("({})*", inner.to_sparql()),
            PathExpr::OneOrMore(inner) => format!("({})+", inner.to_sparql()),
            PathExpr::ZeroOrOne(inner) => format!("({})?", inner.to_sparql()),
            PathExpr::NegatedSet(iris) => {
                if iris.is_empty() {
                    "!()".to_string()
                } else if iris.len() == 1 {
                    format!("!{}", iris[0])
                } else {
                    format!("!({})", iris.join("|"))
                }
            }
        }
    }

    /// Returns `true` if the path is a plain IRI step (no operators).
    pub fn is_simple_iri(&self) -> bool {
        matches!(self, PathExpr::Iri(_))
    }

    /// Returns the IRI string if this is a simple IRI step, otherwise `None`.
    pub fn as_iri(&self) -> Option<&str> {
        if let PathExpr::Iri(iri) = self {
            Some(iri.as_str())
        } else {
            None
        }
    }

    /// Counts the total number of IRI references (including duplicates) in the path.
    pub fn iri_count(&self) -> usize {
        match self {
            PathExpr::Iri(_) => 1,
            PathExpr::NegatedSet(iris) => iris.len(),
            PathExpr::Inverse(inner) => inner.iri_count(),
            PathExpr::ZeroOrMore(inner) => inner.iri_count(),
            PathExpr::OneOrMore(inner) => inner.iri_count(),
            PathExpr::ZeroOrOne(inner) => inner.iri_count(),
            PathExpr::Sequence(left, right) => left.iri_count() + right.iri_count(),
            PathExpr::Alternative(left, right) => left.iri_count() + right.iri_count(),
        }
    }
}

impl std::fmt::Display for PathExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sparql())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Constructor tests ---

    #[test]
    fn test_iri_constructor() {
        let p = PathExpr::iri("http://example.org/p");
        assert_eq!(p, PathExpr::Iri("http://example.org/p".to_string()));
    }

    #[test]
    fn test_inverse_constructor() {
        let p = PathExpr::inverse(PathExpr::iri(":p"));
        assert!(matches!(p, PathExpr::Inverse(_)));
    }

    #[test]
    fn test_sequence_constructor() {
        let s = PathExpr::sequence(PathExpr::iri(":p"), PathExpr::iri(":q"));
        assert!(matches!(s, PathExpr::Sequence(_, _)));
    }

    #[test]
    fn test_alternative_constructor() {
        let a = PathExpr::alternative(PathExpr::iri(":p"), PathExpr::iri(":q"));
        assert!(matches!(a, PathExpr::Alternative(_, _)));
    }

    #[test]
    fn test_zero_or_more_constructor() {
        let z = PathExpr::zero_or_more(PathExpr::iri(":p"));
        assert!(matches!(z, PathExpr::ZeroOrMore(_)));
    }

    #[test]
    fn test_one_or_more_constructor() {
        let o = PathExpr::one_or_more(PathExpr::iri(":p"));
        assert!(matches!(o, PathExpr::OneOrMore(_)));
    }

    #[test]
    fn test_zero_or_one_constructor() {
        let z = PathExpr::zero_or_one(PathExpr::iri(":p"));
        assert!(matches!(z, PathExpr::ZeroOrOne(_)));
    }

    #[test]
    fn test_negated_set_constructor() {
        let n = PathExpr::negated_set(&[":p", ":q"]);
        assert!(matches!(n, PathExpr::NegatedSet(_)));
        if let PathExpr::NegatedSet(iris) = &n {
            assert_eq!(iris.len(), 2);
        }
    }

    #[test]
    fn test_negated_set_empty() {
        let n = PathExpr::negated_set(&[]);
        if let PathExpr::NegatedSet(iris) = &n {
            assert!(iris.is_empty());
        }
    }

    #[test]
    fn test_negated_set_single() {
        let n = PathExpr::negated_set(&[":p"]);
        if let PathExpr::NegatedSet(iris) = &n {
            assert_eq!(iris.len(), 1);
            assert_eq!(iris[0], ":p");
        }
    }

    // --- depth() tests ---

    #[test]
    fn test_depth_iri() {
        assert_eq!(PathExpr::iri(":p").depth(), 1);
    }

    #[test]
    fn test_depth_negated_set() {
        assert_eq!(PathExpr::negated_set(&[":p"]).depth(), 1);
    }

    #[test]
    fn test_depth_inverse() {
        let p = PathExpr::inverse(PathExpr::iri(":p"));
        assert_eq!(p.depth(), 2);
    }

    #[test]
    fn test_depth_zero_or_more() {
        let p = PathExpr::zero_or_more(PathExpr::iri(":p"));
        assert_eq!(p.depth(), 2);
    }

    #[test]
    fn test_depth_one_or_more() {
        let p = PathExpr::one_or_more(PathExpr::iri(":p"));
        assert_eq!(p.depth(), 2);
    }

    #[test]
    fn test_depth_zero_or_one() {
        let p = PathExpr::zero_or_one(PathExpr::iri(":p"));
        assert_eq!(p.depth(), 2);
    }

    #[test]
    fn test_depth_sequence() {
        // depth = 1 + max(1,1) = 2
        let p = PathExpr::sequence(PathExpr::iri(":p"), PathExpr::iri(":q"));
        assert_eq!(p.depth(), 2);
    }

    #[test]
    fn test_depth_alternative() {
        let p = PathExpr::alternative(PathExpr::iri(":p"), PathExpr::iri(":q"));
        assert_eq!(p.depth(), 2);
    }

    #[test]
    fn test_depth_nested_three_levels() {
        // inverse(sequence(iri, iri)) → depth 3
        let seq = PathExpr::sequence(PathExpr::iri(":p"), PathExpr::iri(":q"));
        let p = PathExpr::inverse(seq);
        assert_eq!(p.depth(), 3);
    }

    #[test]
    fn test_depth_deeply_nested() {
        // zero_or_more(one_or_more(inverse(iri))) → 4
        let inner = PathExpr::inverse(PathExpr::iri(":x"));
        let mid = PathExpr::one_or_more(inner);
        let outer = PathExpr::zero_or_more(mid);
        assert_eq!(outer.depth(), 4);
    }

    #[test]
    fn test_depth_asymmetric_sequence() {
        // sequence(iri, zero_or_more(iri)) → 1 + max(1,2) = 3
        let p = PathExpr::sequence(
            PathExpr::iri(":p"),
            PathExpr::zero_or_more(PathExpr::iri(":q")),
        );
        assert_eq!(p.depth(), 3);
    }

    // --- iris() tests ---

    #[test]
    fn test_iris_single_iri() {
        let p = PathExpr::iri(":a");
        assert_eq!(p.iris(), vec![":a".to_string()]);
    }

    #[test]
    fn test_iris_sequence_two() {
        let p = PathExpr::sequence(PathExpr::iri(":a"), PathExpr::iri(":b"));
        let mut iris = p.iris();
        iris.sort();
        assert!(iris.contains(&":a".to_string()));
        assert!(iris.contains(&":b".to_string()));
    }

    #[test]
    fn test_iris_deduplicates() {
        // alternative(:a|:a) should return only one :a
        let p = PathExpr::alternative(PathExpr::iri(":a"), PathExpr::iri(":a"));
        assert_eq!(p.iris(), vec![":a".to_string()]);
    }

    #[test]
    fn test_iris_negated_set() {
        let p = PathExpr::negated_set(&[":x", ":y"]);
        let mut iris = p.iris();
        iris.sort();
        assert_eq!(iris, vec![":x".to_string(), ":y".to_string()]);
    }

    #[test]
    fn test_iris_sorted_order() {
        let p = PathExpr::sequence(PathExpr::iri("z:c"), PathExpr::iri("a:b"));
        let iris = p.iris();
        // BTreeSet gives sorted output
        assert_eq!(iris, vec!["a:b".to_string(), "z:c".to_string()]);
    }

    #[test]
    fn test_iris_negated_set_empty() {
        let p = PathExpr::negated_set(&[]);
        assert!(p.iris().is_empty());
    }

    // --- can_match_zero() tests ---

    #[test]
    fn test_can_match_zero_iri_false() {
        assert!(!PathExpr::iri(":p").can_match_zero());
    }

    #[test]
    fn test_can_match_zero_negated_set_false() {
        assert!(!PathExpr::negated_set(&[":p"]).can_match_zero());
    }

    #[test]
    fn test_can_match_zero_star_true() {
        assert!(PathExpr::zero_or_more(PathExpr::iri(":p")).can_match_zero());
    }

    #[test]
    fn test_can_match_zero_question_mark_true() {
        assert!(PathExpr::zero_or_one(PathExpr::iri(":p")).can_match_zero());
    }

    #[test]
    fn test_can_match_zero_plus_non_nullable_inner() {
        // p+ where p is IRI: cannot match zero
        assert!(!PathExpr::one_or_more(PathExpr::iri(":p")).can_match_zero());
    }

    #[test]
    fn test_can_match_zero_plus_nullable_inner() {
        // (p*)+ → inner is nullable so result is nullable
        let star = PathExpr::zero_or_more(PathExpr::iri(":p"));
        assert!(PathExpr::one_or_more(star).can_match_zero());
    }

    #[test]
    fn test_can_match_zero_sequence_both_nullable() {
        let s = PathExpr::sequence(
            PathExpr::zero_or_more(PathExpr::iri(":p")),
            PathExpr::zero_or_one(PathExpr::iri(":q")),
        );
        assert!(s.can_match_zero());
    }

    #[test]
    fn test_can_match_zero_sequence_one_not_nullable() {
        let s = PathExpr::sequence(
            PathExpr::zero_or_more(PathExpr::iri(":p")),
            PathExpr::iri(":q"),
        );
        assert!(!s.can_match_zero());
    }

    #[test]
    fn test_can_match_zero_alternative_one_nullable() {
        let a = PathExpr::alternative(
            PathExpr::iri(":p"),
            PathExpr::zero_or_more(PathExpr::iri(":q")),
        );
        assert!(a.can_match_zero());
    }

    #[test]
    fn test_can_match_zero_alternative_none_nullable() {
        let a = PathExpr::alternative(PathExpr::iri(":p"), PathExpr::iri(":q"));
        assert!(!a.can_match_zero());
    }

    #[test]
    fn test_can_match_zero_inverse_non_nullable() {
        assert!(!PathExpr::inverse(PathExpr::iri(":p")).can_match_zero());
    }

    #[test]
    fn test_can_match_zero_inverse_of_star() {
        let star = PathExpr::zero_or_more(PathExpr::iri(":p"));
        assert!(PathExpr::inverse(star).can_match_zero());
    }

    // --- to_sparql() tests ---

    #[test]
    fn test_to_sparql_iri() {
        assert_eq!(PathExpr::iri(":p").to_sparql(), ":p");
    }

    #[test]
    fn test_to_sparql_inverse() {
        let p = PathExpr::inverse(PathExpr::iri(":p"));
        assert_eq!(p.to_sparql(), "^(:p)");
    }

    #[test]
    fn test_to_sparql_sequence() {
        let p = PathExpr::sequence(PathExpr::iri(":p"), PathExpr::iri(":q"));
        assert_eq!(p.to_sparql(), "(:p/:q)");
    }

    #[test]
    fn test_to_sparql_alternative() {
        let p = PathExpr::alternative(PathExpr::iri(":p"), PathExpr::iri(":q"));
        assert_eq!(p.to_sparql(), "(:p|:q)");
    }

    #[test]
    fn test_to_sparql_zero_or_more() {
        let p = PathExpr::zero_or_more(PathExpr::iri(":p"));
        assert_eq!(p.to_sparql(), "(:p)*");
    }

    #[test]
    fn test_to_sparql_one_or_more() {
        let p = PathExpr::one_or_more(PathExpr::iri(":p"));
        assert_eq!(p.to_sparql(), "(:p)+");
    }

    #[test]
    fn test_to_sparql_zero_or_one() {
        let p = PathExpr::zero_or_one(PathExpr::iri(":p"));
        assert_eq!(p.to_sparql(), "(:p)?");
    }

    #[test]
    fn test_to_sparql_negated_set_empty() {
        let p = PathExpr::negated_set(&[]);
        assert_eq!(p.to_sparql(), "!()");
    }

    #[test]
    fn test_to_sparql_negated_set_single() {
        let p = PathExpr::negated_set(&[":p"]);
        assert_eq!(p.to_sparql(), "!:p");
    }

    #[test]
    fn test_to_sparql_negated_set_multiple() {
        let p = PathExpr::negated_set(&[":p", ":q"]);
        assert_eq!(p.to_sparql(), "!(:p|:q)");
    }

    #[test]
    fn test_to_sparql_nested_sequence_of_alternatives() {
        // (:a|:b)/(:c|:d)
        let alt1 = PathExpr::alternative(PathExpr::iri(":a"), PathExpr::iri(":b"));
        let alt2 = PathExpr::alternative(PathExpr::iri(":c"), PathExpr::iri(":d"));
        let seq = PathExpr::sequence(alt1, alt2);
        assert_eq!(seq.to_sparql(), "((:a|:b)/(:c|:d))");
    }

    #[test]
    fn test_to_sparql_round_trip_display() {
        let p = PathExpr::one_or_more(PathExpr::inverse(PathExpr::iri(":knows")));
        let s = p.to_sparql();
        assert!(s.contains("knows"));
        assert!(s.contains('+'));
        assert!(s.contains('^'));
    }

    // --- Nested / complex path tests ---

    #[test]
    fn test_nested_sequence_depth() {
        // ((a/b)/c) → depth 3
        let ab = PathExpr::sequence(PathExpr::iri(":a"), PathExpr::iri(":b"));
        let abc = PathExpr::sequence(ab, PathExpr::iri(":c"));
        assert_eq!(abc.depth(), 3);
    }

    #[test]
    fn test_nested_alternative_iris() {
        let a = PathExpr::alternative(
            PathExpr::iri(":p"),
            PathExpr::alternative(PathExpr::iri(":q"), PathExpr::iri(":r")),
        );
        let mut iris = a.iris();
        iris.sort();
        assert_eq!(
            iris,
            vec![":p".to_string(), ":q".to_string(), ":r".to_string()]
        );
    }

    #[test]
    fn test_clone_equality() {
        let p =
            PathExpr::zero_or_more(PathExpr::sequence(PathExpr::iri(":a"), PathExpr::iri(":b")));
        assert_eq!(p.clone(), p);
    }

    #[test]
    fn test_is_simple_iri_true() {
        assert!(PathExpr::iri(":p").is_simple_iri());
    }

    #[test]
    fn test_is_simple_iri_false() {
        assert!(!PathExpr::inverse(PathExpr::iri(":p")).is_simple_iri());
    }

    #[test]
    fn test_as_iri_some() {
        let p = PathExpr::iri(":x");
        assert_eq!(p.as_iri(), Some(":x"));
    }

    #[test]
    fn test_as_iri_none_for_inverse() {
        let p = PathExpr::inverse(PathExpr::iri(":x"));
        assert_eq!(p.as_iri(), None);
    }

    #[test]
    fn test_iri_count_sequence() {
        let p = PathExpr::sequence(PathExpr::iri(":a"), PathExpr::iri(":a"));
        assert_eq!(p.iri_count(), 2);
    }

    #[test]
    fn test_iri_count_negated_set() {
        let p = PathExpr::negated_set(&[":a", ":b", ":c"]);
        assert_eq!(p.iri_count(), 3);
    }

    #[test]
    fn test_display_trait() {
        let p = PathExpr::iri(":hello");
        assert_eq!(format!("{}", p), ":hello");
    }

    #[test]
    fn test_deep_nesting_can_match_zero() {
        // sequence(star(iri), star(iri)) → nullable
        let s = PathExpr::sequence(
            PathExpr::zero_or_more(PathExpr::iri(":x")),
            PathExpr::zero_or_more(PathExpr::iri(":y")),
        );
        assert!(s.can_match_zero());
    }

    #[test]
    fn test_alternative_of_sequences_iris() {
        // (a/b) | (c/d)
        let s1 = PathExpr::sequence(PathExpr::iri(":a"), PathExpr::iri(":b"));
        let s2 = PathExpr::sequence(PathExpr::iri(":c"), PathExpr::iri(":d"));
        let alt = PathExpr::alternative(s1, s2);
        let mut iris = alt.iris();
        iris.sort();
        assert_eq!(
            iris,
            vec![
                ":a".to_string(),
                ":b".to_string(),
                ":c".to_string(),
                ":d".to_string()
            ]
        );
    }

    #[test]
    fn test_to_sparql_complex_path() {
        // ^(:a/:b)*
        let seq = PathExpr::sequence(PathExpr::iri(":a"), PathExpr::iri(":b"));
        let star = PathExpr::zero_or_more(seq);
        let inv = PathExpr::inverse(star);
        let sparql = inv.to_sparql();
        assert!(sparql.contains(":a"));
        assert!(sparql.contains(":b"));
        assert!(sparql.contains('*'));
        assert!(sparql.contains('^'));
    }

    #[test]
    fn test_negated_set_three_iris() {
        let n = PathExpr::negated_set(&[":a", ":b", ":c"]);
        if let PathExpr::NegatedSet(iris) = &n {
            assert_eq!(iris.len(), 3);
        } else {
            panic!("Expected NegatedSet");
        }
    }

    #[test]
    fn test_depth_alternative_asymmetric() {
        // alternative(iri, zero_or_more(inverse(iri))) → 1 + max(1,3) = 4
        let deep = PathExpr::zero_or_more(PathExpr::inverse(PathExpr::iri(":x")));
        let alt = PathExpr::alternative(PathExpr::iri(":y"), deep);
        assert_eq!(alt.depth(), 4);
    }
}
