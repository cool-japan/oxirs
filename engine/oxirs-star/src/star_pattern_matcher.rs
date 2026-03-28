/// RDF-star pattern matching for nested triple queries.
///
/// Provides pattern matching over RDF-star triples, including variable binding,
/// nested pattern matching, unification, and variable substitution.
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A term in an RDF-star pattern — may be a variable, ground term or a
/// nested quoted-triple pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum StarTerm {
    /// A SPARQL-style variable (without the `?` prefix, e.g. `"subject"`).
    Var(String),
    /// A ground IRI (e.g. `"http://example.org/knows"`).
    Iri(String),
    /// A ground literal (e.g. `"hello"` or `"42"`).
    Literal(String),
    /// A blank node identifier.
    Blank(String),
    /// A nested quoted-triple pattern.
    Nested(Box<StarPattern>),
}

/// A triple-shaped pattern where each position is a `StarTerm`.
#[derive(Debug, Clone, PartialEq)]
pub struct StarPattern {
    pub subject: StarTerm,
    pub predicate: StarTerm,
    pub object: StarTerm,
}

/// A set of variable bindings produced by matching.
pub type Bindings = HashMap<String, String>;

// ---------------------------------------------------------------------------
// Matcher
// ---------------------------------------------------------------------------

/// Stateless RDF-star pattern matcher.
pub struct StarPatternMatcher;

impl StarPatternMatcher {
    /// Try to match `term` against concrete string `value`.
    ///
    /// - If `term` is `Var(name)`:
    ///   - If `name` is already bound in `bindings` to something other than
    ///     `value`, the match fails.
    ///   - Otherwise the variable is bound to `value` and `true` is returned.
    /// - For ground terms the match succeeds iff the value equals the term's
    ///   string representation.
    /// - `Nested` terms cannot be matched against a plain string; returns `false`.
    pub fn match_term(term: &StarTerm, value: &str, bindings: &mut Bindings) -> bool {
        match term {
            StarTerm::Var(name) => {
                if let Some(existing) = bindings.get(name) {
                    existing == value
                } else {
                    bindings.insert(name.clone(), value.to_string());
                    true
                }
            }
            StarTerm::Iri(iri) => iri == value,
            StarTerm::Literal(lit) => lit == value,
            StarTerm::Blank(id) => id == value,
            StarTerm::Nested(_) => false,
        }
    }

    /// Try to match `pattern` against a concrete ground triple `(s, p, o)`.
    ///
    /// Each position is matched independently with `match_term`, accumulating
    /// bindings.  If any position fails the overall match fails and `bindings`
    /// may be partially updated (callers should clone before calling if they
    /// need rollback).
    pub fn match_pattern(
        pattern: &StarPattern,
        s: &str,
        p: &str,
        o: &str,
        bindings: &mut Bindings,
    ) -> bool {
        Self::match_term(&pattern.subject, s, bindings)
            && Self::match_term(&pattern.predicate, p, bindings)
            && Self::match_term(&pattern.object, o, bindings)
    }

    /// Match `pattern` against a nested (quoted) triple represented by the
    /// component strings `nested_s`, `nested_p`, `nested_o`.
    ///
    /// This is a convenience wrapper over `match_pattern` intended for use when
    /// the triple being matched originates from a quoted position.
    pub fn match_nested(
        pattern: &StarPattern,
        nested_s: &str,
        nested_p: &str,
        nested_o: &str,
        bindings: &mut Bindings,
    ) -> bool {
        Self::match_pattern(pattern, nested_s, nested_p, nested_o, bindings)
    }

    /// Collect all variable names that appear anywhere in `pattern`
    /// (including inside `Nested` patterns recursively).
    pub fn variables(pattern: &StarPattern) -> Vec<String> {
        let mut vars: Vec<String> = Vec::new();
        Self::collect_vars_term(&pattern.subject, &mut vars);
        Self::collect_vars_term(&pattern.predicate, &mut vars);
        Self::collect_vars_term(&pattern.object, &mut vars);
        vars
    }

    /// Return `true` when `pattern` contains no variables (all positions are
    /// ground terms or nested ground patterns).
    pub fn is_ground(pattern: &StarPattern) -> bool {
        Self::variables(pattern).is_empty()
    }

    /// Substitute variables in `term` with values from `bindings`.  Variables
    /// not present in `bindings` are returned unchanged.
    pub fn apply_bindings(term: &StarTerm, bindings: &Bindings) -> StarTerm {
        match term {
            StarTerm::Var(name) => {
                if let Some(val) = bindings.get(name) {
                    // Heuristic: IRIs look like URLs or use prefixes
                    if val.starts_with("http://")
                        || val.starts_with("https://")
                        || val.contains(':')
                    {
                        StarTerm::Iri(val.clone())
                    } else {
                        StarTerm::Literal(val.clone())
                    }
                } else {
                    term.clone()
                }
            }
            StarTerm::Nested(inner) => {
                let subst = StarPattern {
                    subject: Self::apply_bindings(&inner.subject, bindings),
                    predicate: Self::apply_bindings(&inner.predicate, bindings),
                    object: Self::apply_bindings(&inner.object, bindings),
                };
                StarTerm::Nested(Box::new(subst))
            }
            other => other.clone(),
        }
    }

    /// Attempt to unify two patterns into a common binding set.
    ///
    /// Returns `Some(bindings)` if the patterns can be unified, `None` if they
    /// contain incompatible ground terms.
    pub fn unify(pattern: &StarPattern, other: &StarPattern) -> Option<Bindings> {
        let mut bindings = Bindings::new();
        if !Self::unify_term(&pattern.subject, &other.subject, &mut bindings) {
            return None;
        }
        if !Self::unify_term(&pattern.predicate, &other.predicate, &mut bindings) {
            return None;
        }
        if !Self::unify_term(&pattern.object, &other.object, &mut bindings) {
            return None;
        }
        Some(bindings)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn collect_vars_term(term: &StarTerm, vars: &mut Vec<String>) {
        match term {
            StarTerm::Var(name) if !vars.contains(name) => {
                vars.push(name.clone());
            }
            StarTerm::Nested(inner) => {
                Self::collect_vars_term(&inner.subject, vars);
                Self::collect_vars_term(&inner.predicate, vars);
                Self::collect_vars_term(&inner.object, vars);
            }
            _ => {}
        }
    }

    /// Unify two terms, updating `bindings`.
    fn unify_term(a: &StarTerm, b: &StarTerm, bindings: &mut Bindings) -> bool {
        match (a, b) {
            (StarTerm::Var(name_a), StarTerm::Var(name_b)) => {
                if name_a == name_b {
                    true
                } else {
                    // Bind one to the other (using a placeholder value)
                    bindings
                        .entry(name_a.clone())
                        .or_insert_with(|| format!("?{name_b}"));
                    true
                }
            }
            (StarTerm::Var(name), other) | (other, StarTerm::Var(name)) => {
                let val = Self::term_to_string(other);
                if let Some(existing) = bindings.get(name) {
                    existing == &val
                } else {
                    bindings.insert(name.clone(), val);
                    true
                }
            }
            (StarTerm::Iri(a), StarTerm::Iri(b)) => a == b,
            (StarTerm::Literal(a), StarTerm::Literal(b)) => a == b,
            (StarTerm::Blank(a), StarTerm::Blank(b)) => a == b,
            (StarTerm::Nested(pa), StarTerm::Nested(pb)) => {
                Self::unify_term(&pa.subject, &pb.subject, bindings)
                    && Self::unify_term(&pa.predicate, &pb.predicate, bindings)
                    && Self::unify_term(&pa.object, &pb.object, bindings)
            }
            _ => false,
        }
    }

    fn term_to_string(term: &StarTerm) -> String {
        match term {
            StarTerm::Iri(s) | StarTerm::Literal(s) | StarTerm::Blank(s) | StarTerm::Var(s) => {
                s.clone()
            }
            StarTerm::Nested(_) => "<nested>".to_string(),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> Bindings {
        Bindings::new()
    }

    fn ground_pattern(s: &str, p: &str, o: &str) -> StarPattern {
        StarPattern {
            subject: StarTerm::Iri(s.to_string()),
            predicate: StarTerm::Iri(p.to_string()),
            object: StarTerm::Iri(o.to_string()),
        }
    }

    fn var_pattern(s: &str, p: &str, o: &str) -> StarPattern {
        StarPattern {
            subject: StarTerm::Var(s.to_string()),
            predicate: StarTerm::Var(p.to_string()),
            object: StarTerm::Var(o.to_string()),
        }
    }

    // --- match_term ---
    #[test]
    fn test_match_term_var_unbound_binds() {
        let mut b = empty();
        assert!(StarPatternMatcher::match_term(
            &StarTerm::Var("x".to_string()),
            "value",
            &mut b
        ));
        assert_eq!(b.get("x").map(String::as_str), Some("value"));
    }

    #[test]
    fn test_match_term_var_already_bound_same() {
        let mut b: Bindings = [("x".to_string(), "value".to_string())]
            .into_iter()
            .collect();
        assert!(StarPatternMatcher::match_term(
            &StarTerm::Var("x".to_string()),
            "value",
            &mut b
        ));
    }

    #[test]
    fn test_match_term_var_already_bound_different() {
        let mut b: Bindings = [("x".to_string(), "old".to_string())].into_iter().collect();
        assert!(!StarPatternMatcher::match_term(
            &StarTerm::Var("x".to_string()),
            "new",
            &mut b
        ));
    }

    #[test]
    fn test_match_term_iri_match() {
        let mut b = empty();
        assert!(StarPatternMatcher::match_term(
            &StarTerm::Iri("http://example.org/".to_string()),
            "http://example.org/",
            &mut b
        ));
    }

    #[test]
    fn test_match_term_iri_no_match() {
        let mut b = empty();
        assert!(!StarPatternMatcher::match_term(
            &StarTerm::Iri("http://a/".to_string()),
            "http://b/",
            &mut b
        ));
    }

    #[test]
    fn test_match_term_literal_match() {
        let mut b = empty();
        assert!(StarPatternMatcher::match_term(
            &StarTerm::Literal("hello".to_string()),
            "hello",
            &mut b
        ));
    }

    #[test]
    fn test_match_term_blank_match() {
        let mut b = empty();
        assert!(StarPatternMatcher::match_term(
            &StarTerm::Blank("b1".to_string()),
            "b1",
            &mut b
        ));
    }

    #[test]
    fn test_match_term_nested_returns_false() {
        let mut b = empty();
        let nested = StarTerm::Nested(Box::new(ground_pattern("s", "p", "o")));
        assert!(!StarPatternMatcher::match_term(&nested, "anything", &mut b));
    }

    // --- match_pattern ---
    #[test]
    fn test_match_pattern_all_variables() {
        let pattern = var_pattern("s", "p", "o");
        let mut b = empty();
        assert!(StarPatternMatcher::match_pattern(
            &pattern, "S", "P", "O", &mut b
        ));
        assert_eq!(b.get("s").map(String::as_str), Some("S"));
        assert_eq!(b.get("p").map(String::as_str), Some("P"));
        assert_eq!(b.get("o").map(String::as_str), Some("O"));
    }

    #[test]
    fn test_match_pattern_all_ground_success() {
        let pattern = ground_pattern("http://s/", "http://p/", "http://o/");
        let mut b = empty();
        assert!(StarPatternMatcher::match_pattern(
            &pattern,
            "http://s/",
            "http://p/",
            "http://o/",
            &mut b
        ));
    }

    #[test]
    fn test_match_pattern_ground_fail() {
        let pattern = ground_pattern("http://s/", "http://p/", "http://o/");
        let mut b = empty();
        assert!(!StarPatternMatcher::match_pattern(
            &pattern,
            "http://s/",
            "http://p/",
            "http://X/",
            &mut b
        ));
    }

    #[test]
    fn test_match_pattern_mixed() {
        let pattern = StarPattern {
            subject: StarTerm::Iri("http://s/".to_string()),
            predicate: StarTerm::Var("p".to_string()),
            object: StarTerm::Var("o".to_string()),
        };
        let mut b = empty();
        assert!(StarPatternMatcher::match_pattern(
            &pattern,
            "http://s/",
            "http://knows/",
            "Alice",
            &mut b
        ));
        assert_eq!(b.get("p").map(String::as_str), Some("http://knows/"));
        assert_eq!(b.get("o").map(String::as_str), Some("Alice"));
    }

    // --- match_nested ---
    #[test]
    fn test_match_nested_delegates_to_match_pattern() {
        let pattern = var_pattern("s", "p", "o");
        let mut b = empty();
        assert!(StarPatternMatcher::match_nested(
            &pattern, "S", "P", "O", &mut b
        ));
        assert_eq!(b.get("s").map(String::as_str), Some("S"));
    }

    // --- variables ---
    #[test]
    fn test_variables_all_vars() {
        let pattern = var_pattern("s", "p", "o");
        let mut vars = StarPatternMatcher::variables(&pattern);
        vars.sort();
        assert_eq!(vars, vec!["o", "p", "s"]);
    }

    #[test]
    fn test_variables_no_vars() {
        let pattern = ground_pattern("s", "p", "o");
        assert!(StarPatternMatcher::variables(&pattern).is_empty());
    }

    #[test]
    fn test_variables_partial_vars() {
        let pattern = StarPattern {
            subject: StarTerm::Var("s".to_string()),
            predicate: StarTerm::Iri("http://p/".to_string()),
            object: StarTerm::Var("o".to_string()),
        };
        let mut vars = StarPatternMatcher::variables(&pattern);
        vars.sort();
        assert_eq!(vars, vec!["o", "s"]);
    }

    #[test]
    fn test_variables_nested() {
        let inner = ground_pattern("a", "b", "c");
        let pattern = StarPattern {
            subject: StarTerm::Nested(Box::new(inner)),
            predicate: StarTerm::Iri("http://p/".to_string()),
            object: StarTerm::Var("x".to_string()),
        };
        let vars = StarPatternMatcher::variables(&pattern);
        assert!(vars.contains(&"x".to_string()));
    }

    // --- is_ground ---
    #[test]
    fn test_is_ground_all_iri() {
        assert!(StarPatternMatcher::is_ground(&ground_pattern(
            "s", "p", "o"
        )));
    }

    #[test]
    fn test_is_ground_false_with_var() {
        let pattern = StarPattern {
            subject: StarTerm::Var("s".to_string()),
            predicate: StarTerm::Iri("p".to_string()),
            object: StarTerm::Iri("o".to_string()),
        };
        assert!(!StarPatternMatcher::is_ground(&pattern));
    }

    // --- apply_bindings ---
    #[test]
    fn test_apply_bindings_iri_substitution() {
        let mut b = Bindings::new();
        b.insert("x".to_string(), "http://example.org/Bob".to_string());
        let term = StarTerm::Var("x".to_string());
        let result = StarPatternMatcher::apply_bindings(&term, &b);
        assert_eq!(result, StarTerm::Iri("http://example.org/Bob".to_string()));
    }

    #[test]
    fn test_apply_bindings_literal_substitution() {
        let mut b = Bindings::new();
        b.insert("x".to_string(), "hello".to_string());
        let term = StarTerm::Var("x".to_string());
        let result = StarPatternMatcher::apply_bindings(&term, &b);
        assert_eq!(result, StarTerm::Literal("hello".to_string()));
    }

    #[test]
    fn test_apply_bindings_unbound_var_unchanged() {
        let b = Bindings::new();
        let term = StarTerm::Var("x".to_string());
        let result = StarPatternMatcher::apply_bindings(&term, &b);
        assert_eq!(result, StarTerm::Var("x".to_string()));
    }

    #[test]
    fn test_apply_bindings_iri_passthrough() {
        let b = Bindings::new();
        let term = StarTerm::Iri("http://x/".to_string());
        let result = StarPatternMatcher::apply_bindings(&term, &b);
        assert_eq!(result, StarTerm::Iri("http://x/".to_string()));
    }

    // --- unify ---
    #[test]
    fn test_unify_two_vars() {
        let a = var_pattern("s", "p", "o");
        let b = ground_pattern("S", "P", "O");
        let bindings = StarPatternMatcher::unify(&a, &b).unwrap();
        assert_eq!(bindings.get("s").map(String::as_str), Some("S"));
        assert_eq!(bindings.get("p").map(String::as_str), Some("P"));
        assert_eq!(bindings.get("o").map(String::as_str), Some("O"));
    }

    #[test]
    fn test_unify_identical_ground() {
        let a = ground_pattern("S", "P", "O");
        let b = ground_pattern("S", "P", "O");
        let bindings = StarPatternMatcher::unify(&a, &b);
        assert!(bindings.is_some());
    }

    #[test]
    fn test_unify_incompatible_ground() {
        let a = ground_pattern("S", "P", "O1");
        let b = ground_pattern("S", "P", "O2");
        assert!(StarPatternMatcher::unify(&a, &b).is_none());
    }

    #[test]
    fn test_unify_partial_vars() {
        let a = StarPattern {
            subject: StarTerm::Var("s".to_string()),
            predicate: StarTerm::Iri("http://p/".to_string()),
            object: StarTerm::Var("o".to_string()),
        };
        let b = StarPattern {
            subject: StarTerm::Iri("http://s/".to_string()),
            predicate: StarTerm::Iri("http://p/".to_string()),
            object: StarTerm::Literal("42".to_string()),
        };
        let bindings = StarPatternMatcher::unify(&a, &b).unwrap();
        assert_eq!(bindings.get("s").map(String::as_str), Some("http://s/"));
        assert_eq!(bindings.get("o").map(String::as_str), Some("42"));
    }
}
