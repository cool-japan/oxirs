//! SPARQL ASK query evaluation engine.
//!
//! Provides an in-memory triple store and an evaluator that checks whether
//! a SPARQL-style graph pattern has at least one solution.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// A set of variable→value bindings produced by pattern matching.
pub type Binding = HashMap<String, String>;

/// A graph pattern used in ASK evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum AskPattern {
    /// A basic triple pattern.  Subject / predicate / object strings that
    /// start with `?` are treated as variables.
    TriplePattern { s: String, p: String, o: String },
    /// A FILTER constraint applied to the incoming bindings.
    Filter(String),
    /// An optional (LEFT JOIN) sub-pattern.
    Optional(Box<AskPattern>),
    /// All sub-patterns must hold simultaneously (conjunction).
    And(Vec<AskPattern>),
    /// At least one sub-pattern must hold (disjunction).
    Union(Vec<AskPattern>),
}

// ────────────────────────────────────────────────────────────────────────────
// TripleStore
// ────────────────────────────────────────────────────────────────────────────

/// Minimal in-memory triple store.
#[derive(Debug, Default, Clone)]
pub struct TripleStore {
    triples: Vec<(String, String, String)>,
}

impl TripleStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a triple.
    pub fn insert(&mut self, s: &str, p: &str, o: &str) {
        self.triples
            .push((s.to_string(), p.to_string(), o.to_string()));
    }

    /// Return the number of triples.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Return `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Match triples against an optional subject / predicate / object.
    ///
    /// - `None`  → wildcard (match any value)
    /// - `Some("?x")` → variable (always matches; captured into the returned binding)
    /// - `Some("literal")` → concrete value (match only equal triples)
    pub fn match_pattern(&self, s: Option<&str>, p: Option<&str>, o: Option<&str>) -> Vec<Binding> {
        let mut results = Vec::new();

        for (ts, tp, to) in &self.triples {
            let mut binding: Binding = HashMap::new();

            if !Self::matches_slot(s, ts, &mut binding, "s") {
                continue;
            }
            if !Self::matches_slot(p, tp, &mut binding, "p") {
                continue;
            }
            if !Self::matches_slot(o, to, &mut binding, "o") {
                continue;
            }

            results.push(binding);
        }

        results
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Check whether a pattern slot matches a concrete triple component.
    ///
    /// `slot_name` is only used to pick a fallback binding key when the
    /// slot is `None` (wildcard without a named variable).
    fn matches_slot(
        slot: Option<&str>,
        value: &str,
        binding: &mut Binding,
        _slot_name: &str,
    ) -> bool {
        match slot {
            None => true, // wildcard – always matches, nothing captured
            Some(s) if s.starts_with('?') => {
                // variable – check consistency
                let var = s.to_string();
                if let Some(existing) = binding.get(&var) {
                    existing == value
                } else {
                    binding.insert(var, value.to_string());
                    true
                }
            }
            Some(s) => s == value,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Filter evaluation
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate a simple filter expression against a binding.
///
/// Supported syntax:
/// - `?x = "value"` / `?x = value`
/// - `?x != "value"` / `?x != value`
/// - `ISIRI(?x)` / `isiri(?x)`
/// - `ISLITERAL(?x)` / `isliteral(?x)`
fn evaluate_filter(expr: &str, binding: &Binding) -> bool {
    let expr = expr.trim();
    let upper = expr.to_uppercase();

    // ISIRI(?x) or isiri(?x)
    if upper.starts_with("ISIRI(") {
        let inner = extract_paren_arg(expr);
        if let Some(val) = resolve_slot(inner.trim(), binding) {
            return is_iri(&val);
        }
        return false;
    }

    // ISLITERAL(?x) or isliteral(?x)
    if upper.starts_with("ISLITERAL(") {
        let inner = extract_paren_arg(expr);
        if let Some(val) = resolve_slot(inner.trim(), binding) {
            return is_literal(&val);
        }
        return false;
    }

    // ?x != "value" or ?x = "value"
    if let Some(idx) = expr.find("!=") {
        let lhs = expr[..idx].trim();
        let rhs = expr[idx + 2..].trim();
        let lhs_val = resolve_slot(lhs, binding);
        let rhs_val = strip_quotes(rhs);
        return match lhs_val {
            Some(v) => v != rhs_val,
            None => true,
        };
    }

    if let Some(idx) = expr.find('=') {
        let lhs = expr[..idx].trim();
        let rhs = expr[idx + 1..].trim();
        let lhs_val = resolve_slot(lhs, binding);
        let rhs_val = strip_quotes(rhs);
        return match lhs_val {
            Some(v) => v == rhs_val,
            None => false,
        };
    }

    // Unknown / unsupported – default to true (pass-through)
    true
}

/// Resolve a slot that is either a variable reference or a literal.
fn resolve_slot<'a>(slot: &'a str, binding: &'a Binding) -> Option<String> {
    if slot.starts_with('?') {
        binding.get(slot).cloned()
    } else {
        Some(strip_quotes(slot).to_string())
    }
}

/// Strip surrounding double-quotes from a string, if present.
fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Extract the argument inside the outermost parentheses.
fn extract_paren_arg(expr: &str) -> &str {
    if let Some(open) = expr.find('(') {
        let rest = &expr[open + 1..];
        if let Some(close) = rest.rfind(')') {
            return &rest[..close];
        }
        return rest;
    }
    expr
}

/// Return `true` if a value looks like an IRI (starts with `http`, `urn`,
/// or is enclosed in `<…>`).
fn is_iri(value: &str) -> bool {
    let v = value.trim();
    (v.starts_with('<') && v.ends_with('>'))
        || v.starts_with("http://")
        || v.starts_with("https://")
        || v.starts_with("urn:")
        || v.starts_with("ftp://")
}

/// Return `true` if a value looks like a literal (starts/ends with `"`).
fn is_literal(value: &str) -> bool {
    let v = value.trim();
    v.starts_with('"') || (!is_iri(v) && !v.starts_with('_'))
}

// ────────────────────────────────────────────────────────────────────────────
// AskEvaluator
// ────────────────────────────────────────────────────────────────────────────

/// Evaluator for SPARQL ASK-style graph patterns.
#[derive(Debug, Default)]
pub struct AskEvaluator;

impl AskEvaluator {
    /// Create a new evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Return `true` if there is at least one binding that satisfies `pattern`.
    pub fn evaluate(store: &TripleStore, pattern: &AskPattern) -> bool {
        !Self::evaluate_with_bindings(store, pattern).is_empty()
    }

    /// Return all bindings that satisfy `pattern` starting from an empty binding.
    pub fn evaluate_with_bindings(store: &TripleStore, pattern: &AskPattern) -> Vec<Binding> {
        let initial = vec![HashMap::new()];
        Self::eval_pattern(store, pattern, initial)
    }

    /// Return the number of satisfying solutions.
    pub fn count_solutions(store: &TripleStore, pattern: &AskPattern) -> usize {
        Self::evaluate_with_bindings(store, pattern).len()
    }

    // ── recursive core ────────────────────────────────────────────────────────

    fn eval_pattern(
        store: &TripleStore,
        pattern: &AskPattern,
        input: Vec<Binding>,
    ) -> Vec<Binding> {
        match pattern {
            AskPattern::TriplePattern { s, p, o } => {
                Self::eval_triple_pattern(store, s, p, o, input)
            }
            AskPattern::Filter(expr) => input
                .into_iter()
                .filter(|b| evaluate_filter(expr, b))
                .collect(),
            AskPattern::Optional(inner) => {
                // LEFT JOIN: for each input binding, try to extend it via the
                // inner pattern; if it produces no solutions keep the original.
                let mut out = Vec::new();
                for b in &input {
                    let extended = Self::eval_pattern(store, inner, vec![b.clone()]);
                    if extended.is_empty() {
                        out.push(b.clone());
                    } else {
                        out.extend(extended);
                    }
                }
                out
            }
            AskPattern::And(patterns) => {
                // Fold left: each pattern refines the previous solutions.
                patterns
                    .iter()
                    .fold(input, |acc, p| Self::eval_pattern(store, p, acc))
            }
            AskPattern::Union(branches) => {
                // Union: collect solutions from every branch.
                let mut out = Vec::new();
                for branch in branches {
                    out.extend(Self::eval_pattern(store, branch, input.clone()));
                }
                out
            }
        }
    }

    /// Extend each input binding with the solutions from a basic triple pattern.
    fn eval_triple_pattern(
        store: &TripleStore,
        s: &str,
        p: &str,
        o: &str,
        input: Vec<Binding>,
    ) -> Vec<Binding> {
        let mut out = Vec::new();

        for binding in &input {
            // Substitute any already-bound variables.
            let s_resolved = Self::resolve_in_binding(s, binding);
            let p_resolved = Self::resolve_in_binding(p, binding);
            let o_resolved = Self::resolve_in_binding(o, binding);

            let matches = store.match_pattern(
                s_resolved.as_deref(),
                p_resolved.as_deref(),
                o_resolved.as_deref(),
            );

            for mut m in matches {
                // Merge with the current binding; skip on conflict.
                let mut merged = binding.clone();
                let mut conflict = false;
                for (k, v) in &m {
                    if let Some(existing) = merged.get(k) {
                        if existing != v {
                            conflict = true;
                            break;
                        }
                    }
                }
                if !conflict {
                    merged.extend(m.drain());
                    out.push(merged);
                }
            }
        }

        out
    }

    /// If `slot` is a variable already present in `binding`, return the
    /// concrete value; otherwise return `None` so the store sees it as a
    /// variable / wildcard.
    fn resolve_in_binding(slot: &str, binding: &Binding) -> Option<String> {
        if slot.starts_with('?') {
            // If already bound, return the concrete value; otherwise let the
            // triple store capture it.
            binding.get(slot).cloned().or(Some(slot.to_string()))
        } else {
            Some(slot.to_string())
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn basic_store() -> TripleStore {
        let mut store = TripleStore::new();
        store.insert(
            "http://example.org/alice",
            "http://example.org/name",
            "Alice",
        );
        store.insert("http://example.org/bob", "http://example.org/name", "Bob");
        store.insert(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        );
        store.insert("http://example.org/bob", "http://example.org/age", "30");
        store
    }

    // ── TripleStore ───────────────────────────────────────────────────────────

    #[test]
    fn test_triple_store_new_is_empty() {
        let store = TripleStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_triple_store_insert_increases_len() {
        let mut store = TripleStore::new();
        store.insert("s", "p", "o");
        assert_eq!(store.len(), 1);
        store.insert("s2", "p2", "o2");
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_match_pattern_exact() {
        let store = basic_store();
        let results = store.match_pattern(
            Some("http://example.org/alice"),
            Some("http://example.org/name"),
            Some("Alice"),
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_match_pattern_no_match() {
        let store = basic_store();
        let results = store.match_pattern(Some("unknown"), None, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_match_pattern_wildcard_subject() {
        let store = basic_store();
        // All triples with predicate = name
        let results = store.match_pattern(None, Some("http://example.org/name"), None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_match_pattern_variable_subject() {
        let store = basic_store();
        let results =
            store.match_pattern(Some("?s"), Some("http://example.org/name"), Some("Alice"));
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].get("?s").map(String::as_str),
            Some("http://example.org/alice")
        );
    }

    #[test]
    fn test_match_pattern_all_variables() {
        let store = basic_store();
        let results = store.match_pattern(Some("?s"), Some("?p"), Some("?o"));
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_match_pattern_two_variables() {
        let store = basic_store();
        let results = store.match_pattern(None, Some("?p"), Some("Alice"));
        assert_eq!(results.len(), 1);
        assert!(results[0].contains_key("?p"));
    }

    #[test]
    fn test_match_pattern_empty_store() {
        let store = TripleStore::new();
        let results = store.match_pattern(None, None, None);
        assert!(results.is_empty());
    }

    // ── AskEvaluator – TriplePattern ─────────────────────────────────────────

    #[test]
    fn test_ask_triple_pattern_true() {
        let store = basic_store();
        let pattern = AskPattern::TriplePattern {
            s: "http://example.org/alice".to_string(),
            p: "http://example.org/name".to_string(),
            o: "Alice".to_string(),
        };
        assert!(AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_ask_triple_pattern_false() {
        let store = basic_store();
        let pattern = AskPattern::TriplePattern {
            s: "http://example.org/alice".to_string(),
            p: "http://example.org/name".to_string(),
            o: "Zara".to_string(),
        };
        assert!(!AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_ask_triple_pattern_variable() {
        let store = basic_store();
        let pattern = AskPattern::TriplePattern {
            s: "?s".to_string(),
            p: "http://example.org/name".to_string(),
            o: "?name".to_string(),
        };
        assert!(AskEvaluator::evaluate(&store, &pattern));
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_ask_count_solutions() {
        let store = basic_store();
        let pattern = AskPattern::TriplePattern {
            s: "?s".to_string(),
            p: "?p".to_string(),
            o: "?o".to_string(),
        };
        assert_eq!(AskEvaluator::count_solutions(&store, &pattern), 4);
    }

    #[test]
    fn test_ask_empty_store_returns_false() {
        let store = TripleStore::new();
        let pattern = AskPattern::TriplePattern {
            s: "?s".to_string(),
            p: "?p".to_string(),
            o: "?o".to_string(),
        };
        assert!(!AskEvaluator::evaluate(&store, &pattern));
    }

    // ── AskEvaluator – Filter ─────────────────────────────────────────────────

    #[test]
    fn test_filter_equality_pass() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?name".to_string(),
            },
            AskPattern::Filter("?name = \"Alice\"".to_string()),
        ]);
        assert!(AskEvaluator::evaluate(&store, &pattern));
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 1);
    }

    #[test]
    fn test_filter_equality_fail() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?name".to_string(),
            },
            AskPattern::Filter("?name = \"Nobody\"".to_string()),
        ]);
        assert!(!AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_filter_inequality() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?name".to_string(),
            },
            AskPattern::Filter("?name != \"Alice\"".to_string()),
        ]);
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].get("?name").map(String::as_str), Some("Bob"));
    }

    #[test]
    fn test_filter_isiri_true() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/knows".to_string(),
                o: "?target".to_string(),
            },
            AskPattern::Filter("ISIRI(?target)".to_string()),
        ]);
        assert!(AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_filter_isiri_false() {
        let store = basic_store();
        // "Alice" is not an IRI
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?n".to_string(),
            },
            AskPattern::Filter("ISIRI(?n)".to_string()),
        ]);
        assert!(!AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_filter_isliteral_true() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?n".to_string(),
            },
            AskPattern::Filter("ISLITERAL(?n)".to_string()),
        ]);
        assert!(AskEvaluator::evaluate(&store, &pattern));
    }

    // ── AskEvaluator – AND ────────────────────────────────────────────────────

    #[test]
    fn test_and_two_patterns_match() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "Alice".to_string(),
            },
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/knows".to_string(),
                o: "?target".to_string(),
            },
        ]);
        assert!(AskEvaluator::evaluate(&store, &pattern));
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].get("?s").map(String::as_str),
            Some("http://example.org/alice")
        );
    }

    #[test]
    fn test_and_no_match() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "Alice".to_string(),
            },
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/age".to_string(),
                o: "?age".to_string(),
            },
        ]);
        // Alice has no age triple
        assert!(!AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_and_empty_patterns() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![]);
        // Empty conjunction – the empty binding is always valid
        assert!(AskEvaluator::evaluate(&store, &pattern));
    }

    // ── AskEvaluator – Union ──────────────────────────────────────────────────

    #[test]
    fn test_union_first_branch_matches() {
        let store = basic_store();
        let pattern = AskPattern::Union(vec![
            AskPattern::TriplePattern {
                s: "http://example.org/alice".to_string(),
                p: "http://example.org/name".to_string(),
                o: "Alice".to_string(),
            },
            AskPattern::TriplePattern {
                s: "http://example.org/alice".to_string(),
                p: "http://example.org/name".to_string(),
                o: "Nope".to_string(),
            },
        ]);
        assert!(AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_union_second_branch_matches() {
        let store = basic_store();
        let pattern = AskPattern::Union(vec![
            AskPattern::TriplePattern {
                s: "unknown".to_string(),
                p: "x".to_string(),
                o: "y".to_string(),
            },
            AskPattern::TriplePattern {
                s: "http://example.org/alice".to_string(),
                p: "http://example.org/name".to_string(),
                o: "Alice".to_string(),
            },
        ]);
        assert!(AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_union_no_branch_matches() {
        let store = basic_store();
        let pattern = AskPattern::Union(vec![
            AskPattern::TriplePattern {
                s: "x".to_string(),
                p: "y".to_string(),
                o: "z".to_string(),
            },
            AskPattern::TriplePattern {
                s: "a".to_string(),
                p: "b".to_string(),
                o: "c".to_string(),
            },
        ]);
        assert!(!AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_union_aggregates_bindings() {
        let store = basic_store();
        let pattern = AskPattern::Union(vec![
            AskPattern::TriplePattern {
                s: "http://example.org/alice".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?n".to_string(),
            },
            AskPattern::TriplePattern {
                s: "http://example.org/bob".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?n".to_string(),
            },
        ]);
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 2);
    }

    // ── AskEvaluator – Optional ───────────────────────────────────────────────

    #[test]
    fn test_optional_inner_matches() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?name".to_string(),
            },
            AskPattern::Optional(Box::new(AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/age".to_string(),
                o: "?age".to_string(),
            })),
        ]);
        // Both alice (no age) and bob (age=30) match; optional extends bob
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_optional_inner_does_not_match_still_returns_outer() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "http://example.org/alice".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?name".to_string(),
            },
            AskPattern::Optional(Box::new(AskPattern::TriplePattern {
                s: "http://example.org/alice".to_string(),
                p: "http://example.org/age".to_string(),
                o: "?age".to_string(),
            })),
        ]);
        // Alice has no age triple but should still appear
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 1);
        assert!(!bindings[0].contains_key("?age"));
    }

    #[test]
    fn test_optional_empty_store() {
        let store = TripleStore::new();
        let pattern = AskPattern::Optional(Box::new(AskPattern::TriplePattern {
            s: "?s".to_string(),
            p: "?p".to_string(),
            o: "?o".to_string(),
        }));
        // The empty input binding is kept
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 1);
    }

    // ── evaluate_filter unit tests ────────────────────────────────────────────

    #[test]
    fn test_filter_equality_no_quotes() {
        let mut b = Binding::new();
        b.insert("?x".to_string(), "hello".to_string());
        assert!(evaluate_filter("?x = hello", &b));
        assert!(!evaluate_filter("?x = world", &b));
    }

    #[test]
    fn test_filter_inequality_direct() {
        let mut b = Binding::new();
        b.insert("?x".to_string(), "hello".to_string());
        assert!(evaluate_filter("?x != world", &b));
        assert!(!evaluate_filter("?x != hello", &b));
    }

    #[test]
    fn test_filter_isiri_http() {
        let mut b = Binding::new();
        b.insert("?x".to_string(), "http://example.org/foo".to_string());
        assert!(evaluate_filter("ISIRI(?x)", &b));
    }

    #[test]
    fn test_filter_isiri_literal_value() {
        let mut b = Binding::new();
        b.insert("?x".to_string(), "just a string".to_string());
        assert!(!evaluate_filter("ISIRI(?x)", &b));
    }

    #[test]
    fn test_filter_isliteral_plain() {
        let mut b = Binding::new();
        b.insert("?x".to_string(), "hello world".to_string());
        assert!(evaluate_filter("ISLITERAL(?x)", &b));
    }

    #[test]
    fn test_filter_isliteral_iri_returns_false() {
        let mut b = Binding::new();
        b.insert("?x".to_string(), "http://example.org/foo".to_string());
        assert!(!evaluate_filter("ISLITERAL(?x)", &b));
    }

    #[test]
    fn test_filter_unbound_variable_equality_returns_false() {
        let b = Binding::new();
        // ?x is not bound; equality should return false
        assert!(!evaluate_filter("?x = \"something\"", &b));
    }

    #[test]
    fn test_filter_unbound_variable_inequality_returns_true() {
        let b = Binding::new();
        // ?x is not bound; inequality should return true (vacuous)
        assert!(evaluate_filter("?x != \"something\"", &b));
    }

    // ── complex nested patterns ───────────────────────────────────────────────

    #[test]
    fn test_and_union_combined() {
        let store = basic_store();
        // ?s has a name AND (knows someone OR has an age)
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?name".to_string(),
            },
            AskPattern::Union(vec![
                AskPattern::TriplePattern {
                    s: "?s".to_string(),
                    p: "http://example.org/knows".to_string(),
                    o: "?target".to_string(),
                },
                AskPattern::TriplePattern {
                    s: "?s".to_string(),
                    p: "http://example.org/age".to_string(),
                    o: "?age".to_string(),
                },
            ]),
        ]);
        assert!(AskEvaluator::evaluate(&store, &pattern));
        // alice (knows) + bob (age) → 2 results
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_count_with_filter() {
        let store = basic_store();
        let pattern = AskPattern::And(vec![
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/name".to_string(),
                o: "?name".to_string(),
            },
            AskPattern::Filter("?name = \"Bob\"".to_string()),
        ]);
        assert_eq!(AskEvaluator::count_solutions(&store, &pattern), 1);
    }

    #[test]
    fn test_triple_pattern_subject_object_same_variable() {
        // ?x <p> ?x  — subject and object must be the same value
        let mut store = TripleStore::new();
        store.insert("a", "p", "a");
        store.insert("a", "p", "b");
        let pattern = AskPattern::TriplePattern {
            s: "?x".to_string(),
            p: "p".to_string(),
            o: "?x".to_string(),
        };
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].get("?x").map(String::as_str), Some("a"));
    }

    #[test]
    fn test_nested_and_in_union() {
        let store = basic_store();
        let pattern = AskPattern::Union(vec![
            AskPattern::And(vec![
                AskPattern::TriplePattern {
                    s: "?s".to_string(),
                    p: "http://example.org/name".to_string(),
                    o: "Alice".to_string(),
                },
                AskPattern::TriplePattern {
                    s: "?s".to_string(),
                    p: "http://example.org/knows".to_string(),
                    o: "?t".to_string(),
                },
            ]),
            AskPattern::TriplePattern {
                s: "?s".to_string(),
                p: "http://example.org/age".to_string(),
                o: "30".to_string(),
            },
        ]);
        let bindings = AskEvaluator::evaluate_with_bindings(&store, &pattern);
        // alice+knows + bob+age  → 2
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_ask_evaluate_returns_true_on_match() {
        let store = basic_store();
        let pattern = AskPattern::TriplePattern {
            s: "?s".to_string(),
            p: "http://example.org/age".to_string(),
            o: "30".to_string(),
        };
        assert!(AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_ask_evaluate_returns_false_on_empty() {
        let store = TripleStore::new();
        let pattern = AskPattern::TriplePattern {
            s: "?s".to_string(),
            p: "?p".to_string(),
            o: "?o".to_string(),
        };
        assert!(!AskEvaluator::evaluate(&store, &pattern));
    }

    #[test]
    fn test_multiple_inserts_same_triple() {
        let mut store = TripleStore::new();
        store.insert("s", "p", "o");
        store.insert("s", "p", "o");
        assert_eq!(store.len(), 2);
        let results = store.match_pattern(Some("s"), Some("p"), Some("o"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_case_insensitive_isiri() {
        let mut b = Binding::new();
        b.insert("?x".to_string(), "http://example.org/r".to_string());
        // lowercase version
        assert!(evaluate_filter("isiri(?x)", &b));
    }

    #[test]
    fn test_filter_case_insensitive_isliteral() {
        let mut b = Binding::new();
        b.insert("?x".to_string(), "plain literal".to_string());
        assert!(evaluate_filter("isliteral(?x)", &b));
    }
}
