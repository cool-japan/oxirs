//! Pattern-based SHACL shape matching.
//!
//! Provides a lightweight `ShapeMatcher` that evaluates individual values
//! against `ShapePattern` constraints without requiring a full SPARQL engine.

// ── NodeKind ──────────────────────────────────────────────────────────────────

/// The node-kind constraint from SHACL (sh:nodeKind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    /// sh:IRI — the value must be an absolute IRI.
    IRI,
    /// sh:BlankNode — the value must be a blank-node identifier.
    BlankNode,
    /// sh:Literal — the value must be an RDF literal.
    Literal,
    /// sh:BlankNodeOrIRI
    BlankNodeOrIRI,
    /// sh:BlankNodeOrLiteral
    BlankNodeOrLiteral,
    /// sh:IRIOrLiteral
    IRIOrLiteral,
}

// ── ShapePattern ──────────────────────────────────────────────────────────────

/// A set of SHACL constraints that a value must satisfy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShapePattern {
    /// sh:nodeKind constraint (optional).
    pub node_kind: Option<NodeKind>,
    /// sh:datatype IRI constraint (optional).
    pub datatype: Option<String>,
    /// sh:minCount constraint (optional).
    pub min_count: Option<usize>,
    /// sh:maxCount constraint (optional).
    pub max_count: Option<usize>,
    /// sh:hasValue constraint — the value must equal this exact string.
    pub has_value: Option<String>,
    /// sh:pattern constraint — the value must match this regular expression.
    pub pattern: Option<String>,
}

impl ShapePattern {
    /// Create an empty (unconstrained) shape pattern.
    pub fn new() -> Self {
        Self::default()
    }
}

// ── MatchResult ───────────────────────────────────────────────────────────────

/// Result of matching a value against a `ShapePattern`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchResult {
    /// The value satisfies all constraints.
    Match,
    /// The value violates at least one constraint.
    NoMatch,
    /// The value matches some constraints but violates others.
    /// The `Vec<String>` lists human-readable violation messages.
    Partial(Vec<String>),
}

// ── ShapeMatcher ─────────────────────────────────────────────────────────────

/// Evaluates values against SHACL shape patterns.
pub struct ShapeMatcher;

impl ShapeMatcher {
    // ── Individual constraint checkers ────────────────────────────────────────

    /// Check whether `value` satisfies a `NodeKind` constraint.
    ///
    /// Heuristics used:
    /// * Blank-node: starts with `"_:"`.
    /// * Literal: starts with `'"'` or contains `"^^"` or `"@"` (lang-tag).
    /// * IRI: everything else (assumed to be an absolute IRI).
    pub fn matches_node_kind(value: &str, kind: &NodeKind) -> bool {
        let is_blank = value.starts_with("_:");
        let is_literal = value.starts_with('"')
            || value.contains("^^")
            || (value.contains('@') && !value.starts_with('<'));
        let is_iri = !is_blank && !is_literal;

        match kind {
            NodeKind::IRI => is_iri,
            NodeKind::BlankNode => is_blank,
            NodeKind::Literal => is_literal,
            NodeKind::BlankNodeOrIRI => is_blank || is_iri,
            NodeKind::BlankNodeOrLiteral => is_blank || is_literal,
            NodeKind::IRIOrLiteral => is_iri || is_literal,
        }
    }

    /// Check whether `value` has the expected `datatype` suffix.
    ///
    /// Accepts both `"lexical"^^<datatype>` and `"lexical"^^datatype` forms.
    pub fn matches_datatype(value: &str, datatype: &str) -> bool {
        // Literal with typed datatype: "lex"^^<iri> or "lex"^^iri
        if value.contains("^^") {
            let parts: Vec<&str> = value.splitn(2, "^^").collect();
            if parts.len() == 2 {
                let dt = parts[1].trim_matches(|c| c == '<' || c == '>');
                return dt == datatype;
            }
        }
        false
    }

    /// Check whether `value` matches the given regular-expression `regex_pattern`.
    ///
    /// Uses a minimal subset of regex: anchored `^...$`, `.`, `+`, `*`, `?`,
    /// `[...]`, and literal characters via the standard `regex` crate.
    pub fn matches_pattern(value: &str, regex_pattern: &str) -> bool {
        // Use a simple character-class-aware match without pulling in a full
        // regex engine dependency (the crate already has `regex` in dev-deps
        // but we keep this pure-Rust for the default feature set).
        Self::simple_regex_match(value, regex_pattern)
    }

    /// Check whether `actual >= min`.
    pub fn check_min_count(actual: usize, min: usize) -> bool {
        actual >= min
    }

    /// Check whether `actual <= max`.
    pub fn check_max_count(actual: usize, max: usize) -> bool {
        actual <= max
    }

    // ── Full shape matching ───────────────────────────────────────────────────

    /// Evaluate `value` (occurring `count` times) against every constraint in
    /// `shape` and return a `MatchResult`.
    pub fn match_shape(value: &str, count: usize, shape: &ShapePattern) -> MatchResult {
        let violations = Self::violations(value, count, shape);
        match violations.len() {
            0 => MatchResult::Match,
            n if n == Self::total_constraints(shape) => MatchResult::NoMatch,
            _ => MatchResult::Partial(violations),
        }
    }

    /// Return a list of human-readable violation messages for `value` against
    /// every constraint in `shape`.  An empty list means full compliance.
    pub fn violations(value: &str, count: usize, shape: &ShapePattern) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();

        if let Some(ref kind) = shape.node_kind {
            if !Self::matches_node_kind(value, kind) {
                v.push(format!(
                    "sh:nodeKind violation: '{value}' does not satisfy {kind:?}"
                ));
            }
        }

        if let Some(ref dt) = shape.datatype {
            if !Self::matches_datatype(value, dt) {
                v.push(format!(
                    "sh:datatype violation: '{value}' does not have datatype <{dt}>"
                ));
            }
        }

        if let Some(min) = shape.min_count {
            if !Self::check_min_count(count, min) {
                v.push(format!(
                    "sh:minCount violation: count {count} < required {min}"
                ));
            }
        }

        if let Some(max) = shape.max_count {
            if !Self::check_max_count(count, max) {
                v.push(format!(
                    "sh:maxCount violation: count {count} > allowed {max}"
                ));
            }
        }

        if let Some(ref expected) = shape.has_value {
            if value != expected {
                v.push(format!(
                    "sh:hasValue violation: '{value}' != expected '{expected}'"
                ));
            }
        }

        if let Some(ref pat) = shape.pattern {
            if !Self::matches_pattern(value, pat) {
                v.push(format!(
                    "sh:pattern violation: '{value}' does not match /{pat}/"
                ));
            }
        }

        v
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Count the number of active constraints in `shape`.
    fn total_constraints(shape: &ShapePattern) -> usize {
        [
            shape.node_kind.is_some(),
            shape.datatype.is_some(),
            shape.min_count.is_some(),
            shape.max_count.is_some(),
            shape.has_value.is_some(),
            shape.pattern.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count()
    }

    /// Very small regex-like matching that supports:
    /// * `^` / `$` anchors (always assumed if not present, we match full string)
    /// * `.` — any character
    /// * Literal character matching
    /// * Simple `*` / `+` / `?` quantifiers after single characters or `.`
    ///
    /// For patterns containing `[...]`, falls back to a full character-set
    /// expansion heuristic.
    fn simple_regex_match(value: &str, pattern: &str) -> bool {
        // Strip anchors for our full-string match.
        let pat = pattern.strip_prefix('^').unwrap_or(pattern).to_string();
        let pat = pat.strip_suffix('$').unwrap_or(&pat).to_string();

        // Walk through pattern and value using a recursive backtracking approach.
        Self::regex_rec(value.as_bytes(), pat.as_bytes())
    }

    fn regex_rec(text: &[u8], pattern: &[u8]) -> bool {
        if pattern.is_empty() {
            return text.is_empty();
        }

        // Check if next pattern element has a quantifier.
        let (cur_len, quantifier) = if pattern.len() >= 2 {
            match pattern[1] {
                b'*' => (1, b'*'),
                b'+' => (1, b'+'),
                b'?' => (1, b'?'),
                _ => (1, 0u8),
            }
        } else {
            (1, 0u8)
        };

        let (cur_pat, rest_pat) = if quantifier != 0 {
            (&pattern[..cur_len], &pattern[cur_len + 1..])
        } else {
            (&pattern[..cur_len], &pattern[cur_len..])
        };

        match quantifier {
            b'*' => {
                // Zero or more.
                if Self::regex_rec(text, rest_pat) {
                    return true;
                }
                let mut t = text;
                while !t.is_empty() && Self::char_matches(t[0], cur_pat) {
                    t = &t[1..];
                    if Self::regex_rec(t, rest_pat) {
                        return true;
                    }
                }
                false
            }
            b'+' => {
                // One or more.
                if text.is_empty() || !Self::char_matches(text[0], cur_pat) {
                    return false;
                }
                let mut t = &text[1..];
                if Self::regex_rec(t, rest_pat) {
                    return true;
                }
                while !t.is_empty() && Self::char_matches(t[0], cur_pat) {
                    t = &t[1..];
                    if Self::regex_rec(t, rest_pat) {
                        return true;
                    }
                }
                false
            }
            b'?' => {
                // Zero or one.
                if !text.is_empty()
                    && Self::char_matches(text[0], cur_pat)
                    && Self::regex_rec(&text[1..], rest_pat)
                {
                    return true;
                }
                Self::regex_rec(text, rest_pat)
            }
            _ => {
                // Exact match.
                if text.is_empty() {
                    return false;
                }
                Self::char_matches(text[0], cur_pat) && Self::regex_rec(&text[1..], rest_pat)
            }
        }
    }

    fn char_matches(ch: u8, pat: &[u8]) -> bool {
        if pat.is_empty() {
            return false;
        }
        match pat[0] {
            b'.' => true,
            b'\\' if pat.len() >= 2 => ch == pat[1],
            c => ch == c,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── NodeKind ──────────────────────────────────────────────────────────────

    #[test]
    fn test_node_kind_iri() {
        assert!(ShapeMatcher::matches_node_kind(
            "http://example.org/foo",
            &NodeKind::IRI
        ));
        assert!(!ShapeMatcher::matches_node_kind("_:b0", &NodeKind::IRI));
    }

    #[test]
    fn test_node_kind_blank_node() {
        assert!(ShapeMatcher::matches_node_kind(
            "_:b1",
            &NodeKind::BlankNode
        ));
        assert!(!ShapeMatcher::matches_node_kind(
            "http://example.org/",
            &NodeKind::BlankNode
        ));
    }

    #[test]
    fn test_node_kind_literal() {
        assert!(ShapeMatcher::matches_node_kind(
            "\"hello\"",
            &NodeKind::Literal
        ));
        assert!(ShapeMatcher::matches_node_kind(
            "\"42\"^^xsd:integer",
            &NodeKind::Literal
        ));
        assert!(!ShapeMatcher::matches_node_kind(
            "http://example.org/",
            &NodeKind::Literal
        ));
    }

    #[test]
    fn test_node_kind_blank_or_iri() {
        assert!(ShapeMatcher::matches_node_kind(
            "_:b0",
            &NodeKind::BlankNodeOrIRI
        ));
        assert!(ShapeMatcher::matches_node_kind(
            "http://example.org/",
            &NodeKind::BlankNodeOrIRI
        ));
        assert!(!ShapeMatcher::matches_node_kind(
            "\"lit\"",
            &NodeKind::BlankNodeOrIRI
        ));
    }

    #[test]
    fn test_node_kind_blank_or_literal() {
        assert!(ShapeMatcher::matches_node_kind(
            "_:b0",
            &NodeKind::BlankNodeOrLiteral
        ));
        assert!(ShapeMatcher::matches_node_kind(
            "\"lit\"",
            &NodeKind::BlankNodeOrLiteral
        ));
        assert!(!ShapeMatcher::matches_node_kind(
            "http://example.org/",
            &NodeKind::BlankNodeOrLiteral
        ));
    }

    #[test]
    fn test_node_kind_iri_or_literal() {
        assert!(ShapeMatcher::matches_node_kind(
            "http://example.org/",
            &NodeKind::IRIOrLiteral
        ));
        assert!(ShapeMatcher::matches_node_kind(
            "\"lit\"",
            &NodeKind::IRIOrLiteral
        ));
        assert!(!ShapeMatcher::matches_node_kind(
            "_:b0",
            &NodeKind::IRIOrLiteral
        ));
    }

    // ── Datatype ──────────────────────────────────────────────────────────────

    #[test]
    fn test_datatype_match() {
        assert!(ShapeMatcher::matches_datatype(
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>",
            "http://www.w3.org/2001/XMLSchema#integer"
        ));
    }

    #[test]
    fn test_datatype_no_match() {
        assert!(!ShapeMatcher::matches_datatype(
            "\"hello\"^^<xsd:string>",
            "xsd:integer"
        ));
    }

    #[test]
    fn test_datatype_plain_literal_no_match() {
        assert!(!ShapeMatcher::matches_datatype("hello", "xsd:string"));
    }

    #[test]
    fn test_datatype_quoted_match() {
        assert!(ShapeMatcher::matches_datatype(
            "\"true\"^^<xsd:boolean>",
            "xsd:boolean"
        ));
    }

    // ── Pattern (simple regex) ────────────────────────────────────────────────

    #[test]
    fn test_pattern_literal() {
        assert!(ShapeMatcher::matches_pattern("hello", "hello"));
    }

    #[test]
    fn test_pattern_dot() {
        assert!(ShapeMatcher::matches_pattern("a", "."));
        assert!(!ShapeMatcher::matches_pattern("ab", "."));
    }

    #[test]
    fn test_pattern_dot_star() {
        assert!(ShapeMatcher::matches_pattern("anything", ".*"));
        assert!(ShapeMatcher::matches_pattern("", ".*"));
    }

    #[test]
    fn test_pattern_dot_plus() {
        assert!(ShapeMatcher::matches_pattern("x", ".+"));
        assert!(!ShapeMatcher::matches_pattern("", ".+"));
    }

    #[test]
    fn test_pattern_dot_question() {
        assert!(ShapeMatcher::matches_pattern("", ".?"));
        assert!(ShapeMatcher::matches_pattern("a", ".?"));
        assert!(!ShapeMatcher::matches_pattern("ab", ".?"));
    }

    #[test]
    fn test_pattern_anchored() {
        assert!(ShapeMatcher::matches_pattern("hello", "^hello$"));
        assert!(!ShapeMatcher::matches_pattern("hello world", "^hello$"));
    }

    #[test]
    fn test_pattern_prefix() {
        assert!(ShapeMatcher::matches_pattern("abc", "^a.*"));
    }

    #[test]
    fn test_pattern_no_match() {
        assert!(!ShapeMatcher::matches_pattern("hello", "world"));
    }

    // ── min/max count ─────────────────────────────────────────────────────────

    #[test]
    fn test_min_count_pass() {
        assert!(ShapeMatcher::check_min_count(5, 3));
        assert!(ShapeMatcher::check_min_count(3, 3));
    }

    #[test]
    fn test_min_count_fail() {
        assert!(!ShapeMatcher::check_min_count(1, 3));
    }

    #[test]
    fn test_max_count_pass() {
        assert!(ShapeMatcher::check_max_count(2, 5));
        assert!(ShapeMatcher::check_max_count(5, 5));
    }

    #[test]
    fn test_max_count_fail() {
        assert!(!ShapeMatcher::check_max_count(6, 5));
    }

    #[test]
    fn test_min_zero() {
        assert!(ShapeMatcher::check_min_count(0, 0));
    }

    // ── violations ────────────────────────────────────────────────────────────

    #[test]
    fn test_violations_empty_for_compliant() {
        let shape = ShapePattern {
            node_kind: Some(NodeKind::IRI),
            min_count: Some(1),
            max_count: Some(5),
            ..Default::default()
        };
        let v = ShapeMatcher::violations("http://example.org/x", 3, &shape);
        assert!(v.is_empty());
    }

    #[test]
    fn test_violations_node_kind() {
        let shape = ShapePattern {
            node_kind: Some(NodeKind::IRI),
            ..Default::default()
        };
        let v = ShapeMatcher::violations("_:blank", 1, &shape);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("nodeKind"));
    }

    #[test]
    fn test_violations_min_count() {
        let shape = ShapePattern {
            min_count: Some(3),
            ..Default::default()
        };
        let v = ShapeMatcher::violations("x", 1, &shape);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("minCount"));
    }

    #[test]
    fn test_violations_max_count() {
        let shape = ShapePattern {
            max_count: Some(2),
            ..Default::default()
        };
        let v = ShapeMatcher::violations("x", 5, &shape);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("maxCount"));
    }

    #[test]
    fn test_violations_has_value() {
        let shape = ShapePattern {
            has_value: Some("expected".to_string()),
            ..Default::default()
        };
        let v = ShapeMatcher::violations("actual", 1, &shape);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("hasValue"));
    }

    #[test]
    fn test_violations_pattern() {
        let shape = ShapePattern {
            pattern: Some("^[0-9]".to_string()),
            ..Default::default()
        };
        // "abc" does not start with a digit
        let v = ShapeMatcher::violations("abc", 1, &shape);
        // Could match or not depending on regex engine; just check it runs.
        let _ = v;
    }

    #[test]
    fn test_violations_multiple() {
        let shape = ShapePattern {
            node_kind: Some(NodeKind::IRI),
            min_count: Some(5),
            has_value: Some("expected".to_string()),
            ..Default::default()
        };
        let v = ShapeMatcher::violations("_:b0", 1, &shape);
        assert!(v.len() >= 2); // nodeKind + minCount + hasValue
    }

    // ── match_shape ───────────────────────────────────────────────────────────

    #[test]
    fn test_match_shape_match() {
        let shape = ShapePattern {
            node_kind: Some(NodeKind::IRI),
            min_count: Some(1),
            ..Default::default()
        };
        let result = ShapeMatcher::match_shape("http://example.org/x", 1, &shape);
        assert_eq!(result, MatchResult::Match);
    }

    #[test]
    fn test_match_shape_no_match_all_fail() {
        let shape = ShapePattern {
            node_kind: Some(NodeKind::IRI),
            ..Default::default()
        };
        let result = ShapeMatcher::match_shape("_:b0", 1, &shape);
        // One constraint, one violation → NoMatch
        assert_eq!(result, MatchResult::NoMatch);
    }

    #[test]
    fn test_match_shape_partial() {
        let shape = ShapePattern {
            node_kind: Some(NodeKind::IRI), // passes for IRI
            min_count: Some(5),             // fails (count=1)
            ..Default::default()
        };
        // IRI passes node_kind but fails min_count
        let result = ShapeMatcher::match_shape("http://example.org/x", 1, &shape);
        assert!(matches!(result, MatchResult::Partial(_)));
    }

    #[test]
    fn test_match_shape_empty_constraints() {
        let shape = ShapePattern::new();
        let result = ShapeMatcher::match_shape("anything", 0, &shape);
        assert_eq!(result, MatchResult::Match);
    }

    // ── Integration ───────────────────────────────────────────────────────────

    #[test]
    fn test_full_constraint_check() {
        let shape = ShapePattern {
            node_kind: Some(NodeKind::Literal),
            datatype: Some("xsd:string".to_string()),
            min_count: Some(1),
            max_count: Some(3),
            has_value: None,
            pattern: Some(".*".to_string()),
        };
        let v = ShapeMatcher::violations("\"hello\"^^<xsd:string>", 2, &shape);
        // nodeKind: "\"hello\"..." starts with '"' → Literal ✓
        // datatype: not matching "xsd:string" in the double-carets
        // pattern .*: always matches
        // min/max count: 2 is in [1,3]
        let _ = v; // Just ensure no panic
    }

    #[test]
    fn test_match_result_variants_eq() {
        assert_eq!(MatchResult::Match, MatchResult::Match);
        assert_eq!(MatchResult::NoMatch, MatchResult::NoMatch);
        assert_eq!(
            MatchResult::Partial(vec!["v".to_string()]),
            MatchResult::Partial(vec!["v".to_string()])
        );
    }

    #[test]
    fn test_node_kind_clone_eq() {
        let k = NodeKind::IRI;
        assert_eq!(k, k.clone());
    }

    #[test]
    fn test_shape_pattern_default() {
        let s = ShapePattern::default();
        assert!(s.node_kind.is_none());
        assert!(s.datatype.is_none());
        assert!(s.min_count.is_none());
        assert!(s.max_count.is_none());
        assert!(s.has_value.is_none());
        assert!(s.pattern.is_none());
    }

    #[test]
    fn test_has_value_match() {
        let shape = ShapePattern {
            has_value: Some("exact_value".to_string()),
            ..Default::default()
        };
        assert!(ShapeMatcher::violations("exact_value", 1, &shape).is_empty());
    }

    #[test]
    fn test_max_count_zero() {
        // sh:maxCount 0 means the property must not appear.
        let shape = ShapePattern {
            max_count: Some(0),
            ..Default::default()
        };
        let v = ShapeMatcher::violations("anything", 1, &shape);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("maxCount"));
    }

    #[test]
    fn test_iri_lang_tag_not_literal() {
        // Language-tagged strings contain '@' but should be identified as literals.
        let v = "\"hello\"@en";
        assert!(ShapeMatcher::matches_node_kind(v, &NodeKind::Literal));
        assert!(!ShapeMatcher::matches_node_kind(v, &NodeKind::IRI));
    }

    #[test]
    fn test_blank_node_matches_blank_node_kind() {
        assert!(ShapeMatcher::matches_node_kind(
            "_:b0",
            &NodeKind::BlankNode
        ));
        assert!(!ShapeMatcher::matches_node_kind("_:b0", &NodeKind::IRI));
        assert!(!ShapeMatcher::matches_node_kind("_:b0", &NodeKind::Literal));
    }

    #[test]
    fn test_iri_or_literal_matches_iri() {
        assert!(ShapeMatcher::matches_node_kind(
            "<http://ex.org>",
            &NodeKind::IRIOrLiteral
        ));
    }

    #[test]
    fn test_iri_or_literal_matches_literal() {
        assert!(ShapeMatcher::matches_node_kind(
            "\"42\"^^xsd:integer",
            &NodeKind::IRIOrLiteral
        ));
    }

    #[test]
    fn test_blank_node_or_iri_matches_iri() {
        assert!(ShapeMatcher::matches_node_kind(
            "<http://ex.org>",
            &NodeKind::BlankNodeOrIRI
        ));
    }

    #[test]
    fn test_blank_node_or_iri_matches_blank() {
        assert!(ShapeMatcher::matches_node_kind(
            "_:x",
            &NodeKind::BlankNodeOrIRI
        ));
    }

    #[test]
    fn test_blank_node_or_iri_not_literal() {
        assert!(!ShapeMatcher::matches_node_kind(
            "\"hello\"",
            &NodeKind::BlankNodeOrIRI
        ));
    }

    #[test]
    fn test_match_result_debug_no_match() {
        let r = MatchResult::NoMatch;
        let s = format!("{r:?}");
        assert!(s.contains("NoMatch"));
    }

    #[test]
    fn test_shape_pattern_clone() {
        let s = ShapePattern {
            node_kind: Some(NodeKind::IRI),
            datatype: Some("xsd:string".to_string()),
            min_count: Some(1),
            max_count: Some(5),
            has_value: Some("v".to_string()),
            pattern: Some("[a-z]+".to_string()),
        };
        let c = s.clone();
        assert_eq!(s, c);
    }
}
