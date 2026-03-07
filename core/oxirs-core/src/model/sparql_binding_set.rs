//! SPARQL Binding Set operations for MINUS and EXISTS evaluation.
//!
//! Implements the core set-theoretic operations used in SPARQL query processing,
//! following the W3C SPARQL 1.1 specification for set algebra.

use std::collections::{HashMap, HashSet};

/// An RDF term that can appear in a SPARQL solution binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfTerm {
    /// An IRI (Internationalized Resource Identifier)
    Iri(String),
    /// A literal value with optional language tag or datatype
    Literal {
        value: String,
        datatype: String,
        lang: Option<String>,
    },
    /// A blank node with a local identifier
    BlankNode(String),
}

impl RdfTerm {
    /// Create an IRI term
    pub fn iri(iri: impl Into<String>) -> Self {
        RdfTerm::Iri(iri.into())
    }

    /// Create a plain string literal with xsd:string datatype
    pub fn string_literal(value: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            datatype: "http://www.w3.org/2001/XMLSchema#string".to_string(),
            lang: None,
        }
    }

    /// Create a language-tagged literal
    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            datatype: "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string(),
            lang: Some(lang.into()),
        }
    }

    /// Create a blank node
    pub fn blank(id: impl Into<String>) -> Self {
        RdfTerm::BlankNode(id.into())
    }
}

impl std::fmt::Display for RdfTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfTerm::Iri(iri) => write!(f, "<{}>", iri),
            RdfTerm::Literal {
                value,
                lang: Some(lang),
                ..
            } => write!(f, "\"{}\"@{}", value, lang),
            RdfTerm::Literal {
                value,
                datatype,
                lang: None,
            } => write!(f, "\"{}\"^^<{}>", value, datatype),
            RdfTerm::BlankNode(id) => write!(f, "_:{}", id),
        }
    }
}

/// A set of SPARQL solution mappings (bindings).
///
/// Each element is a mapping from variable names to RDF terms.
/// Implements the SPARQL algebra operations: MINUS, EXISTS, UNION, PROJECT, JOIN, DISTINCT.
#[derive(Debug, Clone, Default)]
pub struct BindingSet {
    bindings: Vec<HashMap<String, RdfTerm>>,
}

impl BindingSet {
    /// Create an empty binding set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a binding set from a vector of solution mappings.
    pub fn from_vec(bindings: Vec<HashMap<String, RdfTerm>>) -> Self {
        BindingSet { bindings }
    }

    /// Add a single solution mapping.
    pub fn add(&mut self, binding: HashMap<String, RdfTerm>) {
        self.bindings.push(binding);
    }

    /// Number of solution mappings in this set.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns true if this binding set contains no solutions.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// SPARQL MINUS operation.
    ///
    /// Returns a binding set containing only those rows from `self` that are NOT
    /// compatible with any row in `other` **that shares at least one variable**.
    ///
    /// W3C SPARQL 1.1 spec §18.5: A row μ₁ from Ω₁ is removed if there exists
    /// μ₂ in Ω₂ such that:
    /// - dom(μ₁) ∩ dom(μ₂) ≠ ∅  (they share at least one variable)
    /// - μ₁ and μ₂ are compatible on shared variables
    ///
    /// If no row in `other` shares any variable with a row in `self`, the row is KEPT.
    pub fn minus(&self, other: &BindingSet) -> BindingSet {
        let kept = self
            .bindings
            .iter()
            .filter(|row_self| {
                !other.bindings.iter().any(|row_other| {
                    let shared: HashSet<&String> = row_self
                        .keys()
                        .collect::<HashSet<_>>()
                        .intersection(&row_other.keys().collect::<HashSet<_>>())
                        .copied()
                        .collect();
                    !shared.is_empty() && Self::is_compatible(row_self, row_other)
                })
            })
            .cloned()
            .collect();
        BindingSet { bindings: kept }
    }

    /// EXISTS filter: keep rows from `self` that are compatible with at least
    /// one row in `pattern`.
    ///
    /// Two rows are compatible if all their shared variables agree in value.
    /// Rows with no shared variables are considered compatible.
    pub fn exists_filter(&self, pattern: &BindingSet) -> BindingSet {
        let kept = self
            .bindings
            .iter()
            .filter(|row_self| {
                pattern
                    .bindings
                    .iter()
                    .any(|row_pattern| Self::is_compatible(row_self, row_pattern))
            })
            .cloned()
            .collect();
        BindingSet { bindings: kept }
    }

    /// UNION: combine all rows from both binding sets.
    pub fn union(&self, other: &BindingSet) -> BindingSet {
        let mut result = self.bindings.clone();
        result.extend(other.bindings.iter().cloned());
        BindingSet { bindings: result }
    }

    /// PROJECT: keep only the named variables in each solution mapping.
    ///
    /// Variables not in `vars` are dropped. Solutions that become identical
    /// after projection are NOT automatically deduplicated (call `distinct` separately).
    pub fn project(&self, vars: &[&str]) -> BindingSet {
        let projected = self
            .bindings
            .iter()
            .map(|row| {
                vars.iter()
                    .filter_map(|v| row.get(*v).map(|term| (v.to_string(), term.clone())))
                    .collect::<HashMap<String, RdfTerm>>()
            })
            .collect();
        BindingSet {
            bindings: projected,
        }
    }

    /// Natural JOIN: for each pair of rows from `self` and `other` that are
    /// compatible (agree on all shared variables), produce the merged row.
    pub fn join(&self, other: &BindingSet) -> BindingSet {
        let mut result = Vec::new();
        for row_self in &self.bindings {
            for row_other in &other.bindings {
                if Self::is_compatible(row_self, row_other) {
                    result.push(Self::merge_rows(row_self, row_other));
                }
            }
        }
        BindingSet { bindings: result }
    }

    /// DISTINCT: remove duplicate solution mappings.
    pub fn distinct(&self) -> BindingSet {
        let mut seen: HashSet<Vec<(String, RdfTerm)>> = HashSet::new();
        let unique = self
            .bindings
            .iter()
            .filter(|row| {
                let mut sorted: Vec<(String, RdfTerm)> =
                    row.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                seen.insert(sorted)
            })
            .cloned()
            .collect();
        BindingSet { bindings: unique }
    }

    /// Iterate over all solution mappings in this set.
    pub fn iter(&self) -> impl Iterator<Item = &HashMap<String, RdfTerm>> {
        self.bindings.iter()
    }

    /// Two solution mappings are **compatible** if, for every variable they
    /// both bind, they agree on the value.
    ///
    /// Rows with no shared variables are always compatible.
    pub fn is_compatible(a: &HashMap<String, RdfTerm>, b: &HashMap<String, RdfTerm>) -> bool {
        a.iter()
            .all(|(var, term_a)| b.get(var).map_or(true, |term_b| term_a == term_b))
    }

    /// Merge two compatible rows into one by taking the union of their variables.
    fn merge_rows(
        a: &HashMap<String, RdfTerm>,
        b: &HashMap<String, RdfTerm>,
    ) -> HashMap<String, RdfTerm> {
        let mut merged = a.clone();
        for (k, v) in b {
            merged.entry(k.clone()).or_insert_with(|| v.clone());
        }
        merged
    }
}

impl IntoIterator for BindingSet {
    type Item = HashMap<String, RdfTerm>;
    type IntoIter = std::vec::IntoIter<HashMap<String, RdfTerm>>;

    fn into_iter(self) -> Self::IntoIter {
        self.bindings.into_iter()
    }
}

/// Helper to build a single solution mapping from pairs.
pub fn solution(pairs: &[(&str, RdfTerm)]) -> HashMap<String, RdfTerm> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn iri(s: &str) -> RdfTerm {
        RdfTerm::iri(s)
    }

    fn lit(s: &str) -> RdfTerm {
        RdfTerm::string_literal(s)
    }

    fn bnode(s: &str) -> RdfTerm {
        RdfTerm::blank(s)
    }

    fn row(pairs: &[(&str, RdfTerm)]) -> HashMap<String, RdfTerm> {
        solution(pairs)
    }

    fn single(var: &str, term: RdfTerm) -> HashMap<String, RdfTerm> {
        let mut m = HashMap::new();
        m.insert(var.to_string(), term);
        m
    }

    // ── BindingSet::new ───────────────────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let bs = BindingSet::new();
        assert!(bs.is_empty());
        assert_eq!(bs.len(), 0);
    }

    #[test]
    fn test_from_vec_empty() {
        let bs = BindingSet::from_vec(vec![]);
        assert!(bs.is_empty());
    }

    #[test]
    fn test_from_vec_non_empty() {
        let bs = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        assert_eq!(bs.len(), 1);
    }

    #[test]
    fn test_add() {
        let mut bs = BindingSet::new();
        bs.add(single("x", iri("http://a")));
        bs.add(single("x", iri("http://b")));
        assert_eq!(bs.len(), 2);
    }

    // ── is_compatible ────────────────────────────────────────────────────────

    #[test]
    fn test_compatible_no_shared_vars() {
        let a = single("x", iri("http://a"));
        let b = single("y", iri("http://b"));
        assert!(BindingSet::is_compatible(&a, &b));
    }

    #[test]
    fn test_compatible_same_shared_var_same_value() {
        let a = single("x", iri("http://a"));
        let b = single("x", iri("http://a"));
        assert!(BindingSet::is_compatible(&a, &b));
    }

    #[test]
    fn test_incompatible_shared_var_different_value() {
        let a = single("x", iri("http://a"));
        let b = single("x", iri("http://b"));
        assert!(!BindingSet::is_compatible(&a, &b));
    }

    #[test]
    fn test_compatible_partial_overlap() {
        let a = row(&[("x", iri("http://a")), ("y", iri("http://y"))]);
        let b = row(&[("x", iri("http://a")), ("z", iri("http://z"))]);
        assert!(BindingSet::is_compatible(&a, &b));
    }

    #[test]
    fn test_incompatible_partial_overlap() {
        let a = row(&[("x", iri("http://a")), ("y", iri("http://y"))]);
        let b = row(&[("x", iri("http://DIFFERENT")), ("z", iri("http://z"))]);
        assert!(!BindingSet::is_compatible(&a, &b));
    }

    #[test]
    fn test_compatible_all_shared_vars_agree() {
        let a = row(&[("x", iri("http://x")), ("y", lit("hello"))]);
        let b = row(&[("x", iri("http://x")), ("y", lit("hello"))]);
        assert!(BindingSet::is_compatible(&a, &b));
    }

    #[test]
    fn test_compatible_empty_rows() {
        let a: HashMap<String, RdfTerm> = HashMap::new();
        let b: HashMap<String, RdfTerm> = HashMap::new();
        assert!(BindingSet::is_compatible(&a, &b));
    }

    // ── MINUS semantics ───────────────────────────────────────────────────────

    #[test]
    fn test_minus_empty_self() {
        let s = BindingSet::new();
        let o = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        assert!(s.minus(&o).is_empty());
    }

    #[test]
    fn test_minus_empty_other() {
        let s = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let o = BindingSet::new();
        assert_eq!(s.minus(&o).len(), 1);
    }

    #[test]
    fn test_minus_removes_compatible_row_with_shared_var() {
        // x=a in self, x=a in other → row is removed
        let s = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let o = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        assert_eq!(s.minus(&o).len(), 0);
    }

    #[test]
    fn test_minus_keeps_row_different_value_shared_var() {
        // x=a in self, x=b in other → incompatible, row is kept
        let s = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let o = BindingSet::from_vec(vec![single("x", iri("http://b"))]);
        assert_eq!(s.minus(&o).len(), 1);
    }

    #[test]
    fn test_minus_keeps_row_no_shared_vars() {
        // SPARQL MINUS rule: no shared vars → row is KEPT
        let s = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let o = BindingSet::from_vec(vec![single("y", iri("http://b"))]);
        assert_eq!(s.minus(&o).len(), 1);
    }

    #[test]
    fn test_minus_partial_filter() {
        // Two rows in self: x=a (matches other) and x=b (does not)
        let s = BindingSet::from_vec(vec![
            single("x", iri("http://a")),
            single("x", iri("http://b")),
        ]);
        let o = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let result = s.minus(&o);
        assert_eq!(result.len(), 1);
        assert_eq!(result.bindings[0].get("x"), Some(&iri("http://b")));
    }

    #[test]
    fn test_minus_multiple_rows_in_other() {
        let s = BindingSet::from_vec(vec![
            single("x", iri("http://a")),
            single("x", iri("http://b")),
            single("x", iri("http://c")),
        ]);
        let o = BindingSet::from_vec(vec![
            single("x", iri("http://a")),
            single("x", iri("http://c")),
        ]);
        let result = s.minus(&o);
        assert_eq!(result.len(), 1);
        assert_eq!(result.bindings[0].get("x"), Some(&iri("http://b")));
    }

    #[test]
    fn test_minus_multi_variable_rows() {
        // Both rows share x, y but other has different y → row kept
        let s = BindingSet::from_vec(vec![row(&[("x", iri("http://a")), ("y", lit("foo"))])]);
        let o = BindingSet::from_vec(vec![row(&[("x", iri("http://a")), ("y", lit("bar"))])]);
        assert_eq!(s.minus(&o).len(), 1);
    }

    #[test]
    fn test_minus_row_with_no_vars_kept_always() {
        // Empty binding row has no shared vars with anything → kept
        let empty_row: HashMap<String, RdfTerm> = HashMap::new();
        let s = BindingSet::from_vec(vec![empty_row]);
        let o = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        assert_eq!(s.minus(&o).len(), 1);
    }

    // ── EXISTS filter ─────────────────────────────────────────────────────────

    #[test]
    fn test_exists_filter_empty_pattern() {
        let s = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let p = BindingSet::new();
        // No row in pattern → nothing is compatible → result is empty
        assert_eq!(s.exists_filter(&p).len(), 0);
    }

    #[test]
    fn test_exists_filter_compatible_keeps_row() {
        let s = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let p = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        assert_eq!(s.exists_filter(&p).len(), 1);
    }

    #[test]
    fn test_exists_filter_incompatible_removes_row() {
        let s = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let p = BindingSet::from_vec(vec![single("x", iri("http://b"))]);
        assert_eq!(s.exists_filter(&p).len(), 0);
    }

    #[test]
    fn test_exists_filter_no_shared_vars_compatible() {
        // No shared variables → compatible → row is kept
        let s = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let p = BindingSet::from_vec(vec![single("y", iri("http://b"))]);
        assert_eq!(s.exists_filter(&p).len(), 1);
    }

    #[test]
    fn test_exists_filter_mixed() {
        let s = BindingSet::from_vec(vec![
            single("x", iri("http://a")),
            single("x", iri("http://b")),
        ]);
        let p = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        assert_eq!(s.exists_filter(&p).len(), 1);
    }

    // ── UNION ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_union_empty_both() {
        let a = BindingSet::new();
        let b = BindingSet::new();
        assert!(a.union(&b).is_empty());
    }

    #[test]
    fn test_union_one_empty() {
        let a = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let b = BindingSet::new();
        assert_eq!(a.union(&b).len(), 1);
    }

    #[test]
    fn test_union_both_non_empty() {
        let a = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let b = BindingSet::from_vec(vec![single("y", iri("http://b"))]);
        assert_eq!(a.union(&b).len(), 2);
    }

    #[test]
    fn test_union_preserves_duplicates() {
        let a = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let b = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        assert_eq!(a.union(&b).len(), 2); // union does NOT deduplicate
    }

    // ── PROJECT ───────────────────────────────────────────────────────────────

    #[test]
    fn test_project_keeps_named_vars() {
        let bs = BindingSet::from_vec(vec![row(&[
            ("x", iri("http://x")),
            ("y", iri("http://y")),
            ("z", iri("http://z")),
        ])]);
        let proj = bs.project(&["x", "z"]);
        assert_eq!(proj.len(), 1);
        let r = &proj.bindings[0];
        assert!(r.contains_key("x"));
        assert!(!r.contains_key("y"));
        assert!(r.contains_key("z"));
    }

    #[test]
    fn test_project_no_vars() {
        let bs = BindingSet::from_vec(vec![single("x", iri("http://x"))]);
        let proj = bs.project(&[]);
        assert_eq!(proj.len(), 1);
        assert!(proj.bindings[0].is_empty());
    }

    #[test]
    fn test_project_missing_var_omitted() {
        let bs = BindingSet::from_vec(vec![single("x", iri("http://x"))]);
        let proj = bs.project(&["x", "missing"]);
        assert_eq!(proj.len(), 1);
        assert!(proj.bindings[0].contains_key("x"));
        assert!(!proj.bindings[0].contains_key("missing"));
    }

    #[test]
    fn test_project_empty_set() {
        let bs = BindingSet::new();
        let proj = bs.project(&["x"]);
        assert!(proj.is_empty());
    }

    // ── JOIN ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_join_empty_left() {
        let a = BindingSet::new();
        let b = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        assert!(a.join(&b).is_empty());
    }

    #[test]
    fn test_join_empty_right() {
        let a = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let b = BindingSet::new();
        assert!(a.join(&b).is_empty());
    }

    #[test]
    fn test_join_no_shared_vars_cross_product() {
        let a = BindingSet::from_vec(vec![single("x", iri("http://1"))]);
        let b = BindingSet::from_vec(vec![single("y", iri("http://2"))]);
        let j = a.join(&b);
        assert_eq!(j.len(), 1);
        assert!(j.bindings[0].contains_key("x"));
        assert!(j.bindings[0].contains_key("y"));
    }

    #[test]
    fn test_join_shared_var_compatible() {
        let a = BindingSet::from_vec(vec![row(&[("x", iri("http://a")), ("y", lit("foo"))])]);
        let b = BindingSet::from_vec(vec![row(&[("x", iri("http://a")), ("z", lit("bar"))])]);
        let j = a.join(&b);
        assert_eq!(j.len(), 1);
        assert_eq!(j.bindings[0].get("x"), Some(&iri("http://a")));
        assert_eq!(j.bindings[0].get("y"), Some(&lit("foo")));
        assert_eq!(j.bindings[0].get("z"), Some(&lit("bar")));
    }

    #[test]
    fn test_join_shared_var_incompatible_excluded() {
        let a = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        let b = BindingSet::from_vec(vec![single("x", iri("http://b"))]);
        assert!(a.join(&b).is_empty());
    }

    #[test]
    fn test_join_multiple_rows() {
        let a = BindingSet::from_vec(vec![
            single("x", iri("http://1")),
            single("x", iri("http://2")),
        ]);
        let b = BindingSet::from_vec(vec![
            single("x", iri("http://1")),
            single("x", iri("http://3")),
        ]);
        let j = a.join(&b);
        // Only (x=1, x=1) is compatible
        assert_eq!(j.len(), 1);
        assert_eq!(j.bindings[0].get("x"), Some(&iri("http://1")));
    }

    // ── DISTINCT ─────────────────────────────────────────────────────────────

    #[test]
    fn test_distinct_empty() {
        let bs = BindingSet::new();
        assert!(bs.distinct().is_empty());
    }

    #[test]
    fn test_distinct_no_duplicates() {
        let bs = BindingSet::from_vec(vec![
            single("x", iri("http://a")),
            single("x", iri("http://b")),
        ]);
        assert_eq!(bs.distinct().len(), 2);
    }

    #[test]
    fn test_distinct_with_duplicates() {
        let bs = BindingSet::from_vec(vec![
            single("x", iri("http://a")),
            single("x", iri("http://a")),
            single("x", iri("http://b")),
        ]);
        assert_eq!(bs.distinct().len(), 2);
    }

    #[test]
    fn test_distinct_multi_var_duplicates() {
        let r1 = row(&[("x", iri("http://x")), ("y", lit("foo"))]);
        let r2 = row(&[("x", iri("http://x")), ("y", lit("foo"))]);
        let r3 = row(&[("x", iri("http://x")), ("y", lit("bar"))]);
        let bs = BindingSet::from_vec(vec![r1, r2, r3]);
        assert_eq!(bs.distinct().len(), 2);
    }

    // ── iter ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_iter() {
        let bs = BindingSet::from_vec(vec![
            single("x", iri("http://1")),
            single("x", iri("http://2")),
        ]);
        let collected: Vec<_> = bs.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    // ── RdfTerm display / helpers ─────────────────────────────────────────────

    #[test]
    fn test_rdf_term_iri_display() {
        let t = iri("http://example.org/thing");
        assert_eq!(format!("{}", t), "<http://example.org/thing>");
    }

    #[test]
    fn test_rdf_term_literal_display() {
        let t = lit("hello");
        assert!(format!("{}", t).contains("hello"));
    }

    #[test]
    fn test_rdf_term_lang_display() {
        let t = RdfTerm::lang_literal("hello", "en");
        assert!(format!("{}", t).contains("@en"));
    }

    #[test]
    fn test_rdf_term_blank_display() {
        let t = bnode("b0");
        assert_eq!(format!("{}", t), "_:b0");
    }

    #[test]
    fn test_rdf_term_eq() {
        assert_eq!(iri("http://a"), iri("http://a"));
        assert_ne!(iri("http://a"), iri("http://b"));
        assert_ne!(iri("http://a"), lit("http://a"));
        assert_ne!(iri("http://a"), bnode("b0"));
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_minus_all_kept_when_no_shared_vars() {
        // All rows in self have no shared variables with other → all kept
        let s = BindingSet::from_vec(vec![
            single("x", iri("http://1")),
            single("x", iri("http://2")),
            single("x", iri("http://3")),
        ]);
        let o = BindingSet::from_vec(vec![single("y", iri("http://y"))]);
        assert_eq!(s.minus(&o).len(), 3);
    }

    #[test]
    fn test_minus_both_empty() {
        let s = BindingSet::new();
        let o = BindingSet::new();
        assert!(s.minus(&o).is_empty());
    }

    #[test]
    fn test_exists_filter_self_empty() {
        let s = BindingSet::new();
        let p = BindingSet::from_vec(vec![single("x", iri("http://a"))]);
        assert!(s.exists_filter(&p).is_empty());
    }

    #[test]
    fn test_join_all_compatible_no_shared() {
        // 2×2 = 4 rows in cross product
        let a = BindingSet::from_vec(vec![
            single("x", iri("http://1")),
            single("x", iri("http://2")),
        ]);
        let b = BindingSet::from_vec(vec![
            single("y", iri("http://a")),
            single("y", iri("http://b")),
        ]);
        assert_eq!(a.join(&b).len(), 4);
    }

    #[test]
    fn test_project_and_distinct() {
        // Project x,y then distinct: two identical projected rows become one
        let bs = BindingSet::from_vec(vec![
            row(&[("x", iri("http://x")), ("y", lit("foo")), ("z", lit("A"))]),
            row(&[("x", iri("http://x")), ("y", lit("foo")), ("z", lit("B"))]),
        ]);
        let proj = bs.project(&["x", "y"]).distinct();
        assert_eq!(proj.len(), 1);
    }

    #[test]
    fn test_union_order_preserved() {
        let a = BindingSet::from_vec(vec![single("x", iri("http://1"))]);
        let b = BindingSet::from_vec(vec![single("x", iri("http://2"))]);
        let u = a.union(&b);
        assert_eq!(u.bindings[0].get("x"), Some(&iri("http://1")));
        assert_eq!(u.bindings[1].get("x"), Some(&iri("http://2")));
    }

    #[test]
    fn test_minus_literal_terms() {
        let s = BindingSet::from_vec(vec![single("label", lit("hello"))]);
        let o = BindingSet::from_vec(vec![single("label", lit("hello"))]);
        assert_eq!(s.minus(&o).len(), 0);
    }

    #[test]
    fn test_minus_blank_node_terms() {
        let s = BindingSet::from_vec(vec![single("b", bnode("b0"))]);
        let o = BindingSet::from_vec(vec![single("b", bnode("b0"))]);
        assert_eq!(s.minus(&o).len(), 0);
    }

    #[test]
    fn test_exists_filter_multiple_pattern_rows() {
        // Row is kept if ANY pattern row is compatible
        let s = BindingSet::from_vec(vec![single("x", iri("http://c"))]);
        let p = BindingSet::from_vec(vec![
            single("x", iri("http://a")),
            single("x", iri("http://b")),
            single("x", iri("http://c")),
        ]);
        assert_eq!(s.exists_filter(&p).len(), 1);
    }

    #[test]
    fn test_join_preserves_all_vars() {
        let a = BindingSet::from_vec(vec![row(&[
            ("subject", iri("http://s")),
            ("predicate", iri("http://p")),
        ])]);
        let b = BindingSet::from_vec(vec![row(&[
            ("predicate", iri("http://p")),
            ("object", lit("value")),
        ])]);
        let j = a.join(&b);
        assert_eq!(j.len(), 1);
        let r = &j.bindings[0];
        assert_eq!(r.get("subject"), Some(&iri("http://s")));
        assert_eq!(r.get("predicate"), Some(&iri("http://p")));
        assert_eq!(r.get("object"), Some(&lit("value")));
    }
}
