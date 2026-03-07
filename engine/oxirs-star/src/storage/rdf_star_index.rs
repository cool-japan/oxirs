//! RDF-star storage index with compound B-tree key layout.
//!
//! `RdfStarIndex` stores quoted triples using a compound key:
//! `(outer_s, outer_p, outer_o, inner_s, inner_p, inner_o)`
//! enabling efficient prefix scans for nested triple pattern matching.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::model::{StarTerm, StarTriple};
use crate::{StarError, StarResult};

// ============================================================================
// Compound key
// ============================================================================

/// A serialisable representation of a `StarTerm` component used as part of
/// a B-tree key.  We use the canonical Turtle-star string representation so
/// that terms sort lexicographically and prefix scans work correctly.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TermKey(pub String);

impl TermKey {
    /// Produce a `TermKey` from a `StarTerm`.
    pub fn from_term(term: &StarTerm) -> Self {
        TermKey(term_to_string(term))
    }

    /// A wildcard key (lexicographically minimal sentinel).
    pub fn wildcard() -> Self {
        TermKey(String::new())
    }

    /// Returns `true` if this key is a wildcard.
    pub fn is_wildcard(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for TermKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Convert a `StarTerm` to its canonical string for use in keys.
fn term_to_string(term: &StarTerm) -> String {
    match term {
        StarTerm::NamedNode(n) => format!("<{}>", n.iri),
        StarTerm::BlankNode(b) => format!("_:{}", b.id),
        StarTerm::Literal(l) => {
            if let Some(lang) = &l.language {
                format!("{:?}@{lang}", l.value)
            } else if let Some(dt) = &l.datatype {
                format!("{:?}^^<{}>", l.value, dt.iri)
            } else {
                format!("{:?}", l.value)
            }
        }
        StarTerm::QuotedTriple(t) => {
            format!(
                "<<{} {} {}>>",
                term_to_string(&t.subject),
                term_to_string(&t.predicate),
                term_to_string(&t.object),
            )
        }
        StarTerm::Variable(v) => format!("?{}", v.name),
    }
}

// ============================================================================
// Compound key for the outer+inner triple pair
// ============================================================================

/// Six-component compound key: `(outer_s, outer_p, outer_o, inner_s, inner_p, inner_o)`.
///
/// The first three components identify the *outer* (meta) triple.
/// The last three identify the *inner* quoted triple embedded in its subject.
/// If the outer triple's subject is not a quoted triple, the inner components
/// are set to `TermKey("")` (wildcard sentinel).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CompoundKey {
    pub outer_s: TermKey,
    pub outer_p: TermKey,
    pub outer_o: TermKey,
    pub inner_s: TermKey,
    pub inner_p: TermKey,
    pub inner_o: TermKey,
}

impl CompoundKey {
    /// Build a compound key from an outer triple.
    ///
    /// If `outer.subject` is a quoted triple, the inner components are set
    /// from that quoted triple.  Otherwise they are all set to `TermKey("")`.
    pub fn from_triple(outer: &StarTriple) -> Self {
        let (inner_s, inner_p, inner_o) = match &outer.subject {
            StarTerm::QuotedTriple(inner) => (
                TermKey::from_term(&inner.subject),
                TermKey::from_term(&inner.predicate),
                TermKey::from_term(&inner.object),
            ),
            other => (
                TermKey::from_term(other),
                TermKey::wildcard(),
                TermKey::wildcard(),
            ),
        };

        Self {
            outer_s: TermKey::from_term(&outer.subject),
            outer_p: TermKey::from_term(&outer.predicate),
            outer_o: TermKey::from_term(&outer.object),
            inner_s,
            inner_p,
            inner_o,
        }
    }
}

// ============================================================================
// Pattern for prefix scans
// ============================================================================

/// A partial compound key used for prefix scanning.
///
/// `None` in any position is treated as a wildcard that matches any value.
#[derive(Debug, Clone, Default)]
pub struct CompoundPattern {
    pub outer_s: Option<TermKey>,
    pub outer_p: Option<TermKey>,
    pub outer_o: Option<TermKey>,
    pub inner_s: Option<TermKey>,
    pub inner_p: Option<TermKey>,
    pub inner_o: Option<TermKey>,
}

impl CompoundPattern {
    /// Create a pattern where all components are wildcards.
    pub fn wildcard() -> Self {
        Self::default()
    }

    /// Build a `CompoundPattern` from a triple pattern.
    /// Variables in the triple → `None` in the pattern.
    pub fn from_triple_pattern(pattern: &StarTriple) -> Self {
        Self {
            outer_s: if is_variable(&pattern.subject) {
                None
            } else {
                Some(TermKey::from_term(&pattern.subject))
            },
            outer_p: if is_variable(&pattern.predicate) {
                None
            } else {
                Some(TermKey::from_term(&pattern.predicate))
            },
            outer_o: if is_variable(&pattern.object) {
                None
            } else {
                Some(TermKey::from_term(&pattern.object))
            },
            inner_s: extract_inner_s(&pattern.subject),
            inner_p: extract_inner_p(&pattern.subject),
            inner_o: extract_inner_o(&pattern.subject),
        }
    }

    /// Return `true` if `key` matches this pattern.
    pub fn matches(&self, key: &CompoundKey) -> bool {
        matches_component(&self.outer_s, &key.outer_s)
            && matches_component(&self.outer_p, &key.outer_p)
            && matches_component(&self.outer_o, &key.outer_o)
            && matches_component(&self.inner_s, &key.inner_s)
            && matches_component(&self.inner_p, &key.inner_p)
            && matches_component(&self.inner_o, &key.inner_o)
    }
}

fn is_variable(term: &StarTerm) -> bool {
    matches!(term, StarTerm::Variable(_))
}

fn extract_inner_s(term: &StarTerm) -> Option<TermKey> {
    if let StarTerm::QuotedTriple(inner) = term {
        if !is_variable(&inner.subject) {
            return Some(TermKey::from_term(&inner.subject));
        }
    }
    None
}

fn extract_inner_p(term: &StarTerm) -> Option<TermKey> {
    if let StarTerm::QuotedTriple(inner) = term {
        if !is_variable(&inner.predicate) {
            return Some(TermKey::from_term(&inner.predicate));
        }
    }
    None
}

fn extract_inner_o(term: &StarTerm) -> Option<TermKey> {
    if let StarTerm::QuotedTriple(inner) = term {
        if !is_variable(&inner.object) {
            return Some(TermKey::from_term(&inner.object));
        }
    }
    None
}

fn matches_component(pattern: &Option<TermKey>, key: &TermKey) -> bool {
    match pattern {
        None => true, // wildcard
        Some(p) => p == key,
    }
}

// ============================================================================
// RdfStarIndex
// ============================================================================

/// Efficient storage index for RDF-star triples using a compound B-tree key.
///
/// Supports:
/// - `insert`: O(log n) insertion.
/// - `remove`: O(log n) removal.
/// - `prefix_scan`: efficient prefix scan for nested triple pattern matching.
/// - `len` / `is_empty`: O(1) size queries.
#[derive(Debug, Default)]
pub struct RdfStarIndex {
    /// Sorted map from compound key → original triple.
    btree: BTreeMap<CompoundKey, StarTriple>,
}

impl RdfStarIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a triple into the index.  Returns `true` if the triple was
    /// newly inserted (i.e. it was not already present).
    pub fn insert(&mut self, triple: StarTriple) -> bool {
        let key = CompoundKey::from_triple(&triple);
        let was_absent = !self.btree.contains_key(&key);
        self.btree.insert(key, triple);
        was_absent
    }

    /// Remove a triple from the index.  Returns `true` if the triple was
    /// present and removed.
    pub fn remove(&mut self, triple: &StarTriple) -> bool {
        let key = CompoundKey::from_triple(triple);
        self.btree.remove(&key).is_some()
    }

    /// Returns `true` if the triple is in the index.
    pub fn contains(&self, triple: &StarTriple) -> bool {
        let key = CompoundKey::from_triple(triple);
        self.btree.contains_key(&key)
    }

    /// Number of triples in the index.
    pub fn len(&self) -> usize {
        self.btree.len()
    }

    /// `true` if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.btree.is_empty()
    }

    /// Prefix scan: return all triples whose compound key matches `pattern`.
    ///
    /// This iterates over all stored entries (full scan with pattern filter).
    /// Future work: implement a true prefix range scan using the first bound
    /// component as the B-tree lower bound.
    pub fn prefix_scan(&self, pattern: &CompoundPattern) -> Vec<StarTriple> {
        self.btree
            .iter()
            .filter(|(key, _)| pattern.matches(key))
            .map(|(_, triple)| triple.clone())
            .collect()
    }

    /// Scan for all triples whose outer subject is a specific quoted triple.
    pub fn scan_by_quoted_subject(&self, inner: &StarTriple) -> Vec<StarTriple> {
        let inner_s = TermKey::from_term(&inner.subject);
        let inner_p = TermKey::from_term(&inner.predicate);
        let inner_o = TermKey::from_term(&inner.object);

        let pattern = CompoundPattern {
            outer_s: None,
            outer_p: None,
            outer_o: None,
            inner_s: Some(inner_s),
            inner_p: Some(inner_p),
            inner_o: Some(inner_o),
        };
        self.prefix_scan(&pattern)
    }

    /// Iterate over all triples in key order.
    pub fn iter(&self) -> impl Iterator<Item = &StarTriple> {
        self.btree.values()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.btree.clear();
    }

    /// Validate the index integrity (checks that each stored key matches the
    /// stored triple).
    pub fn validate(&self) -> StarResult<()> {
        for (key, triple) in &self.btree {
            let expected = CompoundKey::from_triple(triple);
            if *key != expected {
                return Err(StarError::QueryError {
                    message: format!(
                        "Index corruption: key {key:?} does not match triple {triple:?}"
                    ),
                    query_fragment: None,
                    position: None,
                    suggestion: None,
                });
            }
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StarTerm, StarTriple, Variable};

    fn iri(s: &str) -> StarTerm {
        StarTerm::iri(s).expect("valid IRI")
    }

    fn lit(s: &str) -> StarTerm {
        StarTerm::literal(s).expect("ok")
    }

    fn triple(s: &str, p: &str, o: &str) -> StarTriple {
        StarTriple::new(iri(s), iri(p), iri(o))
    }

    fn var(name: &str) -> StarTerm {
        StarTerm::Variable(Variable { name: name.into() })
    }

    // ------------------------------------------------------------------
    // Basic insert / contains / remove
    // ------------------------------------------------------------------

    #[test]
    fn test_index_insert_and_contains() {
        let mut idx = RdfStarIndex::new();
        let t = triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        assert!(idx.insert(t.clone()));
        assert!(idx.contains(&t));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_index_insert_duplicate() {
        let mut idx = RdfStarIndex::new();
        let t = triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        assert!(idx.insert(t.clone())); // first insert
        assert!(!idx.insert(t.clone())); // duplicate
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_index_remove() {
        let mut idx = RdfStarIndex::new();
        let t = triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        idx.insert(t.clone());
        assert!(idx.remove(&t));
        assert!(!idx.contains(&t));
        assert!(idx.is_empty());
    }

    #[test]
    fn test_index_remove_nonexistent() {
        let mut idx = RdfStarIndex::new();
        let t = triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        assert!(!idx.remove(&t));
    }

    // ------------------------------------------------------------------
    // Quoted triple insertion
    // ------------------------------------------------------------------

    #[test]
    fn test_index_quoted_triple_subject() {
        let mut idx = RdfStarIndex::new();
        let inner = triple(
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://ex.org/certainty"),
            lit("0.9"),
        );
        assert!(idx.insert(meta.clone()));
        assert!(idx.contains(&meta));
        assert_eq!(idx.len(), 1);
    }

    // ------------------------------------------------------------------
    // Prefix scan tests
    // ------------------------------------------------------------------

    #[test]
    fn test_prefix_scan_wildcard_returns_all() {
        let mut idx = RdfStarIndex::new();
        idx.insert(triple(
            "http://ex.org/s1",
            "http://ex.org/p",
            "http://ex.org/o1",
        ));
        idx.insert(triple(
            "http://ex.org/s2",
            "http://ex.org/p",
            "http://ex.org/o2",
        ));
        idx.insert(triple(
            "http://ex.org/s3",
            "http://ex.org/p",
            "http://ex.org/o3",
        ));

        let results = idx.prefix_scan(&CompoundPattern::wildcard());
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_prefix_scan_bound_predicate() {
        let mut idx = RdfStarIndex::new();
        idx.insert(triple(
            "http://ex.org/s1",
            "http://ex.org/knows",
            "http://ex.org/o1",
        ));
        idx.insert(triple(
            "http://ex.org/s2",
            "http://ex.org/age",
            "http://ex.org/o2",
        ));
        idx.insert(triple(
            "http://ex.org/s3",
            "http://ex.org/knows",
            "http://ex.org/o3",
        ));

        let pattern = CompoundPattern {
            outer_p: Some(TermKey::from_term(&iri("http://ex.org/knows"))),
            ..CompoundPattern::default()
        };
        let results = idx.prefix_scan(&pattern);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_prefix_scan_no_match() {
        let mut idx = RdfStarIndex::new();
        idx.insert(triple(
            "http://ex.org/s",
            "http://ex.org/p",
            "http://ex.org/o",
        ));

        let pattern = CompoundPattern {
            outer_s: Some(TermKey::from_term(&iri("http://ex.org/nobody"))),
            ..CompoundPattern::default()
        };
        let results = idx.prefix_scan(&pattern);
        assert!(results.is_empty());
    }

    #[test]
    fn test_scan_by_quoted_subject() {
        let mut idx = RdfStarIndex::new();
        let inner = triple(
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
        );
        let meta1 = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://ex.org/certainty"),
            lit("0.9"),
        );
        let meta2 = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://ex.org/source"),
            iri("http://ex.org/db"),
        );
        let unrelated = triple("http://ex.org/bob", "http://ex.org/age", "http://ex.org/25");

        idx.insert(meta1.clone());
        idx.insert(meta2.clone());
        idx.insert(unrelated.clone());

        let results = idx.scan_by_quoted_subject(&inner);
        assert_eq!(results.len(), 2);
    }

    // ------------------------------------------------------------------
    // CompoundPattern from triple pattern
    // ------------------------------------------------------------------

    #[test]
    fn test_pattern_from_triple_pattern_bound_s() {
        let pattern = StarTriple::new(iri("http://ex.org/s"), var("p"), var("o"));
        let cp = CompoundPattern::from_triple_pattern(&pattern);
        assert!(cp.outer_s.is_some());
        assert!(cp.outer_p.is_none()); // variable
        assert!(cp.outer_o.is_none()); // variable
    }

    #[test]
    fn test_pattern_from_triple_pattern_all_wildcards() {
        let pattern = StarTriple::new(var("s"), var("p"), var("o"));
        let cp = CompoundPattern::from_triple_pattern(&pattern);
        assert!(cp.outer_s.is_none());
        assert!(cp.outer_p.is_none());
        assert!(cp.outer_o.is_none());
    }

    #[test]
    fn test_pattern_from_quoted_subject() {
        let inner = triple("http://ex.org/a", "http://ex.org/b", "http://ex.org/c");
        let pattern = StarTriple::new(StarTerm::quoted_triple(inner), var("p"), var("o"));
        let cp = CompoundPattern::from_triple_pattern(&pattern);
        // inner_s should be bound.
        assert!(cp.inner_s.is_some());
        assert!(cp.inner_p.is_some());
        assert!(cp.inner_o.is_some());
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    #[test]
    fn test_validate_clean_index() {
        let mut idx = RdfStarIndex::new();
        idx.insert(triple(
            "http://ex.org/s",
            "http://ex.org/p",
            "http://ex.org/o",
        ));
        assert!(idx.validate().is_ok());
    }

    // ------------------------------------------------------------------
    // Clear
    // ------------------------------------------------------------------

    #[test]
    fn test_clear() {
        let mut idx = RdfStarIndex::new();
        idx.insert(triple(
            "http://ex.org/s",
            "http://ex.org/p",
            "http://ex.org/o",
        ));
        idx.clear();
        assert!(idx.is_empty());
    }

    // ------------------------------------------------------------------
    // Sorting / ordering
    // ------------------------------------------------------------------

    #[test]
    fn test_term_key_ordering() {
        let a = TermKey::from_term(&iri("http://ex.org/a"));
        let b = TermKey::from_term(&iri("http://ex.org/b"));
        assert!(a < b);
    }

    #[test]
    fn test_wildcard_is_lexicographically_smallest() {
        let wc = TermKey::wildcard();
        let any = TermKey::from_term(&iri("http://ex.org/any"));
        assert!(wc < any);
    }
}
