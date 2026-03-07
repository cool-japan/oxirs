//! Optimized Storage for Quoted Triple Graphs (RDF-star)
//!
//! This module provides `QuotedGraph`: a compact, indexed storage structure for
//! *quoted triples* as defined in RDF-star / RDF 1.2.
//!
//! A **quoted triple** is an RDF triple that appears as the subject or object of
//! another triple (an *asserted triple*), enabling statements-about-statements.
//!
//! # Design
//!
//! - Quoted triples are **interned** on insertion: each unique triple gets a
//!   `QuotedId` (a monotonically increasing `u64`).
//! - Three secondary indexes (`by_subject`, `by_predicate`, `by_object`) allow
//!   O(1) average-case lookup by any single component.
//! - **Assertion triples** (`<<s p o>> pred obj`) are stored separately and
//!   indexed by `QuotedId`.
//!
//! # Example
//!
//! ```rust
//! use oxirs_star::quoted_graph::{QuotedGraph, QuotedTriple};
//!
//! let mut g = QuotedGraph::new();
//!
//! let qid = g.intern(QuotedTriple {
//!     subject: "http://example.org/alice".into(),
//!     predicate: "http://example.org/age".into(),
//!     object: "25".into(),
//! });
//!
//! g.add_assertion(qid, "http://example.org/certainty", "0.9");
//!
//! let assertions = g.get_assertions(qid);
//! assert_eq!(assertions.len(), 1);
//! ```

use std::collections::HashMap;

// ─── QuotedId ────────────────────────────────────────────────────────────────

/// A compact, unique identifier for an interned quoted triple.
///
/// IDs start from 1; 0 is reserved as a sentinel "not found" value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuotedId(pub u64);

impl QuotedId {
    /// The null/sentinel ID.
    pub const NULL: Self = Self(0);

    /// Returns `true` if this is the null sentinel.
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

// ─── QuotedTriple ────────────────────────────────────────────────────────────

/// An RDF-star quoted triple `<< subject predicate object >>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuotedTriple {
    /// Subject component (IRI or blank-node identifier)
    pub subject: String,
    /// Predicate component (must be an IRI)
    pub predicate: String,
    /// Object component (IRI, blank node, or literal)
    pub object: String,
}

impl QuotedTriple {
    /// Construct a new quoted triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Format in Turtle-star notation.
    pub fn to_turtle_star(&self) -> String {
        format!("<< {} {} {} >>", self.subject, self.predicate, self.object)
    }
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Summary statistics for a `QuotedGraph`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuotedGraphStats {
    /// Number of unique interned quoted triples
    pub unique_quoted: usize,
    /// Total number of assertion triples across all quoted IDs
    pub total_assertions: usize,
}

// ─── QuotedGraph ──────────────────────────────────────────────────────────────

/// An optimized, indexed storage for quoted triples and their assertion metadata.
///
/// Invariants maintained at all times:
/// - Every `QuotedId` in any index is present in both `triples_by_id` and `id_by_triple`
/// - IDs are assigned sequentially starting from 1
/// - Duplicate triples receive the same `QuotedId`
#[derive(Debug, Default)]
pub struct QuotedGraph {
    /// Forward map: ID → triple
    triples_by_id: HashMap<QuotedId, QuotedTriple>,
    /// Reverse map: triple → ID (for deduplication)
    id_by_triple: HashMap<QuotedTriple, QuotedId>,
    /// Index by subject string
    by_subject: HashMap<String, Vec<QuotedId>>,
    /// Index by predicate string
    by_predicate: HashMap<String, Vec<QuotedId>>,
    /// Index by object string
    by_object: HashMap<String, Vec<QuotedId>>,
    /// Assertion triples keyed by the cited `QuotedId`
    /// Value: list of `(predicate, object)` pairs
    assertions: HashMap<QuotedId, Vec<(String, String)>>,
    /// Next ID to assign
    next_id: u64,
}

impl QuotedGraph {
    /// Create a new, empty graph.
    pub fn new() -> Self {
        Self {
            next_id: 1,
            ..Default::default()
        }
    }

    /// Intern a quoted triple, returning its stable `QuotedId`.
    ///
    /// If the triple is already stored, the existing ID is returned (no duplicate).
    pub fn intern(&mut self, triple: QuotedTriple) -> QuotedId {
        if let Some(&existing_id) = self.id_by_triple.get(&triple) {
            return existing_id;
        }

        let id = QuotedId(self.next_id);
        self.next_id += 1;

        // Update indexes
        self.by_subject
            .entry(triple.subject.clone())
            .or_default()
            .push(id);
        self.by_predicate
            .entry(triple.predicate.clone())
            .or_default()
            .push(id);
        self.by_object
            .entry(triple.object.clone())
            .or_default()
            .push(id);

        self.id_by_triple.insert(triple.clone(), id);
        self.triples_by_id.insert(id, triple);

        id
    }

    /// Look up a quoted triple by its ID.
    pub fn lookup(&self, id: QuotedId) -> Option<&QuotedTriple> {
        self.triples_by_id.get(&id)
    }

    /// Find all quoted triples with the given subject.
    pub fn find_by_subject(&self, s: &str) -> Vec<QuotedId> {
        self.by_subject.get(s).cloned().unwrap_or_default()
    }

    /// Find all quoted triples with the given predicate.
    pub fn find_by_predicate(&self, p: &str) -> Vec<QuotedId> {
        self.by_predicate.get(p).cloned().unwrap_or_default()
    }

    /// Find all quoted triples with the given object.
    pub fn find_by_object(&self, o: &str) -> Vec<QuotedId> {
        self.by_object.get(o).cloned().unwrap_or_default()
    }

    /// Add an assertion triple about a quoted triple: `<<triple>> predicate object`.
    ///
    /// Multiple assertions about the same quoted ID are accumulated.
    pub fn add_assertion(
        &mut self,
        quoted_id: QuotedId,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) {
        self.assertions
            .entry(quoted_id)
            .or_default()
            .push((predicate.into(), object.into()));
    }

    /// Retrieve all assertion `(predicate, object)` pairs for a quoted triple.
    pub fn get_assertions(&self, quoted_id: QuotedId) -> Vec<(String, String)> {
        self.assertions.get(&quoted_id).cloned().unwrap_or_default()
    }

    /// Return summary statistics.
    pub fn stats(&self) -> QuotedGraphStats {
        let total_assertions = self.assertions.values().map(Vec::len).sum();
        QuotedGraphStats {
            unique_quoted: self.triples_by_id.len(),
            total_assertions,
        }
    }

    /// Number of unique quoted triples stored.
    pub fn len(&self) -> usize {
        self.triples_by_id.len()
    }

    /// Returns `true` if no quoted triples are stored.
    pub fn is_empty(&self) -> bool {
        self.triples_by_id.is_empty()
    }

    /// Clear all stored triples and assertions (reset to initial state).
    pub fn clear(&mut self) {
        self.triples_by_id.clear();
        self.id_by_triple.clear();
        self.by_subject.clear();
        self.by_predicate.clear();
        self.by_object.clear();
        self.assertions.clear();
        self.next_id = 1;
    }

    /// Check whether a quoted triple has been interned.
    pub fn contains(&self, triple: &QuotedTriple) -> bool {
        self.id_by_triple.contains_key(triple)
    }

    /// Retrieve the `QuotedId` for an existing triple without interning.
    pub fn get_id(&self, triple: &QuotedTriple) -> Option<QuotedId> {
        self.id_by_triple.get(triple).copied()
    }

    /// All interned `QuotedId`s.
    pub fn all_ids(&self) -> Vec<QuotedId> {
        let mut ids: Vec<QuotedId> = self.triples_by_id.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Find quoted triples by subject AND object (conjunction of two indexes).
    pub fn find_by_subject_and_object(&self, s: &str, o: &str) -> Vec<QuotedId> {
        let by_s: std::collections::HashSet<QuotedId> =
            self.find_by_subject(s).into_iter().collect();
        self.find_by_object(o)
            .into_iter()
            .filter(|id| by_s.contains(id))
            .collect()
    }

    /// Find quoted triples that match a subject-predicate-object pattern.
    ///
    /// Pass `None` for positions to be treated as wildcards.
    pub fn find_pattern(&self, s: Option<&str>, p: Option<&str>, o: Option<&str>) -> Vec<QuotedId> {
        // Compute candidate sets for each specified component, then intersect
        let s_set: Option<std::collections::HashSet<QuotedId>> =
            s.map(|v| self.find_by_subject(v).into_iter().collect());
        let p_set: Option<std::collections::HashSet<QuotedId>> =
            p.map(|v| self.find_by_predicate(v).into_iter().collect());
        let o_set: Option<std::collections::HashSet<QuotedId>> =
            o.map(|v| self.find_by_object(v).into_iter().collect());

        let mut result: Vec<QuotedId> = self.all_ids();

        if let Some(s_ids) = &s_set {
            result.retain(|id| s_ids.contains(id));
        }
        if let Some(p_ids) = &p_set {
            result.retain(|id| p_ids.contains(id));
        }
        if let Some(o_ids) = &o_set {
            result.retain(|id| o_ids.contains(id));
        }

        result
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn qt(s: &str, p: &str, o: &str) -> QuotedTriple {
        QuotedTriple::new(s, p, o)
    }

    // ── QuotedId ─────────────────────────────────────────────────────────────

    #[test]
    fn test_quoted_id_null() {
        assert!(QuotedId::NULL.is_null());
        assert_eq!(QuotedId::NULL.0, 0);
    }

    #[test]
    fn test_quoted_id_non_null() {
        assert!(!QuotedId(1).is_null());
    }

    #[test]
    fn test_quoted_id_ordering() {
        assert!(QuotedId(1) < QuotedId(2));
        assert!(QuotedId(10) > QuotedId(5));
    }

    // ── QuotedTriple ─────────────────────────────────────────────────────────

    #[test]
    fn test_quoted_triple_new() {
        let t = qt("s", "p", "o");
        assert_eq!(t.subject, "s");
        assert_eq!(t.predicate, "p");
        assert_eq!(t.object, "o");
    }

    #[test]
    fn test_quoted_triple_turtle_star() {
        let t = qt("s", "p", "o");
        let turtle = t.to_turtle_star();
        assert!(turtle.contains("<<"));
        assert!(turtle.contains(">>"));
        assert!(turtle.contains("s"));
    }

    #[test]
    fn test_quoted_triple_equality() {
        let a = qt("s", "p", "o");
        let b = qt("s", "p", "o");
        assert_eq!(a, b);
    }

    // ── Intern and lookup ────────────────────────────────────────────────────

    #[test]
    fn test_intern_new_triple() {
        let mut g = QuotedGraph::new();
        let id = g.intern(qt("s", "p", "o"));
        assert!(!id.is_null());
        assert_eq!(id.0, 1);
    }

    #[test]
    fn test_intern_duplicate_returns_same_id() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("s", "p", "o"));
        let id2 = g.intern(qt("s", "p", "o"));
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_intern_different_triples_different_ids() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("s", "p", "o1"));
        let id2 = g.intern(qt("s", "p", "o2"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_lookup_existing() {
        let mut g = QuotedGraph::new();
        let triple = qt("subj", "pred", "obj");
        let id = g.intern(triple.clone());
        let found = g.lookup(id).expect("should exist");
        assert_eq!(found, &triple);
    }

    #[test]
    fn test_lookup_missing() {
        let g = QuotedGraph::new();
        assert!(g.lookup(QuotedId(99)).is_none());
    }

    #[test]
    fn test_len() {
        let mut g = QuotedGraph::new();
        assert_eq!(g.len(), 0);
        g.intern(qt("s", "p", "o"));
        assert_eq!(g.len(), 1);
        g.intern(qt("s", "p", "o")); // duplicate
        assert_eq!(g.len(), 1);
        g.intern(qt("s", "p", "o2"));
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn test_is_empty() {
        let mut g = QuotedGraph::new();
        assert!(g.is_empty());
        g.intern(qt("s", "p", "o"));
        assert!(!g.is_empty());
    }

    #[test]
    fn test_contains() {
        let mut g = QuotedGraph::new();
        let triple = qt("s", "p", "o");
        assert!(!g.contains(&triple));
        g.intern(triple.clone());
        assert!(g.contains(&triple));
    }

    #[test]
    fn test_get_id() {
        let mut g = QuotedGraph::new();
        let triple = qt("s", "p", "o");
        assert!(g.get_id(&triple).is_none());
        let id = g.intern(triple.clone());
        assert_eq!(g.get_id(&triple), Some(id));
    }

    // ── Index lookups ─────────────────────────────────────────────────────────

    #[test]
    fn test_find_by_subject() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("alice", "age", "30"));
        let id2 = g.intern(qt("alice", "name", "Alice"));
        let _id3 = g.intern(qt("bob", "age", "25"));

        let mut found = g.find_by_subject("alice");
        found.sort();
        assert!(found.contains(&id1));
        assert!(found.contains(&id2));
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_find_by_predicate() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("alice", "age", "30"));
        let id2 = g.intern(qt("bob", "age", "25"));
        let _id3 = g.intern(qt("alice", "name", "Alice"));

        let found = g.find_by_predicate("age");
        assert_eq!(found.len(), 2);
        assert!(found.contains(&id1));
        assert!(found.contains(&id2));
    }

    #[test]
    fn test_find_by_object() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("alice", "type", "Person"));
        let id2 = g.intern(qt("bob", "type", "Person"));
        let _id3 = g.intern(qt("corp", "type", "Organization"));

        let found = g.find_by_object("Person");
        assert_eq!(found.len(), 2);
        assert!(found.contains(&id1));
        assert!(found.contains(&id2));
    }

    #[test]
    fn test_find_by_subject_none() {
        let mut g = QuotedGraph::new();
        g.intern(qt("alice", "age", "30"));
        let found = g.find_by_subject("nobody");
        assert!(found.is_empty());
    }

    // ── Assertions ────────────────────────────────────────────────────────────

    #[test]
    fn test_add_assertion_basic() {
        let mut g = QuotedGraph::new();
        let id = g.intern(qt("alice", "age", "30"));
        g.add_assertion(id, "certainty", "0.9");
        let assertions = g.get_assertions(id);
        assert_eq!(assertions.len(), 1);
        assert_eq!(assertions[0], ("certainty".to_string(), "0.9".to_string()));
    }

    #[test]
    fn test_add_assertion_multiple() {
        let mut g = QuotedGraph::new();
        let id = g.intern(qt("alice", "age", "30"));
        g.add_assertion(id, "certainty", "0.9");
        g.add_assertion(id, "source", "http://example.org/census");
        let assertions = g.get_assertions(id);
        assert_eq!(assertions.len(), 2);
    }

    #[test]
    fn test_get_assertions_no_assertions() {
        let mut g = QuotedGraph::new();
        let id = g.intern(qt("s", "p", "o"));
        let assertions = g.get_assertions(id);
        assert!(assertions.is_empty());
    }

    #[test]
    fn test_assertions_per_quoted_id() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("a", "p", "o"));
        let id2 = g.intern(qt("b", "p", "o"));
        g.add_assertion(id1, "k", "v1");
        g.add_assertion(id2, "k", "v2");
        assert_eq!(g.get_assertions(id1)[0].1, "v1");
        assert_eq!(g.get_assertions(id2)[0].1, "v2");
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let g = QuotedGraph::new();
        let s = g.stats();
        assert_eq!(s.unique_quoted, 0);
        assert_eq!(s.total_assertions, 0);
    }

    #[test]
    fn test_stats_after_insert() {
        let mut g = QuotedGraph::new();
        let id = g.intern(qt("s", "p", "o"));
        g.add_assertion(id, "k1", "v1");
        g.add_assertion(id, "k2", "v2");
        let id2 = g.intern(qt("s2", "p", "o"));
        g.add_assertion(id2, "k", "v");
        let s = g.stats();
        assert_eq!(s.unique_quoted, 2);
        assert_eq!(s.total_assertions, 3);
    }

    // ── all_ids and clear ─────────────────────────────────────────────────────

    #[test]
    fn test_all_ids_sorted() {
        let mut g = QuotedGraph::new();
        g.intern(qt("a", "p", "o1"));
        g.intern(qt("a", "p", "o2"));
        g.intern(qt("a", "p", "o3"));
        let ids = g.all_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_clear() {
        let mut g = QuotedGraph::new();
        let id = g.intern(qt("s", "p", "o"));
        g.add_assertion(id, "k", "v");
        g.clear();
        assert!(g.is_empty());
        assert!(g.get_assertions(id).is_empty());
        // After clear, IDs restart from 1
        let new_id = g.intern(qt("s", "p", "o"));
        assert_eq!(new_id.0, 1);
    }

    // ── find_by_subject_and_object ────────────────────────────────────────────

    #[test]
    fn test_find_by_subject_and_object() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("alice", "knows", "bob"));
        let _id2 = g.intern(qt("alice", "age", "30"));
        let _id3 = g.intern(qt("carol", "knows", "bob"));

        let found = g.find_by_subject_and_object("alice", "bob");
        assert_eq!(found, vec![id1]);
    }

    // ── find_pattern ─────────────────────────────────────────────────────────

    #[test]
    fn test_find_pattern_wildcard() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("a", "p", "o"));
        let id2 = g.intern(qt("b", "p", "o"));
        let mut found = g.find_pattern(None, None, None);
        found.sort();
        assert!(found.contains(&id1));
        assert!(found.contains(&id2));
    }

    #[test]
    fn test_find_pattern_by_predicate() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("s1", "rdf:type", "Person"));
        let _id2 = g.intern(qt("s2", "other", "val"));
        let found = g.find_pattern(None, Some("rdf:type"), None);
        assert_eq!(found, vec![id1]);
    }

    #[test]
    fn test_find_pattern_full_triple() {
        let mut g = QuotedGraph::new();
        let id1 = g.intern(qt("s", "p", "o"));
        let _id2 = g.intern(qt("s", "p", "o2"));
        let found = g.find_pattern(Some("s"), Some("p"), Some("o"));
        assert_eq!(found, vec![id1]);
    }

    #[test]
    fn test_find_pattern_no_match() {
        let mut g = QuotedGraph::new();
        g.intern(qt("s", "p", "o"));
        let found = g.find_pattern(Some("nobody"), None, None);
        assert!(found.is_empty());
    }

    // ── Large graph ───────────────────────────────────────────────────────────

    #[test]
    fn test_large_insert() {
        let mut g = QuotedGraph::new();
        for i in 0..100 {
            let id = g.intern(qt(&format!("subj_{i}"), "rdf:type", "Entity"));
            g.add_assertion(id, "confidence", format!("{}", 0.5 + i as f64 * 0.005));
        }
        let s = g.stats();
        assert_eq!(s.unique_quoted, 100);
        assert_eq!(s.total_assertions, 100);

        // All share the same predicate
        let by_pred = g.find_by_predicate("rdf:type");
        assert_eq!(by_pred.len(), 100);
    }
}
