/// In-memory store for RDF-star quoted triples.
///
/// RDF-star allows triples themselves to be used as subjects or objects of
/// other triples ("annotation" triples).  This module provides a simple
/// `QuotedTripleStore` that manages such quoted triples and their annotations.
use std::collections::HashMap;

// ── Data structures ───────────────────────────────────────────────────────────

/// A plain RDF triple (subject, predicate, object — all strings / IRIs).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub s: String,
    pub p: String,
    pub o: String,
}

impl Triple {
    /// Create a new triple from anything that can be converted to `String`.
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }

    /// Return the canonical map key for this triple.
    pub fn key(&self) -> (String, String, String) {
        (self.s.clone(), self.p.clone(), self.o.clone())
    }
}

/// An RDF-star term — either a resource, literal, blank node, or a quoted triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfStarTerm {
    Iri(String),
    Literal(String),
    BlankNode(String),
    QuotedTriple(Box<Triple>),
}

impl RdfStarTerm {
    /// Returns the quoted triple if this term is `QuotedTriple`, otherwise `None`.
    pub fn as_quoted_triple(&self) -> Option<&Triple> {
        match self {
            Self::QuotedTriple(t) => Some(t),
            _ => None,
        }
    }
}

/// A triple whose subject and/or object may themselves be quoted triples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuotedTriple {
    pub s: RdfStarTerm,
    pub p: String,
    pub o: RdfStarTerm,
}

/// An annotation triple: `<< s p o >> ann_p ann_o`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnotationTriple {
    /// The quoted (asserted) triple being annotated.
    pub quoted: Triple,
    /// Annotation predicate.
    pub p: String,
    /// Annotation object.
    pub o: RdfStarTerm,
}

// ── QuotedTripleStore ─────────────────────────────────────────────────────────

/// In-memory RDF-star store.
#[derive(Debug, Default)]
pub struct QuotedTripleStore {
    triples: Vec<QuotedTriple>,
    /// `(s, p, o)` key → list of annotation triples.
    annotations: HashMap<(String, String, String), Vec<AnnotationTriple>>,
}

impl QuotedTripleStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a `QuotedTriple` to the store.
    pub fn add_triple(&mut self, triple: QuotedTriple) {
        self.triples.push(triple);
    }

    /// Add an annotation triple to the quoted `triple`.
    pub fn add_annotation(&mut self, quoted: &Triple, p: &str, o: RdfStarTerm) {
        let annotation = AnnotationTriple {
            quoted: quoted.clone(),
            p: p.to_string(),
            o,
        };
        self.annotations
            .entry(quoted.key())
            .or_default()
            .push(annotation);
    }

    /// Get all annotations for the given quoted triple.
    pub fn get_annotations(&self, quoted: &Triple) -> Vec<&AnnotationTriple> {
        self.annotations
            .get(&quoted.key())
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// All `QuotedTriple`s whose subject is itself a quoted triple.
    pub fn triples_with_quoted_subject(&self) -> Vec<&QuotedTriple> {
        self.triples
            .iter()
            .filter(|qt| matches!(qt.s, RdfStarTerm::QuotedTriple(_)))
            .collect()
    }

    /// All `QuotedTriple`s whose object is itself a quoted triple.
    pub fn triples_with_quoted_object(&self) -> Vec<&QuotedTriple> {
        self.triples
            .iter()
            .filter(|qt| matches!(qt.o, RdfStarTerm::QuotedTriple(_)))
            .collect()
    }

    /// Find all `QuotedTriple`s with the given predicate.
    pub fn find_by_predicate(&self, p: &str) -> Vec<&QuotedTriple> {
        self.triples.iter().filter(|qt| qt.p == p).collect()
    }

    /// Total number of `QuotedTriple`s in the store.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Total number of annotation triples across all quoted triples.
    pub fn annotation_count(&self) -> usize {
        self.annotations.values().map(|v| v.len()).sum()
    }

    /// Remove all annotations for the given quoted triple.
    pub fn clear_annotations(&mut self, quoted: &Triple) {
        self.annotations.remove(&quoted.key());
    }

    /// Collect all `Triple` objects that appear as the subject or object of any
    /// stored `QuotedTriple`.
    pub fn all_quoted_triples(&self) -> Vec<&Triple> {
        let mut result: Vec<&Triple> = Vec::new();
        for qt in &self.triples {
            if let RdfStarTerm::QuotedTriple(t) = &qt.s {
                result.push(t);
            }
            if let RdfStarTerm::QuotedTriple(t) = &qt.o {
                result.push(t);
            }
        }
        result
    }

    /// Serialize a `QuotedTriple` to the RDF-star `<< s p o >>` notation.
    pub fn to_rdf_star_string(triple: &QuotedTriple) -> String {
        let s = term_to_string(&triple.s);
        let o = term_to_string(&triple.o);
        format!("{} {} {}", s, triple.p, o)
    }
}

// ── Private helper ────────────────────────────────────────────────────────────

fn term_to_string(term: &RdfStarTerm) -> String {
    match term {
        RdfStarTerm::Iri(iri) => format!("<{iri}>"),
        RdfStarTerm::Literal(lit) => format!("\"{lit}\""),
        RdfStarTerm::BlankNode(bn) => format!("_:{bn}"),
        RdfStarTerm::QuotedTriple(t) => {
            format!("<< {} {} {} >>", t.s, t.p, t.o)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn iri(s: &str) -> RdfStarTerm {
        RdfStarTerm::Iri(s.into())
    }

    fn lit(s: &str) -> RdfStarTerm {
        RdfStarTerm::Literal(s.into())
    }

    fn bnode(s: &str) -> RdfStarTerm {
        RdfStarTerm::BlankNode(s.into())
    }

    fn quoted(s: &str, p: &str, o: &str) -> RdfStarTerm {
        RdfStarTerm::QuotedTriple(Box::new(Triple::new(s, p, o)))
    }

    fn qt(s: RdfStarTerm, p: &str, o: RdfStarTerm) -> QuotedTriple {
        QuotedTriple { s, p: p.into(), o }
    }

    fn triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    // ── Triple ────────────────────────────────────────────────────────────────

    #[test]
    fn test_triple_new() {
        let t = triple("s", "p", "o");
        assert_eq!(t.s, "s");
        assert_eq!(t.p, "p");
        assert_eq!(t.o, "o");
    }

    #[test]
    fn test_triple_key() {
        let t = triple("a", "b", "c");
        assert_eq!(t.key(), ("a".into(), "b".into(), "c".into()));
    }

    // ── add_triple / triple_count ─────────────────────────────────────────────

    #[test]
    fn test_add_triple_increments_count() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(iri("s"), "p", iri("o")));
        assert_eq!(store.triple_count(), 1);
    }

    #[test]
    fn test_add_multiple_triples() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(iri("s1"), "p", iri("o1")));
        store.add_triple(qt(iri("s2"), "p", iri("o2")));
        assert_eq!(store.triple_count(), 2);
    }

    #[test]
    fn test_empty_store_counts() {
        let store = QuotedTripleStore::new();
        assert_eq!(store.triple_count(), 0);
        assert_eq!(store.annotation_count(), 0);
    }

    // ── add_annotation / get_annotations ─────────────────────────────────────

    #[test]
    fn test_add_and_get_annotation() {
        let mut store = QuotedTripleStore::new();
        let q = triple("alice", "knows", "bob");
        store.add_annotation(&q, "certainty", lit("0.9"));
        let anns = store.get_annotations(&q);
        assert_eq!(anns.len(), 1);
        assert_eq!(anns[0].p, "certainty");
    }

    #[test]
    fn test_get_annotations_empty() {
        let store = QuotedTripleStore::new();
        let q = triple("x", "y", "z");
        assert!(store.get_annotations(&q).is_empty());
    }

    #[test]
    fn test_multiple_annotations_same_triple() {
        let mut store = QuotedTripleStore::new();
        let q = triple("a", "b", "c");
        store.add_annotation(&q, "p1", lit("v1"));
        store.add_annotation(&q, "p2", lit("v2"));
        let anns = store.get_annotations(&q);
        assert_eq!(anns.len(), 2);
    }

    #[test]
    fn test_annotation_count_across_triples() {
        let mut store = QuotedTripleStore::new();
        let q1 = triple("a", "b", "c");
        let q2 = triple("x", "y", "z");
        store.add_annotation(&q1, "p", lit("v"));
        store.add_annotation(&q2, "p", lit("w"));
        assert_eq!(store.annotation_count(), 2);
    }

    // ── clear_annotations ─────────────────────────────────────────────────────

    #[test]
    fn test_clear_annotations() {
        let mut store = QuotedTripleStore::new();
        let q = triple("a", "b", "c");
        store.add_annotation(&q, "p", lit("v"));
        store.clear_annotations(&q);
        assert!(store.get_annotations(&q).is_empty());
        assert_eq!(store.annotation_count(), 0);
    }

    #[test]
    fn test_clear_annotations_only_affects_target() {
        let mut store = QuotedTripleStore::new();
        let q1 = triple("a", "b", "c");
        let q2 = triple("x", "y", "z");
        store.add_annotation(&q1, "p", lit("v1"));
        store.add_annotation(&q2, "p", lit("v2"));
        store.clear_annotations(&q1);
        assert_eq!(store.annotation_count(), 1);
        assert!(!store.get_annotations(&q2).is_empty());
    }

    // ── triples_with_quoted_subject / object ───────────────────────────────────

    #[test]
    fn test_triples_with_quoted_subject() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(quoted("a", "b", "c"), "p", iri("o")));
        store.add_triple(qt(iri("s"), "p", iri("o")));
        let result = store.triples_with_quoted_subject();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_triples_with_quoted_object() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(iri("s"), "p", quoted("a", "b", "c")));
        store.add_triple(qt(iri("s"), "p", iri("o")));
        let result = store.triples_with_quoted_object();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_no_quoted_subject_or_object() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(iri("s"), "p", lit("v")));
        assert!(store.triples_with_quoted_subject().is_empty());
        assert!(store.triples_with_quoted_object().is_empty());
    }

    // ── find_by_predicate ─────────────────────────────────────────────────────

    #[test]
    fn test_find_by_predicate() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(iri("s"), "knows", iri("o")));
        store.add_triple(qt(iri("s"), "likes", iri("o")));
        let result = store.find_by_predicate("knows");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].p, "knows");
    }

    #[test]
    fn test_find_by_predicate_no_match() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(iri("s"), "p", iri("o")));
        assert!(store.find_by_predicate("missing").is_empty());
    }

    #[test]
    fn test_find_by_predicate_multiple() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(iri("s1"), "p", iri("o1")));
        store.add_triple(qt(iri("s2"), "p", iri("o2")));
        let result = store.find_by_predicate("p");
        assert_eq!(result.len(), 2);
    }

    // ── all_quoted_triples ────────────────────────────────────────────────────

    #[test]
    fn test_all_quoted_triples_from_subject() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(quoted("a", "b", "c"), "p", iri("o")));
        let all = store.all_quoted_triples();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].s, "a");
    }

    #[test]
    fn test_all_quoted_triples_from_object() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(iri("s"), "p", quoted("x", "y", "z")));
        let all = store.all_quoted_triples();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].s, "x");
    }

    #[test]
    fn test_all_quoted_triples_empty_when_no_nested() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(iri("s"), "p", iri("o")));
        assert!(store.all_quoted_triples().is_empty());
    }

    // ── to_rdf_star_string ────────────────────────────────────────────────────

    #[test]
    fn test_to_rdf_star_string_iri_terms() {
        let t = qt(iri("http://ex/s"), "http://ex/p", iri("http://ex/o"));
        let s = QuotedTripleStore::to_rdf_star_string(&t);
        assert!(s.contains("<http://ex/s>"));
        assert!(s.contains("http://ex/p"));
        assert!(s.contains("<http://ex/o>"));
    }

    #[test]
    fn test_to_rdf_star_string_literal_object() {
        let t = qt(iri("s"), "p", lit("42"));
        let s = QuotedTripleStore::to_rdf_star_string(&t);
        assert!(s.contains("\"42\""));
    }

    #[test]
    fn test_to_rdf_star_string_bnode() {
        let t = qt(bnode("b1"), "p", iri("o"));
        let s = QuotedTripleStore::to_rdf_star_string(&t);
        assert!(s.contains("_:b1"));
    }

    #[test]
    fn test_to_rdf_star_string_nested_quoted() {
        let t = qt(quoted("a", "b", "c"), "p", iri("o"));
        let s = QuotedTripleStore::to_rdf_star_string(&t);
        assert!(s.contains("<<"));
        assert!(s.contains(">>"));
    }

    // ── RdfStarTerm helpers ───────────────────────────────────────────────────

    #[test]
    fn test_rdf_star_term_as_quoted_triple_some() {
        let term = quoted("a", "b", "c");
        assert!(term.as_quoted_triple().is_some());
    }

    #[test]
    fn test_rdf_star_term_as_quoted_triple_none_for_iri() {
        let term = iri("x");
        assert!(term.as_quoted_triple().is_none());
    }

    // ── Nested quoted triple ──────────────────────────────────────────────────

    #[test]
    fn test_nested_quoted_triple_in_subject() {
        let inner = Triple::new("alice", "knows", "bob");
        let nested = RdfStarTerm::QuotedTriple(Box::new(inner.clone()));
        let t = QuotedTriple {
            s: nested,
            p: "believedBy".into(),
            o: iri("carol"),
        };
        let mut store = QuotedTripleStore::new();
        store.add_triple(t);
        let qs = store.triples_with_quoted_subject();
        assert_eq!(qs.len(), 1);
        let inner_ref = qs[0].s.as_quoted_triple().unwrap();
        assert_eq!(inner_ref.s, "alice");
    }

    // ── Additional tests for round 12 (reaching ≥45 total) ───────────────────

    #[test]
    fn test_triple_clone_and_eq() {
        let t = triple("s", "p", "o");
        assert_eq!(t, t.clone());
    }

    #[test]
    fn test_triple_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(triple("a", "b", "c"));
        set.insert(triple("a", "b", "c")); // duplicate
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_rdf_star_term_iri_variant() {
        let t = iri("http://example.org/foo");
        match t {
            RdfStarTerm::Iri(v) => assert_eq!(v, "http://example.org/foo"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_rdf_star_term_literal_variant() {
        let t = lit("hello");
        match t {
            RdfStarTerm::Literal(v) => assert_eq!(v, "hello"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_rdf_star_term_blank_node_variant() {
        let t = bnode("b42");
        match t {
            RdfStarTerm::BlankNode(v) => assert_eq!(v, "b42"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_rdf_star_term_quoted_triple_as_quoted() {
        let t = quoted("s", "p", "o");
        let inner = t.as_quoted_triple().unwrap();
        assert_eq!(inner.s, "s");
        assert_eq!(inner.p, "p");
        assert_eq!(inner.o, "o");
    }

    #[test]
    fn test_rdf_star_term_literal_as_quoted_is_none() {
        let t = lit("x");
        assert!(t.as_quoted_triple().is_none());
    }

    #[test]
    fn test_rdf_star_term_blank_node_as_quoted_is_none() {
        let t = bnode("b0");
        assert!(t.as_quoted_triple().is_none());
    }

    #[test]
    fn test_rdf_star_term_clone_eq() {
        let t = iri("http://example.org/");
        assert_eq!(t, t.clone());
    }

    #[test]
    fn test_quoted_triple_clone() {
        let t = qt(iri("s"), "p", lit("o"));
        let c = t.clone();
        assert_eq!(t.p, c.p);
    }

    #[test]
    fn test_store_default() {
        let store = QuotedTripleStore::default();
        assert_eq!(store.triple_count(), 0);
        assert_eq!(store.annotation_count(), 0);
    }

    #[test]
    fn test_annotation_with_iri_object() {
        let mut store = QuotedTripleStore::new();
        let q = triple("s", "p", "o");
        store.add_annotation(&q, "source", iri("http://example.org/src"));
        let anns = store.get_annotations(&q);
        assert_eq!(anns.len(), 1);
        match &anns[0].o {
            RdfStarTerm::Iri(v) => assert_eq!(v, "http://example.org/src"),
            _ => panic!("wrong term type"),
        }
    }

    #[test]
    fn test_annotation_with_blank_node_object() {
        let mut store = QuotedTripleStore::new();
        let q = triple("a", "b", "c");
        store.add_annotation(&q, "rel", bnode("bn1"));
        let anns = store.get_annotations(&q);
        assert_eq!(anns.len(), 1);
        match &anns[0].o {
            RdfStarTerm::BlankNode(v) => assert_eq!(v, "bn1"),
            _ => panic!("wrong term type"),
        }
    }

    #[test]
    fn test_find_by_predicate_empty_store() {
        let store = QuotedTripleStore::new();
        assert!(store.find_by_predicate("any").is_empty());
    }

    #[test]
    fn test_all_quoted_triples_both_subject_and_object() {
        let mut store = QuotedTripleStore::new();
        store.add_triple(qt(
            quoted("s1", "p1", "o1"),
            "meta",
            quoted("s2", "p2", "o2"),
        ));
        let all = store.all_quoted_triples();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_triple_count_after_multiple_adds() {
        let mut store = QuotedTripleStore::new();
        for i in 0..10 {
            store.add_triple(qt(iri(&format!("s{i}")), "p", iri(&format!("o{i}"))));
        }
        assert_eq!(store.triple_count(), 10);
    }

    #[test]
    fn test_annotation_triple_fields() {
        let ann = AnnotationTriple {
            quoted: triple("a", "b", "c"),
            p: "certainty".to_string(),
            o: lit("0.8"),
        };
        assert_eq!(ann.p, "certainty");
        assert_eq!(ann.quoted.s, "a");
    }

    #[test]
    fn test_to_rdf_star_string_with_literal_subject() {
        // A QuotedTriple with a Literal subject (unusual but valid in code)
        let t = qt(lit("42"), "ex:value", iri("ex:Thing"));
        let s = QuotedTripleStore::to_rdf_star_string(&t);
        assert!(s.contains("\"42\""));
    }

    #[test]
    fn test_clear_annotations_noop_if_not_set() {
        let mut store = QuotedTripleStore::new();
        let q = triple("x", "y", "z");
        // Should not panic even if no annotations exist
        store.clear_annotations(&q);
        assert_eq!(store.annotation_count(), 0);
    }

    #[test]
    fn test_multiple_predicates_find_each() {
        let mut store = QuotedTripleStore::new();
        let preds = ["rdf:type", "rdfs:label", "owl:sameAs"];
        for p in &preds {
            store.add_triple(qt(iri("s"), p, iri("o")));
        }
        for p in &preds {
            assert_eq!(store.find_by_predicate(p).len(), 1);
        }
    }
}
