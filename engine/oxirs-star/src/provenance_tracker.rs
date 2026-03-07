//! RDF-star provenance tracking.
//!
//! Stores provenance records for RDF triples with metadata about who asserted them,
//! when, with what confidence, and in which named graph. Supports RDF-star notation
//! for export and N-Quads export.

use std::collections::HashMap;
use std::fmt;

/// A provenance record for a single asserted triple.
#[derive(Debug, Clone)]
pub struct ProvenanceRecord {
    /// Subject IRI of the asserted triple.
    pub subject: String,
    /// Predicate IRI of the asserted triple.
    pub predicate: String,
    /// Object IRI or literal of the asserted triple.
    pub object: String,
    /// Named graph this triple belongs to.
    pub named_graph: String,
    /// Entity (IRI or name) that asserted the triple.
    pub asserted_by: String,
    /// Unix timestamp in milliseconds when the triple was asserted.
    pub asserted_at_ms: u64,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f64,
    /// Arbitrary key-value annotations (e.g., source, license, method).
    pub annotations: HashMap<String, String>,
}

impl ProvenanceRecord {
    /// Create a new provenance record.
    pub fn new(
        subject: &str,
        predicate: &str,
        object: &str,
        named_graph: &str,
        asserted_by: &str,
        asserted_at_ms: u64,
        confidence: f64,
    ) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            named_graph: named_graph.to_string(),
            asserted_by: asserted_by.to_string(),
            asserted_at_ms,
            confidence,
            annotations: HashMap::new(),
        }
    }
}

/// Error type for provenance store operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ProvError {
    /// The requested index was out of bounds.
    IndexOutOfBounds(usize),
}

impl fmt::Display for ProvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvError::IndexOutOfBounds(i) => {
                write!(f, "Provenance record index {i} is out of bounds")
            }
        }
    }
}

impl std::error::Error for ProvError {}

/// A store of provenance records indexed sequentially.
#[derive(Debug, Default)]
pub struct ProvenanceStore {
    records: Vec<ProvenanceRecord>,
}

impl ProvenanceStore {
    /// Create an empty provenance store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pre-built provenance record.
    pub fn add(&mut self, record: ProvenanceRecord) {
        self.records.push(record);
    }

    /// Assert a triple with provenance metadata and empty annotations.
    ///
    /// Returns the index of the newly inserted record.
    pub fn assert_triple(
        &mut self,
        s: &str,
        p: &str,
        o: &str,
        graph: &str,
        asserted_by: &str,
        confidence: f64,
    ) -> usize {
        let idx = self.records.len();
        self.records.push(ProvenanceRecord::new(
            s,
            p,
            o,
            graph,
            asserted_by,
            0,
            confidence,
        ));
        idx
    }

    /// Add an annotation to the record at `index`.
    pub fn annotate(&mut self, index: usize, key: &str, value: &str) -> Result<(), ProvError> {
        let record = self
            .records
            .get_mut(index)
            .ok_or(ProvError::IndexOutOfBounds(index))?;
        record
            .annotations
            .insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Get the record at `index`.
    pub fn get(&self, index: usize) -> Option<&ProvenanceRecord> {
        self.records.get(index)
    }

    /// Find all records for a given (subject, predicate, object) triple, regardless of graph.
    pub fn find_by_triple(&self, s: &str, p: &str, o: &str) -> Vec<&ProvenanceRecord> {
        self.records
            .iter()
            .filter(|r| r.subject == s && r.predicate == p && r.object == o)
            .collect()
    }

    /// Find all records in the given named graph.
    pub fn find_by_graph(&self, graph: &str) -> Vec<&ProvenanceRecord> {
        self.records
            .iter()
            .filter(|r| r.named_graph == graph)
            .collect()
    }

    /// Find all records asserted by the given entity.
    pub fn find_by_author(&self, asserted_by: &str) -> Vec<&ProvenanceRecord> {
        self.records
            .iter()
            .filter(|r| r.asserted_by == asserted_by)
            .collect()
    }

    /// Return all records with confidence >= `threshold`.
    pub fn high_confidence(&self, threshold: f64) -> Vec<&ProvenanceRecord> {
        self.records
            .iter()
            .filter(|r| r.confidence >= threshold)
            .collect()
    }

    /// Serialize a record to RDF-star Turtle notation.
    ///
    /// Format:
    /// `<< <s> <p> <o> >> <asserted_by_pred> <asserted_by_val> .`
    pub fn to_rdf_star_notation(record: &ProvenanceRecord) -> String {
        format!(
            "<< <{s}> <{p}> <{o}> >> <http://oxirs.io/prov#assertedBy> <{by}> .\n\
             << <{s}> <{p}> <{o}> >> <http://oxirs.io/prov#assertedAt> \"{ts}\"^^xsd:long .",
            s = record.subject,
            p = record.predicate,
            o = record.object,
            by = record.asserted_by,
            ts = record.asserted_at_ms,
        )
    }

    /// Export all triples in the named graph as N-Quads strings.
    pub fn export_named_graph(&self, graph: &str) -> Vec<String> {
        self.find_by_graph(graph)
            .iter()
            .map(|r| {
                format!(
                    "<{}> <{}> <{}> <{}> .",
                    r.subject, r.predicate, r.object, r.named_graph
                )
            })
            .collect()
    }

    /// Merge all records from `other` into this store.
    pub fn merge(&mut self, other: ProvenanceStore) {
        self.records.extend(other.records);
    }

    /// Return total number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Return true if the store has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_store() -> ProvenanceStore {
        let mut store = ProvenanceStore::new();
        store.assert_triple(
            "http://example.org/Alice",
            "http://example.org/knows",
            "http://example.org/Bob",
            "http://example.org/graph1",
            "http://example.org/agent1",
            0.9,
        );
        store.assert_triple(
            "http://example.org/Bob",
            "http://example.org/knows",
            "http://example.org/Carol",
            "http://example.org/graph2",
            "http://example.org/agent2",
            0.5,
        );
        store
    }

    // ------ ProvenanceRecord ------

    #[test]
    fn test_record_new() {
        let r = ProvenanceRecord::new("s", "p", "o", "g", "agent", 1000, 0.8);
        assert_eq!(r.subject, "s");
        assert_eq!(r.predicate, "p");
        assert_eq!(r.object, "o");
        assert_eq!(r.named_graph, "g");
        assert_eq!(r.asserted_by, "agent");
        assert_eq!(r.asserted_at_ms, 1000);
        assert!((r.confidence - 0.8).abs() < 1e-12);
        assert!(r.annotations.is_empty());
    }

    // ------ ProvenanceStore::add ------

    #[test]
    fn test_store_add() {
        let mut store = ProvenanceStore::new();
        let r = ProvenanceRecord::new("s", "p", "o", "g", "a", 0, 1.0);
        store.add(r);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_store_empty() {
        let store = ProvenanceStore::new();
        assert!(store.is_empty());
    }

    // ------ assert_triple ------

    #[test]
    fn test_assert_triple_returns_index() {
        let mut store = ProvenanceStore::new();
        let i0 = store.assert_triple("s0", "p", "o", "g", "a", 1.0);
        let i1 = store.assert_triple("s1", "p", "o", "g", "a", 1.0);
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
    }

    #[test]
    fn test_assert_triple_stores_values() {
        let mut store = ProvenanceStore::new();
        let idx = store.assert_triple("s", "p", "o", "g", "agent", 0.7);
        let r = store.get(idx).expect("record");
        assert_eq!(r.subject, "s");
        assert_eq!(r.predicate, "p");
        assert_eq!(r.object, "o");
        assert_eq!(r.named_graph, "g");
        assert_eq!(r.asserted_by, "agent");
        assert!((r.confidence - 0.7).abs() < 1e-12);
    }

    // ------ annotate ------

    #[test]
    fn test_annotate_success() {
        let mut store = simple_store();
        store.annotate(0, "source", "wikipedia").expect("annotate");
        let r = store.get(0).expect("record");
        assert_eq!(r.annotations.get("source"), Some(&"wikipedia".to_string()));
    }

    #[test]
    fn test_annotate_out_of_bounds() {
        let mut store = simple_store();
        let e = store.annotate(999, "k", "v").expect_err("out of bounds");
        assert_eq!(e, ProvError::IndexOutOfBounds(999));
    }

    #[test]
    fn test_annotate_multiple_keys() {
        let mut store = simple_store();
        store.annotate(0, "source", "web").expect("ok");
        store.annotate(0, "license", "CC0").expect("ok");
        let r = store.get(0).expect("record");
        assert_eq!(r.annotations.len(), 2);
    }

    // ------ get ------

    #[test]
    fn test_get_valid_index() {
        let store = simple_store();
        assert!(store.get(0).is_some());
        assert!(store.get(1).is_some());
    }

    #[test]
    fn test_get_out_of_bounds() {
        let store = simple_store();
        assert!(store.get(99).is_none());
    }

    // ------ find_by_triple ------

    #[test]
    fn test_find_by_triple_found() {
        let store = simple_store();
        let results = store.find_by_triple(
            "http://example.org/Alice",
            "http://example.org/knows",
            "http://example.org/Bob",
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_by_triple_not_found() {
        let store = simple_store();
        let results = store.find_by_triple("x", "y", "z");
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_by_triple_multiple_graphs() {
        let mut store = ProvenanceStore::new();
        store.assert_triple("s", "p", "o", "g1", "a", 0.9);
        store.assert_triple("s", "p", "o", "g2", "b", 0.8);
        let results = store.find_by_triple("s", "p", "o");
        assert_eq!(results.len(), 2);
    }

    // ------ find_by_graph ------

    #[test]
    fn test_find_by_graph() {
        let store = simple_store();
        let results = store.find_by_graph("http://example.org/graph1");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "http://example.org/Alice");
    }

    #[test]
    fn test_find_by_graph_empty() {
        let store = simple_store();
        assert!(store.find_by_graph("nonexistent").is_empty());
    }

    // ------ find_by_author ------

    #[test]
    fn test_find_by_author() {
        let store = simple_store();
        let results = store.find_by_author("http://example.org/agent1");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_by_author_empty() {
        let store = simple_store();
        assert!(store.find_by_author("nobody").is_empty());
    }

    // ------ high_confidence ------

    #[test]
    fn test_high_confidence_threshold() {
        let store = simple_store();
        let high = store.high_confidence(0.8);
        assert_eq!(high.len(), 1);
        assert!((high[0].confidence - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_high_confidence_all() {
        let store = simple_store();
        let all = store.high_confidence(0.0);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_high_confidence_none() {
        let store = simple_store();
        let none = store.high_confidence(1.1);
        assert!(none.is_empty());
    }

    // ------ to_rdf_star_notation ------

    #[test]
    fn test_rdf_star_notation_format() {
        let r = ProvenanceRecord::new(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
            "http://example.org/g",
            "http://example.org/agent",
            1000,
            0.9,
        );
        let notation = ProvenanceStore::to_rdf_star_notation(&r);
        assert!(notation.contains("<<"));
        assert!(notation.contains(">>"));
        assert!(notation.contains("http://example.org/s"));
        assert!(notation.contains("xsd:long"));
    }

    #[test]
    fn test_rdf_star_notation_contains_timestamp() {
        let mut r = ProvenanceRecord::new("s", "p", "o", "g", "a", 0, 1.0);
        r.asserted_at_ms = 1_700_000_000_000;
        let notation = ProvenanceStore::to_rdf_star_notation(&r);
        assert!(notation.contains("1700000000000"));
    }

    // ------ export_named_graph ------

    #[test]
    fn test_export_named_graph_nquads() {
        let store = simple_store();
        let nquads = store.export_named_graph("http://example.org/graph1");
        assert_eq!(nquads.len(), 1);
        assert!(nquads[0].contains("<http://example.org/Alice>"));
        assert!(nquads[0].ends_with('.'));
    }

    #[test]
    fn test_export_named_graph_empty() {
        let store = simple_store();
        let nquads = store.export_named_graph("http://example.org/graph_not_exist");
        assert!(nquads.is_empty());
    }

    // ------ merge ------

    #[test]
    fn test_merge() {
        let mut store1 = simple_store();
        let store2 = simple_store();
        let count_before = store1.len();
        store1.merge(store2);
        assert_eq!(store1.len(), count_before * 2);
    }

    #[test]
    fn test_merge_empty() {
        let mut store = simple_store();
        let count = store.len();
        store.merge(ProvenanceStore::new());
        assert_eq!(store.len(), count);
    }

    // ------ ProvError ------

    #[test]
    fn test_prov_error_display() {
        let e = ProvError::IndexOutOfBounds(42);
        let s = e.to_string();
        assert!(s.contains("42"));
    }

    #[test]
    fn test_prov_error_is_error() {
        let e: Box<dyn std::error::Error> = Box::new(ProvError::IndexOutOfBounds(0));
        assert!(!e.to_string().is_empty());
    }
}
