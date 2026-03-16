//! High-level public API for WASM consumers
//!
//! This module provides clean, ergonomic Rust types that can be bound to
//! JavaScript with `wasm_bindgen`, or used directly from Rust tests.
//!
//! Design principles:
//! - All public methods use plain Rust types (no `JsValue`) so they can be
//!   unit-tested natively without a WASM runtime
//! - Error results are `Result<_, String>` for easy JS conversion
//! - Streaming loading is supported via `feed_chunk` / `finish_loading`

use crate::parser::{RdfFormat, StreamingParser};
use crate::store::{CompactTripleStore, RdfTerm};

// -----------------------------------------------------------------------
// Conversion helpers
// -----------------------------------------------------------------------

/// Convert a serialized N-Triples-style string into an [`RdfTerm`].
///
/// Accepts:
/// - `<iri>` -> `RdfTerm::Iri`
/// - `_:id` -> `RdfTerm::BlankNode`
/// - `"value"@lang` -> `RdfTerm::LangLiteral`
/// - `"value"^^<dt>` -> `RdfTerm::TypedLiteral`
/// - `"value"` → `RdfTerm::PlainLiteral`
/// - anything else → `RdfTerm::Iri` (treated as bare IRI)
fn str_to_rdf_term(s: &str) -> RdfTerm {
    let s = s.trim();
    if s.starts_with('<') && s.ends_with('>') {
        RdfTerm::iri(&s[1..s.len() - 1])
    } else if let Some(id) = s.strip_prefix("_:") {
        RdfTerm::blank(id)
    } else if s.starts_with('"') {
        // Find closing quote (accounting for escapes)
        let chars: Vec<char> = s.chars().collect();
        let mut pos = 1usize;
        let mut escaped = false;
        while pos < chars.len() {
            if escaped {
                escaped = false;
                pos += 1;
                continue;
            }
            if chars[pos] == '\\' {
                escaped = true;
                pos += 1;
                continue;
            }
            if chars[pos] == '"' {
                break;
            }
            pos += 1;
        }
        // Value is chars[1..pos]
        let value: String = chars[1..pos].iter().collect();
        let rest: String = if pos + 1 < chars.len() {
            chars[pos + 1..].iter().collect()
        } else {
            String::new()
        };

        if let Some(lang) = rest.strip_prefix('@') {
            RdfTerm::lang_literal(value, lang)
        } else if let Some(dt_raw) = rest.strip_prefix("^^") {
            let datatype = if dt_raw.starts_with('<') && dt_raw.ends_with('>') {
                dt_raw[1..dt_raw.len() - 1].to_string()
            } else {
                dt_raw.to_string()
            };
            RdfTerm::typed_literal(value, datatype)
        } else {
            RdfTerm::literal(value)
        }
    } else {
        // Bare string – treat as IRI
        RdfTerm::iri(s)
    }
}

/// Convert an [`RdfTerm`] to its N-Triples string representation
fn rdf_term_to_str(term: &RdfTerm) -> String {
    term.to_string()
}

// -----------------------------------------------------------------------
// WasmSparqlStore
// -----------------------------------------------------------------------

/// A SPARQL-capable RDF store suitable for WASM binding.
///
/// Combines a [`CompactTripleStore`] (memory-efficient storage) with a
/// [`StreamingParser`] (incremental RDF parsing) into a single convenient API.
///
/// # Usage (Rust)
/// ```no_run
/// # use oxirs_wasm::api::wasm_api::WasmSparqlStore;
/// let mut store = WasmSparqlStore::new();
/// let count = store.load_ntriples("<http://s> <http://p> <http://o> .\n").expect("should succeed");
/// assert_eq!(count, 1);
///
/// let results = store.query_pattern(Some("<http://s>"), None, None);
/// assert_eq!(results.len(), 1);
/// ```
pub struct WasmSparqlStore {
    /// Compact triple store (dictionary + sorted indexes)
    store: CompactTripleStore,
    /// Streaming parser state (for incremental chunk loading)
    streaming_parser: Option<StreamingParser>,
    /// Format for the active streaming session
    streaming_format: RdfFormat,
    /// Total triples loaded via streaming in the current session
    streaming_count: usize,
}

impl WasmSparqlStore {
    /// Create a new empty store
    pub fn new() -> Self {
        Self {
            store: CompactTripleStore::new(),
            streaming_parser: None,
            streaming_format: RdfFormat::NTriples,
            streaming_count: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Bulk loading
    // -----------------------------------------------------------------------

    /// Load a complete Turtle document.
    ///
    /// Returns the number of triples inserted.
    pub fn load_turtle(&mut self, data: &str) -> Result<usize, String> {
        self.load_format(data, RdfFormat::Turtle)
    }

    /// Load a complete N-Triples document.
    ///
    /// Returns the number of triples inserted.
    pub fn load_ntriples(&mut self, data: &str) -> Result<usize, String> {
        self.load_format(data, RdfFormat::NTriples)
    }

    /// Load a complete N-Quads document (graph names are ignored for storage).
    ///
    /// Returns the number of triples inserted.
    pub fn load_nquads(&mut self, data: &str) -> Result<usize, String> {
        self.load_format(data, RdfFormat::NQuads)
    }

    fn load_format(&mut self, data: &str, format: RdfFormat) -> Result<usize, String> {
        let mut parser = StreamingParser::new(format);
        let mut stmts = parser
            .feed(data)
            .map_err(|e| format!("Parse error: {}", e))?;
        let mut tail = parser
            .finish()
            .map_err(|e| format!("Parse error during finalization: {}", e))?;
        stmts.append(&mut tail);

        let before = self.store.triple_count();
        self.store.bulk_insert(stmts.iter().map(|s| {
            let subj = s.subject().to_ntriples_string();
            let pred = s.predicate().to_ntriples_string();
            let obj = s.object().to_ntriples_string();
            (
                str_to_rdf_term(&subj),
                str_to_rdf_term(&pred),
                str_to_rdf_term(&obj),
            )
        }));
        let inserted = self.store.triple_count() - before;
        Ok(inserted)
    }

    // -----------------------------------------------------------------------
    // Streaming loading (chunk-by-chunk)
    // -----------------------------------------------------------------------

    /// Begin a streaming load session in the given format.
    ///
    /// Must be called before [`feed_chunk`](Self::feed_chunk). Any previously
    /// active streaming session is discarded.
    pub fn begin_streaming(&mut self, format: RdfFormat) {
        self.streaming_parser = Some(StreamingParser::new(format));
        self.streaming_format = format;
        self.streaming_count = 0;
    }

    /// Feed a chunk of RDF data in the current streaming session.
    ///
    /// Returns the number of triples parsed from complete statements in this
    /// chunk. Incomplete statements are buffered for subsequent chunks.
    ///
    /// # Errors
    /// Returns an error string if the chunk contains a syntax error.
    pub fn feed_chunk(&mut self, chunk: &str) -> Result<usize, String> {
        let parser = self.streaming_parser.as_mut().ok_or_else(|| {
            "No active streaming session. Call begin_streaming() first.".to_string()
        })?;

        let stmts = parser
            .feed(chunk)
            .map_err(|e| format!("Parse error in chunk: {}", e))?;

        let count = stmts.len();
        self.streaming_count += count;
        self.store.bulk_insert(stmts.iter().map(|s| {
            let subj = s.subject().to_ntriples_string();
            let pred = s.predicate().to_ntriples_string();
            let obj = s.object().to_ntriples_string();
            (
                str_to_rdf_term(&subj),
                str_to_rdf_term(&pred),
                str_to_rdf_term(&obj),
            )
        }));

        Ok(count)
    }

    /// Finish the current streaming session, flushing any remaining buffered data.
    ///
    /// Returns the total number of triples inserted during this session.
    pub fn finish_loading(&mut self) -> Result<usize, String> {
        let parser = self.streaming_parser.as_mut().ok_or_else(|| {
            "No active streaming session. Call begin_streaming() first.".to_string()
        })?;

        let stmts = parser
            .finish()
            .map_err(|e| format!("Parse error in final flush: {}", e))?;

        let count = stmts.len();
        self.streaming_count += count;
        self.store.bulk_insert(stmts.iter().map(|s| {
            let subj = s.subject().to_ntriples_string();
            let pred = s.predicate().to_ntriples_string();
            let obj = s.object().to_ntriples_string();
            (
                str_to_rdf_term(&subj),
                str_to_rdf_term(&pred),
                str_to_rdf_term(&obj),
            )
        }));

        let total = self.streaming_count;
        self.streaming_parser = None;
        self.streaming_count = 0;

        Ok(total)
    }

    // -----------------------------------------------------------------------
    // Triple management
    // -----------------------------------------------------------------------

    /// Insert a single triple.
    ///
    /// Terms should be in N-Triples serialization format:
    /// - IRIs: `<http://example.org/>`
    /// - Blank nodes: `_:b0`
    /// - Literals: `"hello"`, `"hello"@en`, `"42"^^<xsd:integer>`
    pub fn insert_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        self.store.insert(
            &str_to_rdf_term(subject),
            &str_to_rdf_term(predicate),
            &str_to_rdf_term(object),
        );
    }

    /// Delete a triple.
    ///
    /// Returns `true` if the triple existed and was deleted.
    pub fn delete_triple(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        self.store.delete(
            &str_to_rdf_term(subject),
            &str_to_rdf_term(predicate),
            &str_to_rdf_term(object),
        )
    }

    /// Check whether a triple exists
    pub fn triple_exists(&self, subject: &str, predicate: &str, object: &str) -> bool {
        self.store.contains(
            &str_to_rdf_term(subject),
            &str_to_rdf_term(predicate),
            &str_to_rdf_term(object),
        )
    }

    // -----------------------------------------------------------------------
    // Querying
    // -----------------------------------------------------------------------

    /// Execute a triple pattern query.
    ///
    /// Each optional argument is an N-Triples-serialized term acting as a filter.
    /// `None` means "match anything" (wildcard).
    ///
    /// Returns a list of matching triples, each as `[subject, predicate, object]`
    /// in N-Triples serialization format.
    ///
    /// # Example
    /// ```no_run
    /// # use oxirs_wasm::api::wasm_api::WasmSparqlStore;
    /// let mut store = WasmSparqlStore::new();
    /// store.load_ntriples("<http://s> <http://p> <http://o> .\n").expect("should succeed");
    ///
    /// // Find all triples with subject <http://s>
    /// let results = store.query_pattern(Some("<http://s>"), None, None);
    /// assert_eq!(results.len(), 1);
    ///
    /// // Find all triples
    /// let all = store.query_pattern(None, None, None);
    /// assert_eq!(all.len(), 1);
    /// ```
    pub fn query_pattern(
        &mut self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<Vec<String>> {
        match (subject, predicate, object) {
            // Subject bound, predicate and object unbound → use SPO index
            (Some(s), None, None) => {
                let s_term = str_to_rdf_term(s);
                self.store
                    .find_by_subject(&s_term)
                    .into_iter()
                    .map(|(s, p, o)| vec![s.to_string(), p.to_string(), o.to_string()])
                    .collect()
            }
            // Predicate bound, subject and object unbound → use PSO index
            (None, Some(p), None) => {
                let p_term = str_to_rdf_term(p);
                self.store
                    .find_by_predicate(&p_term)
                    .into_iter()
                    .map(|(s, p, o)| vec![s.to_string(), p.to_string(), o.to_string()])
                    .collect()
            }
            // Predicate and object bound → specialized index lookup
            (None, Some(p), Some(o)) => {
                let p_term = str_to_rdf_term(p);
                let o_term = str_to_rdf_term(o);
                self.store
                    .find_by_predicate_object(&p_term, &o_term)
                    .into_iter()
                    .map(|s| {
                        vec![
                            s.to_string(),
                            rdf_term_to_str(&str_to_rdf_term(p)),
                            rdf_term_to_str(&str_to_rdf_term(o)),
                        ]
                    })
                    .collect()
            }
            // Full wildcard or other combinations → linear scan with filters
            _ => {
                let s_filter = subject.map(str_to_rdf_term);
                let p_filter = predicate.map(str_to_rdf_term);
                let o_filter = object.map(str_to_rdf_term);

                self.store
                    .iter_all()
                    .filter(|(s, p, o)| {
                        s_filter.as_ref().map_or(true, |f| f == s)
                            && p_filter.as_ref().map_or(true, |f| f == p)
                            && o_filter.as_ref().map_or(true, |f| f == o)
                    })
                    .map(|(s, p, o)| vec![s.to_string(), p.to_string(), o.to_string()])
                    .collect()
            }
        }
    }

    /// Find all triples with the given subject
    pub fn find_by_subject(&mut self, subject: &str) -> Vec<Vec<String>> {
        self.query_pattern(Some(subject), None, None)
    }

    /// Find all triples with the given predicate
    pub fn find_by_predicate(&mut self, predicate: &str) -> Vec<Vec<String>> {
        self.query_pattern(None, Some(predicate), None)
    }

    /// Find all subjects that have a specific predicate-object pair
    pub fn find_subjects_by_predicate_object(
        &mut self,
        predicate: &str,
        object: &str,
    ) -> Vec<String> {
        let p_term = str_to_rdf_term(predicate);
        let o_term = str_to_rdf_term(object);
        self.store
            .find_by_predicate_object(&p_term, &o_term)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Export all triples as N-Triples text
    pub fn to_ntriples(&mut self) -> String {
        self.store
            .iter_all()
            .map(|(s, p, o)| format!("{} {} {} .\n", s, p, o))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return the number of triples currently stored
    pub fn triple_count(&self) -> usize {
        self.store.triple_count()
    }

    /// Return the number of unique RDF terms in the dictionary
    pub fn term_count(&self) -> usize {
        self.store.term_count()
    }

    /// Estimate memory usage in bytes (triples + dictionary)
    pub fn memory_bytes(&self) -> usize {
        self.store.memory_estimate_bytes()
    }
}

impl Default for WasmSparqlStore {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store_with_data() -> WasmSparqlStore {
        let mut store = WasmSparqlStore::new();
        store
            .load_ntriples(
                "<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .\n\
                 <http://example.org/alice> <http://example.org/name> \"Alice\" .\n\
                 <http://example.org/bob> <http://example.org/name> \"Bob\" .\n",
            )
            .expect("load_ntriples");
        store
    }

    #[test]
    fn test_load_ntriples() {
        let mut store = WasmSparqlStore::new();
        let count = store
            .load_ntriples("<http://s> <http://p> <http://o> .\n")
            .expect("load");
        assert_eq!(count, 1);
        assert_eq!(store.triple_count(), 1);
    }

    #[test]
    fn test_load_turtle() {
        let mut store = WasmSparqlStore::new();
        let ttl = "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\n";
        let count = store.load_turtle(ttl).expect("load turtle");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_streaming_ntriples() {
        let mut store = WasmSparqlStore::new();
        store.begin_streaming(RdfFormat::NTriples);

        // Feed in three chunks (split mid-line)
        let c1 = store.feed_chunk("<http://s> ").expect("chunk 1");
        assert_eq!(c1, 0); // No complete statement yet

        let c2 = store
            .feed_chunk("<http://p> <http://o> .\n")
            .expect("chunk 2");
        assert_eq!(c2, 1);

        let total = store.finish_loading().expect("finish");
        assert_eq!(total, 1);
        assert_eq!(store.triple_count(), 1);
    }

    #[test]
    fn test_streaming_without_begin_fails() {
        let mut store = WasmSparqlStore::new();
        let result = store.feed_chunk("data");
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_multiple_chunks() {
        let mut store = WasmSparqlStore::new();
        store.begin_streaming(RdfFormat::NTriples);

        // Feed 10 triples one at a time
        for i in 0..10 {
            let line = format!("<http://s{}> <http://p> <http://o{}> .\n", i, i);
            let c = store.feed_chunk(&line).expect("chunk");
            assert_eq!(c, 1);
        }

        let total = store.finish_loading().expect("finish");
        assert_eq!(total, 10);
        assert_eq!(store.triple_count(), 10);
    }

    #[test]
    fn test_insert_and_delete() {
        let mut store = WasmSparqlStore::new();
        store.insert_triple("<http://s>", "<http://p>", "<http://o>");
        assert!(store.triple_exists("<http://s>", "<http://p>", "<http://o>"));

        let deleted = store.delete_triple("<http://s>", "<http://p>", "<http://o>");
        assert!(deleted);
        assert!(!store.triple_exists("<http://s>", "<http://p>", "<http://o>"));
    }

    #[test]
    fn test_query_pattern_all_wildcards() {
        let mut store = make_store_with_data();
        let all = store.query_pattern(None, None, None);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_query_pattern_subject_bound() {
        let mut store = make_store_with_data();
        let results = store.query_pattern(Some("<http://example.org/alice>"), None, None);
        assert_eq!(results.len(), 2); // knows + name
    }

    #[test]
    fn test_query_pattern_predicate_bound() {
        let mut store = make_store_with_data();
        let results = store.query_pattern(None, Some("<http://example.org/name>"), None);
        assert_eq!(results.len(), 2); // Alice + Bob
    }

    #[test]
    fn test_query_pattern_predicate_object_bound() {
        let mut store = make_store_with_data();
        let results = store.query_pattern(
            None,
            Some("<http://example.org/knows>"),
            Some("<http://example.org/bob>"),
        );
        assert_eq!(results.len(), 1);
        assert!(results[0][0].contains("alice"));
    }

    #[test]
    fn test_query_pattern_no_match() {
        let mut store = make_store_with_data();
        let results = store.query_pattern(Some("<http://example.org/nobody>"), None, None);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_find_by_subject() {
        let mut store = make_store_with_data();
        let results = store.find_by_subject("<http://example.org/bob>");
        assert_eq!(results.len(), 1); // name only
    }

    #[test]
    fn test_find_by_predicate() {
        let mut store = make_store_with_data();
        let results = store.find_by_predicate("<http://example.org/name>");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_subjects_by_predicate_object() {
        let mut store = make_store_with_data();
        let subjects = store.find_subjects_by_predicate_object(
            "<http://example.org/knows>",
            "<http://example.org/bob>",
        );
        assert_eq!(subjects.len(), 1);
        assert!(subjects[0].contains("alice"));
    }

    #[test]
    fn test_to_ntriples() {
        let mut store = make_store_with_data();
        let nt = store.to_ntriples();
        assert!(nt.contains("<http://example.org/alice>"));
        assert!(nt.contains("<http://example.org/bob>"));
    }

    #[test]
    fn test_triple_count_and_term_count() {
        let store = make_store_with_data();
        assert_eq!(store.triple_count(), 3);
        // At least 5 unique terms (alice, knows, bob, name, "Alice", "Bob")
        assert!(store.term_count() >= 5);
    }

    #[test]
    fn test_memory_bytes() {
        let store = make_store_with_data();
        assert!(store.memory_bytes() > 0);
    }

    #[test]
    fn test_str_to_rdf_term_iri() {
        let term = str_to_rdf_term("<http://example.org/>");
        assert!(matches!(term, RdfTerm::Iri(_)));
        assert_eq!(term.value(), "http://example.org/");
    }

    #[test]
    fn test_str_to_rdf_term_blank() {
        let term = str_to_rdf_term("_:b0");
        assert!(matches!(term, RdfTerm::BlankNode(_)));
        assert_eq!(term.value(), "b0");
    }

    #[test]
    fn test_str_to_rdf_term_literal() {
        let term = str_to_rdf_term("\"hello\"");
        assert!(matches!(term, RdfTerm::PlainLiteral(_)));
        assert_eq!(term.value(), "hello");
    }

    #[test]
    fn test_str_to_rdf_term_lang_literal() {
        let term = str_to_rdf_term("\"hello\"@en");
        assert!(matches!(term, RdfTerm::LangLiteral { .. }));
        assert_eq!(term.value(), "hello");
        assert_eq!(term.lang(), Some("en"));
    }

    #[test]
    fn test_str_to_rdf_term_typed_literal() {
        let term = str_to_rdf_term("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>");
        assert!(matches!(term, RdfTerm::TypedLiteral { .. }));
        assert_eq!(term.value(), "42");
        assert_eq!(
            term.datatype(),
            Some("http://www.w3.org/2001/XMLSchema#integer")
        );
    }

    #[test]
    fn test_load_turtle_with_semicolon() {
        let mut store = WasmSparqlStore::new();
        let ttl = "@prefix ex: <http://example.org/> .\n\
                   ex:alice ex:knows ex:bob ; ex:name \"Alice\" .\n";
        let count = store.load_turtle(ttl).expect("load turtle");
        assert_eq!(count, 2);
        assert_eq!(store.triple_count(), 2);
    }
}
