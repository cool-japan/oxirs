//! RDF triple store implementations for OxiRS WASM
//!
//! This module provides two store implementations:
//! - [`compact_store`]: Memory-efficient dictionary-based store optimized for WASM
//! - The `OxiRSStore` (re-exported for backward compatibility with existing JS bindings)

pub mod compact_store;

pub use compact_store::{CompactDictionary, CompactTripleStore, NodeId, RdfTerm};

// Re-export the original OxiRSStore for JS/wasm_bindgen use
// (implementation lives in oxirs_store module below)

use crate::error::{WasmError, WasmResult};
use crate::Triple;
use std::collections::{HashMap, HashSet};
use wasm_bindgen::prelude::*;

/// Internal triple representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct InternalTriple {
    pub(crate) subject: String,
    pub(crate) predicate: String,
    pub(crate) object: String,
}

impl From<&Triple> for InternalTriple {
    fn from(t: &Triple) -> Self {
        Self {
            subject: t.subject.clone(),
            predicate: t.predicate.clone(),
            object: t.object.clone(),
        }
    }
}

impl From<crate::parser::ParsedTriple> for InternalTriple {
    fn from(t: crate::parser::ParsedTriple) -> Self {
        Self {
            subject: t.subject,
            predicate: t.predicate,
            object: t.object,
        }
    }
}

/// OxiRS in-memory RDF store exposed to JavaScript via wasm_bindgen
#[wasm_bindgen]
pub struct OxiRSStore {
    /// All triples (deduplication set)
    triples: HashSet<InternalTriple>,
    /// Subject index: subject string → list of positions in `triple_list`
    subject_index: HashMap<String, Vec<usize>>,
    /// Predicate index
    predicate_index: HashMap<String, Vec<usize>>,
    /// Object index
    object_index: HashMap<String, Vec<usize>>,
    /// Ordered list of triples (for indexing by position)
    triple_list: Vec<InternalTriple>,
    /// Namespace prefix bindings
    prefixes: HashMap<String, String>,
    /// Ceiling on intermediate solutions per pattern (None = unlimited)
    solution_budget: Option<usize>,
    /// Ceiling on the number of triples this store may hold (None = unlimited)
    triple_limit: Option<usize>,
}

#[wasm_bindgen]
impl OxiRSStore {
    /// Create a new empty store
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            triples: HashSet::new(),
            subject_index: HashMap::new(),
            predicate_index: HashMap::new(),
            object_index: HashMap::new(),
            triple_list: Vec::new(),
            prefixes: HashMap::new(),
            solution_budget: None,
            triple_limit: None,
        }
    }

    /// Cap the number of intermediate solutions a single query may produce.
    ///
    /// A join is evaluated left to right, so an unselective pattern early in a
    /// WHERE clause can build a huge intermediate result before a later pattern
    /// cuts it back down — the join is what costs, not the rows returned, and a
    /// LIMIT cannot bound it because it applies at the end. A caller that
    /// answers queries under a time or CPU budget (an endpoint on a serverless
    /// platform, say) sets this so that such a query fails fast with a query
    /// error instead of running to completion.
    ///
    /// Unset by default: queries are unbounded.
    #[wasm_bindgen(js_name = setSolutionBudget)]
    pub fn set_solution_budget(&mut self, budget: usize) {
        self.solution_budget = Some(budget);
    }

    /// Remove the solution budget set by [`OxiRSStore::set_solution_budget`].
    #[wasm_bindgen(js_name = clearSolutionBudget)]
    pub fn clear_solution_budget(&mut self) {
        self.solution_budget = None;
    }

    /// Cap the total number of triples this store may hold.
    ///
    /// `load_turtle`/`load_ntriples` check this budget as they insert, and
    /// fail with an explicit, catchable error the moment a document would
    /// push the store past the cap — rather than growing without bound until
    /// the WASM heap itself is exhausted (which surfaces to JS as an
    /// uncatchable trap, not a clean `Err`).
    ///
    /// Unset by default: the store is unbounded.
    #[wasm_bindgen(js_name = setTripleLimit)]
    pub fn set_triple_limit(&mut self, limit: usize) {
        self.triple_limit = Some(limit);
    }

    /// Remove the triple-count ceiling set by [`OxiRSStore::set_triple_limit`].
    #[wasm_bindgen(js_name = clearTripleLimit)]
    pub fn clear_triple_limit(&mut self) {
        self.triple_limit = None;
    }

    /// Load Turtle data
    #[wasm_bindgen(js_name = loadTurtle)]
    pub fn load_turtle(&mut self, turtle: &str) -> Result<usize, JsValue> {
        let triples =
            crate::parser::parse_turtle(turtle).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.bulk_insert_checked(triples).map_err(JsValue::from)
    }

    /// Load N-Triples data
    #[wasm_bindgen(js_name = loadNTriples)]
    pub fn load_ntriples(&mut self, ntriples: &str) -> Result<usize, JsValue> {
        let triples = crate::parser::parse_ntriples(ntriples)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.bulk_insert_checked(triples).map_err(JsValue::from)
    }

    /// Insert a single triple
    pub fn insert(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        let triple = InternalTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        };

        if self.triples.contains(&triple) {
            return false;
        }

        self.insert_internal(triple);
        true
    }

    /// Delete a single triple, returning true if it was found.
    ///
    /// Rebuilds the subject/predicate/object indexes so that [`OxiRSStore::subjects`],
    /// [`OxiRSStore::predicates`] and [`OxiRSStore::objects`] never report a term whose
    /// last triple was just deleted, and so `triple_list`/the indexes do not grow
    /// unboundedly across long insert/delete churn.
    pub fn delete(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        let triple = InternalTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        };

        if self.triples.remove(&triple) {
            self.rebuild_indexes();
            true
        } else {
            false
        }
    }

    /// Execute a SPARQL UPDATE (INSERT DATA, DELETE DATA, INSERT/DELETE WHERE,
    /// CLEAR, DROP). Returns the number of triples affected.
    pub fn update(&mut self, sparql: &str) -> Result<u32, JsValue> {
        crate::update::execute_update(sparql, self).map_err(JsValue::from)
    }

    /// Check if a triple exists
    pub fn contains(&self, subject: &str, predicate: &str, object: &str) -> bool {
        let triple = InternalTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        };

        self.triples.contains(&triple)
    }

    /// Get the number of triples
    pub fn size(&self) -> usize {
        self.triples.len()
    }

    /// Clear all triples
    pub fn clear(&mut self) {
        self.triples.clear();
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
        self.triple_list.clear();
    }

    /// Execute a SPARQL SELECT query
    pub fn query(&self, sparql: &str) -> Result<JsValue, JsValue> {
        let results = crate::query::execute_select(sparql, self)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let array = js_sys::Array::new();
        for binding in results {
            let obj = js_sys::Object::new();
            for (key, value) in binding {
                js_sys::Reflect::set(&obj, &JsValue::from_str(&key), &JsValue::from_str(&value))
                    .map_err(|_| JsValue::from_str("Failed to set property"))?;
            }
            array.push(&obj);
        }

        Ok(array.into())
    }

    /// Execute a SPARQL ASK query
    pub fn ask(&self, sparql: &str) -> Result<bool, JsValue> {
        crate::query::execute_ask(sparql, self).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Execute a SPARQL CONSTRUCT query
    pub fn construct(&self, sparql: &str) -> Result<Vec<Triple>, JsValue> {
        crate::query::execute_construct(sparql, self).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Export to Turtle format
    #[wasm_bindgen(js_name = toTurtle)]
    pub fn to_turtle(&self) -> String {
        let mut result = String::new();

        for (prefix, uri) in &self.prefixes {
            result.push_str(&format!("@prefix {}: <{}> .\n", prefix, uri));
        }
        if !self.prefixes.is_empty() {
            result.push('\n');
        }

        for triple in &self.triples {
            result.push_str(&format!(
                "<{}> <{}> {} .\n",
                triple.subject,
                triple.predicate,
                self.format_object(&triple.object)
            ));
        }

        result
    }

    /// Export to N-Triples format
    #[wasm_bindgen(js_name = toNTriples)]
    pub fn to_ntriples(&self) -> String {
        let mut result = String::new();

        for triple in &self.triples {
            result.push_str(&format!(
                "<{}> <{}> {} .\n",
                triple.subject,
                triple.predicate,
                self.format_object(&triple.object)
            ));
        }

        result
    }

    /// Get all subjects
    pub fn subjects(&self) -> Vec<String> {
        self.subject_index.keys().cloned().collect()
    }

    /// Get all predicates
    pub fn predicates(&self) -> Vec<String> {
        self.predicate_index.keys().cloned().collect()
    }

    /// Get all objects
    pub fn objects(&self) -> Vec<String> {
        self.object_index.keys().cloned().collect()
    }

    /// Register a namespace prefix
    #[wasm_bindgen(js_name = addPrefix)]
    pub fn add_prefix(&mut self, prefix: &str, uri: &str) {
        self.prefixes.insert(prefix.to_string(), uri.to_string());
    }

    /// Apply RDFS forward-chaining entailment rules and materialise inferred triples.
    ///
    /// Returns a JS object `{ added: number }` describing how many new triples
    /// were derived.  Calling this multiple times is idempotent — subsequent
    /// calls return 0 once the fixed-point has been reached.
    ///
    /// Rules implemented: rdfs:subClassOf transitivity (rdfs11), rdf:type
    /// propagation via rdfs:subClassOf (rdfs9), rdfs:subPropertyOf transitivity
    /// (rdfs5), rdfs:subPropertyOf usage propagation (rdfs7), rdfs:domain
    /// subject-typing (rdfs2), rdfs:range object-typing (rdfs3).
    #[wasm_bindgen(js_name = inferRdfs)]
    pub fn infer_rdfs(&mut self) -> Result<JsValue, JsValue> {
        let added = crate::inference::apply_rdfs_inference(self);
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("added"),
            &JsValue::from_f64(added as f64),
        )
        .map_err(|_| JsValue::from_str("Failed to build inferRdfs result object"))?;
        Ok(obj.into())
    }
}

// Internal (non-wasm_bindgen) methods
impl OxiRSStore {
    pub(crate) fn insert_internal(&mut self, triple: InternalTriple) {
        if self.triples.insert(triple.clone()) {
            let idx = self.triple_list.len();
            self.triple_list.push(triple.clone());

            self.subject_index
                .entry(triple.subject.clone())
                .or_default()
                .push(idx);
            self.predicate_index
                .entry(triple.predicate.clone())
                .or_default()
                .push(idx);
            self.object_index
                .entry(triple.object.clone())
                .or_default()
                .push(idx);
        }
    }

    /// Rebuild `triple_list` and all three term indexes from `self.triples`.
    ///
    /// Called after a deletion so that stale index entries (positions that no
    /// longer point at a live triple) never accumulate: without this, `subjects()`
    /// / `predicates()` / `objects()` would keep reporting terms whose last triple
    /// was deleted, and `triple_list` would grow without bound across long
    /// insert/delete churn in a browser tab.
    fn rebuild_indexes(&mut self) {
        self.triple_list = self.triples.iter().cloned().collect();
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
        for (idx, triple) in self.triple_list.iter().enumerate() {
            self.subject_index
                .entry(triple.subject.clone())
                .or_default()
                .push(idx);
            self.predicate_index
                .entry(triple.predicate.clone())
                .or_default()
                .push(idx);
            self.object_index
                .entry(triple.object.clone())
                .or_default()
                .push(idx);
        }
    }

    /// Insert a batch of parsed triples, honoring `triple_limit`.
    ///
    /// Returns the number of triples processed (matching the historical
    /// `load_turtle`/`load_ntriples` contract of returning the parsed count on
    /// success), or a [`WasmError::StoreError`] the moment inserting a
    /// genuinely new triple would push the store past the configured
    /// ceiling. Already-inserted triples from this call remain in the store —
    /// the caller can inspect `size()` to see how far loading got before
    /// hitting the limit.
    ///
    /// Returns a plain [`WasmError`] rather than `JsValue` so this logic stays
    /// natively unit-testable (constructing a `JsValue` requires a real
    /// `wasm32` + JS host and panics under `cargo test`); the `#[wasm_bindgen]`
    /// callers convert with `.map_err(JsValue::from)` at the crate boundary.
    pub(crate) fn bulk_insert_checked<T: Into<InternalTriple>>(
        &mut self,
        triples: Vec<T>,
    ) -> WasmResult<usize> {
        let count = triples.len();
        for triple in triples {
            let triple: InternalTriple = triple.into();
            if let Some(limit) = self.triple_limit {
                if !self.triples.contains(&triple) && self.triples.len() >= limit {
                    return Err(WasmError::StoreError(format!(
                        "triple limit exceeded: cannot grow store past the configured \
                         maximum of {limit} triples (call setTripleLimit to raise it, or \
                         clearTripleLimit to remove the cap)"
                    )));
                }
            }
            self.insert_internal(triple);
        }
        Ok(count)
    }

    fn format_object(&self, object: &str) -> String {
        if object.starts_with("http://")
            || object.starts_with("https://")
            || object.starts_with("urn:")
        {
            format!("<{}>", object)
        } else if object.starts_with('"') {
            object.to_string()
        } else {
            format!("\"{}\"", object)
        }
    }

    pub(crate) fn get_by_subject(&self, subject: &str) -> Vec<&InternalTriple> {
        self.subject_index
            .get(subject)
            .map(|indices| indices.iter().map(|&i| &self.triple_list[i]).collect())
            .unwrap_or_default()
    }

    pub(crate) fn get_by_predicate(&self, predicate: &str) -> Vec<&InternalTriple> {
        self.predicate_index
            .get(predicate)
            .map(|indices| indices.iter().map(|&i| &self.triple_list[i]).collect())
            .unwrap_or_default()
    }

    pub(crate) fn all_triples(&self) -> impl Iterator<Item = &InternalTriple> {
        self.triples.iter()
    }

    /// The ceiling on intermediate solutions, if one was set.
    pub(crate) fn solution_budget(&self) -> Option<usize> {
        self.solution_budget
    }

    /// Triples with this subject, via the subject index.
    pub(crate) fn triples_with_subject(&self, term: &str) -> Vec<&InternalTriple> {
        self.lookup(&self.subject_index, term)
    }

    /// Triples with this predicate, via the predicate index.
    pub(crate) fn triples_with_predicate(&self, term: &str) -> Vec<&InternalTriple> {
        self.lookup(&self.predicate_index, term)
    }

    /// Triples with this object, via the object index.
    pub(crate) fn triples_with_object(&self, term: &str) -> Vec<&InternalTriple> {
        self.lookup(&self.object_index, term)
    }

    /// Look a term up in an index under both of the forms an IRI can be held
    /// in: `<iri>` as the parsers produce it, and bare `iri` as
    /// [`OxiRSStore::insert`] accepts it.
    fn lookup<'a>(
        &'a self,
        index: &'a HashMap<String, Vec<usize>>,
        term: &str,
    ) -> Vec<&'a InternalTriple> {
        let mut hits = self.lookup_exact(index, term);
        if let Some(alternate) = alternate_iri_form(term) {
            hits.extend(self.lookup_exact(index, &alternate));
        }
        hits
    }

    /// The index maps a term to positions in `triple_list`. `delete` rebuilds
    /// both on every call, so in practice every position is already live —
    /// this liveness check is retained as a defensive belt-and-braces
    /// guarantee (self-healing against any future indexing bug) rather than
    /// because it is load-bearing today.
    fn lookup_exact<'a>(
        &'a self,
        index: &'a HashMap<String, Vec<usize>>,
        term: &str,
    ) -> Vec<&'a InternalTriple> {
        match index.get(term) {
            Some(positions) => positions
                .iter()
                .filter_map(|&i| self.triple_list.get(i))
                .filter(|triple| self.triples.contains(*triple))
                .collect(),
            None => Vec::new(),
        }
    }
}

/// `<iri>` ↔ `iri`: the other spelling of the same IRI, or None for a literal
/// or a blank node, which have only one form.
fn alternate_iri_form(term: &str) -> Option<String> {
    match term.strip_prefix('<').and_then(|t| t.strip_suffix('>')) {
        Some(body) => Some(body.to_string()),
        None if !term.starts_with('"') && !term.starts_with("_:") => Some(format!("<{}>", term)),
        None => None,
    }
}

impl Default for OxiRSStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_operations() {
        let mut store = OxiRSStore::new();

        assert!(store.insert("http://a", "http://b", "http://c"));
        assert!(!store.insert("http://a", "http://b", "http://c"));

        assert_eq!(store.size(), 1);
        assert!(store.contains("http://a", "http://b", "http://c"));

        assert!(store.delete("http://a", "http://b", "http://c"));
        assert_eq!(store.size(), 0);
    }

    #[test]
    fn test_export() {
        let mut store = OxiRSStore::new();
        store.insert(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        );

        let nt = store.to_ntriples();
        assert!(nt.contains("<http://example.org/s>"));
        assert!(nt.contains("<http://example.org/p>"));
        assert!(nt.contains("<http://example.org/o>"));
    }

    #[test]
    fn regression_delete_removes_subject_from_subjects_list() {
        // Before the fix, subject_index/predicate_index/object_index kept
        // stale keys forever once every triple referencing them was deleted.
        let mut store = OxiRSStore::new();
        store.insert("http://only-triple-s", "http://p", "http://o");
        assert!(store
            .subjects()
            .contains(&"http://only-triple-s".to_string()));

        assert!(store.delete("http://only-triple-s", "http://p", "http://o"));

        assert!(
            !store
                .subjects()
                .contains(&"http://only-triple-s".to_string()),
            "deleted subject must not still be reported by subjects(): {:?}",
            store.subjects()
        );
    }

    #[test]
    fn regression_delete_removes_predicate_and_object_from_index_lists() {
        let mut store = OxiRSStore::new();
        store.insert("http://s", "http://only-triple-p", "http://only-triple-o");
        assert!(store.delete("http://s", "http://only-triple-p", "http://only-triple-o"));

        assert!(!store
            .predicates()
            .contains(&"http://only-triple-p".to_string()));
        assert!(!store
            .objects()
            .contains(&"http://only-triple-o".to_string()));
    }

    #[test]
    fn regression_delete_keeps_shared_terms_reported() {
        // A term used by more than one triple must survive deleting just one
        // of them.
        let mut store = OxiRSStore::new();
        store.insert("http://alice", "http://knows", "http://bob");
        store.insert("http://alice", "http://knows", "http://carol");

        assert!(store.delete("http://alice", "http://knows", "http://bob"));

        assert!(store.subjects().contains(&"http://alice".to_string()));
        assert!(store.predicates().contains(&"http://knows".to_string()));
        assert!(store.objects().contains(&"http://carol".to_string()));
        assert!(!store.objects().contains(&"http://bob".to_string()));
    }

    #[test]
    fn regression_delete_does_not_leave_dangling_index_positions() {
        // Insert three triples sharing no terms, delete the middle one, and
        // confirm the surviving two are still findable via subjects() /
        // triples_with_subject — this exercises that `triple_list` positions
        // referenced by the indexes are rebuilt (not merely masked).
        let mut store = OxiRSStore::new();
        store.insert("http://s1", "http://p1", "http://o1");
        store.insert("http://s2", "http://p2", "http://o2");
        store.insert("http://s3", "http://p3", "http://o3");

        assert!(store.delete("http://s2", "http://p2", "http://o2"));

        let subjects = store.subjects();
        assert!(subjects.contains(&"http://s1".to_string()));
        assert!(subjects.contains(&"http://s3".to_string()));
        assert!(!subjects.contains(&"http://s2".to_string()));
        assert_eq!(store.size(), 2);
    }

    #[test]
    fn regression_update_binding_executes_sparql_update() {
        // OxiRSStore::update() delegates to crate::update::execute_update —
        // previously there was no wasm_bindgen-reachable way to run a
        // SPARQL UPDATE at all.
        let mut store = OxiRSStore::new();
        let affected = store
            .update("INSERT DATA { <http://a> <http://b> <http://c> }")
            .expect("update should succeed");
        assert_eq!(affected, 1);
        assert!(store.contains("http://a", "http://b", "http://c"));

        let affected = store
            .update("DELETE DATA { <http://a> <http://b> <http://c> }")
            .expect("update should succeed");
        assert_eq!(affected, 1);
        assert!(!store.contains("http://a", "http://b", "http://c"));
    }

    #[test]
    fn regression_triple_limit_rejects_oversized_load_instead_of_growing_unbounded() {
        // Exercised at the `bulk_insert_checked` (plain `WasmError`) layer
        // rather than through the `#[wasm_bindgen]` `load_ntriples` method:
        // constructing the `JsValue` that method's `Result` requires calls
        // into a real `wasm32` + JS host and is not something a native
        // `cargo test`/nextest run can do; `bulk_insert_checked` carries the
        // exact same limit-enforcement logic without that boundary crossing.
        let mut store = OxiRSStore::new();
        store.set_triple_limit(2);

        let ntriples = "<http://s0> <http://p> <http://o0> .\n\
                         <http://s1> <http://p> <http://o1> .\n\
                         <http://s2> <http://p> <http://o2> .\n";
        let parsed = crate::parser::parse_ntriples(ntriples).expect("parse");
        let result = store.bulk_insert_checked(parsed);
        assert!(
            result.is_err(),
            "loading past the configured triple limit must fail loudly"
        );
        // The first two (within budget) should have been inserted before the
        // limit was hit; the store must not silently exceed the cap.
        assert!(store.size() <= 2);
    }

    #[test]
    fn regression_triple_limit_allows_load_within_budget() {
        let mut store = OxiRSStore::new();
        store.set_triple_limit(5);
        let parsed =
            crate::parser::parse_ntriples("<http://s> <http://p> <http://o> .\n").expect("parse");
        let n = store.bulk_insert_checked(parsed).expect("within budget");
        assert_eq!(n, 1);
    }

    #[test]
    fn regression_clear_triple_limit_removes_the_cap() {
        let mut store = OxiRSStore::new();
        store.set_triple_limit(1);
        store.clear_triple_limit();
        let parsed = crate::parser::parse_ntriples(
            "<http://s0> <http://p> <http://o0> .\n<http://s1> <http://p> <http://o1> .\n",
        )
        .expect("parse");
        let n = store
            .bulk_insert_checked(parsed)
            .expect("cap was cleared, so this must not error");
        assert_eq!(n, 2);
    }
}
