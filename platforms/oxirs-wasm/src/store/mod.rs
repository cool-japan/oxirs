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

    /// Load Turtle data
    #[wasm_bindgen(js_name = loadTurtle)]
    pub fn load_turtle(&mut self, turtle: &str) -> Result<usize, JsValue> {
        let triples =
            crate::parser::parse_turtle(turtle).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let count = triples.len();
        for triple in triples {
            self.insert_internal(triple.into());
        }

        Ok(count)
    }

    /// Load N-Triples data
    #[wasm_bindgen(js_name = loadNTriples)]
    pub fn load_ntriples(&mut self, ntriples: &str) -> Result<usize, JsValue> {
        let triples = crate::parser::parse_ntriples(ntriples)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let count = triples.len();
        for triple in triples {
            self.insert_internal(triple.into());
        }

        Ok(count)
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

    /// Delete a single triple, returning true if it was found
    pub fn delete(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        let triple = InternalTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        };

        self.triples.remove(&triple)
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

    /// The index maps a term to positions in `triple_list`, which `delete` does
    /// not compact — so every hit is checked against the live triple set.
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
}
