//! In-memory RDF store for WASM

use crate::error::{WasmError, WasmResult};
use crate::{QueryResult, Triple};
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

/// OxiRS in-memory RDF store
#[wasm_bindgen]
pub struct OxiRSStore {
    /// All triples
    triples: HashSet<InternalTriple>,
    /// Subject index
    subject_index: HashMap<String, Vec<usize>>,
    /// Predicate index
    predicate_index: HashMap<String, Vec<usize>>,
    /// Object index
    object_index: HashMap<String, Vec<usize>>,
    /// Triple list for indexing
    triple_list: Vec<InternalTriple>,
    /// Namespace prefixes
    prefixes: HashMap<String, String>,
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
        }
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

    /// Delete a single triple
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

        // Convert to JS array
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

        // Prefixes
        for (prefix, uri) in &self.prefixes {
            result.push_str(&format!("@prefix {}: <{}> .\n", prefix, uri));
        }
        if !self.prefixes.is_empty() {
            result.push('\n');
        }

        // Triples
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
}

// Internal methods
impl OxiRSStore {
    fn insert_internal(&mut self, triple: InternalTriple) {
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

    /// Get triples by subject (internal use)
    pub(crate) fn get_by_subject(&self, subject: &str) -> Vec<&InternalTriple> {
        self.subject_index
            .get(subject)
            .map(|indices| indices.iter().map(|&i| &self.triple_list[i]).collect())
            .unwrap_or_default()
    }

    /// Get triples by predicate (internal use)
    pub(crate) fn get_by_predicate(&self, predicate: &str) -> Vec<&InternalTriple> {
        self.predicate_index
            .get(predicate)
            .map(|indices| indices.iter().map(|&i| &self.triple_list[i]).collect())
            .unwrap_or_default()
    }

    /// Get all triples (internal use)
    pub(crate) fn all_triples(&self) -> impl Iterator<Item = &InternalTriple> {
        self.triples.iter()
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
        assert!(!store.insert("http://a", "http://b", "http://c")); // Duplicate

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
