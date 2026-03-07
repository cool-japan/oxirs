//! # OxiRS WASM
//!
//! [![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Development Release (v0.2.0)
//!
//! WebAssembly bindings for OxiRS - Run RDF/SPARQL in the browser.
//!
//! ## Features
//!
//! - **Streaming RDF Parser**: Turtle, N-Triples, N-Quads (incremental chunk processing)
//! - **Compact Triple Store**: Dictionary-based, 12 bytes per triple, dual sorted indexes
//! - **JS/TS API**: Clean `wasm_bindgen` surface via `OxiRSStore` and `WasmSparqlStore`
//! - **SPARQL Queries**: SELECT, ASK, CONSTRUCT (triple-pattern based)
//! - **SPARQL UPDATE**: INSERT DATA, DELETE DATA, INSERT/DELETE WHERE, CLEAR, DROP
//! - **SPARQL Enhancements**: OPTIONAL, UNION, FILTER expressions
//! - **RDFS Inference**: subClassOf/subPropertyOf transitivity, domain/range typing
//! - **Named Graphs**: Multi-graph storage with GRAPH pattern queries
//! - **Memory Efficient**: Optimized for WASM's 4 GB address space limit
//!
//! ## Example (JavaScript)
//!
//! ```javascript
//! import init, { OxiRSStore } from 'oxirs-wasm';
//!
//! await init();
//!
//! const store = new OxiRSStore();
//! await store.loadTurtle(`
//!     @prefix : <http://example.org/> .
//!     :alice :knows :bob .
//!     :bob :name "Bob" .
//! `);
//!
//! const results = store.query('SELECT ?name WHERE { ?person :name ?name }');
//! console.log(results);
//! ```

pub mod api;
pub mod error;
pub mod inference;
pub mod named_graph;
pub mod parser;
pub mod query;
pub mod store;
pub mod update;

// v1.1.0 SPARQL Results CSV/TSV serializer
pub mod results_csv;

// v1.1.0 round 5: GeoJSON serialization/deserialization for SPARQL geo results
pub mod geojson_support;

// v1.1.0 round 6: SPARQL query pretty-printer and formatter
pub mod sparql_formatter;

// v1.1.0 round 7: Namespace/prefix management for RDF serializers
pub mod prefix_manager;

// v1.1.0 round 11: Streaming / incremental chunk-based RDF document parser
pub mod streaming_parser;

// v1.1.0 round 12: WASM/JS interop bridge type conversions
pub mod wasm_bridge;

// v1.1.0 round 13: Client-side SPARQL query validation (syntax/semantic checks)
pub mod query_validator;

// v1.1.0 round 11: RDF graph visualization with Fruchterman-Reingold layout
pub mod graph_visualizer;

// v1.1.0 round 12: In-memory indexed triple store for WASM
pub mod triple_store;

// v1.1.0 round 13: In-memory SPARQL executor (SELECT/ASK/CONSTRUCT/DESCRIBE + FILTER)
pub mod sparql_executor;

// v1.1.0 round 14: WASM storage abstraction layer (key-value with namespace isolation + TTL)
pub mod storage_adapter;

// v1.1.0 round 15: WASM event dispatching system (pub/sub for browser events)
pub mod event_dispatcher;

// v1.1.0 round 16: RDF namespace prefix management for WASM bindings
pub mod namespace_manager;

use wasm_bindgen::prelude::*;

pub use api::WasmSparqlStore;
pub use error::WasmError;
pub use store::OxiRSStore;

// Initialize panic hook for better error messages in browser DevTools
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// RDF Triple representation for JavaScript
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Triple {
    subject: String,
    predicate: String,
    object: String,
}

#[wasm_bindgen]
impl Triple {
    #[wasm_bindgen(constructor)]
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn subject(&self) -> String {
        self.subject.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn predicate(&self) -> String {
        self.predicate.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn object(&self) -> String {
        self.object.clone()
    }
}

/// Query result for JavaScript
#[wasm_bindgen]
pub struct QueryResult {
    bindings: Vec<JsValue>,
}

#[wasm_bindgen]
impl QueryResult {
    #[wasm_bindgen(getter)]
    pub fn bindings(&self) -> Vec<JsValue> {
        self.bindings.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.bindings.len()
    }
}

/// Get OxiRS WASM version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Log a message to the browser console
#[wasm_bindgen]
pub fn log(message: &str) {
    web_sys::console::log_1(&message.into());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple() {
        let triple = Triple::new("http://a", "http://b", "http://c");
        assert_eq!(triple.subject(), "http://a");
        assert_eq!(triple.predicate(), "http://b");
        assert_eq!(triple.object(), "http://c");
    }
}
