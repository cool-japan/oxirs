//! # OxiRS WASM
//!
//! WebAssembly bindings for OxiRS - Run RDF/SPARQL in the browser.
//!
//! ## Features
//!
//! - **RDF Parsing**: Turtle, N-Triples, JSON-LD
//! - **In-Memory Store**: HashMap-based triple store
//! - **SPARQL Queries**: SELECT, ASK, CONSTRUCT
//! - **Validation**: Basic SHACL validation
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

mod error;
mod parser;
mod query;
mod store;

use wasm_bindgen::prelude::*;

pub use error::WasmError;
pub use store::OxiRSStore;

// Initialize panic hook for better error messages
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
