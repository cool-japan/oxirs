//! Public API layer for OxiRS WASM
//!
//! Provides clean Rust types for JavaScript binding via `wasm_bindgen`,
//! and for direct use in native tests.

pub mod wasm_api;

pub use wasm_api::WasmSparqlStore;
