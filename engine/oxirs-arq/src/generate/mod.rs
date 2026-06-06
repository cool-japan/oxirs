//! SPARQL-Generate extension module for OxiRS ARQ.
//!
//! Implements a meaningful subset of the W3C community SPARQL-Generate
//! specification (<https://ci.mines-stetienne.fr/sparql-generate/>), enabling
//! text/JSON/CSV generation from RDF data via a `GENERATE { ... } WHERE { ... }`
//! query template.
//!
//! # Architecture
//!
//! | Sub-module     | Responsibility                                        |
//! |----------------|------------------------------------------------------|
//! | [`ast`]        | AST types: `GenerateQuery`, `TemplateClause`, etc.   |
//! | [`parser`]     | Hand-rolled tokenizer + parser                       |
//! | [`executor`]   | Template evaluation against SPARQL binding rows      |
//!
//! # Quick example
//!
//! ```rust
//! use oxirs_arq::generate::{parse_generate, GenerateExecutor, Bindings};
//! use std::collections::HashMap;
//!
//! let src = r#"
//!     GENERATE { "name=" ?name }
//!     WHERE { ?s foaf:name ?name . }
//! "#;
//!
//! let query = parse_generate(src).unwrap();
//! let exec  = GenerateExecutor::new(query);
//!
//! let mut row = HashMap::new();
//! row.insert("name".to_string(), "Alice".to_string());
//!
//! let result = exec.evaluate_one(&row).unwrap();
//! assert_eq!(result.text, "name=Alice");
//! ```

pub mod ast;
pub mod executor;
pub mod parser;

#[cfg(test)]
mod tests;

pub use ast::{GenerateLiteral, GenerateQuery, TemplateClause};
pub use executor::{Bindings, GenerateExecutor, GenerateResult};
pub use parser::parse as parse_generate;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during SPARQL-Generate parsing or template evaluation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GenerateError {
    /// A syntax error encountered while parsing the GENERATE query.
    #[error("Parse error at position {pos}: {msg}")]
    ParseError {
        /// Byte offset (or token index) in the input where the error was detected.
        pos: usize,
        /// Human-readable description of the parse error.
        msg: String,
    },

    /// A template variable reference could not be resolved in the current
    /// binding row.  The inner `String` is the variable name (without `?`).
    #[error("Unbound variable: {0}")]
    UnboundVariable(String),

    /// A generic template evaluation error.
    #[error("Template error: {0}")]
    TemplateError(String),
}
