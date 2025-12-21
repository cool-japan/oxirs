//! Language Server Protocol (LSP) implementation for SHACL shapes.
//!
//! This module provides IDE integration for SHACL shape authoring with:
//! - Real-time validation diagnostics
//! - Code completion for SHACL properties and classes
//! - Hover documentation
//! - Go-to-definition for shape references
//! - Find references for shapes
//! - Semantic tokens for syntax highlighting
//!
//! # Features
//!
//! The LSP server supports:
//! - Turtle, JSON-LD, and RDF/XML file formats
//! - Full SHACL Core and SHACL-AF validation
//! - Cross-file shape imports
//! - Custom constraint components
//! - Performance-optimized validation with SciRS2
//!
//! # Usage
//!
//! Start the LSP server:
//! ```bash
//! cargo run --bin shacl_lsp --features lsp
//! ```
//!
//! Configure your IDE to use the server (e.g., VS Code, IntelliJ).

#[cfg(feature = "lsp")]
pub mod backend;
#[cfg(feature = "lsp")]
pub mod completion;
#[cfg(feature = "lsp")]
pub mod diagnostics;
#[cfg(feature = "lsp")]
pub mod hover;
#[cfg(feature = "lsp")]
pub mod semantic_tokens;
#[cfg(feature = "lsp")]
pub mod server;

#[cfg(feature = "lsp")]
pub use backend::ShaclBackend;
#[cfg(feature = "lsp")]
pub use server::ShaclLanguageServer;
