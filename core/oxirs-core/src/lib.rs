//! # OxiRS Core
//!
//! Core RDF and SPARQL functionality for OxiRS - a thin, safe re-export of oxigraph
//! with additional abstractions for the OxiRS ecosystem.
//!
//! This crate provides the foundational RDF and SPARQL operations that all other
//! OxiRS crates depend on, including:
//!
//! - RDF 1.2 parsing and serialization
//! - Triple store operations
//! - SPARQL query execution
//! - Memory and disk storage backends
//!
//! ## Examples
//!
//! ```rust
//! use oxirs_core::store::Store;
//! 
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let store = Store::new()?;
//! // Use the store for RDF operations
//! # Ok(())
//! # }
//! ```

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(private_interfaces)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::manual_strip)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::new_without_default)]
#![allow(clippy::single_match)]
#![allow(clippy::manual_map)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::collapsible_str_replace)]
#![allow(clippy::get_first)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::while_let_on_iterator)]
#![allow(clippy::needless_range_loop)]
#![allow(unused_doc_comments)]


pub mod model;
pub mod store;
pub mod query;
pub mod graph;
pub mod parser;
pub mod serializer;
pub mod interning;
pub mod indexing;
pub mod optimization;

// Re-export core types for convenience
pub use model::*;

/// Core error type for OxiRS operations
#[derive(Debug, thiserror::Error)]
pub enum OxirsError {
    #[error("Store error: {0}")]
    Store(String),
    #[error("Query error: {0}")]
    Query(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type alias for OxiRS operations
pub type Result<T> = std::result::Result<T, OxirsError>;

/// Version information for OxiRS Core
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize OxiRS Core with default configuration
pub fn init() -> Result<()> {
    tracing::info!("Initializing OxiRS Core v{}", VERSION);
    Ok(())
}