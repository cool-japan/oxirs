//! JenaText full-text SPARQL integration for OxiRS ARQ.
//!
//! Provides the `text:query` property function backed by a tantivy full-text
//! index.  Enable with the `text-search` Cargo feature.
//!
//! ## Quick Start
//!
//! ```rust
//! # #[cfg(feature = "text-search")]
//! # {
//! use std::sync::Arc;
//! use oxirs_arq::text_search::{TextSearchIndex, register_text_query};
//! use oxirs_arq::property_functions::PropertyFunctionRegistry;
//!
//! let index = Arc::new(TextSearchIndex::new_in_memory().expect("index"));
//! let registry = PropertyFunctionRegistry::new();
//! register_text_query(&registry, index).expect("register");
//! # }
//! ```

pub mod index;
pub mod property_fn;

pub use index::{TextSearchError, TextSearchIndex, TextSearchResult, TEXT_NAMESPACE};
pub use property_fn::{TextQueryPropertyFunction, TEXT_QUERY_IRI};

use std::sync::Arc;

use anyhow::Result;

use crate::property_functions::PropertyFunctionRegistry;

/// Register the `text:query` property function into `registry` backed by `index`.
///
/// After registration, any SPARQL query containing a `text:query` triple
/// pattern is dispatched to the tantivy index for full-text evaluation.
pub fn register_text_query(
    registry: &PropertyFunctionRegistry,
    index: Arc<TextSearchIndex>,
) -> Result<()> {
    registry.register(TextQueryPropertyFunction::new(index))
}
