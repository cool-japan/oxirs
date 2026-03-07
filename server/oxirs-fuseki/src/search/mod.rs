//! Full-text search subsystem for OxiRS Fuseki
//!
//! Provides indexing and querying of RDF literals via the `text:` service extension.

pub mod sparql_text_ext;
pub mod text_index;
pub mod text_search_index;

pub use sparql_text_ext::{SparqlTextSearchExtension, TextQueryBinding, TextQueryCall};
pub use text_index::{
    IndexedLiteral, SimpleTextIndex, TantivyTextIndex, TextIndex, TextIndexBackend, TextSearchHit,
};
pub use text_search_index::{SearchHit, TextSearchIndex};
