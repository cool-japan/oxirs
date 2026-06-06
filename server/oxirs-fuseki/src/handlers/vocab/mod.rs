//! VocPrez-style vocabulary publishing handlers.
//!
//! Exposes vocabulary metadata (title, description, namespace, contributors
//! and concept counts) to clients in HTML, JSON-LD or Turtle. Inspired by the
//! Australian Government Linked Data Working Group's VocPrez Linked Data
//! viewer.

pub mod handler;
pub mod metadata;
pub mod registry;
pub mod serializer;

#[cfg(test)]
mod tests;

pub use handler::{vocab_detail_handler, vocab_list_handler};
pub use metadata::{build_metadata, VocabularyMetadata};
pub use registry::{VocabularyEntry, VocabularyRegistry};
pub use serializer::{serialize_metadata, VocabFormat};
