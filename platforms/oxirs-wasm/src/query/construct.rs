//! Enhanced SPARQL CONSTRUCT query support for WASM.
//!
//! Implements proper CONSTRUCT template parsing per SPARQL 1.1 specification,
//! including template triple patterns, scoped blank node generation, literal
//! propagation, deduplication, the `CONSTRUCT WHERE` shorthand, multi-format
//! serialization, and template-expansion statistics.
//!
//! This module is a thin facade. The implementation is split across:
//! - [`crate::query::construct_types`]     — [`ConstructConfig`], [`ConstructQuery`],
//!   [`TemplateTriple`], [`TemplateTerm`], and [`ConstructStats`].
//! - [`crate::query::construct_engine`]    — the [`ConstructEngine`] executor.
//! - [`crate::query::construct_parser`]    — [`parse_construct_query`] and parsing helpers.
//! - [`crate::query::construct_serialize`] — [`serialize_construct`] and output formats.

pub use super::construct_engine::ConstructEngine;
pub use super::construct_parser::parse_construct_query;
pub use super::construct_serialize::{serialize_construct, ConstructOutputFormat};
pub use super::construct_types::{
    ConstructConfig, ConstructQuery, ConstructStats, TemplateTerm, TemplateTriple,
};
