//! GraphQL schema generation from RDF ontologies.
//!
//! This module is a thin facade. The implementation is split across:
//! - [`crate::schema_types`]      — RDF vocabulary value types (`RdfVocabulary`,
//!   `RdfClass`, `RdfProperty`, `PropertyType`) and `SchemaGenerationConfig`.
//! - [`crate::schema_generator`]  — the [`SchemaGenerator`] and the RDF-to-GraphQL
//!   type generation logic (objects, queries, mutations, subscriptions).
//! - [`crate::schema_sdl`]        — SDL serialization and naming helpers.
//! - [`crate::schema_loader`]     — vocabulary extraction and ontology loading.

pub use crate::schema_generator::SchemaGenerator;
pub use crate::schema_types::{
    PropertyType, RdfClass, RdfProperty, RdfVocabulary, SchemaGenerationConfig,
};
