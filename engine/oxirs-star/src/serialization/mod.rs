//! RDF-star serialization module.
//!
//! This module provides dedicated serialization and parsing for RDF-star formats,
//! with a focus on JSON-LD-star.

pub mod jsonld_star;

pub use jsonld_star::{JsonLdStarParser, JsonLdStarSerializer};
