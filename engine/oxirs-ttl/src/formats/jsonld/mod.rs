//! JSON-LD 1.1 serialization, deserialization, and processing algorithms.
//!
//! Implements the W3C JSON-LD 1.1 specification:
//! <https://www.w3.org/TR/json-ld11/>
//!
//! Supports:
//! - Expansion (CURIE/prefix resolution to full IRIs)
//! - Compaction (IRI shortening with a context)
//! - Flattening (nested → flat @graph)
//! - Framing (reshape output to match a frame)
//! - RDF serialization/deserialization (JSON-LD ↔ N-Quads)
//! - Streaming writer with compact/pretty options

pub mod jsonld_context;
pub mod jsonld_processor;
pub mod jsonld_writer;

pub use jsonld_context::{
    is_absolute_iri, ContainerType, JsonLdContext, JsonLdError, JsonLdQuad, JsonLdResult,
    JsonLdTerm, TermDefinition, RDF_LANG_STRING, XSD_BOOLEAN, XSD_DOUBLE, XSD_INTEGER, XSD_STRING,
};
pub use jsonld_processor::JsonLdProcessor;
pub use jsonld_writer::{JsonLdWriter, Quad, Triple, WriterObject};

#[cfg(test)]
mod tests;
