//! W3C SPARQL Service Description (SD ontology) implementation — thin facade.
//!
//! Reference: <https://www.w3.org/TR/sparql11-service-description/>
//!
//! Allows SPARQL endpoints to self-describe their capabilities by returning
//! an RDF graph conforming to the SPARQL Service Description vocabulary.
//!
//! The implementation is split across sibling modules and surfaced here via
//! re-exports so consumers can keep importing from
//! `crate::service_description::*`:
//!
//! - [`service_description_types`]: the SD vocabulary types
//!   (`ServiceDescription`, `SdFeature`, `SdLanguage`, `SdResultFormat`,
//!   `SdInputFormat`, `EntailmentRegime`) and namespace constants.
//! - [`service_description_builder`]: [`ServiceDescriptionBuilder`] and the
//!   `ServiceDescription` constructor / query / merge helpers.
//! - [`service_description_serializer`]: Turtle, JSON-LD, and RDF/XML
//!   serializers plus string-escaping helpers.

mod service_description_builder;
mod service_description_serializer;
mod service_description_types;

#[cfg(test)]
mod service_description_tests;

pub use service_description_builder::*;
pub use service_description_serializer::*;
pub use service_description_types::*;
