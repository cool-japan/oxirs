//! Format-specific parsers and serializers
//!
//! This module contains implementations for each supported RDF format.

pub mod jsonld;
pub mod jsonld_context;
pub mod n3;
pub mod n3_backward_chaining;
pub mod n3_builtins;
pub mod n3_parser;
pub mod n3_reasoning;
pub mod n3_rule_parser;
pub mod n3_serializer;
pub mod n3_types;
pub mod nquads;
// W3C N-Quads 1.1 parser (v1.1.0 round 5)
pub mod nquads_parser;
pub mod ntriples;
pub mod rdf_thrift;
pub mod rdf_xml_writer;
pub mod trig;
pub mod turtle;

// Re-export format APIs
pub use jsonld::{
    ContainerType, JsonLdContext, JsonLdError, JsonLdProcessor, JsonLdQuad, JsonLdResult,
    JsonLdTerm, JsonLdWriter, Quad, TermDefinition, Triple, WriterObject,
};
pub use n3_parser::*;
pub use n3_reasoning::*;
pub use n3_serializer::*;
pub use n3_types::*;
pub use nquads::*;
pub use ntriples::*;
pub use rdf_thrift::{
    graph_name_to_thrift, object_to_thrift, predicate_to_thrift, subject_to_thrift,
    RdfThriftReader, RdfThriftWriter, ThriftRow, ThriftTerm, ThriftWriteMode,
};
pub use trig::*;
pub use turtle::*;
