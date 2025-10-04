//! # RDF Patch Support
//!
//! RDF Patch format support for atomic updates.
//!
//! This module implements the RDF Patch specification for describing
//! atomic changes to RDF datasets. RDF Patch provides a standardized
//! way to represent additions, deletions, and graph operations.
//!
//! Reference: https://afs.github.io/rdf-patch/

pub mod parser;
pub mod serializer;
pub mod context;
pub mod result;
pub mod conflict;
pub mod normalizer;
pub mod compressor;

// Re-export main types
pub use parser::PatchParser;
pub use serializer::PatchSerializer;
pub use context::PatchContext;
pub use result::PatchResult;
pub use conflict::{ConflictResolver, ConflictStrategy, ConflictResolution};
pub use normalizer::PatchNormalizer;
pub use compressor::PatchCompressor;
