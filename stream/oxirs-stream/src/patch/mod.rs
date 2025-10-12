//! # RDF Patch Support
//!
//! RDF Patch format support for atomic updates.
//!
//! This module implements the RDF Patch specification for describing
//! atomic changes to RDF datasets. RDF Patch provides a standardized
//! way to represent additions, deletions, and graph operations.
//!
//! Reference: https://afs.github.io/rdf-patch/

pub mod compressor;
pub mod conflict;
pub mod context;
pub mod normalizer;
pub mod parser;
pub mod result;
pub mod serializer;

// Re-export main types
pub use compressor::PatchCompressor;
pub use conflict::{ConflictResolution, ConflictResolver, ConflictStrategy};
pub use context::PatchContext;
pub use normalizer::PatchNormalizer;
pub use parser::PatchParser;
pub use result::PatchResult;
pub use serializer::PatchSerializer;
