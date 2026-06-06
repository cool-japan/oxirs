//! RDF Patch Protocol implementation
//!
//! RDF Patch is a format for expressing changes to RDF datasets.
//! Each patch consists of optional header lines followed by change lines.
//!
//! # Format Overview
//!
//! ```text
//! H id <uuid>
//! H prev <uuid>
//! TX
//! PA ex <http://example.org/>
//! A <http://example.org/s> <http://example.org/p> <http://example.org/o>
//! D <http://example.org/s> <http://example.org/p> <http://example.org/old>
//! TC
//! ```
//!
//! Line prefixes:
//! - `H`  — header (version, id, prev)
//! - `TX` — transaction begin
//! - `TC` — transaction commit
//! - `TA` — transaction abort
//! - `PA` — add prefix
//! - `PD` — delete prefix
//! - `A`  — add triple or quad
//! - `D`  — delete triple or quad
//!
//! # References
//!
//! <https://afs.github.io/rdf-patch/>

pub mod patch_parser;
pub mod patch_types;
pub mod patch_writer;

pub use patch_parser::{apply_patch, diff_to_patch, PatchParser};
pub use patch_types::{
    Graph, PatchChange, PatchError, PatchHeader, PatchQuad, PatchResult, PatchStats, PatchTerm,
    PatchTriple, RdfPatch,
};
pub use patch_writer::PatchSerializer;

#[cfg(test)]
mod tests;
