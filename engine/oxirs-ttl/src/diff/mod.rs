//! RDF graph diff and patch utilities
//!
//! Provides [`RdfDiff`] for representing the symmetric difference between two
//! RDF graphs, along with [`compute_diff`] to compute diffs and [`parse_patch`]
//! to reconstruct a diff from a serialised patch document.

pub mod rdf_diff;

pub use rdf_diff::{compute_diff, parse_patch, NTriple, PatchParseError, RdfDiff};
