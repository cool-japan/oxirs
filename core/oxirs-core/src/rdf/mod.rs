//! RDF dataset utilities for OxiRS.
//!
//! - [`diff`]: Compute diffs between RDF datasets and apply patches.

pub mod blank_node_allocator;
pub mod diff;
pub mod namespace_registry;

pub use diff::{DatasetDiff, DatasetPatch, DiffStats, RdfDiffEngine, Triple as DiffTriple};
pub use namespace_registry::{NamespaceEntry, NamespaceRegistry, DEFAULT_PREFIXES};
