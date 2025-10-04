//! Triple index structures (SPO, POS, OSP)
//!
//! This module implements triple indexes for efficient
//! SPARQL query execution.

pub mod triple;
pub mod triple_index;

pub use triple::{EmptyValue, OspKey, PosKey, SpoKey, Triple};
pub use triple_index::{OspIndex, PosIndex, SpoIndex, TripleIndex, TripleIndexes};
