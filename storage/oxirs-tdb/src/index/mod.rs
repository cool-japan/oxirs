//! Triple and quad index structures (SPO, POS, OSP, GSPO, GPOS, GOSP)
//!
//! This module implements triple indexes for efficient SPARQL query execution
//! and quad indexes for named graph (RDF dataset) support.

pub mod bloom_filter;
pub mod quad;
pub mod triple;
pub mod triple_index;

pub use bloom_filter::{BloomFilter, BloomFilterConfig, BloomFilterStats, CountingBloomFilter};
pub use quad::{GospKey, GposKey, GspoKey, Quad, QuadIndexes};
pub use triple::{EmptyValue, OspKey, PosKey, SpoKey, Triple};
pub use triple_index::{OspIndex, PosIndex, SpoIndex, TripleIndex, TripleIndexes};
