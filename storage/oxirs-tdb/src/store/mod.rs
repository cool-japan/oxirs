//! High-level TDB store API
//!
//! Provides the main TDBStore interface for interacting with the storage engine.
//! Integrates all components: dictionary, indexes, transactions, compression.

pub mod store_impl;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod store_index;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod store_params;
pub mod store_quad;
pub mod store_stream;
mod store_tests;
pub mod store_types;

pub use store_impl::*;
pub use store_index::*;
pub use store_params::{
    CompressionAlgorithm, ReplicationMode, StoreParams, StoreParamsBuilder, StorePresets,
};
pub use store_quad::{GraphName, GraphTarget, QuadResult, QuadTermIter};
pub use store_stream::TripleTermIter;
pub use store_types::*;
