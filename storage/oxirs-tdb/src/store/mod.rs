//! High-level TDB store API
//!
//! Provides the main TDBStore interface for interacting with the storage engine.
//! Integrates all components: dictionary, indexes, transactions, compression.

pub mod store_impl;
pub mod store_index;
pub mod store_params;
mod store_tests;
pub mod store_types;

pub use store_impl::*;
pub use store_index::*;
pub use store_params::{
    CompressionAlgorithm, ReplicationMode, StoreParams, StoreParamsBuilder, StorePresets,
};
pub use store_types::*;
