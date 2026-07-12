//! Indexing structures and query helpers for the TDB store.
//!
//! This module re-exports index access helpers from [`store_impl`](crate::store::store_impl) and provides
//! convenience wrappers around B-tree page scanning and range queries used
//! internally by the store.

// The actual indexing structures (TripleIndexes, SPO/POS/OSP B-tree pages)
// live in crate::index. This file serves as the index-layer facade within
// the store module and can be extended with store-specific index helpers.

// Public re-export to keep the facade clean.
pub use crate::index::{Triple, TripleIndexes};
