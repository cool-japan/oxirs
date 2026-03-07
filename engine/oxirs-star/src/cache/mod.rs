//! Cache modules for RDF-star query results.
//!
//! - [`pattern_cache`] – Statistical eviction cache for quoted triple
//!   pattern query results.

/// Quoted triple pattern cache with statistical eviction policy.
pub mod pattern_cache;
