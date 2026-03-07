//! SPARQL-star query processing modules.
//!
//! This module provides the complete SPARQL-star query engine, combining the
//! original BGP executor with new high-performance extensions:
//!
//! - [`legacy`] – original SPARQL-star query executor (BGP evaluation, filters,
//!   SPARQL-star functions, query parser)
//! - [`parallel`] – parallel BGP evaluation using Rayon for large datasets
//! - [`streaming`] – lazy, iterator-based streaming of query results

/// Original SPARQL-star query executor.
pub mod legacy;

/// Parallel Basic Graph Pattern (BGP) evaluation for SPARQL-star.
pub mod parallel;

/// Streaming SPARQL-star query result iteration.
pub mod streaming;

// Re-export the original query types at the module level for backward
// compatibility with code that previously used `crate::query::*`.
pub use legacy::{
    BasicGraphPattern, Binding, QueryExecutor, QueryParser, QueryType, TermPattern, TriplePattern,
};
