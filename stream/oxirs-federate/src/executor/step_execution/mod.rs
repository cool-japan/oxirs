//! Individual step execution functions for federated queries
//!
//! This module contains the implementation of individual step execution functions
//! including service queries, GraphQL queries, joins, filters, aggregation, sorting,
//! entity resolution, and result stitching.

pub mod execution;
pub mod sorting;
pub mod aggregation;
pub mod result_processing;
pub mod joins;
pub mod entity_resolution;
pub mod stitching;

// Re-export main execution functions
pub use execution::*;
pub use sorting::*;
pub use aggregation::*;
pub use result_processing::*;
pub use joins::*;
pub use entity_resolution::*;
pub use stitching::*;
