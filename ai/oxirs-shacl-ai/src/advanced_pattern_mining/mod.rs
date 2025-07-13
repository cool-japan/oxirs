//! Advanced Pattern Mining Engine for SHACL AI
//!
//! This module implements state-of-the-art pattern mining algorithms for improved
//! constraint discovery and shape learning performance.

pub mod algorithms;
pub mod analytics;
pub mod cache;
pub mod engine;
pub mod patterns;
pub mod sparql;
pub mod types;

// Re-export main types for convenience
pub use algorithms::*;
pub use analytics::*;
pub use cache::*;
pub use engine::*;
pub use patterns::*;
pub use sparql::*;
pub use types::*;
