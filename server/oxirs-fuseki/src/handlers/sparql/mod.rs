//! SPARQL handler module organization
//!
//! This module provides the main SPARQL protocol implementation
//! broken down into smaller, manageable components.

pub mod aggregation_engine;
pub mod bind_processor;
pub mod content_types;
pub mod core;
pub mod optimizers;
pub mod service_delegation;
pub mod sparql12_features;

// Re-export main items for backwards compatibility
pub use core::*;
pub use sparql12_features::{
    extract_quoted_triple_patterns, parse_quoted_triple_value, ParsedQuotedTriple, Sparql12Features,
};

// Re-export handlers from sparql_refactored for server routing
pub use crate::handlers::sparql_refactored::{query_handler, update_handler};
