//! Neural-Symbolic Bridge Module
//!
//! This module provides advanced integration between neural (vector) and symbolic (RDF/SPARQL)
//! reasoning, enabling hybrid AI queries that combine the best of both paradigms.
//!
//! The module has been refactored from a single large file into smaller, focused modules
//! to comply with the 2000-line refactoring policy.

pub mod types;

// Re-export all types for convenience
pub use types::*;

// TODO: Add other modules as they are extracted:
// pub mod bridge;      // Main NeuralSymbolicBridge implementation
// pub mod config;      // Configuration management
// pub mod metrics;     // Performance monitoring
// pub mod temporal;    // Temporal reasoning functionality