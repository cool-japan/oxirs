//! Neural Pattern Recognition for Advanced SHACL Shape Learning
//!
//! This module implements advanced neural pattern recognition using deep learning
//! to discover complex patterns in RDF data for intelligent SHACL shape generation.

pub mod attention;
pub mod correlation;
pub mod hierarchies;
pub mod learning;
pub mod recognizer;
pub mod types;

// Re-export main types and functions
pub use attention::*;
pub use correlation::*;
pub use hierarchies::*;
pub use learning::*;
pub use recognizer::*;
pub use types::*;
