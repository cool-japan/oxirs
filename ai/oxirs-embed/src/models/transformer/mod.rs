//! Transformer-based embedding models module
//!
//! This module provides a modular implementation of transformer-based models
//! for generating embeddings including BERT, RoBERTa, Sentence-BERT, and
//! domain-specific variants.

pub mod preprocessing;
pub mod training;
pub mod types;

// Re-export commonly used types and functions
pub use preprocessing::*;
pub use training::*;
pub use types::*;