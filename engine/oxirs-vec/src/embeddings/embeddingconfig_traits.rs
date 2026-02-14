//! # EmbeddingConfig - Trait Implementations
//!
//! This module contains trait implementations for `EmbeddingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::EmbeddingConfig;

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimensions: 384,
            max_sequence_length: 512,
            normalize: true,
        }
    }
}
