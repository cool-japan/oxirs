//! # SentenceTransformerGenerator - Trait Implementations
//!
//! This module contains trait implementations for `SentenceTransformerGenerator`.
//!
//! ## Implemented Traits
//!
//! - `EmbeddingGenerator`
//! - `AsAny`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Vector;
use anyhow::Result;

use super::functions::{AsAny, EmbeddingGenerator};
use super::sentencetransformergenerator_type::SentenceTransformerGenerator;
use super::types::{EmbeddableContent, EmbeddingConfig};

impl EmbeddingGenerator for SentenceTransformerGenerator {
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector> {
        let text = content.to_text();
        self.generate_with_model(&text)
    }
    fn dimensions(&self) -> usize {
        self.config.dimensions
    }
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

impl AsAny for SentenceTransformerGenerator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
