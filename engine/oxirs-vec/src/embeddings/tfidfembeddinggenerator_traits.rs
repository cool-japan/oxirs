//! # TfIdfEmbeddingGenerator - Trait Implementations
//!
//! This module contains trait implementations for `TfIdfEmbeddingGenerator`.
//!
//! ## Implemented Traits
//!
//! - `EmbeddingGenerator`
//! - `AsAny`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Vector;
use anyhow::{anyhow, Result};

use super::functions::{AsAny, EmbeddingGenerator};
use super::types::{EmbeddableContent, EmbeddingConfig, TfIdfEmbeddingGenerator};

impl EmbeddingGenerator for TfIdfEmbeddingGenerator {
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector> {
        if self.vocabulary.is_empty() {
            return Err(anyhow!(
                "Vocabulary not built. Call build_vocabulary first."
            ));
        }
        let text = content.to_text();
        Ok(self.calculate_tf_idf(&text))
    }
    fn dimensions(&self) -> usize {
        self.config.dimensions
    }
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

impl AsAny for TfIdfEmbeddingGenerator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
