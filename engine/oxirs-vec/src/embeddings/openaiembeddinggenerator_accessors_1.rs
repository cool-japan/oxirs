//! # OpenAIEmbeddingGenerator - accessors Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::OpenAIMetrics;

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Get API usage metrics
    pub fn get_metrics(&self) -> &OpenAIMetrics {
        &self.metrics
    }
}
