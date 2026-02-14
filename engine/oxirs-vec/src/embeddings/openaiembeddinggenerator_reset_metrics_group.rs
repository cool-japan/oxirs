//! # OpenAIEmbeddingGenerator - reset_metrics_group Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::OpenAIMetrics;

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = OpenAIMetrics::default();
    }
}
