//! # OpenAIEmbeddingGenerator - caching Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Update cache metrics
    pub(super) fn update_cache_hit(&mut self) {
        self.metrics.cache_hits += 1;
    }
}
