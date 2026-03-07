//! # OpenAIEmbeddingGenerator - caching Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, Option<usize>) {
        match self.request_cache.lock() {
            Ok(cache) => (cache.len(), Some(cache.cap().into())),
            _ => (0, None),
        }
    }
}
