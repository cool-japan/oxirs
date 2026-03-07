//! # OpenAIEmbeddingGenerator - caching Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Clear the request cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.request_cache.lock() {
            cache.clear();
        }
    }
}
