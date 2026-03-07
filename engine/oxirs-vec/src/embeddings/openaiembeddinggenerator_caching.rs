//! # OpenAIEmbeddingGenerator - caching Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CachedEmbedding;

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Check if cached embedding is still valid
    fn is_cache_valid(&self, cached: &CachedEmbedding) -> bool {
        if self.openai_config.cache_ttl_seconds == 0 {
            return true;
        }
        let elapsed = cached
            .cached_at
            .elapsed()
            .unwrap_or(std::time::Duration::from_secs(u64::MAX));
        elapsed.as_secs() < self.openai_config.cache_ttl_seconds
    }
}
