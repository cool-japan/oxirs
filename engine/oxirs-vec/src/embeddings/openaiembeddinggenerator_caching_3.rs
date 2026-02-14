//! # OpenAIEmbeddingGenerator - caching Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Get total cache cost
    pub fn get_cache_cost(&self) -> f64 {
        match self.request_cache.lock() {
            Ok(cache) => cache.iter().map(|(_, cached)| cached.cost_usd).sum(),
            _ => 0.0,
        }
    }
}
