//! # OpenAIEmbeddingGenerator - calculate_cost_from_tokens_group Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Calculate cost for embeddings request
    pub(super) fn calculate_cost_from_tokens(&self, total_tokens: u64) -> f64 {
        let cost_per_1k_tokens = match self.openai_config.model.as_str() {
            "text-embedding-ada-002" => 0.0001,
            "text-embedding-3-small" => 0.00002,
            "text-embedding-3-large" => 0.00013,
            _ => 0.0001,
        };
        (total_tokens as f64 / 1000.0) * cost_per_1k_tokens
    }
}
