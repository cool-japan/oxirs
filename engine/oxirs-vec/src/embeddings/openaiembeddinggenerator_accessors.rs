//! # OpenAIEmbeddingGenerator - accessors Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Get cost per 1k tokens for different models (in USD)
    fn get_model_cost_per_1k_tokens(model: &str) -> f64 {
        match model {
            "text-embedding-ada-002" => 0.0001,
            "text-embedding-3-small" => 0.00002,
            "text-embedding-3-large" => 0.00013,
            "text-embedding-004" => 0.00002,
            _ => 0.0001,
        }
    }
    /// Calculate cost for processing texts
    pub(super) fn calculate_cost(&self, texts: &[String]) -> f64 {
        if !self.openai_config.track_costs {
            return 0.0;
        }
        let total_tokens: usize = texts.iter().map(|t| t.len() / 4).sum();
        let cost_per_1k = Self::get_model_cost_per_1k_tokens(&self.openai_config.model);
        (total_tokens as f64 / 1000.0) * cost_per_1k
    }
}
