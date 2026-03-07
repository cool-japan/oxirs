//! # OpenAIEmbeddingGenerator - estimate_tokens_group Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Estimate token count for text (approximate)
    fn estimate_tokens(&self, text: &str) -> u64 {
        (text.len() / 4).max(1) as u64
    }
    /// Update metrics after successful request
    pub(super) fn update_metrics_success(&mut self, texts: &[String]) {
        self.metrics.total_requests += 1;
        self.metrics.successful_requests += 1;
        let total_tokens: u64 = texts.iter().map(|text| self.estimate_tokens(text)).sum();
        self.metrics.total_tokens_processed += total_tokens;
        self.metrics.total_cost_usd += self.calculate_cost_from_tokens(total_tokens);
    }
}
