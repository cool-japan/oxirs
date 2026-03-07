//! # OpenAIEmbeddingGenerator - update_metrics_failure_group Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Update metrics after failed request
    pub(super) fn update_metrics_failure(&mut self) {
        self.metrics.total_requests += 1;
        self.metrics.failed_requests += 1;
    }
}
