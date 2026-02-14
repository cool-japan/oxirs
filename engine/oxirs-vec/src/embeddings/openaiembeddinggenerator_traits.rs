//! # OpenAIEmbeddingGenerator - Trait Implementations
//!
//! This module contains trait implementations for `OpenAIEmbeddingGenerator`.
//!
//! ## Implemented Traits
//!
//! - `EmbeddingGenerator`
//! - `AsAny`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Vector;
use anyhow::{anyhow, Result};

use super::functions::{AsAny, EmbeddingGenerator};
use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;
use super::types::{EmbeddableContent, EmbeddingConfig, RateLimiter};

impl EmbeddingGenerator for OpenAIEmbeddingGenerator {
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector> {
        if self.openai_config.enable_cache {
            let hash = content.content_hash();
            if let Ok(mut cache) = self.request_cache.lock() {
                if let Some(cached) = cache.get(&hash) {
                    return Ok(cached.vector.clone());
                }
            }
        }
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow!("Failed to create async runtime: {}", e))?;
        let mut temp_generator = OpenAIEmbeddingGenerator {
            config: self.config.clone(),
            openai_config: self.openai_config.clone(),
            client: self.client.clone(),
            rate_limiter: RateLimiter::new(self.openai_config.requests_per_minute),
            request_cache: self.request_cache.clone(),
            metrics: self.metrics.clone(),
        };
        rt.block_on(temp_generator.generate_async(content))
    }
    fn dimensions(&self) -> usize {
        self.config.dimensions
    }
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

impl AsAny for OpenAIEmbeddingGenerator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
