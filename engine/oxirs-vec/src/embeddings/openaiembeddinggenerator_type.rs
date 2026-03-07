//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CachedEmbedding, EmbeddingConfig, OpenAIConfig, OpenAIMetrics, RateLimiter};

/// OpenAI embeddings generator with rate limiting and retry logic
pub struct OpenAIEmbeddingGenerator {
    pub(super) config: EmbeddingConfig,
    pub(super) openai_config: OpenAIConfig,
    pub(super) client: reqwest::Client,
    pub(super) rate_limiter: RateLimiter,
    pub(super) request_cache: std::sync::Arc<std::sync::Mutex<lru::LruCache<u64, CachedEmbedding>>>,
    pub(super) metrics: OpenAIMetrics,
}
