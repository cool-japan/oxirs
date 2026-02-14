//! # OpenAIConfig - Trait Implementations
//!
//! This module contains trait implementations for `OpenAIConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{OpenAIConfig, RetryStrategy};

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: "text-embedding-3-small".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            timeout_seconds: 30,
            requests_per_minute: 3000,
            batch_size: 100,
            enable_cache: true,
            cache_size: 10000,
            cache_ttl_seconds: 3600,
            max_retries: 3,
            retry_delay_ms: 1000,
            retry_strategy: RetryStrategy::ExponentialBackoff,
            track_costs: true,
            enable_metrics: true,
            user_agent: "oxirs-vec/0.1.0".to_string(),
        }
    }
}
