//! LLM Provider Implementations
//!
//! Contains provider trait definition and implementations for various LLM services
//! including OpenAI, Anthropic, and local models.

use anyhow::Result;
use async_trait::async_trait;

use super::types::{LLMRequest, LLMResponse, LLMResponseStream};

/// Core trait for LLM providers
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse>;
    async fn generate_stream(&self, model: &str, request: &LLMRequest)
        -> Result<LLMResponseStream>;
    fn get_available_models(&self) -> Vec<String>;
    fn supports_streaming(&self) -> bool;
    fn get_provider_name(&self) -> &str;
    fn estimate_cost(&self, model: &str, input_tokens: usize, output_tokens: usize) -> f64;
}
