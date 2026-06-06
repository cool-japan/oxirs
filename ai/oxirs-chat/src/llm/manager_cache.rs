//! LLM Manager Cache
//!
//! Response caching: semantic cache integration, TTL management, cache invalidation,
//! token budget enforcement, and usage recording.

use anyhow::Result;
use tracing::debug;

use super::{
    cache::ResponseCache,
    token_budget::TokenBudget,
    types::{LLMRequest, LLMResponse},
};

/// Check whether a request is already cached and return the cached response.
pub async fn cache_get(cache: &ResponseCache, request: &LLMRequest) -> Option<LLMResponse> {
    let cached = cache.get(request).await;
    if cached.is_some() {
        debug!("Cache hit for request");
    }
    cached
}

/// Store a response in the cache, keyed by the originating provider.
pub async fn cache_put(
    cache: &ResponseCache,
    request: &LLMRequest,
    response: LLMResponse,
    provider_name: String,
) {
    cache.put(request, response, provider_name).await;
}

/// Verify that the user has sufficient token budget before executing a request.
///
/// Returns `Err` when the budget is exhausted.
pub async fn check_budget(
    token_budget: &TokenBudget,
    user_id: &str,
    estimated_tokens: u64,
) -> Result<()> {
    let user_id_owned = user_id.to_string();
    token_budget
        .check_budget(&user_id_owned, estimated_tokens)
        .await
}

/// Record actual token usage after a successful response.
pub async fn record_usage(
    token_budget: &TokenBudget,
    user_id: &str,
    actual_tokens: u64,
) -> Result<()> {
    let user_id_owned = user_id.to_string();
    token_budget
        .record_usage(&user_id_owned, actual_tokens)
        .await
}

/// Estimate the number of input tokens for a request.
///
/// Uses a simple heuristic (1 token ≈ 4 characters) that callers can refine.
pub fn estimate_input_tokens(request: &LLMRequest) -> usize {
    let total_content: String = request
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .chain(request.system_prompt.as_deref())
        .collect::<Vec<_>>()
        .join(" ");

    // Rough estimate: 1 token ≈ 4 characters
    total_content.len() / 4
}
