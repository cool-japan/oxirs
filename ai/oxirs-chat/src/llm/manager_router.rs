//! LLM Manager Router
//!
//! Model routing logic: provider selection, load balancing, fallback chains,
//! circuit breaker integration, and health-based routing.

use anyhow::Result;
use std::{collections::HashMap, time::Duration};
use tracing::{debug, error, warn};

use super::{
    circuit_breaker::CircuitBreaker,
    config::LLMConfig,
    health_checker::HealthChecker,
    providers::LLMProvider,
    types::{LLMRequest, LLMResponse},
    CircuitBreakerStats,
};

/// Provider routing context: an ordered list of (provider_name, model_name) pairs
pub type FallbackChain = Vec<(String, String)>;

/// Build an ordered fallback chain based on configuration.
///
/// Priority order: openai → anthropic → local
pub fn build_fallback_chain(
    config: &LLMConfig,
    active_providers: &HashMap<String, Box<dyn LLMProvider + Send + Sync>>,
) -> FallbackChain {
    let priority_order = ["openai", "anthropic", "local"];
    let mut chain = Vec::new();

    for provider_name in priority_order {
        if let Some(provider_config) = config.providers.get(provider_name) {
            if provider_config.enabled && active_providers.contains_key(provider_name) {
                if let Some(model) = provider_config.models.first() {
                    chain.push((provider_name.to_string(), model.name.clone()));
                }
            }
        }
    }

    chain
}

/// Attempt a single provider call.
pub async fn try_provider(
    providers: &HashMap<String, Box<dyn LLMProvider + Send + Sync>>,
    provider_name: &str,
    model_name: &str,
    request: &LLMRequest,
) -> Result<LLMResponse> {
    let provider = providers
        .get(provider_name)
        .ok_or_else(|| anyhow::anyhow!("Provider {} not found", provider_name))?;

    provider.generate(model_name, request).await
}

/// Record a successful provider call against circuit breaker and health checker.
pub async fn record_success(
    circuit_breakers: &HashMap<String, std::sync::Arc<CircuitBreaker>>,
    health_checker: &HealthChecker,
    provider_name: &str,
    latency: Duration,
) {
    if let Some(cb) = circuit_breakers.get(provider_name) {
        cb.record_result(true, latency).await;
    }

    let provider_id = provider_name.to_string();
    if let Err(e) = health_checker
        .record_call(&provider_id, true, latency)
        .await
    {
        error!("Failed to record success for {}: {}", provider_name, e);
    }
}

/// Record a failed provider call against circuit breaker and health checker.
pub async fn record_failure(
    circuit_breakers: &HashMap<String, std::sync::Arc<CircuitBreaker>>,
    health_checker: &HealthChecker,
    provider_name: &str,
    latency: Duration,
) {
    if let Some(cb) = circuit_breakers.get(provider_name) {
        cb.record_result(false, latency).await;
    }

    let provider_id = provider_name.to_string();
    if let Err(e) = health_checker
        .record_call(&provider_id, false, latency)
        .await
    {
        error!("Failed to record failure for {}: {}", provider_name, e);
    }
}

/// Check whether a provider is allowed to execute (circuit breaker + health check).
///
/// Returns `true` when execution is allowed.
pub async fn provider_is_allowed(
    circuit_breakers: &HashMap<String, std::sync::Arc<CircuitBreaker>>,
    health_checker: &HealthChecker,
    provider_name: &str,
) -> bool {
    if let Some(cb) = circuit_breakers.get(provider_name) {
        if !cb.can_execute().await {
            warn!("Circuit breaker is open for provider: {}", provider_name);
            return false;
        }
    }

    let provider_id = provider_name.to_string();
    if !health_checker.is_provider_healthy(&provider_id).await {
        warn!("Provider {} is unhealthy, skipping", provider_name);
        return false;
    }

    true
}

/// Collect circuit-breaker statistics for all tracked providers.
pub async fn collect_circuit_breaker_stats(
    circuit_breakers: &HashMap<String, std::sync::Arc<CircuitBreaker>>,
) -> HashMap<String, CircuitBreakerStats> {
    let mut stats = HashMap::new();
    for (name, cb) in circuit_breakers {
        stats.insert(name.clone(), cb.get_stats().await);
    }
    stats
}

/// Log fast-failover diagnostics when elapsed time is below the threshold.
pub fn log_fast_failover(elapsed: Duration) {
    if elapsed.as_millis() < 500 {
        debug!("Fast failover to next provider ({}ms)", elapsed.as_millis());
    }
}
