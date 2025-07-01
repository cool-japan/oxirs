//! LLM Configuration Module
//!
//! Contains all configuration structures for LLM providers, routing, fallback,
//! rate limiting, and circuit breaker functionality.

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{atomic::AtomicU64, Arc},
    time::{Duration, Instant},
};
use tokio::sync::RwLock;

/// LLM Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub providers: HashMap<String, ProviderConfig>,
    pub routing: RoutingConfig,
    pub fallback: FallbackConfig,
    pub rate_limits: RateLimitConfig,
    pub circuit_breaker: CircuitBreakerConfig,
}

impl Default for LLMConfig {
    fn default() -> Self {
        let mut providers = HashMap::new();
        providers.insert("openai".to_string(), ProviderConfig::openai_default());
        providers.insert("anthropic".to_string(), ProviderConfig::anthropic_default());

        Self {
            providers,
            routing: RoutingConfig::default(),
            fallback: FallbackConfig::default(),
            rate_limits: RateLimitConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

/// Provider-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub models: Vec<ModelConfig>,
    pub timeout: Duration,
    pub max_retries: usize,
}

impl ProviderConfig {
    pub fn openai_default() -> Self {
        Self {
            enabled: true,
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            base_url: None,
            models: vec![
                ModelConfig {
                    name: "gpt-4".to_string(),
                    max_tokens: 4096,
                    cost_per_token: 0.00003,
                    capabilities: vec!["reasoning".to_string(), "code".to_string()],
                    use_cases: vec!["complex_queries".to_string(), "analysis".to_string()],
                },
                ModelConfig {
                    name: "gpt-3.5-turbo".to_string(),
                    max_tokens: 4096,
                    cost_per_token: 0.000002,
                    capabilities: vec!["general".to_string(), "fast".to_string()],
                    use_cases: vec!["quick_responses".to_string(), "simple_queries".to_string()],
                },
            ],
            timeout: Duration::from_secs(30),
            max_retries: 3,
        }
    }

    pub fn anthropic_default() -> Self {
        Self {
            enabled: true,
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            base_url: Some("https://api.anthropic.com".to_string()),
            models: vec![
                ModelConfig {
                    name: "claude-3-opus".to_string(),
                    max_tokens: 4096,
                    cost_per_token: 0.000015,
                    capabilities: vec!["reasoning".to_string(), "analysis".to_string()],
                    use_cases: vec!["complex_analysis".to_string(), "research".to_string()],
                },
                ModelConfig {
                    name: "claude-3-sonnet".to_string(),
                    max_tokens: 4096,
                    cost_per_token: 0.000003,
                    capabilities: vec!["general".to_string(), "balanced".to_string()],
                    use_cases: vec!["general_chat".to_string(), "sparql_generation".to_string()],
                },
            ],
            timeout: Duration::from_secs(30),
            max_retries: 3,
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub max_tokens: usize,
    pub cost_per_token: f64,
    pub capabilities: Vec<String>,
    pub use_cases: Vec<String>,
}

/// Routing configuration for intelligent model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub strategy: RoutingStrategy,
    pub quality_threshold: f32,
    pub latency_threshold: Duration,
    pub cost_threshold: f64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::QualityFirst,
            quality_threshold: 0.8,
            latency_threshold: Duration::from_secs(5),
            cost_threshold: 0.01,
        }
    }
}

/// Routing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    QualityFirst,
    CostOptimized,
    LatencyOptimized,
    Balanced,
    RoundRobin,
}

/// Fallback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackConfig {
    pub enabled: bool,
    pub max_attempts: usize,
    pub backoff_strategy: BackoffStrategy,
    pub quality_degradation_allowed: bool,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 3,
            backoff_strategy: BackoffStrategy::Exponential,
            quality_degradation_allowed: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Linear,
    Exponential,
    Fixed(Duration),
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: usize,
    pub tokens_per_minute: usize,
    pub burst_allowed: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: 10000,
            burst_allowed: true,
        }
    }
}

/// Circuit breaker configuration for resilient LLM API calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit
    pub failure_threshold: usize,
    /// Duration to keep circuit open before attempting recovery
    pub timeout_duration: Duration,
    /// Number of successful requests required to close the circuit
    pub recovery_threshold: usize,
    /// Maximum response time considered acceptable
    pub slow_call_threshold: Duration,
    /// Percentage of slow calls that triggers circuit opening
    pub slow_call_rate_threshold: f32,
    /// Window size for calculating failure rates
    pub sliding_window_size: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout_duration: Duration::from_secs(60),
            recovery_threshold: 3,
            slow_call_threshold: Duration::from_secs(10),
            slow_call_rate_threshold: 0.5, // 50%
            sliding_window_size: 20,
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Failing fast
    HalfOpen, // Testing recovery
}