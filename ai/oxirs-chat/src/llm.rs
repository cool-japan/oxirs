//! LLM Integration Module for OxiRS Chat
//!
//! Provides unified interface for multiple LLM providers including OpenAI, Anthropic Claude,
//! and local models with intelligent routing and fallback strategies.

use anyhow::{anyhow, Result};
use async_openai::{
    types::{
        ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
        Role,
    },
    Client as OpenAIClient,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// LLM Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub providers: HashMap<String, ProviderConfig>,
    pub routing: RoutingConfig,
    pub fallback: FallbackConfig,
    pub rate_limits: RateLimitConfig,
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

/// LLM request context
#[derive(Debug, Clone)]
pub struct LLMRequest {
    pub messages: Vec<ChatMessage>,
    pub system_prompt: Option<String>,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    pub use_case: UseCase,
    pub priority: Priority,
    pub timeout: Option<Duration>,
}

/// Chat message for LLM interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// Use case classification for intelligent routing
#[derive(Debug, Clone, PartialEq)]
pub enum UseCase {
    SimpleQuery,
    ComplexReasoning,
    SparqlGeneration,
    KnowledgeExtraction,
    Conversation,
    Analysis,
    CodeGeneration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

/// LLM response
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub content: String,
    pub model_used: String,
    pub provider_used: String,
    pub usage: Usage,
    pub latency: Duration,
    pub quality_score: Option<f32>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Token usage information
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub cost: f64,
}

/// Main LLM manager
pub struct LLMManager {
    config: LLMConfig,
    openai_client: Option<OpenAIClient>,
    providers: HashMap<String, Box<dyn LLMProvider + Send + Sync>>,
    usage_tracker: UsageTracker,
}

impl LLMManager {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let mut manager = Self {
            openai_client: Self::create_openai_client(&config)?,
            providers: HashMap::new(),
            usage_tracker: UsageTracker::new(),
            config,
        };

        manager.initialize_providers()?;
        Ok(manager)
    }

    fn create_openai_client(config: &LLMConfig) -> Result<Option<OpenAIClient>> {
        if let Some(openai_config) = config.providers.get("openai") {
            if openai_config.enabled && openai_config.api_key.is_some() {
                let client = OpenAIClient::new();
                return Ok(Some(client));
            }
        }
        Ok(None)
    }

    fn initialize_providers(&mut self) -> Result<()> {
        // Initialize OpenAI provider
        if let Some(config) = self.config.providers.get("openai") {
            if config.enabled {
                let provider = Box::new(OpenAIProvider::new(config.clone())?);
                self.providers.insert("openai".to_string(), provider);
            }
        }

        // Initialize Anthropic provider
        if let Some(config) = self.config.providers.get("anthropic") {
            if config.enabled {
                let provider = Box::new(AnthropicProvider::new(config.clone())?);
                self.providers.insert("anthropic".to_string(), provider);
            }
        }

        // TODO: Initialize local model providers

        Ok(())
    }

    /// Generate response using intelligent routing
    pub async fn generate_response(&mut self, request: LLMRequest) -> Result<LLMResponse> {
        let start_time = Instant::now();

        // Select the best provider and model
        let (provider_name, model_name) = self.select_provider_and_model(&request)?;

        debug!(
            "Selected provider: {}, model: {}",
            provider_name, model_name
        );

        // Attempt generation with fallback
        let mut attempts = 0;
        let max_attempts = self.config.fallback.max_attempts;

        while attempts < max_attempts {
            match self
                .try_generate(&provider_name, &model_name, &request)
                .await
            {
                Ok(mut response) => {
                    response.latency = start_time.elapsed();
                    self.usage_tracker.record_usage(&response);
                    return Ok(response);
                }
                Err(e) => {
                    attempts += 1;
                    warn!("Attempt {} failed: {}", attempts, e);

                    if attempts < max_attempts {
                        // Try fallback provider/model
                        if let Ok((fallback_provider, fallback_model)) =
                            self.select_fallback_provider(&provider_name, &model_name)
                        {
                            continue;
                        }
                    }

                    if attempts == max_attempts {
                        error!("All attempts failed, returning error");
                        return Err(e);
                    }
                }
            }
        }

        Err(anyhow!(
            "Failed to generate response after {} attempts",
            max_attempts
        ))
    }

    async fn try_generate(
        &self,
        provider_name: &str,
        model_name: &str,
        request: &LLMRequest,
    ) -> Result<LLMResponse> {
        if let Some(provider) = self.providers.get(provider_name) {
            let timeout_duration = request.timeout.unwrap_or(Duration::from_secs(30));

            timeout(timeout_duration, provider.generate(model_name, request))
                .await
                .map_err(|_| anyhow!("Request timed out"))?
        } else {
            Err(anyhow!("Provider {} not found", provider_name))
        }
    }

    fn select_provider_and_model(&self, request: &LLMRequest) -> Result<(String, String)> {
        // Implement intelligent routing based on use case and configuration
        match request.use_case {
            UseCase::SimpleQuery | UseCase::Conversation => {
                // Use fast, cost-effective model
                if self.providers.contains_key("openai") {
                    Ok(("openai".to_string(), "gpt-3.5-turbo".to_string()))
                } else {
                    Err(anyhow!("No suitable provider found"))
                }
            }
            UseCase::ComplexReasoning | UseCase::Analysis => {
                // Use high-quality model
                if self.providers.contains_key("openai") {
                    Ok(("openai".to_string(), "gpt-4".to_string()))
                } else if self.providers.contains_key("anthropic") {
                    Ok(("anthropic".to_string(), "claude-3-opus".to_string()))
                } else {
                    Err(anyhow!("No suitable provider found"))
                }
            }
            UseCase::SparqlGeneration | UseCase::CodeGeneration => {
                // Use code-specialized model
                if self.providers.contains_key("openai") {
                    Ok(("openai".to_string(), "gpt-4".to_string()))
                } else {
                    Err(anyhow!("No suitable provider found"))
                }
            }
            UseCase::KnowledgeExtraction => {
                // Use reasoning-optimized model
                if self.providers.contains_key("anthropic") {
                    Ok(("anthropic".to_string(), "claude-3-sonnet".to_string()))
                } else if self.providers.contains_key("openai") {
                    Ok(("openai".to_string(), "gpt-4".to_string()))
                } else {
                    Err(anyhow!("No suitable provider found"))
                }
            }
        }
    }

    fn select_fallback_provider(
        &self,
        failed_provider: &str,
        _failed_model: &str,
    ) -> Result<(String, String)> {
        // Simple fallback logic - use different provider
        for (provider_name, _) in &self.providers {
            if provider_name != failed_provider {
                // Return first available alternative
                if provider_name == "openai" {
                    return Ok((provider_name.clone(), "gpt-3.5-turbo".to_string()));
                } else if provider_name == "anthropic" {
                    return Ok((provider_name.clone(), "claude-3-sonnet".to_string()));
                }
            }
        }
        Err(anyhow!("No fallback provider available"))
    }

    /// Get usage statistics
    pub fn get_usage_stats(&self) -> UsageStats {
        self.usage_tracker.get_stats()
    }
}

/// Provider trait for implementing different LLM providers
#[async_trait::async_trait]
pub trait LLMProvider {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse>;
    fn get_available_models(&self) -> Vec<String>;
    fn supports_streaming(&self) -> bool;
}

/// OpenAI provider implementation
pub struct OpenAIProvider {
    client: OpenAIClient,
    config: ProviderConfig,
}

impl OpenAIProvider {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let client = OpenAIClient::new();
        Ok(Self { client, config })
    }
}

#[async_trait::async_trait]
impl LLMProvider for OpenAIProvider {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse> {
        use async_openai::types::{
            ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
        };

        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();

        // Add system message if provided
        if let Some(system_prompt) = &request.system_prompt {
            messages.push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: system_prompt.clone(),
                    name: None,
                },
            ));
        }

        // Add user messages
        for msg in &request.messages {
            match msg.role {
                ChatRole::System => {
                    messages.push(ChatCompletionRequestMessage::System(
                        ChatCompletionRequestSystemMessage {
                            content: msg.content.clone(),
                            name: None,
                        },
                    ));
                }
                ChatRole::User => {
                    messages.push(ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessage {
                            content: msg.content.clone().into(),
                            name: None,
                        },
                    ));
                }
                ChatRole::Assistant => {
                    // Handle assistant messages - simplified for now
                    continue;
                }
            }
        }

        let openai_request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .messages(messages)
            .temperature(request.temperature)
            .max_tokens(request.max_tokens.unwrap_or(1000) as u16)
            .build()?;

        let response = self
            .client
            .chat()
            .completions()
            .create(openai_request)
            .await?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow!("No response choices received"))?;

        let content = choice
            .message
            .content
            .clone()
            .unwrap_or_else(|| "No content received".to_string());

        let usage = response
            .usage
            .map(|u| Usage {
                prompt_tokens: u.prompt_tokens as usize,
                completion_tokens: u.completion_tokens as usize,
                total_tokens: u.total_tokens as usize,
                cost: (u.total_tokens as f64) * 0.000002, // Approximate cost
            })
            .unwrap_or(Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                cost: 0.0,
            });

        Ok(LLMResponse {
            content,
            model_used: model.to_string(),
            provider_used: "openai".to_string(),
            usage,
            latency: Duration::from_secs(0), // Will be set by caller
            quality_score: None,
            metadata: HashMap::new(),
        })
    }

    fn get_available_models(&self) -> Vec<String> {
        self.config.models.iter().map(|m| m.name.clone()).collect()
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

/// Usage tracking
pub struct UsageTracker {
    stats: UsageStats,
}

impl UsageTracker {
    pub fn new() -> Self {
        Self {
            stats: UsageStats::default(),
        }
    }

    pub fn record_usage(&mut self, response: &LLMResponse) {
        self.stats.total_requests += 1;
        self.stats.total_tokens += response.usage.total_tokens;
        self.stats.total_cost += response.usage.cost;
        self.stats.average_latency = Duration::from_millis(
            ((self.stats.average_latency.as_millis() * (self.stats.total_requests - 1) as u128
                + response.latency.as_millis())
                / self.stats.total_requests as u128) as u64,
        );
    }

    pub fn get_stats(&self) -> UsageStats {
        self.stats.clone()
    }
}

#[derive(Debug, Clone, Default)]
pub struct UsageStats {
    pub total_requests: usize,
    pub total_tokens: usize,
    pub total_cost: f64,
    pub average_latency: Duration,
    pub success_rate: f32,
}

/// Anthropic Claude provider implementation
pub struct AnthropicProvider {
    client: reqwest::Client,
    config: ProviderConfig,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let client = reqwest::Client::builder().timeout(config.timeout).build()?;
        Ok(Self { client, config })
    }

    /// Build Anthropic API request
    fn build_anthropic_request(
        &self,
        model: &str,
        request: &LLMRequest,
    ) -> Result<serde_json::Value> {
        let mut messages = Vec::new();

        // Add system message if provided
        if let Some(ref system_prompt) = request.system_prompt {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system_prompt
            }));
        }

        // Add conversation messages
        for msg in &request.messages {
            let role = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            messages.push(serde_json::json!({
                "role": role,
                "content": msg.content
            }));
        }

        let anthropic_request = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(1000),
            "temperature": request.temperature,
        });

        Ok(anthropic_request)
    }
}

#[async_trait::async_trait]
impl LLMProvider for AnthropicProvider {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow!("Anthropic API key not configured"))?;

        let request_body = self.build_anthropic_request(model, request)?;

        let base_url = self
            .config
            .base_url
            .as_ref()
            .map(|url| url.as_str())
            .unwrap_or("https://api.anthropic.com");

        let response = self
            .client
            .post(&format!("{}/v1/messages", base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!("Anthropic API error: {}", error_text));
        }

        let response_data: serde_json::Value = response.json().await?;

        // Extract content from Anthropic response format
        let content = response_data["content"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|content| content["text"].as_str())
            .unwrap_or("No response content")
            .to_string();

        // Extract usage information if available
        let usage = if let Some(usage_data) = response_data.get("usage") {
            Usage {
                prompt_tokens: usage_data["input_tokens"].as_u64().unwrap_or(0) as usize,
                completion_tokens: usage_data["output_tokens"].as_u64().unwrap_or(0) as usize,
                total_tokens: (usage_data["input_tokens"].as_u64().unwrap_or(0)
                    + usage_data["output_tokens"].as_u64().unwrap_or(0))
                    as usize,
                cost: 0.0, // TODO: Calculate actual cost based on model
            }
        } else {
            Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                cost: 0.0,
            }
        };

        Ok(LLMResponse {
            content,
            model_used: model.to_string(),
            provider_used: "anthropic".to_string(),
            usage,
            latency: Duration::from_secs(0), // Will be set by caller
            quality_score: None,
            metadata: HashMap::new(),
        })
    }

    fn get_available_models(&self) -> Vec<String> {
        self.config.models.iter().map(|m| m.name.clone()).collect()
    }

    fn supports_streaming(&self) -> bool {
        true // Anthropic supports streaming
    }
}

/// Rate limiter for LLM providers
pub struct RateLimiter {
    // TODO: Implement rate limiting using governor crate
}

impl RateLimiter {
    pub fn new(_config: &RateLimitConfig) -> Self {
        Self {}
    }

    pub async fn check_rate_limit(&self, _provider: &str) -> Result<()> {
        // TODO: Implement actual rate limiting
        Ok(())
    }
}

/// Enhanced LLM manager with rate limiting and monitoring
pub struct EnhancedLLMManager {
    inner: LLMManager,
    rate_limiter: RateLimiter,
    // TODO: Add monitoring and metrics
}

impl EnhancedLLMManager {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let rate_limiter = RateLimiter::new(&config.rate_limits);
        let inner = LLMManager::new(config)?;

        Ok(Self {
            inner,
            rate_limiter,
        })
    }

    pub async fn generate_response_with_limits(
        &mut self,
        request: LLMRequest,
    ) -> Result<LLMResponse> {
        // Check rate limits before processing
        self.rate_limiter.check_rate_limit("default").await?;

        // Generate response using the inner manager
        self.inner.generate_response(request).await
    }
}
