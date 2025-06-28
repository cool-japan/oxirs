//! LLM Integration Module for OxiRS Chat
//!
//! Provides unified interface for multiple LLM providers including OpenAI, Anthropic Claude,
//! and local models with intelligent routing and fallback strategies.

use anyhow::{anyhow, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, Role,
    },
    Client as OpenAIClient,
};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    time::{Duration, Instant},
};
use tokio::sync::Mutex as TokioMutex;
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

/// Streaming response chunk
#[derive(Debug, Clone)]
pub struct LLMResponseChunk {
    pub content: String,
    pub is_final: bool,
    pub chunk_index: usize,
    pub model_used: String,
    pub provider_used: String,
    pub latency: Duration,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Streaming response wrapper
pub struct LLMResponseStream {
    pub stream:
        std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<LLMResponseChunk>> + Send>>,
    pub model_used: String,
    pub provider_used: String,
    pub started_at: std::time::Instant,
}

/// Token usage information
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub cost: f64,
}

/// Routing candidate for model selection
#[derive(Debug, Clone)]
struct RoutingCandidate {
    provider: String,
    model: String,
    score: f32,
}

/// Main LLM manager
pub struct LLMManager {
    config: LLMConfig,
    openai_client: Option<OpenAIClient<OpenAIConfig>>,
    providers: HashMap<String, Box<dyn LLMProvider + Send + Sync>>,
    usage_tracker: TokioMutex<UsageTracker>,
}

impl LLMManager {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let mut manager = Self {
            openai_client: Self::create_openai_client(&config)?,
            providers: HashMap::new(),
            usage_tracker: TokioMutex::new(UsageTracker::new()),
            config,
        };

        manager.initialize_providers()?;
        Ok(manager)
    }

    fn create_openai_client(config: &LLMConfig) -> Result<Option<OpenAIClient<OpenAIConfig>>> {
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

        // Initialize local model provider
        if let Some(config) = self.config.providers.get("local") {
            if config.enabled {
                let provider = Box::new(LocalModelProvider::new(config.clone())?);
                self.providers.insert("local".to_string(), provider);
            }
        }

        Ok(())
    }

    /// Generate response using intelligent routing
    pub async fn generate_response(&self, request: LLMRequest) -> Result<LLMResponse> {
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
                    self.usage_tracker.lock().await.record_usage(&response);
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

    /// Generate streaming response using intelligent routing
    pub async fn generate_response_stream(&self, request: LLMRequest) -> Result<LLMResponseStream> {
        // Select the best provider and model
        let (provider_name, model_name) = self.select_provider_and_model(&request)?;

        debug!(
            "Selected provider: {} (streaming), model: {}",
            provider_name, model_name
        );

        // Check if provider supports streaming
        if let Some(provider) = self.providers.get(&provider_name) {
            if !provider.supports_streaming() {
                warn!(
                    "Provider {} doesn't support streaming, falling back to non-streaming",
                    provider_name
                );
                // Convert non-streaming response to streaming
                let response = self.generate_response(request).await?;
                return self.convert_to_stream(response);
            }
        }

        // Attempt streaming generation with fallback
        let mut attempts = 0;
        let max_attempts = self.config.fallback.max_attempts;

        while attempts < max_attempts {
            match self
                .try_generate_stream(&provider_name, &model_name, &request)
                .await
            {
                Ok(stream) => {
                    info!(
                        "Streaming response initiated with {} provider",
                        provider_name
                    );
                    return Ok(stream);
                }
                Err(e) => {
                    attempts += 1;
                    warn!("Streaming attempt {} failed: {}", attempts, e);

                    if attempts < max_attempts {
                        // Try fallback provider/model
                        if let Ok((fallback_provider, fallback_model)) =
                            self.select_fallback_provider(&provider_name, &model_name)
                        {
                            continue;
                        }
                    }

                    if attempts == max_attempts {
                        error!("All streaming attempts failed, returning error");
                        return Err(e);
                    }
                }
            }
        }

        Err(anyhow!(
            "Failed to generate streaming response after {} attempts",
            max_attempts
        ))
    }

    /// Convert a regular response to a streaming response
    fn convert_to_stream(&self, response: LLMResponse) -> Result<LLMResponseStream> {
        let words: Vec<String> = response
            .content
            .split_whitespace()
            .map(String::from)
            .collect();
        let chunk_size = 8; // Words per chunk
        let started_at = Instant::now();

        // Create chunks with owned data
        let chunks: Vec<Result<LLMResponseChunk>> = words
            .chunks(chunk_size)
            .enumerate()
            .map(|(index, chunk)| {
                let total_words = words.len();
                let is_final = (index + 1) * chunk_size >= total_words;
                Ok(LLMResponseChunk {
                    content: chunk.join(" ") + if !is_final { " " } else { "" },
                    is_final,
                    chunk_index: index,
                    model_used: response.model_used.clone(),
                    provider_used: response.provider_used.clone(),
                    latency: started_at.elapsed(),
                    metadata: HashMap::new(),
                })
            })
            .collect();

        // Create stream from owned chunks
        let stream = futures_util::stream::iter(chunks);

        Ok(LLMResponseStream {
            stream: Box::pin(stream),
            model_used: response.model_used,
            provider_used: response.provider_used,
            started_at,
        })
    }

    async fn try_generate_stream(
        &self,
        provider_name: &str,
        model_name: &str,
        request: &LLMRequest,
    ) -> Result<LLMResponseStream> {
        if let Some(provider) = self.providers.get(provider_name) {
            let timeout_duration = request.timeout.unwrap_or(Duration::from_secs(30));

            timeout(
                timeout_duration,
                provider.generate_stream(model_name, request),
            )
            .await
            .map_err(|_| anyhow!("Streaming request timed out"))?
        } else {
            Err(anyhow!("Provider {} not found", provider_name))
        }
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

    /// Enhanced provider and model selection with sophisticated routing
    fn select_provider_and_model(&self, request: &LLMRequest) -> Result<(String, String)> {
        // Calculate routing scores for all available providers/models
        let mut routing_candidates = Vec::new();

        for (provider_name, provider) in &self.providers {
            for model_name in provider.get_available_models() {
                let score = self.calculate_routing_score(provider_name, &model_name, request)?;
                routing_candidates.push(RoutingCandidate {
                    provider: provider_name.clone(),
                    model: model_name,
                    score,
                });
            }
        }

        // Sort by score and select the best candidate
        routing_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(best_candidate) = routing_candidates.first() {
            debug!(
                "Selected provider: {}, model: {}, score: {}",
                best_candidate.provider, best_candidate.model, best_candidate.score
            );
            Ok((
                best_candidate.provider.clone(),
                best_candidate.model.clone(),
            ))
        } else {
            Err(anyhow!("No suitable provider/model combination found"))
        }
    }

    /// Calculate comprehensive routing score for a provider/model combination
    fn calculate_routing_score(
        &self,
        provider_name: &str,
        model_name: &str,
        request: &LLMRequest,
    ) -> Result<f32> {
        let mut score = 0.0f32;

        // Get base scores from different factors
        let quality_score = self.get_quality_score(provider_name, model_name, &request.use_case);
        let cost_score = self.get_cost_efficiency_score(provider_name, model_name, request);
        let latency_score = self.get_latency_score(provider_name, model_name);
        let availability_score = self.get_availability_score(provider_name);
        let use_case_fit_score =
            self.get_use_case_fit_score(provider_name, model_name, &request.use_case);
        let priority_score = self.get_priority_score(&request.priority);

        // Apply routing strategy weights
        match self.config.routing.strategy {
            RoutingStrategy::QualityFirst => {
                score = quality_score * 0.6
                    + use_case_fit_score * 0.2
                    + availability_score * 0.1
                    + latency_score * 0.1;
            }
            RoutingStrategy::CostOptimized => {
                score = cost_score * 0.5
                    + quality_score * 0.2
                    + use_case_fit_score * 0.2
                    + availability_score * 0.1;
            }
            RoutingStrategy::LatencyOptimized => {
                score = latency_score * 0.5
                    + availability_score * 0.2
                    + quality_score * 0.2
                    + cost_score * 0.1;
            }
            RoutingStrategy::Balanced => {
                score = quality_score * 0.3
                    + cost_score * 0.25
                    + latency_score * 0.25
                    + use_case_fit_score * 0.15
                    + availability_score * 0.05;
            }
            RoutingStrategy::RoundRobin => {
                // For round-robin, use a base score and add randomness
                score = (quality_score + cost_score + latency_score) / 3.0;
                score += (fastrand::f32() - 0.5) * 0.2; // Add some randomness
            }
        }

        // Apply priority multiplier
        score *= priority_score;

        // Apply threshold filters
        if quality_score < self.config.routing.quality_threshold {
            score *= 0.1; // Heavily penalize low-quality options
        }

        Ok(score.max(0.0).min(1.0))
    }

    /// Get quality score for provider/model combination
    fn get_quality_score(&self, provider_name: &str, model_name: &str, use_case: &UseCase) -> f32 {
        let base_quality = match (provider_name, model_name) {
            ("openai", "gpt-4") | ("openai", "gpt-4-turbo") => 0.95f32,
            ("anthropic", "claude-3-opus-20240229") => 0.95f32,
            ("anthropic", "claude-3-sonnet-20240229") => 0.85f32,
            ("openai", "gpt-3.5-turbo") => 0.80f32,
            ("anthropic", "claude-3-haiku-20240307") => 0.75f32,
            ("local", "llama-2-13b") => 0.70f32,
            ("local", "mistral-7b") => 0.65f32,
            ("local", "codellama-7b") => 0.75f32, // Better for code
            _ => 0.50f32,                         // Default for unknown models
        };

        // Adjust for use case specific quality
        let use_case_adjustment = match use_case {
            UseCase::ComplexReasoning | UseCase::Analysis => {
                if model_name.contains("gpt-4") || model_name.contains("opus") {
                    1.1f32
                } else if model_name.contains("3.5") || model_name.contains("haiku") {
                    0.9f32
                } else {
                    1.0f32
                }
            }
            UseCase::SparqlGeneration | UseCase::CodeGeneration => {
                if model_name.contains("gpt-4") || model_name.contains("codellama") {
                    1.1f32
                } else if model_name.contains("sonnet") {
                    1.05f32
                } else {
                    1.0f32
                }
            }
            UseCase::SimpleQuery | UseCase::Conversation => {
                if model_name.contains("haiku") || model_name.contains("3.5") {
                    1.05f32 // These are optimized for fast, simple queries
                } else {
                    1.0f32
                }
            }
            _ => 1.0f32,
        };

        (base_quality * use_case_adjustment).min(1.0f32)
    }

    /// Get cost efficiency score (higher score = better cost efficiency)
    fn get_cost_efficiency_score(
        &self,
        provider_name: &str,
        model_name: &str,
        request: &LLMRequest,
    ) -> f32 {
        let estimated_input_tokens = self.estimate_input_tokens(request);
        let estimated_output_tokens = 500; // Rough estimate

        if let Some(provider) = self.providers.get(provider_name) {
            let estimated_cost =
                provider.estimate_cost(model_name, estimated_input_tokens, estimated_output_tokens);

            // Convert cost to efficiency score (lower cost = higher score)
            if estimated_cost <= 0.0 {
                return 1.0; // Free models get perfect efficiency score
            }

            // Normalize cost to 0-1 scale, where $0.10 = 0.0 score and $0.001 = 1.0 score
            let max_acceptable_cost = 0.10;
            let min_cost = 0.001;

            if estimated_cost >= max_acceptable_cost {
                0.0
            } else if estimated_cost <= min_cost {
                1.0
            } else {
                1.0 - ((estimated_cost - min_cost) / (max_acceptable_cost - min_cost)) as f32
            }
        } else {
            0.5 // Default score if provider not found
        }
    }

    /// Get latency score (higher score = lower expected latency)
    fn get_latency_score(&self, provider_name: &str, model_name: &str) -> f32 {
        match (provider_name, model_name) {
            ("local", _) => 0.95, // Local models are typically fastest
            ("openai", "gpt-3.5-turbo") => 0.85,
            ("anthropic", "claude-3-haiku-20240307") => 0.85,
            ("openai", "gpt-4-turbo") => 0.75,
            ("anthropic", "claude-3-sonnet-20240229") => 0.70,
            ("openai", "gpt-4") => 0.60,
            ("anthropic", "claude-3-opus-20240229") => 0.55,
            _ => 0.50,
        }
    }

    /// Get availability score based on current provider status
    fn get_availability_score(&self, provider_name: &str) -> f32 {
        // In a production system, this would check actual provider health/status
        // For now, assume all providers are available with different reliability scores
        match provider_name {
            "local" => 0.99, // Local is most reliable
            "openai" => 0.95,
            "anthropic" => 0.95,
            _ => 0.80,
        }
    }

    /// Get use case fit score
    fn get_use_case_fit_score(
        &self,
        provider_name: &str,
        model_name: &str,
        use_case: &UseCase,
    ) -> f32 {
        match use_case {
            UseCase::SparqlGeneration | UseCase::CodeGeneration => {
                if model_name.contains("gpt-4") || model_name.contains("codellama") {
                    0.95
                } else if model_name.contains("sonnet") {
                    0.85
                } else {
                    0.70
                }
            }
            UseCase::ComplexReasoning | UseCase::Analysis => {
                if model_name.contains("opus") || model_name.contains("gpt-4") {
                    0.95
                } else if model_name.contains("sonnet") {
                    0.85
                } else {
                    0.70
                }
            }
            UseCase::SimpleQuery | UseCase::Conversation => {
                if model_name.contains("haiku") || model_name.contains("3.5") {
                    0.95
                } else if provider_name == "local" {
                    0.90
                } else {
                    0.80
                }
            }
            UseCase::KnowledgeExtraction => {
                if model_name.contains("sonnet") || model_name.contains("gpt-4") {
                    0.90
                } else {
                    0.75
                }
            }
        }
    }

    /// Get priority score multiplier
    fn get_priority_score(&self, priority: &Priority) -> f32 {
        match priority {
            Priority::Critical => 1.2,
            Priority::High => 1.1,
            Priority::Normal => 1.0,
            Priority::Low => 0.9,
        }
    }

    /// Estimate input tokens from request
    fn estimate_input_tokens(&self, request: &LLMRequest) -> usize {
        let mut total_chars = 0;

        if let Some(ref system_prompt) = request.system_prompt {
            total_chars += system_prompt.len();
        }

        for message in &request.messages {
            total_chars += message.content.len();
        }

        // Rough approximation: 1 token ≈ 4 characters
        (total_chars / 4).max(10)
    }

    fn select_fallback_provider(
        &self,
        failed_provider: &str,
        failed_model: &str,
    ) -> Result<(String, String)> {
        // Sophisticated fallback logic based on model capabilities
        let fallback_order = match failed_provider {
            "openai" => vec!["anthropic", "local"],
            "anthropic" => vec!["openai", "local"],
            "local" => vec!["openai", "anthropic"],
            _ => vec!["openai", "anthropic", "local"],
        };

        for provider_name in fallback_order {
            if let Some(provider) = self.providers.get(provider_name) {
                // Select appropriate fallback model based on the failed model's capability
                // Select appropriate fallback model based on the failed model's capability
                let fallback_model = match (provider_name, failed_model) {
                    // OpenAI fallbacks
                    ("openai", model) if model.contains("gpt-4") => "gpt-4".to_string(),
                    ("openai", model) if model.contains("claude-3-opus") => "gpt-4".to_string(),
                    ("openai", _) => "gpt-3.5-turbo".to_string(),

                    // Anthropic fallbacks
                    ("anthropic", model) if model.contains("gpt-4") => {
                        "claude-3-opus-20240229".to_string()
                    }
                    ("anthropic", model) if model.contains("opus") => {
                        "claude-3-opus-20240229".to_string()
                    }
                    ("anthropic", model) if model.contains("turbo") => {
                        "claude-3-haiku-20240307".to_string()
                    }
                    ("anthropic", _) => "claude-3-sonnet-20240229".to_string(),

                    // Local fallbacks
                    ("local", model) if model.contains("code") => "codellama-7b".to_string(),
                    ("local", model) if model.contains("13b") => "llama-2-13b".to_string(),
                    ("local", _) => "mistral-7b".to_string(),

                    _ => {
                        let models = provider.get_available_models();
                        if models.is_empty() {
                            continue;
                        } else {
                            models[0].clone()
                        }
                    }
                };

                return Ok((provider_name.to_string(), fallback_model));
            }
        }

        Err(anyhow!("No fallback provider available"))
    }

    /// Get usage statistics
    pub async fn get_usage_stats(&self) -> UsageStats {
        self.usage_tracker.lock().await.get_stats()
    }
}

/// Provider trait for implementing different LLM providers
#[async_trait::async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse>;
    async fn generate_stream(&self, model: &str, request: &LLMRequest)
        -> Result<LLMResponseStream>;
    fn get_available_models(&self) -> Vec<String>;
    fn supports_streaming(&self) -> bool;
    fn get_provider_name(&self) -> &str;
    fn estimate_cost(&self, model: &str, input_tokens: usize, output_tokens: usize) -> f64;
}

/// OpenAI provider implementation
pub struct OpenAIProvider {
    client: OpenAIClient<OpenAIConfig>,
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
                    content: ChatCompletionRequestSystemMessageContent::Text(system_prompt.clone()),
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
                            content: ChatCompletionRequestSystemMessageContent::Text(
                                msg.content.clone(),
                            ),
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

        let response = self.client.chat().create(openai_request).await?;

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

    fn get_provider_name(&self) -> &str {
        "openai"
    }

    fn estimate_cost(&self, model: &str, input_tokens: usize, output_tokens: usize) -> f64 {
        // Pricing as of 2024 (per 1K tokens)
        let (input_price, output_price) = match model {
            "gpt-4" | "gpt-4-0314" => (0.03, 0.06),
            "gpt-4-32k" | "gpt-4-32k-0314" => (0.06, 0.12),
            "gpt-4-turbo" | "gpt-4-1106-preview" => (0.01, 0.03),
            "gpt-3.5-turbo" | "gpt-3.5-turbo-0301" => (0.0015, 0.002),
            "gpt-3.5-turbo-16k" => (0.003, 0.004),
            _ => (0.002, 0.002), // Default pricing
        };

        (input_tokens as f64 * input_price / 1000.0)
            + (output_tokens as f64 * output_price / 1000.0)
    }

    async fn generate_stream(
        &self,
        model: &str,
        request: &LLMRequest,
    ) -> Result<LLMResponseStream> {
        use async_openai::types::{
            ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
        };

        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();

        // Add system message if provided
        if let Some(system_prompt) = &request.system_prompt {
            messages.push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(system_prompt.clone()),
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
                            content: ChatCompletionRequestSystemMessageContent::Text(
                                msg.content.clone(),
                            ),
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
            .stream(true)
            .build()?;

        let stream = self.client.chat().create_stream(openai_request).await?;

        let model_name = model.to_string();
        let provider_name = "openai".to_string();
        let started_at = Instant::now();

        // Transform the OpenAI stream into our custom stream
        let transformed_stream =
            stream
                .enumerate()
                .map(move |(index, chunk_result)| match chunk_result {
                    Ok(chunk) => {
                        let content = chunk
                            .choices
                            .first()
                            .and_then(|choice| choice.delta.content.as_ref())
                            .map(|s| s.clone())
                            .unwrap_or_default();

                        let is_final = chunk
                            .choices
                            .first()
                            .map(|choice| choice.finish_reason.is_some())
                            .unwrap_or(false);

                        Ok(LLMResponseChunk {
                            content,
                            is_final,
                            chunk_index: index,
                            model_used: model_name.clone(),
                            provider_used: provider_name.clone(),
                            latency: started_at.elapsed(),
                            metadata: HashMap::new(),
                        })
                    }
                    Err(e) => Err(anyhow!("Stream error: {}", e)),
                });

        Ok(LLMResponseStream {
            stream: Box::pin(transformed_stream),
            model_used: model.to_string(),
            provider_used: "openai".to_string(),
            started_at,
        })
    }
}

/// Anthropic Claude provider implementation
pub struct AnthropicProvider {
    api_key: String,
    config: ProviderConfig,
    client: reqwest::Client,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow!("Anthropic API key not provided"))?
            .clone();

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .default_headers(headers)
            .build()?;

        Ok(Self {
            api_key,
            config,
            client,
            base_url,
        })
    }
}

#[async_trait::async_trait]
impl LLMProvider for AnthropicProvider {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse> {
        let mut messages = Vec::new();
        let mut system_messages = Vec::new();

        // Separate system messages from conversation messages
        for msg in &request.messages {
            match msg.role {
                ChatRole::System => {
                    system_messages.push(msg.content.clone());
                }
                ChatRole::User => {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": msg.content
                    }));
                }
                ChatRole::Assistant => {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": msg.content
                    }));
                }
            }
        }

        // Combine system messages
        let mut system_content = Vec::new();
        if let Some(ref system_prompt) = request.system_prompt {
            system_content.push(system_prompt.clone());
        }
        system_content.extend(system_messages);
        let combined_system = if system_content.is_empty() {
            None
        } else {
            Some(system_content.join("\n\n"))
        };

        // Prepare request body
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "temperature": request.temperature,
        });

        if let Some(system) = combined_system {
            body["system"] = serde_json::Value::String(system);
        }

        // Add metadata if present
        let mut metadata = HashMap::new();
        metadata.insert(
            "user_id".to_string(),
            serde_json::Value::String("oxirs-chat".to_string()),
        );
        body["metadata"] = serde_json::to_value(&metadata)?;

        debug!(
            "Sending request to Anthropic API: {}",
            serde_json::to_string_pretty(&body)?
        );

        let response = self
            .client
            .post(&format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let response_text = response.text().await?;

        if !status.is_success() {
            error!("Anthropic API error: {} - {}", status, response_text);
            return Err(anyhow!(
                "Anthropic API error: {} - {}",
                status,
                response_text
            ));
        }

        debug!("Anthropic API response: {}", response_text);

        let response_json: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| {
                anyhow!(
                    "Failed to parse Anthropic response: {} - Response: {}",
                    e,
                    response_text
                )
            })?;

        // Extract content with better error handling
        let content = response_json
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or_else(|| {
                warn!(
                    "Unexpected response format from Anthropic: {}",
                    response_json
                );
                "No content available"
            })
            .to_string();

        // Extract usage statistics
        let usage_data = response_json.get("usage");
        let input_tokens = usage_data
            .and_then(|u| u.get("input_tokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as usize;
        let output_tokens = usage_data
            .and_then(|u| u.get("output_tokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as usize;

        let total_tokens = input_tokens + output_tokens;
        let cost = self.estimate_cost(model, input_tokens, output_tokens);

        // Create response metadata
        let mut response_metadata = HashMap::new();
        if let Some(id) = response_json.get("id") {
            response_metadata.insert("anthropic_id".to_string(), id.clone());
        }
        if let Some(stop_reason) = response_json.get("stop_reason") {
            response_metadata.insert("stop_reason".to_string(), stop_reason.clone());
        }

        Ok(LLMResponse {
            content,
            model_used: model.to_string(),
            provider_used: "anthropic".to_string(),
            usage: Usage {
                prompt_tokens: input_tokens,
                completion_tokens: output_tokens,
                total_tokens,
                cost,
            },
            latency: Duration::from_secs(0), // Will be set by caller
            quality_score: Some(0.85),       // Anthropic generally provides high quality
            metadata: response_metadata,
        })
    }

    fn get_available_models(&self) -> Vec<String> {
        vec![
            "claude-3-opus-20240229".to_string(),
            "claude-3-sonnet-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
            "claude-2.1".to_string(),
            "claude-2.0".to_string(),
            "claude-instant-1.2".to_string(),
        ]
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn get_provider_name(&self) -> &str {
        "anthropic"
    }

    fn estimate_cost(&self, model: &str, input_tokens: usize, output_tokens: usize) -> f64 {
        // Pricing as of 2024 (per 1K tokens)
        let (input_price, output_price) = match model {
            "claude-3-opus-20240229" => (0.015, 0.075),
            "claude-3-sonnet-20240229" => (0.003, 0.015),
            "claude-3-haiku-20240307" => (0.00025, 0.00125),
            "claude-2.1" | "claude-2.0" => (0.008, 0.024),
            "claude-instant-1.2" => (0.0008, 0.0024),
            _ => (0.001, 0.003), // Default pricing
        };

        (input_tokens as f64 * input_price / 1000.0)
            + (output_tokens as f64 * output_price / 1000.0)
    }

    async fn generate_stream(
        &self,
        model: &str,
        request: &LLMRequest,
    ) -> Result<LLMResponseStream> {
        // Note: This is a simplified implementation. Production version would implement actual streaming
        // using Anthropic's streaming API when available.

        // For now, simulate streaming by breaking response into chunks
        // In production, this would use Anthropic's streaming API
        let response = self.generate(model, request).await?;
        let words: Vec<String> = response
            .content
            .split_whitespace()
            .map(String::from)
            .collect();
        let chunk_size = 5; // Words per chunk

        let model_name = model.to_string();
        let provider_name = "anthropic".to_string();
        let started_at = Instant::now();

        // Create chunks with owned data
        let chunks: Vec<Result<LLMResponseChunk>> = words
            .chunks(chunk_size)
            .enumerate()
            .map(|(index, chunk)| {
                let total_words = words.len();
                let is_final = (index + 1) * chunk_size >= total_words;
                Ok(LLMResponseChunk {
                    content: chunk.join(" ") + if !is_final { " " } else { "" },
                    is_final,
                    chunk_index: index,
                    model_used: model_name.clone(),
                    provider_used: provider_name.clone(),
                    latency: started_at.elapsed(),
                    metadata: HashMap::new(),
                })
            })
            .collect();

        // Create stream from owned chunks
        let stream = futures_util::stream::iter(chunks);

        Ok(LLMResponseStream {
            stream: Box::pin(stream),
            model_used: model.to_string(),
            provider_used: "anthropic".to_string(),
            started_at,
        })
    }
}

/// Local model provider implementation (using llama.cpp or similar)
pub struct LocalModelProvider {
    config: ProviderConfig,
    model_path: PathBuf,
}

impl LocalModelProvider {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let model_path = PathBuf::from(
            config
                .base_url
                .as_ref()
                .ok_or_else(|| anyhow!("Model path not specified for local provider"))?,
        );

        if !model_path.exists() {
            return Err(anyhow!("Model file does not exist: {:?}", model_path));
        }

        Ok(Self { config, model_path })
    }
}

#[async_trait::async_trait]
impl LLMProvider for LocalModelProvider {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse> {
        // This is a placeholder implementation
        // In production, this would interface with llama.cpp, candle, or another local inference engine

        let prompt = format!(
            "{}\n\n{}",
            request.system_prompt.as_deref().unwrap_or(""),
            request
                .messages
                .iter()
                .map(|m| format!("{:?}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n")
        );

        // Simulate local model response
        let content = format!(
            "Local model response to: {}... (Model: {})",
            &prompt.chars().take(50).collect::<String>(),
            model
        );

        let prompt_tokens = prompt.split_whitespace().count();
        let completion_tokens = content.split_whitespace().count();

        Ok(LLMResponse {
            content,
            model_used: model.to_string(),
            provider_used: "local".to_string(),
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                cost: 0.0, // Local models are free
            },
            latency: Duration::from_millis(100),
            quality_score: Some(0.7),
            metadata: HashMap::new(),
        })
    }

    fn get_available_models(&self) -> Vec<String> {
        vec![
            "llama-2-7b".to_string(),
            "llama-2-13b".to_string(),
            "mistral-7b".to_string(),
            "codellama-7b".to_string(),
        ]
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn get_provider_name(&self) -> &str {
        "local"
    }

    fn estimate_cost(&self, _model: &str, _input_tokens: usize, _output_tokens: usize) -> f64 {
        0.0 // Local models are free
    }

    async fn generate_stream(
        &self,
        model: &str,
        request: &LLMRequest,
    ) -> Result<LLMResponseStream> {
        // Simulate streaming for local models
        let response = self.generate(model, request).await?;
        let words: Vec<String> = response
            .content
            .split_whitespace()
            .map(String::from)
            .collect();
        let chunk_size = 3; // Words per chunk

        let model_name = model.to_string();
        let provider_name = "local".to_string();
        let started_at = Instant::now();

        // Create chunks with owned data
        let chunks: Vec<Result<LLMResponseChunk>> = words
            .chunks(chunk_size)
            .enumerate()
            .map(|(index, chunk)| {
                let total_words = words.len();
                let is_final = (index + 1) * chunk_size >= total_words;
                Ok(LLMResponseChunk {
                    content: chunk.join(" ") + if !is_final { " " } else { "" },
                    is_final,
                    chunk_index: index,
                    model_used: model_name.clone(),
                    provider_used: provider_name.clone(),
                    latency: started_at.elapsed(),
                    metadata: HashMap::new(),
                })
            })
            .collect();

        // Create stream from owned chunks
        let stream = futures_util::stream::iter(chunks);

        Ok(LLMResponseStream {
            stream: Box::pin(stream),
            model_used: model.to_string(),
            provider_used: "local".to_string(),
            started_at,
        })
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
