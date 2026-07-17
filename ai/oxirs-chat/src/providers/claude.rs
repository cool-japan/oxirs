//! Anthropic Claude LLM provider (model catalog / config types)
//!
//! Provides the [`ClaudeModel`] catalog and [`ClaudeConfig`] builder used to
//! describe Claude 3/4 model families. [`ClaudeClient::complete`] and
//! [`ClaudeClient::chat`] do **not** call the real Anthropic API — no HTTP
//! client is wired up here — and return [`ChatError::LlmGenerationError`]
//! rather than a fabricated response. For an actual working Claude
//! integration, use [`crate::llm::anthropic_provider::AnthropicProvider`],
//! which implements [`crate::llm::LLMProvider`] against the real Anthropic
//! Messages API.

use crate::error::{ChatError, Result};
use crate::llm::types::ChatMessage;

/// Claude model variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaudeModel {
    Claude3Haiku,
    Claude3Sonnet,
    Claude3Opus,
    Claude35Sonnet,
    Claude4Sonnet,
}

impl ClaudeModel {
    /// Return the model ID string for API usage
    pub fn as_str(&self) -> &'static str {
        match self {
            ClaudeModel::Claude3Haiku => "claude-3-haiku-20240307",
            ClaudeModel::Claude3Sonnet => "claude-3-sonnet-20240229",
            ClaudeModel::Claude3Opus => "claude-3-opus-20240229",
            ClaudeModel::Claude35Sonnet => "claude-3-5-sonnet-20241022",
            ClaudeModel::Claude4Sonnet => "claude-sonnet-4-5",
        }
    }

    /// Maximum context window in tokens
    pub fn max_context_tokens(&self) -> usize {
        match self {
            ClaudeModel::Claude3Haiku => 200_000,
            ClaudeModel::Claude3Sonnet => 200_000,
            ClaudeModel::Claude3Opus => 200_000,
            ClaudeModel::Claude35Sonnet => 200_000,
            ClaudeModel::Claude4Sonnet => 200_000,
        }
    }
}

/// Configuration for the Claude client
#[derive(Debug, Clone)]
pub struct ClaudeConfig {
    pub api_key: String,
    pub model: ClaudeModel,
    pub max_tokens: usize,
    pub temperature: f64,
    pub system_prompt: Option<String>,
}

impl ClaudeConfig {
    /// Create a new configuration
    pub fn new(api_key: impl Into<String>, model: ClaudeModel) -> Self {
        Self {
            api_key: api_key.into(),
            model,
            max_tokens: 4096,
            temperature: 0.7,
            system_prompt: None,
        }
    }

    /// Set max tokens for the response
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature (0.0 – 1.0)
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature.clamp(0.0, 1.0);
        self
    }

    /// Set a system prompt
    pub fn with_system_prompt(mut self, system: impl Into<String>) -> Self {
        self.system_prompt = Some(system.into());
        self
    }
}

/// Response from the Claude API
#[derive(Debug, Clone)]
pub struct ClaudeResponse {
    pub content: String,
    pub stop_reason: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

impl ClaudeResponse {
    /// Total token usage
    pub fn total_tokens(&self) -> usize {
        self.input_tokens + self.output_tokens
    }

    /// Check whether the response stopped normally
    pub fn is_end_turn(&self) -> bool {
        self.stop_reason == "end_turn"
    }
}

/// Claude API client (mock implementation)
pub struct ClaudeClient {
    config: ClaudeConfig,
}

impl ClaudeClient {
    /// Create a new Claude client
    pub fn new(config: ClaudeConfig) -> Self {
        Self { config }
    }

    /// Return the configured model name
    pub fn model_name(&self) -> &str {
        self.config.model.as_str()
    }

    /// Return the maximum context window size for the current model
    pub fn max_context_tokens(&self) -> usize {
        self.config.model.max_context_tokens()
    }

    /// Send a single-turn completion request.
    ///
    /// # Errors
    /// Always returns [`ChatError::LlmGenerationError`]: this client has no
    /// HTTP transport wired to the Anthropic Messages API, so it cannot
    /// produce a real completion and must not fabricate one. Use
    /// [`crate::llm::anthropic_provider::AnthropicProvider`] for a working
    /// integration.
    pub async fn complete(&self, prompt: &str) -> Result<ClaudeResponse> {
        if prompt.is_empty() {
            return Err(ChatError::ValidationError(
                "Prompt must not be empty".to_string(),
            ));
        }
        if self.config.api_key.is_empty() {
            return Err(ChatError::ConfigError(
                "Claude API key is not configured".to_string(),
            ));
        }

        Err(ChatError::LlmGenerationError(format!(
            "ClaudeClient (providers::claude) has no real Anthropic API transport wired up for \
             model {model}; use crate::llm::anthropic_provider::AnthropicProvider for a working \
             Claude integration instead of this config-only client",
            model = self.config.model.as_str()
        )))
    }

    /// Send a multi-turn chat request.
    ///
    /// # Errors
    /// Always returns [`ChatError::LlmGenerationError`], for the same reason
    /// as [`complete`](Self::complete).
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<ClaudeResponse> {
        if messages.is_empty() {
            return Err(ChatError::ValidationError(
                "Message list must not be empty".to_string(),
            ));
        }
        if self.config.api_key.is_empty() {
            return Err(ChatError::ConfigError(
                "Claude API key is not configured".to_string(),
            ));
        }

        Err(ChatError::LlmGenerationError(format!(
            "ClaudeClient (providers::claude) has no real Anthropic API transport wired up for \
             model {model}; use crate::llm::anthropic_provider::AnthropicProvider for a working \
             Claude integration instead of this config-only client",
            model = self.config.model.as_str()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ChatMessage, ChatRole};

    fn make_client(key: &str) -> ClaudeClient {
        ClaudeClient::new(ClaudeConfig::new(key, ClaudeModel::Claude3Sonnet))
    }

    // --- ClaudeModel tests ---

    #[test]
    fn test_model_names() {
        assert_eq!(
            ClaudeModel::Claude3Haiku.as_str(),
            "claude-3-haiku-20240307"
        );
        assert_eq!(
            ClaudeModel::Claude35Sonnet.as_str(),
            "claude-3-5-sonnet-20241022"
        );
        assert_eq!(ClaudeModel::Claude4Sonnet.as_str(), "claude-sonnet-4-5");
    }

    #[test]
    fn test_model_context_windows() {
        assert_eq!(ClaudeModel::Claude3Opus.max_context_tokens(), 200_000);
        assert_eq!(ClaudeModel::Claude4Sonnet.max_context_tokens(), 200_000);
    }

    // --- ClaudeConfig builder tests ---

    #[test]
    fn test_config_builder() {
        let cfg = ClaudeConfig::new("sk-test", ClaudeModel::Claude3Opus)
            .with_max_tokens(8192)
            .with_temperature(0.5)
            .with_system_prompt("You are a SPARQL expert.");
        assert_eq!(cfg.max_tokens, 8192);
        assert_eq!(cfg.temperature, 0.5);
        assert!(cfg.system_prompt.is_some());
    }

    #[test]
    fn test_temperature_clamping() {
        let cfg = ClaudeConfig::new("k", ClaudeModel::Claude3Haiku).with_temperature(3.0);
        assert_eq!(cfg.temperature, 1.0);
        let cfg2 = ClaudeConfig::new("k", ClaudeModel::Claude3Haiku).with_temperature(-0.5);
        assert_eq!(cfg2.temperature, 0.0);
    }

    // --- Client tests ---

    #[test]
    fn test_model_name() {
        let client = make_client("key");
        assert_eq!(client.model_name(), "claude-3-sonnet-20240229");
    }

    #[test]
    fn test_max_context_tokens() {
        let client = make_client("key");
        assert_eq!(client.max_context_tokens(), 200_000);
    }

    /// Regression: even with a valid prompt and a configured API key,
    /// `complete()` must not fabricate a response — it has no real
    /// Anthropic API transport wired up, so it must fail loudly instead of
    /// silently returning simulated content.
    #[tokio::test]
    async fn test_complete_never_fabricates_response() {
        let client = make_client("test-key");
        let result = client.complete("Explain RDF triples.").await;
        assert!(result.is_err());
        let err = result.expect_err("must fail loudly instead of fabricating a completion");
        assert!(matches!(err, ChatError::LlmGenerationError(_)));
        let msg = err.to_string();
        assert!(
            !msg.contains("Simulated"),
            "error must not resemble a fabricated completion: {msg}"
        );
    }

    #[tokio::test]
    async fn test_complete_empty_prompt_error() {
        let client = make_client("test-key");
        let result = client.complete("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_complete_no_api_key_error() {
        let client = make_client("");
        let result = client.complete("Hello").await;
        assert!(result.is_err());
    }

    /// Regression: same guarantee as
    /// [`test_complete_never_fabricates_response`] for the multi-turn path.
    #[tokio::test]
    async fn test_chat_never_fabricates_response() {
        let client = make_client("test-key");
        let messages = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "What is OWL?".to_string(),
                metadata: None,
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "OWL is the Web Ontology Language.".to_string(),
                metadata: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: "How does it relate to RDF?".to_string(),
                metadata: None,
            },
        ];
        let result = client.chat(&messages).await;
        assert!(result.is_err());
        let err = result.expect_err("must fail loudly instead of fabricating a completion");
        assert!(matches!(err, ChatError::LlmGenerationError(_)));
    }

    #[tokio::test]
    async fn test_chat_empty_messages_error() {
        let client = make_client("test-key");
        let result = client.chat(&[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_chat_no_api_key_error() {
        let client = make_client("");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
            metadata: None,
        }];
        let result = client.chat(&messages).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_response_total_tokens() {
        let r = ClaudeResponse {
            content: "test".to_string(),
            stop_reason: "end_turn".to_string(),
            input_tokens: 100,
            output_tokens: 50,
        };
        assert_eq!(r.total_tokens(), 150);
        assert!(r.is_end_turn());
    }
}
