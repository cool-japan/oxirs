//! Google Gemini LLM provider
//!
//! Mock implementation for testing and development. In production, this
//! would make actual HTTP requests to the Gemini API.

use crate::error::{ChatError, Result};
use crate::llm::types::ChatMessage;

/// Gemini model variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeminiModel {
    GeminiPro,
    GeminiProVision,
    Gemini15Flash,
    Gemini15Pro,
}

impl GeminiModel {
    /// Return the model name string for API usage
    pub fn as_str(&self) -> &'static str {
        match self {
            GeminiModel::GeminiPro => "gemini-pro",
            GeminiModel::GeminiProVision => "gemini-pro-vision",
            GeminiModel::Gemini15Flash => "gemini-1.5-flash",
            GeminiModel::Gemini15Pro => "gemini-1.5-pro",
        }
    }

    /// Maximum context window tokens for each model
    pub fn max_context_tokens(&self) -> usize {
        match self {
            GeminiModel::GeminiPro => 32_768,
            GeminiModel::GeminiProVision => 16_384,
            GeminiModel::Gemini15Flash => 1_048_576,
            GeminiModel::Gemini15Pro => 2_097_152,
        }
    }
}

/// Harm categories for safety settings
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarmCategory {
    DangerousContent,
    Harassment,
    HateSpeech,
    SexuallyExplicit,
}

/// Threshold for blocking harmful content
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarmBlockThreshold {
    BlockNone,
    BlockLowAndAbove,
    BlockMediumAndAbove,
    BlockHighAndAbove,
}

/// A safety setting pairing a harm category with a blocking threshold
#[derive(Debug, Clone)]
pub struct SafetySetting {
    pub category: HarmCategory,
    pub threshold: HarmBlockThreshold,
}

impl SafetySetting {
    /// Create a new safety setting
    pub fn new(category: HarmCategory, threshold: HarmBlockThreshold) -> Self {
        Self {
            category,
            threshold,
        }
    }
}

/// Configuration for the Gemini client
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub model: GeminiModel,
    pub temperature: f64,
    pub max_output_tokens: usize,
    pub safety_settings: Vec<SafetySetting>,
}

impl GeminiConfig {
    /// Create a new configuration with the given API key and model
    pub fn new(api_key: impl Into<String>, model: GeminiModel) -> Self {
        Self {
            api_key: api_key.into(),
            model,
            temperature: 0.7,
            max_output_tokens: 2048,
            safety_settings: Vec::new(),
        }
    }

    /// Set temperature (0.0 – 1.0)
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature.clamp(0.0, 1.0);
        self
    }

    /// Set maximum output tokens
    pub fn with_max_output_tokens(mut self, tokens: usize) -> Self {
        self.max_output_tokens = tokens;
        self
    }

    /// Add a safety setting
    pub fn with_safety_setting(mut self, setting: SafetySetting) -> Self {
        self.safety_settings.push(setting);
        self
    }
}

/// Response from the Gemini API
#[derive(Debug, Clone)]
pub struct GeminiResponse {
    pub text: String,
    pub finish_reason: String,
    pub token_count: usize,
}

impl GeminiResponse {
    /// Check whether the response finished normally
    pub fn is_complete(&self) -> bool {
        self.finish_reason == "STOP"
    }
}

/// Gemini API client (mock implementation)
pub struct GeminiClient {
    config: GeminiConfig,
}

impl GeminiClient {
    /// Create a new Gemini client
    pub fn new(config: GeminiConfig) -> Self {
        Self { config }
    }

    /// Return the configured model name
    pub fn model_name(&self) -> &str {
        self.config.model.as_str()
    }

    /// Count tokens using a simple word-based estimate (4 chars ≈ 1 token)
    pub fn count_tokens(text: &str) -> usize {
        // Rough approximation: every 4 characters is ~1 token
        let char_count = text.chars().count();
        (char_count + 3) / 4
    }

    /// Generate a response for a single prompt (mock)
    pub async fn generate(&self, prompt: &str) -> Result<GeminiResponse> {
        if prompt.is_empty() {
            return Err(ChatError::ValidationError(
                "Prompt must not be empty".to_string(),
            ));
        }
        if self.config.api_key.is_empty() {
            return Err(ChatError::ConfigError(
                "Gemini API key is not configured".to_string(),
            ));
        }

        // Mock response — in production this would call the Gemini REST API
        let response_text = format!(
            "[Gemini/{model}] Simulated response to: {prompt}",
            model = self.config.model.as_str()
        );
        let token_count = Self::count_tokens(&response_text);

        Ok(GeminiResponse {
            text: response_text,
            finish_reason: "STOP".to_string(),
            token_count,
        })
    }

    /// Generate a response given a conversation history (mock)
    pub async fn generate_with_context(&self, messages: &[ChatMessage]) -> Result<GeminiResponse> {
        if messages.is_empty() {
            return Err(ChatError::ValidationError(
                "Message list must not be empty".to_string(),
            ));
        }
        if self.config.api_key.is_empty() {
            return Err(ChatError::ConfigError(
                "Gemini API key is not configured".to_string(),
            ));
        }

        let last_user = messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, crate::llm::types::ChatRole::User))
            .map(|m| m.content.as_str())
            .unwrap_or("(no user message)");

        let response_text = format!(
            "[Gemini/{model}] Context-aware response to: {last}",
            model = self.config.model.as_str(),
            last = last_user
        );
        let token_count = Self::count_tokens(&response_text);

        Ok(GeminiResponse {
            text: response_text,
            finish_reason: "STOP".to_string(),
            token_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ChatMessage, ChatRole};

    fn make_client(key: &str) -> GeminiClient {
        GeminiClient::new(GeminiConfig::new(key, GeminiModel::GeminiPro))
    }

    // --- GeminiModel tests ---

    #[test]
    fn test_model_names() {
        assert_eq!(GeminiModel::GeminiPro.as_str(), "gemini-pro");
        assert_eq!(GeminiModel::GeminiProVision.as_str(), "gemini-pro-vision");
        assert_eq!(GeminiModel::Gemini15Flash.as_str(), "gemini-1.5-flash");
        assert_eq!(GeminiModel::Gemini15Pro.as_str(), "gemini-1.5-pro");
    }

    #[test]
    fn test_model_context_windows() {
        assert_eq!(GeminiModel::GeminiPro.max_context_tokens(), 32_768);
        assert!(GeminiModel::Gemini15Pro.max_context_tokens() > 1_000_000);
    }

    // --- SafetySetting tests ---

    #[test]
    fn test_safety_setting_construction() {
        let s = SafetySetting::new(HarmCategory::Harassment, HarmBlockThreshold::BlockNone);
        assert_eq!(s.category, HarmCategory::Harassment);
        assert_eq!(s.threshold, HarmBlockThreshold::BlockNone);
    }

    // --- GeminiConfig builder tests ---

    #[test]
    fn test_config_builder() {
        let cfg = GeminiConfig::new("key123", GeminiModel::Gemini15Flash)
            .with_temperature(0.9)
            .with_max_output_tokens(4096)
            .with_safety_setting(SafetySetting::new(
                HarmCategory::DangerousContent,
                HarmBlockThreshold::BlockHighAndAbove,
            ));
        assert_eq!(cfg.temperature, 0.9);
        assert_eq!(cfg.max_output_tokens, 4096);
        assert_eq!(cfg.safety_settings.len(), 1);
    }

    #[test]
    fn test_temperature_clamping() {
        let cfg = GeminiConfig::new("k", GeminiModel::GeminiPro).with_temperature(5.0);
        assert_eq!(cfg.temperature, 1.0);
        let cfg2 = GeminiConfig::new("k", GeminiModel::GeminiPro).with_temperature(-1.0);
        assert_eq!(cfg2.temperature, 0.0);
    }

    // --- Token counting tests ---

    #[test]
    fn test_count_tokens_empty() {
        assert_eq!(GeminiClient::count_tokens(""), 0);
    }

    #[test]
    fn test_count_tokens_basic() {
        // 4 chars → 1 token
        assert_eq!(GeminiClient::count_tokens("abcd"), 1);
        // 5 chars → 2 tokens (ceil)
        assert_eq!(GeminiClient::count_tokens("abcde"), 2);
    }

    #[test]
    fn test_count_tokens_sentence() {
        let tokens = GeminiClient::count_tokens("Hello, world!");
        assert!(tokens > 0);
    }

    // --- Client tests ---

    #[test]
    fn test_model_name() {
        let client = make_client("test-key");
        assert_eq!(client.model_name(), "gemini-pro");
    }

    #[tokio::test]
    async fn test_generate_success() {
        let client = make_client("test-key");
        let response = client.generate("What is RDF?").await;
        assert!(response.is_ok());
        let r = response.expect("should succeed");
        assert!(!r.text.is_empty());
        assert_eq!(r.finish_reason, "STOP");
        assert!(r.is_complete());
    }

    #[tokio::test]
    async fn test_generate_empty_prompt_error() {
        let client = make_client("test-key");
        let result = client.generate("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_no_api_key_error() {
        let client = make_client("");
        let result = client.generate("Hello").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_with_context() {
        let client = make_client("test-key");
        let messages = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "Tell me about SPARQL.".to_string(),
                metadata: None,
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "SPARQL is a query language for RDF.".to_string(),
                metadata: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Give me an example query.".to_string(),
                metadata: None,
            },
        ];
        let result = client.generate_with_context(&messages).await;
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        assert!(r.text.contains("gemini-pro"));
    }

    #[tokio::test]
    async fn test_generate_with_empty_messages_error() {
        let client = make_client("test-key");
        let result = client.generate_with_context(&[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_with_context_no_api_key() {
        let client = make_client("");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
            metadata: None,
        }];
        let result = client.generate_with_context(&messages).await;
        assert!(result.is_err());
    }
}
