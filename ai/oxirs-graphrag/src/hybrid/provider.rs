//! LLM provider trait and deterministic local mock.

use async_trait::async_trait;

/// Capabilities of an LLM provider.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub supports_streaming: bool,
    pub max_context_tokens: usize,
    pub supports_embeddings: bool,
}

/// A completion request.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: usize,
}

/// A completion response.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub text: String,
    pub tokens_used: usize,
}

/// Errors from an LLM provider.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("provider error: {0}")]
    Provider(String),
    #[error("rate limit exceeded")]
    RateLimit,
    #[error("context too long")]
    ContextTooLong,
}

/// An LLM provider that can complete prompts and optionally produce embeddings.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LlmError>;
    fn capabilities(&self) -> &Capabilities;
    fn name(&self) -> &str;
}

/// Deterministic local provider for tests — returns canned responses.
pub struct LocalProvider {
    capabilities: Capabilities,
    /// Default response text.
    response: String,
}

impl LocalProvider {
    pub fn new() -> Self {
        Self {
            capabilities: Capabilities {
                supports_streaming: false,
                max_context_tokens: 4096,
                supports_embeddings: false,
            },
            response: "The answer is 42.".to_string(),
        }
    }

    pub fn with_response(response: impl Into<String>) -> Self {
        Self {
            capabilities: Self::new().capabilities,
            response: response.into(),
        }
    }
}

impl Default for LocalProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmProvider for LocalProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LlmError> {
        // Use char-safe truncation to avoid UTF-8 boundary panics.
        let context_preview: String = request.prompt.chars().take(40).collect();
        let text = format!("{} [context: {}]", self.response, context_preview);
        Ok(CompletionResponse {
            text,
            tokens_used: request.max_tokens / 10,
        })
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn name(&self) -> &str {
        "local"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_provider_default_response() {
        let provider = LocalProvider::new();
        let req = CompletionRequest {
            prompt: "Hello world".to_string(),
            max_tokens: 100,
        };
        let resp = provider.complete(&req).await.expect("should succeed");
        assert!(resp.text.contains("The answer is 42."));
        assert_eq!(resp.tokens_used, 10);
    }

    #[tokio::test]
    async fn test_local_provider_with_custom_response() {
        let provider = LocalProvider::with_response("Custom answer");
        let req = CompletionRequest {
            prompt: "test".to_string(),
            max_tokens: 200,
        };
        let resp = provider.complete(&req).await.expect("should succeed");
        assert!(resp.text.contains("Custom answer"));
    }

    #[test]
    fn test_capabilities() {
        let provider = LocalProvider::new();
        let caps = provider.capabilities();
        assert_eq!(caps.max_context_tokens, 4096);
        assert!(!caps.supports_streaming);
        assert!(!caps.supports_embeddings);
    }

    #[tokio::test]
    async fn test_prompt_longer_than_40_chars_no_panic() {
        let provider = LocalProvider::new();
        let long_prompt = "a".repeat(200);
        let req = CompletionRequest {
            prompt: long_prompt,
            max_tokens: 50,
        };
        let resp = provider.complete(&req).await.expect("should not panic");
        assert!(!resp.text.is_empty());
    }
}
