//! Anthropic Claude completion provider (requires `llm-network` feature).
//!
//! This module is compiled only when the `llm-network` Cargo feature is
//! enabled, which in turn makes `reqwest` available.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::provider::{
    Capabilities, CompletionProvider, CompletionRequest, CompletionResponse, LlmError, Message,
    Role, TokenUsage,
};

// ---------------------------------------------------------------------------
// Wire types for the Anthropic Messages API
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    model: String,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ---------------------------------------------------------------------------
// Provider implementation
// ---------------------------------------------------------------------------

/// LLM provider backed by the Anthropic Messages API.
///
/// Requires the `llm-network` feature and a valid `ANTHROPIC_API_KEY`.
///
/// ```rust,no_run
/// # #[cfg(feature = "llm-network")]
/// # {
/// use oxirs_shacl_ai::llm::AnthropicProvider;
/// let provider = AnthropicProvider::new("sk-ant-…");
/// # }
/// ```
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    default_model: String,
    caps: Capabilities,
}

impl AnthropicProvider {
    /// Create an `AnthropicProvider` with the given API key and sensible defaults.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            default_model: "claude-sonnet-4-5".to_string(),
            caps: Capabilities {
                supports_tools: true,
                supports_embeddings: false, // Anthropic does not offer an embeddings endpoint
                supports_streaming: true,
                max_context_tokens: 200_000,
            },
        }
    }

    /// Override the default model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    fn role_str(role: &Role) -> &'static str {
        match role {
            Role::System => "user", // System messages are handled separately
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[async_trait]
impl CompletionProvider for AnthropicProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let model = if request.model == "local" || request.model.is_empty() {
            self.default_model.as_str()
        } else {
            request.model.as_str()
        };

        // Extract system message if present (Anthropic uses a top-level `system` field)
        let system_content: Option<String> = request
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone());

        // Non-system messages become the `messages` array
        let non_system: Vec<&Message> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect();

        let ant_messages: Vec<AnthropicMessage<'_>> = non_system
            .iter()
            .map(|m| AnthropicMessage {
                role: Self::role_str(&m.role),
                content: m.content.as_str(),
            })
            .collect();

        let max_tokens = request.max_tokens.unwrap_or(1024);

        let body = AnthropicRequest {
            model,
            messages: ant_messages,
            max_tokens,
            system: system_content.as_deref(),
            temperature: request.temperature,
        };

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Network(e.to_string()))?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(LlmError::AuthFailed);
        }

        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(LlmError::RateLimited {
                retry_after_secs: 60,
            });
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(LlmError::Network(format!("HTTP {status}: {body_text}")));
        }

        let ant: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let content = ant
            .content
            .into_iter()
            .next()
            .map(|b| b.text)
            .ok_or_else(|| LlmError::InvalidResponse("empty content array".to_string()))?;

        Ok(CompletionResponse {
            content,
            model: ant.model,
            usage: Some(TokenUsage {
                prompt_tokens: ant.usage.input_tokens,
                completion_tokens: ant.usage.output_tokens,
            }),
        })
    }

    /// Anthropic does not expose a public embeddings endpoint.
    ///
    /// This implementation always returns [`LlmError::Unavailable`].
    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        Err(LlmError::Unavailable(
            "Anthropic does not provide an embeddings endpoint".to_string(),
        ))
    }

    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}
