//! OpenAI GPT completion provider (requires `llm-network` feature).
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
// Wire types for the OpenAI chat completions API
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct OpenAiChatRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingDatum {
    embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Provider implementation
// ---------------------------------------------------------------------------

/// LLM provider backed by the OpenAI chat completions API.
///
/// Requires the `llm-network` feature and a valid `OPENAI_API_KEY`.
///
/// ```rust,no_run
/// # #[cfg(feature = "llm-network")]
/// # {
/// use oxirs_shacl_ai::llm::OpenAiProvider;
/// let provider = OpenAiProvider::new("sk-…");
/// # }
/// ```
pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    default_model: String,
    embedding_model: String,
    caps: Capabilities,
}

impl OpenAiProvider {
    /// Create an `OpenAiProvider` with the given API key and sensible defaults.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            default_model: "gpt-4o".to_string(),
            embedding_model: "text-embedding-3-small".to_string(),
            caps: Capabilities {
                supports_tools: true,
                supports_embeddings: true,
                supports_streaming: true,
                max_context_tokens: 128_000,
            },
        }
    }

    /// Override the default chat model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Override the embedding model.
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = model.into();
        self
    }

    fn role_str(role: &Role) -> &'static str {
        match role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[async_trait]
impl CompletionProvider for OpenAiProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let model = if request.model == "local" || request.model.is_empty() {
            self.default_model.as_str()
        } else {
            request.model.as_str()
        };

        let oai_messages: Vec<OpenAiMessage<'_>> = request
            .messages
            .iter()
            .map(|m: &Message| OpenAiMessage {
                role: Self::role_str(&m.role),
                content: m.content.as_str(),
            })
            .collect();

        let body = OpenAiChatRequest {
            model,
            messages: oai_messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
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
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Network(format!("HTTP {status}: {body}")));
        }

        let oai: OpenAiChatResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let content = oai
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| LlmError::InvalidResponse("empty choices array".to_string()))?;

        let usage = oai.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
        });

        Ok(CompletionResponse {
            content,
            model: oai.model,
            usage,
        })
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        #[derive(Serialize)]
        struct EmbedRequest<'a> {
            model: &'a str,
            input: &'a [String],
        }

        let body = EmbedRequest {
            model: &self.embedding_model,
            input: texts,
        };

        let resp = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Network(e.to_string()))?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(LlmError::AuthFailed);
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Network(format!("HTTP {status}: {body}")));
        }

        let oai: OpenAiEmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let embeddings = oai.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }

    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn name(&self) -> &str {
        "openai"
    }
}
