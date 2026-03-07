//! Mistral AI Provider Implementation
//!
//! Implements the LLM provider trait for Mistral AI's API.
//! Mistral uses an OpenAI-compatible format with some Mistral-specific extensions
//! such as `safe_prompt` and `random_seed`.
//!
//! Supported models: open-mistral-7b, open-mixtral-8x7b, mistral-small-latest,
//! mistral-medium-latest, mistral-large-latest, and embedding model.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tracing::{debug, error};

use super::{
    config::ProviderConfig,
    providers::LLMProvider,
    types::{ChatRole, LLMRequest, LLMResponse, LLMResponseChunk, LLMResponseStream, Usage},
};

/// Mistral AI model variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MistralModel {
    /// open-mistral-7b (formerly mistral-tiny)
    OpenMistral7b,
    /// open-mixtral-8x7b (formerly mistral-small)
    OpenMixtral8x7b,
    /// open-mixtral-8x22b
    OpenMixtral8x22b,
    /// mistral-small-latest
    MistralSmallLatest,
    /// mistral-medium-latest
    MistralMediumLatest,
    /// mistral-large-latest
    MistralLargeLatest,
    /// codestral-latest (code specialised)
    CodestralLatest,
    /// mistral-embed (embeddings only)
    MistralEmbed,
    /// Custom model identifier
    Custom(String),
}

impl MistralModel {
    pub fn model_id(&self) -> &str {
        match self {
            Self::OpenMistral7b => "open-mistral-7b",
            Self::OpenMixtral8x7b => "open-mixtral-8x7b",
            Self::OpenMixtral8x22b => "open-mixtral-8x22b",
            Self::MistralSmallLatest => "mistral-small-latest",
            Self::MistralMediumLatest => "mistral-medium-latest",
            Self::MistralLargeLatest => "mistral-large-latest",
            Self::CodestralLatest => "codestral-latest",
            Self::MistralEmbed => "mistral-embed",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Whether this model supports chat completions (as opposed to embedding-only)
    pub fn supports_chat(&self) -> bool {
        !matches!(self, Self::MistralEmbed)
    }

    /// Pricing per 1K tokens (input_price, output_price) in USD
    pub fn cost_per_1k_tokens(&self) -> (f64, f64) {
        match self {
            Self::OpenMistral7b => (0.00025, 0.00025),
            Self::OpenMixtral8x7b => (0.0007, 0.0007),
            Self::OpenMixtral8x22b => (0.002, 0.006),
            Self::MistralSmallLatest => (0.001, 0.003),
            Self::MistralMediumLatest => (0.0027, 0.0081),
            Self::MistralLargeLatest => (0.004, 0.012),
            Self::CodestralLatest => (0.001, 0.003),
            Self::MistralEmbed => (0.0001, 0.0),
            Self::Custom(_) => (0.002, 0.006),
        }
    }
}

impl std::fmt::Display for MistralModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model_id())
    }
}

/// Chat message in Mistral's OpenAI-compatible format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralMessage {
    /// "system" | "user" | "assistant"
    pub role: String,
    pub content: String,
}

impl MistralMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Mistral chat completion request
#[derive(Debug, Serialize)]
pub struct MistralChatRequest {
    pub model: String,
    pub messages: Vec<MistralMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Whether to inject a safety prompt before all conversations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_prompt: Option<bool>,
    /// Seed for deterministic sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Token usage in a Mistral response
#[derive(Debug, Deserialize)]
pub struct MistralUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// A single choice in a Mistral chat completion
#[derive(Debug, Deserialize)]
pub struct MistralChoice {
    pub index: u32,
    pub message: MistralMessage,
    pub finish_reason: Option<String>,
}

/// Mistral chat completion response
#[derive(Debug, Deserialize)]
pub struct MistralChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<MistralChoice>,
    pub usage: Option<MistralUsage>,
}

impl MistralChatResponse {
    /// Return the first choice's content, or empty string if absent.
    pub fn first_content(&self) -> &str {
        self.choices
            .first()
            .map(|c| c.message.content.as_str())
            .unwrap_or("")
    }

    /// Extract token counts as (prompt, completion, total).
    pub fn token_counts(&self) -> (usize, usize, usize) {
        self.usage
            .as_ref()
            .map(|u| {
                (
                    u.prompt_tokens as usize,
                    u.completion_tokens as usize,
                    u.total_tokens as usize,
                )
            })
            .unwrap_or((0, 0, 0))
    }
}

/// Mistral provider implementing the unified LLMProvider trait
pub struct MistralProvider {
    api_key: String,
    config: ProviderConfig,
    client: reqwest::Client,
    base_url: String,
}

impl MistralProvider {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow!("Mistral API key not provided"))?
            .clone();

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.mistral.ai".to_string());

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;

        Ok(Self {
            api_key,
            config,
            client,
            base_url,
        })
    }

    /// Convert an LLMRequest into Mistral's message list
    fn build_messages(&self, request: &LLMRequest) -> Vec<MistralMessage> {
        let mut messages: Vec<MistralMessage> = Vec::new();

        if let Some(ref sp) = request.system_prompt {
            messages.push(MistralMessage::system(sp.clone()));
        }

        for msg in &request.messages {
            match msg.role {
                ChatRole::System => messages.push(MistralMessage::system(msg.content.clone())),
                ChatRole::User => messages.push(MistralMessage::user(msg.content.clone())),
                ChatRole::Assistant => {
                    messages.push(MistralMessage::assistant(msg.content.clone()))
                }
            }
        }

        messages
    }

    /// Send a raw MistralChatRequest and parse the response
    async fn send_request(&self, mistral_req: &MistralChatRequest) -> Result<MistralChatResponse> {
        debug!("Sending request to Mistral API model={}", mistral_req.model);

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(mistral_req)
            .send()
            .await
            .map_err(|e| anyhow!("Mistral HTTP request failed: {}", e))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read Mistral response body: {}", e))?;

        if !status.is_success() {
            error!("Mistral API error: {} - {}", status, body);
            return Err(anyhow!("Mistral API error {}: {}", status, body));
        }

        let parsed: MistralChatResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow!("Failed to parse Mistral response: {} - body: {}", e, body))?;

        Ok(parsed)
    }
}

#[async_trait]
impl LLMProvider for MistralProvider {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse> {
        let started_at = Instant::now();
        let messages = self.build_messages(request);

        let mistral_req = MistralChatRequest {
            model: model.to_string(),
            messages,
            temperature: Some(request.temperature as f64),
            top_p: None,
            max_tokens: request.max_tokens.map(|t| t as u32),
            safe_prompt: None,
            random_seed: None,
            stream: None,
        };

        let mistral_resp = self.send_request(&mistral_req).await?;
        let latency = started_at.elapsed();

        let (prompt_tokens, completion_tokens, total_tokens) = mistral_resp.token_counts();
        let cost = self.estimate_cost(model, prompt_tokens, completion_tokens);

        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        metadata.insert(
            "response_id".to_string(),
            serde_json::Value::String(mistral_resp.id.clone()),
        );
        metadata.insert(
            "object".to_string(),
            serde_json::Value::String(mistral_resp.object.clone()),
        );
        if let Some(choice) = mistral_resp.choices.first() {
            if let Some(ref finish_reason) = choice.finish_reason {
                metadata.insert(
                    "finish_reason".to_string(),
                    serde_json::Value::String(finish_reason.clone()),
                );
            }
        }

        Ok(LLMResponse {
            content: mistral_resp.first_content().to_string(),
            model_used: mistral_resp.model.clone(),
            provider_used: "mistral".to_string(),
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
            },
            latency,
            quality_score: Some(0.83),
            metadata,
        })
    }

    async fn generate_stream(
        &self,
        model: &str,
        request: &LLMRequest,
    ) -> Result<LLMResponseStream> {
        // Mistral supports SSE streaming; simulate with full response for now
        let response = self.generate(model, request).await?;
        let words: Vec<String> = response
            .content
            .split_whitespace()
            .map(String::from)
            .collect();
        let chunk_size = 5usize;

        let model_name = model.to_string();
        let provider_name = "mistral".to_string();
        let started_at = Instant::now();
        let total_words = words.len();

        let chunks: Vec<Result<LLMResponseChunk>> = words
            .chunks(chunk_size)
            .enumerate()
            .map(|(index, chunk)| {
                let is_final = (index + 1) * chunk_size >= total_words;
                Ok(LLMResponseChunk {
                    content: chunk.join(" ") + if is_final { "" } else { " " },
                    is_final,
                    chunk_index: index,
                    model_used: model_name.clone(),
                    provider_used: provider_name.clone(),
                    latency: started_at.elapsed(),
                    metadata: HashMap::new(),
                })
            })
            .collect();

        Ok(LLMResponseStream {
            stream: Box::pin(futures_util::stream::iter(chunks)),
            model_used: model.to_string(),
            provider_used: "mistral".to_string(),
            started_at,
        })
    }

    fn get_available_models(&self) -> Vec<String> {
        let config_models: Vec<String> =
            self.config.models.iter().map(|m| m.name.clone()).collect();

        let defaults = vec![
            "open-mistral-7b".to_string(),
            "open-mixtral-8x7b".to_string(),
            "open-mixtral-8x22b".to_string(),
            "mistral-small-latest".to_string(),
            "mistral-medium-latest".to_string(),
            "mistral-large-latest".to_string(),
            "codestral-latest".to_string(),
            "mistral-embed".to_string(),
        ];

        let mut all: std::collections::HashSet<String> =
            config_models.into_iter().chain(defaults).collect();
        let mut sorted: Vec<String> = all.drain().collect();
        sorted.sort();
        sorted
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn get_provider_name(&self) -> &str {
        "mistral"
    }

    fn estimate_cost(&self, model: &str, input_tokens: usize, output_tokens: usize) -> f64 {
        let mistral_model = match model {
            "open-mistral-7b" => MistralModel::OpenMistral7b,
            "open-mixtral-8x7b" => MistralModel::OpenMixtral8x7b,
            "open-mixtral-8x22b" => MistralModel::OpenMixtral8x22b,
            "mistral-small-latest" | "mistral-small-2402" => MistralModel::MistralSmallLatest,
            "mistral-medium-latest" | "mistral-medium-2312" => MistralModel::MistralMediumLatest,
            "mistral-large-latest" | "mistral-large-2402" => MistralModel::MistralLargeLatest,
            "codestral-latest" => MistralModel::CodestralLatest,
            "mistral-embed" => MistralModel::MistralEmbed,
            _ => MistralModel::Custom(model.to_string()),
        };
        let (ip, op) = mistral_model.cost_per_1k_tokens();
        (input_tokens as f64 * ip / 1000.0) + (output_tokens as f64 * op / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_model_ids() {
        assert_eq!(MistralModel::OpenMistral7b.model_id(), "open-mistral-7b");
        assert_eq!(
            MistralModel::OpenMixtral8x7b.model_id(),
            "open-mixtral-8x7b"
        );
        assert_eq!(
            MistralModel::MistralLargeLatest.model_id(),
            "mistral-large-latest"
        );
        assert_eq!(MistralModel::MistralEmbed.model_id(), "mistral-embed");
        assert_eq!(MistralModel::Custom("foo".to_string()).model_id(), "foo");
    }

    #[test]
    fn test_mistral_supports_chat() {
        assert!(MistralModel::MistralLargeLatest.supports_chat());
        assert!(!MistralModel::MistralEmbed.supports_chat());
    }

    #[test]
    fn test_mistral_message_construction() {
        let sys = MistralMessage::system("You are helpful.");
        assert_eq!(sys.role, "system");
        let usr = MistralMessage::user("What is RDF?");
        assert_eq!(usr.role, "user");
        let asst = MistralMessage::assistant("RDF is ...");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_mistral_request_serialization() {
        let req = MistralChatRequest {
            model: "mistral-small-latest".to_string(),
            messages: vec![
                MistralMessage::system("Be concise."),
                MistralMessage::user("Hello"),
            ],
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(128),
            safe_prompt: Some(true),
            random_seed: Some(42),
            stream: None,
        };

        let json = serde_json::to_string(&req).expect("serialization must succeed");
        assert!(json.contains("mistral-small-latest"));
        assert!(json.contains("system"));
        assert!(json.contains("safe_prompt"));
        assert!(json.contains("random_seed"));
        // None fields must be absent
        assert!(!json.contains("\"stream\""));
        assert!(!json.contains("\"top_p\""));
    }

    #[test]
    fn test_mistral_response_deserialization() {
        let json = r#"{
            "id": "cmpl-abc",
            "object": "chat.completion",
            "created": 1714000000,
            "model": "mistral-small-latest",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "RDF is a graph data model."},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28
            }
        }"#;

        let resp: MistralChatResponse =
            serde_json::from_str(json).expect("deserialization must succeed");
        assert_eq!(resp.id, "cmpl-abc");
        assert_eq!(resp.first_content(), "RDF is a graph data model.");
        let (p, c, t) = resp.token_counts();
        assert_eq!(p, 20);
        assert_eq!(c, 8);
        assert_eq!(t, 28);
    }

    #[test]
    fn test_mistral_cost_estimation() {
        // mistral-large-latest: $0.004 input / $0.012 output per 1K tokens
        let (ip, op) = MistralModel::MistralLargeLatest.cost_per_1k_tokens();
        let cost = (1000.0 * ip / 1000.0) + (1000.0 * op / 1000.0);
        let expected = 0.004 + 0.012;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_mistral_model_display() {
        assert_eq!(
            format!("{}", MistralModel::MistralSmallLatest),
            "mistral-small-latest"
        );
    }
}

// ────────────────────────────────────────────────────────────────────
// Mistral Function Calling / Tool Calls support
// ────────────────────────────────────────────────────────────────────

/// Type of tool supported by Mistral (currently only "function")
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MistralToolType {
    Function,
}

/// JSON Schema for a function parameter or property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaProperty {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub description: Option<String>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

/// JSON Schema describing function parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParameters {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: std::collections::HashMap<String, JsonSchemaProperty>,
    pub required: Vec<String>,
}

/// A callable function definition for Mistral tool use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralFunction {
    pub name: String,
    pub description: String,
    pub parameters: FunctionParameters,
}

/// A tool that can be called by Mistral
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralTool {
    #[serde(rename = "type")]
    pub tool_type: MistralToolType,
    pub function: MistralFunction,
}

impl MistralTool {
    /// Create a new function tool
    pub fn function(func: MistralFunction) -> Self {
        Self {
            tool_type: MistralToolType::Function,
            function: func,
        }
    }
}

/// Controls how Mistral selects a tool: auto, none, or a specific function
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MistralToolChoice {
    /// "auto" or "none" string
    Mode(String),
    /// Force a specific function call
    Function {
        r#type: String,
        function: MistralToolChoiceFunction,
    },
}

/// Named function for forced tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralToolChoiceFunction {
    pub name: String,
}

/// A tool call returned in a Mistral choice (when the model calls a function)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: MistralToolCallFunction,
}

/// Function invocation inside a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments string
    pub arguments: String,
}

impl MistralToolCallFunction {
    /// Parse the JSON-encoded arguments into a `serde_json::Value`
    pub fn parse_arguments(&self) -> Result<serde_json::Value> {
        serde_json::from_str(&self.arguments)
            .map_err(|e| anyhow!("Failed to parse tool call arguments: {}", e))
    }
}

/// Extended Mistral chat request with tool support
#[derive(Debug, Serialize)]
pub struct MistralToolChatRequest {
    pub model: String,
    pub messages: Vec<MistralMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<MistralTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
}

/// A choice in a Mistral tool-calling response
#[derive(Debug, Deserialize, Clone)]
pub struct MistralToolChoice2 {
    pub index: u32,
    pub message: MistralToolMessage,
    pub finish_reason: Option<String>,
}

/// A Mistral message that may contain tool calls
#[derive(Debug, Deserialize, Clone)]
pub struct MistralToolMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<MistralToolCall>>,
}

/// Response from a Mistral chat completion with tool use
#[derive(Debug, Deserialize)]
pub struct MistralToolChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<MistralToolChoice2>,
    pub usage: Option<MistralUsage>,
}

impl MistralToolChatResponse {
    /// Return all tool calls from the first choice, if any
    pub fn tool_calls(&self) -> Vec<&MistralToolCall> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|calls| calls.iter().collect())
            .unwrap_or_default()
    }

    /// Whether the model decided to call a tool
    pub fn has_tool_calls(&self) -> bool {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|calls| !calls.is_empty())
            .unwrap_or(false)
    }
}

impl MistralProvider {
    /// Send a chat request with function-calling tools to the Mistral API.
    pub async fn generate_with_tools(
        &self,
        model: &str,
        messages: Vec<MistralMessage>,
        tools: Vec<MistralTool>,
        tool_choice: Option<String>,
    ) -> Result<MistralToolChatResponse> {
        let req = MistralToolChatRequest {
            model: model.to_string(),
            messages,
            temperature: None,
            max_tokens: None,
            tools: Some(tools),
            tool_choice,
            stream: None,
            safe_prompt: None,
            random_seed: None,
        };

        debug!(
            "Sending tool-calling request to Mistral API model={}",
            req.model
        );

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| anyhow!("Mistral tool call HTTP request failed: {}", e))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read Mistral tool response body: {}", e))?;

        if !status.is_success() {
            error!("Mistral tool API error: {} - {}", status, body);
            return Err(anyhow!("Mistral tool API error {}: {}", status, body));
        }

        let parsed: MistralToolChatResponse = serde_json::from_str(&body).map_err(|e| {
            anyhow!(
                "Failed to parse Mistral tool response: {} - body: {}",
                e,
                body
            )
        })?;

        Ok(parsed)
    }
}

// ────────────────────────────────────────────────────────────────────
// Extended tests – targeting 20+ total for this module
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod extended_tests {
    use super::*;

    // ── Tool type ─────────────────────────────────────────────────────

    #[test]
    fn test_mistral_tool_type_serialization() {
        let t = MistralToolType::Function;
        let json = serde_json::to_string(&t).expect("serialize");
        assert_eq!(json, r#""function""#);
    }

    // ── Function definition ───────────────────────────────────────────

    #[test]
    fn test_function_parameters_serialization() {
        let mut props = std::collections::HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchemaProperty {
                schema_type: "string".to_string(),
                description: Some("SPARQL query string".to_string()),
                enum_values: None,
            },
        );
        let params = FunctionParameters {
            schema_type: "object".to_string(),
            properties: props,
            required: vec!["query".to_string()],
        };
        let json = serde_json::to_string(&params).expect("serialize");
        assert!(json.contains("\"type\":\"object\"") || json.contains("\"type\": \"object\""));
        assert!(json.contains("query"));
    }

    #[test]
    fn test_mistral_tool_creation() {
        let func = MistralFunction {
            name: "execute_sparql".to_string(),
            description: "Execute a SPARQL query against the RDF store".to_string(),
            parameters: FunctionParameters {
                schema_type: "object".to_string(),
                properties: std::collections::HashMap::new(),
                required: vec![],
            },
        };
        let tool = MistralTool::function(func.clone());
        assert_eq!(tool.tool_type, MistralToolType::Function);
        assert_eq!(tool.function.name, "execute_sparql");
    }

    #[test]
    fn test_mistral_tool_serialization() {
        let func = MistralFunction {
            name: "rdf_lookup".to_string(),
            description: "Look up RDF resource by URI".to_string(),
            parameters: FunctionParameters {
                schema_type: "object".to_string(),
                properties: {
                    let mut m = std::collections::HashMap::new();
                    m.insert(
                        "uri".to_string(),
                        JsonSchemaProperty {
                            schema_type: "string".to_string(),
                            description: Some("URI of the RDF resource".to_string()),
                            enum_values: None,
                        },
                    );
                    m
                },
                required: vec!["uri".to_string()],
            },
        };
        let tool = MistralTool::function(func);
        let json = serde_json::to_string(&tool).expect("serialize");
        assert!(json.contains("rdf_lookup"));
        assert!(json.contains("function"));
    }

    // ── Tool call deserialization ─────────────────────────────────────

    #[test]
    fn test_tool_call_deserialization() {
        let json = r#"{
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "execute_sparql",
                "arguments": "{\"query\": \"SELECT ?s WHERE { ?s ?p ?o }\"}"
            }
        }"#;
        let call: MistralToolCall = serde_json::from_str(json).expect("deserialize");
        assert_eq!(call.id, "call_abc123");
        assert_eq!(call.function.name, "execute_sparql");
    }

    #[test]
    fn test_tool_call_argument_parsing() {
        let call = MistralToolCall {
            id: "call_xyz".to_string(),
            call_type: "function".to_string(),
            function: MistralToolCallFunction {
                name: "execute_sparql".to_string(),
                arguments: r#"{"query": "SELECT ?s WHERE { ?s a <http://schema.org/Person> }"}"#
                    .to_string(),
            },
        };
        let args = call.function.parse_arguments().expect("parse args");
        assert!(args.is_object());
        assert!(args.get("query").is_some());
    }

    #[test]
    fn test_tool_call_invalid_arguments() {
        let call = MistralToolCall {
            id: "call_bad".to_string(),
            call_type: "function".to_string(),
            function: MistralToolCallFunction {
                name: "bad_fn".to_string(),
                arguments: "{ invalid json }".to_string(),
            },
        };
        assert!(call.function.parse_arguments().is_err());
    }

    // ── Tool response helpers ─────────────────────────────────────────

    #[test]
    fn test_tool_response_has_tool_calls_true() {
        let resp = MistralToolChatResponse {
            id: "resp-1".to_string(),
            object: "chat.completion".to_string(),
            created: 1714000000,
            model: "mistral-large-latest".to_string(),
            choices: vec![MistralToolChoice2 {
                index: 0,
                message: MistralToolMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![MistralToolCall {
                        id: "call_1".to_string(),
                        call_type: "function".to_string(),
                        function: MistralToolCallFunction {
                            name: "rdf_lookup".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
        };
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls().len(), 1);
        assert_eq!(resp.tool_calls()[0].function.name, "rdf_lookup");
    }

    #[test]
    fn test_tool_response_has_tool_calls_false() {
        let resp = MistralToolChatResponse {
            id: "resp-2".to_string(),
            object: "chat.completion".to_string(),
            created: 1714000000,
            model: "mistral-large-latest".to_string(),
            choices: vec![MistralToolChoice2 {
                index: 0,
                message: MistralToolMessage {
                    role: "assistant".to_string(),
                    content: Some("I will answer directly.".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        assert!(!resp.has_tool_calls());
        assert!(resp.tool_calls().is_empty());
    }

    // ── Provider construction ─────────────────────────────────────────

    #[test]
    fn test_provider_construction_fails_without_api_key() {
        let cfg = ProviderConfig {
            api_key: None,
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let result = MistralProvider::new(cfg);
        assert!(result.is_err());
        let msg = result.err().expect("has err").to_string();
        assert!(msg.contains("API key"));
    }

    #[test]
    fn test_provider_construction_succeeds() {
        let cfg = ProviderConfig {
            api_key: Some("test-key".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        assert!(MistralProvider::new(cfg).is_ok());
    }

    #[test]
    fn test_available_models_includes_defaults() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = MistralProvider::new(cfg).expect("construct");
        let models = p.get_available_models();
        assert!(models.contains(&"mistral-large-latest".to_string()));
        assert!(models.contains(&"mistral-embed".to_string()));
        assert!(models.contains(&"codestral-latest".to_string()));
    }

    #[test]
    fn test_get_provider_name() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = MistralProvider::new(cfg).expect("construct");
        assert_eq!(p.get_provider_name(), "mistral");
    }

    #[test]
    fn test_supports_streaming() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = MistralProvider::new(cfg).expect("construct");
        assert!(p.supports_streaming());
    }

    // ── Cost ──────────────────────────────────────────────────────────

    #[test]
    fn test_cost_estimation_codestral() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = MistralProvider::new(cfg).expect("construct");
        // codestral-latest: $0.001 input / $0.003 output per 1K tokens
        let cost = p.estimate_cost("codestral-latest", 1000, 1000);
        let expected = 0.001 + 0.003;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_cost_embed_output_zero() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = MistralProvider::new(cfg).expect("construct");
        // mistral-embed: $0.0001 input / $0.0 output per 1K tokens
        let cost = p.estimate_cost("mistral-embed", 1000, 0);
        let expected = 0.0001;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_cost_estimation_unknown_model() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = MistralProvider::new(cfg).expect("construct");
        // Custom fallback: $0.002 / $0.006
        let cost = p.estimate_cost("future-model", 1000, 1000);
        let expected = 0.002 + 0.006;
        assert!((cost - expected).abs() < 1e-9);
    }

    // ── Tool-enabled request serialization ────────────────────────────

    #[test]
    fn test_tool_chat_request_serialization() {
        let req = MistralToolChatRequest {
            model: "mistral-large-latest".to_string(),
            messages: vec![MistralMessage::user("Execute this SPARQL query")],
            temperature: Some(0.0),
            max_tokens: Some(512),
            tools: Some(vec![MistralTool::function(MistralFunction {
                name: "run_sparql".to_string(),
                description: "Execute SPARQL".to_string(),
                parameters: FunctionParameters {
                    schema_type: "object".to_string(),
                    properties: std::collections::HashMap::new(),
                    required: vec![],
                },
            })]),
            tool_choice: Some("auto".to_string()),
            stream: None,
            safe_prompt: None,
            random_seed: None,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(json.contains("run_sparql"));
        assert!(json.contains("auto"));
        assert!(!json.contains("\"stream\""));
    }

    #[test]
    fn test_mistral_model_embed_not_chat() {
        assert!(!MistralModel::MistralEmbed.supports_chat());
        assert!(MistralModel::MistralSmallLatest.supports_chat());
        assert!(MistralModel::CodestralLatest.supports_chat());
        assert!(MistralModel::OpenMixtral8x22b.supports_chat());
    }
}
