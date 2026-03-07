//! Groq Provider Implementation
//!
//! Implements the LLM provider trait for Groq's inference API.
//! Groq uses an OpenAI-compatible REST API with extremely fast inference via LPU hardware.
//!
//! Supported models: llama-3.1-8b-instant, llama-3.1-70b-versatile,
//! mixtral-8x7b-32768, gemma-7b-it, and others.

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

/// Groq model variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroqModel {
    /// Llama 3.1 8B Instant – ultra-low latency
    Llama31_8bInstant,
    /// Llama 3.1 70B Versatile – high quality
    Llama31_70bVersatile,
    /// Llama 3 8B (8192 context)
    Llama3_8b8192,
    /// Llama 3 70B (8192 context)
    Llama3_70b8192,
    /// Mixtral 8x7B (32768 context)
    Mixtral8x7b32768,
    /// Gemma 7B Instruction
    Gemma7bIt,
    /// Gemma2 9B Instruction
    Gemma2_9bIt,
    /// Custom / future models
    Custom(String),
}

impl GroqModel {
    pub fn model_id(&self) -> &str {
        match self {
            Self::Llama31_8bInstant => "llama-3.1-8b-instant",
            Self::Llama31_70bVersatile => "llama-3.1-70b-versatile",
            Self::Llama3_8b8192 => "llama3-8b-8192",
            Self::Llama3_70b8192 => "llama3-70b-8192",
            Self::Mixtral8x7b32768 => "mixtral-8x7b-32768",
            Self::Gemma7bIt => "gemma-7b-it",
            Self::Gemma2_9bIt => "gemma2-9b-it",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Groq pricing per 1K tokens (input_price, output_price) in USD.
    /// Groq is extremely cheap; these are approximate figures.
    pub fn cost_per_1k_tokens(&self) -> (f64, f64) {
        match self {
            Self::Llama31_8bInstant | Self::Llama3_8b8192 => (0.00005, 0.00008),
            Self::Llama31_70bVersatile | Self::Llama3_70b8192 => (0.00059, 0.00079),
            Self::Mixtral8x7b32768 => (0.00024, 0.00024),
            Self::Gemma7bIt => (0.00007, 0.00007),
            Self::Gemma2_9bIt => (0.0002, 0.0002),
            Self::Custom(_) => (0.0002, 0.0002),
        }
    }
}

impl std::fmt::Display for GroqModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model_id())
    }
}

/// Chat message in OpenAI-compatible format used by Groq
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqMessage {
    /// "system" | "user" | "assistant"
    pub role: String,
    pub content: String,
}

impl GroqMessage {
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

/// Groq chat completion request (OpenAI-compatible)
#[derive(Debug, Serialize)]
pub struct GroqChatRequest {
    pub model: String,
    pub messages: Vec<GroqMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// Groq token usage including Groq-specific timing fields
#[derive(Debug, Deserialize)]
pub struct GroqUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Seconds spent processing the prompt
    pub prompt_time: Option<f64>,
    /// Seconds spent generating completions
    pub completion_time: Option<f64>,
    /// Total wall-clock seconds
    pub total_time: Option<f64>,
}

/// A single choice in the Groq response
#[derive(Debug, Deserialize)]
pub struct GroqChoice {
    pub index: u32,
    pub message: GroqMessage,
    pub finish_reason: Option<String>,
}

/// Groq chat completion response
#[derive(Debug, Deserialize)]
pub struct GroqChatResponse {
    pub id: String,
    pub object: Option<String>,
    pub created: Option<u64>,
    pub model: String,
    pub choices: Vec<GroqChoice>,
    pub usage: Option<GroqUsage>,
    /// Groq-specific metadata (request_id, timing, etc.)
    pub x_groq: Option<serde_json::Value>,
}

impl GroqChatResponse {
    /// Extract the first choice's message content, or empty string if absent.
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

/// Groq provider implementing the unified LLMProvider trait
pub struct GroqProvider {
    api_key: String,
    config: ProviderConfig,
    client: reqwest::Client,
    base_url: String,
}

impl GroqProvider {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow!("Groq API key not provided"))?
            .clone();

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.groq.com".to_string());

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;

        Ok(Self {
            api_key,
            config,
            client,
            base_url,
        })
    }

    /// Convert an LLMRequest to Groq's OpenAI-compatible message list
    fn build_messages(&self, request: &LLMRequest) -> Vec<GroqMessage> {
        let mut messages: Vec<GroqMessage> = Vec::new();

        if let Some(ref sp) = request.system_prompt {
            messages.push(GroqMessage::system(sp.clone()));
        }

        for msg in &request.messages {
            match msg.role {
                ChatRole::System => messages.push(GroqMessage::system(msg.content.clone())),
                ChatRole::User => messages.push(GroqMessage::user(msg.content.clone())),
                ChatRole::Assistant => messages.push(GroqMessage::assistant(msg.content.clone())),
            }
        }

        messages
    }

    /// Send a raw GroqChatRequest and parse the response
    async fn send_request(&self, groq_req: &GroqChatRequest) -> Result<GroqChatResponse> {
        debug!("Sending request to Groq API model={}", groq_req.model);

        let response = self
            .client
            .post(format!("{}/openai/v1/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(groq_req)
            .send()
            .await
            .map_err(|e| anyhow!("Groq HTTP request failed: {}", e))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read Groq response body: {}", e))?;

        if !status.is_success() {
            error!("Groq API error: {} - {}", status, body);
            return Err(anyhow!("Groq API error {}: {}", status, body));
        }

        let parsed: GroqChatResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow!("Failed to parse Groq response: {} - body: {}", e, body))?;

        Ok(parsed)
    }
}

#[async_trait]
impl LLMProvider for GroqProvider {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse> {
        let started_at = Instant::now();
        let messages = self.build_messages(request);

        let groq_req = GroqChatRequest {
            model: model.to_string(),
            messages,
            temperature: Some(request.temperature as f64),
            max_tokens: request.max_tokens.map(|t| t as u32),
            stream: None,
            top_p: None,
            stop: None,
        };

        let groq_resp = self.send_request(&groq_req).await?;
        let latency = started_at.elapsed();

        let (prompt_tokens, completion_tokens, total_tokens) = groq_resp.token_counts();
        let cost = self.estimate_cost(model, prompt_tokens, completion_tokens);

        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        metadata.insert(
            "response_id".to_string(),
            serde_json::Value::String(groq_resp.id.clone()),
        );
        if let Some(x_groq) = &groq_resp.x_groq {
            metadata.insert("x_groq".to_string(), x_groq.clone());
        }
        if let Some(choice) = groq_resp.choices.first() {
            if let Some(ref finish_reason) = choice.finish_reason {
                metadata.insert(
                    "finish_reason".to_string(),
                    serde_json::Value::String(finish_reason.clone()),
                );
            }
        }

        Ok(LLMResponse {
            content: groq_resp.first_content().to_string(),
            model_used: groq_resp.model.clone(),
            provider_used: "groq".to_string(),
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
            },
            latency,
            quality_score: Some(0.82),
            metadata,
        })
    }

    async fn generate_stream(
        &self,
        model: &str,
        request: &LLMRequest,
    ) -> Result<LLMResponseStream> {
        // Groq supports SSE streaming; simulate with full response for now
        let response = self.generate(model, request).await?;
        let words: Vec<String> = response
            .content
            .split_whitespace()
            .map(String::from)
            .collect();
        let chunk_size = 5usize;

        let model_name = model.to_string();
        let provider_name = "groq".to_string();
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
            provider_used: "groq".to_string(),
            started_at,
        })
    }

    fn get_available_models(&self) -> Vec<String> {
        let config_models: Vec<String> =
            self.config.models.iter().map(|m| m.name.clone()).collect();

        let defaults = vec![
            "llama-3.1-8b-instant".to_string(),
            "llama-3.1-70b-versatile".to_string(),
            "llama3-8b-8192".to_string(),
            "llama3-70b-8192".to_string(),
            "mixtral-8x7b-32768".to_string(),
            "gemma-7b-it".to_string(),
            "gemma2-9b-it".to_string(),
        ];

        let mut all: std::collections::HashSet<String> =
            config_models.into_iter().chain(defaults).collect();
        // keep deterministic order
        let mut sorted: Vec<String> = all.drain().collect();
        sorted.sort();
        sorted
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn get_provider_name(&self) -> &str {
        "groq"
    }

    fn estimate_cost(&self, model: &str, input_tokens: usize, output_tokens: usize) -> f64 {
        let groq_model = match model {
            "llama-3.1-8b-instant" | "llama3-8b-8192" => GroqModel::Llama31_8bInstant,
            "llama-3.1-70b-versatile" | "llama3-70b-8192" => GroqModel::Llama31_70bVersatile,
            "mixtral-8x7b-32768" => GroqModel::Mixtral8x7b32768,
            "gemma-7b-it" => GroqModel::Gemma7bIt,
            "gemma2-9b-it" => GroqModel::Gemma2_9bIt,
            _ => GroqModel::Custom(model.to_string()),
        };
        let (ip, op) = groq_model.cost_per_1k_tokens();
        (input_tokens as f64 * ip / 1000.0) + (output_tokens as f64 * op / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groq_model_ids() {
        assert_eq!(
            GroqModel::Llama31_8bInstant.model_id(),
            "llama-3.1-8b-instant"
        );
        assert_eq!(
            GroqModel::Llama31_70bVersatile.model_id(),
            "llama-3.1-70b-versatile"
        );
        assert_eq!(GroqModel::Mixtral8x7b32768.model_id(), "mixtral-8x7b-32768");
        assert_eq!(GroqModel::Gemma7bIt.model_id(), "gemma-7b-it");
        assert_eq!(GroqModel::Gemma2_9bIt.model_id(), "gemma2-9b-it");
        assert_eq!(GroqModel::Custom("my-llm".to_string()).model_id(), "my-llm");
    }

    #[test]
    fn test_groq_message_construction() {
        let sys = GroqMessage::system("Be helpful.");
        assert_eq!(sys.role, "system");
        let usr = GroqMessage::user("Hello");
        assert_eq!(usr.role, "user");
        let asst = GroqMessage::assistant("Hi!");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_groq_request_serialization() {
        let req = GroqChatRequest {
            model: "llama-3.1-8b-instant".to_string(),
            messages: vec![
                GroqMessage::system("Be terse."),
                GroqMessage::user("What is SPARQL?"),
            ],
            temperature: Some(0.5),
            max_tokens: Some(256),
            stream: None,
            top_p: None,
            stop: None,
        };

        let json = serde_json::to_string(&req).expect("serialization must succeed");
        assert!(json.contains("llama-3.1-8b-instant"));
        assert!(json.contains("system"));
        assert!(json.contains("user"));
        assert!(json.contains("SPARQL"));
        // None fields must be omitted
        assert!(!json.contains("\"stream\""));
        assert!(!json.contains("\"top_p\""));
    }

    #[test]
    fn test_groq_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-xyz",
            "object": "chat.completion",
            "created": 1714000000,
            "model": "llama-3.1-8b-instant",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "SPARQL is a query language for RDF."},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "total_tokens": 42,
                "prompt_time": 0.01,
                "completion_time": 0.05,
                "total_time": 0.06
            },
            "x_groq": {"id": "req_abc"}
        }"#;

        let resp: GroqChatResponse =
            serde_json::from_str(json).expect("deserialization must succeed");
        assert_eq!(resp.id, "chatcmpl-xyz");
        assert_eq!(resp.first_content(), "SPARQL is a query language for RDF.");
        let (p, c, t) = resp.token_counts();
        assert_eq!(p, 30);
        assert_eq!(c, 12);
        assert_eq!(t, 42);
        assert!(resp.x_groq.is_some());
    }

    #[test]
    fn test_groq_cost_estimation() {
        // llama-3.1-8b-instant: $0.00005 input / $0.00008 output per 1K tokens
        let (ip, op) = GroqModel::Llama31_8bInstant.cost_per_1k_tokens();
        let cost = (1000.0 * ip / 1000.0) + (500.0 * op / 1000.0);
        // 1K input + 500 output
        let expected = 0.00005 + 0.00004;
        assert!((cost - expected).abs() < 1e-10);
    }

    #[test]
    fn test_groq_model_display() {
        assert_eq!(
            format!("{}", GroqModel::Mixtral8x7b32768),
            "mixtral-8x7b-32768"
        );
    }
}

// ────────────────────────────────────────────────────────────────────
// Groq speed metrics: parsed from x_groq metadata and usage timing
// ────────────────────────────────────────────────────────────────────

/// Speed metrics extracted from a Groq API response.
/// Groq's LPU hardware provides extremely fast inference; these metrics
/// allow callers to track tokens/sec and latency budgets.
#[derive(Debug, Clone, Default)]
pub struct GroqSpeedMetrics {
    /// Time in seconds to process the prompt
    pub prompt_time_secs: f64,
    /// Time in seconds to generate the completion
    pub completion_time_secs: f64,
    /// Total wall-clock time in seconds
    pub total_time_secs: f64,
    /// Tokens per second (prompt phase)
    pub prompt_tokens_per_sec: f64,
    /// Tokens per second (completion phase)
    pub completion_tokens_per_sec: f64,
    /// Groq request ID from x_groq metadata
    pub request_id: Option<String>,
}

impl GroqSpeedMetrics {
    /// Parse speed metrics from a `GroqChatResponse`.
    ///
    /// Groq embeds timing data in the `usage` field and request
    /// identification in the `x_groq` metadata field.
    pub fn from_response(resp: &GroqChatResponse) -> Self {
        let (prompt_tokens, completion_tokens, _) = resp.token_counts();

        let (prompt_time, completion_time, total_time) = resp
            .usage
            .as_ref()
            .map(|u| {
                (
                    u.prompt_time.unwrap_or(0.0),
                    u.completion_time.unwrap_or(0.0),
                    u.total_time.unwrap_or(0.0),
                )
            })
            .unwrap_or((0.0, 0.0, 0.0));

        let prompt_tokens_per_sec = if prompt_time > 0.0 {
            prompt_tokens as f64 / prompt_time
        } else {
            0.0
        };

        let completion_tokens_per_sec = if completion_time > 0.0 {
            completion_tokens as f64 / completion_time
        } else {
            0.0
        };

        let request_id = resp
            .x_groq
            .as_ref()
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        GroqSpeedMetrics {
            prompt_time_secs: prompt_time,
            completion_time_secs: completion_time,
            total_time_secs: total_time,
            prompt_tokens_per_sec,
            completion_tokens_per_sec,
            request_id,
        }
    }

    /// Whether the speed metrics indicate ultra-fast Groq LPU inference
    /// (> 100 completion tokens/sec is a typical Groq benchmark).
    pub fn is_ultra_fast(&self) -> bool {
        self.completion_tokens_per_sec > 100.0
    }
}

impl GroqProvider {
    /// Parse speed metrics from a completed response
    pub fn parse_speed_metrics(resp: &GroqChatResponse) -> GroqSpeedMetrics {
        GroqSpeedMetrics::from_response(resp)
    }
}

// ────────────────────────────────────────────────────────────────────
// Extended tests – targeting 20+ total for this module
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod extended_tests {
    use super::*;

    fn make_usage(
        prompt: u32,
        completion: u32,
        prompt_time: f64,
        completion_time: f64,
    ) -> GroqUsage {
        GroqUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            prompt_time: Some(prompt_time),
            completion_time: Some(completion_time),
            total_time: Some(prompt_time + completion_time),
        }
    }

    fn make_response_with_timing(
        prompt: u32,
        completion: u32,
        prompt_time: f64,
        completion_time: f64,
        request_id: Option<&str>,
    ) -> GroqChatResponse {
        let x_groq = request_id.map(|id| serde_json::json!({"id": id}));
        GroqChatResponse {
            id: "chatcmpl-test".to_string(),
            object: Some("chat.completion".to_string()),
            created: Some(1714000000),
            model: "llama-3.1-8b-instant".to_string(),
            choices: vec![GroqChoice {
                index: 0,
                message: GroqMessage::assistant("Test response"),
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(make_usage(prompt, completion, prompt_time, completion_time)),
            x_groq,
        }
    }

    // ── Speed metrics ─────────────────────────────────────────────────

    #[test]
    fn test_speed_metrics_basic() {
        let resp = make_response_with_timing(100, 200, 0.5, 1.0, Some("req-abc"));
        let metrics = GroqSpeedMetrics::from_response(&resp);

        assert!((metrics.prompt_time_secs - 0.5).abs() < 1e-9);
        assert!((metrics.completion_time_secs - 1.0).abs() < 1e-9);
        assert!((metrics.total_time_secs - 1.5).abs() < 1e-9);
        assert_eq!(metrics.request_id.as_deref(), Some("req-abc"));
    }

    #[test]
    fn test_speed_metrics_tokens_per_sec() {
        // 100 prompt tokens in 0.5s = 200 t/s
        // 200 completion tokens in 1.0s = 200 t/s
        let resp = make_response_with_timing(100, 200, 0.5, 1.0, None);
        let metrics = GroqSpeedMetrics::from_response(&resp);
        assert!((metrics.prompt_tokens_per_sec - 200.0).abs() < 1e-6);
        assert!((metrics.completion_tokens_per_sec - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_speed_metrics_ultra_fast() {
        // 500 completion tokens in 1.0s = 500 t/s > 100 threshold
        let resp = make_response_with_timing(50, 500, 0.01, 1.0, None);
        let metrics = GroqSpeedMetrics::from_response(&resp);
        assert!(metrics.is_ultra_fast());
    }

    #[test]
    fn test_speed_metrics_not_ultra_fast() {
        // 50 completion tokens in 2.0s = 25 t/s < 100 threshold
        let resp = make_response_with_timing(50, 50, 0.5, 2.0, None);
        let metrics = GroqSpeedMetrics::from_response(&resp);
        assert!(!metrics.is_ultra_fast());
    }

    #[test]
    fn test_speed_metrics_zero_time() {
        let resp = make_response_with_timing(100, 100, 0.0, 0.0, None);
        let metrics = GroqSpeedMetrics::from_response(&resp);
        // Should not divide by zero
        assert!((metrics.prompt_tokens_per_sec - 0.0).abs() < 1e-9);
        assert!((metrics.completion_tokens_per_sec - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_speed_metrics_missing_usage() {
        let resp = GroqChatResponse {
            id: "test".to_string(),
            object: None,
            created: None,
            model: "llama-3.1-8b-instant".to_string(),
            choices: vec![],
            usage: None,
            x_groq: None,
        };
        let metrics = GroqSpeedMetrics::from_response(&resp);
        assert!((metrics.total_time_secs - 0.0).abs() < 1e-9);
        assert!(metrics.request_id.is_none());
    }

    #[test]
    fn test_speed_metrics_request_id_extraction() {
        let resp = make_response_with_timing(10, 10, 0.01, 0.05, Some("groq-req-12345"));
        let metrics = GroqSpeedMetrics::from_response(&resp);
        assert_eq!(metrics.request_id.as_deref(), Some("groq-req-12345"));
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
        let result = GroqProvider::new(cfg);
        assert!(result.is_err());
        let msg = result.err().expect("has err").to_string();
        assert!(msg.contains("API key"));
    }

    #[test]
    fn test_provider_construction_succeeds() {
        let cfg = ProviderConfig {
            api_key: Some("gsk_test".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        assert!(GroqProvider::new(cfg).is_ok());
    }

    #[test]
    fn test_provider_custom_base_url() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: Some("https://proxy.groq.local".to_string()),
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = GroqProvider::new(cfg).expect("construct");
        assert_eq!(p.base_url, "https://proxy.groq.local");
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
        let p = GroqProvider::new(cfg).expect("construct");
        let models = p.get_available_models();
        assert!(models.contains(&"llama-3.1-8b-instant".to_string()));
        assert!(models.contains(&"mixtral-8x7b-32768".to_string()));
        assert!(models.contains(&"gemma-7b-it".to_string()));
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
        let p = GroqProvider::new(cfg).expect("construct");
        assert_eq!(p.get_provider_name(), "groq");
    }

    // ── Message construction ──────────────────────────────────────────

    #[test]
    fn test_groq_message_content() {
        let msg = GroqMessage::user("Query the RDF graph");
        assert_eq!(msg.content, "Query the RDF graph");
    }

    #[test]
    fn test_groq_message_system_role() {
        let msg = GroqMessage::system("You are an RDF expert.");
        assert_eq!(msg.role, "system");
    }

    #[test]
    fn test_groq_message_assistant_role() {
        let msg = GroqMessage::assistant("The SPARQL result is...");
        assert_eq!(msg.role, "assistant");
    }

    // ── Cost estimation ───────────────────────────────────────────────

    #[test]
    fn test_cost_estimation_mixtral() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = GroqProvider::new(cfg).expect("construct");
        // mixtral-8x7b-32768: $0.00024 per 1K for both input and output
        let cost = p.estimate_cost("mixtral-8x7b-32768", 1000, 1000);
        let expected = 0.00024 + 0.00024;
        assert!((cost - expected).abs() < 1e-10);
    }

    #[test]
    fn test_cost_estimation_llama_70b() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = GroqProvider::new(cfg).expect("construct");
        let cost = p.estimate_cost("llama-3.1-70b-versatile", 0, 0);
        assert!((cost - 0.0).abs() < 1e-12);
    }

    // ── Response helpers ──────────────────────────────────────────────

    #[test]
    fn test_first_content_empty_choices() {
        let resp = GroqChatResponse {
            id: "test".to_string(),
            object: None,
            created: None,
            model: "llama-3.1-8b-instant".to_string(),
            choices: vec![],
            usage: None,
            x_groq: None,
        };
        assert_eq!(resp.first_content(), "");
    }

    #[test]
    fn test_token_counts_no_usage() {
        let resp = GroqChatResponse {
            id: "test".to_string(),
            object: None,
            created: None,
            model: "llama-3.1-8b-instant".to_string(),
            choices: vec![],
            usage: None,
            x_groq: None,
        };
        let (p, c, t) = resp.token_counts();
        assert_eq!(p, 0);
        assert_eq!(c, 0);
        assert_eq!(t, 0);
    }

    #[test]
    fn test_groq_model_custom_fallback_cost() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = GroqProvider::new(cfg).expect("construct");
        // Unknown model falls back to Custom with $0.0002/$0.0002
        let cost = p.estimate_cost("unknown-llm", 1000, 1000);
        let expected = 0.0002 + 0.0002;
        assert!((cost - expected).abs() < 1e-10);
    }
}
