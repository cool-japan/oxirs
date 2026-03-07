//! Cohere Provider Implementation
//!
//! Implements the LLM provider trait for Cohere's Command models using the Cohere REST API.
//! Supports: Command, Command-R, Command-R+, Command-Light

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

/// Cohere chat model variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CohereModel {
    Command,
    CommandR,
    CommandRPlus,
    CommandLight,
    Custom(String),
}

impl CohereModel {
    pub fn model_id(&self) -> &str {
        match self {
            Self::Command => "command",
            Self::CommandR => "command-r",
            Self::CommandRPlus => "command-r-plus",
            Self::CommandLight => "command-light",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Estimate cost per 1K tokens (input_price, output_price)
    pub fn cost_per_1k_tokens(&self) -> (f64, f64) {
        match self {
            Self::CommandRPlus => (0.003, 0.015),
            Self::CommandR => (0.0005, 0.0015),
            Self::Command => (0.001, 0.002),
            Self::CommandLight => (0.0003, 0.0006),
            Self::Custom(_) => (0.001, 0.002),
        }
    }
}

impl std::fmt::Display for CohereModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model_id())
    }
}

/// Chat message for Cohere API (role is "USER" or "CHATBOT")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereChatMessage {
    pub role: String,
    pub message: String,
}

impl CohereChatMessage {
    pub fn user(message: impl Into<String>) -> Self {
        Self {
            role: "USER".to_string(),
            message: message.into(),
        }
    }

    pub fn chatbot(message: impl Into<String>) -> Self {
        Self {
            role: "CHATBOT".to_string(),
            message: message.into(),
        }
    }
}

/// Cohere chat request body
#[derive(Debug, Serialize)]
pub struct CohereChatRequest {
    pub model: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_history: Option<Vec<CohereChatMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preamble: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Cohere API billed units metadata
#[derive(Debug, Deserialize)]
pub struct CohereUsage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

/// Cohere API-level metadata
#[derive(Debug, Deserialize)]
pub struct CohereMetadata {
    pub api_version: Option<serde_json::Value>,
    pub billed_units: Option<CohereUsage>,
}

/// Cohere chat API response
#[derive(Debug, Deserialize)]
pub struct CohereChatResponse {
    pub text: String,
    pub generation_id: Option<String>,
    pub chat_history: Option<Vec<CohereChatMessage>>,
    pub finish_reason: Option<String>,
    pub meta: Option<CohereMetadata>,
}

impl CohereChatResponse {
    /// Extract input/output token counts from metadata
    pub fn token_usage(&self) -> (usize, usize) {
        let input = self
            .meta
            .as_ref()
            .and_then(|m| m.billed_units.as_ref())
            .and_then(|u| u.input_tokens)
            .unwrap_or(0) as usize;
        let output = self
            .meta
            .as_ref()
            .and_then(|m| m.billed_units.as_ref())
            .and_then(|u| u.output_tokens)
            .unwrap_or(0) as usize;
        (input, output)
    }
}

/// Cohere provider implementing the unified LLMProvider trait
pub struct CohereProvider {
    api_key: String,
    config: ProviderConfig,
    client: reqwest::Client,
    base_url: String,
}

impl CohereProvider {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow!("Cohere API key not provided"))?
            .clone();

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.cohere.com".to_string());

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

    /// Construct a Cohere chat request from an LLMRequest
    fn build_cohere_request(&self, model: &str, request: &LLMRequest) -> CohereChatRequest {
        let mut chat_history: Vec<CohereChatMessage> = Vec::new();
        let mut last_user_message = String::new();

        // Build preamble from system prompts
        let mut preamble_parts: Vec<String> = Vec::new();
        if let Some(ref sp) = request.system_prompt {
            preamble_parts.push(sp.clone());
        }

        for msg in &request.messages {
            match msg.role {
                ChatRole::System => {
                    preamble_parts.push(msg.content.clone());
                }
                ChatRole::User => {
                    // If we already had a user message queued, move it to history
                    if !last_user_message.is_empty() {
                        chat_history.push(CohereChatMessage::user(last_user_message.clone()));
                        last_user_message.clear();
                    }
                    last_user_message = msg.content.clone();
                }
                ChatRole::Assistant => {
                    if !last_user_message.is_empty() {
                        chat_history.push(CohereChatMessage::user(last_user_message.clone()));
                        last_user_message.clear();
                    }
                    chat_history.push(CohereChatMessage::chatbot(msg.content.clone()));
                }
            }
        }

        // The final user message becomes `message`; everything before becomes `chat_history`
        let preamble = if preamble_parts.is_empty() {
            None
        } else {
            Some(preamble_parts.join("\n\n"))
        };

        CohereChatRequest {
            model: model.to_string(),
            message: last_user_message,
            chat_history: if chat_history.is_empty() {
                None
            } else {
                Some(chat_history)
            },
            preamble,
            temperature: Some(request.temperature as f64),
            max_tokens: request.max_tokens.map(|t| t as u32),
            stream: None,
        }
    }

    /// Send a raw Cohere chat request and return the parsed response
    async fn send_request(&self, cohere_req: &CohereChatRequest) -> Result<CohereChatResponse> {
        debug!("Sending request to Cohere API model={}", cohere_req.model);

        let response = self
            .client
            .post(format!("{}/v2/chat", self.base_url))
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(cohere_req)
            .send()
            .await
            .map_err(|e| anyhow!("Cohere HTTP request failed: {}", e))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read Cohere response body: {}", e))?;

        if !status.is_success() {
            error!("Cohere API error: {} - {}", status, body);
            return Err(anyhow!("Cohere API error {}: {}", status, body));
        }

        let parsed: CohereChatResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow!("Failed to parse Cohere response: {} - body: {}", e, body))?;

        Ok(parsed)
    }
}

#[async_trait]
impl LLMProvider for CohereProvider {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse> {
        let started_at = Instant::now();
        let cohere_req = self.build_cohere_request(model, request);
        let cohere_resp = self.send_request(&cohere_req).await?;
        let latency = started_at.elapsed();

        let (input_tokens, output_tokens) = cohere_resp.token_usage();
        let total_tokens = input_tokens + output_tokens;
        let cost = self.estimate_cost(model, input_tokens, output_tokens);

        let mut metadata = HashMap::new();
        if let Some(ref gen_id) = cohere_resp.generation_id {
            metadata.insert(
                "generation_id".to_string(),
                serde_json::Value::String(gen_id.clone()),
            );
        }
        if let Some(ref finish) = cohere_resp.finish_reason {
            metadata.insert(
                "finish_reason".to_string(),
                serde_json::Value::String(finish.clone()),
            );
        }

        Ok(LLMResponse {
            content: cohere_resp.text,
            model_used: model.to_string(),
            provider_used: "cohere".to_string(),
            usage: Usage {
                prompt_tokens: input_tokens,
                completion_tokens: output_tokens,
                total_tokens,
                cost,
            },
            latency,
            quality_score: Some(0.80),
            metadata,
        })
    }

    async fn generate_stream(
        &self,
        model: &str,
        request: &LLMRequest,
    ) -> Result<LLMResponseStream> {
        // Cohere streaming requires SSE parsing; simulate with full response for now
        let response = self.generate(model, request).await?;
        let words: Vec<String> = response
            .content
            .split_whitespace()
            .map(String::from)
            .collect();
        let chunk_size = 5usize;

        let model_name = model.to_string();
        let provider_name = "cohere".to_string();
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
            provider_used: "cohere".to_string(),
            started_at,
        })
    }

    fn get_available_models(&self) -> Vec<String> {
        self.config
            .models
            .iter()
            .map(|m| m.name.clone())
            .collect::<Vec<_>>()
            .into_iter()
            .chain([
                "command-r-plus".to_string(),
                "command-r".to_string(),
                "command".to_string(),
                "command-light".to_string(),
            ])
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn get_provider_name(&self) -> &str {
        "cohere"
    }

    fn estimate_cost(&self, model: &str, input_tokens: usize, output_tokens: usize) -> f64 {
        let cohere_model = match model {
            "command-r-plus" => CohereModel::CommandRPlus,
            "command-r" => CohereModel::CommandR,
            "command-light" => CohereModel::CommandLight,
            _ => CohereModel::Command,
        };
        let (input_price, output_price) = cohere_model.cost_per_1k_tokens();
        (input_tokens as f64 * input_price / 1000.0)
            + (output_tokens as f64 * output_price / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohere_model_ids() {
        assert_eq!(CohereModel::Command.model_id(), "command");
        assert_eq!(CohereModel::CommandR.model_id(), "command-r");
        assert_eq!(CohereModel::CommandRPlus.model_id(), "command-r-plus");
        assert_eq!(CohereModel::CommandLight.model_id(), "command-light");
        assert_eq!(
            CohereModel::Custom("my-model".to_string()).model_id(),
            "my-model"
        );
    }

    #[test]
    fn test_cohere_message_roles() {
        let user_msg = CohereChatMessage::user("Hello");
        assert_eq!(user_msg.role, "USER");
        assert_eq!(user_msg.message, "Hello");

        let bot_msg = CohereChatMessage::chatbot("World");
        assert_eq!(bot_msg.role, "CHATBOT");
        assert_eq!(bot_msg.message, "World");
    }

    #[test]
    fn test_cohere_request_serialization() {
        let req = CohereChatRequest {
            model: "command-r".to_string(),
            message: "What is RDF?".to_string(),
            chat_history: Some(vec![
                CohereChatMessage::user("Hello"),
                CohereChatMessage::chatbot("Hi there!"),
            ]),
            preamble: Some("You are an expert in RDF.".to_string()),
            temperature: Some(0.7),
            max_tokens: Some(512),
            stream: None,
        };

        let serialized = serde_json::to_string(&req).expect("serialization must succeed");
        assert!(serialized.contains("command-r"));
        assert!(serialized.contains("What is RDF?"));
        assert!(serialized.contains("USER"));
        assert!(serialized.contains("CHATBOT"));
        // stream=None should be omitted
        assert!(!serialized.contains("\"stream\""));
    }

    #[test]
    fn test_cohere_response_deserialization() {
        let json = r#"{
            "text": "RDF stands for Resource Description Framework.",
            "generation_id": "gen-abc-123",
            "finish_reason": "COMPLETE",
            "meta": {
                "api_version": {"version": "2"},
                "billed_units": {
                    "input_tokens": 42,
                    "output_tokens": 10
                }
            }
        }"#;

        let resp: CohereChatResponse =
            serde_json::from_str(json).expect("deserialization must succeed");
        assert_eq!(resp.text, "RDF stands for Resource Description Framework.");
        assert_eq!(resp.generation_id.as_deref(), Some("gen-abc-123"));
        assert_eq!(resp.finish_reason.as_deref(), Some("COMPLETE"));

        let (input, output) = resp.token_usage();
        assert_eq!(input, 42);
        assert_eq!(output, 10);
    }

    #[test]
    fn test_cohere_cost_estimation() {
        // Command-R-Plus: $0.003 / 1K input, $0.015 / 1K output
        let (ip, op) = CohereModel::CommandRPlus.cost_per_1k_tokens();
        let cost = (1000.0 * ip / 1000.0) + (1000.0 * op / 1000.0);
        assert!((cost - 0.018).abs() < 1e-9);
    }

    #[test]
    fn test_cohere_model_display() {
        assert_eq!(format!("{}", CohereModel::CommandR), "command-r");
    }
}

// ────────────────────────────────────────────────────────────────────
// Cohere Reranking API
// ────────────────────────────────────────────────────────────────────

/// A document passed to the Cohere Rerank endpoint
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankDocument {
    /// Document text (flat string)
    pub text: String,
}

/// Request body for `POST /v1/rerank`
#[derive(Debug, Serialize)]
pub struct CohereRerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<RerankDocument>,
    pub top_n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
}

/// One result from the Cohere Rerank endpoint
#[derive(Debug, Deserialize, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f64,
    pub document: Option<RerankDocument>,
}

/// Response from `POST /v1/rerank`
#[derive(Debug, Deserialize)]
pub struct CohereRerankResponse {
    pub results: Vec<RerankResult>,
    pub id: Option<String>,
    pub meta: Option<CohereMetadata>,
}

// ────────────────────────────────────────────────────────────────────
// Cohere Embeddings API
// ────────────────────────────────────────────────────────────────────

/// Input type for Cohere Embed endpoint
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CohereEmbedInputType {
    SearchDocument,
    SearchQuery,
    Classification,
    Clustering,
}

/// Request body for `POST /v1/embed`
#[derive(Debug, Serialize)]
pub struct CohereEmbedRequest {
    pub model: String,
    pub texts: Vec<String>,
    pub input_type: CohereEmbedInputType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<String>,
}

/// Response from `POST /v1/embed`
#[derive(Debug, Deserialize)]
pub struct CohereEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub id: Option<String>,
    pub meta: Option<CohereMetadata>,
}

/// Groq speed metrics parsed from response headers / x_groq metadata
#[derive(Debug, Clone, Default)]
pub struct CohereSpeedMetrics {
    pub generation_id: Option<String>,
    pub finish_reason: Option<String>,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

impl CohereProvider {
    /// Rerank a list of documents against a query using Cohere's Rerank API.
    ///
    /// Returns results sorted by descending relevance score, capped to `top_n`.
    pub async fn rerank(
        &self,
        query: impl Into<String>,
        documents: Vec<String>,
        model: &str,
        top_n: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let docs: Vec<RerankDocument> = documents
            .into_iter()
            .map(|text| RerankDocument { text })
            .collect();

        let req = CohereRerankRequest {
            model: model.to_string(),
            query: query.into(),
            documents: docs,
            top_n,
            return_documents: Some(true),
        };

        debug!("Sending rerank request to Cohere API model={}", req.model);

        let response = self
            .client
            .post(format!("{}/v1/rerank", self.base_url))
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| anyhow!("Cohere rerank HTTP request failed: {}", e))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read Cohere rerank response: {}", e))?;

        if !status.is_success() {
            error!("Cohere rerank API error: {} - {}", status, body);
            return Err(anyhow!("Cohere rerank API error {}: {}", status, body));
        }

        let parsed: CohereRerankResponse = serde_json::from_str(&body).map_err(|e| {
            anyhow!(
                "Failed to parse Cohere rerank response: {} body: {}",
                e,
                body
            )
        })?;

        Ok(parsed.results)
    }

    /// Compute embeddings for a batch of texts using Cohere's Embed API.
    pub async fn embed(
        &self,
        texts: Vec<String>,
        model: &str,
        input_type: CohereEmbedInputType,
    ) -> Result<Vec<Vec<f32>>> {
        let req = CohereEmbedRequest {
            model: model.to_string(),
            texts,
            input_type,
            truncate: Some("END".to_string()),
        };

        debug!("Sending embed request to Cohere API model={}", req.model);

        let response = self
            .client
            .post(format!("{}/v1/embed", self.base_url))
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| anyhow!("Cohere embed HTTP request failed: {}", e))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read Cohere embed response: {}", e))?;

        if !status.is_success() {
            error!("Cohere embed API error: {} - {}", status, body);
            return Err(anyhow!("Cohere embed API error {}: {}", status, body));
        }

        let parsed: CohereEmbedResponse = serde_json::from_str(&body).map_err(|e| {
            anyhow!(
                "Failed to parse Cohere embed response: {} body: {}",
                e,
                body
            )
        })?;

        Ok(parsed.embeddings)
    }

    /// Parse speed / usage metrics from a `CohereChatResponse`.
    pub fn parse_speed_metrics(resp: &CohereChatResponse) -> CohereSpeedMetrics {
        let (input_tokens, output_tokens) = resp.token_usage();
        CohereSpeedMetrics {
            generation_id: resp.generation_id.clone(),
            finish_reason: resp.finish_reason.clone(),
            input_tokens,
            output_tokens,
        }
    }
}

// Additional tests – targeting 20+ total
#[cfg(test)]
mod extended_tests {
    use super::*;

    // ── Rerank structs ────────────────────────────────────────────────

    #[test]
    fn test_rerank_document_roundtrip() {
        let doc = RerankDocument {
            text: "Semantic web technologies enable linked data.".to_string(),
        };
        let json = serde_json::to_string(&doc).expect("serialize");
        let back: RerankDocument = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.text, doc.text);
    }

    #[test]
    fn test_rerank_request_serialization() {
        let req = CohereRerankRequest {
            model: "rerank-english-v3.0".to_string(),
            query: "What is SPARQL?".to_string(),
            documents: vec![
                RerankDocument {
                    text: "SPARQL is a query language for RDF.".to_string(),
                },
                RerankDocument {
                    text: "Turtle is a syntax for RDF.".to_string(),
                },
            ],
            top_n: Some(1),
            return_documents: Some(true),
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(json.contains("rerank-english-v3.0"));
        assert!(json.contains("SPARQL"));
        assert!(json.contains("top_n"));
    }

    #[test]
    fn test_rerank_response_deserialization() {
        let json = r#"{
            "id": "rerank-abc",
            "results": [
                {"index": 0, "relevance_score": 0.95, "document": {"text": "SPARQL is a query language."}},
                {"index": 1, "relevance_score": 0.42, "document": {"text": "Turtle is an RDF syntax."}}
            ]
        }"#;
        let resp: CohereRerankResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(resp.results.len(), 2);
        assert!((resp.results[0].relevance_score - 0.95).abs() < 1e-6);
        assert_eq!(resp.results[0].index, 0);
    }

    #[test]
    fn test_rerank_results_ordering_by_score() {
        let mut results: Vec<RerankResult> = vec![
            RerankResult {
                index: 2,
                relevance_score: 0.30,
                document: None,
            },
            RerankResult {
                index: 0,
                relevance_score: 0.95,
                document: None,
            },
            RerankResult {
                index: 1,
                relevance_score: 0.60,
                document: None,
            },
        ];
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .expect("cmp")
        });
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].index, 1);
        assert_eq!(results[2].index, 2);
    }

    // ── Embed structs ─────────────────────────────────────────────────

    #[test]
    fn test_embed_input_type_serialization() {
        let search_doc = CohereEmbedInputType::SearchDocument;
        let json = serde_json::to_string(&search_doc).expect("serialize");
        assert_eq!(json, r#""search_document""#);

        let search_query = CohereEmbedInputType::SearchQuery;
        let json2 = serde_json::to_string(&search_query).expect("serialize");
        assert_eq!(json2, r#""search_query""#);
    }

    #[test]
    fn test_embed_request_serialization() {
        let req = CohereEmbedRequest {
            model: "embed-english-v3.0".to_string(),
            texts: vec!["RDF triple store".to_string(), "SPARQL query".to_string()],
            input_type: CohereEmbedInputType::SearchDocument,
            truncate: Some("END".to_string()),
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(json.contains("embed-english-v3.0"));
        assert!(json.contains("search_document"));
        assert!(json.contains("RDF triple store"));
    }

    #[test]
    fn test_embed_response_deserialization() {
        let json = r#"{
            "id": "embed-xyz",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }"#;
        let resp: CohereEmbedResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(resp.embeddings.len(), 2);
        assert_eq!(resp.embeddings[0].len(), 3);
        assert!((resp.embeddings[0][0] - 0.1f32).abs() < 1e-6);
    }

    #[test]
    fn test_embed_classification_type() {
        let t = CohereEmbedInputType::Classification;
        let json = serde_json::to_string(&t).expect("serialize");
        assert_eq!(json, r#""classification""#);
    }

    #[test]
    fn test_embed_clustering_type() {
        let t = CohereEmbedInputType::Clustering;
        let json = serde_json::to_string(&t).expect("serialize");
        assert_eq!(json, r#""clustering""#);
    }

    // ── Speed metrics ─────────────────────────────────────────────────

    #[test]
    fn test_speed_metrics_from_response() {
        let resp = CohereChatResponse {
            text: "Answer".to_string(),
            generation_id: Some("gen-111".to_string()),
            chat_history: None,
            finish_reason: Some("COMPLETE".to_string()),
            meta: Some(CohereMetadata {
                api_version: None,
                billed_units: Some(CohereUsage {
                    input_tokens: Some(50),
                    output_tokens: Some(25),
                }),
            }),
        };
        let metrics = CohereProvider::parse_speed_metrics(&resp);
        assert_eq!(metrics.generation_id.as_deref(), Some("gen-111"));
        assert_eq!(metrics.finish_reason.as_deref(), Some("COMPLETE"));
        assert_eq!(metrics.input_tokens, 50);
        assert_eq!(metrics.output_tokens, 25);
    }

    #[test]
    fn test_speed_metrics_defaults_on_empty_meta() {
        let resp = CohereChatResponse {
            text: "Hello".to_string(),
            generation_id: None,
            chat_history: None,
            finish_reason: None,
            meta: None,
        };
        let metrics = CohereProvider::parse_speed_metrics(&resp);
        assert!(metrics.generation_id.is_none());
        assert_eq!(metrics.input_tokens, 0);
        assert_eq!(metrics.output_tokens, 0);
    }

    // ── Provider construction & config ────────────────────────────────

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
        let result = CohereProvider::new(cfg);
        assert!(result.is_err());
        let msg = result.err().expect("has err").to_string();
        assert!(msg.contains("API key"));
    }

    #[test]
    fn test_provider_construction_succeeds_with_api_key() {
        let cfg = ProviderConfig {
            api_key: Some("test-key".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let result = CohereProvider::new(cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_provider_custom_base_url() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: Some("https://proxy.example.com".to_string()),
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = CohereProvider::new(cfg).expect("construct");
        assert_eq!(p.base_url, "https://proxy.example.com");
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
        let p = CohereProvider::new(cfg).expect("construct");
        let models = p.get_available_models();
        assert!(models.contains(&"command-r-plus".to_string()));
        assert!(models.contains(&"command-r".to_string()));
        assert!(models.contains(&"command".to_string()));
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
        let p = CohereProvider::new(cfg).expect("construct");
        assert_eq!(p.get_provider_name(), "cohere");
    }

    #[test]
    fn test_supports_streaming_true() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = CohereProvider::new(cfg).expect("construct");
        assert!(p.supports_streaming());
    }

    // ── Cost estimation edge cases ────────────────────────────────────

    #[test]
    fn test_cost_estimation_zero_tokens() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = CohereProvider::new(cfg).expect("construct");
        let cost = p.estimate_cost("command-r", 0, 0);
        assert!((cost - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_cost_estimation_command_model() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = CohereProvider::new(cfg).expect("construct");
        // command: $0.001 input + $0.002 output per 1K
        let cost = p.estimate_cost("command", 2000, 1000);
        let expected = 2.0 * 0.001 + 1.0 * 0.002;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_cost_estimation_command_light() {
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = CohereProvider::new(cfg).expect("construct");
        // command-light: $0.0003 input + $0.0006 output per 1K
        let cost = p.estimate_cost("command-light", 1000, 1000);
        let expected = 0.0003 + 0.0006;
        assert!((cost - expected).abs() < 1e-9);
    }

    // ── Build cohere request ──────────────────────────────────────────

    #[test]
    fn test_build_cohere_request_system_prompt() {
        use super::super::types::{ChatMessage, ChatRole, Priority, UseCase};
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = CohereProvider::new(cfg).expect("construct");
        let request = LLMRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Hello".to_string(),
                metadata: None,
            }],
            system_prompt: Some("Be helpful.".to_string()),
            temperature: 0.7,
            max_tokens: Some(256),
            use_case: UseCase::SimpleQuery,
            priority: Priority::Normal,
            timeout: None,
        };
        let cohere_req = p.build_cohere_request("command-r", &request);
        assert_eq!(cohere_req.message, "Hello");
        assert!(cohere_req.preamble.is_some());
        assert!(cohere_req
            .preamble
            .as_ref()
            .expect("preamble")
            .contains("Be helpful"));
    }

    #[test]
    fn test_build_cohere_request_multi_turn() {
        use super::super::types::{ChatMessage, ChatRole, Priority, UseCase};
        let cfg = ProviderConfig {
            api_key: Some("k".to_string()),
            base_url: None,
            models: vec![],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
            enabled: true,
        };
        let p = CohereProvider::new(cfg).expect("construct");
        let request = LLMRequest {
            messages: vec![
                ChatMessage {
                    role: ChatRole::User,
                    content: "First question".to_string(),
                    metadata: None,
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "First answer".to_string(),
                    metadata: None,
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Second question".to_string(),
                    metadata: None,
                },
            ],
            system_prompt: None,
            temperature: 0.5,
            max_tokens: None,
            use_case: UseCase::Conversation,
            priority: Priority::Normal,
            timeout: None,
        };
        let cohere_req = p.build_cohere_request("command-r", &request);
        // The final user message becomes `message`
        assert_eq!(cohere_req.message, "Second question");
        // Previous turns become chat_history
        let history = cohere_req.chat_history.as_ref().expect("history present");
        assert!(!history.is_empty());
    }
}
