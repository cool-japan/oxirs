//! Anthropic Provider Implementation
//!
//! Implements the LLM provider trait for Anthropic's Claude models using reqwest HTTP client.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json;
use std::{collections::HashMap, time::{Duration, Instant}};
use tracing::{debug, error, warn};
use futures_util::StreamExt;

use super::{
    config::ProviderConfig,
    providers::LLMProvider,
    types::{ChatRole, LLMRequest, LLMResponse, LLMResponseChunk, LLMResponseStream, Usage},
};

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

#[async_trait]
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