//! OpenAI Provider Implementation
//!
//! Implements the LLM provider trait for OpenAI's GPT models using async-openai crate.

use anyhow::{anyhow, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageContent,
        CreateChatCompletionRequestArgs,
    },
    Client as OpenAIClient,
};
use async_trait::async_trait;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use super::{
    config::ProviderConfig,
    providers::LLMProvider,
    types::{ChatRole, LLMRequest, LLMResponse, LLMResponseChunk, LLMResponseStream, Usage},
};

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

#[async_trait]
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
        use futures_util::StreamExt;

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
                            .and_then(|choice| choice.delta.content.as_ref()).cloned()
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
