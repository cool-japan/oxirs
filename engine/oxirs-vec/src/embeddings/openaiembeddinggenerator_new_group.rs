//! # OpenAIEmbeddingGenerator - new_group Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Vector;
use anyhow::{anyhow, Result};
use std::time::Duration;

use super::types::{
    CachedEmbedding, EmbeddableContent, EmbeddingConfig, OpenAIConfig, OpenAIMetrics, RateLimiter,
};

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    pub fn new(openai_config: OpenAIConfig) -> Result<Self> {
        openai_config.validate()?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                openai_config.timeout_seconds,
            ))
            .user_agent(&openai_config.user_agent)
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;
        let embedding_config = EmbeddingConfig {
            model_name: openai_config.model.clone(),
            dimensions: Self::get_model_dimensions(&openai_config.model),
            max_sequence_length: 8191,
            normalize: true,
        };
        let cache_size = if openai_config.enable_cache {
            std::num::NonZeroUsize::new(openai_config.cache_size)
                .unwrap_or(std::num::NonZeroUsize::new(1000).expect("constant 1000 is non-zero"))
        } else {
            std::num::NonZeroUsize::new(1).expect("constant 1 is non-zero")
        };
        Ok(Self {
            config: embedding_config,
            openai_config: openai_config.clone(),
            client,
            rate_limiter: RateLimiter::new(openai_config.requests_per_minute),
            request_cache: std::sync::Arc::new(std::sync::Mutex::new(lru::LruCache::new(
                cache_size,
            ))),
            metrics: OpenAIMetrics::default(),
        })
    }
    /// Get dimensions for different OpenAI models
    fn get_model_dimensions(model: &str) -> usize {
        match model {
            "text-embedding-ada-002" => 1536,
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-004" => 1536,
            _ => 1536,
        }
    }
    pub(super) async fn try_request(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.rate_limiter.wait_if_needed().await;
        let request_body = serde_json::json!(
            { "model" : self.openai_config.model, "input" : texts, "encoding_format" :
            "float" }
        );
        let response = self
            .client
            .post(format!("{}/embeddings", self.openai_config.base_url))
            .header(
                "Authorization",
                format!("Bearer {}", self.openai_config.api_key),
            )
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("Request failed: {}", e))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "API request failed with status {}: {}",
                status,
                error_text
            ));
        }
        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse response: {}", e))?;
        let embeddings_data = response_data["data"]
            .as_array()
            .ok_or_else(|| anyhow!("Invalid response format: missing data array"))?;
        let mut embeddings = Vec::new();
        for item in embeddings_data {
            let embedding = item["embedding"]
                .as_array()
                .ok_or_else(|| anyhow!("Invalid response format: missing embedding"))?;
            let vec: Result<Vec<f32>, _> = embedding
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| anyhow!("Invalid embedding value"))
                        .map(|f| f as f32)
                })
                .collect();
            embeddings.push(vec?);
        }
        Ok(embeddings)
    }
    /// Generate embeddings with batching support
    pub async fn generate_async(&mut self, content: &EmbeddableContent) -> Result<Vector> {
        let text = content.to_text();
        if self.openai_config.enable_cache {
            let hash = content.content_hash();
            let cached_vector = match self.request_cache.lock() {
                Ok(mut cache) => {
                    if let Some(cached) = cache.get(&hash) {
                        let is_valid = cached.cached_at.elapsed().unwrap_or_default()
                            < Duration::from_secs(self.openai_config.cache_ttl_seconds);
                        if is_valid {
                            Some(cached.vector.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(result) = cached_vector {
                self.update_cache_hit();
                return Ok(result);
            } else {
                if let Ok(mut cache) = self.request_cache.lock() {
                    cache.pop(&hash);
                }
                self.update_cache_miss();
            }
        }
        let embeddings = match self.make_request(std::slice::from_ref(&text)).await {
            Ok(embeddings) => {
                self.update_metrics_success(std::slice::from_ref(&text));
                embeddings
            }
            Err(e) => {
                self.update_metrics_failure();
                return Err(e);
            }
        };
        if embeddings.is_empty() {
            self.update_metrics_failure();
            return Err(anyhow!("No embeddings returned from API"));
        }
        let vector = Vector::new(embeddings[0].clone());
        if self.openai_config.enable_cache {
            let hash = content.content_hash();
            let cost = self.calculate_cost(std::slice::from_ref(&text));
            let cached_embedding = CachedEmbedding {
                vector: vector.clone(),
                cached_at: std::time::SystemTime::now(),
                model: self.openai_config.model.clone(),
                cost_usd: cost,
            };
            if let Ok(mut cache) = self.request_cache.lock() {
                cache.put(hash, cached_embedding);
            }
        }
        Ok(vector)
    }
    /// Generate embeddings for multiple texts in batch
    pub async fn generate_batch_async(
        &mut self,
        contents: &[EmbeddableContent],
    ) -> Result<Vec<Vector>> {
        if contents.is_empty() {
            return Ok(Vec::new());
        }
        let mut results = Vec::with_capacity(contents.len());
        let batch_size = self.openai_config.batch_size;
        for chunk in contents.chunks(batch_size) {
            let texts: Vec<String> = chunk.iter().map(|c| c.to_text()).collect();
            let embeddings = match self.make_request(&texts).await {
                Ok(embeddings) => {
                    self.update_metrics_success(&texts);
                    embeddings
                }
                Err(e) => {
                    self.update_metrics_failure();
                    return Err(e);
                }
            };
            if embeddings.len() != chunk.len() {
                self.update_metrics_failure();
                return Err(anyhow!("Mismatch between request and response sizes"));
            }
            let batch_cost = self.calculate_cost(&texts) / chunk.len() as f64;
            for (content, embedding) in chunk.iter().zip(embeddings) {
                let vector = Vector::new(embedding);
                if self.openai_config.enable_cache {
                    let hash = content.content_hash();
                    let cached_embedding = CachedEmbedding {
                        vector: vector.clone(),
                        cached_at: std::time::SystemTime::now(),
                        model: self.openai_config.model.clone(),
                        cost_usd: batch_cost,
                    };
                    if let Ok(mut cache) = self.request_cache.lock() {
                        cache.put(hash, cached_embedding);
                    }
                }
                results.push(vector);
            }
        }
        Ok(results)
    }
}
