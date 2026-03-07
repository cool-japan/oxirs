//! # OpenAIEmbeddingGenerator - make_request_group Methods
//!
//! This module contains method implementations for `OpenAIEmbeddingGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::types::RetryStrategy;

use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;

impl OpenAIEmbeddingGenerator {
    /// Make API request to OpenAI with retry logic
    pub(super) async fn make_request(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let start_time = std::time::Instant::now();
        let mut attempts = 0;
        while attempts < self.openai_config.max_retries {
            match self.try_request(texts).await {
                Ok(embeddings) => {
                    if self.openai_config.enable_metrics {
                        let response_time = start_time.elapsed().as_millis() as f64;
                        self.update_response_time(response_time);
                        let cost = self.calculate_cost(texts);
                        self.metrics.total_cost_usd += cost;
                        *self
                            .metrics
                            .requests_by_model
                            .entry(self.openai_config.model.clone())
                            .or_insert(0) += 1;
                    }
                    return Ok(embeddings);
                }
                Err(e) => {
                    attempts += 1;
                    self.metrics.retry_count += 1;
                    let error_type = if e.to_string().contains("rate_limit") {
                        "rate_limit"
                    } else if e.to_string().contains("timeout") {
                        "timeout"
                    } else if e.to_string().contains("401") {
                        "unauthorized"
                    } else if e.to_string().contains("400") {
                        "bad_request"
                    } else {
                        "other"
                    };
                    *self
                        .metrics
                        .errors_by_type
                        .entry(error_type.to_string())
                        .or_insert(0) += 1;
                    if attempts >= self.openai_config.max_retries {
                        return Err(e);
                    }
                    let delay = self.calculate_retry_delay(attempts);
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                }
            }
        }
        Err(anyhow!("Max retries exceeded"))
    }
    /// Calculate retry delay based on strategy
    fn calculate_retry_delay(&self, attempt: u32) -> u64 {
        let base_delay = self.openai_config.retry_delay_ms;
        match self.openai_config.retry_strategy {
            RetryStrategy::Fixed => base_delay,
            RetryStrategy::LinearBackoff => base_delay * attempt as u64,
            RetryStrategy::ExponentialBackoff => {
                let delay = base_delay * (2_u64.pow(attempt - 1));
                let jitter = {
                    #[allow(unused_imports)]
                    use scirs2_core::random::{Random, Rng};
                    let mut rng = Random::seed(42);
                    (delay as f64 * 0.25 * (rng.gen_range(0.0..1.0) - 0.5)) as u64
                };
                delay.saturating_add(jitter).min(30000)
            }
        }
    }
    /// Update response time metrics
    fn update_response_time(&mut self, response_time_ms: f64) {
        if self.metrics.successful_requests == 0 {
            self.metrics.average_response_time_ms = response_time_ms;
        } else {
            let total =
                self.metrics.average_response_time_ms * self.metrics.successful_requests as f64;
            self.metrics.average_response_time_ms =
                (total + response_time_ms) / (self.metrics.successful_requests + 1) as f64;
        }
    }
}
