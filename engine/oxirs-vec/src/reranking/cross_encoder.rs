//! Cross-encoder model wrapper for query-document relevance scoring
//!
//! Cross-encoders jointly encode query and document pairs to produce
//! accurate relevance scores. Unlike bi-encoders, they see both inputs
//! together, enabling fine-grained relevance modeling at higher computational cost.

use crate::reranking::types::{RerankingError, RerankingResult};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Backend for cross-encoder inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossEncoderBackend {
    /// Local model inference (PyTorch/ONNX)
    Local,
    /// API-based inference (OpenAI, Cohere, etc.)
    Api,
    /// Remote inference server
    Remote,
    /// Mock backend for testing
    Mock,
}

/// Trait for cross-encoder backends
pub trait CrossEncoderBackendTrait: Send + Sync {
    /// Score a single query-document pair
    fn score(&self, query: &str, document: &str) -> RerankingResult<f32>;

    /// Score multiple query-document pairs in batch
    fn batch_score(&self, pairs: &[(String, String)]) -> RerankingResult<Vec<f32>> {
        // Default implementation: score one by one
        pairs.iter().map(|(q, d)| self.score(q, d)).collect()
    }
}

/// Local model backend using sentence transformers
#[derive(Debug, Clone)]
pub struct LocalBackend {
    model_name: String,
    max_length: usize,
    device: String,
    model_loaded: Arc<RwLock<bool>>,
}

impl LocalBackend {
    pub fn new(model_name: String, max_length: usize, device: String) -> Self {
        Self {
            model_name,
            max_length,
            device,
            model_loaded: Arc::new(RwLock::new(false)),
        }
    }

    fn ensure_loaded(&self) -> RerankingResult<()> {
        let mut loaded = self
            .model_loaded
            .write()
            .map_err(|e| RerankingError::BackendError {
                message: format!("Lock poisoned: {}", e),
            })?;

        if !*loaded {
            tracing::info!("Loading cross-encoder model: {}", self.model_name);
            // In real implementation: load model using tokenizers + PyTorch/ONNX
            // For now, mark as loaded
            *loaded = true;
        }
        Ok(())
    }

    fn compute_similarity(&self, query: &str, document: &str) -> f32 {
        // Simplified similarity computation
        // Real implementation would use transformer model inference

        // Normalize texts
        let q = query.to_lowercase();
        let d = document.to_lowercase();

        // Exact match bonus
        if d.contains(&q) {
            return 0.95;
        }

        // Word overlap score
        let q_words: Vec<&str> = q.split_whitespace().collect();
        let d_words: Vec<&str> = d.split_whitespace().collect();

        if q_words.is_empty() {
            return 0.5;
        }

        let overlap_count = q_words
            .iter()
            .filter(|qw| d_words.iter().any(|dw| dw.contains(*qw) || qw.contains(dw)))
            .count();

        let overlap_ratio = overlap_count as f32 / q_words.len() as f32;

        // Length penalty for very short or very long documents
        let doc_len = d_words.len();
        let length_factor = if doc_len < 10 {
            0.8
        } else if doc_len > 500 {
            0.85
        } else {
            1.0
        };

        // Combine scores
        let base_score = 0.4 + overlap_ratio * 0.5;
        (base_score * length_factor).min(0.99)
    }
}

impl CrossEncoderBackendTrait for LocalBackend {
    fn score(&self, query: &str, document: &str) -> RerankingResult<f32> {
        self.ensure_loaded()?;

        if query.is_empty() || document.is_empty() {
            return Ok(0.0);
        }

        let score = self.compute_similarity(query, document);
        Ok(score)
    }

    fn batch_score(&self, pairs: &[(String, String)]) -> RerankingResult<Vec<f32>> {
        self.ensure_loaded()?;

        // Real implementation would batch through model inference
        // For now, process individually
        Ok(pairs
            .iter()
            .map(|(q, d)| self.compute_similarity(q, d))
            .collect())
    }
}

/// API-based backend (e.g., Cohere Rerank API)
#[derive(Debug, Clone)]
pub struct ApiBackend {
    api_key: String,
    endpoint: String,
    model: String,
    timeout_ms: u64,
}

impl ApiBackend {
    pub fn new(api_key: String, endpoint: String, model: String, timeout_ms: u64) -> Self {
        Self {
            api_key,
            endpoint,
            model,
            timeout_ms,
        }
    }
}

impl CrossEncoderBackendTrait for ApiBackend {
    fn score(&self, query: &str, document: &str) -> RerankingResult<f32> {
        // In real implementation: make API call to reranking service
        // For now, return mock score
        tracing::debug!(
            "API reranking: {} chars query, {} chars doc",
            query.len(),
            document.len()
        );

        // Simulate API delay with random jitter
        let mut rng = Random::seed(42);
        let base_score = rng.gen_range(0.4..0.9);
        Ok(base_score)
    }

    fn batch_score(&self, pairs: &[(String, String)]) -> RerankingResult<Vec<f32>> {
        // Real implementation would batch API call
        tracing::debug!("Batch API reranking: {} pairs", pairs.len());

        let mut rng = Random::seed(42);
        Ok(pairs.iter().map(|_| rng.gen_range(0.4..0.9)).collect())
    }
}

/// Mock backend for testing
#[derive(Debug, Clone)]
pub struct MockBackend {
    scores: Arc<RwLock<HashMap<String, f32>>>,
}

impl MockBackend {
    pub fn new() -> Self {
        Self {
            scores: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn set_score(&self, query: &str, document: &str, score: f32) {
        let key = format!("{}||{}", query, document);
        if let Ok(mut scores) = self.scores.write() {
            scores.insert(key, score);
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEncoderBackendTrait for MockBackend {
    fn score(&self, query: &str, document: &str) -> RerankingResult<f32> {
        let key = format!("{}||{}", query, document);

        if let Ok(scores) = self.scores.read() {
            if let Some(&score) = scores.get(&key) {
                return Ok(score);
            }
        }

        // Default mock score based on text overlap
        let overlap = query
            .split_whitespace()
            .filter(|w| document.contains(w))
            .count();

        let query_words = query.split_whitespace().count().max(1);
        let score = 0.5 + (overlap as f32 / query_words as f32) * 0.4;

        Ok(score.min(0.95))
    }
}

/// Cross-encoder model for relevance scoring
#[derive(Clone)]
pub struct CrossEncoder {
    model_name: String,
    backend: Arc<dyn CrossEncoderBackendTrait>,
    batch_size: usize,
}

impl CrossEncoder {
    /// Create new cross-encoder with specified backend
    pub fn new(model_name: &str, backend_type: &str) -> RerankingResult<Self> {
        let backend: Arc<dyn CrossEncoderBackendTrait> = match backend_type {
            "local" => Arc::new(LocalBackend::new(
                model_name.to_string(),
                512,
                "cpu".to_string(),
            )),
            "api" => {
                // Read API key from environment
                let api_key =
                    std::env::var("RERANK_API_KEY").unwrap_or_else(|_| "mock_api_key".to_string());

                Arc::new(ApiBackend::new(
                    api_key,
                    "https://api.cohere.ai/v1/rerank".to_string(),
                    model_name.to_string(),
                    5000,
                ))
            }
            "mock" => Arc::new(MockBackend::new()),
            _ => {
                return Err(RerankingError::InvalidConfiguration {
                    message: format!("Unknown backend type: {}", backend_type),
                });
            }
        };

        Ok(Self {
            model_name: model_name.to_string(),
            backend,
            batch_size: 32,
        })
    }

    /// Create with mock backend for testing
    pub fn with_mock_backend() -> Self {
        Self {
            model_name: "mock".to_string(),
            backend: Arc::new(MockBackend::new()),
            batch_size: 32,
        }
    }

    /// Score a single query-document pair
    pub fn score(&self, query: &str, document: &str) -> RerankingResult<f32> {
        self.backend.score(query, document)
    }

    /// Score multiple query-document pairs in batch
    pub fn batch_score(&self, pairs: &[(String, String)]) -> RerankingResult<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        // Process in batches
        let mut all_scores = Vec::with_capacity(pairs.len());

        for chunk in pairs.chunks(self.batch_size) {
            let scores = self.backend.batch_score(chunk)?;
            all_scores.extend(scores);
        }

        Ok(all_scores)
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Set batch size
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_backend_basic() {
        let backend = LocalBackend::new(
            "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            512,
            "cpu".to_string(),
        );

        let score = backend
            .score("machine learning", "deep learning tutorial")
            .unwrap();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_local_backend_exact_match() {
        let backend = LocalBackend::new("test-model".to_string(), 512, "cpu".to_string());

        let score = backend
            .score("rust programming", "This is about rust programming")
            .unwrap();
        assert!(score > 0.9);
    }

    #[test]
    fn test_local_backend_no_match() {
        let backend = LocalBackend::new("test-model".to_string(), 512, "cpu".to_string());

        let score = backend.score("python", "javascript tutorial").unwrap();
        assert!(score < 0.6);
    }

    #[test]
    fn test_mock_backend() {
        let backend = MockBackend::new();
        backend.set_score("test", "document", 0.85);

        let score = backend.score("test", "document").unwrap();
        assert!((score - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_cross_encoder_creation() {
        let encoder = CrossEncoder::new("ms-marco-MiniLM", "local").unwrap();
        assert_eq!(encoder.model_name(), "ms-marco-MiniLM");
    }

    #[test]
    fn test_cross_encoder_scoring() {
        let encoder = CrossEncoder::with_mock_backend();
        let score = encoder.score("query", "relevant document").unwrap();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_batch_scoring() {
        let encoder = CrossEncoder::with_mock_backend();
        let pairs = vec![
            ("query1".to_string(), "doc1".to_string()),
            ("query2".to_string(), "doc2".to_string()),
            ("query3".to_string(), "doc3".to_string()),
        ];

        let scores = encoder.batch_score(&pairs).unwrap();
        assert_eq!(scores.len(), 3);

        for score in scores {
            assert!((0.0..=1.0).contains(&score));
        }
    }

    #[test]
    fn test_empty_input() {
        let backend = LocalBackend::new("test-model".to_string(), 512, "cpu".to_string());

        let score = backend.score("", "document").unwrap();
        assert_eq!(score, 0.0);

        let score = backend.score("query", "").unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_invalid_backend() {
        let result = CrossEncoder::new("model", "invalid_backend");
        assert!(result.is_err());
    }
}
