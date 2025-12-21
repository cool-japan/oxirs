//! Core types for re-ranking

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type for re-ranking operations
pub type RerankingResult<T> = std::result::Result<T, RerankingError>;

/// Errors that can occur during re-ranking
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum RerankingError {
    #[error("Model not loaded: {model_name}")]
    ModelNotLoaded { model_name: String },

    #[error("Model error: {message}")]
    ModelError { message: String },

    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Batch size exceeded: {size} > {max}")]
    BatchSizeExceeded { size: usize, max: usize },

    #[error("Cache error: {message}")]
    CacheError { message: String },

    #[error("Score fusion error: {message}")]
    FusionError { message: String },

    #[error("API error: {message}")]
    ApiError { message: String },

    #[error("Timeout: operation took longer than {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Backend error: {message}")]
    BackendError { message: String },

    #[error("Internal error: {message}")]
    InternalError { message: String },
}

/// Candidate document with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredCandidate {
    /// Document ID
    pub id: String,

    /// Original retrieval score (from bi-encoder)
    pub retrieval_score: f32,

    /// Re-ranking score (from cross-encoder)
    pub reranking_score: Option<f32>,

    /// Fused final score
    pub final_score: f32,

    /// Document content (optional)
    pub content: Option<String>,

    /// Metadata
    pub metadata: std::collections::HashMap<String, String>,

    /// Original rank in retrieval results
    pub original_rank: usize,
}

impl ScoredCandidate {
    /// Create new candidate
    pub fn new(id: impl Into<String>, retrieval_score: f32, original_rank: usize) -> Self {
        Self {
            id: id.into(),
            retrieval_score,
            reranking_score: None,
            final_score: retrieval_score,
            content: None,
            metadata: std::collections::HashMap::new(),
            original_rank,
        }
    }

    /// Set re-ranking score
    pub fn with_reranking_score(mut self, score: f32) -> Self {
        self.reranking_score = Some(score);
        self.final_score = score;
        self
    }

    /// Set content
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get effective score (reranking if available, otherwise retrieval)
    pub fn effective_score(&self) -> f32 {
        self.reranking_score.unwrap_or(self.retrieval_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scored_candidate_creation() {
        let candidate = ScoredCandidate::new("doc1", 0.85, 0);
        assert_eq!(candidate.id, "doc1");
        assert_eq!(candidate.retrieval_score, 0.85);
        assert_eq!(candidate.final_score, 0.85);
        assert_eq!(candidate.original_rank, 0);
        assert!(candidate.reranking_score.is_none());
    }

    #[test]
    fn test_candidate_with_reranking() {
        let candidate = ScoredCandidate::new("doc1", 0.85, 0).with_reranking_score(0.92);

        assert_eq!(candidate.retrieval_score, 0.85);
        assert_eq!(candidate.reranking_score, Some(0.92));
        assert_eq!(candidate.final_score, 0.92);
        assert_eq!(candidate.effective_score(), 0.92);
    }

    #[test]
    fn test_candidate_with_content() {
        let candidate = ScoredCandidate::new("doc1", 0.85, 0).with_content("Test document");

        assert_eq!(candidate.content, Some("Test document".to_string()));
    }

    #[test]
    fn test_candidate_with_metadata() {
        let candidate = ScoredCandidate::new("doc1", 0.85, 0)
            .with_metadata("source", "wikipedia")
            .with_metadata("lang", "en");

        assert_eq!(
            candidate.metadata.get("source"),
            Some(&"wikipedia".to_string())
        );
        assert_eq!(candidate.metadata.get("lang"), Some(&"en".to_string()));
    }

    #[test]
    fn test_effective_score() {
        let mut candidate = ScoredCandidate::new("doc1", 0.85, 0);
        assert_eq!(candidate.effective_score(), 0.85);

        candidate.reranking_score = Some(0.92);
        assert_eq!(candidate.effective_score(), 0.92);
    }

    #[test]
    fn test_error_display() {
        let err = RerankingError::ModelNotLoaded {
            model_name: "cross-encoder-ms-marco".to_string(),
        };
        assert!(err.to_string().contains("cross-encoder-ms-marco"));

        let err = RerankingError::BatchSizeExceeded { size: 100, max: 50 };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));

        let err = RerankingError::Timeout { timeout_ms: 5000 };
        assert!(err.to_string().contains("5000"));
    }
}
