//! Core types for hybrid search

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A hybrid search query combining keyword and semantic components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridQuery {
    /// The query text for keyword search
    pub query_text: String,
    /// The query vector for semantic search (optional, can be computed from text)
    pub query_vector: Option<Vec<f32>>,
    /// Maximum number of results to return
    pub top_k: usize,
    /// Weights for combining keyword and semantic scores
    pub weights: SearchWeights,
    /// Metadata filters
    pub filters: HashMap<String, String>,
}

/// Weights for combining different search signals
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SearchWeights {
    /// Weight for keyword search (0.0 - 1.0)
    pub keyword_weight: f32,
    /// Weight for semantic search (0.0 - 1.0)
    pub semantic_weight: f32,
    /// Weight for recency/freshness (0.0 - 1.0)
    pub recency_weight: f32,
}

impl Default for SearchWeights {
    fn default() -> Self {
        Self {
            keyword_weight: 0.3,
            semantic_weight: 0.7,
            recency_weight: 0.0,
        }
    }
}

impl SearchWeights {
    /// Validate weights (should sum to approximately 1.0)
    pub fn validate(&self) -> anyhow::Result<()> {
        let sum = self.keyword_weight + self.semantic_weight + self.recency_weight;
        if (sum - 1.0).abs() > 0.1 {
            anyhow::bail!(
                "Search weights should sum to approximately 1.0, got {}",
                sum
            );
        }
        if self.keyword_weight < 0.0 || self.semantic_weight < 0.0 || self.recency_weight < 0.0 {
            anyhow::bail!("Search weights must be non-negative");
        }
        Ok(())
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.keyword_weight + self.semantic_weight + self.recency_weight;
        if sum > 0.0 {
            self.keyword_weight /= sum;
            self.semantic_weight /= sum;
            self.recency_weight /= sum;
        }
    }
}

/// Result from hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    /// Document ID
    pub doc_id: String,
    /// Final combined score
    pub score: f32,
    /// Breakdown of score components
    pub score_breakdown: ScoreBreakdown,
    /// Document metadata
    pub metadata: HashMap<String, String>,
}

/// Breakdown of score components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Keyword search score
    pub keyword_score: f32,
    /// Semantic search score
    pub semantic_score: f32,
    /// Recency score
    pub recency_score: f32,
    /// Rank in keyword results (0-based)
    pub keyword_rank: Option<usize>,
    /// Rank in semantic results (0-based)
    pub semantic_rank: Option<usize>,
}

/// Document score from a single search method
#[derive(Debug, Clone)]
pub struct DocumentScore {
    /// Document ID
    pub doc_id: String,
    /// Score
    pub score: f32,
    /// Rank (0-based)
    pub rank: usize,
}

/// Keyword match information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordMatch {
    /// Document ID
    pub doc_id: String,
    /// BM25 or TF-IDF score
    pub score: f32,
    /// Matched terms
    pub matched_terms: Vec<String>,
    /// Term frequencies in document
    pub term_frequencies: HashMap<String, usize>,
}

impl HybridResult {
    /// Create a new hybrid result
    pub fn new(
        doc_id: String,
        keyword_score: f32,
        semantic_score: f32,
        recency_score: f32,
        weights: &SearchWeights,
    ) -> Self {
        let score = keyword_score * weights.keyword_weight
            + semantic_score * weights.semantic_weight
            + recency_score * weights.recency_weight;

        Self {
            doc_id,
            score,
            score_breakdown: ScoreBreakdown {
                keyword_score,
                semantic_score,
                recency_score,
                keyword_rank: None,
                semantic_rank: None,
            },
            metadata: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_weights_validation() {
        let weights = SearchWeights {
            keyword_weight: 0.3,
            semantic_weight: 0.7,
            recency_weight: 0.0,
        };
        assert!(weights.validate().is_ok());

        let bad_weights = SearchWeights {
            keyword_weight: 0.5,
            semantic_weight: 0.8,
            recency_weight: 0.0,
        };
        assert!(bad_weights.validate().is_err());
    }

    #[test]
    fn test_weights_normalization() {
        let mut weights = SearchWeights {
            keyword_weight: 1.0,
            semantic_weight: 2.0,
            recency_weight: 1.0,
        };
        weights.normalize();
        assert!((weights.keyword_weight - 0.25).abs() < 0.001);
        assert!((weights.semantic_weight - 0.5).abs() < 0.001);
        assert!((weights.recency_weight - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_hybrid_result_scoring() {
        let weights = SearchWeights {
            keyword_weight: 0.4,
            semantic_weight: 0.6,
            recency_weight: 0.0,
        };
        let result = HybridResult::new("doc1".to_string(), 0.8, 0.9, 0.0, &weights);
        let expected_score = 0.8 * 0.4 + 0.9 * 0.6;
        assert!((result.score - expected_score).abs() < 0.001);
    }
}
