//! Configuration for hybrid search

use super::types::SearchWeights;
use serde::{Deserialize, Serialize};

/// Hybrid search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Search mode
    pub mode: SearchMode,
    /// Keyword search algorithm
    pub keyword_algorithm: KeywordAlgorithm,
    /// Fusion strategy for combining results
    pub fusion_strategy: RankFusionStrategy,
    /// Default search weights
    pub default_weights: SearchWeights,
    /// Enable query expansion
    pub enable_query_expansion: bool,
    /// Maximum expanded terms
    pub max_expanded_terms: usize,
    /// Minimum keyword score threshold
    pub min_keyword_score: f32,
    /// Minimum semantic score threshold
    pub min_semantic_score: f32,
    /// Enable re-ranking
    pub enable_reranking: bool,
    /// Number of candidates for re-ranking
    pub reranking_candidates: usize,
}

/// Search mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchMode {
    /// Keyword search only
    KeywordOnly,
    /// Semantic search only
    SemanticOnly,
    /// Hybrid search (both keyword and semantic)
    Hybrid,
    /// Adaptive (automatically choose based on query)
    Adaptive,
}

/// Keyword search algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeywordAlgorithm {
    /// BM25 (Okapi BM25)
    Bm25,
    /// TF-IDF (Term Frequency - Inverse Document Frequency)
    Tfidf,
    /// Combined (use both and take max)
    Combined,
}

/// Strategy for fusing keyword and semantic results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankFusionStrategy {
    /// Weighted sum of scores
    WeightedSum,
    /// Reciprocal Rank Fusion (RRF)
    ReciprocalRankFusion,
    /// Cascade (filter with keyword, re-rank with semantic)
    Cascade,
    /// Interleave results from both
    Interleave,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            mode: SearchMode::Hybrid,
            keyword_algorithm: KeywordAlgorithm::Bm25,
            fusion_strategy: RankFusionStrategy::ReciprocalRankFusion,
            default_weights: SearchWeights::default(),
            enable_query_expansion: true,
            max_expanded_terms: 5,
            min_keyword_score: 0.1,
            min_semantic_score: 0.3,
            enable_reranking: true,
            reranking_candidates: 100,
        }
    }
}

impl HybridSearchConfig {
    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        self.default_weights.validate()?;

        if self.max_expanded_terms == 0 {
            anyhow::bail!("max_expanded_terms must be positive");
        }

        if self.min_keyword_score < 0.0 || self.min_keyword_score > 1.0 {
            anyhow::bail!("min_keyword_score must be in [0.0, 1.0]");
        }

        if self.min_semantic_score < 0.0 || self.min_semantic_score > 1.0 {
            anyhow::bail!("min_semantic_score must be in [0.0, 1.0]");
        }

        if self.reranking_candidates == 0 {
            anyhow::bail!("reranking_candidates must be positive");
        }

        Ok(())
    }

    /// Create a keyword-only configuration
    pub fn keyword_only() -> Self {
        Self {
            mode: SearchMode::KeywordOnly,
            default_weights: SearchWeights {
                keyword_weight: 1.0,
                semantic_weight: 0.0,
                recency_weight: 0.0,
            },
            ..Default::default()
        }
    }

    /// Create a semantic-only configuration
    pub fn semantic_only() -> Self {
        Self {
            mode: SearchMode::SemanticOnly,
            default_weights: SearchWeights {
                keyword_weight: 0.0,
                semantic_weight: 1.0,
                recency_weight: 0.0,
            },
            ..Default::default()
        }
    }

    /// Create a balanced hybrid configuration
    pub fn balanced() -> Self {
        Self {
            mode: SearchMode::Hybrid,
            default_weights: SearchWeights {
                keyword_weight: 0.5,
                semantic_weight: 0.5,
                recency_weight: 0.0,
            },
            fusion_strategy: RankFusionStrategy::ReciprocalRankFusion,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = HybridSearchConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_keyword_only_config() {
        let config = HybridSearchConfig::keyword_only();
        assert_eq!(config.mode, SearchMode::KeywordOnly);
        assert!((config.default_weights.keyword_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_semantic_only_config() {
        let config = HybridSearchConfig::semantic_only();
        assert_eq!(config.mode, SearchMode::SemanticOnly);
        assert!((config.default_weights.semantic_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_balanced_config() {
        let config = HybridSearchConfig::balanced();
        assert_eq!(config.mode, SearchMode::Hybrid);
        assert!((config.default_weights.keyword_weight - 0.5).abs() < 0.001);
        assert!((config.default_weights.semantic_weight - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_invalid_thresholds() {
        let config = HybridSearchConfig {
            min_keyword_score: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
