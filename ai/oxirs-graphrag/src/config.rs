//! GraphRAG configuration

use serde::{Deserialize, Serialize};

/// GraphRAG configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRAGConfig {
    /// Number of seed nodes from vector search
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Maximum number of seeds after fusion
    #[serde(default = "default_max_seeds")]
    pub max_seeds: usize,

    /// Graph expansion hops
    #[serde(default = "default_expansion_hops")]
    pub expansion_hops: usize,

    /// Maximum subgraph size (triples)
    #[serde(default = "default_max_subgraph_size")]
    pub max_subgraph_size: usize,

    /// Maximum triples to include in LLM context
    #[serde(default = "default_max_context_triples")]
    pub max_context_triples: usize,

    /// Enable community detection
    #[serde(default = "default_enable_communities")]
    pub enable_communities: bool,

    /// Community detection algorithm
    #[serde(default)]
    pub community_algorithm: CommunityAlgorithm,

    /// Fusion strategy
    #[serde(default)]
    pub fusion_strategy: FusionStrategy,

    /// Weight for vector similarity scores (0.0 - 1.0)
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,

    /// Weight for keyword/BM25 scores (0.0 - 1.0)
    #[serde(default = "default_keyword_weight")]
    pub keyword_weight: f32,

    /// Path patterns for graph expansion (SPARQL property paths)
    #[serde(default)]
    pub path_patterns: Vec<String>,

    /// Similarity threshold for vector search
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,

    /// Cache size for query results
    #[serde(default)]
    pub cache_size: Option<usize>,

    /// Enable query expansion
    #[serde(default)]
    pub enable_query_expansion: bool,

    /// Enable hierarchical summarization
    #[serde(default)]
    pub enable_hierarchical_summary: bool,

    /// Maximum community levels for hierarchical summarization
    #[serde(default = "default_max_community_levels")]
    pub max_community_levels: usize,

    /// LLM model to use for generation
    #[serde(default)]
    pub llm_model: Option<String>,

    /// Temperature for LLM generation
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum tokens for LLM response
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

impl Default for GraphRAGConfig {
    fn default() -> Self {
        Self {
            top_k: default_top_k(),
            max_seeds: default_max_seeds(),
            expansion_hops: default_expansion_hops(),
            max_subgraph_size: default_max_subgraph_size(),
            max_context_triples: default_max_context_triples(),
            enable_communities: default_enable_communities(),
            community_algorithm: CommunityAlgorithm::default(),
            fusion_strategy: FusionStrategy::default(),
            vector_weight: default_vector_weight(),
            keyword_weight: default_keyword_weight(),
            path_patterns: vec![],
            similarity_threshold: default_similarity_threshold(),
            cache_size: Some(1000),
            enable_query_expansion: false,
            enable_hierarchical_summary: false,
            max_community_levels: default_max_community_levels(),
            llm_model: None,
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
        }
    }
}

/// Community detection algorithm
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum CommunityAlgorithm {
    /// Louvain algorithm (fast, good quality)
    #[default]
    Louvain,
    /// Leiden algorithm (improved Louvain)
    Leiden,
    /// Label propagation (very fast, lower quality)
    LabelPropagation,
    /// Connected components (simplest)
    ConnectedComponents,
}

/// Fusion strategy for combining retrieval results
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion (default, robust)
    #[default]
    ReciprocalRankFusion,
    /// Linear combination of scores
    LinearCombination,
    /// Take highest score per entity
    HighestScore,
    /// Learning-to-rank (requires model)
    LearningToRank,
}

// Default value functions
fn default_top_k() -> usize {
    20
}
fn default_max_seeds() -> usize {
    10
}
fn default_expansion_hops() -> usize {
    2
}
fn default_max_subgraph_size() -> usize {
    500
}
fn default_max_context_triples() -> usize {
    100
}
fn default_enable_communities() -> bool {
    true
}
fn default_vector_weight() -> f32 {
    0.7
}
fn default_keyword_weight() -> f32 {
    0.3
}
fn default_similarity_threshold() -> f32 {
    0.7
}
fn default_max_community_levels() -> usize {
    3
}
fn default_temperature() -> f32 {
    0.7
}
fn default_max_tokens() -> usize {
    2048
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GraphRAGConfig::default();
        assert_eq!(config.top_k, 20);
        assert_eq!(config.expansion_hops, 2);
        assert!(config.enable_communities);
        assert_eq!(config.fusion_strategy, FusionStrategy::ReciprocalRankFusion);
    }

    #[test]
    fn test_config_serialization() {
        let config = GraphRAGConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: GraphRAGConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.top_k, config.top_k);
    }
}
