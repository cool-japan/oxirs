//! Configuration for re-ranking

use serde::{Deserialize, Serialize};

/// Re-ranking mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RerankingMode {
    /// Re-rank all candidates
    Full,
    /// Re-rank top-k candidates only
    TopK,
    /// Adaptive re-ranking based on score distribution
    Adaptive,
    /// No re-ranking (passthrough)
    Disabled,
}

/// Strategy for fusing retrieval and re-ranking scores
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Use only re-ranking scores
    RerankingOnly,
    /// Use only retrieval scores
    RetrievalOnly,
    /// Linear combination: α * retrieval + (1-α) * reranking
    Linear,
    /// Reciprocal rank fusion
    ReciprocalRank,
    /// Learned score fusion (trained weights)
    Learned,
    /// Harmonic mean
    Harmonic,
    /// Geometric mean
    Geometric,
}

/// Configuration for re-ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingConfig {
    /// Re-ranking mode
    pub mode: RerankingMode,

    /// Maximum number of candidates to re-rank
    pub max_candidates: usize,

    /// Number of final results to return
    pub top_k: usize,

    /// Fusion strategy for combining scores
    pub fusion_strategy: FusionStrategy,

    /// Weight for retrieval score in linear fusion (0.0 to 1.0)
    pub retrieval_weight: f32,

    /// Batch size for cross-encoder inference
    pub batch_size: usize,

    /// Timeout for re-ranking (milliseconds)
    pub timeout_ms: Option<u64>,

    /// Enable result caching
    pub enable_caching: bool,

    /// Cache size (number of entries)
    pub cache_size: usize,

    /// Enable diversity-aware re-ranking
    pub enable_diversity: bool,

    /// Diversity weight (0.0 to 1.0)
    pub diversity_weight: f32,

    /// Model name or path
    pub model_name: String,

    /// Model backend (local, api, etc.)
    pub model_backend: String,

    /// Enable parallel batch processing
    pub enable_parallel: bool,

    /// Number of worker threads
    pub num_workers: usize,
}

impl RerankingConfig {
    /// Create default configuration
    pub fn default_config() -> Self {
        Self {
            mode: RerankingMode::TopK,
            max_candidates: 100,
            top_k: 10,
            fusion_strategy: FusionStrategy::Linear,
            retrieval_weight: 0.3,
            batch_size: 32,
            timeout_ms: Some(5000),
            enable_caching: true,
            cache_size: 1000,
            enable_diversity: false,
            diversity_weight: 0.2,
            model_name: "cross-encoder/ms-marco-MiniLM-L-12-v2".to_string(),
            model_backend: "local".to_string(),
            enable_parallel: true,
            num_workers: 4,
        }
    }

    /// Create configuration optimized for accuracy
    pub fn accuracy_optimized() -> Self {
        Self {
            mode: RerankingMode::Full,
            max_candidates: 200,
            top_k: 10,
            fusion_strategy: FusionStrategy::RerankingOnly,
            retrieval_weight: 0.0,
            batch_size: 16,
            timeout_ms: Some(10000),
            enable_caching: true,
            cache_size: 2000,
            enable_diversity: true,
            diversity_weight: 0.3,
            model_name: "cross-encoder/ms-marco-TinyBERT-L-6-v2".to_string(),
            model_backend: "local".to_string(),
            enable_parallel: true,
            num_workers: 8,
        }
    }

    /// Create configuration optimized for speed
    pub fn speed_optimized() -> Self {
        Self {
            mode: RerankingMode::TopK,
            max_candidates: 50,
            top_k: 10,
            fusion_strategy: FusionStrategy::Linear,
            retrieval_weight: 0.5,
            batch_size: 64,
            timeout_ms: Some(2000),
            enable_caching: true,
            cache_size: 500,
            enable_diversity: false,
            diversity_weight: 0.0,
            model_name: "cross-encoder/ms-marco-MiniLM-L-2-v2".to_string(),
            model_backend: "local".to_string(),
            enable_parallel: true,
            num_workers: 2,
        }
    }

    /// Create configuration for API-based models
    pub fn api_based(api_backend: impl Into<String>) -> Self {
        Self {
            mode: RerankingMode::TopK,
            max_candidates: 100,
            top_k: 10,
            fusion_strategy: FusionStrategy::Linear,
            retrieval_weight: 0.3,
            batch_size: 16,
            timeout_ms: Some(30000), // Longer timeout for API
            enable_caching: true,
            cache_size: 5000, // Larger cache for API
            enable_diversity: false,
            diversity_weight: 0.2,
            model_name: "rerank-v2".to_string(),
            model_backend: api_backend.into(),
            enable_parallel: false, // API handles parallelism
            num_workers: 1,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_candidates == 0 {
            return Err("max_candidates must be greater than 0".to_string());
        }

        if self.top_k == 0 {
            return Err("top_k must be greater than 0".to_string());
        }

        if self.top_k > self.max_candidates {
            return Err("top_k cannot exceed max_candidates".to_string());
        }

        if self.retrieval_weight < 0.0 || self.retrieval_weight > 1.0 {
            return Err("retrieval_weight must be between 0.0 and 1.0".to_string());
        }

        if self.diversity_weight < 0.0 || self.diversity_weight > 1.0 {
            return Err("diversity_weight must be between 0.0 and 1.0".to_string());
        }

        if self.batch_size == 0 {
            return Err("batch_size must be greater than 0".to_string());
        }

        if self.cache_size == 0 && self.enable_caching {
            return Err("cache_size must be greater than 0 when caching is enabled".to_string());
        }

        if self.num_workers == 0 && self.enable_parallel {
            return Err(
                "num_workers must be greater than 0 when parallel processing is enabled"
                    .to_string(),
            );
        }

        if self.model_name.is_empty() {
            return Err("model_name cannot be empty".to_string());
        }

        Ok(())
    }

    /// Get fusion weight for reranking score
    pub fn reranking_weight(&self) -> f32 {
        1.0 - self.retrieval_weight
    }
}

impl Default for RerankingConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RerankingConfig::default_config();
        assert_eq!(config.mode, RerankingMode::TopK);
        assert_eq!(config.max_candidates, 100);
        assert_eq!(config.top_k, 10);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_accuracy_optimized() {
        let config = RerankingConfig::accuracy_optimized();
        assert_eq!(config.mode, RerankingMode::Full);
        assert_eq!(config.fusion_strategy, FusionStrategy::RerankingOnly);
        assert!(config.enable_diversity);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_speed_optimized() {
        let config = RerankingConfig::speed_optimized();
        assert_eq!(config.max_candidates, 50);
        assert!(config.batch_size > 32); // Larger batches for speed
        assert!(!config.enable_diversity); // No diversity for speed
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_api_based() {
        let config = RerankingConfig::api_based("cohere");
        assert_eq!(config.model_backend, "cohere");
        assert!(config.timeout_ms.unwrap() > 10000); // Longer timeout
        assert!(config.cache_size > 1000); // Larger cache
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation() {
        let mut config = RerankingConfig::default_config();
        assert!(config.validate().is_ok());

        config.max_candidates = 0;
        assert!(config.validate().is_err());

        config = RerankingConfig::default_config();
        config.top_k = 0;
        assert!(config.validate().is_err());

        config = RerankingConfig::default_config();
        config.top_k = 200;
        config.max_candidates = 100;
        assert!(config.validate().is_err());

        config = RerankingConfig::default_config();
        config.retrieval_weight = 1.5;
        assert!(config.validate().is_err());

        config = RerankingConfig::default_config();
        config.model_name = "".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_reranking_weight() {
        let mut config = RerankingConfig::default_config();
        config.retrieval_weight = 0.3;
        assert!((config.reranking_weight() - 0.7).abs() < 0.001);

        config.retrieval_weight = 0.0;
        assert_eq!(config.reranking_weight(), 1.0);

        config.retrieval_weight = 1.0;
        assert_eq!(config.reranking_weight(), 0.0);
    }

    #[test]
    fn test_fusion_strategies() {
        let strategies = vec![
            FusionStrategy::RerankingOnly,
            FusionStrategy::RetrievalOnly,
            FusionStrategy::Linear,
            FusionStrategy::ReciprocalRank,
            FusionStrategy::Learned,
            FusionStrategy::Harmonic,
            FusionStrategy::Geometric,
        ];

        for strategy in strategies {
            let mut config = RerankingConfig::default_config();
            config.fusion_strategy = strategy;
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_reranking_modes() {
        let modes = vec![
            RerankingMode::Full,
            RerankingMode::TopK,
            RerankingMode::Adaptive,
            RerankingMode::Disabled,
        ];

        for mode in modes {
            let mut config = RerankingConfig::default_config();
            config.mode = mode;
            assert!(config.validate().is_ok());
        }
    }
}
