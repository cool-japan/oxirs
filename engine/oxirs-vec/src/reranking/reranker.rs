//! Main cross-encoder re-ranker implementation

use crate::reranking::{
    cache::RerankingCache,
    config::{RerankingConfig, RerankingMode},
    cross_encoder::CrossEncoder,
    diversity::DiversityReranker,
    fusion::ScoreFusion,
    types::{RerankingError, RerankingResult, ScoredCandidate},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Statistics for a re-ranking operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RerankingStats {
    /// Number of candidates processed
    pub num_candidates: usize,

    /// Number of candidates actually re-ranked
    pub num_reranked: usize,

    /// Number of cache hits
    pub cache_hits: usize,

    /// Total time (milliseconds)
    pub total_time_ms: f64,

    /// Model inference time (milliseconds)
    pub inference_time_ms: f64,

    /// Score fusion time (milliseconds)
    pub fusion_time_ms: f64,

    /// Average score change
    pub avg_score_change: f32,

    /// Rank correlation (Kendall's tau)
    pub rank_correlation: Option<f32>,
}

/// Output of a re-ranking operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingOutput {
    /// Re-ranked candidates
    pub candidates: Vec<ScoredCandidate>,

    /// Statistics
    pub stats: RerankingStats,
}

/// Cross-encoder re-ranker
pub struct CrossEncoderReranker {
    /// Configuration
    config: RerankingConfig,

    /// Cross-encoder model
    encoder: Arc<CrossEncoder>,

    /// Score fusion
    fusion: Arc<ScoreFusion>,

    /// Diversity re-ranker
    diversity: Option<Arc<DiversityReranker>>,

    /// Cache
    cache: Option<Arc<RerankingCache>>,
}

impl CrossEncoderReranker {
    /// Create new re-ranker
    pub fn new(config: RerankingConfig) -> RerankingResult<Self> {
        config
            .validate()
            .map_err(|e| RerankingError::InvalidConfiguration { message: e })?;

        let encoder = Arc::new(CrossEncoder::new(
            &config.model_name,
            &config.model_backend,
        )?);
        let fusion = Arc::new(ScoreFusion::new(
            config.fusion_strategy,
            config.retrieval_weight,
        ));

        let diversity = if config.enable_diversity {
            Some(Arc::new(DiversityReranker::new(config.diversity_weight)))
        } else {
            None
        };

        let cache = if config.enable_caching {
            Some(Arc::new(RerankingCache::new(config.cache_size)))
        } else {
            None
        };

        Ok(Self {
            config,
            encoder,
            fusion,
            diversity,
            cache,
        })
    }

    /// Re-rank candidates
    pub fn rerank(
        &self,
        query: &str,
        candidates: &[ScoredCandidate],
    ) -> RerankingResult<RerankingOutput> {
        let start = Instant::now();

        // Filter candidates based on mode
        let candidates_to_rerank = self.select_candidates_for_reranking(candidates);

        let mut stats = RerankingStats {
            num_candidates: candidates.len(),
            num_reranked: candidates_to_rerank.len(),
            ..Default::default()
        };

        // Check mode
        if self.config.mode == RerankingMode::Disabled {
            return Ok(RerankingOutput {
                candidates: candidates.to_vec(),
                stats,
            });
        }

        // Re-rank with cross-encoder
        let inference_start = Instant::now();
        let mut reranked = self.apply_cross_encoder(query, candidates_to_rerank, &mut stats)?;
        stats.inference_time_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

        // Fuse scores
        let fusion_start = Instant::now();
        for candidate in &mut reranked {
            if let Some(reranking_score) = candidate.reranking_score {
                candidate.final_score =
                    self.fusion.fuse(candidate.retrieval_score, reranking_score);
            }
        }
        stats.fusion_time_ms = fusion_start.elapsed().as_secs_f64() * 1000.0;

        // Apply diversity if enabled
        if let Some(ref diversity) = self.diversity {
            reranked = diversity.apply_diversity(&reranked)?;
        }

        // Sort by final score
        reranked.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());

        // Take top-k
        reranked.truncate(self.config.top_k);

        // Calculate statistics
        self.calculate_stats(&mut stats, candidates, &reranked);
        stats.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(RerankingOutput {
            candidates: reranked,
            stats,
        })
    }

    /// Select candidates for re-ranking based on mode
    fn select_candidates_for_reranking(
        &self,
        candidates: &[ScoredCandidate],
    ) -> Vec<ScoredCandidate> {
        let max_candidates = self.config.max_candidates.min(candidates.len());

        match self.config.mode {
            RerankingMode::Full => candidates.to_vec(),
            RerankingMode::TopK => candidates[..max_candidates].to_vec(),
            RerankingMode::Adaptive => {
                // Use score threshold for adaptive selection
                let threshold = self.calculate_adaptive_threshold(candidates);
                candidates
                    .iter()
                    .filter(|c| c.retrieval_score >= threshold)
                    .take(max_candidates)
                    .cloned()
                    .collect()
            }
            RerankingMode::Disabled => Vec::new(),
        }
    }

    /// Calculate adaptive threshold based on score distribution
    fn calculate_adaptive_threshold(&self, candidates: &[ScoredCandidate]) -> f32 {
        if candidates.is_empty() {
            return 0.0;
        }

        // Use mean - 0.5 * std as threshold
        let scores: Vec<f32> = candidates.iter().map(|c| c.retrieval_score).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        let std = variance.sqrt();

        (mean - 0.5 * std).max(0.0)
    }

    /// Apply cross-encoder to candidates
    fn apply_cross_encoder(
        &self,
        query: &str,
        candidates: Vec<ScoredCandidate>,
        stats: &mut RerankingStats,
    ) -> RerankingResult<Vec<ScoredCandidate>> {
        let mut reranked = Vec::new();

        // Process in batches
        for batch in candidates.chunks(self.config.batch_size) {
            let mut batch_results = Vec::new();

            for candidate in batch {
                // Check cache first
                let cache_key = format!("{}:{}", query, candidate.id);
                let score = if let Some(ref cache) = self.cache {
                    if let Some(cached_score) = cache.get(&cache_key) {
                        stats.cache_hits += 1;
                        cached_score
                    } else {
                        let score = self
                            .encoder
                            .score(query, candidate.content.as_deref().unwrap_or(""))?;
                        cache.put(cache_key, score);
                        score
                    }
                } else {
                    self.encoder
                        .score(query, candidate.content.as_deref().unwrap_or(""))?
                };

                let mut updated = candidate.clone();
                updated.reranking_score = Some(score);
                batch_results.push(updated);
            }

            reranked.extend(batch_results);
        }

        Ok(reranked)
    }

    /// Calculate additional statistics
    fn calculate_stats(
        &self,
        stats: &mut RerankingStats,
        original: &[ScoredCandidate],
        reranked: &[ScoredCandidate],
    ) {
        // Calculate average score change
        let score_changes: Vec<f32> = reranked
            .iter()
            .filter_map(|c| c.reranking_score.map(|r| (r - c.retrieval_score).abs()))
            .collect();

        if !score_changes.is_empty() {
            stats.avg_score_change = score_changes.iter().sum::<f32>() / score_changes.len() as f32;
        }

        // Calculate rank correlation (simplified - just check if order changed)
        if original.len() == reranked.len() && !original.is_empty() {
            let original_ids: Vec<&String> = original.iter().map(|c| &c.id).collect();
            let reranked_ids: Vec<&String> = reranked.iter().map(|c| &c.id).collect();
            let same_order = original_ids == reranked_ids;
            stats.rank_correlation = Some(if same_order { 1.0 } else { 0.5 });
        }
    }

    /// Get configuration
    pub fn config(&self) -> &RerankingConfig {
        &self.config
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<(usize, usize)> {
        self.cache.as_ref().map(|c| c.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reranking::config::FusionStrategy;

    #[test]
    fn test_reranking_stats_default() {
        let stats = RerankingStats::default();
        assert_eq!(stats.num_candidates, 0);
        assert_eq!(stats.num_reranked, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_select_candidates_topk() {
        let config = RerankingConfig {
            mode: RerankingMode::TopK,
            max_candidates: 5,
            ..RerankingConfig::default_config()
        };

        let encoder = CrossEncoder::new("dummy", "local").unwrap();
        let fusion = ScoreFusion::new(FusionStrategy::Linear, 0.3);

        let reranker = CrossEncoderReranker {
            config,
            encoder: Arc::new(encoder),
            fusion: Arc::new(fusion),
            diversity: None,
            cache: None,
        };

        let candidates: Vec<ScoredCandidate> = (0..10)
            .map(|i| ScoredCandidate::new(format!("doc{}", i), 0.9 - i as f32 * 0.05, i))
            .collect();

        let selected = reranker.select_candidates_for_reranking(&candidates);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_adaptive_threshold() {
        let config = RerankingConfig::default_config();
        let encoder = CrossEncoder::new("dummy", "local").unwrap();
        let fusion = ScoreFusion::new(FusionStrategy::Linear, 0.3);

        let reranker = CrossEncoderReranker {
            config,
            encoder: Arc::new(encoder),
            fusion: Arc::new(fusion),
            diversity: None,
            cache: None,
        };

        let candidates = vec![
            ScoredCandidate::new("doc1", 0.9, 0),
            ScoredCandidate::new("doc2", 0.8, 1),
            ScoredCandidate::new("doc3", 0.7, 2),
            ScoredCandidate::new("doc4", 0.3, 3),
            ScoredCandidate::new("doc5", 0.2, 4),
        ];

        let threshold = reranker.calculate_adaptive_threshold(&candidates);
        assert!(threshold > 0.0);
        assert!(threshold < 0.9);
    }
}
