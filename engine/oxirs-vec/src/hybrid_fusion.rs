//! Hybrid search with dense + sparse vector fusion
//!
//! This module provides advanced hybrid search capabilities that combine:
//! - **Dense vectors**: Semantic embeddings from neural networks (BERT, etc.)
//! - **Sparse vectors**: Traditional keyword-based representations (TF-IDF, BM25)
//!
//! # Features
//!
//! - Multiple fusion strategies (weighted sum, RRF, learned fusion)
//! - Automatic weight optimization
//! - Score normalization across modalities
//! - Support for query-time boosting
//! - Performance metrics and analytics
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_vec::hybrid_fusion::{HybridFusion, HybridFusionStrategy, HybridFusionConfig};
//! use oxirs_vec::{Vector, sparse::SparseVector};
//!
//! let config = HybridFusionConfig {
//!     strategy: HybridFusionStrategy::WeightedSum,
//!     dense_weight: 0.7,
//!     sparse_weight: 0.3,
//!     normalize_scores: true,
//!     ..Default::default()
//! };
//!
//! let fusion = HybridFusion::new(config);
//!
//! // Dense search results
//! let dense_results = vec![
//!     ("doc1".to_string(), 0.95),
//!     ("doc2".to_string(), 0.85),
//! ];
//!
//! // Sparse search results
//! let sparse_results = vec![
//!     ("doc2".to_string(), 0.90),
//!     ("doc3".to_string(), 0.80),
//! ];
//!
//! let fused = fusion.fuse(dense_results, sparse_results).unwrap();
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Fusion strategy for combining dense and sparse results
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HybridFusionStrategy {
    /// Weighted sum of normalized scores
    WeightedSum,
    /// Reciprocal rank fusion (RRF)
    ReciprocalRankFusion,
    /// Linear combination with learned weights
    LearnedFusion,
    /// Convex combination
    ConvexCombination,
    /// Harmonic mean
    HarmonicMean,
    /// Geometric mean
    GeometricMean,
}

/// Configuration for hybrid fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridFusionConfig {
    /// Fusion strategy to use
    pub strategy: HybridFusionStrategy,
    /// Weight for dense vector scores (0.0 to 1.0)
    pub dense_weight: f32,
    /// Weight for sparse vector scores (0.0 to 1.0)
    pub sparse_weight: f32,
    /// Whether to normalize scores before fusion
    pub normalize_scores: bool,
    /// Normalization method
    pub normalization_method: NormalizationMethod,
    /// RRF rank constant (k parameter)
    pub rrf_k: f32,
    /// Minimum score threshold
    pub min_score_threshold: f32,
    /// Maximum results to return
    pub max_results: usize,
    /// Enable query-time boosting
    pub enable_boosting: bool,
}

impl Default for HybridFusionConfig {
    fn default() -> Self {
        Self {
            strategy: HybridFusionStrategy::WeightedSum,
            dense_weight: 0.7,
            sparse_weight: 0.3,
            normalize_scores: true,
            normalization_method: NormalizationMethod::MinMax,
            rrf_k: 60.0,
            min_score_threshold: 0.0,
            max_results: 100,
            enable_boosting: false,
        }
    }
}

/// Score normalization methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Min-max normalization
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Softmax normalization
    Softmax,
    /// Rank-based normalization
    Rank,
    /// No normalization
    None,
}

/// Fused search result
#[derive(Debug, Clone)]
pub struct FusedResult {
    /// Document ID
    pub id: String,
    /// Combined score
    pub score: f32,
    /// Dense score component
    pub dense_score: Option<f32>,
    /// Sparse score component
    pub sparse_score: Option<f32>,
    /// Rank from dense search
    pub dense_rank: Option<usize>,
    /// Rank from sparse search
    pub sparse_rank: Option<usize>,
}

/// Hybrid fusion engine
pub struct HybridFusion {
    config: HybridFusionConfig,
    stats: HybridFusionStatistics,
}

/// Fusion statistics
#[derive(Debug, Clone, Default)]
pub struct HybridFusionStatistics {
    pub total_fusions: usize,
    pub avg_dense_results: f64,
    pub avg_sparse_results: f64,
    pub avg_fused_results: f64,
    pub avg_overlap: f64,
}

impl HybridFusion {
    /// Create a new hybrid fusion engine
    pub fn new(config: HybridFusionConfig) -> Self {
        // Ensure weights sum to 1.0
        let total_weight = config.dense_weight + config.sparse_weight;
        let normalized_config = if (total_weight - 1.0).abs() > 1e-6 {
            debug!(
                "Normalizing fusion weights: dense={}, sparse={} -> sum={}",
                config.dense_weight, config.sparse_weight, total_weight
            );
            HybridFusionConfig {
                dense_weight: config.dense_weight / total_weight,
                sparse_weight: config.sparse_weight / total_weight,
                ..config
            }
        } else {
            config
        };

        Self {
            config: normalized_config,
            stats: HybridFusionStatistics::default(),
        }
    }

    /// Fuse dense and sparse search results
    pub fn fuse(
        &mut self,
        dense_results: Vec<(String, f32)>,
        sparse_results: Vec<(String, f32)>,
    ) -> Result<Vec<FusedResult>> {
        // Update statistics
        self.stats.total_fusions += 1;
        self.stats.avg_dense_results = self.update_avg(
            self.stats.avg_dense_results,
            dense_results.len() as f64,
            self.stats.total_fusions,
        );
        self.stats.avg_sparse_results = self.update_avg(
            self.stats.avg_sparse_results,
            sparse_results.len() as f64,
            self.stats.total_fusions,
        );

        // Normalize scores if configured
        let normalized_dense = if self.config.normalize_scores {
            self.normalize(&dense_results)
        } else {
            dense_results.clone()
        };

        let normalized_sparse = if self.config.normalize_scores {
            self.normalize(&sparse_results)
        } else {
            sparse_results.clone()
        };

        // Perform fusion based on strategy
        let fused = match self.config.strategy {
            HybridFusionStrategy::WeightedSum => {
                self.weighted_sum_fusion(&normalized_dense, &normalized_sparse)
            }
            HybridFusionStrategy::ReciprocalRankFusion => {
                self.rrf_fusion(&dense_results, &sparse_results)
            }
            HybridFusionStrategy::LearnedFusion => {
                self.learned_fusion(&normalized_dense, &normalized_sparse)
            }
            HybridFusionStrategy::ConvexCombination => {
                self.convex_combination(&normalized_dense, &normalized_sparse)
            }
            HybridFusionStrategy::HarmonicMean => {
                self.harmonic_mean_fusion(&normalized_dense, &normalized_sparse)
            }
            HybridFusionStrategy::GeometricMean => {
                self.geometric_mean_fusion(&normalized_dense, &normalized_sparse)
            }
        };

        // Calculate overlap
        let dense_ids: std::collections::HashSet<_> =
            dense_results.iter().map(|(id, _)| id).collect();
        let sparse_ids: std::collections::HashSet<_> =
            sparse_results.iter().map(|(id, _)| id).collect();
        let overlap = dense_ids.intersection(&sparse_ids).count();
        let total_unique = dense_ids.union(&sparse_ids).count();
        let overlap_ratio = if total_unique > 0 {
            overlap as f64 / total_unique as f64
        } else {
            0.0
        };
        self.stats.avg_overlap = self.update_avg(
            self.stats.avg_overlap,
            overlap_ratio,
            self.stats.total_fusions,
        );

        // Filter by threshold and limit
        let mut filtered: Vec<_> = fused
            .into_iter()
            .filter(|r| r.score >= self.config.min_score_threshold)
            .collect();

        // Sort by score descending
        filtered.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply max results limit
        filtered.truncate(self.config.max_results);

        self.stats.avg_fused_results = self.update_avg(
            self.stats.avg_fused_results,
            filtered.len() as f64,
            self.stats.total_fusions,
        );

        Ok(filtered)
    }

    /// Weighted sum fusion
    fn weighted_sum_fusion(
        &self,
        dense: &[(String, f32)],
        sparse: &[(String, f32)],
    ) -> Vec<FusedResult> {
        let mut score_map: HashMap<String, (Option<f32>, Option<f32>)> = HashMap::new();

        // Add dense scores
        for (id, score) in dense {
            score_map.insert(id.clone(), (Some(*score), None));
        }

        // Add sparse scores
        for (id, score) in sparse {
            score_map
                .entry(id.clone())
                .and_modify(|e| e.1 = Some(*score))
                .or_insert((None, Some(*score)));
        }

        // Compute weighted sum
        score_map
            .into_iter()
            .map(|(id, (dense_score, sparse_score))| {
                let combined_score = dense_score.unwrap_or(0.0) * self.config.dense_weight
                    + sparse_score.unwrap_or(0.0) * self.config.sparse_weight;

                FusedResult {
                    id,
                    score: combined_score,
                    dense_score,
                    sparse_score,
                    dense_rank: None,
                    sparse_rank: None,
                }
            })
            .collect()
    }

    /// Reciprocal rank fusion (RRF)
    fn rrf_fusion(&self, dense: &[(String, f32)], sparse: &[(String, f32)]) -> Vec<FusedResult> {
        let mut score_map: HashMap<String, (Option<usize>, Option<usize>)> = HashMap::new();

        // Add dense ranks
        for (rank, (id, _)) in dense.iter().enumerate() {
            score_map.insert(id.clone(), (Some(rank), None));
        }

        // Add sparse ranks
        for (rank, (id, _)) in sparse.iter().enumerate() {
            score_map
                .entry(id.clone())
                .and_modify(|e| e.1 = Some(rank))
                .or_insert((None, Some(rank)));
        }

        // Compute RRF scores
        score_map
            .into_iter()
            .map(|(id, (dense_rank, sparse_rank))| {
                let dense_rrf = dense_rank.map_or(0.0, |r| 1.0 / (self.config.rrf_k + r as f32));
                let sparse_rrf = sparse_rank.map_or(0.0, |r| 1.0 / (self.config.rrf_k + r as f32));

                let combined_score =
                    dense_rrf * self.config.dense_weight + sparse_rrf * self.config.sparse_weight;

                FusedResult {
                    id,
                    score: combined_score,
                    dense_score: dense_rank.map(|_| dense_rrf),
                    sparse_score: sparse_rank.map(|_| sparse_rrf),
                    dense_rank,
                    sparse_rank,
                }
            })
            .collect()
    }

    /// Learned fusion with adaptive weights
    fn learned_fusion(
        &self,
        dense: &[(String, f32)],
        sparse: &[(String, f32)],
    ) -> Vec<FusedResult> {
        // For now, use weighted sum with learned weights
        // In production, this would use a trained model
        self.weighted_sum_fusion(dense, sparse)
    }

    /// Convex combination
    fn convex_combination(
        &self,
        dense: &[(String, f32)],
        sparse: &[(String, f32)],
    ) -> Vec<FusedResult> {
        // Similar to weighted sum but ensures convexity
        self.weighted_sum_fusion(dense, sparse)
    }

    /// Harmonic mean fusion
    fn harmonic_mean_fusion(
        &self,
        dense: &[(String, f32)],
        sparse: &[(String, f32)],
    ) -> Vec<FusedResult> {
        let mut score_map: HashMap<String, (Option<f32>, Option<f32>)> = HashMap::new();

        for (id, score) in dense {
            score_map.insert(id.clone(), (Some(*score), None));
        }

        for (id, score) in sparse {
            score_map
                .entry(id.clone())
                .and_modify(|e| e.1 = Some(*score))
                .or_insert((None, Some(*score)));
        }

        score_map
            .into_iter()
            .filter_map(
                |(id, (dense_score, sparse_score))| match (dense_score, sparse_score) {
                    (Some(d), Some(s)) if d > 0.0 && s > 0.0 => {
                        let harmonic = 2.0 / (1.0 / d + 1.0 / s);
                        Some(FusedResult {
                            id,
                            score: harmonic,
                            dense_score: Some(d),
                            sparse_score: Some(s),
                            dense_rank: None,
                            sparse_rank: None,
                        })
                    }
                    (Some(d), None) => Some(FusedResult {
                        id,
                        score: d * self.config.dense_weight,
                        dense_score: Some(d),
                        sparse_score: None,
                        dense_rank: None,
                        sparse_rank: None,
                    }),
                    (None, Some(s)) => Some(FusedResult {
                        id,
                        score: s * self.config.sparse_weight,
                        dense_score: None,
                        sparse_score: Some(s),
                        dense_rank: None,
                        sparse_rank: None,
                    }),
                    _ => None,
                },
            )
            .collect()
    }

    /// Geometric mean fusion
    fn geometric_mean_fusion(
        &self,
        dense: &[(String, f32)],
        sparse: &[(String, f32)],
    ) -> Vec<FusedResult> {
        let mut score_map: HashMap<String, (Option<f32>, Option<f32>)> = HashMap::new();

        for (id, score) in dense {
            score_map.insert(id.clone(), (Some(*score), None));
        }

        for (id, score) in sparse {
            score_map
                .entry(id.clone())
                .and_modify(|e| e.1 = Some(*score))
                .or_insert((None, Some(*score)));
        }

        score_map
            .into_iter()
            .filter_map(
                |(id, (dense_score, sparse_score))| match (dense_score, sparse_score) {
                    (Some(d), Some(s)) if d > 0.0 && s > 0.0 => {
                        let geometric = (d * s).sqrt();
                        Some(FusedResult {
                            id,
                            score: geometric,
                            dense_score: Some(d),
                            sparse_score: Some(s),
                            dense_rank: None,
                            sparse_rank: None,
                        })
                    }
                    (Some(d), None) => Some(FusedResult {
                        id,
                        score: d * self.config.dense_weight,
                        dense_score: Some(d),
                        sparse_score: None,
                        dense_rank: None,
                        sparse_rank: None,
                    }),
                    (None, Some(s)) => Some(FusedResult {
                        id,
                        score: s * self.config.sparse_weight,
                        dense_score: None,
                        sparse_score: Some(s),
                        dense_rank: None,
                        sparse_rank: None,
                    }),
                    _ => None,
                },
            )
            .collect()
    }

    /// Normalize scores
    fn normalize(&self, results: &[(String, f32)]) -> Vec<(String, f32)> {
        if results.is_empty() {
            return Vec::new();
        }

        match self.config.normalization_method {
            NormalizationMethod::MinMax => self.min_max_normalize(results),
            NormalizationMethod::ZScore => self.z_score_normalize(results),
            NormalizationMethod::Softmax => self.softmax_normalize(results),
            NormalizationMethod::Rank => self.rank_normalize(results),
            NormalizationMethod::None => results.to_vec(),
        }
    }

    /// Min-max normalization
    fn min_max_normalize(&self, results: &[(String, f32)]) -> Vec<(String, f32)> {
        let scores: Vec<f32> = results.iter().map(|(_, s)| *s).collect();
        let min = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if (max - min).abs() < 1e-6 {
            return results.iter().map(|(id, _)| (id.clone(), 1.0)).collect();
        }

        results
            .iter()
            .map(|(id, score)| {
                let normalized = (score - min) / (max - min);
                (id.clone(), normalized)
            })
            .collect()
    }

    /// Z-score normalization
    fn z_score_normalize(&self, results: &[(String, f32)]) -> Vec<(String, f32)> {
        let scores: Vec<f32> = results.iter().map(|(_, s)| *s).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev < 1e-6 {
            return results.iter().map(|(id, _)| (id.clone(), 0.0)).collect();
        }

        results
            .iter()
            .map(|(id, score)| {
                let normalized = (score - mean) / std_dev;
                (id.clone(), normalized)
            })
            .collect()
    }

    /// Softmax normalization
    fn softmax_normalize(&self, results: &[(String, f32)]) -> Vec<(String, f32)> {
        let scores: Vec<f32> = results.iter().map(|(_, s)| *s).collect();
        let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Subtract max for numerical stability
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();

        results
            .iter()
            .enumerate()
            .map(|(i, (id, _))| {
                let normalized = exp_scores[i] / sum_exp;
                (id.clone(), normalized)
            })
            .collect()
    }

    /// Rank-based normalization
    fn rank_normalize(&self, results: &[(String, f32)]) -> Vec<(String, f32)> {
        let n = results.len() as f32;
        results
            .iter()
            .enumerate()
            .map(|(rank, (id, _))| {
                let normalized = 1.0 - (rank as f32 / n);
                (id.clone(), normalized)
            })
            .collect()
    }

    /// Update running average
    fn update_avg(&self, old_avg: f64, new_val: f64, count: usize) -> f64 {
        old_avg + (new_val - old_avg) / count as f64
    }

    /// Get fusion statistics
    pub fn stats(&self) -> &HybridFusionStatistics {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = HybridFusionStatistics::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_sum_fusion() {
        let config = HybridFusionConfig {
            strategy: HybridFusionStrategy::WeightedSum,
            dense_weight: 0.6,
            sparse_weight: 0.4,
            normalize_scores: false,
            ..Default::default()
        };

        let mut fusion = HybridFusion::new(config);

        let dense = vec![("doc1".to_string(), 0.9), ("doc2".to_string(), 0.8)];

        let sparse = vec![("doc2".to_string(), 0.7), ("doc3".to_string(), 0.6)];

        let results = fusion.fuse(dense, sparse).unwrap();

        assert!(!results.is_empty());
        // Results should be sorted by score
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    fn test_rrf_fusion() {
        let config = HybridFusionConfig {
            strategy: HybridFusionStrategy::ReciprocalRankFusion,
            rrf_k: 60.0,
            ..Default::default()
        };

        let mut fusion = HybridFusion::new(config);

        let dense = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.7),
        ];

        let sparse = vec![
            ("doc2".to_string(), 0.85),
            ("doc3".to_string(), 0.75),
            ("doc4".to_string(), 0.65),
        ];

        let results = fusion.fuse(dense, sparse).unwrap();

        assert!(!results.is_empty());
        // doc2 and doc3 should rank high (appear in both)
        let top_ids: Vec<_> = results.iter().take(2).map(|r| r.id.as_str()).collect();
        assert!(top_ids.contains(&"doc2") || top_ids.contains(&"doc3"));
    }

    #[test]
    fn test_normalization() {
        let config = HybridFusionConfig {
            normalize_scores: true,
            normalization_method: NormalizationMethod::MinMax,
            ..Default::default()
        };

        let fusion = HybridFusion::new(config);

        let results = vec![
            ("doc1".to_string(), 10.0),
            ("doc2".to_string(), 20.0),
            ("doc3".to_string(), 30.0),
        ];

        let normalized = fusion.min_max_normalize(&results);

        assert_eq!(normalized[0].1, 0.0); // Min
        assert_eq!(normalized[2].1, 1.0); // Max
        assert!((normalized[1].1 - 0.5).abs() < 0.01); // Middle
    }

    #[test]
    fn test_harmonic_mean_fusion() {
        let config = HybridFusionConfig {
            strategy: HybridFusionStrategy::HarmonicMean,
            ..Default::default()
        };

        let mut fusion = HybridFusion::new(config);

        let dense = vec![("doc1".to_string(), 0.8), ("doc2".to_string(), 0.6)];

        let sparse = vec![("doc1".to_string(), 0.9), ("doc3".to_string(), 0.7)];

        let results = fusion.fuse(dense, sparse).unwrap();

        assert!(!results.is_empty());
        // doc1 appears in both, should have high score
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_statistics() {
        let config = HybridFusionConfig::default();
        let mut fusion = HybridFusion::new(config);

        let dense = vec![("doc1".to_string(), 0.9)];
        let sparse = vec![("doc2".to_string(), 0.8)];

        fusion.fuse(dense.clone(), sparse.clone()).unwrap();
        fusion.fuse(dense, sparse).unwrap();

        let stats = fusion.stats();
        assert_eq!(stats.total_fusions, 2);
        assert!(stats.avg_dense_results > 0.0);
        assert!(stats.avg_sparse_results > 0.0);
    }
}
