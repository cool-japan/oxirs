//! Result fusion and score combination algorithms for merging vector search results
//!
//! This module provides advanced algorithms for combining vector search results from
//! multiple sources, including federated endpoints, different similarity metrics,
//! and heterogeneous scoring schemes.

use crate::{sparql_integration::VectorServiceResult, Vector};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for result fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Maximum number of results to return after fusion
    pub max_results: usize,
    /// Minimum score threshold for inclusion
    pub min_score_threshold: f32,
    /// Score normalization strategy
    pub normalization_strategy: ScoreNormalizationStrategy,
    /// Fusion algorithm to use
    pub fusion_algorithm: FusionAlgorithm,
    /// Weights for different sources (source_id -> weight)
    pub source_weights: HashMap<String, f32>,
    /// Enable result diversification
    pub enable_diversification: bool,
    /// Diversification factor (0.0 = no diversification, 1.0 = maximum diversification)
    pub diversification_factor: f32,
    /// Enable result explanation
    pub enable_explanation: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            max_results: 100,
            min_score_threshold: 0.0,
            normalization_strategy: ScoreNormalizationStrategy::MinMax,
            fusion_algorithm: FusionAlgorithm::CombSum,
            source_weights: HashMap::new(),
            enable_diversification: false,
            diversification_factor: 0.2,
            enable_explanation: false,
        }
    }
}

/// Score normalization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreNormalizationStrategy {
    /// No normalization
    None,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization (standardization)
    ZScore,
    /// Rank-based normalization
    Rank,
    /// Sigmoid normalization
    Sigmoid,
    /// Softmax normalization
    Softmax,
}

/// Fusion algorithms for combining scores
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum FusionAlgorithm {
    /// Sum of normalized scores
    #[default]
    CombSum,
    /// Maximum score across sources
    CombMax,
    /// Minimum score across sources
    CombMin,
    /// Average of scores
    CombAvg,
    /// Median of scores
    CombMedian,
    /// Weighted sum with source weights
    WeightedSum,
    /// Reciprocal rank fusion
    RRF,
    /// Borda count fusion
    BordaCount,
    /// Condorcet fusion
    Condorcet,
    /// Machine learning-based fusion
    MLFusion,
}


/// A single result from a vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// Resource identifier
    pub resource: String,
    /// Similarity score
    pub score: f32,
    /// Normalized score (computed during fusion)
    pub normalized_score: Option<f32>,
    /// Source identifier
    pub source: String,
    /// Original rank in source results
    pub original_rank: usize,
    /// Final rank after fusion
    pub final_rank: Option<usize>,
    /// Associated vector (optional)
    pub vector: Option<Vector>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Explanation of score computation
    pub explanation: Option<String>,
}

/// Collection of results from a single source
#[derive(Debug, Clone)]
pub struct SourceResults {
    /// Source identifier
    pub source_id: String,
    /// Results from this source
    pub results: Vec<VectorSearchResult>,
    /// Source-specific metadata
    pub metadata: HashMap<String, String>,
    /// Response time from source
    pub response_time: Option<Duration>,
    /// Source weight (if different from config)
    pub weight: Option<f32>,
}

/// Fused results from multiple sources
#[derive(Debug, Clone)]
pub struct FusedResults {
    /// Final ranked results
    pub results: Vec<VectorSearchResult>,
    /// Fusion statistics
    pub fusion_stats: FusionStats,
    /// Configuration used for fusion
    pub config: FusionConfig,
    /// Total processing time
    pub processing_time: Duration,
}

/// Statistics about the fusion process
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// Number of input sources
    pub source_count: usize,
    /// Total number of input results
    pub total_input_results: usize,
    /// Number of unique resources
    pub unique_resources: usize,
    /// Number of results after fusion
    pub final_result_count: usize,
    /// Average score before normalization
    pub avg_score_before: f32,
    /// Average score after normalization
    pub avg_score_after: f32,
    /// Score distribution by source
    pub score_distribution: HashMap<String, ScoreDistribution>,
    /// Fusion algorithm used
    pub fusion_algorithm: FusionAlgorithm,
}

/// Score distribution statistics for a source
#[derive(Debug, Clone, Default)]
pub struct ScoreDistribution {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub count: usize,
}

/// Result fusion engine
pub struct ResultFusionEngine {
    config: FusionConfig,
}

impl ResultFusionEngine {
    /// Create a new fusion engine with default configuration
    pub fn new() -> Self {
        Self {
            config: FusionConfig::default(),
        }
    }

    /// Create fusion engine with custom configuration
    pub fn with_config(config: FusionConfig) -> Self {
        Self { config }
    }

    /// Fuse results from multiple sources
    pub fn fuse_results(&self, sources: Vec<SourceResults>) -> Result<FusedResults> {
        let start_time = std::time::Instant::now();

        if sources.is_empty() {
            return Ok(FusedResults {
                results: Vec::new(),
                fusion_stats: FusionStats::default(),
                config: self.config.clone(),
                processing_time: start_time.elapsed(),
            });
        }

        // Collect all results with source information
        let mut all_results = Vec::new();
        let mut fusion_stats = FusionStats {
            source_count: sources.len(),
            fusion_algorithm: self.config.fusion_algorithm.clone(),
            ..Default::default()
        };

        for source in &sources {
            for (rank, result) in source.results.iter().enumerate() {
                let mut enriched_result = result.clone();
                enriched_result.original_rank = rank;
                enriched_result.source = source.source_id.clone();
                all_results.push(enriched_result);
            }
            fusion_stats.total_input_results += source.results.len();
        }

        // Calculate score distributions
        self.calculate_score_distributions(&sources, &mut fusion_stats);

        // Normalize scores
        let normalized_results = self.normalize_scores(all_results)?;

        // Group results by resource
        let grouped_results = self.group_by_resource(normalized_results);
        fusion_stats.unique_resources = grouped_results.len();

        // Apply fusion algorithm
        let fused_results = self.apply_fusion_algorithm(grouped_results)?;

        // Apply diversification if enabled
        let diversified_results = if self.config.enable_diversification {
            self.apply_diversification(fused_results)?
        } else {
            fused_results
        };

        // Filter by threshold and limit
        let mut final_results = diversified_results
            .into_iter()
            .filter(|r| r.score >= self.config.min_score_threshold)
            .take(self.config.max_results)
            .collect::<Vec<_>>();

        // Assign final ranks
        for (rank, result) in final_results.iter_mut().enumerate() {
            result.final_rank = Some(rank + 1);
        }

        // Update statistics
        fusion_stats.final_result_count = final_results.len();
        if !final_results.is_empty() {
            fusion_stats.avg_score_after =
                final_results.iter().map(|r| r.score).sum::<f32>() / final_results.len() as f32;
        }

        Ok(FusedResults {
            results: final_results,
            fusion_stats,
            config: self.config.clone(),
            processing_time: start_time.elapsed(),
        })
    }

    /// Calculate score distributions for each source
    fn calculate_score_distributions(&self, sources: &[SourceResults], stats: &mut FusionStats) {
        for source in sources {
            if source.results.is_empty() {
                continue;
            }

            let scores: Vec<f32> = source.results.iter().map(|r| r.score).collect();
            let min = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mean = scores.iter().sum::<f32>() / scores.len() as f32;

            let variance =
                scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;
            let std_dev = variance.sqrt();

            stats.score_distribution.insert(
                source.source_id.clone(),
                ScoreDistribution {
                    min,
                    max,
                    mean,
                    std_dev,
                    count: scores.len(),
                },
            );
        }

        // Calculate overall statistics
        let all_scores: Vec<f32> = sources
            .iter()
            .flat_map(|s| s.results.iter().map(|r| r.score))
            .collect();

        if !all_scores.is_empty() {
            stats.avg_score_before = all_scores.iter().sum::<f32>() / all_scores.len() as f32;
        }
    }

    /// Normalize scores across all results
    fn normalize_scores(
        &self,
        mut results: Vec<VectorSearchResult>,
    ) -> Result<Vec<VectorSearchResult>> {
        match self.config.normalization_strategy {
            ScoreNormalizationStrategy::None => {
                for result in &mut results {
                    result.normalized_score = Some(result.score);
                }
            }
            ScoreNormalizationStrategy::MinMax => {
                self.apply_minmax_normalization(&mut results)?;
            }
            ScoreNormalizationStrategy::ZScore => {
                self.apply_zscore_normalization(&mut results)?;
            }
            ScoreNormalizationStrategy::Rank => {
                self.apply_rank_normalization(&mut results)?;
            }
            ScoreNormalizationStrategy::Sigmoid => {
                self.apply_sigmoid_normalization(&mut results)?;
            }
            ScoreNormalizationStrategy::Softmax => {
                self.apply_softmax_normalization(&mut results)?;
            }
        }

        Ok(results)
    }

    /// Apply min-max normalization
    fn apply_minmax_normalization(&self, results: &mut [VectorSearchResult]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        let min_score = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let range = max_score - min_score;
        if range == 0.0 {
            for result in results {
                result.normalized_score = Some(1.0);
            }
        } else {
            for result in results {
                result.normalized_score = Some((result.score - min_score) / range);
            }
        }

        Ok(())
    }

    /// Apply z-score normalization
    fn apply_zscore_normalization(&self, results: &mut [VectorSearchResult]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance =
            scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            for result in results {
                result.normalized_score = Some(0.0);
            }
        } else {
            for result in results {
                result.normalized_score = Some((result.score - mean) / std_dev);
            }
        }

        Ok(())
    }

    /// Apply rank-based normalization
    fn apply_rank_normalization(&self, results: &mut [VectorSearchResult]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        // Sort by score descending
        let mut indexed_results: Vec<(usize, f32)> = results
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.score))
            .collect();
        indexed_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign rank-based scores
        let total_count = results.len() as f32;
        for (rank, (original_index, _)) in indexed_results.iter().enumerate() {
            let normalized_score = (total_count - rank as f32) / total_count;
            results[*original_index].normalized_score = Some(normalized_score);
        }

        Ok(())
    }

    /// Apply sigmoid normalization
    fn apply_sigmoid_normalization(&self, results: &mut [VectorSearchResult]) -> Result<()> {
        for result in results {
            let sigmoid_score = 1.0 / (1.0 + (-result.score).exp());
            result.normalized_score = Some(sigmoid_score);
        }
        Ok(())
    }

    /// Apply softmax normalization
    fn apply_softmax_normalization(&self, results: &mut [VectorSearchResult]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        // Calculate softmax
        let max_score = results
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        let exp_scores: Vec<f32> = results
            .iter()
            .map(|r| (r.score - max_score).exp())
            .collect();

        let sum_exp: f32 = exp_scores.iter().sum();

        for (i, result) in results.iter_mut().enumerate() {
            result.normalized_score = Some(exp_scores[i] / sum_exp);
        }

        Ok(())
    }

    /// Group results by resource identifier
    fn group_by_resource(
        &self,
        results: Vec<VectorSearchResult>,
    ) -> HashMap<String, Vec<VectorSearchResult>> {
        let mut grouped = HashMap::new();

        for result in results {
            grouped
                .entry(result.resource.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        grouped
    }

    /// Apply the configured fusion algorithm
    fn apply_fusion_algorithm(
        &self,
        grouped_results: HashMap<String, Vec<VectorSearchResult>>,
    ) -> Result<Vec<VectorSearchResult>> {
        let mut fused_results = Vec::new();

        for (_resource, mut resource_results) in grouped_results {
            let fused_result = match &self.config.fusion_algorithm {
                FusionAlgorithm::CombSum => self.apply_combsum(&resource_results)?,
                FusionAlgorithm::CombMax => self.apply_combmax(&resource_results)?,
                FusionAlgorithm::CombMin => self.apply_combmin(&resource_results)?,
                FusionAlgorithm::CombAvg => self.apply_combavg(&resource_results)?,
                FusionAlgorithm::CombMedian => self.apply_combmedian(&mut resource_results)?,
                FusionAlgorithm::WeightedSum => self.apply_weighted_sum(&resource_results)?,
                FusionAlgorithm::RRF => self.apply_rrf(&resource_results)?,
                FusionAlgorithm::BordaCount => self.apply_borda_count(&resource_results)?,
                FusionAlgorithm::Condorcet => self.apply_condorcet(&resource_results)?,
                FusionAlgorithm::MLFusion => self.apply_ml_fusion(&resource_results)?,
            };

            fused_results.push(fused_result);
        }

        // Sort by fused score descending
        fused_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// CombSum: Sum of normalized scores
    fn apply_combsum(&self, results: &[VectorSearchResult]) -> Result<VectorSearchResult> {
        let sum_score = results
            .iter()
            .map(|r| r.normalized_score.unwrap_or(r.score))
            .sum::<f32>();

        Ok(self.create_fused_result(results, sum_score, "CombSum"))
    }

    /// CombMax: Maximum normalized score
    fn apply_combmax(&self, results: &[VectorSearchResult]) -> Result<VectorSearchResult> {
        let max_score = results
            .iter()
            .map(|r| r.normalized_score.unwrap_or(r.score))
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        Ok(self.create_fused_result(results, max_score, "CombMax"))
    }

    /// CombMin: Minimum normalized score
    fn apply_combmin(&self, results: &[VectorSearchResult]) -> Result<VectorSearchResult> {
        let min_score = results
            .iter()
            .map(|r| r.normalized_score.unwrap_or(r.score))
            .fold(f32::INFINITY, |a, b| a.min(b));

        Ok(self.create_fused_result(results, min_score, "CombMin"))
    }

    /// CombAvg: Average of normalized scores
    fn apply_combavg(&self, results: &[VectorSearchResult]) -> Result<VectorSearchResult> {
        let avg_score = results
            .iter()
            .map(|r| r.normalized_score.unwrap_or(r.score))
            .sum::<f32>()
            / results.len() as f32;

        Ok(self.create_fused_result(results, avg_score, "CombAvg"))
    }

    /// CombMedian: Median of normalized scores
    fn apply_combmedian(&self, results: &mut [VectorSearchResult]) -> Result<VectorSearchResult> {
        let mut scores: Vec<f32> = results
            .iter()
            .map(|r| r.normalized_score.unwrap_or(r.score))
            .collect();

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_score = if scores.len() % 2 == 0 {
            let mid = scores.len() / 2;
            (scores[mid - 1] + scores[mid]) / 2.0
        } else {
            scores[scores.len() / 2]
        };

        Ok(self.create_fused_result(results, median_score, "CombMedian"))
    }

    /// WeightedSum: Weighted sum with source weights
    fn apply_weighted_sum(&self, results: &[VectorSearchResult]) -> Result<VectorSearchResult> {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for result in results {
            let weight = self
                .config
                .source_weights
                .get(&result.source)
                .copied()
                .unwrap_or(1.0);
            let score = result.normalized_score.unwrap_or(result.score);
            weighted_sum += score * weight;
            total_weight += weight;
        }

        let final_score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        Ok(self.create_fused_result(results, final_score, "WeightedSum"))
    }

    /// Reciprocal Rank Fusion (RRF)
    fn apply_rrf(&self, results: &[VectorSearchResult]) -> Result<VectorSearchResult> {
        let k = 60.0; // Standard RRF parameter
        let rrf_score = results
            .iter()
            .map(|r| 1.0 / (k + r.original_rank as f32 + 1.0))
            .sum::<f32>();

        Ok(self.create_fused_result(results, rrf_score, "RRF"))
    }

    /// Borda Count fusion
    fn apply_borda_count(&self, results: &[VectorSearchResult]) -> Result<VectorSearchResult> {
        // For each source, assign points based on rank (higher rank = more points)
        let total_sources = results.len();
        let borda_score = results
            .iter()
            .map(|r| (total_sources - r.original_rank) as f32)
            .sum::<f32>();

        Ok(self.create_fused_result(results, borda_score, "BordaCount"))
    }

    /// Condorcet fusion (simplified pairwise comparison)
    fn apply_condorcet(&self, results: &[VectorSearchResult]) -> Result<VectorSearchResult> {
        // Simplified Condorcet: average of normalized scores with rank consideration
        let condorcet_score = results
            .iter()
            .map(|r| {
                let score = r.normalized_score.unwrap_or(r.score);
                let rank_penalty = 1.0 / (r.original_rank as f32 + 1.0);
                score * rank_penalty
            })
            .sum::<f32>()
            / results.len() as f32;

        Ok(self.create_fused_result(results, condorcet_score, "Condorcet"))
    }

    /// Machine learning-based fusion (placeholder)
    fn apply_ml_fusion(&self, results: &[VectorSearchResult]) -> Result<VectorSearchResult> {
        // Placeholder: In practice, this would use a trained ML model
        // For now, use a weighted combination of features
        let mut ml_score = 0.0;

        for result in results {
            let score = result.normalized_score.unwrap_or(result.score);
            let rank_feature = 1.0 / (result.original_rank as f32 + 1.0);
            let source_weight = self
                .config
                .source_weights
                .get(&result.source)
                .copied()
                .unwrap_or(1.0);

            // Simple linear combination (in practice, would use trained weights)
            ml_score += 0.5 * score + 0.3 * rank_feature + 0.2 * source_weight;
        }

        ml_score /= results.len() as f32;

        Ok(self.create_fused_result(results, ml_score, "MLFusion"))
    }

    /// Create a fused result from multiple source results
    fn create_fused_result(
        &self,
        results: &[VectorSearchResult],
        fused_score: f32,
        algorithm: &str,
    ) -> VectorSearchResult {
        let first_result = &results[0];
        let mut metadata = first_result.metadata.clone();

        // Add fusion information
        metadata.insert("fusion_algorithm".to_string(), algorithm.to_string());
        metadata.insert("source_count".to_string(), results.len().to_string());
        metadata.insert(
            "sources".to_string(),
            results
                .iter()
                .map(|r| r.source.clone())
                .collect::<Vec<_>>()
                .join(","),
        );

        let explanation = if self.config.enable_explanation {
            Some(format!(
                "{} fusion of {} results from sources: [{}] with final score: {:.4}",
                algorithm,
                results.len(),
                results
                    .iter()
                    .map(|r| format!("{}:{:.3}", r.source, r.score))
                    .collect::<Vec<_>>()
                    .join(", "),
                fused_score
            ))
        } else {
            None
        };

        VectorSearchResult {
            resource: first_result.resource.clone(),
            score: fused_score,
            normalized_score: Some(fused_score),
            source: "FUSED".to_string(),
            original_rank: 0,
            final_rank: None,
            vector: first_result.vector.clone(),
            metadata,
            explanation,
        }
    }

    /// Apply result diversification to reduce redundancy
    fn apply_diversification(
        &self,
        results: Vec<VectorSearchResult>,
    ) -> Result<Vec<VectorSearchResult>> {
        if results.len() <= 1 || self.config.diversification_factor == 0.0 {
            return Ok(results);
        }

        let mut diversified = Vec::new();
        let mut remaining = results;

        // Always take the top result
        if !remaining.is_empty() {
            diversified.push(remaining.remove(0));
        }

        // For each subsequent position, balance relevance and diversity
        while !remaining.is_empty() && diversified.len() < self.config.max_results {
            let mut best_index = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (i, candidate) in remaining.iter().enumerate() {
                // Calculate diversity penalty
                let diversity_penalty = self.calculate_diversity_penalty(candidate, &diversified);

                // Combine relevance and diversity
                let combined_score = (1.0 - self.config.diversification_factor) * candidate.score
                    + self.config.diversification_factor * diversity_penalty;

                if combined_score > best_score {
                    best_score = combined_score;
                    best_index = i;
                }
            }

            diversified.push(remaining.remove(best_index));
        }

        Ok(diversified)
    }

    /// Calculate diversity penalty for a candidate result
    fn calculate_diversity_penalty(
        &self,
        candidate: &VectorSearchResult,
        selected: &[VectorSearchResult],
    ) -> f32 {
        if selected.is_empty() {
            return 1.0;
        }

        // Simple diversity measure based on string similarity
        let mut min_similarity = f32::INFINITY;

        for selected_result in selected {
            let similarity =
                self.calculate_string_similarity(&candidate.resource, &selected_result.resource);
            min_similarity = min_similarity.min(similarity);
        }

        // Convert similarity to diversity (higher diversity = lower similarity)
        1.0 - min_similarity
    }

    /// Calculate string similarity between two resources
    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f32 {
        // Simple Jaccard similarity on character bigrams
        let bigrams1 = self.get_character_bigrams(s1);
        let bigrams2 = self.get_character_bigrams(s2);

        let intersection: usize = bigrams1
            .iter()
            .filter(|&bigram| bigrams2.contains(bigram))
            .count();

        let union_size = bigrams1.len() + bigrams2.len() - intersection;

        if union_size == 0 {
            1.0
        } else {
            intersection as f32 / union_size as f32
        }
    }

    /// Get character bigrams from a string
    fn get_character_bigrams(&self, s: &str) -> std::collections::HashSet<String> {
        let chars: Vec<char> = s.chars().collect();
        let mut bigrams = std::collections::HashSet::new();

        for i in 0..chars.len().saturating_sub(1) {
            let bigram = format!("{}{}", chars[i], chars[i + 1]);
            bigrams.insert(bigram);
        }

        bigrams
    }
}

impl Default for ResultFusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for working with fusion results
pub mod fusion_utils {
    use super::*;

    /// Convert vector service results to source results
    pub fn convert_service_results(
        source_id: String,
        service_result: VectorServiceResult,
    ) -> Result<SourceResults> {
        let results = match service_result {
            VectorServiceResult::SimilarityList(list) => list
                .into_iter()
                .enumerate()
                .map(|(rank, (resource, score))| VectorSearchResult {
                    resource,
                    score,
                    normalized_score: None,
                    source: source_id.clone(),
                    original_rank: rank,
                    final_rank: None,
                    vector: None,
                    metadata: HashMap::new(),
                    explanation: None,
                })
                .collect(),
            VectorServiceResult::DetailedSimilarityList(detailed_list) => detailed_list
                .into_iter()
                .enumerate()
                .map(|(rank, detailed)| VectorSearchResult {
                    resource: detailed.0,
                    score: detailed.1,
                    normalized_score: None,
                    source: source_id.clone(),
                    original_rank: rank,
                    final_rank: None,
                    vector: None,
                    metadata: detailed.2,
                    explanation: None,
                })
                .collect(),
            _ => {
                return Err(anyhow!(
                    "Cannot convert non-similarity result to source results"
                ));
            }
        };

        Ok(SourceResults {
            source_id,
            results,
            metadata: HashMap::new(),
            response_time: None,
            weight: None,
        })
    }

    /// Create source results from simple tuples
    pub fn create_source_results(source_id: String, results: Vec<(String, f32)>) -> SourceResults {
        let search_results = results
            .into_iter()
            .enumerate()
            .map(|(rank, (resource, score))| VectorSearchResult {
                resource,
                score,
                normalized_score: None,
                source: source_id.clone(),
                original_rank: rank,
                final_rank: None,
                vector: None,
                metadata: HashMap::new(),
                explanation: None,
            })
            .collect();

        SourceResults {
            source_id,
            results: search_results,
            metadata: HashMap::new(),
            response_time: None,
            weight: None,
        }
    }

    /// Calculate fusion quality metrics
    pub fn calculate_fusion_quality(
        fused_results: &FusedResults,
        ground_truth: Option<&[String]>,
    ) -> FusionQualityMetrics {
        let mut metrics = FusionQualityMetrics {
            result_count: fused_results.results.len(),
            ..Default::default()
        };
        if !fused_results.results.is_empty() {
            metrics.avg_score = fused_results.results.iter().map(|r| r.score).sum::<f32>()
                / fused_results.results.len() as f32;
            metrics.min_score = fused_results
                .results
                .iter()
                .map(|r| r.score)
                .fold(f32::INFINITY, |a, b| a.min(b));
            metrics.max_score = fused_results
                .results
                .iter()
                .map(|r| r.score)
                .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        }

        // Calculate diversity
        metrics.diversity = calculate_result_diversity(&fused_results.results);

        // Calculate relevance metrics if ground truth is provided
        if let Some(gt) = ground_truth {
            let relevant_count = fused_results
                .results
                .iter()
                .filter(|r| gt.contains(&r.resource))
                .count();

            metrics.precision = if fused_results.results.is_empty() {
                0.0
            } else {
                relevant_count as f32 / fused_results.results.len() as f32
            };

            metrics.recall = if gt.is_empty() {
                0.0
            } else {
                relevant_count as f32 / gt.len() as f32
            };

            metrics.f1_score = if metrics.precision + metrics.recall == 0.0 {
                0.0
            } else {
                2.0 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            };
        }

        metrics
    }

    /// Calculate diversity among results
    fn calculate_result_diversity(results: &[VectorSearchResult]) -> f32 {
        if results.len() <= 1 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..results.len() {
            for j in i + 1..results.len() {
                // Simple string similarity
                let sim = jaccard_similarity(&results[i].resource, &results[j].resource);
                total_similarity += sim;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            1.0
        } else {
            1.0 - (total_similarity / pair_count as f32)
        }
    }

    /// Calculate Jaccard similarity between two strings
    fn jaccard_similarity(s1: &str, s2: &str) -> f32 {
        let chars1: std::collections::HashSet<char> = s1.chars().collect();
        let chars2: std::collections::HashSet<char> = s2.chars().collect();

        let intersection = chars1.intersection(&chars2).count();
        let union = chars1.union(&chars2).count();

        if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

/// Quality metrics for fusion results
#[derive(Debug, Clone, Default)]
pub struct FusionQualityMetrics {
    pub result_count: usize,
    pub avg_score: f32,
    pub min_score: f32,
    pub max_score: f32,
    pub diversity: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combsum_fusion() {
        let fusion_engine = ResultFusionEngine::new();

        let source1 = SourceResults {
            source_id: "source1".to_string(),
            results: vec![
                VectorSearchResult {
                    resource: "doc1".to_string(),
                    score: 0.9,
                    normalized_score: None,
                    source: "source1".to_string(),
                    original_rank: 0,
                    final_rank: None,
                    vector: None,
                    metadata: HashMap::new(),
                    explanation: None,
                },
                VectorSearchResult {
                    resource: "doc2".to_string(),
                    score: 0.7,
                    normalized_score: None,
                    source: "source1".to_string(),
                    original_rank: 1,
                    final_rank: None,
                    vector: None,
                    metadata: HashMap::new(),
                    explanation: None,
                },
            ],
            metadata: HashMap::new(),
            response_time: None,
            weight: None,
        };

        let source2 = SourceResults {
            source_id: "source2".to_string(),
            results: vec![
                VectorSearchResult {
                    resource: "doc1".to_string(),
                    score: 0.8,
                    normalized_score: None,
                    source: "source2".to_string(),
                    original_rank: 0,
                    final_rank: None,
                    vector: None,
                    metadata: HashMap::new(),
                    explanation: None,
                },
                VectorSearchResult {
                    resource: "doc3".to_string(),
                    score: 0.6,
                    normalized_score: None,
                    source: "source2".to_string(),
                    original_rank: 1,
                    final_rank: None,
                    vector: None,
                    metadata: HashMap::new(),
                    explanation: None,
                },
            ],
            metadata: HashMap::new(),
            response_time: None,
            weight: None,
        };

        let result = fusion_engine.fuse_results(vec![source1, source2]).unwrap();

        assert_eq!(result.results.len(), 3); // doc1, doc2, doc3
        assert_eq!(result.fusion_stats.source_count, 2);
        assert_eq!(result.fusion_stats.unique_resources, 3);

        // doc1 should have highest score (fusion of 0.9 and 0.8)
        assert_eq!(result.results[0].resource, "doc1");
        assert!(result.results[0].score > result.results[1].score);
    }

    #[test]
    fn test_rrf_fusion() {
        let config = FusionConfig {
            fusion_algorithm: FusionAlgorithm::RRF,
            ..Default::default()
        };
        let fusion_engine = ResultFusionEngine::with_config(config);

        // Create test data where doc2 appears in both sources with different ranks
        let source1 = fusion_utils::create_source_results(
            "source1".to_string(),
            vec![("doc1".to_string(), 0.9), ("doc2".to_string(), 0.7)],
        );

        let source2 = fusion_utils::create_source_results(
            "source2".to_string(),
            vec![("doc2".to_string(), 0.8), ("doc3".to_string(), 0.6)],
        );

        let result = fusion_engine.fuse_results(vec![source1, source2]).unwrap();

        assert!(!result.results.is_empty());
        assert_eq!(result.fusion_stats.unique_resources, 3);
    }

    #[test]
    fn test_score_normalization() {
        let config = FusionConfig {
            normalization_strategy: ScoreNormalizationStrategy::MinMax,
            ..Default::default()
        };
        let fusion_engine = ResultFusionEngine::with_config(config);

        let source = fusion_utils::create_source_results(
            "test".to_string(),
            vec![
                ("doc1".to_string(), 0.2),
                ("doc2".to_string(), 0.8),
                ("doc3".to_string(), 0.5),
            ],
        );

        let result = fusion_engine.fuse_results(vec![source]).unwrap();

        // After min-max normalization, scores should be in [0, 1]
        for res in &result.results {
            assert!(res.score >= 0.0 && res.score <= 1.0);
        }
    }

    #[test]
    fn test_fusion_quality_metrics() {
        let fusion_results = FusedResults {
            results: vec![
                VectorSearchResult {
                    resource: "relevant1".to_string(),
                    score: 0.9,
                    normalized_score: Some(0.9),
                    source: "test".to_string(),
                    original_rank: 0,
                    final_rank: Some(1),
                    vector: None,
                    metadata: HashMap::new(),
                    explanation: None,
                },
                VectorSearchResult {
                    resource: "irrelevant1".to_string(),
                    score: 0.8,
                    normalized_score: Some(0.8),
                    source: "test".to_string(),
                    original_rank: 1,
                    final_rank: Some(2),
                    vector: None,
                    metadata: HashMap::new(),
                    explanation: None,
                },
            ],
            fusion_stats: FusionStats::default(),
            config: FusionConfig::default(),
            processing_time: Duration::from_millis(10),
        };

        let ground_truth = vec!["relevant1".to_string(), "relevant2".to_string()];
        let metrics = fusion_utils::calculate_fusion_quality(&fusion_results, Some(&ground_truth));

        assert_eq!(metrics.result_count, 2);
        assert_eq!(metrics.precision, 0.5); // 1 relevant out of 2 results
        assert_eq!(metrics.recall, 0.5); // 1 relevant out of 2 ground truth
        assert!(metrics.diversity > 0.0);
    }
}
