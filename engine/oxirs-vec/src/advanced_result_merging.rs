//! Advanced Result Merging and Score Combination System
//!
//! This module provides sophisticated result merging capabilities for combining
//! vector search results from multiple sources, algorithms, and modalities.

use crate::{similarity::SimilarityMetric, Vector, VectorError};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for result merging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMergingConfig {
    /// Score combination strategy
    pub combination_strategy: ScoreCombinationStrategy,
    /// Normalization method for scores
    pub normalization_method: ScoreNormalizationMethod,
    /// Fusion algorithm for rank-based combination
    pub fusion_algorithm: RankFusionAlgorithm,
    /// Weights for different result sources
    pub source_weights: HashMap<String, f32>,
    /// Confidence interval calculation
    pub confidence_intervals: bool,
    /// Enable explanation generation
    pub enable_explanations: bool,
    /// Result diversity enhancement
    pub diversity_config: Option<DiversityConfig>,
}

impl Default for ResultMergingConfig {
    fn default() -> Self {
        let mut source_weights = HashMap::new();
        source_weights.insert("primary".to_string(), 1.0);
        
        Self {
            combination_strategy: ScoreCombinationStrategy::WeightedSum,
            normalization_method: ScoreNormalizationMethod::MinMax,
            fusion_algorithm: RankFusionAlgorithm::CombSUM,
            source_weights,
            confidence_intervals: true,
            enable_explanations: false,
            diversity_config: None,
        }
    }
}

/// Score combination strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreCombinationStrategy {
    /// Simple average of scores
    Average,
    /// Weighted sum of scores
    WeightedSum,
    /// Maximum score across sources
    Maximum,
    /// Minimum score across sources
    Minimum,
    /// Geometric mean
    GeometricMean,
    /// Harmonic mean
    HarmonicMean,
    /// Product of scores
    Product,
    /// Borda count method
    BordaCount,
    /// Custom combination function
    Custom(String),
}

/// Score normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreNormalizationMethod {
    /// No normalization
    None,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Rank-based normalization
    RankBased,
    /// Softmax normalization
    Softmax,
    /// Sigmoid normalization
    Sigmoid,
}

/// Rank fusion algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RankFusionAlgorithm {
    /// CombSUM - sum of scores
    CombSUM,
    /// CombMNZ - multiply sum by number of non-zero scores
    CombMNZ,
    /// Reciprocal Rank Fusion
    ReciprocalRankFusion,
    /// Borda fusion
    BordaFusion,
    /// Condorcet fusion
    CondorcetFusion,
}

/// Diversity configuration for result enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConfig {
    /// Enable diversity enhancement
    pub enable: bool,
    /// Diversity metric
    pub metric: DiversityMetric,
    /// Diversity weight (0.0 = no diversity, 1.0 = maximum diversity)
    pub diversity_weight: f32,
    /// Maximum results to consider for diversity
    pub max_diverse_results: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityMetric {
    /// Maximum Marginal Relevance
    MMR,
    /// Angular diversity
    Angular,
    /// Clustering-based diversity
    ClusterBased,
    /// Content-based diversity
    ContentBased,
}

/// Result from a single source with metadata
#[derive(Debug, Clone)]
pub struct SourceResult {
    pub source_id: String,
    pub results: Vec<ScoredResult>,
    pub metadata: ResultMetadata,
}

/// Individual scored result
#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub item_id: String,
    pub score: f32,
    pub rank: usize,
    pub vector: Option<Vector>,
    pub metadata: Option<HashMap<String, String>>,
}

/// Metadata for result source
#[derive(Debug, Clone)]
pub struct ResultMetadata {
    pub source_type: SourceType,
    pub algorithm_used: String,
    pub total_candidates: usize,
    pub processing_time: std::time::Duration,
    pub quality_metrics: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub enum SourceType {
    VectorSearch,
    TextSearch,
    KnowledgeGraph,
    MultiModal,
    Hybrid,
}

/// Merged result with explanation
#[derive(Debug, Clone)]
pub struct MergedResult {
    pub item_id: String,
    pub final_score: f32,
    pub confidence_interval: Option<ConfidenceInterval>,
    pub source_contributions: Vec<SourceContribution>,
    pub explanation: Option<ResultExplanation>,
    pub diversity_score: Option<f32>,
}

/// Confidence interval for a result
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    pub lower_bound: f32,
    pub upper_bound: f32,
    pub confidence_level: f32,
}

/// Contribution from each source
#[derive(Debug, Clone)]
pub struct SourceContribution {
    pub source_id: String,
    pub original_score: f32,
    pub normalized_score: f32,
    pub weight: f32,
    pub rank: usize,
}

/// Explanation for result ranking
#[derive(Debug, Clone)]
pub struct ResultExplanation {
    pub ranking_factors: Vec<RankingFactor>,
    pub score_breakdown: HashMap<String, f32>,
    pub similar_items: Vec<String>,
    pub differentiating_features: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RankingFactor {
    pub factor_name: String,
    pub importance: f32,
    pub description: String,
}

/// Advanced result merging engine
pub struct AdvancedResultMerger {
    config: ResultMergingConfig,
    normalization_cache: HashMap<String, NormalizationParams>,
    fusion_stats: FusionStatistics,
}

/// Parameters for score normalization
#[derive(Debug, Clone)]
struct NormalizationParams {
    min_score: f32,
    max_score: f32,
    mean_score: f32,
    std_dev: f32,
}

/// Statistics for fusion operations
#[derive(Debug, Clone, Default)]
pub struct FusionStatistics {
    pub total_merges: usize,
    pub average_sources_per_merge: f32,
    pub score_distribution: HashMap<String, f32>,
    pub fusion_quality_metrics: HashMap<String, f32>,
}

impl AdvancedResultMerger {
    /// Create new result merger
    pub fn new(config: ResultMergingConfig) -> Self {
        Self {
            config,
            normalization_cache: HashMap::new(),
            fusion_stats: FusionStatistics::default(),
        }
    }

    /// Merge results from multiple sources
    pub fn merge_results(&mut self, sources: Vec<SourceResult>) -> Result<Vec<MergedResult>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }

        // Update statistics
        self.fusion_stats.total_merges += 1;
        self.fusion_stats.average_sources_per_merge = 
            (self.fusion_stats.average_sources_per_merge * (self.fusion_stats.total_merges - 1) as f32 + sources.len() as f32) 
            / self.fusion_stats.total_merges as f32;

        // Step 1: Normalize scores from each source
        let normalized_sources = self.normalize_sources(&sources)?;

        // Step 2: Collect all unique items
        let all_items = self.collect_unique_items(&normalized_sources);

        // Step 3: Apply fusion algorithm
        let mut merged_results = match self.config.fusion_algorithm {
            RankFusionAlgorithm::CombSUM => self.apply_combsum(&normalized_sources, &all_items)?,
            RankFusionAlgorithm::CombMNZ => self.apply_combmnz(&normalized_sources, &all_items)?,
            RankFusionAlgorithm::ReciprocalRankFusion => self.apply_rrf(&normalized_sources, &all_items)?,
            RankFusionAlgorithm::BordaFusion => self.apply_borda(&normalized_sources, &all_items)?,
            RankFusionAlgorithm::CondorcetFusion => self.apply_condorcet(&normalized_sources, &all_items)?,
        };

        // Step 4: Apply score combination strategy
        merged_results = self.apply_score_combination(merged_results, &normalized_sources)?;

        // Step 5: Calculate confidence intervals if enabled
        if self.config.confidence_intervals {
            merged_results = self.calculate_confidence_intervals(merged_results, &normalized_sources)?;
        }

        // Step 6: Generate explanations if enabled
        if self.config.enable_explanations {
            merged_results = self.generate_explanations(merged_results, &normalized_sources)?;
        }

        // Step 7: Apply diversity enhancement if configured
        if let Some(diversity_config) = &self.config.diversity_config {
            if diversity_config.enable {
                merged_results = self.enhance_diversity(merged_results, diversity_config)?;
            }
        }

        // Step 8: Sort by final score
        merged_results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());

        Ok(merged_results)
    }

    /// Normalize scores from all sources
    fn normalize_sources(&mut self, sources: &[SourceResult]) -> Result<Vec<SourceResult>> {
        let mut normalized = Vec::new();

        for source in sources {
            let normalized_source = self.normalize_source(source)?;
            normalized.push(normalized_source);
        }

        Ok(normalized)
    }

    /// Normalize a single source
    fn normalize_source(&mut self, source: &SourceResult) -> Result<SourceResult> {
        if source.results.is_empty() {
            return Ok(source.clone());
        }

        let scores: Vec<f32> = source.results.iter().map(|r| r.score).collect();
        let normalization_params = self.calculate_normalization_params(&scores);

        // Cache normalization parameters
        self.normalization_cache.insert(source.source_id.clone(), normalization_params.clone());

        let normalized_results: Vec<ScoredResult> = source.results
            .iter()
            .map(|result| {
                let normalized_score = self.normalize_score(result.score, &normalization_params);
                ScoredResult {
                    item_id: result.item_id.clone(),
                    score: normalized_score,
                    rank: result.rank,
                    vector: result.vector.clone(),
                    metadata: result.metadata.clone(),
                }
            })
            .collect();

        Ok(SourceResult {
            source_id: source.source_id.clone(),
            results: normalized_results,
            metadata: source.metadata.clone(),
        })
    }

    /// Calculate normalization parameters
    fn calculate_normalization_params(&self, scores: &[f32]) -> NormalizationParams {
        let min_score = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
        
        let variance = scores.iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f32>() / scores.len() as f32;
        let std_dev = variance.sqrt();

        NormalizationParams {
            min_score,
            max_score,
            mean_score,
            std_dev,
        }
    }

    /// Normalize a single score
    fn normalize_score(&self, score: f32, params: &NormalizationParams) -> f32 {
        match self.config.normalization_method {
            ScoreNormalizationMethod::None => score,
            ScoreNormalizationMethod::MinMax => {
                if params.max_score == params.min_score {
                    0.5 // Default to middle value if no variance
                } else {
                    (score - params.min_score) / (params.max_score - params.min_score)
                }
            }
            ScoreNormalizationMethod::ZScore => {
                if params.std_dev == 0.0 {
                    0.0 // Default to zero if no variance
                } else {
                    (score - params.mean_score) / params.std_dev
                }
            }
            ScoreNormalizationMethod::Softmax => {
                // For softmax, we need all scores, so this is a simplified version
                (score - params.min_score).exp()
            }
            ScoreNormalizationMethod::Sigmoid => {
                1.0 / (1.0 + (-score).exp())
            }
            ScoreNormalizationMethod::RankBased => {
                // This would require rank information
                score / params.max_score
            }
        }
    }

    /// Collect all unique items from sources
    fn collect_unique_items(&self, sources: &[SourceResult]) -> HashSet<String> {
        let mut items = HashSet::new();
        for source in sources {
            for result in &source.results {
                items.insert(result.item_id.clone());
            }
        }
        items
    }

    /// Apply CombSUM fusion algorithm
    fn apply_combsum(&self, sources: &[SourceResult], items: &HashSet<String>) -> Result<Vec<MergedResult>> {
        let mut merged_results = Vec::new();

        for item_id in items {
            let mut total_score = 0.0;
            let mut source_contributions = Vec::new();

            for source in sources {
                if let Some(result) = source.results.iter().find(|r| r.item_id == *item_id) {
                    let weight = self.config.source_weights.get(&source.source_id).copied().unwrap_or(1.0);
                    let weighted_score = result.score * weight;
                    total_score += weighted_score;

                    source_contributions.push(SourceContribution {
                        source_id: source.source_id.clone(),
                        original_score: result.score,
                        normalized_score: result.score,
                        weight,
                        rank: result.rank,
                    });
                }
            }

            merged_results.push(MergedResult {
                item_id: item_id.clone(),
                final_score: total_score,
                confidence_interval: None,
                source_contributions,
                explanation: None,
                diversity_score: None,
            });
        }

        Ok(merged_results)
    }

    /// Apply CombMNZ fusion algorithm
    fn apply_combmnz(&self, sources: &[SourceResult], items: &HashSet<String>) -> Result<Vec<MergedResult>> {
        let mut merged_results = Vec::new();

        for item_id in items {
            let mut total_score = 0.0;
            let mut non_zero_count = 0;
            let mut source_contributions = Vec::new();

            for source in sources {
                if let Some(result) = source.results.iter().find(|r| r.item_id == *item_id) {
                    let weight = self.config.source_weights.get(&source.source_id).copied().unwrap_or(1.0);
                    let weighted_score = result.score * weight;
                    
                    if weighted_score > 0.0 {
                        total_score += weighted_score;
                        non_zero_count += 1;
                    }

                    source_contributions.push(SourceContribution {
                        source_id: source.source_id.clone(),
                        original_score: result.score,
                        normalized_score: result.score,
                        weight,
                        rank: result.rank,
                    });
                }
            }

            let final_score = if non_zero_count > 0 {
                total_score * non_zero_count as f32
            } else {
                0.0
            };

            merged_results.push(MergedResult {
                item_id: item_id.clone(),
                final_score,
                confidence_interval: None,
                source_contributions,
                explanation: None,
                diversity_score: None,
            });
        }

        Ok(merged_results)
    }

    /// Apply Reciprocal Rank Fusion
    fn apply_rrf(&self, sources: &[SourceResult], items: &HashSet<String>) -> Result<Vec<MergedResult>> {
        let k = 60.0; // RRF constant
        let mut merged_results = Vec::new();

        for item_id in items {
            let mut rrf_score = 0.0;
            let mut source_contributions = Vec::new();

            for source in sources {
                if let Some(result) = source.results.iter().find(|r| r.item_id == *item_id) {
                    let weight = self.config.source_weights.get(&source.source_id).copied().unwrap_or(1.0);
                    let rrf_contribution = weight / (k + result.rank as f32);
                    rrf_score += rrf_contribution;

                    source_contributions.push(SourceContribution {
                        source_id: source.source_id.clone(),
                        original_score: result.score,
                        normalized_score: rrf_contribution,
                        weight,
                        rank: result.rank,
                    });
                }
            }

            merged_results.push(MergedResult {
                item_id: item_id.clone(),
                final_score: rrf_score,
                confidence_interval: None,
                source_contributions,
                explanation: None,
                diversity_score: None,
            });
        }

        Ok(merged_results)
    }

    /// Apply Borda fusion
    fn apply_borda(&self, sources: &[SourceResult], items: &HashSet<String>) -> Result<Vec<MergedResult>> {
        let mut merged_results = Vec::new();

        for item_id in items {
            let mut borda_score = 0.0;
            let mut source_contributions = Vec::new();

            for source in sources {
                if let Some(result) = source.results.iter().find(|r| r.item_id == *item_id) {
                    let weight = self.config.source_weights.get(&source.source_id).copied().unwrap_or(1.0);
                    let max_rank = source.results.len() as f32;
                    let borda_contribution = weight * (max_rank - result.rank as f32);
                    borda_score += borda_contribution;

                    source_contributions.push(SourceContribution {
                        source_id: source.source_id.clone(),
                        original_score: result.score,
                        normalized_score: borda_contribution,
                        weight,
                        rank: result.rank,
                    });
                }
            }

            merged_results.push(MergedResult {
                item_id: item_id.clone(),
                final_score: borda_score,
                confidence_interval: None,
                source_contributions,
                explanation: None,
                diversity_score: None,
            });
        }

        Ok(merged_results)
    }

    /// Apply Condorcet fusion (simplified)
    fn apply_condorcet(&self, sources: &[SourceResult], items: &HashSet<String>) -> Result<Vec<MergedResult>> {
        // For simplicity, we'll use a vote-based approach
        // In a full implementation, this would involve pairwise comparisons
        self.apply_borda(sources, items)
    }

    /// Apply score combination strategy
    fn apply_score_combination(&self, mut results: Vec<MergedResult>, sources: &[SourceResult]) -> Result<Vec<MergedResult>> {
        match self.config.combination_strategy {
            ScoreCombinationStrategy::Average => {
                for result in &mut results {
                    if !result.source_contributions.is_empty() {
                        result.final_score = result.source_contributions.iter()
                            .map(|c| c.normalized_score)
                            .sum::<f32>() / result.source_contributions.len() as f32;
                    }
                }
            }
            ScoreCombinationStrategy::WeightedSum => {
                // Already handled in fusion algorithms
            }
            ScoreCombinationStrategy::Maximum => {
                for result in &mut results {
                    result.final_score = result.source_contributions.iter()
                        .map(|c| c.normalized_score)
                        .fold(0.0, f32::max);
                }
            }
            ScoreCombinationStrategy::Minimum => {
                for result in &mut results {
                    result.final_score = result.source_contributions.iter()
                        .map(|c| c.normalized_score)
                        .fold(f32::INFINITY, f32::min);
                }
            }
            ScoreCombinationStrategy::GeometricMean => {
                for result in &mut results {
                    let product: f32 = result.source_contributions.iter()
                        .map(|c| c.normalized_score.max(0.001)) // Avoid zero values
                        .product();
                    result.final_score = product.powf(1.0 / result.source_contributions.len() as f32);
                }
            }
            _ => {
                // Other strategies would be implemented here
            }
        }

        Ok(results)
    }

    /// Calculate confidence intervals
    fn calculate_confidence_intervals(&self, mut results: Vec<MergedResult>, _sources: &[SourceResult]) -> Result<Vec<MergedResult>> {
        for result in &mut results {
            if result.source_contributions.len() > 1 {
                let scores: Vec<f32> = result.source_contributions.iter()
                    .map(|c| c.normalized_score)
                    .collect();
                
                let mean = scores.iter().sum::<f32>() / scores.len() as f32;
                let variance = scores.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>() / scores.len() as f32;
                let std_dev = variance.sqrt();
                
                // 95% confidence interval (approximation)
                let margin = 1.96 * std_dev / (scores.len() as f32).sqrt();
                
                result.confidence_interval = Some(ConfidenceInterval {
                    lower_bound: (mean - margin).max(0.0),
                    upper_bound: (mean + margin).min(1.0),
                    confidence_level: 0.95,
                });
            }
        }

        Ok(results)
    }

    /// Generate explanations for results
    fn generate_explanations(&self, mut results: Vec<MergedResult>, _sources: &[SourceResult]) -> Result<Vec<MergedResult>> {
        for result in &mut results {
            let mut ranking_factors = Vec::new();
            let mut score_breakdown = HashMap::new();

            // Analyze source contributions
            for contribution in &result.source_contributions {
                ranking_factors.push(RankingFactor {
                    factor_name: format!("Source: {}", contribution.source_id),
                    importance: contribution.normalized_score,
                    description: format!("Contribution from {} with weight {}", 
                                       contribution.source_id, contribution.weight),
                });
                
                score_breakdown.insert(contribution.source_id.clone(), contribution.normalized_score);
            }

            result.explanation = Some(ResultExplanation {
                ranking_factors,
                score_breakdown,
                similar_items: Vec::new(), // Would be populated in a full implementation
                differentiating_features: Vec::new(), // Would be populated in a full implementation
            });
        }

        Ok(results)
    }

    /// Enhance diversity of results
    fn enhance_diversity(&self, mut results: Vec<MergedResult>, diversity_config: &DiversityConfig) -> Result<Vec<MergedResult>> {
        if results.len() <= diversity_config.max_diverse_results {
            return Ok(results);
        }

        // Simple diversity enhancement using Maximum Marginal Relevance (MMR)
        let mut selected = Vec::new();
        let mut remaining = results;

        // Always select the top result first
        if !remaining.is_empty() {
            let top_result = remaining.remove(0);
            selected.push(top_result);
        }

        // Select remaining results balancing relevance and diversity
        while selected.len() < diversity_config.max_diverse_results && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_mmr = f32::NEG_INFINITY;

            for (i, candidate) in remaining.iter().enumerate() {
                // Calculate MMR score
                let relevance = candidate.final_score;
                let max_similarity = self.calculate_max_similarity_to_selected(candidate, &selected);
                let mmr = diversity_config.diversity_weight * relevance - 
                         (1.0 - diversity_config.diversity_weight) * max_similarity;

                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = i;
                }
            }

            let selected_result = remaining.remove(best_idx);
            selected.push(selected_result);
        }

        // Add diversity scores
        for result in &mut selected {
            result.diversity_score = Some(0.8); // Placeholder - would be calculated properly
        }

        Ok(selected)
    }

    /// Calculate maximum similarity to already selected results
    fn calculate_max_similarity_to_selected(&self, candidate: &MergedResult, selected: &[MergedResult]) -> f32 {
        if selected.is_empty() {
            return 0.0;
        }

        // Simplified similarity calculation
        // In a full implementation, this would use actual vector similarities
        let mut max_similarity: f32 = 0.0;
        
        for selected_result in selected {
            // Simple similarity based on score difference
            let similarity: f32 = 1.0 - (candidate.final_score - selected_result.final_score).abs();
            max_similarity = max_similarity.max(similarity);
        }

        max_similarity
    }

    /// Get fusion statistics
    pub fn get_statistics(&self) -> &FusionStatistics {
        &self.fusion_stats
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.fusion_stats = FusionStatistics::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_source(source_id: &str, results: Vec<(String, f32, usize)>) -> SourceResult {
        let scored_results = results
            .into_iter()
            .map(|(id, score, rank)| ScoredResult {
                item_id: id,
                score,
                rank,
                vector: None,
                metadata: None,
            })
            .collect();

        SourceResult {
            source_id: source_id.to_string(),
            results: scored_results,
            metadata: ResultMetadata {
                source_type: SourceType::VectorSearch,
                algorithm_used: "test".to_string(),
                total_candidates: 100,
                processing_time: Duration::from_millis(10),
                quality_metrics: HashMap::new(),
            },
        }
    }

    #[test]
    fn test_combsum_fusion() {
        let config = ResultMergingConfig::default();
        let mut merger = AdvancedResultMerger::new(config);

        let source1 = create_test_source("source1", vec![
            ("doc1".to_string(), 0.9, 1),
            ("doc2".to_string(), 0.8, 2),
        ]);

        let source2 = create_test_source("source2", vec![
            ("doc1".to_string(), 0.7, 1),
            ("doc3".to_string(), 0.6, 2),
        ]);

        let merged = merger.merge_results(vec![source1, source2]).unwrap();
        
        assert_eq!(merged.len(), 3); // doc1, doc2, doc3
        
        // doc1 should have the highest score (appears in both sources)
        let doc1_result = merged.iter().find(|r| r.item_id == "doc1").unwrap();
        assert!(doc1_result.final_score > 1.0); // Should be sum of normalized scores
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let mut config = ResultMergingConfig::default();
        config.fusion_algorithm = RankFusionAlgorithm::ReciprocalRankFusion;
        
        let mut merger = AdvancedResultMerger::new(config);

        let source1 = create_test_source("source1", vec![
            ("doc1".to_string(), 0.9, 1),
            ("doc2".to_string(), 0.8, 2),
        ]);

        let source2 = create_test_source("source2", vec![
            ("doc2".to_string(), 0.7, 1),
            ("doc1".to_string(), 0.6, 2),
        ]);

        let merged = merger.merge_results(vec![source1, source2]).unwrap();
        
        assert_eq!(merged.len(), 2);
        
        // Both documents appear in both sources, so both should have RRF scores
        for result in &merged {
            assert!(result.final_score > 0.0);
            assert_eq!(result.source_contributions.len(), 2);
        }
    }

    #[test]
    fn test_confidence_intervals() {
        let mut config = ResultMergingConfig::default();
        config.confidence_intervals = true;
        
        let mut merger = AdvancedResultMerger::new(config);

        let source1 = create_test_source("source1", vec![
            ("doc1".to_string(), 0.9, 1),
        ]);

        let source2 = create_test_source("source2", vec![
            ("doc1".to_string(), 0.7, 1),
        ]);

        let merged = merger.merge_results(vec![source1, source2]).unwrap();
        
        assert_eq!(merged.len(), 1);
        
        let result = &merged[0];
        assert!(result.confidence_interval.is_some());
        
        let ci = result.confidence_interval.as_ref().unwrap();
        assert!(ci.lower_bound <= ci.upper_bound);
        assert_eq!(ci.confidence_level, 0.95);
    }

    #[test]
    fn test_score_normalization() {
        let mut config = ResultMergingConfig::default();
        config.normalization_method = ScoreNormalizationMethod::MinMax;
        
        let mut merger = AdvancedResultMerger::new(config);

        let source = create_test_source("source1", vec![
            ("doc1".to_string(), 10.0, 1),
            ("doc2".to_string(), 5.0, 2),
            ("doc3".to_string(), 0.0, 3),
        ]);

        let normalized = merger.normalize_source(&source).unwrap();
        
        // After min-max normalization, scores should be in [0, 1]
        for result in &normalized.results {
            assert!(result.score >= 0.0 && result.score <= 1.0);
        }
    }
}