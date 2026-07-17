//! Recommendation system evaluation module
//!
//! This module provides comprehensive evaluation for recommendation systems using
//! embedding models, including precision, recall, coverage, diversity, and user
//! satisfaction metrics.

use super::ApplicationEvalConfig;
use crate::{EmbeddingModel, Vector};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// User interaction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteraction {
    /// User identifier
    pub user_id: String,
    /// Item identifier
    pub item_id: String,
    /// Interaction type (view, like, purchase, etc.)
    pub interaction_type: InteractionType,
    /// Rating (if applicable)
    pub rating: Option<f64>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Contextual features
    pub context: HashMap<String, String>,
}

/// Types of user interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    View,
    Like,
    Dislike,
    Purchase,
    AddToCart,
    Share,
    Comment,
    Rating,
}

/// Item metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemMetadata {
    /// Item identifier
    pub item_id: String,
    /// Item category
    pub category: String,
    /// Item features
    pub features: HashMap<String, String>,
    /// Item popularity score
    pub popularity: f64,
    /// Item embedding (if available)
    pub embedding: Option<Vec<f32>>,
}

/// Recommendation evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationMetric {
    /// Precision at K
    PrecisionAtK(usize),
    /// Recall at K
    RecallAtK(usize),
    /// F1 score at K
    F1AtK(usize),
    /// Mean Average Precision
    MAP,
    /// Normalized Discounted Cumulative Gain
    NDCG(usize),
    /// Mean Reciprocal Rank
    MRR,
    /// Coverage (catalog coverage)
    Coverage,
    /// Diversity
    Diversity,
    /// Novelty
    Novelty,
    /// Serendipity
    Serendipity,
}

/// Per-user recommendation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRecommendationResults {
    /// User identifier
    pub user_id: String,
    /// Precision scores at different K values
    pub precision_scores: HashMap<usize, f64>,
    /// Recall scores at different K values
    pub recall_scores: HashMap<usize, f64>,
    /// NDCG scores
    pub ndcg_scores: HashMap<usize, f64>,
    /// Personalization score
    pub personalization_score: f64,
    /// Full ranked list of recommended item IDs (highest-scored first), kept
    /// so aggregate metrics needing the whole ranking (MAP, MRR) and
    /// catalog-level statistics (coverage, diversity, novelty) can be
    /// computed for real instead of guessed at.
    pub recommended_items: Vec<String>,
    /// Ground-truth relevant item IDs for this user (from held-out test
    /// interactions), used alongside `recommended_items` for MAP/MRR.
    pub ground_truth: HashSet<String>,
}

/// Coverage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageStats {
    /// Catalog coverage percentage
    pub catalog_coverage: f64,
    /// Number of unique items recommended
    pub unique_items_recommended: usize,
    /// Total items in catalog
    pub total_catalog_items: usize,
    /// Long-tail coverage
    pub long_tail_coverage: f64,
}

/// Diversity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityAnalysis {
    /// Intra-list diversity (average)
    pub intra_list_diversity: f64,
    /// Inter-user diversity
    pub inter_user_diversity: f64,
    /// Category diversity
    pub category_diversity: f64,
    /// Feature diversity
    pub feature_diversity: f64,
}

/// A/B test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestResults {
    /// Control group performance
    pub control_performance: f64,
    /// Treatment group performance
    pub treatment_performance: f64,
    /// Statistical significance
    pub p_value: f64,
    /// Effect size
    pub effect_size: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Recommendation evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationResults {
    /// Metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Per-user results
    pub per_user_results: HashMap<String, UserRecommendationResults>,
    /// Coverage statistics
    pub coverage_stats: CoverageStats,
    /// Diversity analysis
    pub diversity_analysis: DiversityAnalysis,
    /// User satisfaction scores
    pub user_satisfaction: Option<HashMap<String, f64>>,
    /// A/B test results (if applicable)
    pub ab_test_results: Option<ABTestResults>,
}

/// Recommendation system evaluator
pub struct RecommendationEvaluator {
    /// User interaction history
    user_interactions: HashMap<String, Vec<UserInteraction>>,
    /// Item catalog
    item_catalog: HashMap<String, ItemMetadata>,
    /// Evaluation metrics
    metrics: Vec<RecommendationMetric>,
}

impl RecommendationEvaluator {
    /// Create a new recommendation evaluator
    pub fn new() -> Self {
        Self {
            user_interactions: HashMap::new(),
            item_catalog: HashMap::new(),
            metrics: vec![
                RecommendationMetric::PrecisionAtK(5),
                RecommendationMetric::PrecisionAtK(10),
                RecommendationMetric::RecallAtK(5),
                RecommendationMetric::RecallAtK(10),
                RecommendationMetric::NDCG(10),
                RecommendationMetric::MAP,
                RecommendationMetric::Coverage,
                RecommendationMetric::Diversity,
            ],
        }
    }

    /// Add user interaction data
    pub fn add_interaction(&mut self, interaction: UserInteraction) {
        self.user_interactions
            .entry(interaction.user_id.clone())
            .or_default()
            .push(interaction);
    }

    /// Add item to catalog
    pub fn add_item(&mut self, item: ItemMetadata) {
        self.item_catalog.insert(item.item_id.clone(), item);
    }

    /// Evaluate recommendation quality
    pub async fn evaluate(
        &self,
        model: &dyn EmbeddingModel,
        config: &ApplicationEvalConfig,
    ) -> Result<RecommendationResults> {
        let mut metric_scores = HashMap::new();
        let mut per_user_results = HashMap::new();

        // Sample users for evaluation
        let users_to_evaluate: Vec<_> = self
            .user_interactions
            .keys()
            .take(config.sample_size)
            .cloned()
            .collect();

        for user_id in &users_to_evaluate {
            let user_results = self
                .evaluate_user_recommendations(user_id, model, config)
                .await?;
            per_user_results.insert(user_id.clone(), user_results);
        }

        // Calculate aggregate metrics
        for metric in &self.metrics {
            let score = self.calculate_metric(metric, &per_user_results)?;
            metric_scores.insert(format!("{metric:?}"), score);
        }

        // Calculate coverage and diversity
        let coverage_stats = self.calculate_coverage_stats(&per_user_results)?;
        let diversity_analysis = self.calculate_diversity_analysis(&per_user_results)?;

        // User satisfaction (if enabled)
        let user_satisfaction = if config.enable_user_satisfaction {
            Some(self.simulate_user_satisfaction(&per_user_results)?)
        } else {
            None
        };

        Ok(RecommendationResults {
            metric_scores,
            per_user_results,
            coverage_stats,
            diversity_analysis,
            user_satisfaction,
            ab_test_results: None, // Would be populated in real A/B testing scenarios
        })
    }

    /// Evaluate recommendations for a specific user
    async fn evaluate_user_recommendations(
        &self,
        user_id: &str,
        model: &dyn EmbeddingModel,
        config: &ApplicationEvalConfig,
    ) -> Result<UserRecommendationResults> {
        let user_interactions = self
            .user_interactions
            .get(user_id)
            .expect("user_id should exist in user_interactions");

        // Split interactions into training and test sets
        let split_point = (user_interactions.len() as f64 * 0.8) as usize;
        let training_interactions = &user_interactions[..split_point];
        let test_interactions = &user_interactions[split_point..];

        if test_interactions.is_empty() {
            return Err(anyhow!("No test interactions for user {}", user_id));
        }

        // Generate recommendations based on training interactions
        let recommendations = self
            .generate_recommendations(
                user_id,
                training_interactions,
                model,
                config.num_recommendations,
            )
            .await?;

        // Extract ground truth items from test interactions
        let ground_truth: HashSet<String> = test_interactions
            .iter()
            .filter(|i| {
                matches!(
                    i.interaction_type,
                    InteractionType::Like | InteractionType::Purchase
                )
            })
            .map(|i| i.item_id.clone())
            .collect();

        // Calculate precision and recall at different K values
        let mut precision_scores = HashMap::new();
        let mut recall_scores = HashMap::new();
        let mut ndcg_scores = HashMap::new();

        for &k in &[1, 3, 5, 10] {
            if k <= recommendations.len() {
                let top_k_recs: HashSet<String> = recommendations
                    .iter()
                    .take(k)
                    .map(|(item_id, _)| item_id.clone())
                    .collect();

                let tp = top_k_recs.intersection(&ground_truth).count() as f64;
                let precision = tp / k as f64;
                let recall = if !ground_truth.is_empty() {
                    tp / ground_truth.len() as f64
                } else {
                    0.0
                };

                precision_scores.insert(k, precision);
                recall_scores.insert(k, recall);

                // Calculate NDCG
                let ndcg = self.calculate_ndcg(&recommendations, &ground_truth, k)?;
                ndcg_scores.insert(k, ndcg);
            }
        }

        // Calculate personalization score
        let personalization_score =
            self.calculate_personalization_score(user_id, &recommendations, training_interactions)?;

        Ok(UserRecommendationResults {
            user_id: user_id.to_string(),
            precision_scores,
            recall_scores,
            ndcg_scores,
            personalization_score,
            recommended_items: recommendations
                .iter()
                .map(|(item_id, _)| item_id.clone())
                .collect(),
            ground_truth,
        })
    }

    /// Generate recommendations for a user
    async fn generate_recommendations(
        &self,
        _user_id: &str,
        interactions: &[UserInteraction],
        model: &dyn EmbeddingModel,
        num_recommendations: usize,
    ) -> Result<Vec<(String, f64)>> {
        // Create user profile based on interactions
        let user_profile = self.create_user_profile(interactions, model).await?;

        // Score all items in catalog
        let mut item_scores = Vec::new();
        for (item_id, item_metadata) in &self.item_catalog {
            // Skip items the user has already interacted with
            if interactions.iter().any(|i| &i.item_id == item_id) {
                continue;
            }

            let item_score = self
                .score_item_for_user(&user_profile, item_metadata, model)
                .await?;
            item_scores.push((item_id.clone(), item_score));
        }

        // Sort by score and return top K
        item_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        item_scores.truncate(num_recommendations);

        Ok(item_scores)
    }

    /// Create user profile from interactions
    async fn create_user_profile(
        &self,
        interactions: &[UserInteraction],
        model: &dyn EmbeddingModel,
    ) -> Result<Vector> {
        let mut profile_embeddings = Vec::new();

        for interaction in interactions {
            if let Ok(item_embedding) = model.get_entity_embedding(&interaction.item_id) {
                // Weight by interaction type
                let weight = match interaction.interaction_type {
                    InteractionType::Purchase => 3.0,
                    InteractionType::Like => 2.0,
                    InteractionType::View => 1.0,
                    InteractionType::Dislike => -1.0,
                    _ => 1.0,
                };

                // Weight by rating if available
                let rating_weight = interaction.rating.unwrap_or(1.0);
                let final_weight = weight * rating_weight;

                let weighted_embedding: Vec<f32> = item_embedding
                    .values
                    .iter()
                    .map(|&x| x * final_weight as f32)
                    .collect();

                profile_embeddings.push(weighted_embedding);
            }
        }

        if profile_embeddings.is_empty() {
            return Ok(Vector::new(vec![0.0; 100])); // Default empty profile
        }

        // Average the embeddings
        let dim = profile_embeddings[0].len();
        let mut avg_embedding = vec![0.0f32; dim];

        for embedding in &profile_embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                avg_embedding[i] += value;
            }
        }

        for value in &mut avg_embedding {
            *value /= profile_embeddings.len() as f32;
        }

        Ok(Vector::new(avg_embedding))
    }

    /// Score an item for a user
    async fn score_item_for_user(
        &self,
        user_profile: &Vector,
        item: &ItemMetadata,
        model: &dyn EmbeddingModel,
    ) -> Result<f64> {
        // Get item embedding
        let item_embedding = if let Some(ref embedding) = item.embedding {
            Vector::new(embedding.clone())
        } else {
            model.get_entity_embedding(&item.item_id)?
        };

        // Calculate cosine similarity
        let similarity = self.cosine_similarity(user_profile, &item_embedding);

        // Add popularity bias (small weight)
        let popularity_score = item.popularity * 0.1;

        Ok(similarity + popularity_score)
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, v1: &Vector, v2: &Vector) -> f64 {
        let dot_product: f32 = v1
            .values
            .iter()
            .zip(v2.values.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = v1.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = v2.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot_product / (norm_a * norm_b)) as f64
        } else {
            0.0
        }
    }

    /// Calculate NDCG score
    fn calculate_ndcg(
        &self,
        recommendations: &[(String, f64)],
        ground_truth: &HashSet<String>,
        k: usize,
    ) -> Result<f64> {
        if k == 0 || recommendations.is_empty() {
            return Ok(0.0);
        }

        let mut dcg = 0.0;
        for (i, (item_id, _)) in recommendations.iter().take(k).enumerate() {
            if ground_truth.contains(item_id) {
                dcg += 1.0 / (i as f64 + 2.0).log2(); // +2 because rank starts from 1
            }
        }

        // Calculate ideal DCG
        let relevant_items = ground_truth.len().min(k);
        let mut idcg = 0.0;
        for i in 0..relevant_items {
            idcg += 1.0 / (i as f64 + 2.0).log2();
        }

        if idcg > 0.0 {
            Ok(dcg / idcg)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate personalization score
    fn calculate_personalization_score(
        &self,
        _user_id: &str,
        recommendations: &[(String, f64)],
        user_interactions: &[UserInteraction],
    ) -> Result<f64> {
        if recommendations.is_empty() || user_interactions.is_empty() {
            return Ok(0.0);
        }

        // Calculate how well recommendations match user's historical preferences
        let user_categories: HashSet<String> = user_interactions
            .iter()
            .filter_map(|i| self.item_catalog.get(&i.item_id))
            .map(|item| item.category.clone())
            .collect();

        let recommendation_categories: HashSet<String> = recommendations
            .iter()
            .filter_map(|(item_id, _)| self.item_catalog.get(item_id))
            .map(|item| item.category.clone())
            .collect();

        if user_categories.is_empty() {
            return Ok(0.0);
        }

        let overlap = user_categories
            .intersection(&recommendation_categories)
            .count();
        Ok(overlap as f64 / user_categories.len() as f64)
    }

    /// Calculate aggregate metric from per-user results
    fn calculate_metric(
        &self,
        metric: &RecommendationMetric,
        per_user_results: &HashMap<String, UserRecommendationResults>,
    ) -> Result<f64> {
        if per_user_results.is_empty() {
            return Ok(0.0);
        }

        match metric {
            RecommendationMetric::PrecisionAtK(k) => {
                let scores: Vec<f64> = per_user_results
                    .values()
                    .filter_map(|r| r.precision_scores.get(k))
                    .cloned()
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RecommendationMetric::RecallAtK(k) => {
                let scores: Vec<f64> = per_user_results
                    .values()
                    .filter_map(|r| r.recall_scores.get(k))
                    .cloned()
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RecommendationMetric::NDCG(k) => {
                let scores: Vec<f64> = per_user_results
                    .values()
                    .filter_map(|r| r.ndcg_scores.get(k))
                    .cloned()
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RecommendationMetric::F1AtK(k) => {
                let scores: Vec<f64> = per_user_results
                    .values()
                    .filter_map(|r| {
                        let precision = *r.precision_scores.get(k)?;
                        let recall = *r.recall_scores.get(k)?;
                        Some(if precision + recall > 0.0 {
                            2.0 * precision * recall / (precision + recall)
                        } else {
                            0.0
                        })
                    })
                    .collect();
                if scores.is_empty() {
                    return Err(anyhow!(
                        "No precision/recall@{k} scores available to compute F1@{k}"
                    ));
                }
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RecommendationMetric::MAP => {
                let scores: Vec<f64> = per_user_results
                    .values()
                    .map(|r| Self::average_precision(&r.recommended_items, &r.ground_truth))
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RecommendationMetric::MRR => {
                let scores: Vec<f64> = per_user_results
                    .values()
                    .map(|r| Self::reciprocal_rank(&r.recommended_items, &r.ground_truth))
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RecommendationMetric::Coverage => Ok(self
                .calculate_coverage_stats(per_user_results)?
                .catalog_coverage),
            RecommendationMetric::Diversity => Ok(self
                .calculate_diversity_analysis(per_user_results)?
                .intra_list_diversity),
            RecommendationMetric::Novelty => {
                let scores: Vec<f64> = per_user_results
                    .values()
                    .map(|r| self.novelty_for_items(&r.recommended_items))
                    .collect();
                if scores.is_empty() {
                    return Ok(0.0);
                }
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RecommendationMetric::Serendipity => {
                // Simplified but real serendipity: the fraction of
                // recommended items that are both relevant (in the user's
                // ground truth) and unpopular (below the catalog's median
                // popularity) — i.e. "pleasant surprises" rather than
                // popularity-driven hits.
                let median_popularity = self.median_catalog_popularity();
                let scores: Vec<f64> = per_user_results
                    .values()
                    .filter(|r| !r.recommended_items.is_empty())
                    .map(|r| {
                        let surprising_hits = r
                            .recommended_items
                            .iter()
                            .filter(|item_id| {
                                r.ground_truth.contains(*item_id)
                                    && self
                                        .item_catalog
                                        .get(*item_id)
                                        .is_some_and(|item| item.popularity < median_popularity)
                            })
                            .count();
                        surprising_hits as f64 / r.recommended_items.len() as f64
                    })
                    .collect();
                if scores.is_empty() {
                    return Ok(0.0);
                }
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
        }
    }

    /// Average precision of a ranked recommendation list against a
    /// user's ground-truth relevant items.
    fn average_precision(recommended_items: &[String], ground_truth: &HashSet<String>) -> f64 {
        if ground_truth.is_empty() {
            return 0.0;
        }
        let mut hits = 0usize;
        let mut precision_sum = 0.0;
        for (rank, item_id) in recommended_items.iter().enumerate() {
            if ground_truth.contains(item_id) {
                hits += 1;
                precision_sum += hits as f64 / (rank + 1) as f64;
            }
        }
        precision_sum / ground_truth.len() as f64
    }

    /// Reciprocal rank of the first relevant item in a ranked recommendation
    /// list (0.0 if none of the recommended items are relevant).
    fn reciprocal_rank(recommended_items: &[String], ground_truth: &HashSet<String>) -> f64 {
        recommended_items
            .iter()
            .position(|item_id| ground_truth.contains(item_id))
            .map(|rank| 1.0 / (rank + 1) as f64)
            .unwrap_or(0.0)
    }

    /// Novelty of a recommendation list: the average "unpopularity"
    /// (`1 - popularity`) of its items, using the catalog's recorded
    /// popularity scores. Unknown items are skipped.
    fn novelty_for_items(&self, recommended_items: &[String]) -> f64 {
        let popularities: Vec<f64> = recommended_items
            .iter()
            .filter_map(|item_id| self.item_catalog.get(item_id))
            .map(|item| item.popularity.clamp(0.0, 1.0))
            .collect();
        if popularities.is_empty() {
            return 0.0;
        }
        1.0 - popularities.iter().sum::<f64>() / popularities.len() as f64
    }

    /// Median popularity across the whole item catalog, used as the
    /// "unpopular" threshold for serendipity.
    fn median_catalog_popularity(&self) -> f64 {
        let mut popularities: Vec<f64> = self
            .item_catalog
            .values()
            .map(|item| item.popularity)
            .collect();
        if popularities.is_empty() {
            return 0.0;
        }
        popularities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = popularities.len() / 2;
        if popularities.len() % 2 == 0 {
            (popularities[mid - 1] + popularities[mid]) / 2.0
        } else {
            popularities[mid]
        }
    }

    /// Calculate catalog coverage statistics from the items actually
    /// recommended across all evaluated users.
    fn calculate_coverage_stats(
        &self,
        per_user_results: &HashMap<String, UserRecommendationResults>,
    ) -> Result<CoverageStats> {
        let total_catalog_items = self.item_catalog.len();

        let recommended_item_set: HashSet<&str> = per_user_results
            .values()
            .flat_map(|r| r.recommended_items.iter().map(String::as_str))
            .collect();
        let unique_items_recommended = recommended_item_set.len();

        let catalog_coverage = if total_catalog_items > 0 {
            unique_items_recommended as f64 / total_catalog_items as f64
        } else {
            0.0
        };

        // Long-tail items: those in the catalog's bottom-half by popularity.
        let median_popularity = self.median_catalog_popularity();
        let long_tail_items: HashSet<&str> = self
            .item_catalog
            .values()
            .filter(|item| item.popularity < median_popularity)
            .map(|item| item.item_id.as_str())
            .collect();
        let long_tail_coverage = if !long_tail_items.is_empty() {
            recommended_item_set.intersection(&long_tail_items).count() as f64
                / long_tail_items.len() as f64
        } else {
            0.0
        };

        Ok(CoverageStats {
            catalog_coverage,
            unique_items_recommended,
            total_catalog_items,
            long_tail_coverage,
        })
    }

    /// Calculate diversity analysis from the items actually recommended
    /// across all evaluated users, using catalog category/feature metadata
    /// (and item embeddings when available) as the similarity signal.
    fn calculate_diversity_analysis(
        &self,
        per_user_results: &HashMap<String, UserRecommendationResults>,
    ) -> Result<DiversityAnalysis> {
        let non_empty_users: Vec<&UserRecommendationResults> = per_user_results
            .values()
            .filter(|r| !r.recommended_items.is_empty())
            .collect();

        if non_empty_users.is_empty() {
            return Ok(DiversityAnalysis {
                intra_list_diversity: 0.0,
                inter_user_diversity: 0.0,
                category_diversity: 0.0,
                feature_diversity: 0.0,
            });
        }

        // Intra-list diversity: mean pairwise dissimilarity between items
        // within each user's own recommendation list, using item embeddings
        // when available and falling back to "different category" otherwise.
        let mut intra_scores = Vec::new();
        // Category diversity: fraction of a user's list that is distinct
        // categories.
        let mut category_scores = Vec::new();
        // Feature diversity: fraction of a user's list that is distinct
        // (key, value) feature pairs, aggregated across the list.
        let mut feature_scores = Vec::new();

        for result in &non_empty_users {
            let items: Vec<&ItemMetadata> = result
                .recommended_items
                .iter()
                .filter_map(|item_id| self.item_catalog.get(item_id))
                .collect();
            if items.is_empty() {
                continue;
            }

            let mut pair_count = 0usize;
            let mut dissimilarity_sum = 0.0;
            for i in 0..items.len() {
                for j in (i + 1)..items.len() {
                    let dissimilarity = match (&items[i].embedding, &items[j].embedding) {
                        (Some(a), Some(b)) => 1.0 - Self::cosine_similarity_slices(a, b),
                        _ => {
                            if items[i].category == items[j].category {
                                0.0
                            } else {
                                1.0
                            }
                        }
                    };
                    dissimilarity_sum += dissimilarity;
                    pair_count += 1;
                }
            }
            if pair_count > 0 {
                intra_scores.push(dissimilarity_sum / pair_count as f64);
            }

            let distinct_categories: HashSet<&str> =
                items.iter().map(|item| item.category.as_str()).collect();
            category_scores.push(distinct_categories.len() as f64 / items.len() as f64);

            let distinct_features: HashSet<(&str, &str)> = items
                .iter()
                .flat_map(|item| item.features.iter().map(|(k, v)| (k.as_str(), v.as_str())))
                .collect();
            let total_feature_pairs: usize = items.iter().map(|item| item.features.len()).sum();
            if total_feature_pairs > 0 {
                feature_scores.push(distinct_features.len() as f64 / total_feature_pairs as f64);
            }
        }

        // Inter-user diversity: mean pairwise Jaccard *dissimilarity*
        // between different users' recommended-item sets.
        let user_item_sets: Vec<HashSet<&str>> = non_empty_users
            .iter()
            .map(|r| r.recommended_items.iter().map(String::as_str).collect())
            .collect();
        let mut inter_pair_count = 0usize;
        let mut inter_dissimilarity_sum = 0.0;
        for i in 0..user_item_sets.len() {
            for j in (i + 1)..user_item_sets.len() {
                let union = user_item_sets[i].union(&user_item_sets[j]).count();
                if union == 0 {
                    continue;
                }
                let intersection = user_item_sets[i].intersection(&user_item_sets[j]).count();
                let jaccard = intersection as f64 / union as f64;
                inter_dissimilarity_sum += 1.0 - jaccard;
                inter_pair_count += 1;
            }
        }

        Ok(DiversityAnalysis {
            intra_list_diversity: Self::mean(&intra_scores),
            inter_user_diversity: if inter_pair_count > 0 {
                inter_dissimilarity_sum / inter_pair_count as f64
            } else {
                0.0
            },
            category_diversity: Self::mean(&category_scores),
            feature_diversity: Self::mean(&feature_scores),
        })
    }

    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    fn cosine_similarity_slices(a: &[f32], b: &[f32]) -> f64 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a * norm_b)) as f64
        } else {
            0.0
        }
    }

    /// Simulate user satisfaction scores
    fn simulate_user_satisfaction(
        &self,
        per_user_results: &HashMap<String, UserRecommendationResults>,
    ) -> Result<HashMap<String, f64>> {
        let mut satisfaction_scores = HashMap::new();

        for (user_id, results) in per_user_results {
            // Base satisfaction on precision and personalization
            let avg_precision = results.precision_scores.get(&5).copied().unwrap_or(0.0);
            let personalization = results.personalization_score;

            let satisfaction = (avg_precision * 0.7 + personalization * 0.3).clamp(0.0, 1.0);

            satisfaction_scores.insert(user_id.clone(), satisfaction);
        }

        Ok(satisfaction_scores)
    }
}

impl Default for RecommendationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: &str, category: &str, popularity: f64) -> ItemMetadata {
        ItemMetadata {
            item_id: id.to_string(),
            category: category.to_string(),
            features: HashMap::new(),
            popularity,
            embedding: None,
        }
    }

    fn make_user_result(recommended: &[&str], ground_truth: &[&str]) -> UserRecommendationResults {
        UserRecommendationResults {
            user_id: "u1".to_string(),
            precision_scores: HashMap::new(),
            recall_scores: HashMap::new(),
            ndcg_scores: HashMap::new(),
            personalization_score: 0.0,
            recommended_items: recommended.iter().map(|s| s.to_string()).collect(),
            ground_truth: ground_truth.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Regression test for the P3 finding: MAP/MRR must be computed for
    /// real from the ranked `recommended_items` list against
    /// `ground_truth`, instead of a hardcoded 0.5.
    #[test]
    fn test_calculate_metric_map_and_mrr_are_real() -> Result<()> {
        let evaluator = RecommendationEvaluator::new();
        let mut per_user = HashMap::new();
        per_user.insert(
            "u1".to_string(),
            make_user_result(&["i1", "i2", "i3"], &["i2"]),
        );

        // i2 (the only relevant item) is at rank 2: AP = (1/2) / 1 = 0.5;
        // RR = 1/2 = 0.5.
        let map = evaluator.calculate_metric(&RecommendationMetric::MAP, &per_user)?;
        assert!((map - 0.5).abs() < 1e-9, "map = {map}");

        let mrr = evaluator.calculate_metric(&RecommendationMetric::MRR, &per_user)?;
        assert!((mrr - 0.5).abs() < 1e-9, "mrr = {mrr}");

        Ok(())
    }

    /// Regression test: catalog coverage must be computed from the actual
    /// items recommended vs. the real catalog size, instead of the
    /// hardcoded (0.65, 450, 1000, 0.25) placeholder tuple.
    #[test]
    fn test_calculate_coverage_stats_reflects_real_catalog() {
        let mut evaluator = RecommendationEvaluator::new();
        evaluator.add_item(make_item("i1", "books", 0.9));
        evaluator.add_item(make_item("i2", "books", 0.1));
        evaluator.add_item(make_item("i3", "toys", 0.5));

        let mut per_user = HashMap::new();
        per_user.insert("u1".to_string(), make_user_result(&["i1"], &[]));

        let stats = evaluator
            .calculate_coverage_stats(&per_user)
            .expect("should succeed");
        assert_eq!(stats.total_catalog_items, 3);
        assert_eq!(stats.unique_items_recommended, 1);
        assert!(
            (stats.catalog_coverage - (1.0 / 3.0)).abs() < 1e-9,
            "catalog_coverage = {}",
            stats.catalog_coverage
        );
    }

    /// Regression test: novelty must be derived from the catalog's actual
    /// popularity scores instead of being a constant.
    #[test]
    fn test_novelty_for_items_uses_real_popularity() {
        let mut evaluator = RecommendationEvaluator::new();
        evaluator.add_item(make_item("popular", "x", 1.0));
        evaluator.add_item(make_item("obscure", "x", 0.0));

        let popular_novelty = evaluator.novelty_for_items(&["popular".to_string()]);
        let obscure_novelty = evaluator.novelty_for_items(&["obscure".to_string()]);

        assert!(
            (popular_novelty - 0.0).abs() < 1e-9,
            "popular_novelty = {popular_novelty}"
        );
        assert!(
            (obscure_novelty - 1.0).abs() < 1e-9,
            "obscure_novelty = {obscure_novelty}"
        );
    }

    /// Regression test: diversity analysis must genuinely differ between a
    /// single-category recommendation list and a multi-category one,
    /// instead of the hardcoded (0.7, 0.8, 0.6, 0.65) placeholder tuple.
    #[test]
    fn test_calculate_diversity_analysis_varies_with_categories() {
        let mut evaluator = RecommendationEvaluator::new();
        evaluator.add_item(make_item("i1", "books", 0.5));
        evaluator.add_item(make_item("i2", "books", 0.5));
        evaluator.add_item(make_item("i3", "toys", 0.5));

        let mut same_category = HashMap::new();
        same_category.insert("u1".to_string(), make_user_result(&["i1", "i2"], &[]));
        let same_category_diversity = evaluator
            .calculate_diversity_analysis(&same_category)
            .expect("should succeed");

        let mut mixed_category = HashMap::new();
        mixed_category.insert("u1".to_string(), make_user_result(&["i1", "i3"], &[]));
        let mixed_category_diversity = evaluator
            .calculate_diversity_analysis(&mixed_category)
            .expect("should succeed");

        assert_eq!(same_category_diversity.category_diversity, 0.5);
        assert_eq!(mixed_category_diversity.category_diversity, 1.0);
        assert!(
            mixed_category_diversity.intra_list_diversity
                > same_category_diversity.intra_list_diversity
        );
    }
}
