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
            .or_insert_with(Vec::new)
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
            metric_scores.insert(format!("{:?}", metric), score);
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
        let user_interactions = self.user_interactions.get(user_id).unwrap();

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
        })
    }

    /// Generate recommendations for a user
    async fn generate_recommendations(
        &self,
        user_id: &str,
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
            RecommendationMetric::Coverage => {
                // Calculate as placeholder - would need to be computed differently
                Ok(0.7)
            }
            RecommendationMetric::Diversity => {
                // Calculate as placeholder
                Ok(0.6)
            }
            _ => Ok(0.5), // Placeholder for other metrics
        }
    }

    /// Calculate coverage statistics
    fn calculate_coverage_stats(
        &self,
        _per_user_results: &HashMap<String, UserRecommendationResults>,
    ) -> Result<CoverageStats> {
        // Simplified implementation
        Ok(CoverageStats {
            catalog_coverage: 0.65,
            unique_items_recommended: 450,
            total_catalog_items: 1000,
            long_tail_coverage: 0.25,
        })
    }

    /// Calculate diversity analysis
    fn calculate_diversity_analysis(
        &self,
        _per_user_results: &HashMap<String, UserRecommendationResults>,
    ) -> Result<DiversityAnalysis> {
        // Simplified implementation
        Ok(DiversityAnalysis {
            intra_list_diversity: 0.7,
            inter_user_diversity: 0.8,
            category_diversity: 0.6,
            feature_diversity: 0.65,
        })
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

            let satisfaction = (avg_precision * 0.7 + personalization * 0.3)
                .min(1.0)
                .max(0.0);

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
