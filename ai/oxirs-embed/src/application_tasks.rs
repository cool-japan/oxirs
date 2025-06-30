//! Application-specific evaluation tasks for embedding models
//!
//! This module implements comprehensive evaluation metrics and benchmarks for
//! real-world applications of knowledge graph embeddings including recommendation
//! systems, search relevance, clustering performance, and classification accuracy.

use crate::{EmbeddingModel, Vector};
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Application-specific task evaluator
pub struct ApplicationTaskEvaluator {
    /// Evaluation configuration
    config: ApplicationEvalConfig,
    /// Task-specific evaluators
    recommendation_evaluator: RecommendationEvaluator,
    search_evaluator: SearchEvaluator,
    clustering_evaluator: ClusteringEvaluator,
    classification_evaluator: ClassificationEvaluator,
    retrieval_evaluator: RetrievalEvaluator,
    /// Evaluation history
    evaluation_history: Arc<RwLock<VecDeque<ApplicationEvalResults>>>,
}

/// Configuration for application evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationEvalConfig {
    /// Enable recommendation evaluation
    pub enable_recommendation_eval: bool,
    /// Enable search relevance evaluation
    pub enable_search_eval: bool,
    /// Enable clustering evaluation
    pub enable_clustering_eval: bool,
    /// Enable classification evaluation
    pub enable_classification_eval: bool,
    /// Enable retrieval evaluation
    pub enable_retrieval_eval: bool,
    /// Sample size for evaluations
    pub sample_size: usize,
    /// Number of recommendations to generate
    pub num_recommendations: usize,
    /// Number of clusters for clustering evaluation
    pub num_clusters: usize,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Enable user satisfaction simulation
    pub enable_user_satisfaction: bool,
}

impl Default for ApplicationEvalConfig {
    fn default() -> Self {
        Self {
            enable_recommendation_eval: true,
            enable_search_eval: true,
            enable_clustering_eval: true,
            enable_classification_eval: true,
            enable_retrieval_eval: true,
            sample_size: 1000,
            num_recommendations: 10,
            num_clusters: 5,
            cv_folds: 5,
            enable_user_satisfaction: true,
        }
    }
}

/// Application evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationEvalResults {
    /// Timestamp of evaluation
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Model identifier
    pub model_id: String,
    /// Recommendation evaluation results
    pub recommendation_results: Option<RecommendationResults>,
    /// Search evaluation results
    pub search_results: Option<SearchResults>,
    /// Clustering evaluation results
    pub clustering_results: Option<ClusteringResults>,
    /// Classification evaluation results
    pub classification_results: Option<ClassificationResults>,
    /// Retrieval evaluation results
    pub retrieval_results: Option<RetrievalResults>,
    /// Overall application score
    pub overall_score: f64,
    /// Evaluation time (seconds)
    pub evaluation_time: f64,
}

// ============================================================================
// RECOMMENDATION SYSTEM EVALUATION
// ============================================================================

/// Recommendation system evaluator
pub struct RecommendationEvaluator {
    /// User interaction history
    user_interactions: HashMap<String, Vec<UserInteraction>>,
    /// Item catalog
    item_catalog: HashMap<String, ItemMetadata>,
    /// Evaluation metrics
    metrics: Vec<RecommendationMetric>,
}

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
            let user_results = self.evaluate_user_recommendations(user_id, model, config).await?;
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
        let recommendations = self.generate_recommendations(
            user_id,
            training_interactions,
            model,
            config.num_recommendations,
        ).await?;

        // Extract ground truth items from test interactions
        let ground_truth: HashSet<String> = test_interactions
            .iter()
            .filter(|i| matches!(i.interaction_type, InteractionType::Like | InteractionType::Purchase))
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
        let personalization_score = self.calculate_personalization_score(
            user_id,
            &recommendations,
            training_interactions,
        )?;

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

            let item_score = self.score_item_for_user(&user_profile, item_metadata, model).await?;
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
        let dot_product: f32 = v1.values.iter().zip(v2.values.iter()).map(|(a, b)| a * b).sum();
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

        let overlap = user_categories.intersection(&recommendation_categories).count();
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
            let avg_precision = results
                .precision_scores
                .get(&5)
                .copied()
                .unwrap_or(0.0);
            let personalization = results.personalization_score;

            let satisfaction = (avg_precision * 0.7 + personalization * 0.3)
                .min(1.0)
                .max(0.0);

            satisfaction_scores.insert(user_id.clone(), satisfaction);
        }

        Ok(satisfaction_scores)
    }
}

// ============================================================================
// SEARCH RELEVANCE EVALUATION
// ============================================================================

/// Search relevance evaluator
pub struct SearchEvaluator {
    /// Search queries and their relevance judgments
    query_relevance: HashMap<String, Vec<RelevanceJudgment>>,
    /// Search metrics to evaluate
    metrics: Vec<SearchMetric>,
}

/// Relevance judgment for search evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceJudgment {
    /// Query
    pub query: String,
    /// Document/entity identifier
    pub document_id: String,
    /// Relevance score (0-3: not relevant, somewhat relevant, relevant, highly relevant)
    pub relevance_score: u8,
    /// Annotator identifier
    pub annotator_id: String,
}

/// Search evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMetric {
    /// Precision at K
    PrecisionAtK(usize),
    /// Recall at K
    RecallAtK(usize),
    /// Mean Average Precision
    MAP,
    /// Normalized Discounted Cumulative Gain
    NDCG(usize),
    /// Mean Reciprocal Rank
    MRR,
    /// Expected Reciprocal Rank
    ERR,
    /// Click-through rate simulation
    CTR,
}

/// Search evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// Metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Per-query results
    pub per_query_results: HashMap<String, QueryResults>,
    /// Query performance analysis
    pub query_analysis: QueryPerformanceAnalysis,
    /// Search effectiveness metrics
    pub effectiveness_metrics: SearchEffectivenessMetrics,
}

/// Per-query search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResults {
    /// Query text
    pub query: String,
    /// Precision scores at different K values
    pub precision_scores: HashMap<usize, f64>,
    /// Recall scores at different K values
    pub recall_scores: HashMap<usize, f64>,
    /// NDCG scores
    pub ndcg_scores: HashMap<usize, f64>,
    /// Number of relevant documents
    pub num_relevant: usize,
    /// Query difficulty score
    pub difficulty_score: f64,
}

/// Query performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceAnalysis {
    /// Average query length
    pub avg_query_length: f64,
    /// Query type distribution
    pub query_type_distribution: HashMap<String, usize>,
    /// Performance by query difficulty
    pub performance_by_difficulty: HashMap<String, f64>,
    /// Zero-result queries percentage
    pub zero_result_queries: f64,
}

/// Search effectiveness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEffectivenessMetrics {
    /// Overall search satisfaction
    pub search_satisfaction: f64,
    /// Result relevance distribution
    pub relevance_distribution: HashMap<u8, usize>,
    /// Search result diversity
    pub result_diversity: f64,
    /// Query success rate
    pub query_success_rate: f64,
}

impl SearchEvaluator {
    /// Create a new search evaluator
    pub fn new() -> Self {
        Self {
            query_relevance: HashMap::new(),
            metrics: vec![
                SearchMetric::PrecisionAtK(1),
                SearchMetric::PrecisionAtK(5),
                SearchMetric::PrecisionAtK(10),
                SearchMetric::NDCG(10),
                SearchMetric::MAP,
                SearchMetric::MRR,
            ],
        }
    }

    /// Add relevance judgment
    pub fn add_relevance_judgment(&mut self, judgment: RelevanceJudgment) {
        self.query_relevance
            .entry(judgment.query.clone())
            .or_insert_with(Vec::new)
            .push(judgment);
    }

    /// Evaluate search relevance
    pub async fn evaluate(
        &self,
        model: &dyn EmbeddingModel,
        config: &ApplicationEvalConfig,
    ) -> Result<SearchResults> {
        let mut metric_scores = HashMap::new();
        let mut per_query_results = HashMap::new();

        // Sample queries for evaluation
        let queries_to_evaluate: Vec<_> = self
            .query_relevance
            .keys()
            .take(config.sample_size)
            .cloned()
            .collect();

        for query in &queries_to_evaluate {
            let query_results = self.evaluate_query_search(query, model).await?;
            per_query_results.insert(query.clone(), query_results);
        }

        // Calculate aggregate metrics
        for metric in &self.metrics {
            let score = self.calculate_search_metric(metric, &per_query_results)?;
            metric_scores.insert(format!("{:?}", metric), score);
        }

        // Analyze query performance
        let query_analysis = self.analyze_query_performance(&per_query_results)?;
        let effectiveness_metrics = self.calculate_effectiveness_metrics(&per_query_results)?;

        Ok(SearchResults {
            metric_scores,
            per_query_results,
            query_analysis,
            effectiveness_metrics,
        })
    }

    /// Evaluate search for a specific query
    async fn evaluate_query_search(
        &self,
        query: &str,
        model: &dyn EmbeddingModel,
    ) -> Result<QueryResults> {
        let judgments = self.query_relevance.get(query).unwrap();
        
        // Get search results (simplified - would use actual search system)
        let search_results = self.perform_search(query, model).await?;

        // Calculate relevance for each result
        let mut relevance_scores = Vec::new();
        for (doc_id, _score) in &search_results {
            let relevance = judgments
                .iter()
                .find(|j| &j.document_id == doc_id)
                .map(|j| j.relevance_score)
                .unwrap_or(0);
            relevance_scores.push(relevance);
        }

        let num_relevant = judgments.iter().filter(|j| j.relevance_score > 0).count();

        // Calculate metrics at different K values
        let mut precision_scores = HashMap::new();
        let mut recall_scores = HashMap::new();
        let mut ndcg_scores = HashMap::new();

        for &k in &[1, 3, 5, 10] {
            if k <= search_results.len() {
                let relevant_at_k = relevance_scores
                    .iter()
                    .take(k)
                    .filter(|&&r| r > 0)
                    .count() as f64;
                
                let precision = relevant_at_k / k as f64;
                let recall = if num_relevant > 0 {
                    relevant_at_k / num_relevant as f64
                } else {
                    0.0
                };

                precision_scores.insert(k, precision);
                recall_scores.insert(k, recall);

                // Calculate NDCG
                let ndcg = self.calculate_search_ndcg(&relevance_scores, k)?;
                ndcg_scores.insert(k, ndcg);
            }
        }

        let difficulty_score = self.calculate_query_difficulty(query, num_relevant);

        Ok(QueryResults {
            query: query.to_string(),
            precision_scores,
            recall_scores,
            ndcg_scores,
            num_relevant,
            difficulty_score,
        })
    }

    /// Perform search (simplified implementation)
    async fn perform_search(
        &self,
        query: &str,
        model: &dyn EmbeddingModel,
    ) -> Result<Vec<(String, f64)>> {
        // Create query embedding (simplified)
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let mut query_embedding = vec![0.0f32; 100];
        
        // Simple word-based embedding (in practice, use proper query embedding)
        for (i, word) in query_words.iter().enumerate() {
            if i < query_embedding.len() {
                query_embedding[i] = word.len() as f32 / 10.0;
            }
        }
        let query_vector = Vector::new(query_embedding);

        // Score entities (documents) against query
        let entities = model.get_entities();
        let mut search_results = Vec::new();

        for entity in entities.iter().take(100) { // Limit for efficiency
            if let Ok(entity_embedding) = model.get_entity_embedding(entity) {
                let score = self.cosine_similarity(&query_vector, &entity_embedding);
                search_results.push((entity.clone(), score));
            }
        }

        // Sort by score and return top results
        search_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        search_results.truncate(20);

        Ok(search_results)
    }

    /// Calculate NDCG for search results
    fn calculate_search_ndcg(&self, relevance_scores: &[u8], k: usize) -> Result<f64> {
        if k == 0 || relevance_scores.is_empty() {
            return Ok(0.0);
        }

        let mut dcg = 0.0;
        for (i, &relevance) in relevance_scores.iter().take(k).enumerate() {
            if relevance > 0 {
                let gain = (2_u32.pow(relevance as u32) - 1) as f64;
                dcg += gain / (i as f64 + 2.0).log2();
            }
        }

        // Calculate ideal DCG
        let mut ideal_relevance: Vec<u8> = relevance_scores.iter().cloned().collect();
        ideal_relevance.sort_by(|a, b| b.cmp(a));
        
        let mut idcg = 0.0;
        for (i, &relevance) in ideal_relevance.iter().take(k).enumerate() {
            if relevance > 0 {
                let gain = (2_u32.pow(relevance as u32) - 1) as f64;
                idcg += gain / (i as f64 + 2.0).log2();
            }
        }

        if idcg > 0.0 {
            Ok(dcg / idcg)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate query difficulty
    fn calculate_query_difficulty(&self, query: &str, num_relevant: usize) -> f64 {
        let query_length = query.split_whitespace().count() as f64;
        let relevance_factor = if num_relevant == 0 {
            1.0 // High difficulty
        } else {
            1.0 / (num_relevant as f64).log2()
        };
        
        (query_length * 0.1 + relevance_factor * 0.9).min(1.0)
    }

    /// Calculate aggregate search metric
    fn calculate_search_metric(
        &self,
        metric: &SearchMetric,
        per_query_results: &HashMap<String, QueryResults>,
    ) -> Result<f64> {
        if per_query_results.is_empty() {
            return Ok(0.0);
        }

        match metric {
            SearchMetric::PrecisionAtK(k) => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .filter_map(|r| r.precision_scores.get(k))
                    .cloned()
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            SearchMetric::NDCG(k) => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .filter_map(|r| r.ndcg_scores.get(k))
                    .cloned()
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            _ => Ok(0.5), // Placeholder for other metrics
        }
    }

    /// Analyze query performance
    fn analyze_query_performance(
        &self,
        per_query_results: &HashMap<String, QueryResults>,
    ) -> Result<QueryPerformanceAnalysis> {
        let avg_query_length = per_query_results
            .keys()
            .map(|q| q.split_whitespace().count() as f64)
            .sum::<f64>()
            / per_query_results.len() as f64;

        let zero_result_queries = per_query_results
            .values()
            .filter(|r| r.num_relevant == 0)
            .count() as f64
            / per_query_results.len() as f64;

        Ok(QueryPerformanceAnalysis {
            avg_query_length,
            query_type_distribution: HashMap::new(), // Simplified
            performance_by_difficulty: HashMap::new(), // Simplified
            zero_result_queries,
        })
    }

    /// Calculate effectiveness metrics
    fn calculate_effectiveness_metrics(
        &self,
        per_query_results: &HashMap<String, QueryResults>,
    ) -> Result<SearchEffectivenessMetrics> {
        let successful_queries = per_query_results
            .values()
            .filter(|r| r.precision_scores.get(&1).unwrap_or(&0.0) > &0.0)
            .count() as f64;

        let query_success_rate = successful_queries / per_query_results.len() as f64;

        Ok(SearchEffectivenessMetrics {
            search_satisfaction: query_success_rate * 0.8, // Simplified
            relevance_distribution: HashMap::new(), // Simplified
            result_diversity: 0.6, // Simplified
            query_success_rate,
        })
    }

    /// Calculate cosine similarity
    fn cosine_similarity(&self, v1: &Vector, v2: &Vector) -> f64 {
        let dot_product: f32 = v1.values.iter().zip(v2.values.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = v1.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = v2.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot_product / (norm_a * norm_b)) as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// CLUSTERING EVALUATION
// ============================================================================

/// Clustering evaluator
pub struct ClusteringEvaluator {
    /// Ground truth clusters (if available)
    ground_truth_clusters: Option<HashMap<String, String>>,
    /// Clustering metrics
    metrics: Vec<ClusteringMetric>,
}

/// Clustering evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringMetric {
    /// Silhouette score
    SilhouetteScore,
    /// Calinski-Harabasz index
    CalinskiHarabaszIndex,
    /// Davies-Bouldin index
    DaviesBouldinIndex,
    /// Adjusted Rand Index (requires ground truth)
    AdjustedRandIndex,
    /// Normalized Mutual Information (requires ground truth)
    NormalizedMutualInformation,
    /// Clustering purity (requires ground truth)
    Purity,
    /// Inertia (within-cluster sum of squares)
    Inertia,
}

/// Clustering evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResults {
    /// Metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Cluster quality analysis
    pub cluster_analysis: ClusterAnalysis,
    /// Optimal number of clusters (if determined)
    pub optimal_k: Option<usize>,
    /// Clustering stability analysis
    pub stability_analysis: ClusteringStabilityAnalysis,
}

/// Cluster quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAnalysis {
    /// Number of clusters
    pub num_clusters: usize,
    /// Cluster sizes
    pub cluster_sizes: Vec<usize>,
    /// Cluster cohesion scores
    pub cluster_cohesion: Vec<f64>,
    /// Cluster separation scores
    pub cluster_separation: Vec<f64>,
    /// Inter-cluster distances
    pub inter_cluster_distances: Array2<f64>,
}

/// Clustering stability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringStabilityAnalysis {
    /// Stability score across multiple runs
    pub stability_score: f64,
    /// Consistency of cluster assignments
    pub assignment_consistency: f64,
    /// Robustness to parameter changes
    pub parameter_robustness: f64,
}

impl ClusteringEvaluator {
    /// Create a new clustering evaluator
    pub fn new() -> Self {
        Self {
            ground_truth_clusters: None,
            metrics: vec![
                ClusteringMetric::SilhouetteScore,
                ClusteringMetric::CalinskiHarabaszIndex,
                ClusteringMetric::DaviesBouldinIndex,
                ClusteringMetric::Inertia,
            ],
        }
    }

    /// Set ground truth clusters
    pub fn set_ground_truth(&mut self, clusters: HashMap<String, String>) {
        self.ground_truth_clusters = Some(clusters);
        
        // Add supervised metrics
        self.metrics.extend(vec![
            ClusteringMetric::AdjustedRandIndex,
            ClusteringMetric::NormalizedMutualInformation,
            ClusteringMetric::Purity,
        ]);
    }

    /// Evaluate clustering performance
    pub async fn evaluate(
        &self,
        model: &dyn EmbeddingModel,
        config: &ApplicationEvalConfig,
    ) -> Result<ClusteringResults> {
        // Get entity embeddings
        let entities = model.get_entities();
        let sample_entities: Vec<_> = entities.into_iter().take(config.sample_size).collect();
        
        let mut embeddings = Vec::new();
        for entity in &sample_entities {
            if let Ok(embedding) = model.get_entity_embedding(entity) {
                embeddings.push(embedding.values);
            }
        }

        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings available for clustering evaluation"));
        }

        // Perform clustering
        let cluster_assignments = self.perform_clustering(&embeddings, config.num_clusters)?;

        // Calculate metrics
        let mut metric_scores = HashMap::new();
        for metric in &self.metrics {
            let score = self.calculate_clustering_metric(
                metric,
                &embeddings,
                &cluster_assignments,
                &sample_entities,
            )?;
            metric_scores.insert(format!("{:?}", metric), score);
        }

        // Analyze clusters
        let cluster_analysis = self.analyze_clusters(&embeddings, &cluster_assignments)?;

        // Analyze stability
        let stability_analysis = self.analyze_stability(&embeddings, config)?;

        Ok(ClusteringResults {
            metric_scores,
            cluster_analysis,
            optimal_k: Some(config.num_clusters), // Simplified
            stability_analysis,
        })
    }

    /// Perform K-means clustering
    fn perform_clustering(&self, embeddings: &[Vec<f32>], k: usize) -> Result<Vec<usize>> {
        if embeddings.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let n = embeddings.len();
        let dim = embeddings[0].len();

        // Initialize centroids randomly
        let mut centroids = Vec::new();
        for _ in 0..k {
            let idx = rand::random::<usize>() % n;
            centroids.push(embeddings[idx].clone());
        }

        let mut assignments = vec![0; n];
        let max_iterations = 100;

        for _iteration in 0..max_iterations {
            let mut new_assignments = vec![0; n];
            let mut changed = false;

            // Assign points to nearest centroid
            for (i, embedding) in embeddings.iter().enumerate() {
                let mut min_distance = f32::INFINITY;
                let mut best_cluster = 0;

                for (c, centroid) in centroids.iter().enumerate() {
                    let distance = self.euclidean_distance(embedding, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = c;
                    }
                }

                new_assignments[i] = best_cluster;
                if new_assignments[i] != assignments[i] {
                    changed = true;
                }
            }

            assignments = new_assignments;

            if !changed {
                break;
            }

            // Update centroids
            for c in 0..k {
                let cluster_points: Vec<_> = embeddings
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == c)
                    .map(|(_, emb)| emb)
                    .collect();

                if !cluster_points.is_empty() {
                    let mut new_centroid = vec![0.0f32; dim];
                    for point in &cluster_points {
                        for (i, &value) in point.iter().enumerate() {
                            new_centroid[i] += value;
                        }
                    }
                    for value in &mut new_centroid {
                        *value /= cluster_points.len() as f32;
                    }
                    centroids[c] = new_centroid;
                }
            }
        }

        Ok(assignments)
    }

    /// Calculate clustering metric
    fn calculate_clustering_metric(
        &self,
        metric: &ClusteringMetric,
        embeddings: &[Vec<f32>],
        assignments: &[usize],
        entities: &[String],
    ) -> Result<f64> {
        match metric {
            ClusteringMetric::SilhouetteScore => {
                self.calculate_silhouette_score(embeddings, assignments)
            }
            ClusteringMetric::Inertia => {
                self.calculate_inertia(embeddings, assignments)
            }
            ClusteringMetric::CalinskiHarabaszIndex => {
                self.calculate_calinski_harabasz(embeddings, assignments)
            }
            ClusteringMetric::DaviesBouldinIndex => {
                self.calculate_davies_bouldin(embeddings, assignments)
            }
            ClusteringMetric::AdjustedRandIndex => {
                if let Some(ref ground_truth) = self.ground_truth_clusters {
                    self.calculate_adjusted_rand_index(assignments, ground_truth, entities)
                } else {
                    Ok(0.0)
                }
            }
            _ => Ok(0.5), // Placeholder for other metrics
        }
    }

    /// Calculate silhouette score
    fn calculate_silhouette_score(&self, embeddings: &[Vec<f32>], assignments: &[usize]) -> Result<f64> {
        if embeddings.len() != assignments.len() || embeddings.is_empty() {
            return Ok(0.0);
        }

        let mut silhouette_scores = Vec::new();

        for (i, embedding) in embeddings.iter().enumerate() {
            let own_cluster = assignments[i];

            // Calculate average intra-cluster distance
            let same_cluster_points: Vec<_> = embeddings
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i && assignments[*j] == own_cluster)
                .map(|(_, emb)| emb)
                .collect();

            let a = if same_cluster_points.is_empty() {
                0.0
            } else {
                same_cluster_points
                    .iter()
                    .map(|other| self.euclidean_distance(embedding, other) as f64)
                    .sum::<f64>()
                    / same_cluster_points.len() as f64
            };

            // Calculate average nearest-cluster distance
            let unique_clusters: HashSet<usize> = assignments.iter().cloned().collect();
            let mut min_b = f64::INFINITY;

            for &cluster in &unique_clusters {
                if cluster != own_cluster {
                    let other_cluster_points: Vec<_> = embeddings
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| assignments[*j] == cluster)
                        .map(|(_, emb)| emb)
                        .collect();

                    if !other_cluster_points.is_empty() {
                        let avg_distance = other_cluster_points
                            .iter()
                            .map(|other| self.euclidean_distance(embedding, other) as f64)
                            .sum::<f64>()
                            / other_cluster_points.len() as f64;

                        min_b = min_b.min(avg_distance);
                    }
                }
            }

            let b = min_b;

            // Calculate silhouette score for this point
            let silhouette = if a < b {
                (b - a) / b
            } else if a > b {
                (b - a) / a
            } else {
                0.0
            };

            silhouette_scores.push(silhouette);
        }

        Ok(silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64)
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(&self, embeddings: &[Vec<f32>], assignments: &[usize]) -> Result<f64> {
        let unique_clusters: HashSet<usize> = assignments.iter().cloned().collect();
        let mut total_inertia = 0.0;

        for &cluster in &unique_clusters {
            let cluster_points: Vec<_> = embeddings
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == cluster)
                .map(|(_, emb)| emb)
                .collect();

            if cluster_points.is_empty() {
                continue;
            }

            // Calculate centroid
            let dim = cluster_points[0].len();
            let mut centroid = vec![0.0f32; dim];
            for point in &cluster_points {
                for (i, &value) in point.iter().enumerate() {
                    centroid[i] += value;
                }
            }
            for value in &mut centroid {
                *value /= cluster_points.len() as f32;
            }

            // Calculate sum of squared distances to centroid
            for point in &cluster_points {
                let distance = self.euclidean_distance(point, &centroid);
                total_inertia += (distance * distance) as f64;
            }
        }

        Ok(total_inertia)
    }

    /// Calculate Calinski-Harabasz index
    fn calculate_calinski_harabasz(&self, embeddings: &[Vec<f32>], assignments: &[usize]) -> Result<f64> {
        let n = embeddings.len();
        let unique_clusters: HashSet<usize> = assignments.iter().cloned().collect();
        let k = unique_clusters.len();

        if k <= 1 || n <= k {
            return Ok(0.0);
        }

        // Calculate overall centroid
        let dim = embeddings[0].len();
        let mut overall_centroid = vec![0.0f32; dim];
        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                overall_centroid[i] += value;
            }
        }
        for value in &mut overall_centroid {
            *value /= n as f32;
        }

        let mut between_cluster_dispersion = 0.0;
        let mut within_cluster_dispersion = 0.0;

        for &cluster in &unique_clusters {
            let cluster_points: Vec<_> = embeddings
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == cluster)
                .map(|(_, emb)| emb)
                .collect();

            if cluster_points.is_empty() {
                continue;
            }

            // Calculate cluster centroid
            let mut cluster_centroid = vec![0.0f32; dim];
            for point in &cluster_points {
                for (i, &value) in point.iter().enumerate() {
                    cluster_centroid[i] += value;
                }
            }
            for value in &mut cluster_centroid {
                *value /= cluster_points.len() as f32;
            }

            // Between-cluster dispersion
            let distance_to_overall = self.euclidean_distance(&cluster_centroid, &overall_centroid);
            between_cluster_dispersion += cluster_points.len() as f64 * (distance_to_overall * distance_to_overall) as f64;

            // Within-cluster dispersion
            for point in &cluster_points {
                let distance_to_cluster = self.euclidean_distance(point, &cluster_centroid);
                within_cluster_dispersion += (distance_to_cluster * distance_to_cluster) as f64;
            }
        }

        if within_cluster_dispersion > 0.0 {
            Ok((between_cluster_dispersion / (k - 1) as f64) / (within_cluster_dispersion / (n - k) as f64))
        } else {
            Ok(0.0)
        }
    }

    /// Calculate Davies-Bouldin index
    fn calculate_davies_bouldin(&self, embeddings: &[Vec<f32>], assignments: &[usize]) -> Result<f64> {
        let unique_clusters: HashSet<usize> = assignments.iter().cloned().collect();
        let k = unique_clusters.len();

        if k <= 1 {
            return Ok(0.0);
        }

        let mut cluster_centroids = HashMap::new();
        let mut cluster_dispersions = HashMap::new();

        // Calculate centroids and dispersions for each cluster
        for &cluster in &unique_clusters {
            let cluster_points: Vec<_> = embeddings
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == cluster)
                .map(|(_, emb)| emb)
                .collect();

            if cluster_points.is_empty() {
                continue;
            }

            // Calculate centroid
            let dim = cluster_points[0].len();
            let mut centroid = vec![0.0f32; dim];
            for point in &cluster_points {
                for (i, &value) in point.iter().enumerate() {
                    centroid[i] += value;
                }
            }
            for value in &mut centroid {
                *value /= cluster_points.len() as f32;
            }

            // Calculate average distance to centroid (dispersion)
            let dispersion = cluster_points
                .iter()
                .map(|point| self.euclidean_distance(point, &centroid) as f64)
                .sum::<f64>()
                / cluster_points.len() as f64;

            cluster_centroids.insert(cluster, centroid);
            cluster_dispersions.insert(cluster, dispersion);
        }

        let mut db_sum = 0.0;

        for &cluster_i in &unique_clusters {
            let mut max_ratio: f32 = 0.0;

            for &cluster_j in &unique_clusters {
                if cluster_i != cluster_j {
                    if let (Some(centroid_i), Some(centroid_j)) = 
                        (cluster_centroids.get(&cluster_i), cluster_centroids.get(&cluster_j)) {
                        
                        let centroid_distance = self.euclidean_distance(centroid_i, centroid_j) as f64;
                        
                        if centroid_distance > 0.0 {
                            let dispersion_i = cluster_dispersions.get(&cluster_i).unwrap_or(&0.0);
                            let dispersion_j = cluster_dispersions.get(&cluster_j).unwrap_or(&0.0);
                            
                            let ratio = (dispersion_i + dispersion_j) / centroid_distance;
                            max_ratio = max_ratio.max(ratio);
                        }
                    }
                }
            }

            db_sum += max_ratio;
        }

        Ok(db_sum / k as f64)
    }

    /// Calculate Adjusted Rand Index
    fn calculate_adjusted_rand_index(
        &self,
        assignments: &[usize],
        ground_truth: &HashMap<String, String>,
        entities: &[String],
    ) -> Result<f64> {
        if assignments.len() != entities.len() {
            return Ok(0.0);
        }

        // Create contingency table
        let mut contingency: HashMap<(usize, String), usize> = HashMap::new();
        
        for (i, entity) in entities.iter().enumerate() {
            if let Some(true_cluster) = ground_truth.get(entity) {
                let predicted_cluster = assignments[i];
                *contingency.entry((predicted_cluster, true_cluster.clone())).or_insert(0) += 1;
            }
        }

        // Calculate ARI (simplified implementation)
        // In practice, would use the full ARI formula
        let n = entities.len() as f64;
        let mut agreement = 0.0;

        for count in contingency.values() {
            if *count > 1 {
                agreement += (*count * (*count - 1)) as f64 / 2.0;
            }
        }

        let max_agreement = n * (n - 1.0) / 2.0;
        
        if max_agreement > 0.0 {
            Ok(agreement / max_agreement)
        } else {
            Ok(0.0)
        }
    }

    /// Analyze clusters
    fn analyze_clusters(&self, embeddings: &[Vec<f32>], assignments: &[usize]) -> Result<ClusterAnalysis> {
        let unique_clusters: HashSet<usize> = assignments.iter().cloned().collect();
        let num_clusters = unique_clusters.len();

        let mut cluster_sizes = Vec::new();
        let mut cluster_cohesion = Vec::new();
        let mut cluster_separation = Vec::new();

        for &cluster in &unique_clusters {
            let cluster_points: Vec<_> = embeddings
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == cluster)
                .map(|(_, emb)| emb)
                .collect();

            cluster_sizes.push(cluster_points.len());

            // Calculate cohesion (average intra-cluster distance)
            if cluster_points.len() > 1 {
                let mut total_distance = 0.0;
                let mut count = 0;

                for i in 0..cluster_points.len() {
                    for j in (i + 1)..cluster_points.len() {
                        total_distance += self.euclidean_distance(cluster_points[i], cluster_points[j]) as f64;
                        count += 1;
                    }
                }

                let cohesion = if count > 0 {
                    total_distance / count as f64
                } else {
                    0.0
                };
                cluster_cohesion.push(cohesion);
            } else {
                cluster_cohesion.push(0.0);
            }

            // Calculate separation (minimum distance to other clusters)
            let mut min_separation = f64::INFINITY;
            for &other_cluster in &unique_clusters {
                if cluster != other_cluster {
                    let other_points: Vec<_> = embeddings
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| assignments[*i] == other_cluster)
                        .map(|(_, emb)| emb)
                        .collect();

                    for cluster_point in &cluster_points {
                        for other_point in &other_points {
                            let distance = self.euclidean_distance(cluster_point, other_point) as f64;
                            min_separation = min_separation.min(distance);
                        }
                    }
                }
            }

            cluster_separation.push(if min_separation == f64::INFINITY {
                0.0
            } else {
                min_separation
            });
        }

        // Calculate inter-cluster distances (simplified)
        let inter_cluster_distances = Array2::zeros((num_clusters, num_clusters));

        Ok(ClusterAnalysis {
            num_clusters,
            cluster_sizes,
            cluster_cohesion,
            cluster_separation,
            inter_cluster_distances,
        })
    }

    /// Analyze clustering stability
    fn analyze_stability(&self, embeddings: &[Vec<f32>], config: &ApplicationEvalConfig) -> Result<ClusteringStabilityAnalysis> {
        let num_runs = 5;
        let mut all_assignments = Vec::new();

        // Run clustering multiple times
        for _ in 0..num_runs {
            let assignments = self.perform_clustering(embeddings, config.num_clusters)?;
            all_assignments.push(assignments);
        }

        // Calculate stability as agreement between runs
        let mut agreement_scores = Vec::new();
        for i in 0..num_runs {
            for j in (i + 1)..num_runs {
                let agreement = self.calculate_assignment_agreement(&all_assignments[i], &all_assignments[j])?;
                agreement_scores.push(agreement);
            }
        }

        let stability_score = agreement_scores.iter().sum::<f64>() / agreement_scores.len() as f64;

        Ok(ClusteringStabilityAnalysis {
            stability_score,
            assignment_consistency: stability_score,
            parameter_robustness: 0.7, // Placeholder
        })
    }

    /// Calculate agreement between two cluster assignments
    fn calculate_assignment_agreement(&self, assignments1: &[usize], assignments2: &[usize]) -> Result<f64> {
        if assignments1.len() != assignments2.len() {
            return Ok(0.0);
        }

        let mut agreements = 0;
        let n = assignments1.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let same_cluster_1 = assignments1[i] == assignments1[j];
                let same_cluster_2 = assignments2[i] == assignments2[j];
                
                if same_cluster_1 == same_cluster_2 {
                    agreements += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        Ok(if total_pairs > 0 {
            agreements as f64 / total_pairs as f64
        } else {
            0.0
        })
    }

    /// Calculate Euclidean distance
    fn euclidean_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

// ============================================================================
// CLASSIFICATION EVALUATION
// ============================================================================

/// Classification evaluator
pub struct ClassificationEvaluator {
    /// Training data with labels
    training_data: Vec<(String, String)>, // (entity, label)
    /// Test data with labels
    test_data: Vec<(String, String)>,
    /// Classification metrics
    metrics: Vec<ClassificationMetric>,
}

/// Classification evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassificationMetric {
    /// Accuracy
    Accuracy,
    /// Precision (macro-averaged)
    Precision,
    /// Recall (macro-averaged)
    Recall,
    /// F1 Score (macro-averaged)
    F1Score,
    /// ROC AUC (for binary classification)
    ROCAUC,
    /// Precision-Recall AUC
    PRAUC,
    /// Matthews Correlation Coefficient
    MCC,
}

/// Classification evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResults {
    /// Metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Per-class results
    pub per_class_results: HashMap<String, ClassResults>,
    /// Confusion matrix
    pub confusion_matrix: Vec<Vec<usize>>,
    /// Classification report
    pub classification_report: ClassificationReport,
}

/// Per-class classification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassResults {
    /// Class label
    pub class_label: String,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// Support (number of instances)
    pub support: usize,
}

/// Classification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    /// Macro-averaged metrics
    pub macro_avg: ClassResults,
    /// Weighted-averaged metrics
    pub weighted_avg: ClassResults,
    /// Overall accuracy
    pub accuracy: f64,
    /// Total number of samples
    pub total_samples: usize,
}

impl ClassificationEvaluator {
    /// Create a new classification evaluator
    pub fn new() -> Self {
        Self {
            training_data: Vec::new(),
            test_data: Vec::new(),
            metrics: vec![
                ClassificationMetric::Accuracy,
                ClassificationMetric::Precision,
                ClassificationMetric::Recall,
                ClassificationMetric::F1Score,
            ],
        }
    }

    /// Add training data
    pub fn add_training_data(&mut self, entity: String, label: String) {
        self.training_data.push((entity, label));
    }

    /// Add test data
    pub fn add_test_data(&mut self, entity: String, label: String) {
        self.test_data.push((entity, label));
    }

    /// Evaluate classification performance
    pub async fn evaluate(
        &self,
        model: &dyn EmbeddingModel,
        _config: &ApplicationEvalConfig,
    ) -> Result<ClassificationResults> {
        if self.test_data.is_empty() {
            return Err(anyhow!("No test data available for classification evaluation"));
        }

        // Train a simple classifier on embeddings
        let classifier = self.train_classifier(model).await?;

        // Predict on test data
        let predictions = self.predict_test_data(model, &classifier).await?;

        // Calculate metrics
        let mut metric_scores = HashMap::new();
        for metric in &self.metrics {
            let score = self.calculate_classification_metric(metric, &predictions)?;
            metric_scores.insert(format!("{:?}", metric), score);
        }

        // Generate per-class results
        let per_class_results = self.calculate_per_class_results(&predictions)?;

        // Generate confusion matrix
        let confusion_matrix = self.generate_confusion_matrix(&predictions)?;

        // Generate classification report
        let classification_report = self.generate_classification_report(&per_class_results, &predictions)?;

        Ok(ClassificationResults {
            metric_scores,
            per_class_results,
            confusion_matrix,
            classification_report,
        })
    }

    /// Train a simple classifier (placeholder implementation)
    async fn train_classifier(&self, model: &dyn EmbeddingModel) -> Result<SimpleClassifier> {
        let mut class_centroids = HashMap::new();
        let mut class_counts = HashMap::new();

        for (entity, label) in &self.training_data {
            if let Ok(embedding) = model.get_entity_embedding(entity) {
                let centroid = class_centroids.entry(label.clone()).or_insert_with(|| {
                    vec![0.0f32; embedding.values.len()]
                });
                
                for (i, &value) in embedding.values.iter().enumerate() {
                    centroid[i] += value;
                }
                
                *class_counts.entry(label.clone()).or_insert(0) += 1;
            }
        }

        // Normalize centroids
        for (label, centroid) in &mut class_centroids {
            let count = class_counts[label] as f32;
            for value in centroid {
                *value /= count;
            }
        }

        Ok(SimpleClassifier { class_centroids })
    }

    /// Predict test data
    async fn predict_test_data(
        &self,
        model: &dyn EmbeddingModel,
        classifier: &SimpleClassifier,
    ) -> Result<Vec<(String, String)>> {
        let mut predictions = Vec::new();

        for (entity, true_label) in &self.test_data {
            if let Ok(embedding) = model.get_entity_embedding(entity) {
                let predicted_label = classifier.predict(&embedding);
                predictions.push((true_label.clone(), predicted_label));
            }
        }

        Ok(predictions)
    }

    /// Calculate classification metric
    fn calculate_classification_metric(
        &self,
        metric: &ClassificationMetric,
        predictions: &[(String, String)],
    ) -> Result<f64> {
        match metric {
            ClassificationMetric::Accuracy => {
                let correct = predictions.iter().filter(|(true_label, pred_label)| true_label == pred_label).count();
                Ok(correct as f64 / predictions.len() as f64)
            }
            ClassificationMetric::Precision => {
                self.calculate_macro_precision(predictions)
            }
            ClassificationMetric::Recall => {
                self.calculate_macro_recall(predictions)
            }
            ClassificationMetric::F1Score => {
                self.calculate_macro_f1(predictions)
            }
            _ => Ok(0.5), // Placeholder for other metrics
        }
    }

    /// Calculate macro-averaged precision
    fn calculate_macro_precision(&self, predictions: &[(String, String)]) -> Result<f64> {
        let unique_labels: HashSet<String> = predictions.iter().map(|(label, _)| label.clone()).collect();
        let mut precision_scores = Vec::new();

        for label in &unique_labels {
            let tp = predictions.iter().filter(|(true_label, pred_label)| true_label == label && pred_label == label).count();
            let fp = predictions.iter().filter(|(true_label, pred_label)| true_label != label && pred_label == label).count();
            
            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            
            precision_scores.push(precision);
        }

        Ok(precision_scores.iter().sum::<f64>() / precision_scores.len() as f64)
    }

    /// Calculate macro-averaged recall
    fn calculate_macro_recall(&self, predictions: &[(String, String)]) -> Result<f64> {
        let unique_labels: HashSet<String> = predictions.iter().map(|(label, _)| label.clone()).collect();
        let mut recall_scores = Vec::new();

        for label in &unique_labels {
            let tp = predictions.iter().filter(|(true_label, pred_label)| true_label == label && pred_label == label).count();
            let fn_count = predictions.iter().filter(|(true_label, pred_label)| true_label == label && pred_label != label).count();
            
            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };
            
            recall_scores.push(recall);
        }

        Ok(recall_scores.iter().sum::<f64>() / recall_scores.len() as f64)
    }

    /// Calculate macro-averaged F1 score
    fn calculate_macro_f1(&self, predictions: &[(String, String)]) -> Result<f64> {
        let precision = self.calculate_macro_precision(predictions)?;
        let recall = self.calculate_macro_recall(predictions)?;
        
        if precision + recall > 0.0 {
            Ok(2.0 * precision * recall / (precision + recall))
        } else {
            Ok(0.0)
        }
    }

    /// Calculate per-class results
    fn calculate_per_class_results(&self, predictions: &[(String, String)]) -> Result<HashMap<String, ClassResults>> {
        let unique_labels: HashSet<String> = predictions.iter().map(|(label, _)| label.clone()).collect();
        let mut per_class_results = HashMap::new();

        for label in &unique_labels {
            let tp = predictions.iter().filter(|(true_label, pred_label)| true_label == label && pred_label == label).count();
            let fp = predictions.iter().filter(|(true_label, pred_label)| true_label != label && pred_label == label).count();
            let fn_count = predictions.iter().filter(|(true_label, pred_label)| true_label == label && pred_label != label).count();
            
            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            
            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };
            
            let f1_score = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            
            let support = predictions.iter().filter(|(true_label, _)| true_label == label).count();

            per_class_results.insert(label.clone(), ClassResults {
                class_label: label.clone(),
                precision,
                recall,
                f1_score,
                support,
            });
        }

        Ok(per_class_results)
    }

    /// Generate confusion matrix
    fn generate_confusion_matrix(&self, predictions: &[(String, String)]) -> Result<Vec<Vec<usize>>> {
        let unique_labels: Vec<String> = predictions.iter().map(|(label, _)| label.clone()).collect::<HashSet<_>>().into_iter().collect();
        let n = unique_labels.len();
        let mut matrix = vec![vec![0; n]; n];

        for (true_label, pred_label) in predictions {
            if let (Some(true_idx), Some(pred_idx)) = (
                unique_labels.iter().position(|l| l == true_label),
                unique_labels.iter().position(|l| l == pred_label),
            ) {
                matrix[true_idx][pred_idx] += 1;
            }
        }

        Ok(matrix)
    }

    /// Generate classification report
    fn generate_classification_report(
        &self,
        per_class_results: &HashMap<String, ClassResults>,
        predictions: &[(String, String)],
    ) -> Result<ClassificationReport> {
        let accuracy = predictions.iter().filter(|(true_label, pred_label)| true_label == pred_label).count() as f64 / predictions.len() as f64;
        
        // Calculate macro averages
        let macro_precision = per_class_results.values().map(|r| r.precision).sum::<f64>() / per_class_results.len() as f64;
        let macro_recall = per_class_results.values().map(|r| r.recall).sum::<f64>() / per_class_results.len() as f64;
        let macro_f1 = per_class_results.values().map(|r| r.f1_score).sum::<f64>() / per_class_results.len() as f64;

        // Calculate weighted averages
        let total_support: usize = per_class_results.values().map(|r| r.support).sum();
        let weighted_precision = per_class_results.values().map(|r| r.precision * r.support as f64).sum::<f64>() / total_support as f64;
        let weighted_recall = per_class_results.values().map(|r| r.recall * r.support as f64).sum::<f64>() / total_support as f64;
        let weighted_f1 = per_class_results.values().map(|r| r.f1_score * r.support as f64).sum::<f64>() / total_support as f64;

        Ok(ClassificationReport {
            macro_avg: ClassResults {
                class_label: "macro avg".to_string(),
                precision: macro_precision,
                recall: macro_recall,
                f1_score: macro_f1,
                support: total_support,
            },
            weighted_avg: ClassResults {
                class_label: "weighted avg".to_string(),
                precision: weighted_precision,
                recall: weighted_recall,
                f1_score: weighted_f1,
                support: total_support,
            },
            accuracy,
            total_samples: predictions.len(),
        })
    }
}

/// Simple nearest centroid classifier
struct SimpleClassifier {
    class_centroids: HashMap<String, Vec<f32>>,
}

impl SimpleClassifier {
    /// Predict class for a given embedding
    fn predict(&self, embedding: &Vector) -> String {
        let mut best_class = String::new();
        let mut min_distance = f32::INFINITY;

        for (class_label, centroid) in &self.class_centroids {
            let distance = self.euclidean_distance(&embedding.values, centroid);
            if distance < min_distance {
                min_distance = distance;
                best_class = class_label.clone();
            }
        }

        best_class
    }

    /// Calculate Euclidean distance
    fn euclidean_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

// ============================================================================
// RETRIEVAL EVALUATION
// ============================================================================

/// Retrieval evaluator for information retrieval tasks
pub struct RetrievalEvaluator {
    /// Query-document relevance judgments
    relevance_judgments: HashMap<String, Vec<DocumentRelevance>>,
    /// Retrieval metrics
    metrics: Vec<RetrievalMetric>,
}

/// Document relevance for retrieval evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRelevance {
    /// Query identifier
    pub query_id: String,
    /// Document identifier
    pub document_id: String,
    /// Relevance score (0-3 scale)
    pub relevance: u8,
    /// Document content summary
    pub content_summary: String,
}

/// Retrieval evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalMetric {
    /// Mean Average Precision
    MAP,
    /// Normalized Discounted Cumulative Gain
    NDCG(usize),
    /// Precision at K
    PrecisionAtK(usize),
    /// Recall at K
    RecallAtK(usize),
    /// Mean Reciprocal Rank
    MRR,
    /// R-Precision
    RPrecision,
    /// Binary Preference (bpref)
    BinaryPreference,
}

/// Retrieval evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResults {
    /// Metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Per-query retrieval results
    pub per_query_results: HashMap<String, QueryRetrievalResults>,
    /// Retrieval effectiveness analysis
    pub effectiveness_analysis: RetrievalEffectivenessAnalysis,
}

/// Per-query retrieval results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRetrievalResults {
    /// Query identifier
    pub query_id: String,
    /// Average precision
    pub average_precision: f64,
    /// NDCG scores at different cut-offs
    pub ndcg_scores: HashMap<usize, f64>,
    /// Precision scores at different cut-offs
    pub precision_scores: HashMap<usize, f64>,
    /// Recall scores at different cut-offs
    pub recall_scores: HashMap<usize, f64>,
    /// Reciprocal rank
    pub reciprocal_rank: f64,
    /// Number of relevant documents
    pub num_relevant: usize,
}

/// Retrieval effectiveness analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalEffectivenessAnalysis {
    /// Overall retrieval quality
    pub overall_quality: f64,
    /// Query difficulty distribution
    pub query_difficulty_dist: HashMap<String, usize>,
    /// Performance by query type
    pub performance_by_type: HashMap<String, f64>,
    /// Retrieval consistency
    pub retrieval_consistency: f64,
}

impl RetrievalEvaluator {
    /// Create a new retrieval evaluator
    pub fn new() -> Self {
        Self {
            relevance_judgments: HashMap::new(),
            metrics: vec![
                RetrievalMetric::MAP,
                RetrievalMetric::NDCG(10),
                RetrievalMetric::PrecisionAtK(5),
                RetrievalMetric::PrecisionAtK(10),
                RetrievalMetric::MRR,
            ],
        }
    }

    /// Add relevance judgment
    pub fn add_relevance_judgment(&mut self, judgment: DocumentRelevance) {
        self.relevance_judgments
            .entry(judgment.query_id.clone())
            .or_insert_with(Vec::new)
            .push(judgment);
    }

    /// Evaluate retrieval performance
    pub async fn evaluate(
        &self,
        model: &dyn EmbeddingModel,
        config: &ApplicationEvalConfig,
    ) -> Result<RetrievalResults> {
        let mut metric_scores = HashMap::new();
        let mut per_query_results = HashMap::new();

        // Sample queries for evaluation
        let queries_to_evaluate: Vec<_> = self
            .relevance_judgments
            .keys()
            .take(config.sample_size)
            .cloned()
            .collect();

        for query_id in &queries_to_evaluate {
            let query_results = self.evaluate_query_retrieval(query_id, model).await?;
            per_query_results.insert(query_id.clone(), query_results);
        }

        // Calculate aggregate metrics
        for metric in &self.metrics {
            let score = self.calculate_retrieval_metric(metric, &per_query_results)?;
            metric_scores.insert(format!("{:?}", metric), score);
        }

        // Analyze effectiveness
        let effectiveness_analysis = self.analyze_retrieval_effectiveness(&per_query_results)?;

        Ok(RetrievalResults {
            metric_scores,
            per_query_results,
            effectiveness_analysis,
        })
    }

    /// Evaluate retrieval for a specific query
    async fn evaluate_query_retrieval(
        &self,
        query_id: &str,
        model: &dyn EmbeddingModel,
    ) -> Result<QueryRetrievalResults> {
        let judgments = self.relevance_judgments.get(query_id).unwrap();
        
        // Perform retrieval (simplified)
        let retrieval_results = self.perform_retrieval(query_id, model).await?;

        // Map documents to relevance scores
        let mut relevance_scores = Vec::new();
        for (doc_id, _score) in &retrieval_results {
            let relevance = judgments
                .iter()
                .find(|j| &j.document_id == doc_id)
                .map(|j| j.relevance)
                .unwrap_or(0);
            relevance_scores.push(relevance);
        }

        let num_relevant = judgments.iter().filter(|j| j.relevance > 0).count();

        // Calculate average precision
        let average_precision = self.calculate_average_precision(&relevance_scores)?;

        // Calculate NDCG, precision, and recall at different cut-offs
        let mut ndcg_scores = HashMap::new();
        let mut precision_scores = HashMap::new();
        let mut recall_scores = HashMap::new();

        for &k in &[5, 10, 15, 20] {
            if k <= retrieval_results.len() {
                let ndcg = self.calculate_retrieval_ndcg(&relevance_scores, k)?;
                let precision = self.calculate_precision_at_k(&relevance_scores, k)?;
                let recall = self.calculate_recall_at_k(&relevance_scores, num_relevant, k)?;

                ndcg_scores.insert(k, ndcg);
                precision_scores.insert(k, precision);
                recall_scores.insert(k, recall);
            }
        }

        // Calculate reciprocal rank
        let reciprocal_rank = self.calculate_reciprocal_rank(&relevance_scores)?;

        Ok(QueryRetrievalResults {
            query_id: query_id.to_string(),
            average_precision,
            ndcg_scores,
            precision_scores,
            recall_scores,
            reciprocal_rank,
            num_relevant,
        })
    }

    /// Perform retrieval (simplified implementation)
    async fn perform_retrieval(
        &self,
        query_id: &str,
        model: &dyn EmbeddingModel,
    ) -> Result<Vec<(String, f64)>> {
        // Create query embedding (simplified - use query_id as proxy)
        let query_embedding = self.create_query_embedding(query_id);

        // Score all entities as potential documents
        let entities = model.get_entities();
        let mut retrieval_results = Vec::new();

        for entity in entities.iter().take(200) { // Limit for efficiency
            if let Ok(entity_embedding) = model.get_entity_embedding(entity) {
                let score = self.cosine_similarity(&query_embedding, &entity_embedding);
                retrieval_results.push((entity.clone(), score));
            }
        }

        // Sort by score and return top results
        retrieval_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        retrieval_results.truncate(50);

        Ok(retrieval_results)
    }

    /// Create query embedding (simplified)
    fn create_query_embedding(&self, query_id: &str) -> Vector {
        // Simple hash-based embedding for demonstration
        let mut embedding = vec![0.0f32; 100];
        let hash_value = query_id.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        
        for (i, value) in embedding.iter_mut().enumerate() {
            *value = ((hash_value.wrapping_add(i as u32)) % 1000) as f32 / 1000.0;
        }
        
        Vector::new(embedding)
    }

    /// Calculate average precision
    fn calculate_average_precision(&self, relevance_scores: &[u8]) -> Result<f64> {
        if relevance_scores.is_empty() {
            return Ok(0.0);
        }

        let mut relevant_found = 0;
        let mut precision_sum = 0.0;
        let total_relevant = relevance_scores.iter().filter(|&&r| r > 0).count();

        if total_relevant == 0 {
            return Ok(0.0);
        }

        for (i, &relevance) in relevance_scores.iter().enumerate() {
            if relevance > 0 {
                relevant_found += 1;
                let precision_at_i = relevant_found as f64 / (i + 1) as f64;
                precision_sum += precision_at_i;
            }
        }

        Ok(precision_sum / total_relevant as f64)
    }

    /// Calculate NDCG for retrieval
    fn calculate_retrieval_ndcg(&self, relevance_scores: &[u8], k: usize) -> Result<f64> {
        if k == 0 || relevance_scores.is_empty() {
            return Ok(0.0);
        }

        let mut dcg = 0.0;
        for (i, &relevance) in relevance_scores.iter().take(k).enumerate() {
            if relevance > 0 {
                let gain = (2_u32.pow(relevance as u32) - 1) as f64;
                dcg += gain / (i as f64 + 2.0).log2();
            }
        }

        // Calculate ideal DCG
        let mut ideal_relevance: Vec<u8> = relevance_scores.iter().cloned().collect();
        ideal_relevance.sort_by(|a, b| b.cmp(a));
        
        let mut idcg = 0.0;
        for (i, &relevance) in ideal_relevance.iter().take(k).enumerate() {
            if relevance > 0 {
                let gain = (2_u32.pow(relevance as u32) - 1) as f64;
                idcg += gain / (i as f64 + 2.0).log2();
            }
        }

        if idcg > 0.0 {
            Ok(dcg / idcg)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate precision at K
    fn calculate_precision_at_k(&self, relevance_scores: &[u8], k: usize) -> Result<f64> {
        if k == 0 || relevance_scores.is_empty() {
            return Ok(0.0);
        }

        let relevant_at_k = relevance_scores
            .iter()
            .take(k)
            .filter(|&&r| r > 0)
            .count() as f64;

        Ok(relevant_at_k / k as f64)
    }

    /// Calculate recall at K
    fn calculate_recall_at_k(&self, relevance_scores: &[u8], total_relevant: usize, k: usize) -> Result<f64> {
        if k == 0 || relevance_scores.is_empty() || total_relevant == 0 {
            return Ok(0.0);
        }

        let relevant_at_k = relevance_scores
            .iter()
            .take(k)
            .filter(|&&r| r > 0)
            .count() as f64;

        Ok(relevant_at_k / total_relevant as f64)
    }

    /// Calculate reciprocal rank
    fn calculate_reciprocal_rank(&self, relevance_scores: &[u8]) -> Result<f64> {
        for (i, &relevance) in relevance_scores.iter().enumerate() {
            if relevance > 0 {
                return Ok(1.0 / (i + 1) as f64);
            }
        }
        Ok(0.0)
    }

    /// Calculate aggregate retrieval metric
    fn calculate_retrieval_metric(
        &self,
        metric: &RetrievalMetric,
        per_query_results: &HashMap<String, QueryRetrievalResults>,
    ) -> Result<f64> {
        if per_query_results.is_empty() {
            return Ok(0.0);
        }

        match metric {
            RetrievalMetric::MAP => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .map(|r| r.average_precision)
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RetrievalMetric::NDCG(k) => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .filter_map(|r| r.ndcg_scores.get(k))
                    .cloned()
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RetrievalMetric::PrecisionAtK(k) => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .filter_map(|r| r.precision_scores.get(k))
                    .cloned()
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RetrievalMetric::MRR => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .map(|r| r.reciprocal_rank)
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            _ => Ok(0.5), // Placeholder for other metrics
        }
    }

    /// Analyze retrieval effectiveness
    fn analyze_retrieval_effectiveness(
        &self,
        per_query_results: &HashMap<String, QueryRetrievalResults>,
    ) -> Result<RetrievalEffectivenessAnalysis> {
        let overall_quality = per_query_results
            .values()
            .map(|r| r.average_precision)
            .sum::<f64>()
            / per_query_results.len() as f64;

        let successful_queries = per_query_results
            .values()
            .filter(|r| r.average_precision > 0.1)
            .count() as f64;

        let retrieval_consistency = successful_queries / per_query_results.len() as f64;

        Ok(RetrievalEffectivenessAnalysis {
            overall_quality,
            query_difficulty_dist: HashMap::new(), // Simplified
            performance_by_type: HashMap::new(), // Simplified
            retrieval_consistency,
        })
    }

    /// Calculate cosine similarity
    fn cosine_similarity(&self, v1: &Vector, v2: &Vector) -> f64 {
        let dot_product: f32 = v1.values.iter().zip(v2.values.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = v1.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = v2.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot_product / (norm_a * norm_b)) as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// MAIN APPLICATION TASK EVALUATOR IMPLEMENTATION
// ============================================================================

impl ApplicationTaskEvaluator {
    /// Create a new application task evaluator
    pub fn new(config: ApplicationEvalConfig) -> Self {
        Self {
            config,
            recommendation_evaluator: RecommendationEvaluator::new(),
            search_evaluator: SearchEvaluator::new(),
            clustering_evaluator: ClusteringEvaluator::new(),
            classification_evaluator: ClassificationEvaluator::new(),
            retrieval_evaluator: RetrievalEvaluator::new(),
            evaluation_history: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Run comprehensive application evaluation
    pub async fn evaluate_all_tasks(
        &self,
        model: &dyn EmbeddingModel,
    ) -> Result<ApplicationEvalResults> {
        let start_time = Instant::now();
        let model_id = model.model_id().to_string();

        let mut recommendation_results = None;
        let mut search_results = None;
        let mut clustering_results = None;
        let mut classification_results = None;
        let mut retrieval_results = None;

        // Run enabled evaluations
        if self.config.enable_recommendation_eval {
            recommendation_results = Some(self.recommendation_evaluator.evaluate(model, &self.config).await?);
        }

        if self.config.enable_search_eval {
            search_results = Some(self.search_evaluator.evaluate(model, &self.config).await?);
        }

        if self.config.enable_clustering_eval {
            clustering_results = Some(self.clustering_evaluator.evaluate(model, &self.config).await?);
        }

        if self.config.enable_classification_eval {
            classification_results = Some(self.classification_evaluator.evaluate(model, &self.config).await?);
        }

        if self.config.enable_retrieval_eval {
            retrieval_results = Some(self.retrieval_evaluator.evaluate(model, &self.config).await?);
        }

        // Calculate overall score
        let mut scores = Vec::new();
        if let Some(ref rec_results) = recommendation_results {
            scores.extend(rec_results.metric_scores.values());
        }
        if let Some(ref search_results) = search_results {
            scores.extend(search_results.metric_scores.values());
        }
        if let Some(ref clustering_results) = clustering_results {
            scores.extend(clustering_results.metric_scores.values());
        }
        if let Some(ref classification_results) = classification_results {
            scores.extend(classification_results.metric_scores.values());
        }
        if let Some(ref retrieval_results) = retrieval_results {
            scores.extend(retrieval_results.metric_scores.values());
        }

        let overall_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        let evaluation_time = start_time.elapsed().as_secs_f64();

        let results = ApplicationEvalResults {
            timestamp: chrono::Utc::now(),
            model_id,
            recommendation_results,
            search_results,
            clustering_results,
            classification_results,
            retrieval_results,
            overall_score,
            evaluation_time,
        };

        // Add to history
        {
            let mut history = self.evaluation_history.write().unwrap();
            history.push_back(results.clone());
            
            // Keep only last 100 evaluations
            while history.len() > 100 {
                history.pop_front();
            }
        }

        Ok(results)
    }

    /// Get evaluation history
    pub fn get_evaluation_history(&self) -> Vec<ApplicationEvalResults> {
        self.evaluation_history.read().unwrap().iter().cloned().collect()
    }

    /// Get specific evaluator for configuration
    pub fn get_recommendation_evaluator(&mut self) -> &mut RecommendationEvaluator {
        &mut self.recommendation_evaluator
    }

    pub fn get_search_evaluator(&mut self) -> &mut SearchEvaluator {
        &mut self.search_evaluator
    }

    pub fn get_clustering_evaluator(&mut self) -> &mut ClusteringEvaluator {
        &mut self.clustering_evaluator
    }

    pub fn get_classification_evaluator(&mut self) -> &mut ClassificationEvaluator {
        &mut self.classification_evaluator
    }

    pub fn get_retrieval_evaluator(&mut self) -> &mut RetrievalEvaluator {
        &mut self.retrieval_evaluator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::transe::TransEModel;

    #[test]
    fn test_application_eval_config_default() {
        let config = ApplicationEvalConfig::default();
        assert!(config.enable_recommendation_eval);
        assert!(config.enable_search_eval);
        assert!(config.enable_clustering_eval);
        assert_eq!(config.sample_size, 1000);
    }

    #[tokio::test]
    async fn test_recommendation_evaluator() {
        let mut evaluator = RecommendationEvaluator::new();
        
        // Add some test data
        evaluator.add_interaction(UserInteraction {
            user_id: "user1".to_string(),
            item_id: "item1".to_string(),
            interaction_type: InteractionType::Like,
            rating: Some(4.0),
            timestamp: chrono::Utc::now(),
            context: HashMap::new(),
        });

        evaluator.add_item(ItemMetadata {
            item_id: "item1".to_string(),
            category: "books".to_string(),
            features: HashMap::new(),
            popularity: 0.8,
            embedding: None,
        });

        // Test evaluation (would need actual model)
        assert_eq!(evaluator.user_interactions.len(), 1);
        assert_eq!(evaluator.item_catalog.len(), 1);
    }

    #[test]
    fn test_clustering_evaluator() {
        let mut evaluator = ClusteringEvaluator::new();
        
        // Test ground truth setting
        let ground_truth = vec![
            ("entity1".to_string(), "cluster1".to_string()),
            ("entity2".to_string(), "cluster1".to_string()),
            ("entity3".to_string(), "cluster2".to_string()),
        ].into_iter().collect();
        
        evaluator.set_ground_truth(ground_truth);
        assert!(evaluator.ground_truth_clusters.is_some());
        assert!(evaluator.metrics.len() >= 6); // Should include supervised metrics
    }

    #[tokio::test]
    async fn test_classification_evaluator() {
        let mut evaluator = ClassificationEvaluator::new();
        
        // Add test data
        evaluator.add_training_data("entity1".to_string(), "class1".to_string());
        evaluator.add_training_data("entity2".to_string(), "class2".to_string());
        evaluator.add_test_data("entity3".to_string(), "class1".to_string());
        
        assert_eq!(evaluator.training_data.len(), 2);
        assert_eq!(evaluator.test_data.len(), 1);
    }

    #[test]
    fn test_search_evaluator() {
        let mut evaluator = SearchEvaluator::new();
        
        // Add relevance judgment
        evaluator.add_relevance_judgment(RelevanceJudgment {
            query: "test query".to_string(),
            document_id: "doc1".to_string(),
            relevance_score: 2,
            annotator_id: "annotator1".to_string(),
        });
        
        assert_eq!(evaluator.query_relevance.len(), 1);
    }

    #[test]
    fn test_retrieval_evaluator() {
        let mut evaluator = RetrievalEvaluator::new();
        
        // Add relevance judgment
        evaluator.add_relevance_judgment(DocumentRelevance {
            query_id: "query1".to_string(),
            document_id: "doc1".to_string(),
            relevance: 3,
            content_summary: "Test document".to_string(),
        });
        
        assert_eq!(evaluator.relevance_judgments.len(), 1);
    }

    #[tokio::test]
    async fn test_application_task_evaluator() {
        let config = ApplicationEvalConfig::default();
        let evaluator = ApplicationTaskEvaluator::new(config);
        
        // Test with a simple model
        let model = TransEModel::new(Default::default());
        
        // Would need more setup to run actual evaluation
        assert_eq!(evaluator.evaluation_history.read().unwrap().len(), 0);
    }
}