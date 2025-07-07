//! Application-specific evaluation tasks for embedding models
//!
//! This module implements comprehensive evaluation metrics and benchmarks for
//! real-world applications of knowledge graph embeddings including recommendation
//! systems, search relevance, clustering performance, and classification accuracy.

pub mod classification;
pub mod clustering;
pub mod query_answering;
pub mod recommendation;
pub mod retrieval;
pub mod search;

use crate::EmbeddingModel;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::Instant;

// Re-export all public types from submodules
pub use classification::{
    ClassResults, ClassificationEvaluator, ClassificationMetric, ClassificationReport,
    ClassificationResults, SimpleClassifier,
};
pub use clustering::{
    ClusterAnalysis, ClusteringEvaluator, ClusteringMetric, ClusteringResults,
    ClusteringStabilityAnalysis,
};
pub use query_answering::{
    ApplicationQueryAnsweringEvaluator, ComplexityResults, QueryAnsweringMetric,
    QueryAnsweringResults, QueryComplexity, QueryResult, QueryType, QuestionAnswerPair,
    ReasoningAnalysis, TypeResults,
};
pub use recommendation::{
    ABTestResults, CoverageStats, DiversityAnalysis, InteractionType, ItemMetadata,
    RecommendationEvaluator, RecommendationMetric, RecommendationResults, UserInteraction,
    UserRecommendationResults,
};
pub use retrieval::{
    DocumentMetadata, RetrievalAnalysis, RetrievalEvaluator, RetrievalMetric, RetrievalQuery,
    RetrievalResults,
};
pub use search::{
    QueryPerformanceAnalysis, QueryResults, RelevanceJudgment, SearchEffectivenessMetrics,
    SearchEvaluator, SearchMetric, SearchResults,
};

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
    /// Enable query answering evaluation
    pub enable_query_answering_eval: bool,
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
    /// Number of query answering tests
    pub num_query_tests: usize,
}

impl Default for ApplicationEvalConfig {
    fn default() -> Self {
        Self {
            enable_recommendation_eval: true,
            enable_search_eval: true,
            enable_clustering_eval: true,
            enable_classification_eval: true,
            enable_retrieval_eval: true,
            enable_query_answering_eval: true,
            sample_size: 1000,
            num_recommendations: 10,
            num_clusters: 5,
            cv_folds: 5,
            enable_user_satisfaction: true,
            num_query_tests: 100,
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
    /// Query answering evaluation results
    pub query_answering_results: Option<QueryAnsweringResults>,
    /// Overall application score
    pub overall_score: f64,
    /// Evaluation time (seconds)
    pub evaluation_time: f64,
}

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
    query_answering_evaluator: ApplicationQueryAnsweringEvaluator,
    /// Evaluation history
    evaluation_history: Arc<RwLock<VecDeque<ApplicationEvalResults>>>,
}

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
            query_answering_evaluator: ApplicationQueryAnsweringEvaluator::new(),
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
        let mut query_answering_results = None;

        // Run enabled evaluations
        if self.config.enable_recommendation_eval {
            recommendation_results = Some(
                self.recommendation_evaluator
                    .evaluate(model, &self.config)
                    .await?,
            );
        }

        if self.config.enable_search_eval {
            search_results = Some(self.search_evaluator.evaluate(model, &self.config).await?);
        }

        if self.config.enable_clustering_eval {
            clustering_results = Some(
                self.clustering_evaluator
                    .evaluate(model, &self.config)
                    .await?,
            );
        }

        if self.config.enable_classification_eval {
            classification_results = Some(
                self.classification_evaluator
                    .evaluate(model, &self.config)
                    .await?,
            );
        }

        if self.config.enable_retrieval_eval {
            retrieval_results = Some(
                self.retrieval_evaluator
                    .evaluate(model, &self.config)
                    .await?,
            );
        }

        if self.config.enable_query_answering_eval {
            query_answering_results = Some(
                self.query_answering_evaluator
                    .evaluate(model, &self.config)
                    .await?,
            );
        }

        let evaluation_time = start_time.elapsed().as_secs_f64();

        // Calculate overall score
        let overall_score = self.calculate_overall_score(
            &recommendation_results,
            &search_results,
            &clustering_results,
            &classification_results,
            &retrieval_results,
            &query_answering_results,
        );

        let results = ApplicationEvalResults {
            timestamp: chrono::Utc::now(),
            model_id,
            recommendation_results,
            search_results,
            clustering_results,
            classification_results,
            retrieval_results,
            query_answering_results,
            overall_score,
            evaluation_time,
        };

        // Store in history
        if let Ok(mut history) = self.evaluation_history.write() {
            history.push_back(results.clone());
            if history.len() > 100 {
                history.pop_front();
            }
        }

        Ok(results)
    }

    /// Calculate overall score from individual evaluation results
    fn calculate_overall_score(
        &self,
        recommendation: &Option<RecommendationResults>,
        search: &Option<SearchResults>,
        clustering: &Option<ClusteringResults>,
        classification: &Option<ClassificationResults>,
        retrieval: &Option<RetrievalResults>,
        query_answering: &Option<QueryAnsweringResults>,
    ) -> f64 {
        let mut total_score = 0.0;
        let mut component_count = 0;

        if let Some(rec) = recommendation {
            // Extract key metrics from recommendation results
            let precision = rec.metric_scores.get("PrecisionAtK(5)").unwrap_or(&0.0);
            let coverage = rec.metric_scores.get("Coverage").unwrap_or(&0.0);
            total_score += (precision + coverage) / 2.0;
            component_count += 1;
        }

        if let Some(search) = search {
            let ndcg = search.metric_scores.get("NDCG(10)").unwrap_or(&0.0);
            let map = search.metric_scores.get("MAP").unwrap_or(&0.0);
            total_score += (ndcg + map) / 2.0;
            component_count += 1;
        }

        if let Some(clustering) = clustering {
            let silhouette = clustering
                .metric_scores
                .get("SilhouetteScore")
                .unwrap_or(&0.0);
            total_score += silhouette.abs(); // Silhouette can be negative
            component_count += 1;
        }

        if let Some(classification) = classification {
            let accuracy = classification.metric_scores.get("Accuracy").unwrap_or(&0.0);
            let f1 = classification.metric_scores.get("F1Score").unwrap_or(&0.0);
            total_score += (accuracy + f1) / 2.0;
            component_count += 1;
        }

        if let Some(retrieval) = retrieval {
            let precision = retrieval
                .metric_scores
                .get("PrecisionAtK(10)")
                .unwrap_or(&0.0);
            let recall = retrieval.metric_scores.get("RecallAtK(10)").unwrap_or(&0.0);
            total_score += (precision + recall) / 2.0;
            component_count += 1;
        }

        if let Some(qa) = query_answering {
            total_score += qa.overall_accuracy;
            component_count += 1;
        }

        if component_count > 0 {
            total_score / component_count as f64
        } else {
            0.0
        }
    }

    /// Get evaluation history
    pub fn get_evaluation_history(&self) -> Result<Vec<ApplicationEvalResults>> {
        match self.evaluation_history.read() {
            Ok(history) => Ok(history.iter().cloned().collect()),
            _ => Err(anyhow!("Failed to read evaluation history")),
        }
    }

    /// Clear evaluation history
    pub fn clear_history(&self) -> Result<()> {
        match self.evaluation_history.write() {
            Ok(mut history) => {
                history.clear();
                Ok(())
            }
            _ => Err(anyhow!("Failed to clear evaluation history")),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ApplicationEvalConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: ApplicationEvalConfig) {
        self.config = config;
    }
}
