//! Search relevance evaluation module
//!
//! This module provides comprehensive evaluation for search relevance using
//! embedding models, including precision, recall, NDCG, MAP, and other
//! information retrieval metrics.

use super::ApplicationEvalConfig;
use crate::{EmbeddingModel, Vector};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// Average precision over the full ranked result list, for aggregating
    /// into Mean Average Precision (MAP) across queries.
    pub average_precision: f64,
    /// Reciprocal rank of the first relevant result (0.0 if none), for
    /// aggregating into Mean Reciprocal Rank (MRR) across queries.
    pub reciprocal_rank: f64,
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

/// Search relevance evaluator
pub struct SearchEvaluator {
    /// Search queries and their relevance judgments
    query_relevance: HashMap<String, Vec<RelevanceJudgment>>,
    /// Search metrics to evaluate
    metrics: Vec<SearchMetric>,
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
            .or_default()
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
            metric_scores.insert(format!("{metric:?}"), score);
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
        let judgments = self
            .query_relevance
            .get(query)
            .expect("query should exist in query_relevance");

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
                let relevant_at_k =
                    relevance_scores.iter().take(k).filter(|&&r| r > 0).count() as f64;

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

        // Average precision over the full ranked list (for MAP): the mean of
        // precision@k evaluated at every rank that holds a relevant result.
        let average_precision = if num_relevant > 0 {
            let mut hits = 0usize;
            let mut precision_sum = 0.0;
            for (rank, &relevance) in relevance_scores.iter().enumerate() {
                if relevance > 0 {
                    hits += 1;
                    precision_sum += hits as f64 / (rank + 1) as f64;
                }
            }
            precision_sum / num_relevant as f64
        } else {
            0.0
        };

        // Reciprocal rank of the first relevant result (for MRR).
        let reciprocal_rank = relevance_scores
            .iter()
            .position(|&relevance| relevance > 0)
            .map(|rank| 1.0 / (rank + 1) as f64)
            .unwrap_or(0.0);

        Ok(QueryResults {
            query: query.to_string(),
            precision_scores,
            recall_scores,
            ndcg_scores,
            num_relevant,
            difficulty_score,
            average_precision,
            reciprocal_rank,
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

        for entity in entities.iter().take(100) {
            // Limit for efficiency
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
        let mut ideal_relevance: Vec<u8> = relevance_scores.to_vec();
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
            SearchMetric::RecallAtK(k) => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .filter_map(|r| r.recall_scores.get(k))
                    .cloned()
                    .collect();
                if scores.is_empty() {
                    return Err(anyhow!("No recall@{k} scores available across queries"));
                }
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
            SearchMetric::MAP => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .map(|r| r.average_precision)
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            SearchMetric::MRR => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .map(|r| r.reciprocal_rank)
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            SearchMetric::ERR | SearchMetric::CTR => Err(anyhow!(
                "Search metric {metric:?} is not yet implemented: computing it requires a \
                 graded-relevance cascade/click model that QueryResults does not currently track"
            )),
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
            relevance_distribution: HashMap::new(),        // Simplified
            result_diversity: 0.6,                         // Simplified
            query_success_rate,
        })
    }

    /// Calculate cosine similarity
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
}

impl Default for SearchEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_query_results(average_precision: f64, reciprocal_rank: f64) -> QueryResults {
        QueryResults {
            query: "test query".to_string(),
            precision_scores: HashMap::new(),
            recall_scores: [(5usize, 0.5)].into_iter().collect(),
            ndcg_scores: HashMap::new(),
            num_relevant: 2,
            difficulty_score: 0.0,
            average_precision,
            reciprocal_rank,
        }
    }

    /// Regression test: MAP/MRR must be computed for real from
    /// `average_precision`/`reciprocal_rank` instead of a hardcoded 0.5.
    #[test]
    fn test_calculate_search_metric_map_and_mrr_are_real() -> Result<()> {
        let evaluator = SearchEvaluator::new();
        let mut per_query = HashMap::new();
        per_query.insert("q1".to_string(), sample_query_results(1.0, 1.0));
        per_query.insert("q2".to_string(), sample_query_results(0.5, 0.5));

        let map = evaluator.calculate_search_metric(&SearchMetric::MAP, &per_query)?;
        assert!((map - 0.75).abs() < 1e-9, "map = {map}");

        let mrr = evaluator.calculate_search_metric(&SearchMetric::MRR, &per_query)?;
        assert!((mrr - 0.75).abs() < 1e-9, "mrr = {mrr}");

        Ok(())
    }

    #[test]
    fn test_calculate_search_metric_recall_at_k_is_real() -> Result<()> {
        let evaluator = SearchEvaluator::new();
        let mut per_query = HashMap::new();
        per_query.insert("q1".to_string(), sample_query_results(0.0, 0.0));

        let recall = evaluator.calculate_search_metric(&SearchMetric::RecallAtK(5), &per_query)?;
        assert!((recall - 0.5).abs() < 1e-9, "recall = {recall}");

        Ok(())
    }

    /// Metrics that genuinely cannot be computed from the tracked data must
    /// fail loudly rather than return a fabricated placeholder score.
    #[test]
    fn test_calculate_search_metric_err_and_ctr_are_explicit_errors() {
        let evaluator = SearchEvaluator::new();
        let mut per_query = HashMap::new();
        per_query.insert("q1".to_string(), sample_query_results(1.0, 1.0));

        assert!(evaluator
            .calculate_search_metric(&SearchMetric::ERR, &per_query)
            .is_err());
        assert!(evaluator
            .calculate_search_metric(&SearchMetric::CTR, &per_query)
            .is_err());
    }
}
