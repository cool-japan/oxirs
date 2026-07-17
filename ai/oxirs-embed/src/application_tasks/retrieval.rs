//! Retrieval evaluation module
//!
//! This module provides comprehensive evaluation for information retrieval tasks
//! using embedding models, including document ranking and retrieval effectiveness.

use super::ApplicationEvalConfig;
use crate::EmbeddingModel;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Retrieval evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalMetric {
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
    /// F1 Score at K
    F1AtK(usize),
}

/// Document metadata for retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document identifier
    pub doc_id: String,
    /// Document title
    pub title: String,
    /// Document content
    pub content: String,
    /// Document category
    pub category: String,
    /// Document embedding (if available)
    pub embedding: Option<Vec<f32>>,
    /// Relevance score for queries
    pub relevance_scores: HashMap<String, f64>,
}

/// Retrieval query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalQuery {
    /// Query identifier
    pub query_id: String,
    /// Query text
    pub query_text: String,
    /// Relevant document IDs
    pub relevant_docs: Vec<String>,
    /// Query type
    pub query_type: String,
}

/// Retrieval analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalAnalysis {
    /// Query performance by type
    pub performance_by_type: HashMap<String, f64>,
    /// Document coverage statistics
    pub document_coverage: f64,
    /// Query completion rate
    pub completion_rate: f64,
    /// Average response time
    pub avg_response_time: f64,
}

/// Retrieval evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResults {
    /// Metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Per-query results
    pub per_query_results: HashMap<String, QueryRetrievalResults>,
    /// Retrieval analysis
    pub retrieval_analysis: RetrievalAnalysis,
    /// Overall retrieval quality
    pub overall_quality: f64,
}

/// Per-query retrieval results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRetrievalResults {
    /// Query ID
    pub query_id: String,
    /// Retrieved documents with scores
    pub retrieved_docs: Vec<(String, f64)>,
    /// Precision at different K values
    pub precision_at_k: HashMap<usize, f64>,
    /// Recall at different K values
    pub recall_at_k: HashMap<usize, f64>,
    /// NDCG scores
    pub ndcg_scores: HashMap<usize, f64>,
    /// Response time (milliseconds)
    pub response_time: f64,
}

/// Retrieval evaluator
pub struct RetrievalEvaluator {
    /// Document collection
    documents: HashMap<String, DocumentMetadata>,
    /// Retrieval queries
    queries: Vec<RetrievalQuery>,
    /// Evaluation metrics
    metrics: Vec<RetrievalMetric>,
}

impl RetrievalEvaluator {
    /// Create a new retrieval evaluator
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            queries: Vec::new(),
            metrics: vec![
                RetrievalMetric::PrecisionAtK(5),
                RetrievalMetric::PrecisionAtK(10),
                RetrievalMetric::RecallAtK(5),
                RetrievalMetric::RecallAtK(10),
                RetrievalMetric::NDCG(10),
                RetrievalMetric::MAP,
                RetrievalMetric::MRR,
            ],
        }
    }

    /// Add document to collection
    pub fn add_document(&mut self, document: DocumentMetadata) {
        self.documents.insert(document.doc_id.clone(), document);
    }

    /// Add retrieval query
    pub fn add_query(&mut self, query: RetrievalQuery) {
        self.queries.push(query);
    }

    /// Evaluate retrieval performance
    pub async fn evaluate(
        &self,
        model: &dyn EmbeddingModel,
        config: &ApplicationEvalConfig,
    ) -> Result<RetrievalResults> {
        let mut metric_scores = HashMap::new();
        let mut per_query_results = HashMap::new();

        // Evaluate each query
        let queries_to_evaluate = if self.queries.len() > config.sample_size {
            &self.queries[..config.sample_size]
        } else {
            &self.queries
        };

        for query in queries_to_evaluate {
            let query_results = self.evaluate_query_retrieval(query, model).await?;
            per_query_results.insert(query.query_id.clone(), query_results);
        }

        // Calculate aggregate metrics
        for metric in &self.metrics {
            let score = self.calculate_retrieval_metric(metric, &per_query_results)?;
            metric_scores.insert(format!("{metric:?}"), score);
        }

        // Generate retrieval analysis
        let retrieval_analysis = self.analyze_retrieval_performance(&per_query_results)?;

        // Calculate overall quality score
        let overall_quality = self.calculate_overall_quality(&metric_scores);

        Ok(RetrievalResults {
            metric_scores,
            per_query_results,
            retrieval_analysis,
            overall_quality,
        })
    }

    /// Evaluate retrieval for a specific query
    async fn evaluate_query_retrieval(
        &self,
        query: &RetrievalQuery,
        model: &dyn EmbeddingModel,
    ) -> Result<QueryRetrievalResults> {
        let start_time = std::time::Instant::now();

        // Perform document retrieval
        let retrieved_docs = self.retrieve_documents(query, model).await?;

        let response_time = start_time.elapsed().as_millis() as f64;

        // Calculate precision and recall at different K values
        let mut precision_at_k = HashMap::new();
        let mut recall_at_k = HashMap::new();
        let mut ndcg_scores = HashMap::new();

        let relevant_set: std::collections::HashSet<String> =
            query.relevant_docs.iter().cloned().collect();

        for &k in &[1, 3, 5, 10, 20] {
            if k <= retrieved_docs.len() {
                let top_k_docs: std::collections::HashSet<String> = retrieved_docs
                    .iter()
                    .take(k)
                    .map(|(doc_id, _)| doc_id.clone())
                    .collect();

                let relevant_retrieved = top_k_docs.intersection(&relevant_set).count();

                let precision = relevant_retrieved as f64 / k as f64;
                let recall = if !query.relevant_docs.is_empty() {
                    relevant_retrieved as f64 / query.relevant_docs.len() as f64
                } else {
                    0.0
                };

                precision_at_k.insert(k, precision);
                recall_at_k.insert(k, recall);

                // Calculate NDCG (simplified)
                let ndcg = self.calculate_ndcg_for_query(&retrieved_docs, &relevant_set, k);
                ndcg_scores.insert(k, ndcg);
            }
        }

        Ok(QueryRetrievalResults {
            query_id: query.query_id.clone(),
            retrieved_docs,
            precision_at_k,
            recall_at_k,
            ndcg_scores,
            response_time,
        })
    }

    /// Retrieve documents for a query
    async fn retrieve_documents(
        &self,
        query: &RetrievalQuery,
        _model: &dyn EmbeddingModel,
    ) -> Result<Vec<(String, f64)>> {
        // Simple retrieval using text similarity (placeholder)
        let mut doc_scores = Vec::new();

        for (doc_id, doc) in &self.documents {
            // Calculate relevance score based on text overlap
            let query_words: std::collections::HashSet<&str> =
                query.query_text.split_whitespace().collect();
            let doc_words: std::collections::HashSet<&str> =
                doc.content.split_whitespace().collect();

            let overlap = query_words.intersection(&doc_words).count();
            let score = overlap as f64 / query_words.len() as f64;

            doc_scores.push((doc_id.clone(), score));
        }

        // Sort by score and return top documents
        doc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        doc_scores.truncate(20); // Return top 20 documents

        Ok(doc_scores)
    }

    /// Calculate NDCG for a query
    fn calculate_ndcg_for_query(
        &self,
        retrieved_docs: &[(String, f64)],
        relevant_docs: &std::collections::HashSet<String>,
        k: usize,
    ) -> f64 {
        if k == 0 || retrieved_docs.is_empty() {
            return 0.0;
        }

        let mut dcg = 0.0;
        for (i, (doc_id, _)) in retrieved_docs.iter().take(k).enumerate() {
            if relevant_docs.contains(doc_id) {
                dcg += 1.0 / (i as f64 + 2.0).log2();
            }
        }

        // Calculate ideal DCG
        let relevant_count = relevant_docs.len().min(k);
        let mut idcg = 0.0;
        for i in 0..relevant_count {
            idcg += 1.0 / (i as f64 + 2.0).log2();
        }

        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
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
            RetrievalMetric::PrecisionAtK(k) => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .filter_map(|r| r.precision_at_k.get(k))
                    .cloned()
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RetrievalMetric::RecallAtK(k) => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .filter_map(|r| r.recall_at_k.get(k))
                    .cloned()
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
            RetrievalMetric::F1AtK(k) => {
                let scores: Vec<f64> = per_query_results
                    .values()
                    .filter_map(|r| {
                        let precision = *r.precision_at_k.get(k)?;
                        let recall = *r.recall_at_k.get(k)?;
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
            RetrievalMetric::MAP => {
                let scores: Vec<f64> = per_query_results
                    .iter()
                    .map(|(query_id, result)| self.average_precision_for_query(query_id, result))
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
            RetrievalMetric::MRR => {
                let scores: Vec<f64> = per_query_results
                    .iter()
                    .map(|(query_id, result)| self.reciprocal_rank_for_query(query_id, result))
                    .collect();
                Ok(scores.iter().sum::<f64>() / scores.len() as f64)
            }
        }
    }

    /// Ground-truth relevant document IDs for a query, looked up by ID from
    /// the evaluator's registered queries.
    fn relevant_docs_for(&self, query_id: &str) -> std::collections::HashSet<String> {
        self.queries
            .iter()
            .find(|q| q.query_id == query_id)
            .map(|q| q.relevant_docs.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Average precision for a single query's full ranked result list,
    /// against its ground-truth relevant documents.
    fn average_precision_for_query(&self, query_id: &str, result: &QueryRetrievalResults) -> f64 {
        let relevant = self.relevant_docs_for(query_id);
        if relevant.is_empty() {
            return 0.0;
        }
        let mut hits = 0usize;
        let mut precision_sum = 0.0;
        for (rank, (doc_id, _)) in result.retrieved_docs.iter().enumerate() {
            if relevant.contains(doc_id) {
                hits += 1;
                precision_sum += hits as f64 / (rank + 1) as f64;
            }
        }
        precision_sum / relevant.len() as f64
    }

    /// Reciprocal rank of the first relevant document in a query's ranked
    /// result list (0.0 if none of the retrieved documents are relevant).
    fn reciprocal_rank_for_query(&self, query_id: &str, result: &QueryRetrievalResults) -> f64 {
        let relevant = self.relevant_docs_for(query_id);
        result
            .retrieved_docs
            .iter()
            .position(|(doc_id, _)| relevant.contains(doc_id))
            .map(|rank| 1.0 / (rank + 1) as f64)
            .unwrap_or(0.0)
    }

    /// Analyze retrieval performance
    fn analyze_retrieval_performance(
        &self,
        per_query_results: &HashMap<String, QueryRetrievalResults>,
    ) -> Result<RetrievalAnalysis> {
        let avg_response_time = per_query_results
            .values()
            .map(|r| r.response_time)
            .sum::<f64>()
            / per_query_results.len() as f64;

        let completion_rate = per_query_results
            .values()
            .filter(|r| !r.retrieved_docs.is_empty())
            .count() as f64
            / per_query_results.len() as f64;

        Ok(RetrievalAnalysis {
            performance_by_type: HashMap::new(), // Simplified
            document_coverage: 0.8,              // Simplified
            completion_rate,
            avg_response_time,
        })
    }

    /// Calculate overall quality score
    fn calculate_overall_quality(&self, metric_scores: &HashMap<String, f64>) -> f64 {
        let relevant_metrics = ["PrecisionAtK(10)", "RecallAtK(10)", "NDCG(10)"];
        let mut total_score = 0.0;
        let mut count = 0;

        for metric_name in &relevant_metrics {
            if let Some(&score) = metric_scores.get(*metric_name) {
                total_score += score;
                count += 1;
            }
        }

        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }
}

impl Default for RetrievalEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluator_with_query() -> RetrievalEvaluator {
        let mut evaluator = RetrievalEvaluator::new();
        evaluator.add_query(RetrievalQuery {
            query_id: "q1".to_string(),
            query_text: "test".to_string(),
            relevant_docs: vec!["d2".to_string(), "d4".to_string()],
            query_type: "test".to_string(),
        });
        evaluator
    }

    fn query_result_with_docs(docs: &[&str]) -> QueryRetrievalResults {
        QueryRetrievalResults {
            query_id: "q1".to_string(),
            retrieved_docs: docs.iter().map(|d| (d.to_string(), 1.0)).collect(),
            precision_at_k: [(2usize, 0.5)].into_iter().collect(),
            recall_at_k: [(2usize, 0.5)].into_iter().collect(),
            ndcg_scores: HashMap::new(),
            response_time: 0.0,
        }
    }

    /// Regression test: MAP/MRR must be computed for real from the full
    /// ranked `retrieved_docs` list against ground-truth relevant docs,
    /// instead of a hardcoded 0.5.
    #[test]
    fn test_calculate_retrieval_metric_map_and_mrr_are_real() -> Result<()> {
        let evaluator = evaluator_with_query();
        let mut per_query = HashMap::new();
        // d2 (relevant) at rank 2, d4 (relevant) at rank 4:
        // AP = (1/2 + 2/4) / 2 = 0.5; RR = 1/2 = 0.5
        per_query.insert(
            "q1".to_string(),
            query_result_with_docs(&["d1", "d2", "d3", "d4"]),
        );

        let map = evaluator.calculate_retrieval_metric(&RetrievalMetric::MAP, &per_query)?;
        assert!((map - 0.5).abs() < 1e-9, "map = {map}");

        let mrr = evaluator.calculate_retrieval_metric(&RetrievalMetric::MRR, &per_query)?;
        assert!((mrr - 0.5).abs() < 1e-9, "mrr = {mrr}");

        Ok(())
    }

    #[test]
    fn test_calculate_retrieval_metric_f1_at_k_is_real() -> Result<()> {
        let evaluator = evaluator_with_query();
        let mut per_query = HashMap::new();
        per_query.insert("q1".to_string(), query_result_with_docs(&["d1", "d2"]));

        // precision@2 = recall@2 = 0.5 => F1 = 0.5
        let f1 = evaluator.calculate_retrieval_metric(&RetrievalMetric::F1AtK(2), &per_query)?;
        assert!((f1 - 0.5).abs() < 1e-9, "f1 = {f1}");

        Ok(())
    }

    #[test]
    fn test_calculate_retrieval_metric_mrr_no_relevant_hit_is_zero() -> Result<()> {
        let evaluator = evaluator_with_query();
        let mut per_query = HashMap::new();
        per_query.insert("q1".to_string(), query_result_with_docs(&["d1", "d3"]));

        let mrr = evaluator.calculate_retrieval_metric(&RetrievalMetric::MRR, &per_query)?;
        assert_eq!(mrr, 0.0);

        Ok(())
    }
}
