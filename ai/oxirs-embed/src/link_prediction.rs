//! Link Prediction for Knowledge Graph Completion
//!
//! This module provides comprehensive link prediction functionality for knowledge graphs,
//! enabling entity prediction, relation prediction, ranking, and evaluation capabilities.
//!
//! Link prediction is a fundamental task in knowledge graph completion where we predict
//! missing links (triples) based on learned embeddings. This module supports multiple
//! prediction tasks and provides evaluation metrics following standard benchmarks.
//!
//! # Overview
//!
//! The module provides:
//! - **Tail entity prediction**: Given (head, relation, ?), predict the tail entity
//! - **Head entity prediction**: Given (?, relation, tail), predict the head entity
//! - **Relation prediction**: Given (head, ?, tail), predict the relation
//! - **Batch prediction**: Process multiple queries efficiently in parallel
//! - **Evaluation metrics**: MRR, Hits@K, Mean Rank for benchmarking
//! - **Filtered ranking**: Remove known triples from evaluation
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use oxirs_embed::{
//!     TransE, ModelConfig, Triple, NamedNode, EmbeddingModel,
//!     link_prediction::{LinkPredictor, LinkPredictionConfig},
//! };
//!
//! # async fn example() -> anyhow::Result<()> {
//! // 1. Train a knowledge graph embedding model
//! let config = ModelConfig::default().with_dimensions(128);
//! let mut model = TransE::new(config);
//!
//! // Add training triples
//! model.add_triple(Triple::new(
//!     NamedNode::new("alice")?,
//!     NamedNode::new("knows")?,
//!     NamedNode::new("bob")?,
//! ))?;
//! model.add_triple(Triple::new(
//!     NamedNode::new("bob")?,
//!     NamedNode::new("knows")?,
//!     NamedNode::new("charlie")?,
//! ))?;
//!
//! // Train the model
//! model.train(Some(100)).await?;
//!
//! // 2. Create link predictor
//! let pred_config = LinkPredictionConfig {
//!     top_k: 10,
//!     min_confidence: 0.5,
//!     filter_known_triples: true,
//!     ..Default::default()
//! };
//! let predictor = LinkPredictor::new(pred_config, model);
//!
//! // 3. Predict tail entities
//! let candidates = vec!["bob".to_string(), "charlie".to_string(), "dave".to_string()];
//! let predictions = predictor.predict_tail("alice", "knows", &candidates)?;
//!
//! for pred in predictions {
//!     println!("Entity: {}, Score: {:.3}, Confidence: {:.3}, Rank: {}",
//!              pred.predicted_id, pred.score, pred.confidence, pred.rank);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Prediction Tasks
//!
//! ## Tail Entity Prediction
//!
//! Given a subject and predicate, predict the most likely objects:
//!
//! ```rust,no_run
//! # use oxirs_embed::{TransE, ModelConfig, link_prediction::{LinkPredictor, LinkPredictionConfig}};
//! # async fn example() -> anyhow::Result<()> {
//! # let model = TransE::new(ModelConfig::default());
//! # let predictor = LinkPredictor::new(LinkPredictionConfig::default(), model);
//! let candidates = vec!["paris".to_string(), "london".to_string(), "berlin".to_string()];
//! let predictions = predictor.predict_tail("france", "has_capital", &candidates)?;
//! // Expected: "paris" should rank first with high confidence
//! # Ok(())
//! # }
//! ```
//!
//! ## Head Entity Prediction
//!
//! Given a predicate and object, predict the most likely subjects:
//!
//! ```rust,no_run
//! # use oxirs_embed::{TransE, ModelConfig, link_prediction::{LinkPredictor, LinkPredictionConfig}};
//! # async fn example() -> anyhow::Result<()> {
//! # let model = TransE::new(ModelConfig::default());
//! # let predictor = LinkPredictor::new(LinkPredictionConfig::default(), model);
//! let candidates = vec!["france".to_string(), "germany".to_string(), "uk".to_string()];
//! let predictions = predictor.predict_head("has_capital", "paris", &candidates)?;
//! // Expected: "france" should rank first
//! # Ok(())
//! # }
//! ```
//!
//! ## Relation Prediction
//!
//! Given a subject and object, predict the most likely relations:
//!
//! ```rust,no_run
//! # use oxirs_embed::{TransE, ModelConfig, link_prediction::{LinkPredictor, LinkPredictionConfig}};
//! # async fn example() -> anyhow::Result<()> {
//! # let model = TransE::new(ModelConfig::default());
//! # let predictor = LinkPredictor::new(LinkPredictionConfig::default(), model);
//! let candidates = vec!["has_capital".to_string(), "located_in".to_string()];
//! let predictions = predictor.predict_relation("france", "paris", &candidates)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Batch Processing
//!
//! For efficient processing of multiple queries:
//!
//! ```rust,no_run
//! # use oxirs_embed::{TransE, ModelConfig, link_prediction::{LinkPredictor, LinkPredictionConfig}};
//! # async fn example() -> anyhow::Result<()> {
//! # let model = TransE::new(ModelConfig::default());
//! # let predictor = LinkPredictor::new(LinkPredictionConfig::default(), model);
//! let queries = vec![
//!     ("france".to_string(), "has_capital".to_string()),
//!     ("germany".to_string(), "has_capital".to_string()),
//! ];
//! let candidates = vec!["paris".to_string(), "berlin".to_string()];
//! let batch_results = predictor.predict_tails_batch(&queries, &candidates)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Evaluation
//!
//! Evaluate link prediction performance on a test set:
//!
//! ```rust,no_run
//! # use oxirs_embed::{TransE, ModelConfig, Triple, NamedNode, link_prediction::{LinkPredictor, LinkPredictionConfig}};
//! # async fn example() -> anyhow::Result<()> {
//! # let model = TransE::new(ModelConfig::default());
//! # let predictor = LinkPredictor::new(LinkPredictionConfig::default(), model);
//! let test_triples = vec![
//!     Triple::new(
//!         NamedNode::new("france")?,
//!         NamedNode::new("has_capital")?,
//!         NamedNode::new("paris")?,
//!     ),
//! ];
//! let candidates = vec!["paris".to_string(), "london".to_string()];
//!
//! let metrics = predictor.evaluate(&test_triples, &candidates)?;
//! println!("Mean Rank: {:.2}", metrics.mean_rank);
//! println!("MRR: {:.4}", metrics.mrr);
//! println!("Hits@1: {:.4}", metrics.hits_at_1);
//! println!("Hits@10: {:.4}", metrics.hits_at_10);
//! # Ok(())
//! # }
//! ```
//!
//! # Configuration
//!
//! The [`LinkPredictionConfig`] allows fine-tuning prediction behavior:
//!
//! ```rust
//! use oxirs_embed::link_prediction::LinkPredictionConfig;
//!
//! let config = LinkPredictionConfig {
//!     top_k: 10,                      // Return top 10 predictions
//!     min_confidence: 0.5,             // Filter predictions below 50% confidence
//!     filter_known_triples: true,      // Remove known facts from ranking
//!     parallel: true,                  // Use parallel processing
//!     batch_size: 100,                 // Batch size for processing
//! };
//! ```
//!
//! # Evaluation Metrics
//!
//! The module provides standard knowledge graph evaluation metrics:
//!
//! - **Mean Rank (MR)**: Average rank of correct entities (lower is better)
//! - **Mean Reciprocal Rank (MRR)**: Average of 1/rank (higher is better, 0-1)
//! - **Hits@K**: Percentage of correct entities in top-K predictions (higher is better)
//!
//! These metrics are computed following the filtered setting used in standard benchmarks
//! like FB15k-237 and WN18RR, where known triples are removed from ranking to avoid
//! trivial predictions.
//!
//! # Performance Considerations
//!
//! - Enable `parallel: true` for large-scale predictions
//! - Use batch processing for multiple queries
//! - Filter known triples to avoid redundant computation
//! - Adjust `top_k` and `min_confidence` based on application needs
//!
//! # See Also
//!
//! - [`LinkPredictor`]: Main prediction interface
//! - [`LinkPredictionConfig`]: Configuration options
//! - [`LinkPredictionMetrics`]: Evaluation metrics

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::info;

use crate::{EmbeddingModel, Triple};

/// Link prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkPredictionConfig {
    /// Top-K candidates to return
    pub top_k: usize,
    /// Minimum confidence threshold (0.0 to 1.0)
    pub min_confidence: f32,
    /// Use filtering to remove known triples from ranking
    pub filter_known_triples: bool,
    /// Enable parallel processing
    pub parallel: bool,
    /// Batch size for batch predictions
    pub batch_size: usize,
}

impl Default for LinkPredictionConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_confidence: 0.5,
            filter_known_triples: true,
            parallel: true,
            batch_size: 100,
        }
    }
}

/// Link prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkPrediction {
    /// Predicted entity or relation ID
    pub predicted_id: String,
    /// Prediction score (higher is better)
    pub score: f32,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Rank in the candidate list (1-indexed)
    pub rank: usize,
}

/// Link prediction type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionType {
    /// Predict tail entity: (head, relation, ?)
    TailEntity,
    /// Predict head entity: (?, relation, tail)
    HeadEntity,
    /// Predict predicate: (head, ?, tail)
    Relation,
}

/// Evaluation metrics for link prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkPredictionMetrics {
    /// Mean Rank (lower is better)
    pub mean_rank: f32,
    /// Mean Reciprocal Rank (higher is better, 0-1)
    pub mrr: f32,
    /// Hits@1 (percentage, 0-1)
    pub hits_at_1: f32,
    /// Hits@3 (percentage, 0-1)
    pub hits_at_3: f32,
    /// Hits@5 (percentage, 0-1)
    pub hits_at_5: f32,
    /// Hits@10 (percentage, 0-1)
    pub hits_at_10: f32,
    /// Number of predictions evaluated
    pub num_predictions: usize,
}

impl LinkPredictionMetrics {
    /// Create empty metrics
    pub fn new() -> Self {
        Self {
            mean_rank: 0.0,
            mrr: 0.0,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_5: 0.0,
            hits_at_10: 0.0,
            num_predictions: 0,
        }
    }

    /// Update metrics with a new rank
    pub fn update(&mut self, rank: usize) {
        self.num_predictions += 1;
        let n = self.num_predictions as f32;

        // Update mean rank
        self.mean_rank = ((self.mean_rank * (n - 1.0)) + rank as f32) / n;

        // Update MRR
        let reciprocal_rank = 1.0 / rank as f32;
        self.mrr = ((self.mrr * (n - 1.0)) + reciprocal_rank) / n;

        // Update Hits@K
        if rank <= 1 {
            self.hits_at_1 = ((self.hits_at_1 * (n - 1.0)) + 1.0) / n;
        } else {
            self.hits_at_1 = (self.hits_at_1 * (n - 1.0)) / n;
        }

        if rank <= 3 {
            self.hits_at_3 = ((self.hits_at_3 * (n - 1.0)) + 1.0) / n;
        } else {
            self.hits_at_3 = (self.hits_at_3 * (n - 1.0)) / n;
        }

        if rank <= 5 {
            self.hits_at_5 = ((self.hits_at_5 * (n - 1.0)) + 1.0) / n;
        } else {
            self.hits_at_5 = (self.hits_at_5 * (n - 1.0)) / n;
        }

        if rank <= 10 {
            self.hits_at_10 = ((self.hits_at_10 * (n - 1.0)) + 1.0) / n;
        } else {
            self.hits_at_10 = (self.hits_at_10 * (n - 1.0)) / n;
        }
    }
}

impl Default for LinkPredictionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Link predictor for knowledge graph completion
pub struct LinkPredictor<M: EmbeddingModel> {
    config: LinkPredictionConfig,
    model: M,
    known_triples: HashSet<(String, String, String)>,
}

impl<M: EmbeddingModel> LinkPredictor<M> {
    /// Create new link predictor
    pub fn new(config: LinkPredictionConfig, model: M) -> Self {
        Self {
            config,
            model,
            known_triples: HashSet::new(),
        }
    }

    /// Add known triples for filtering
    pub fn add_known_triples(&mut self, triples: &[Triple]) {
        for triple in triples {
            self.known_triples.insert((
                triple.subject.to_string(),
                triple.predicate.to_string(),
                triple.object.to_string(),
            ));
        }
    }

    /// Predict tail entities given head and relation
    pub fn predict_tail(
        &self,
        subject: &str,
        predicate: &str,
        candidate_entities: &[String],
    ) -> Result<Vec<LinkPrediction>> {
        // Score all candidates
        let scored: Vec<(String, f64)> = if self.config.parallel {
            candidate_entities
                .par_iter()
                .filter_map(|tail| {
                    if self.config.filter_known_triples
                        && self.known_triples.contains(&(
                            subject.to_string(),
                            predicate.to_string(),
                            tail.clone(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .score_triple(subject, predicate, tail)
                        .ok()
                        .map(|score| (tail.clone(), score))
                })
                .collect()
        } else {
            candidate_entities
                .iter()
                .filter_map(|tail| {
                    if self.config.filter_known_triples
                        && self.known_triples.contains(&(
                            subject.to_string(),
                            predicate.to_string(),
                            tail.clone(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .score_triple(subject, predicate, tail)
                        .ok()
                        .map(|score| (tail.clone(), score))
                })
                .collect()
        };

        self.rank_and_filter(scored)
    }

    /// Predict head entities given relation and tail
    pub fn predict_head(
        &self,
        predicate: &str,
        object: &str,
        candidate_entities: &[String],
    ) -> Result<Vec<LinkPrediction>> {
        // Score all candidates
        let scored: Vec<(String, f64)> = if self.config.parallel {
            candidate_entities
                .par_iter()
                .filter_map(|head| {
                    if self.config.filter_known_triples
                        && self.known_triples.contains(&(
                            head.clone(),
                            predicate.to_string(),
                            object.to_string(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .score_triple(head, predicate, object)
                        .ok()
                        .map(|score| (head.clone(), score))
                })
                .collect()
        } else {
            candidate_entities
                .iter()
                .filter_map(|head| {
                    if self.config.filter_known_triples
                        && self.known_triples.contains(&(
                            head.clone(),
                            predicate.to_string(),
                            object.to_string(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .score_triple(head, predicate, object)
                        .ok()
                        .map(|score| (head.clone(), score))
                })
                .collect()
        };

        self.rank_and_filter(scored)
    }

    /// Predict relations given head and tail
    pub fn predict_relation(
        &self,
        subject: &str,
        object: &str,
        candidate_relations: &[String],
    ) -> Result<Vec<LinkPrediction>> {
        // Score all candidate relations
        let scored: Vec<(String, f64)> = if self.config.parallel {
            candidate_relations
                .par_iter()
                .filter_map(|relation| {
                    if self.config.filter_known_triples
                        && self.known_triples.contains(&(
                            subject.to_string(),
                            relation.clone(),
                            object.to_string(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .score_triple(subject, relation, object)
                        .ok()
                        .map(|score| (relation.clone(), score))
                })
                .collect()
        } else {
            candidate_relations
                .iter()
                .filter_map(|relation| {
                    if self.config.filter_known_triples
                        && self.known_triples.contains(&(
                            subject.to_string(),
                            relation.clone(),
                            object.to_string(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .score_triple(subject, relation, object)
                        .ok()
                        .map(|score| (relation.clone(), score))
                })
                .collect()
        };

        self.rank_and_filter(scored)
    }

    /// Batch prediction of tails
    pub fn predict_tails_batch(
        &self,
        queries: &[(String, String)], // (head, relation) pairs
        candidate_entities: &[String],
    ) -> Result<Vec<Vec<LinkPrediction>>> {
        queries
            .par_iter()
            .map(|(head, relation)| {
                self.predict_tail(head, relation, candidate_entities)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(Ok)
            .collect()
    }

    /// Evaluate link prediction on a test set
    pub fn evaluate(
        &self,
        test_triples: &[Triple],
        candidate_entities: &[String],
    ) -> Result<LinkPredictionMetrics> {
        let mut metrics = LinkPredictionMetrics::new();

        info!(
            "Evaluating link prediction on {} test triples",
            test_triples.len()
        );

        for triple in test_triples {
            // Predict tail
            if let Ok(predictions) = self.predict_tail(
                &triple.subject.to_string(),
                &triple.predicate.to_string(),
                candidate_entities,
            ) {
                // Find rank of correct tail
                if let Some(rank) = predictions
                    .iter()
                    .position(|pred| pred.predicted_id == triple.object.to_string())
                {
                    metrics.update(rank + 1); // 1-indexed rank
                }
            }
        }

        info!(
            "Evaluation complete: MRR={:.4}, Hits@10={:.4}",
            metrics.mrr, metrics.hits_at_10
        );

        Ok(metrics)
    }

    /// Rank and filter predictions
    fn rank_and_filter(&self, mut scored: Vec<(String, f64)>) -> Result<Vec<LinkPrediction>> {
        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-K
        scored.truncate(self.config.top_k);

        // Normalize scores to confidence (0-1)
        let max_score = scored.first().map(|(_, s)| *s).unwrap_or(1.0);
        let min_score = scored.last().map(|(_, s)| *s).unwrap_or(0.0);
        let score_range = (max_score - min_score).max(1e-10);

        let predictions: Vec<LinkPrediction> = scored
            .into_iter()
            .enumerate()
            .filter_map(|(rank, (id, score))| {
                let confidence = (score - min_score) / score_range;

                if confidence >= self.config.min_confidence as f64 {
                    Some(LinkPrediction {
                        predicted_id: id,
                        score: score as f32,
                        confidence: confidence as f32,
                        rank: rank + 1, // 1-indexed
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(predictions)
    }

    /// Get reference to underlying model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to underlying model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::transe::TransE;
    use crate::{ModelConfig, NamedNode};

    #[tokio::test]
    async fn test_link_prediction_tail() {
        let config = ModelConfig {
            dimensions: 50,
            learning_rate: 0.01,
            max_epochs: 50,
            ..Default::default()
        };

        let mut model = TransE::new(config);

        // Add training data
        model
            .add_triple(Triple::new(
                NamedNode::new("alice").unwrap(),
                NamedNode::new("knows").unwrap(),
                NamedNode::new("bob").unwrap(),
            ))
            .unwrap();

        model
            .add_triple(Triple::new(
                NamedNode::new("alice").unwrap(),
                NamedNode::new("knows").unwrap(),
                NamedNode::new("charlie").unwrap(),
            ))
            .unwrap();

        model
            .add_triple(Triple::new(
                NamedNode::new("bob").unwrap(),
                NamedNode::new("likes").unwrap(),
                NamedNode::new("dave").unwrap(),
            ))
            .unwrap();

        // Train model
        model.train(Some(50)).await.unwrap();

        // Create link predictor
        let pred_config = LinkPredictionConfig {
            top_k: 5,
            filter_known_triples: false,
            ..Default::default()
        };

        let predictor = LinkPredictor::new(pred_config, model);

        // Predict tails
        let candidates = vec!["bob".to_string(), "charlie".to_string(), "dave".to_string()];

        let predictions = predictor
            .predict_tail("alice", "knows", &candidates)
            .unwrap();

        assert!(!predictions.is_empty());
        assert!(predictions.len() <= 5);

        // Check that predictions are ranked
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i].score >= predictions[i + 1].score);
        }
    }

    #[tokio::test]
    async fn test_link_prediction_metrics() {
        let mut metrics = LinkPredictionMetrics::new();

        // Simulate some predictions
        metrics.update(1); // Perfect prediction
        metrics.update(3); // Rank 3
        metrics.update(10); // Rank 10

        assert_eq!(metrics.num_predictions, 3);
        assert!(metrics.mrr > 0.0);
        assert!(metrics.hits_at_1 > 0.0);
        assert!(metrics.hits_at_10 == 1.0); // All within top 10
    }

    #[tokio::test]
    async fn test_batch_prediction() {
        let config = ModelConfig {
            dimensions: 50,
            max_epochs: 30,
            ..Default::default()
        };

        let mut model = TransE::new(config);

        model
            .add_triple(Triple::new(
                NamedNode::new("a").unwrap(),
                NamedNode::new("r1").unwrap(),
                NamedNode::new("b").unwrap(),
            ))
            .unwrap();

        model.train(Some(30)).await.unwrap();

        let predictor = LinkPredictor::new(LinkPredictionConfig::default(), model);

        let queries = vec![("a".to_string(), "r1".to_string())];

        let candidates = vec!["b".to_string()];

        let results = predictor
            .predict_tails_batch(&queries, &candidates)
            .unwrap();

        assert_eq!(results.len(), 1);
    }
}
