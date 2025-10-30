//! Link Prediction for Knowledge Graph Completion
//!
//! This module provides link prediction functionality for knowledge graphs,
//! enabling entity prediction, relation prediction, and ranking capabilities.
//!
//! Link prediction is a fundamental task in knowledge graph completion where
//! we predict missing links (triples) based on learned embeddings.

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

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
    /// Predict relation: (head, ?, tail)
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
                triple.head.clone(),
                triple.relation.clone(),
                triple.tail.clone(),
            ));
        }
    }

    /// Predict tail entities given head and relation
    pub fn predict_tail(
        &self,
        head: &str,
        relation: &str,
        candidate_entities: &[String],
    ) -> Result<Vec<LinkPrediction>> {
        let head_emb = self.model.get_entity_embedding(head)?;
        let rel_emb = self.model.get_relation_embedding(relation)?;

        // Score all candidates
        let scored: Vec<(String, f32)> = if self.config.parallel {
            candidate_entities
                .par_iter()
                .filter_map(|tail| {
                    if self.config.filter_known_triples
                        && self.known_triples.contains(&(
                            head.to_string(),
                            relation.to_string(),
                            tail.clone(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .predict(head, relation, tail)
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
                            head.to_string(),
                            relation.to_string(),
                            tail.clone(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .predict(head, relation, tail)
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
        relation: &str,
        tail: &str,
        candidate_entities: &[String],
    ) -> Result<Vec<LinkPrediction>> {
        let rel_emb = self.model.get_relation_embedding(relation)?;
        let tail_emb = self.model.get_entity_embedding(tail)?;

        // Score all candidates
        let scored: Vec<(String, f32)> = if self.config.parallel {
            candidate_entities
                .par_iter()
                .filter_map(|head| {
                    if self.config.filter_known_triples
                        && self.known_triples.contains(&(
                            head.clone(),
                            relation.to_string(),
                            tail.to_string(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .predict(head, relation, tail)
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
                            relation.to_string(),
                            tail.to_string(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .predict(head, relation, tail)
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
        head: &str,
        tail: &str,
        candidate_relations: &[String],
    ) -> Result<Vec<LinkPrediction>> {
        let head_emb = self.model.get_entity_embedding(head)?;
        let tail_emb = self.model.get_entity_embedding(tail)?;

        // Score all candidate relations
        let scored: Vec<(String, f32)> = if self.config.parallel {
            candidate_relations
                .par_iter()
                .filter_map(|relation| {
                    if self.config.filter_known_triples
                        && self.known_triples.contains(&(
                            head.to_string(),
                            relation.clone(),
                            tail.to_string(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .predict(head, relation, tail)
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
                            head.to_string(),
                            relation.clone(),
                            tail.to_string(),
                        ))
                    {
                        return None;
                    }

                    self.model
                        .predict(head, relation, tail)
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
            if let Ok(predictions) =
                self.predict_tail(&triple.head, &triple.relation, candidate_entities)
            {
                // Find rank of correct tail
                if let Some(rank) = predictions
                    .iter()
                    .position(|pred| pred.predicted_id == triple.tail)
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
    fn rank_and_filter(&self, mut scored: Vec<(String, f32)>) -> Result<Vec<LinkPrediction>> {
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

                if confidence >= self.config.min_confidence {
                    Some(LinkPrediction {
                        predicted_id: id,
                        score,
                        confidence,
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
    use crate::models::transe::{TransE, TransEConfig};
    use crate::ModelConfig;

    #[tokio::test]
    async fn test_link_prediction_tail() {
        let config = TransEConfig {
            base: ModelConfig {
                dimensions: 50,
                learning_rate: 0.01,
                epochs: 50,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut model = TransE::new(config);

        // Add training data
        model
            .add_triple(Triple {
                head: "alice".to_string(),
                relation: "knows".to_string(),
                tail: "bob".to_string(),
            })
            .unwrap();

        model
            .add_triple(Triple {
                head: "alice".to_string(),
                relation: "knows".to_string(),
                tail: "charlie".to_string(),
            })
            .unwrap();

        model
            .add_triple(Triple {
                head: "bob".to_string(),
                relation: "likes".to_string(),
                tail: "dave".to_string(),
            })
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
        let config = TransEConfig {
            base: ModelConfig {
                dimensions: 50,
                epochs: 30,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut model = TransE::new(config);

        model
            .add_triple(Triple {
                head: "a".to_string(),
                relation: "r1".to_string(),
                tail: "b".to_string(),
            })
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
