//! Entity Linking and Relation Prediction for Knowledge Graphs
//!
//! This module provides advanced entity linking and relation prediction capabilities
//! using learned embeddings and similarity metrics with full SciRS2 integration.

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Entity linker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityLinkerConfig {
    /// Similarity threshold for entity matching
    pub similarity_threshold: f32,
    /// Maximum number of candidate entities to consider
    pub max_candidates: usize,
    /// Enable context-aware linking
    pub use_context: bool,
    /// Minimum confidence score for linking
    pub min_confidence: f32,
    /// Enable approximate nearest neighbor search
    pub use_ann: bool,
    /// Number of nearest neighbors to retrieve
    pub k_neighbors: usize,
}

impl Default for EntityLinkerConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_candidates: 10,
            use_context: true,
            min_confidence: 0.5,
            use_ann: true,
            k_neighbors: 50,
        }
    }
}

/// Entity linking result with confidence scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingResult {
    /// Linked entity ID
    pub entity_id: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Similarity score
    pub similarity: f32,
    /// Context features used
    pub context_features: Vec<String>,
}

/// Entity linker for knowledge graph entity resolution
pub struct EntityLinker {
    config: EntityLinkerConfig,
    entity_embeddings: Arc<HashMap<String, Array1<f32>>>,
    entity_index: Vec<String>,
    embedding_matrix: Array2<f32>,
}

impl EntityLinker {
    /// Create new entity linker
    pub fn new(
        config: EntityLinkerConfig,
        entity_embeddings: HashMap<String, Array1<f32>>,
    ) -> Result<Self> {
        let entity_count = entity_embeddings.len();
        if entity_count == 0 {
            return Err(anyhow!("Empty entity embedding set"));
        }

        // Build entity index for fast lookup
        let mut entity_index = Vec::with_capacity(entity_count);
        let embedding_dim = entity_embeddings.values().next().unwrap().len();
        let mut embedding_matrix = Array2::zeros((entity_count, embedding_dim));

        for (idx, (entity_id, embedding)) in entity_embeddings.iter().enumerate() {
            entity_index.push(entity_id.clone());
            embedding_matrix.row_mut(idx).assign(embedding);
        }

        info!(
            "Initialized EntityLinker with {} entities, dim={}",
            entity_count, embedding_dim
        );

        Ok(Self {
            config,
            entity_embeddings: Arc::new(entity_embeddings),
            entity_index,
            embedding_matrix,
        })
    }

    /// Link a mention to knowledge graph entities
    pub fn link_entity(
        &self,
        mention_embedding: &Array1<f32>,
        context_embeddings: Option<&[Array1<f32>]>,
    ) -> Result<Vec<LinkingResult>> {
        // Compute similarities with all entities
        let similarities = self.compute_similarities(mention_embedding)?;

        // Get top-k candidates
        let mut candidates: Vec<(usize, f32)> = similarities
            .iter()
            .enumerate()
            .map(|(idx, &sim)| (idx, sim))
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(self.config.max_candidates);

        // Apply context if available
        let results = if self.config.use_context && context_embeddings.is_some() {
            self.rerank_with_context(&candidates, context_embeddings.unwrap())?
        } else {
            candidates
                .into_iter()
                .filter(|(_, sim)| *sim >= self.config.similarity_threshold)
                .map(|(idx, sim)| LinkingResult {
                    entity_id: self.entity_index[idx].clone(),
                    confidence: sim,
                    similarity: sim,
                    context_features: vec![],
                })
                .collect()
        };

        // Filter by minimum confidence
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|r| r.confidence >= self.config.min_confidence)
            .collect();

        debug!("Linked {} candidate entities", filtered.len());

        Ok(filtered)
    }

    /// Batch entity linking for multiple mentions
    pub fn link_entities_batch(
        &self,
        mention_embeddings: &[Array1<f32>],
    ) -> Result<Vec<Vec<LinkingResult>>> {
        // Parallel processing using rayon
        let results: Vec<Vec<LinkingResult>> = mention_embeddings
            .par_iter()
            .map(|mention| self.link_entity(mention, None).unwrap_or_default())
            .collect();

        Ok(results)
    }

    /// Compute cosine similarities efficiently
    fn compute_similarities(&self, query: &Array1<f32>) -> Result<Vec<f32>> {
        // Normalize query
        let query_norm = query.dot(query).sqrt();
        if query_norm == 0.0 {
            return Err(anyhow!("Zero-norm query vector"));
        }

        let normalized_query = query / query_norm;

        // Compute similarities using matrix multiplication
        let similarities: Vec<f32> = (0..self.embedding_matrix.nrows())
            .into_par_iter()
            .map(|i| {
                let entity_emb = self.embedding_matrix.row(i);
                let entity_norm = entity_emb.dot(&entity_emb).sqrt();

                if entity_norm == 0.0 {
                    0.0
                } else {
                    let normalized_entity = entity_emb.to_owned() / entity_norm;
                    normalized_query.dot(&normalized_entity)
                }
            })
            .collect();

        Ok(similarities)
    }

    /// Rerank candidates using context information
    fn rerank_with_context(
        &self,
        candidates: &[(usize, f32)],
        context_embeddings: &[Array1<f32>],
    ) -> Result<Vec<LinkingResult>> {
        let results: Vec<LinkingResult> = candidates
            .iter()
            .map(|(idx, base_sim)| {
                let entity_embedding = self.embedding_matrix.row(*idx);

                // Compute context similarity
                let context_sim = self
                    .compute_context_similarity(&entity_embedding.to_owned(), context_embeddings);

                // Combine base similarity and context similarity
                let confidence = 0.7 * base_sim + 0.3 * context_sim;

                LinkingResult {
                    entity_id: self.entity_index[*idx].clone(),
                    confidence,
                    similarity: *base_sim,
                    context_features: vec!["context_aware".to_string()],
                }
            })
            .collect();

        Ok(results)
    }

    /// Compute context similarity score
    fn compute_context_similarity(
        &self,
        entity_embedding: &Array1<f32>,
        context_embeddings: &[Array1<f32>],
    ) -> f32 {
        if context_embeddings.is_empty() {
            return 0.0;
        }

        // Average similarity with context
        let total_sim: f32 = context_embeddings
            .iter()
            .map(|ctx| {
                let norm1 = entity_embedding.dot(entity_embedding).sqrt();
                let norm2 = ctx.dot(ctx).sqrt();

                if norm1 == 0.0 || norm2 == 0.0 {
                    0.0
                } else {
                    entity_embedding.dot(ctx) / (norm1 * norm2)
                }
            })
            .sum();

        total_sim / context_embeddings.len() as f32
    }

    /// Get entity embedding by ID
    pub fn get_embedding(&self, entity_id: &str) -> Option<&Array1<f32>> {
        self.entity_embeddings.get(entity_id)
    }
}

/// Relation prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationPredictorConfig {
    /// Score threshold for relation prediction
    pub score_threshold: f32,
    /// Maximum number of relations to predict
    pub max_predictions: usize,
    /// Enable type constraints
    pub use_type_constraints: bool,
    /// Enable path-based reasoning
    pub use_path_reasoning: bool,
}

impl Default for RelationPredictorConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.6,
            max_predictions: 10,
            use_type_constraints: true,
            use_path_reasoning: false,
        }
    }
}

/// Relation prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationPrediction {
    /// Predicted relation type
    pub relation: String,
    /// Tail entity (if predicting tails)
    pub tail_entity: Option<String>,
    /// Prediction score
    pub score: f32,
    /// Confidence level
    pub confidence: f32,
}

/// Relation predictor for knowledge graph completion
pub struct RelationPredictor {
    config: RelationPredictorConfig,
    relation_embeddings: Arc<HashMap<String, Array1<f32>>>,
    entity_embeddings: Arc<HashMap<String, Array1<f32>>>,
}

impl RelationPredictor {
    /// Create new relation predictor
    pub fn new(
        config: RelationPredictorConfig,
        relation_embeddings: HashMap<String, Array1<f32>>,
        entity_embeddings: HashMap<String, Array1<f32>>,
    ) -> Self {
        info!(
            "Initialized RelationPredictor with {} relations, {} entities",
            relation_embeddings.len(),
            entity_embeddings.len()
        );

        Self {
            config,
            relation_embeddings: Arc::new(relation_embeddings),
            entity_embeddings: Arc::new(entity_embeddings),
        }
    }

    /// Predict relations between two entities
    pub fn predict_relations(
        &self,
        head_entity: &str,
        tail_entity: &str,
    ) -> Result<Vec<RelationPrediction>> {
        let head_emb = self
            .entity_embeddings
            .get(head_entity)
            .ok_or_else(|| anyhow!("Unknown head entity: {}", head_entity))?;

        let tail_emb = self
            .entity_embeddings
            .get(tail_entity)
            .ok_or_else(|| anyhow!("Unknown tail entity: {}", tail_entity))?;

        // Score all possible relations
        let mut predictions: Vec<RelationPrediction> = self
            .relation_embeddings
            .par_iter()
            .map(|(rel, rel_emb)| {
                // TransE-style scoring: h + r ≈ t
                let score = self.score_triple(head_emb, rel_emb, tail_emb);

                RelationPrediction {
                    relation: rel.clone(),
                    tail_entity: Some(tail_entity.to_string()),
                    score,
                    confidence: score,
                }
            })
            .filter(|pred| pred.score >= self.config.score_threshold)
            .collect();

        // Sort by score descending
        predictions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        predictions.truncate(self.config.max_predictions);

        Ok(predictions)
    }

    /// Predict tail entities for a given head and relation
    pub fn predict_tails(
        &self,
        head_entity: &str,
        relation: &str,
    ) -> Result<Vec<RelationPrediction>> {
        let head_emb = self
            .entity_embeddings
            .get(head_entity)
            .ok_or_else(|| anyhow!("Unknown head entity: {}", head_entity))?;

        let rel_emb = self
            .relation_embeddings
            .get(relation)
            .ok_or_else(|| anyhow!("Unknown relation: {}", relation))?;

        // Compute expected tail embedding: t = h + r
        let expected_tail = head_emb + rel_emb;

        // Find nearest entities to expected tail
        let mut predictions: Vec<RelationPrediction> = self
            .entity_embeddings
            .par_iter()
            .map(|(entity, entity_emb)| {
                let distance = Self::euclidean_distance(&expected_tail, entity_emb);
                let score = 1.0 / (1.0 + distance); // Convert distance to score

                RelationPrediction {
                    relation: relation.to_string(),
                    tail_entity: Some(entity.clone()),
                    score,
                    confidence: score,
                }
            })
            .filter(|pred| pred.score >= self.config.score_threshold)
            .collect();

        predictions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        predictions.truncate(self.config.max_predictions);

        Ok(predictions)
    }

    /// Score a triple using TransE-style scoring
    fn score_triple(&self, head: &Array1<f32>, relation: &Array1<f32>, tail: &Array1<f32>) -> f32 {
        // TransE: score = -||h + r - t||
        let expected_tail = head + relation;
        let distance = Self::euclidean_distance(&expected_tail, tail);

        // Convert to similarity score (higher is better)
        1.0 / (1.0 + distance)
    }

    /// Compute Euclidean distance
    fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let diff = a - b;
        diff.dot(&diff).sqrt()
    }

    /// Batch prediction of tails
    pub fn predict_tails_batch(
        &self,
        queries: &[(String, String)], // (head, relation) pairs
    ) -> Result<Vec<Vec<RelationPrediction>>> {
        let results: Vec<Vec<RelationPrediction>> = queries
            .par_iter()
            .map(|(head, rel)| self.predict_tails(head, rel).unwrap_or_default())
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_entity_linker_creation() {
        let mut embeddings = HashMap::new();
        embeddings.insert("entity1".to_string(), array![0.1, 0.2, 0.3]);
        embeddings.insert("entity2".to_string(), array![0.4, 0.5, 0.6]);

        let config = EntityLinkerConfig::default();
        let linker = EntityLinker::new(config, embeddings);
        assert!(linker.is_ok());
    }

    #[test]
    fn test_entity_linking() {
        let mut embeddings = HashMap::new();
        embeddings.insert("entity1".to_string(), array![1.0, 0.0, 0.0]);
        embeddings.insert("entity2".to_string(), array![0.0, 1.0, 0.0]);
        embeddings.insert("entity3".to_string(), array![0.7, 0.7, 0.0]);

        let config = EntityLinkerConfig {
            similarity_threshold: 0.5,
            ..Default::default()
        };

        let linker = EntityLinker::new(config, embeddings).unwrap();

        // Query similar to entity1
        let query = array![0.9, 0.1, 0.0];
        let results = linker.link_entity(&query, None).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].entity_id, "entity1");
    }

    #[test]
    fn test_relation_predictor_creation() {
        let mut entity_embeddings = HashMap::new();
        entity_embeddings.insert("entity1".to_string(), array![0.1, 0.2, 0.3]);

        let mut relation_embeddings = HashMap::new();
        relation_embeddings.insert("rel1".to_string(), array![0.1, 0.1, 0.1]);

        let config = RelationPredictorConfig::default();
        let predictor = RelationPredictor::new(config, relation_embeddings, entity_embeddings);

        // Just verify creation succeeds
        assert_eq!(predictor.relation_embeddings.len(), 1);
    }

    #[test]
    fn test_batch_entity_linking() {
        let mut embeddings = HashMap::new();
        embeddings.insert("entity1".to_string(), array![1.0, 0.0, 0.0]);
        embeddings.insert("entity2".to_string(), array![0.0, 1.0, 0.0]);

        let config = EntityLinkerConfig::default();
        let linker = EntityLinker::new(config, embeddings).unwrap();

        let queries = vec![array![0.9, 0.1, 0.0], array![0.1, 0.9, 0.0]];

        let results = linker.link_entities_batch(&queries).unwrap();
        assert_eq!(results.len(), 2);
    }
}
