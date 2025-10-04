//! TransE: Translating Embeddings for Modeling Multi-relational Data
//!
//! TransE models relations as translations in the embedding space:
//! h + r ≈ t for a true triple (h, r, t)
//!
//! Reference: Bordes et al. "Translating Embeddings for Modeling Multi-relational Data" (2013)

use crate::models::{common::*, BaseModel};
use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use scirs2_core::ndarray_ext::{Array1, Array2};
#[allow(unused_imports)]
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, SubAssign};
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

/// TransE embedding model
#[derive(Debug)]
pub struct TransE {
    /// Base model functionality
    base: BaseModel,
    /// Entity embeddings matrix (num_entities × dimensions)
    entity_embeddings: Array2<f64>,
    /// Relation embeddings matrix (num_relations × dimensions)
    relation_embeddings: Array2<f64>,
    /// Whether embeddings have been initialized
    embeddings_initialized: bool,
    /// Distance metric for scoring
    distance_metric: DistanceMetric,
    /// Margin for ranking loss
    margin: f64,
}

/// Distance metrics supported by TransE
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// L1 (Manhattan) distance
    L1,
    /// L2 (Euclidean) distance
    L2,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
}

impl TransE {
    /// Create a new TransE model
    pub fn new(config: ModelConfig) -> Self {
        let base = BaseModel::new(config.clone());

        // Get TransE-specific parameters
        let distance_metric = match config.model_params.get("distance_metric") {
            Some(0.0) => DistanceMetric::L1,
            Some(1.0) => DistanceMetric::L2,
            Some(2.0) => DistanceMetric::Cosine,
            _ => DistanceMetric::L2, // Default to L2
        };

        let margin = config.model_params.get("margin").copied().unwrap_or(1.0);

        Self {
            base,
            entity_embeddings: Array2::zeros((0, config.dimensions)),
            relation_embeddings: Array2::zeros((0, config.dimensions)),
            embeddings_initialized: false,
            distance_metric,
            margin,
        }
    }

    /// Create a new TransE model with L1 (Manhattan) distance metric
    pub fn with_l1_distance(mut config: ModelConfig) -> Self {
        config
            .model_params
            .insert("distance_metric".to_string(), 0.0);
        Self::new(config)
    }

    /// Create a new TransE model with L2 (Euclidean) distance metric
    pub fn with_l2_distance(mut config: ModelConfig) -> Self {
        config
            .model_params
            .insert("distance_metric".to_string(), 1.0);
        Self::new(config)
    }

    /// Create a new TransE model with Cosine distance metric
    pub fn with_cosine_distance(mut config: ModelConfig) -> Self {
        config
            .model_params
            .insert("distance_metric".to_string(), 2.0);
        Self::new(config)
    }

    /// Create a new TransE model with custom margin for ranking loss
    pub fn with_margin(mut config: ModelConfig, margin: f64) -> Self {
        config.model_params.insert("margin".to_string(), margin);
        Self::new(config)
    }

    /// Get the current distance metric
    pub fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    /// Get the current margin value
    pub fn margin(&self) -> f64 {
        self.margin
    }

    /// Initialize embeddings after entities and relations are known
    fn initialize_embeddings(&mut self) {
        if self.embeddings_initialized {
            return;
        }

        let num_entities = self.base.num_entities();
        let num_relations = self.base.num_relations();
        let dimensions = self.base.config.dimensions;

        if num_entities == 0 || num_relations == 0 {
            return;
        }

        let mut rng = Random::default();

        // Initialize entity embeddings with Xavier initialization
        self.entity_embeddings =
            xavier_init((num_entities, dimensions), dimensions, dimensions, &mut rng);

        // Initialize relation embeddings with Xavier initialization
        self.relation_embeddings = xavier_init(
            (num_relations, dimensions),
            dimensions,
            dimensions,
            &mut rng,
        );

        // Normalize entity embeddings to unit sphere
        normalize_embeddings(&mut self.entity_embeddings);

        self.embeddings_initialized = true;
        debug!(
            "Initialized TransE embeddings: {} entities, {} relations, {} dimensions",
            num_entities, num_relations, dimensions
        );
    }

    /// Score a triple using TransE scoring function
    fn score_triple_ids(
        &self,
        subject_id: usize,
        predicate_id: usize,
        object_id: usize,
    ) -> Result<f64> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }

        let h = self.entity_embeddings.row(subject_id);
        let r = self.relation_embeddings.row(predicate_id);
        let t = self.entity_embeddings.row(object_id);

        // Compute h + r - t
        let diff = &h + &r - t;

        // Distance metric determines scoring (lower distance = higher score)
        let distance = match self.distance_metric {
            DistanceMetric::L1 => diff.mapv(|x| x.abs()).sum(),
            DistanceMetric::L2 => diff.mapv(|x| x * x).sum().sqrt(),
            DistanceMetric::Cosine => {
                // For cosine distance, we compute 1 - cosine_similarity(h + r, t)
                let h_plus_r = &h + &r;
                let dot_product = (&h_plus_r * &t).sum();
                let norm_h_plus_r = h_plus_r.mapv(|x| x * x).sum().sqrt();
                let norm_t = t.mapv(|x| x * x).sum().sqrt();

                if norm_h_plus_r == 0.0 || norm_t == 0.0 {
                    1.0 // Maximum distance for zero vectors
                } else {
                    let cosine_sim = dot_product / (norm_h_plus_r * norm_t);
                    1.0 - cosine_sim.clamp(-1.0, 1.0) // Clamp to [-1, 1] and convert to distance
                }
            }
        };

        // Return negative distance as score (higher is better)
        Ok(-distance)
    }

    /// Compute gradients for a training triple
    fn compute_gradients(
        &self,
        pos_triple: (usize, usize, usize),
        neg_triple: (usize, usize, usize),
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (pos_s, pos_p, pos_o) = pos_triple;
        let (neg_s, neg_p, neg_o) = neg_triple;

        let mut entity_grads = Array2::zeros(self.entity_embeddings.raw_dim());
        let mut relation_grads = Array2::zeros(self.relation_embeddings.raw_dim());

        // Get embeddings
        let pos_h = self.entity_embeddings.row(pos_s);
        let pos_r = self.relation_embeddings.row(pos_p);
        let pos_t = self.entity_embeddings.row(pos_o);

        let neg_h = self.entity_embeddings.row(neg_s);
        let neg_r = self.relation_embeddings.row(neg_p);
        let neg_t = self.entity_embeddings.row(neg_o);

        // Compute differences
        let pos_diff = &pos_h + &pos_r - pos_t;
        let neg_diff = &neg_h + &neg_r - neg_t;

        // Compute distances
        let pos_distance = match self.distance_metric {
            DistanceMetric::L1 => pos_diff.mapv(|x| x.abs()).sum(),
            DistanceMetric::L2 => pos_diff.mapv(|x| x * x).sum().sqrt(),
            DistanceMetric::Cosine => {
                let norm = pos_diff.mapv(|x| x * x).sum().sqrt();
                if norm > 1e-10 {
                    1.0 - (pos_diff.dot(&pos_diff) / (norm * norm)).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            }
        };

        let neg_distance = match self.distance_metric {
            DistanceMetric::L1 => neg_diff.mapv(|x| x.abs()).sum(),
            DistanceMetric::L2 => neg_diff.mapv(|x| x * x).sum().sqrt(),
            DistanceMetric::Cosine => {
                let norm = neg_diff.mapv(|x| x * x).sum().sqrt();
                if norm > 1e-10 {
                    1.0 - (neg_diff.dot(&neg_diff) / (norm * norm)).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            }
        };

        // Check if we need to update (margin loss > 0)
        let loss = self.margin + pos_distance - neg_distance;
        if loss > 0.0 {
            // Compute gradient direction based on distance metric
            let pos_grad_direction = match self.distance_metric {
                DistanceMetric::L1 => pos_diff.mapv(|x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }),
                DistanceMetric::L2 => {
                    if pos_distance > 1e-10 {
                        &pos_diff / pos_distance
                    } else {
                        Array1::zeros(pos_diff.len())
                    }
                }
                DistanceMetric::Cosine => {
                    let norm_sq = pos_diff.mapv(|x| x * x).sum();
                    if norm_sq > 1e-10 {
                        &pos_diff / norm_sq.sqrt()
                    } else {
                        Array1::zeros(pos_diff.len())
                    }
                }
            };

            let neg_grad_direction = match self.distance_metric {
                DistanceMetric::L1 => neg_diff.mapv(|x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }),
                DistanceMetric::L2 => {
                    if neg_distance > 1e-10 {
                        &neg_diff / neg_distance
                    } else {
                        Array1::zeros(neg_diff.len())
                    }
                }
                DistanceMetric::Cosine => {
                    let norm_sq = neg_diff.mapv(|x| x * x).sum();
                    if norm_sq > 1e-10 {
                        &neg_diff / norm_sq.sqrt()
                    } else {
                        Array1::zeros(neg_diff.len())
                    }
                }
            };

            // Update gradients for positive triple (increase distance)
            entity_grads.row_mut(pos_s).add_assign(&pos_grad_direction);
            relation_grads
                .row_mut(pos_p)
                .add_assign(&pos_grad_direction);
            entity_grads.row_mut(pos_o).sub_assign(&pos_grad_direction);

            // Update gradients for negative triple (decrease distance)
            entity_grads.row_mut(neg_s).sub_assign(&neg_grad_direction);
            relation_grads
                .row_mut(neg_p)
                .sub_assign(&neg_grad_direction);
            entity_grads.row_mut(neg_o).add_assign(&neg_grad_direction);
        }

        Ok((entity_grads, relation_grads))
    }

    /// Perform one training epoch
    async fn train_epoch(&mut self, learning_rate: f64) -> Result<f64> {
        let mut rng = Random::default();

        let mut total_loss = 0.0;
        let num_batches = (self.base.triples.len() + self.base.config.batch_size - 1)
            / self.base.config.batch_size;

        // Create shuffled batches
        let mut shuffled_triples = self.base.triples.clone();
        // Manual Fisher-Yates shuffle using scirs2-core
        for i in (1..shuffled_triples.len()).rev() {
            let j = rng.random_range(0, i + 1);
            shuffled_triples.swap(i, j);
        }

        for batch_triples in shuffled_triples.chunks(self.base.config.batch_size) {
            let mut batch_entity_grads = Array2::zeros(self.entity_embeddings.raw_dim());
            let mut batch_relation_grads = Array2::zeros(self.relation_embeddings.raw_dim());
            let mut batch_loss = 0.0;

            for &pos_triple in batch_triples {
                // Generate negative samples
                let neg_samples = self
                    .base
                    .generate_negative_samples(self.base.config.negative_samples, &mut rng);

                for neg_triple in neg_samples {
                    // Compute scores
                    let pos_score =
                        self.score_triple_ids(pos_triple.0, pos_triple.1, pos_triple.2)?;
                    let neg_score =
                        self.score_triple_ids(neg_triple.0, neg_triple.1, neg_triple.2)?;

                    // Convert scores to distances (negate because score = -distance)
                    let pos_distance = -pos_score;
                    let neg_distance = -neg_score;

                    // Compute margin loss
                    let loss = margin_loss(pos_distance, neg_distance, self.margin);
                    batch_loss += loss;

                    if loss > 0.0 {
                        // Compute and accumulate gradients
                        let (entity_grads, relation_grads) =
                            self.compute_gradients(pos_triple, neg_triple)?;
                        batch_entity_grads += &entity_grads;
                        batch_relation_grads += &relation_grads;
                    }
                }
            }

            // Apply gradients
            if batch_loss > 0.0 {
                gradient_update(
                    &mut self.entity_embeddings,
                    &batch_entity_grads,
                    learning_rate,
                    self.base.config.l2_reg,
                );

                gradient_update(
                    &mut self.relation_embeddings,
                    &batch_relation_grads,
                    learning_rate,
                    self.base.config.l2_reg,
                );

                // Normalize entity embeddings
                normalize_embeddings(&mut self.entity_embeddings);
            }

            total_loss += batch_loss;
        }

        Ok(total_loss / num_batches as f64)
    }
}

#[async_trait]
impl EmbeddingModel for TransE {
    fn config(&self) -> &ModelConfig {
        &self.base.config
    }

    fn model_id(&self) -> &Uuid {
        &self.base.model_id
    }

    fn model_type(&self) -> &'static str {
        "TransE"
    }

    fn add_triple(&mut self, triple: Triple) -> Result<()> {
        self.base.add_triple(triple)
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let start_time = Instant::now();
        let max_epochs = epochs.unwrap_or(self.base.config.max_epochs);

        // Initialize embeddings if needed
        self.initialize_embeddings();

        if !self.embeddings_initialized {
            return Err(anyhow!("No training data available"));
        }

        let mut loss_history = Vec::new();
        let learning_rate = self.base.config.learning_rate;

        info!("Starting TransE training for {} epochs", max_epochs);

        for epoch in 0..max_epochs {
            let epoch_loss = self.train_epoch(learning_rate).await?;
            loss_history.push(epoch_loss);

            if epoch % 100 == 0 {
                debug!("Epoch {}: loss = {:.6}", epoch, epoch_loss);
            }

            // Simple convergence check
            if epoch > 10 && epoch_loss < 1e-6 {
                info!("Converged at epoch {} with loss {:.6}", epoch, epoch_loss);
                break;
            }
        }

        self.base.mark_trained();
        let training_time = start_time.elapsed().as_secs_f64();

        Ok(TrainingStats {
            epochs_completed: loss_history.len(),
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            training_time_seconds: training_time,
            convergence_achieved: loss_history.last().copied().unwrap_or(f64::INFINITY) < 1e-6,
            loss_history,
        })
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }

        let entity_id = self
            .base
            .get_entity_id(entity)
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))?;

        let embedding = self.entity_embeddings.row(entity_id).to_owned();
        Ok(ndarray_to_vector(&embedding))
    }

    fn getrelation_embedding(&self, relation: &str) -> Result<Vector> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }

        let relation_id = self
            .base
            .get_relation_id(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;

        let embedding = self.relation_embeddings.row(relation_id).to_owned();
        Ok(ndarray_to_vector(&embedding))
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let subject_id = self
            .base
            .get_entity_id(subject)
            .ok_or_else(|| anyhow!("Subject not found: {}", subject))?;
        let predicate_id = self
            .base
            .get_relation_id(predicate)
            .ok_or_else(|| anyhow!("Predicate not found: {}", predicate))?;
        let object_id = self
            .base
            .get_entity_id(object)
            .ok_or_else(|| anyhow!("Object not found: {}", object))?;

        self.score_triple_ids(subject_id, predicate_id, object_id)
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }

        let subject_id = self
            .base
            .get_entity_id(subject)
            .ok_or_else(|| anyhow!("Subject not found: {}", subject))?;
        let predicate_id = self
            .base
            .get_relation_id(predicate)
            .ok_or_else(|| anyhow!("Predicate not found: {}", predicate))?;

        let mut scores = Vec::new();

        for object_id in 0..self.base.num_entities() {
            let score = self.score_triple_ids(subject_id, predicate_id, object_id)?;
            let object_name = self.base.get_entity(object_id).unwrap().clone();
            scores.push((object_name, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);

        Ok(scores)
    }

    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }

        let predicate_id = self
            .base
            .get_relation_id(predicate)
            .ok_or_else(|| anyhow!("Predicate not found: {}", predicate))?;
        let object_id = self
            .base
            .get_entity_id(object)
            .ok_or_else(|| anyhow!("Object not found: {}", object))?;

        let mut scores = Vec::new();

        for subject_id in 0..self.base.num_entities() {
            let score = self.score_triple_ids(subject_id, predicate_id, object_id)?;
            let subject_name = self.base.get_entity(subject_id).unwrap().clone();
            scores.push((subject_name, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);

        Ok(scores)
    }

    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }

        let subject_id = self
            .base
            .get_entity_id(subject)
            .ok_or_else(|| anyhow!("Subject not found: {}", subject))?;
        let object_id = self
            .base
            .get_entity_id(object)
            .ok_or_else(|| anyhow!("Object not found: {}", object))?;

        let mut scores = Vec::new();

        for predicate_id in 0..self.base.num_relations() {
            let score = self.score_triple_ids(subject_id, predicate_id, object_id)?;
            let predicate_name = self.base.get_relation(predicate_id).unwrap().clone();
            scores.push((predicate_name, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);

        Ok(scores)
    }

    fn get_entities(&self) -> Vec<String> {
        self.base.get_entities()
    }

    fn get_relations(&self) -> Vec<String> {
        self.base.get_relations()
    }

    fn get_stats(&self) -> ModelStats {
        self.base.get_stats("TransE")
    }

    fn save(&self, path: &str) -> Result<()> {
        // For now, just a placeholder
        // In a full implementation, this would serialize the model to file
        info!("Saving TransE model to {}", path);
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        // For now, just a placeholder
        // In a full implementation, this would deserialize the model from file
        info!("Loading TransE model from {}", path);
        Ok(())
    }

    fn clear(&mut self) {
        self.base.clear();
        self.entity_embeddings = Array2::zeros((0, self.base.config.dimensions));
        self.relation_embeddings = Array2::zeros((0, self.base.config.dimensions));
        self.embeddings_initialized = false;
    }

    fn is_trained(&self) -> bool {
        self.base.is_trained
    }

    async fn encode(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(anyhow!(
            "TransE is a knowledge graph embedding model and does not support text encoding"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedNode;

    #[tokio::test]
    async fn test_transe_basic() -> Result<()> {
        let config = ModelConfig::default()
            .with_dimensions(50)
            .with_max_epochs(10)
            .with_seed(42);

        let mut model = TransE::new(config);

        // Add test triples
        let alice = NamedNode::new("http://example.org/alice")?;
        let knows = NamedNode::new("http://example.org/knows")?;
        let bob = NamedNode::new("http://example.org/bob")?;

        model.add_triple(Triple::new(alice.clone(), knows.clone(), bob.clone()))?;
        model.add_triple(Triple::new(bob.clone(), knows.clone(), alice.clone()))?;

        // Train
        let stats = model.train(Some(5)).await?;
        assert!(stats.epochs_completed > 0);

        // Test embeddings
        let alice_emb = model.get_entity_embedding("http://example.org/alice")?;
        assert_eq!(alice_emb.dimensions, 50);

        // Test scoring
        let score = model.score_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        )?;

        // Score should be a finite number
        assert!(score.is_finite());

        Ok(())
    }

    #[tokio::test]
    async fn test_transe_distance_metrics() -> Result<()> {
        let base_config = ModelConfig::default()
            .with_dimensions(10)
            .with_max_epochs(5)
            .with_seed(42);

        // Test L1 distance
        let mut model_l1 = TransE::with_l1_distance(base_config.clone());
        assert!(matches!(model_l1.distance_metric(), DistanceMetric::L1));

        // Test L2 distance
        let mut model_l2 = TransE::with_l2_distance(base_config.clone());
        assert!(matches!(model_l2.distance_metric(), DistanceMetric::L2));

        // Test Cosine distance
        let mut model_cosine = TransE::with_cosine_distance(base_config.clone());
        assert!(matches!(
            model_cosine.distance_metric(),
            DistanceMetric::Cosine
        ));

        // Test custom margin
        let model_margin = TransE::with_margin(base_config.clone(), 2.0);
        assert_eq!(model_margin.margin(), 2.0);

        // Add same triples to all models
        let alice = NamedNode::new("http://example.org/alice")?;
        let knows = NamedNode::new("http://example.org/knows")?;
        let bob = NamedNode::new("http://example.org/bob")?;
        let triple = Triple::new(alice, knows, bob);

        model_l1.add_triple(triple.clone())?;
        model_l2.add_triple(triple.clone())?;
        model_cosine.add_triple(triple.clone())?;

        // Train all models
        model_l1.train(Some(3)).await?;
        model_l2.train(Some(3)).await?;
        model_cosine.train(Some(3)).await?;

        // Test that all models produce finite scores
        let score_l1 = model_l1.score_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        )?;
        let score_l2 = model_l2.score_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        )?;
        let score_cosine = model_cosine.score_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        )?;

        assert!(score_l1.is_finite());
        assert!(score_l2.is_finite());
        assert!(score_cosine.is_finite());

        // Scores may differ due to different distance metrics
        // This tests that the cosine distance implementation works
        println!("L1 score: {score_l1}, L2 score: {score_l2}, Cosine score: {score_cosine}");

        Ok(())
    }
}
