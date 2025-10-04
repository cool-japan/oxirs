//! DistMult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases
//!
//! DistMult is a bilinear model that uses element-wise multiplication:
//! score(h, r, t) = h^T diag(r) t = sum(h * r * t)
//!
//! Note: DistMult can only model symmetric relations due to its formulation.
//!
//! Reference: Yang et al. "Embedding Entities and Relations for Learning and Inference in Knowledge Bases" (2014)

use crate::models::{common::*, BaseModel};
use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use scirs2_core::ndarray_ext::{Array1, Array2};
#[allow(unused_imports)]
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;
use std::time::Instant;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// DistMult embedding model
#[derive(Debug)]
pub struct DistMult {
    /// Base model functionality
    base: BaseModel,
    /// Entity embeddings matrix (num_entities × dimensions)
    entity_embeddings: Array2<f64>,
    /// Relation embeddings matrix (num_relations × dimensions)
    relation_embeddings: Array2<f64>,
    /// Whether embeddings have been initialized
    embeddings_initialized: bool,
    /// Whether to apply dropout during training
    #[allow(dead_code)]
    dropout_rate: f64,
    /// Loss function type
    loss_function: LossFunction,
}

/// Loss function types for DistMult
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LossFunction {
    /// Logistic loss (binary cross-entropy)
    Logistic,
    /// Margin-based ranking loss
    MarginRanking,
    /// Squared loss
    SquaredLoss,
}

impl DistMult {
    /// Create a new DistMult model
    pub fn new(config: ModelConfig) -> Self {
        let base = BaseModel::new(config.clone());

        // Get DistMult-specific parameters
        let dropout_rate = config
            .model_params
            .get("dropout_rate")
            .copied()
            .unwrap_or(0.0);

        let loss_function = match config.model_params.get("loss_function") {
            Some(0.0) => LossFunction::Logistic,
            Some(1.0) => LossFunction::MarginRanking,
            Some(2.0) => LossFunction::SquaredLoss,
            _ => LossFunction::Logistic, // Default to logistic
        };

        Self {
            base,
            entity_embeddings: Array2::zeros((0, config.dimensions)),
            relation_embeddings: Array2::zeros((0, config.dimensions)),
            embeddings_initialized: false,
            dropout_rate,
            loss_function,
        }
    }

    /// Initialize embeddings
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

        // Initialize embeddings with Xavier initialization
        self.entity_embeddings =
            xavier_init((num_entities, dimensions), dimensions, dimensions, &mut rng);

        self.relation_embeddings = xavier_init(
            (num_relations, dimensions),
            dimensions,
            dimensions,
            &mut rng,
        );

        // For DistMult, it's common to normalize entity embeddings initially
        normalize_embeddings(&mut self.entity_embeddings);

        self.embeddings_initialized = true;
        debug!(
            "Initialized DistMult embeddings: {} entities, {} relations, {} dimensions",
            num_entities, num_relations, dimensions
        );
    }

    /// Score a triple using DistMult scoring function: sum(h * r * t)
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

        // DistMult score: sum of element-wise multiplication
        let score = (&h * &r * t).sum();

        Ok(score)
    }

    /// Apply dropout to embeddings during training
    #[allow(dead_code)]
    fn apply_dropout(&self, embeddings: &Array1<f64>, rng: &mut Random) -> Array1<f64> {
        if self.dropout_rate > 0.0 {
            embeddings.mapv(|x| {
                if rng.random_f64() < self.dropout_rate {
                    0.0
                } else {
                    x / (1.0 - self.dropout_rate)
                }
            })
        } else {
            embeddings.to_owned()
        }
    }

    /// Compute gradients for DistMult model
    fn compute_gradients(
        &self,
        pos_triple: (usize, usize, usize),
        neg_triple: (usize, usize, usize),
        pos_score: f64,
        neg_score: f64,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let mut entity_grads = Array2::zeros(self.entity_embeddings.raw_dim());
        let mut relation_grads = Array2::zeros(self.relation_embeddings.raw_dim());

        match self.loss_function {
            LossFunction::Logistic => {
                // Logistic loss gradients
                let pos_sigmoid = sigmoid(pos_score);
                let neg_sigmoid = sigmoid(neg_score);

                let pos_grad_coeff = pos_sigmoid - 1.0; // d/dx log(sigmoid(x)) = sigmoid(x) - 1
                let neg_grad_coeff = neg_sigmoid; // d/dx log(1 - sigmoid(x)) = sigmoid(x)

                self.add_triple_gradients(
                    pos_triple,
                    pos_grad_coeff,
                    &mut entity_grads,
                    &mut relation_grads,
                );
                self.add_triple_gradients(
                    neg_triple,
                    neg_grad_coeff,
                    &mut entity_grads,
                    &mut relation_grads,
                );
            }
            LossFunction::MarginRanking => {
                // Margin ranking loss: max(0, margin + neg_score - pos_score)
                let margin = self
                    .base
                    .config
                    .model_params
                    .get("margin")
                    .copied()
                    .unwrap_or(1.0);
                let loss = margin + neg_score - pos_score;

                if loss > 0.0 {
                    // Gradients: -1 for positive triple, +1 for negative triple
                    self.add_triple_gradients(
                        pos_triple,
                        -1.0,
                        &mut entity_grads,
                        &mut relation_grads,
                    );
                    self.add_triple_gradients(
                        neg_triple,
                        1.0,
                        &mut entity_grads,
                        &mut relation_grads,
                    );
                }
            }
            LossFunction::SquaredLoss => {
                // Squared loss: (1 - pos_score)^2 + (0 - neg_score)^2
                let pos_grad_coeff = -2.0 * (1.0 - pos_score);
                let neg_grad_coeff = -2.0 * neg_score;

                self.add_triple_gradients(
                    pos_triple,
                    pos_grad_coeff,
                    &mut entity_grads,
                    &mut relation_grads,
                );
                self.add_triple_gradients(
                    neg_triple,
                    neg_grad_coeff,
                    &mut entity_grads,
                    &mut relation_grads,
                );
            }
        }

        Ok((entity_grads, relation_grads))
    }

    /// Add gradients for a single triple
    fn add_triple_gradients(
        &self,
        triple: (usize, usize, usize),
        grad_coeff: f64,
        entity_grads: &mut Array2<f64>,
        relation_grads: &mut Array2<f64>,
    ) {
        let (s, p, o) = triple;

        let h = self.entity_embeddings.row(s);
        let r = self.relation_embeddings.row(p);
        let t = self.entity_embeddings.row(o);

        // DistMult gradients:
        // ∂score/∂h = r * t
        // ∂score/∂r = h * t
        // ∂score/∂t = h * r

        let h_grad = (&r * &t) * grad_coeff;
        let r_grad = (&h * &t) * grad_coeff;
        let t_grad = (&h * &r) * grad_coeff;

        entity_grads.row_mut(s).add_assign(&h_grad);
        relation_grads.row_mut(p).add_assign(&r_grad);
        entity_grads.row_mut(o).add_assign(&t_grad);
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

                    // Compute loss
                    let triple_loss = match self.loss_function {
                        LossFunction::Logistic => {
                            logistic_loss(pos_score, 1.0) + logistic_loss(neg_score, -1.0)
                        }
                        LossFunction::MarginRanking => {
                            let margin = self
                                .base
                                .config
                                .model_params
                                .get("margin")
                                .copied()
                                .unwrap_or(1.0);
                            margin_loss(pos_score, neg_score, margin)
                        }
                        LossFunction::SquaredLoss => (1.0 - pos_score).powi(2) + neg_score.powi(2),
                    };

                    batch_loss += triple_loss;

                    // Compute and accumulate gradients
                    let (entity_grads, relation_grads) =
                        self.compute_gradients(pos_triple, neg_triple, pos_score, neg_score)?;

                    batch_entity_grads += &entity_grads;
                    batch_relation_grads += &relation_grads;
                }
            }

            // Apply gradients with L2 regularization
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

            // Optional: re-normalize entity embeddings
            if self
                .base
                .config
                .model_params
                .get("normalize_entities")
                .copied()
                .unwrap_or(0.0)
                > 0.0
            {
                normalize_embeddings(&mut self.entity_embeddings);
            }

            total_loss += batch_loss;
        }

        Ok(total_loss / num_batches as f64)
    }
}

#[async_trait]
impl EmbeddingModel for DistMult {
    fn config(&self) -> &ModelConfig {
        &self.base.config
    }

    fn model_id(&self) -> &Uuid {
        &self.base.model_id
    }

    fn model_type(&self) -> &'static str {
        "DistMult"
    }

    fn add_triple(&mut self, triple: Triple) -> Result<()> {
        // Check for asymmetric relations and warn
        let predicate_str = triple.predicate.to_string();
        if predicate_str.contains("parent")
            || predicate_str.contains("child")
            || predicate_str.contains("born")
            || predicate_str.contains("founder")
        {
            warn!(
                "DistMult may not handle asymmetric relation well: {}",
                predicate_str
            );
        }

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

        info!("Starting DistMult training for {} epochs", max_epochs);

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
        self.base.get_stats("DistMult")
    }

    fn save(&self, path: &str) -> Result<()> {
        info!("Saving DistMult model to {}", path);
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        info!("Loading DistMult model from {}", path);
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
            "Knowledge graph embedding model does not support text encoding"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedNode;

    #[tokio::test]
    async fn test_distmult_basic() -> Result<()> {
        let config = ModelConfig::default()
            .with_dimensions(50)
            .with_max_epochs(10)
            .with_seed(42);

        let mut model = DistMult::new(config);

        // Add test triples (use symmetric relation for DistMult)
        let alice = NamedNode::new("http://example.org/alice")?;
        let similar_to = NamedNode::new("http://example.org/similarTo")?;
        let bob = NamedNode::new("http://example.org/bob")?;

        model.add_triple(Triple::new(alice.clone(), similar_to.clone(), bob.clone()))?;
        model.add_triple(Triple::new(bob.clone(), similar_to.clone(), alice.clone()))?;

        // Train
        let stats = model.train(Some(5)).await?;
        assert!(stats.epochs_completed > 0);

        // Test embeddings
        let alice_emb = model.get_entity_embedding("http://example.org/alice")?;
        assert_eq!(alice_emb.dimensions, 50);

        // Test scoring
        let score = model.score_triple(
            "http://example.org/alice",
            "http://example.org/similarTo",
            "http://example.org/bob",
        )?;

        // Score should be a finite number
        assert!(score.is_finite());

        Ok(())
    }
}
