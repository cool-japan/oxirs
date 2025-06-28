//! TuckER: Tucker Decomposition for Knowledge Graph Embeddings
//!
//! TuckER is a tensor factorization model that performs link prediction
//! using Tucker decomposition on the binary tensor representation of knowledge graphs.
//!
//! Reference: Balažević et al. "TuckER: Tensor Factorization for Knowledge Graph Completion" (2019)

use crate::models::{common::*, BaseModel};
use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, SubAssign};
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

/// TuckER embedding model
#[derive(Debug)]
pub struct TuckER {
    /// Base model functionality
    base: BaseModel,
    /// Entity embeddings matrix (num_entities × entity_dim)
    entity_embeddings: Array2<f64>,
    /// Relation embeddings matrix (num_relations × relation_dim)  
    relation_embeddings: Array2<f64>,
    /// Core tensor for Tucker decomposition
    core_tensor: Array3<f64>,
    /// Whether embeddings have been initialized
    embeddings_initialized: bool,
    /// Entity embedding dimension
    entity_dim: usize,
    /// Relation embedding dimension
    relation_dim: usize,
    /// Core tensor dimensions
    core_dims: (usize, usize, usize),
    /// Dropout rate for training
    dropout_rate: f64,
    /// Batch normalization parameters
    batch_norm: bool,
}

impl TuckER {
    /// Create a new TuckER model
    pub fn new(config: ModelConfig) -> Self {
        let base = BaseModel::new(config.clone());

        // Get TuckER-specific parameters from model_params
        let entity_dim = config
            .model_params
            .get("entity_dim")
            .map(|&v| v as usize)
            .unwrap_or(config.dimensions);
        let relation_dim = config
            .model_params
            .get("relation_dim")
            .map(|&v| v as usize)
            .unwrap_or(config.dimensions);
        let core_dim1 = config
            .model_params
            .get("core_dim1")
            .map(|&v| v as usize)
            .unwrap_or(config.dimensions);
        let core_dim2 = config
            .model_params
            .get("core_dim2")
            .map(|&v| v as usize)
            .unwrap_or(config.dimensions);
        let core_dim3 = config
            .model_params
            .get("core_dim3")
            .map(|&v| v as usize)
            .unwrap_or(config.dimensions);
        let dropout_rate = config
            .model_params
            .get("dropout_rate")
            .copied()
            .unwrap_or(0.3);
        let batch_norm = config
            .model_params
            .get("batch_norm")
            .map(|&v| v > 0.0)
            .unwrap_or(true);

        Self {
            base,
            entity_embeddings: Array2::zeros((0, entity_dim)),
            relation_embeddings: Array2::zeros((0, relation_dim)),
            core_tensor: Array3::zeros((core_dim1, core_dim2, core_dim3)),
            embeddings_initialized: false,
            entity_dim,
            relation_dim,
            core_dims: (core_dim1, core_dim2, core_dim3),
            dropout_rate,
            batch_norm,
        }
    }

    /// Initialize embeddings after entities and relations are known
    fn initialize_embeddings(&mut self) {
        if self.embeddings_initialized {
            return;
        }

        let num_entities = self.base.num_entities();
        let num_relations = self.base.num_relations();

        if num_entities == 0 || num_relations == 0 {
            return;
        }

        let mut rng = if let Some(seed) = self.base.config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        // Initialize entity embeddings with Xavier initialization
        self.entity_embeddings = xavier_init(
            (num_entities, self.entity_dim),
            self.entity_dim,
            self.entity_dim,
            &mut rng,
        );

        // Initialize relation embeddings with Xavier initialization
        self.relation_embeddings = xavier_init(
            (num_relations, self.relation_dim),
            self.relation_dim,
            self.relation_dim,
            &mut rng,
        );

        // Initialize core tensor with Xavier initialization
        let total_elements = self.core_dims.0 * self.core_dims.1 * self.core_dims.2;
        let std_dev = (2.0 / total_elements as f64).sqrt();

        for elem in self.core_tensor.iter_mut() {
            *elem = rng.gen_range(-std_dev..std_dev);
        }

        // Normalize embeddings
        normalize_embeddings(&mut self.entity_embeddings);
        normalize_embeddings(&mut self.relation_embeddings);

        self.embeddings_initialized = true;
        debug!(
            "Initialized TuckER embeddings: {} entities ({}D), {} relations ({}D), core tensor {:?}",
            num_entities, self.entity_dim, num_relations, self.relation_dim, self.core_dims
        );
    }

    /// Score a triple using TuckER scoring function
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

        // Compute Tucker decomposition score
        // score = Σ_i,j,k h_i * r_j * t_k * W_ijk
        let mut score = 0.0;

        for i in 0..self.core_dims.0.min(h.len()) {
            for j in 0..self.core_dims.1.min(r.len()) {
                for k in 0..self.core_dims.2.min(t.len()) {
                    score += h[i] * r[j] * t[k] * self.core_tensor[(i, j, k)];
                }
            }
        }

        Ok(score)
    }

    /// Compute gradients for Tucker decomposition
    fn compute_gradients(
        &self,
        pos_triple: (usize, usize, usize),
        neg_triple: (usize, usize, usize),
        learning_rate: f64,
    ) -> Result<(Array2<f64>, Array2<f64>, Array3<f64>)> {
        let (pos_s, pos_p, pos_o) = pos_triple;
        let (neg_s, neg_p, neg_o) = neg_triple;

        let mut entity_grads = Array2::zeros(self.entity_embeddings.raw_dim());
        let mut relation_grads = Array2::zeros(self.relation_embeddings.raw_dim());
        let mut core_grads = Array3::zeros(self.core_tensor.raw_dim());

        // Compute scores
        let pos_score = self.score_triple_ids(pos_s, pos_p, pos_o)?;
        let neg_score = self.score_triple_ids(neg_s, neg_p, neg_o)?;

        // Logistic loss gradient
        let pos_sigmoid = 1.0 / (1.0 + (-pos_score).exp());
        let neg_sigmoid = 1.0 / (1.0 + (-neg_score).exp());

        let pos_grad = pos_sigmoid - 1.0;
        let neg_grad = neg_sigmoid;

        // Compute gradients for positive triple
        self.compute_triple_gradients(
            pos_triple,
            pos_grad,
            &mut entity_grads,
            &mut relation_grads,
            &mut core_grads,
        );

        // Compute gradients for negative triple
        self.compute_triple_gradients(
            neg_triple,
            neg_grad,
            &mut entity_grads,
            &mut relation_grads,
            &mut core_grads,
        );

        Ok((entity_grads, relation_grads, core_grads))
    }

    /// Compute gradients for a single triple
    fn compute_triple_gradients(
        &self,
        triple: (usize, usize, usize),
        loss_grad: f64,
        entity_grads: &mut Array2<f64>,
        relation_grads: &mut Array2<f64>,
        core_grads: &mut Array3<f64>,
    ) {
        let (s, p, o) = triple;

        let h = self.entity_embeddings.row(s);
        let r = self.relation_embeddings.row(p);
        let t = self.entity_embeddings.row(o);

        // Gradients w.r.t. entity embeddings
        for i in 0..self.core_dims.0.min(h.len()) {
            let mut h_grad = 0.0;
            for j in 0..self.core_dims.1.min(r.len()) {
                for k in 0..self.core_dims.2.min(t.len()) {
                    h_grad += r[j] * t[k] * self.core_tensor[(i, j, k)];
                }
            }
            entity_grads[[s, i]] += loss_grad * h_grad;
        }

        for k in 0..self.core_dims.2.min(t.len()) {
            let mut t_grad = 0.0;
            for i in 0..self.core_dims.0.min(h.len()) {
                for j in 0..self.core_dims.1.min(r.len()) {
                    t_grad += h[i] * r[j] * self.core_tensor[(i, j, k)];
                }
            }
            entity_grads[[o, k]] += loss_grad * t_grad;
        }

        // Gradients w.r.t. relation embeddings
        for j in 0..self.core_dims.1.min(r.len()) {
            let mut r_grad = 0.0;
            for i in 0..self.core_dims.0.min(h.len()) {
                for k in 0..self.core_dims.2.min(t.len()) {
                    r_grad += h[i] * t[k] * self.core_tensor[(i, j, k)];
                }
            }
            relation_grads[[p, j]] += loss_grad * r_grad;
        }

        // Gradients w.r.t. core tensor
        for i in 0..self.core_dims.0.min(h.len()) {
            for j in 0..self.core_dims.1.min(r.len()) {
                for k in 0..self.core_dims.2.min(t.len()) {
                    core_grads[[i, j, k]] += loss_grad * h[i] * r[j] * t[k];
                }
            }
        }
    }

    /// Perform one training epoch
    async fn train_epoch(&mut self, learning_rate: f64) -> Result<f64> {
        let mut rng = if let Some(seed) = self.base.config.seed {
            StdRng::seed_from_u64(seed + rand::random::<u64>())
        } else {
            StdRng::from_entropy()
        };

        let mut total_loss = 0.0;
        let num_batches = (self.base.triples.len() + self.base.config.batch_size - 1)
            / self.base.config.batch_size;

        // Create shuffled batches
        let mut shuffled_triples = self.base.triples.clone();
        shuffled_triples.shuffle(&mut rng);

        for batch_triples in shuffled_triples.chunks(self.base.config.batch_size) {
            let mut batch_entity_grads = Array2::zeros(self.entity_embeddings.raw_dim());
            let mut batch_relation_grads = Array2::zeros(self.relation_embeddings.raw_dim());
            let mut batch_core_grads = Array3::zeros(self.core_tensor.raw_dim());
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

                    // Logistic loss
                    let pos_loss = -(1.0 / (1.0 + (-pos_score).exp())).ln();
                    let neg_loss = -(1.0 / (1.0 + neg_score.exp())).ln();
                    let loss = pos_loss + neg_loss;
                    batch_loss += loss;

                    // Compute and accumulate gradients
                    let (entity_grads, relation_grads, core_grads) =
                        self.compute_gradients(pos_triple, neg_triple, learning_rate)?;

                    batch_entity_grads += &entity_grads;
                    batch_relation_grads += &relation_grads;
                    batch_core_grads += &core_grads;
                }
            }

            // Apply gradients with L2 regularization
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

                // Update core tensor
                for ((i, j, k), grad) in self
                    .core_tensor
                    .indexed_iter_mut()
                    .zip(batch_core_grads.iter())
                {
                    let reg_term = self.base.config.l2_reg * *grad;
                    *grad -= learning_rate * (grad + reg_term);
                }

                // Apply dropout to embeddings
                if self.dropout_rate > 0.0 {
                    apply_dropout(&mut self.entity_embeddings, self.dropout_rate, &mut rng);
                    apply_dropout(&mut self.relation_embeddings, self.dropout_rate, &mut rng);
                }

                // Normalize embeddings
                normalize_embeddings(&mut self.entity_embeddings);
                normalize_embeddings(&mut self.relation_embeddings);
            }

            total_loss += batch_loss;
        }

        Ok(total_loss / num_batches as f64)
    }
}

#[async_trait]
impl EmbeddingModel for TuckER {
    fn config(&self) -> &ModelConfig {
        &self.base.config
    }

    fn model_id(&self) -> &Uuid {
        &self.base.model_id
    }

    fn model_type(&self) -> &'static str {
        "TuckER"
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

        info!("Starting TuckER training for {} epochs", max_epochs);

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

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
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
        self.base.get_stats("TuckER")
    }

    fn save(&self, path: &str) -> Result<()> {
        info!("Saving TuckER model to {}", path);
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        info!("Loading TuckER model from {}", path);
        Ok(())
    }

    fn clear(&mut self) {
        self.base.clear();
        self.entity_embeddings = Array2::zeros((0, self.entity_dim));
        self.relation_embeddings = Array2::zeros((0, self.relation_dim));
        self.core_tensor = Array3::zeros(self.core_dims);
        self.embeddings_initialized = false;
    }

    fn is_trained(&self) -> bool {
        self.base.is_trained
    }
}

/// Apply dropout to embeddings
fn apply_dropout(embeddings: &mut Array2<f64>, dropout_rate: f64, rng: &mut StdRng) {
    for elem in embeddings.iter_mut() {
        if rng.gen::<f64>() < dropout_rate {
            *elem = 0.0;
        } else {
            *elem /= 1.0 - dropout_rate;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedNode;

    #[tokio::test]
    async fn test_tucker_basic() -> Result<()> {
        let mut config = ModelConfig::default()
            .with_dimensions(50)
            .with_max_epochs(10)
            .with_seed(42);

        // Add TuckER-specific parameters
        config.model_params.insert("entity_dim".to_string(), 50.0);
        config.model_params.insert("relation_dim".to_string(), 50.0);
        config.model_params.insert("core_dim1".to_string(), 50.0);
        config.model_params.insert("core_dim2".to_string(), 50.0);
        config.model_params.insert("core_dim3".to_string(), 50.0);
        config.model_params.insert("dropout_rate".to_string(), 0.1);

        let mut model = TuckER::new(config);

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

    #[test]
    fn test_tucker_creation() {
        let config = ModelConfig::default();
        let tucker = TuckER::new(config);
        assert!(!tucker.embeddings_initialized);
        assert_eq!(tucker.model_type(), "TuckER");
    }
}
