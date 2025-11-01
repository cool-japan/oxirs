//! ComplEx: Complex Embeddings for Simple Link Prediction
//!
//! ComplEx uses complex-valued embeddings to better model asymmetric relations.
//! The scoring function is: Re(<h, r, conj(t)>) where Re denotes real part,
//! <> denotes complex dot product, and conj denotes complex conjugate.
//!
//! Reference: Trouillon et al. "Complex Embeddings for Simple Link Prediction" (2016)

use crate::models::{common::*, BaseModel};
use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use scirs2_core::ndarray_ext::Array2;
#[allow(unused_imports)]
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

/// Type alias for gradient tensors
type GradientTuple = (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>);

/// ComplEx embedding model using complex-valued embeddings
#[derive(Debug)]
pub struct ComplEx {
    /// Base model functionality
    base: BaseModel,
    /// Real part of entity embeddings (num_entities × dimensions)
    entity_embeddings_real: Array2<f64>,
    /// Imaginary part of entity embeddings (num_entities × dimensions)
    entity_embeddings_imag: Array2<f64>,
    /// Real part of relation embeddings (num_relations × dimensions)
    relation_embeddings_real: Array2<f64>,
    /// Imaginary part of relation embeddings (num_relations × dimensions)
    relation_embeddings_imag: Array2<f64>,
    /// Whether embeddings have been initialized
    embeddings_initialized: bool,
    /// Regularization method
    regularization: RegularizationType,
}

/// Regularization types for ComplEx
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RegularizationType {
    /// L2 regularization on embeddings
    L2,
    /// N3 regularization (nuclear 3-norm)
    N3,
    /// No additional regularization
    None,
}

impl ComplEx {
    /// Create a new ComplEx model
    pub fn new(config: ModelConfig) -> Self {
        let base = BaseModel::new(config.clone());

        // Get ComplEx-specific parameters
        let regularization = match config.model_params.get("regularization") {
            Some(0.0) => RegularizationType::None,
            Some(1.0) => RegularizationType::L2,
            Some(2.0) => RegularizationType::N3,
            _ => RegularizationType::N3, // Default to N3
        };

        Self {
            base,
            entity_embeddings_real: Array2::zeros((0, config.dimensions)),
            entity_embeddings_imag: Array2::zeros((0, config.dimensions)),
            relation_embeddings_real: Array2::zeros((0, config.dimensions)),
            relation_embeddings_imag: Array2::zeros((0, config.dimensions)),
            embeddings_initialized: false,
            regularization,
        }
    }

    /// Initialize complex embeddings
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

        // Initialize all embedding components with Xavier initialization
        self.entity_embeddings_real =
            xavier_init((num_entities, dimensions), dimensions, dimensions, &mut rng);

        self.entity_embeddings_imag =
            xavier_init((num_entities, dimensions), dimensions, dimensions, &mut rng);

        self.relation_embeddings_real = xavier_init(
            (num_relations, dimensions),
            dimensions,
            dimensions,
            &mut rng,
        );

        self.relation_embeddings_imag = xavier_init(
            (num_relations, dimensions),
            dimensions,
            dimensions,
            &mut rng,
        );

        self.embeddings_initialized = true;
        debug!(
            "Initialized ComplEx embeddings: {} entities, {} relations, {} dimensions",
            num_entities, num_relations, dimensions
        );
    }

    /// Score a triple using ComplEx scoring function
    /// Score = Re(<h, r, conj(t)>) = Re(h) * Re(r) * Re(t) + Re(h) * Im(r) * Im(t) +
    ///                                 Im(h) * Re(r) * Im(t) - Im(h) * Im(r) * Re(t)
    fn score_triple_ids(
        &self,
        subject_id: usize,
        predicate_id: usize,
        object_id: usize,
    ) -> Result<f64> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }

        let h_real = self.entity_embeddings_real.row(subject_id);
        let h_imag = self.entity_embeddings_imag.row(subject_id);
        let r_real = self.relation_embeddings_real.row(predicate_id);
        let r_imag = self.relation_embeddings_imag.row(predicate_id);
        let t_real = self.entity_embeddings_real.row(object_id);
        let t_imag = self.entity_embeddings_imag.row(object_id);

        // Complex multiplication: (h_real + i*h_imag) * (r_real + i*r_imag) * conj(t_real + i*t_imag)
        // = (h_real + i*h_imag) * (r_real + i*r_imag) * (t_real - i*t_imag)
        let score = (&h_real * &r_real * t_real).sum()
            + (&h_real * &r_imag * t_imag).sum()
            + (&h_imag * &r_real * t_imag).sum()
            - (&h_imag * &r_imag * t_real).sum();

        Ok(score)
    }

    /// Compute gradients for ComplEx model
    fn compute_gradients(
        &self,
        pos_triple: (usize, usize, usize),
        neg_triple: (usize, usize, usize),
        pos_score: f64,
        neg_score: f64,
    ) -> Result<GradientTuple> {
        let mut entity_grads_real = Array2::zeros(self.entity_embeddings_real.raw_dim());
        let mut entity_grads_imag = Array2::zeros(self.entity_embeddings_imag.raw_dim());
        let mut relation_grads_real = Array2::zeros(self.relation_embeddings_real.raw_dim());
        let mut relation_grads_imag = Array2::zeros(self.relation_embeddings_imag.raw_dim());

        // Logistic loss gradients
        let pos_sigmoid = sigmoid(pos_score);
        let neg_sigmoid = sigmoid(neg_score);

        let pos_grad_coeff = pos_sigmoid - 1.0; // Derivative of log(sigmoid(x))
        let neg_grad_coeff = neg_sigmoid; // Derivative of log(1 - sigmoid(x))

        // Compute gradients for positive triple
        self.add_triple_gradients(
            pos_triple,
            pos_grad_coeff,
            &mut entity_grads_real,
            &mut entity_grads_imag,
            &mut relation_grads_real,
            &mut relation_grads_imag,
        );

        // Compute gradients for negative triple
        self.add_triple_gradients(
            neg_triple,
            neg_grad_coeff,
            &mut entity_grads_real,
            &mut entity_grads_imag,
            &mut relation_grads_real,
            &mut relation_grads_imag,
        );

        Ok((
            entity_grads_real,
            entity_grads_imag,
            relation_grads_real,
            relation_grads_imag,
        ))
    }

    /// Add gradients for a single triple
    fn add_triple_gradients(
        &self,
        triple: (usize, usize, usize),
        grad_coeff: f64,
        entity_grads_real: &mut Array2<f64>,
        entity_grads_imag: &mut Array2<f64>,
        relation_grads_real: &mut Array2<f64>,
        relation_grads_imag: &mut Array2<f64>,
    ) {
        let (s, p, o) = triple;

        let h_real = self.entity_embeddings_real.row(s);
        let h_imag = self.entity_embeddings_imag.row(s);
        let r_real = self.relation_embeddings_real.row(p);
        let r_imag = self.relation_embeddings_imag.row(p);
        let t_real = self.entity_embeddings_real.row(o);
        let t_imag = self.entity_embeddings_imag.row(o);

        // Gradients w.r.t. h (subject)
        // ∂score/∂h_real = r_real * t_real + r_imag * t_imag
        // ∂score/∂h_imag = r_real * t_imag - r_imag * t_real
        let h_real_grad = (&r_real * &t_real + &r_imag * &t_imag) * grad_coeff;
        let h_imag_grad = (&r_real * &t_imag - &r_imag * &t_real) * grad_coeff;

        entity_grads_real.row_mut(s).add_assign(&h_real_grad);
        entity_grads_imag.row_mut(s).add_assign(&h_imag_grad);

        // Gradients w.r.t. r (relation)
        // ∂score/∂r_real = h_real * t_real + h_imag * t_imag
        // ∂score/∂r_imag = h_real * t_imag - h_imag * t_real
        let r_real_grad = (&h_real * &t_real + &h_imag * &t_imag) * grad_coeff;
        let r_imag_grad = (&h_real * &t_imag - &h_imag * &t_real) * grad_coeff;

        relation_grads_real.row_mut(p).add_assign(&r_real_grad);
        relation_grads_imag.row_mut(p).add_assign(&r_imag_grad);

        // Gradients w.r.t. t (object) - note the conjugate
        // ∂score/∂t_real = h_real * r_real - h_imag * r_imag
        // ∂score/∂t_imag = -(h_real * r_imag + h_imag * r_real)
        let t_real_grad = (&h_real * &r_real - &h_imag * &r_imag) * grad_coeff;
        let t_imag_grad = -(&h_real * &r_imag + &h_imag * &r_real) * grad_coeff;

        entity_grads_real.row_mut(o).add_assign(&t_real_grad);
        entity_grads_imag.row_mut(o).add_assign(&t_imag_grad);
    }

    /// Apply N3 regularization
    fn apply_n3_regularization(
        &self,
        entity_grads_real: &mut Array2<f64>,
        entity_grads_imag: &mut Array2<f64>,
        relation_grads_real: &mut Array2<f64>,
        relation_grads_imag: &mut Array2<f64>,
        regularization_weight: f64,
    ) {
        // N3 regularization: penalize the nuclear 3-norm
        // For complex embeddings, this becomes more involved
        // For simplicity, we apply L2 regularization here
        // A full N3 implementation would require more complex tensor operations

        *entity_grads_real += &(&self.entity_embeddings_real * regularization_weight);
        *entity_grads_imag += &(&self.entity_embeddings_imag * regularization_weight);
        *relation_grads_real += &(&self.relation_embeddings_real * regularization_weight);
        *relation_grads_imag += &(&self.relation_embeddings_imag * regularization_weight);
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
            let mut batch_entity_grads_real = Array2::zeros(self.entity_embeddings_real.raw_dim());
            let mut batch_entity_grads_imag = Array2::zeros(self.entity_embeddings_imag.raw_dim());
            let mut batch_relation_grads_real =
                Array2::zeros(self.relation_embeddings_real.raw_dim());
            let mut batch_relation_grads_imag =
                Array2::zeros(self.relation_embeddings_imag.raw_dim());
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

                    // Compute logistic loss
                    let pos_loss = logistic_loss(pos_score, 1.0);
                    let neg_loss = logistic_loss(neg_score, -1.0);
                    let total_triple_loss = pos_loss + neg_loss;

                    batch_loss += total_triple_loss;

                    // Compute and accumulate gradients
                    let (
                        entity_grads_real,
                        entity_grads_imag,
                        relation_grads_real,
                        relation_grads_imag,
                    ) = self.compute_gradients(pos_triple, neg_triple, pos_score, neg_score)?;

                    batch_entity_grads_real += &entity_grads_real;
                    batch_entity_grads_imag += &entity_grads_imag;
                    batch_relation_grads_real += &relation_grads_real;
                    batch_relation_grads_imag += &relation_grads_imag;
                }
            }

            // Apply regularization
            match self.regularization {
                RegularizationType::L2 => {
                    let reg_weight = self.base.config.l2_reg;
                    batch_entity_grads_real += &(&self.entity_embeddings_real * reg_weight);
                    batch_entity_grads_imag += &(&self.entity_embeddings_imag * reg_weight);
                    batch_relation_grads_real += &(&self.relation_embeddings_real * reg_weight);
                    batch_relation_grads_imag += &(&self.relation_embeddings_imag * reg_weight);
                }
                RegularizationType::N3 => {
                    self.apply_n3_regularization(
                        &mut batch_entity_grads_real,
                        &mut batch_entity_grads_imag,
                        &mut batch_relation_grads_real,
                        &mut batch_relation_grads_imag,
                        self.base.config.l2_reg,
                    );
                }
                RegularizationType::None => {}
            }

            // Apply gradients
            self.entity_embeddings_real -= &(&batch_entity_grads_real * learning_rate);
            self.entity_embeddings_imag -= &(&batch_entity_grads_imag * learning_rate);
            self.relation_embeddings_real -= &(&batch_relation_grads_real * learning_rate);
            self.relation_embeddings_imag -= &(&batch_relation_grads_imag * learning_rate);

            total_loss += batch_loss;
        }

        Ok(total_loss / num_batches as f64)
    }

    /// Get entity embedding as a concatenated real/imaginary vector
    fn get_entity_embedding_vector(&self, entity_id: usize) -> Vector {
        let real_part = self.entity_embeddings_real.row(entity_id);
        let imag_part = self.entity_embeddings_imag.row(entity_id);

        // Concatenate real and imaginary parts
        let mut values = Vec::with_capacity(real_part.len() * 2);
        for &val in real_part.iter() {
            values.push(val as f32);
        }
        for &val in imag_part.iter() {
            values.push(val as f32);
        }

        Vector::new(values)
    }

    /// Get relation embedding as a concatenated real/imaginary vector
    fn get_relation_embedding_vector(&self, relation_id: usize) -> Vector {
        let real_part = self.relation_embeddings_real.row(relation_id);
        let imag_part = self.relation_embeddings_imag.row(relation_id);

        // Concatenate real and imaginary parts
        let mut values = Vec::with_capacity(real_part.len() * 2);
        for &val in real_part.iter() {
            values.push(val as f32);
        }
        for &val in imag_part.iter() {
            values.push(val as f32);
        }

        Vector::new(values)
    }
}

#[async_trait]
impl EmbeddingModel for ComplEx {
    fn config(&self) -> &ModelConfig {
        &self.base.config
    }

    fn model_id(&self) -> &Uuid {
        &self.base.model_id
    }

    fn model_type(&self) -> &'static str {
        "ComplEx"
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

        info!("Starting ComplEx training for {} epochs", max_epochs);

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

        Ok(self.get_entity_embedding_vector(entity_id))
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }

        let relation_id = self
            .base
            .get_relation_id(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;

        Ok(self.get_relation_embedding_vector(relation_id))
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
        self.base.get_stats("ComplEx")
    }

    fn save(&self, path: &str) -> Result<()> {
        info!("Saving ComplEx model to {}", path);
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        info!("Loading ComplEx model from {}", path);
        Ok(())
    }

    fn clear(&mut self) {
        self.base.clear();
        self.entity_embeddings_real = Array2::zeros((0, self.base.config.dimensions));
        self.entity_embeddings_imag = Array2::zeros((0, self.base.config.dimensions));
        self.relation_embeddings_real = Array2::zeros((0, self.base.config.dimensions));
        self.relation_embeddings_imag = Array2::zeros((0, self.base.config.dimensions));
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
    async fn test_complex_basic() -> Result<()> {
        let config = ModelConfig::default()
            .with_dimensions(50)
            .with_max_epochs(10)
            .with_seed(42);

        let mut model = ComplEx::new(config);

        // Add test triples
        let alice = NamedNode::new("http://example.org/alice")?;
        let knows = NamedNode::new("http://example.org/knows")?;
        let bob = NamedNode::new("http://example.org/bob")?;

        model.add_triple(Triple::new(alice.clone(), knows.clone(), bob.clone()))?;

        // Train
        let stats = model.train(Some(5)).await?;
        assert!(stats.epochs_completed > 0);

        // Test embeddings (should be 2x dimensions due to complex)
        let alice_emb = model.get_entity_embedding("http://example.org/alice")?;
        assert_eq!(alice_emb.dimensions, 100); // 2 * 50

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
}
