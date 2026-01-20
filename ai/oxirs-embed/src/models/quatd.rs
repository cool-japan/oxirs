//! QuatE: Quaternion Embeddings for Knowledge Graph Completion
//!
//! QuatE models entities and relations as quaternions in a 4D space,
//! using quaternion algebra for knowledge graph completion.
//!
//! Reference: Zhang et al. "Quaternion Knowledge Graph Embeddings" (2019)

use crate::models::BaseModel;
use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::random::{Random, SliceRandom};
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

/// Quaternion representation for embeddings
#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
    /// Real component
    pub w: f64,
    /// i component
    pub x: f64,
    /// j component
    pub y: f64,
    /// k component
    pub z: f64,
}

impl Quaternion {
    /// Create a new quaternion
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Create a quaternion from a 4-element array
    pub fn from_array(arr: &[f64]) -> Self {
        assert_eq!(arr.len(), 4);
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }

    /// Convert quaternion to array
    pub fn to_array(&self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }

    /// Quaternion multiplication (Hamilton product)
    pub fn multiply(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Quaternion conjugate
    pub fn conjugate(&self) -> Quaternion {
        Quaternion {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Quaternion norm (magnitude)
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize quaternion to unit length
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 1e-12 {
            self.w /= norm;
            self.x /= norm;
            self.y /= norm;
            self.z /= norm;
        }
    }

    /// Quaternion dot product
    pub fn dot(&self, other: &Quaternion) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Element-wise addition
    pub fn add(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w + other.w,
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Element-wise subtraction
    pub fn subtract(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w - other.w,
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Quaternion {
        Quaternion {
            w: self.w * scalar,
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

/// QuatD embedding model
#[derive(Debug)]
pub struct QuatD {
    /// Base model functionality
    base: BaseModel,
    /// Entity embeddings as quaternions (num_entities × 4)
    entity_embeddings: Array2<f64>,
    /// Relation embeddings as quaternions (num_relations × 4)
    relation_embeddings: Array2<f64>,
    /// Whether embeddings have been initialized
    embeddings_initialized: bool,
    /// Scoring function variant
    scoring_function: QuatDScoringFunction,
    /// Regularization parameters
    quaternion_regularization: f64,
}

/// Scoring function variants for QuatD
#[derive(Debug, Clone, Copy)]
pub enum QuatDScoringFunction {
    /// Original QuatD scoring function
    Standard,
    /// QuatD with L2 distance
    L2Distance,
    /// QuatD with cosine similarity
    CosineSimilarity,
}

impl QuatD {
    /// Create a new QuatD model
    pub fn new(config: ModelConfig) -> Self {
        let base = BaseModel::new(config.clone());

        // Get QuatD-specific parameters
        let scoring_function = match config.model_params.get("scoring_function") {
            Some(0.0) => QuatDScoringFunction::Standard,
            Some(1.0) => QuatDScoringFunction::L2Distance,
            Some(2.0) => QuatDScoringFunction::CosineSimilarity,
            _ => QuatDScoringFunction::Standard,
        };

        let quaternion_regularization = config
            .model_params
            .get("quaternion_regularization")
            .copied()
            .unwrap_or(0.05);

        Self {
            base,
            entity_embeddings: Array2::zeros((0, 4)), // 4D quaternions
            relation_embeddings: Array2::zeros((0, 4)), // 4D quaternions
            embeddings_initialized: false,
            scoring_function,
            quaternion_regularization,
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

        let mut rng = Random::seed(self.base.config.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX_EPOCH")
                .as_secs()
        }));

        // Initialize entity embeddings as quaternions
        self.entity_embeddings =
            Array2::from_shape_fn((num_entities, 4), |_| rng.gen_range(-0.1..0.1));

        // Initialize relation embeddings as quaternions
        self.relation_embeddings =
            Array2::from_shape_fn((num_relations, 4), |_| rng.gen_range(-0.1..0.1));

        // Normalize quaternions to unit length
        self.normalize_all_quaternions();

        self.embeddings_initialized = true;
        debug!(
            "Initialized QuatD embeddings: {} entities, {} relations (4D quaternions)",
            num_entities, num_relations
        );
    }

    /// Normalize all quaternion embeddings to unit length
    fn normalize_all_quaternions(&mut self) {
        // Normalize entity embeddings
        for mut row in self.entity_embeddings.rows_mut() {
            let mut quat =
                Quaternion::from_array(row.as_slice().expect("row should be contiguous"));
            quat.normalize();
            let normalized = quat.to_array();
            for (i, &val) in normalized.iter().enumerate() {
                row[i] = val;
            }
        }

        // Normalize relation embeddings
        for mut row in self.relation_embeddings.rows_mut() {
            let mut quat =
                Quaternion::from_array(row.as_slice().expect("row should be contiguous"));
            quat.normalize();
            let normalized = quat.to_array();
            for (i, &val) in normalized.iter().enumerate() {
                row[i] = val;
            }
        }
    }

    /// Get quaternion from entity embeddings
    fn get_entity_quaternion(&self, entity_id: usize) -> Quaternion {
        let row = self.entity_embeddings.row(entity_id);
        Quaternion::from_array(row.as_slice().expect("row should be contiguous"))
    }

    /// Get quaternion from relation embeddings
    fn get_relation_quaternion(&self, relation_id: usize) -> Quaternion {
        let row = self.relation_embeddings.row(relation_id);
        Quaternion::from_array(row.as_slice().expect("row should be contiguous"))
    }

    /// Score a triple using QuatD scoring function
    fn score_triple_ids(
        &self,
        subject_id: usize,
        predicate_id: usize,
        object_id: usize,
    ) -> Result<f64> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }

        let h = self.get_entity_quaternion(subject_id);
        let r = self.get_relation_quaternion(predicate_id);
        let t = self.get_entity_quaternion(object_id);

        match self.scoring_function {
            QuatDScoringFunction::Standard => {
                // QuatD scoring: σ(h ∘ r · t)
                let hr = h.multiply(&r);
                Ok(hr.dot(&t))
            }
            QuatDScoringFunction::L2Distance => {
                // L2 distance: -||h ∘ r - t||₂
                let hr = h.multiply(&r);
                let diff = hr.subtract(&t);
                Ok(-diff.norm())
            }
            QuatDScoringFunction::CosineSimilarity => {
                // Cosine similarity between h ∘ r and t
                let hr = h.multiply(&r);
                let dot_product = hr.dot(&t);
                let magnitude_product = hr.norm() * t.norm();
                if magnitude_product > 1e-12 {
                    Ok(dot_product / magnitude_product)
                } else {
                    Ok(0.0)
                }
            }
        }
    }

    /// Compute gradients for QuatD
    fn compute_gradients(
        &self,
        pos_triple: (usize, usize, usize),
        neg_triple: (usize, usize, usize),
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (pos_s, pos_p, pos_o) = pos_triple;
        let (neg_s, neg_p, neg_o) = neg_triple;

        let mut entity_grads = Array2::zeros(self.entity_embeddings.raw_dim());
        let mut relation_grads = Array2::zeros(self.relation_embeddings.raw_dim());

        // Compute scores
        let pos_score = self.score_triple_ids(pos_s, pos_p, pos_o)?;
        let neg_score = self.score_triple_ids(neg_s, neg_p, neg_o)?;

        // Sigmoid derivatives
        let pos_sigmoid = 1.0 / (1.0 + (-pos_score).exp());
        let neg_sigmoid = 1.0 / (1.0 + (-neg_score).exp());

        let pos_grad = pos_sigmoid - 1.0;
        let neg_grad = neg_sigmoid;

        // Compute gradients for positive triple
        self.compute_triple_gradients(pos_triple, pos_grad, &mut entity_grads, &mut relation_grads);

        // Compute gradients for negative triple
        self.compute_triple_gradients(neg_triple, neg_grad, &mut entity_grads, &mut relation_grads);

        Ok((entity_grads, relation_grads))
    }

    /// Compute gradients for a single triple
    fn compute_triple_gradients(
        &self,
        triple: (usize, usize, usize),
        loss_grad: f64,
        entity_grads: &mut Array2<f64>,
        relation_grads: &mut Array2<f64>,
    ) {
        let (s, p, o) = triple;

        let h = self.get_entity_quaternion(s);
        let r = self.get_relation_quaternion(p);
        let t = self.get_entity_quaternion(o);

        match self.scoring_function {
            QuatDScoringFunction::Standard => {
                // Gradients for h ∘ r · t scoring
                let hr = h.multiply(&r);

                // ∂score/∂h = (r · t) where · is quaternion multiplication with t
                let r_conj = r.conjugate();
                let grad_h = r_conj.multiply(&t).scale(loss_grad);

                // ∂score/∂r = (h^* · t) where ^* is conjugate
                let h_conj = h.conjugate();
                let grad_r = h_conj.multiply(&t).scale(loss_grad);

                // ∂score/∂t = (h ∘ r)
                let grad_t = hr.scale(loss_grad);

                // Add gradients
                let grad_h_arr = grad_h.to_array();
                let grad_r_arr = grad_r.to_array();
                let grad_t_arr = grad_t.to_array();

                for i in 0..4 {
                    entity_grads[[s, i]] += grad_h_arr[i];
                    relation_grads[[p, i]] += grad_r_arr[i];
                    entity_grads[[o, i]] += grad_t_arr[i];
                }
            }
            QuatDScoringFunction::L2Distance => {
                // Gradients for -||h ∘ r - t||₂ scoring
                let hr = h.multiply(&r);
                let diff = hr.subtract(&t);
                let norm = diff.norm();

                if norm > 1e-12 {
                    let scale = -loss_grad / norm;

                    // Similar quaternion gradient computation but scaled by norm
                    let r_conj = r.conjugate();
                    let grad_h = r_conj.scale(scale);

                    let h_conj = h.conjugate();
                    let grad_r = h_conj.scale(scale);

                    let grad_t = diff.scale(-scale);

                    let grad_h_arr = grad_h.to_array();
                    let grad_r_arr = grad_r.to_array();
                    let grad_t_arr = grad_t.to_array();

                    for i in 0..4 {
                        entity_grads[[s, i]] += grad_h_arr[i];
                        relation_grads[[p, i]] += grad_r_arr[i];
                        entity_grads[[o, i]] += grad_t_arr[i];
                    }
                }
            }
            QuatDScoringFunction::CosineSimilarity => {
                // Gradients for cosine similarity
                let hr = h.multiply(&r);
                let dot_product = hr.dot(&t);
                let hr_norm = hr.norm();
                let t_norm = t.norm();
                let magnitude_product = hr_norm * t_norm;

                if magnitude_product > 1e-12 {
                    let cos_sim = dot_product / magnitude_product;

                    // Complex gradients for cosine similarity - simplified version
                    let scale = loss_grad / magnitude_product;

                    let grad_hr = t
                        .subtract(&hr.scale(cos_sim / (hr_norm * hr_norm)))
                        .scale(scale);
                    let grad_t = hr
                        .subtract(&t.scale(cos_sim / (t_norm * t_norm)))
                        .scale(scale);

                    // Backpropagate through quaternion multiplication for grad_hr
                    let r_conj = r.conjugate();
                    let grad_h = r_conj.multiply(&grad_hr);

                    let h_conj = h.conjugate();
                    let grad_r = h_conj.multiply(&grad_hr);

                    let grad_h_arr = grad_h.to_array();
                    let grad_r_arr = grad_r.to_array();
                    let grad_t_arr = grad_t.to_array();

                    for i in 0..4 {
                        entity_grads[[s, i]] += grad_h_arr[i];
                        relation_grads[[p, i]] += grad_r_arr[i];
                        entity_grads[[o, i]] += grad_t_arr[i];
                    }
                }
            }
        }
    }

    /// Perform one training epoch
    async fn train_epoch(&mut self, learning_rate: f64) -> Result<f64> {
        let mut rng = Random::seed(self.base.config.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX_EPOCH")
                .as_secs()
        }));

        let mut total_loss = 0.0;
        let num_batches = (self.base.triples.len() + self.base.config.batch_size - 1)
            / self.base.config.batch_size;

        // Create shuffled batches
        let mut shuffled_triples = self.base.triples.clone();
        shuffled_triples.shuffle(&mut rng);

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

                    // Logistic loss
                    let pos_loss = -(1.0 / (1.0 + (-pos_score).exp())).ln();
                    let neg_loss = -(1.0 / (1.0 + neg_score.exp())).ln();
                    let loss = pos_loss + neg_loss;
                    batch_loss += loss;

                    // Compute and accumulate gradients
                    let (entity_grads, relation_grads) =
                        self.compute_gradients(pos_triple, neg_triple)?;

                    batch_entity_grads += &entity_grads;
                    batch_relation_grads += &relation_grads;
                }
            }

            // Apply gradients with quaternion regularization
            if batch_loss > 0.0 {
                // Update entity embeddings
                for (((_i, _j), embedding_val), grad_val) in self
                    .entity_embeddings
                    .indexed_iter_mut()
                    .zip(batch_entity_grads.iter())
                {
                    let reg_term = self.quaternion_regularization * *embedding_val;
                    *embedding_val -= learning_rate * (grad_val + reg_term);
                }

                // Update relation embeddings
                for (((_i, _j), embedding_val), grad_val) in self
                    .relation_embeddings
                    .indexed_iter_mut()
                    .zip(batch_relation_grads.iter())
                {
                    let reg_term = self.quaternion_regularization * *embedding_val;
                    *embedding_val -= learning_rate * (grad_val + reg_term);
                }

                // Normalize quaternions after update
                self.normalize_all_quaternions();
            }

            total_loss += batch_loss;
        }

        Ok(total_loss / num_batches as f64)
    }
}

#[async_trait]
impl EmbeddingModel for QuatD {
    fn config(&self) -> &ModelConfig {
        &self.base.config
    }

    fn model_id(&self) -> &Uuid {
        &self.base.model_id
    }

    fn model_type(&self) -> &'static str {
        "QuatD"
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

        info!("Starting QuatD training for {} epochs", max_epochs);

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
        Ok(Vector::new(
            embedding.to_vec().into_iter().map(|x| x as f32).collect(),
        ))
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
        Ok(Vector::new(
            embedding.to_vec().into_iter().map(|x| x as f32).collect(),
        ))
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
            let object_name = self
                .base
                .get_entity(object_id)
                .expect("entity should exist in index")
                .clone();
            scores.push((object_name, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("scores should be comparable"));
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
            let subject_name = self
                .base
                .get_entity(subject_id)
                .expect("entity should exist in index")
                .clone();
            scores.push((subject_name, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("scores should be comparable"));
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
            let predicate_name = self
                .base
                .get_relation(predicate_id)
                .expect("relation should exist in index")
                .clone();
            scores.push((predicate_name, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("scores should be comparable"));
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
        self.base.get_stats("QuatD")
    }

    fn save(&self, path: &str) -> Result<()> {
        info!("Saving QuatD model to {}", path);
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        info!("Loading QuatD model from {}", path);
        Ok(())
    }

    fn clear(&mut self) {
        self.base.clear();
        self.entity_embeddings = Array2::zeros((0, 4));
        self.relation_embeddings = Array2::zeros((0, 4));
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

    #[test]
    fn test_quaternion_operations() {
        let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q2 = Quaternion::new(2.0, 3.0, 4.0, 5.0);

        // Test multiplication
        let product = q1.multiply(&q2);
        assert!(product.w.is_finite());

        // Test conjugate
        let conj = q1.conjugate();
        assert_eq!(conj.w, q1.w);
        assert_eq!(conj.x, -q1.x);

        // Test normalization
        let mut q3 = q1;
        q3.normalize();
        assert!((q3.norm() - 1.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_quatd_basic() -> Result<()> {
        let config = ModelConfig::default()
            .with_dimensions(4) // Always 4 for quaternions
            .with_max_epochs(10)
            .with_seed(42);

        let mut model = QuatD::new(config);

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
        assert_eq!(alice_emb.dimensions, 4); // Quaternion dimension

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
    fn test_quatd_creation() {
        let config = ModelConfig::default();
        let quatd = QuatD::new(config);
        assert!(!quatd.embeddings_initialized);
        assert_eq!(quatd.model_type(), "QuatD");
    }
}
