//! RotatE: Rotation-based Knowledge Graph Embeddings
//!
//! RotatE models relations as rotations in complex space, which allows it to
//! handle symmetric, antisymmetric, inverse, and compositional relation patterns.
//! Each relation is represented as a rotation from head to tail entity.
//!
//! Reference: Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (2019)

use crate::{EmbeddingModel, ModelConfig, TrainingStats, ModelStats, EmbeddingError};
use crate::models::{BaseModel, common::*};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use oxirs_core::Triple;
use oxirs_vec::Vector;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

/// RotatE embedding model using complex rotations
#[derive(Debug)]
pub struct RotatE {
    /// Base model functionality
    base: BaseModel,
    /// Real part of entity embeddings (num_entities × dimensions)
    entity_embeddings_real: Array2<f64>,
    /// Imaginary part of entity embeddings (num_entities × dimensions)
    entity_embeddings_imag: Array2<f64>,
    /// Relation phases/angles (num_relations × dimensions) - angles in [0, 2π]
    relation_phases: Array2<f64>,
    /// Whether embeddings have been initialized
    embeddings_initialized: bool,
    /// Adversarial temperature for negative sampling
    adversarial_temperature: f64,
    /// Modulus constraint for entity embeddings
    modulus_constraint: bool,
}

impl RotatE {
    /// Create a new RotatE model
    pub fn new(config: ModelConfig) -> Self {
        let base = BaseModel::new(config.clone());
        
        // Get RotatE-specific parameters
        let adversarial_temperature = config.model_params
            .get("adversarial_temperature")
            .copied()
            .unwrap_or(1.0);
        
        let modulus_constraint = config.model_params
            .get("modulus_constraint")
            .map(|&x| x > 0.0)
            .unwrap_or(true);
        
        Self {
            base,
            entity_embeddings_real: Array2::zeros((0, config.dimensions)),
            entity_embeddings_imag: Array2::zeros((0, config.dimensions)),
            relation_phases: Array2::zeros((0, config.dimensions)),
            embeddings_initialized: false,
            adversarial_temperature,
            modulus_constraint,
        }
    }
    
    /// Initialize embeddings with proper constraints
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
        
        let mut rng = if let Some(seed) = self.base.config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        
        // Initialize entity embeddings with uniform distribution
        self.entity_embeddings_real = uniform_init(
            (num_entities, dimensions),
            -1.0,
            1.0,
            &mut rng,
        );
        
        self.entity_embeddings_imag = uniform_init(
            (num_entities, dimensions),
            -1.0,
            1.0,
            &mut rng,
        );
        
        // Initialize relation phases uniformly in [0, 2π]
        self.relation_phases = uniform_init(
            (num_relations, dimensions),
            0.0,
            2.0 * std::f64::consts::PI,
            &mut rng,
        );
        
        // Apply modulus constraint to entity embeddings (normalize to unit circle)
        if self.modulus_constraint {
            self.apply_modulus_constraint();
        }
        
        self.embeddings_initialized = true;
        debug!("Initialized RotatE embeddings: {} entities, {} relations, {} dimensions",
               num_entities, num_relations, dimensions);
    }
    
    /// Apply modulus constraint to entity embeddings
    fn apply_modulus_constraint(&mut self) {
        for i in 0..self.entity_embeddings_real.nrows() {
            let mut real_row = self.entity_embeddings_real.row_mut(i);
            let mut imag_row = self.entity_embeddings_imag.row_mut(i);
            
            for j in 0..real_row.len() {
                let real = real_row[j];
                let imag = imag_row[j];
                let modulus = (real * real + imag * imag).sqrt();
                
                if modulus > 1e-10 {
                    real_row[j] = real / modulus;
                    imag_row[j] = imag / modulus;
                }
            }
        }
    }
    
    /// Score a triple using RotatE scoring function
    /// Score = ||h ○ r - t||, where ○ denotes complex multiplication (rotation)
    fn score_triple_ids(&self, subject_id: usize, predicate_id: usize, object_id: usize) -> Result<f64> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }
        
        let h_real = self.entity_embeddings_real.row(subject_id);
        let h_imag = self.entity_embeddings_imag.row(subject_id);
        let r_phases = self.relation_phases.row(predicate_id);
        let t_real = self.entity_embeddings_real.row(object_id);
        let t_imag = self.entity_embeddings_imag.row(object_id);
        
        // Compute h ○ r (rotation of h by r)
        // r is represented as e^(i*θ) = cos(θ) + i*sin(θ)
        // h ○ r = (h_real + i*h_imag) * (cos(θ) + i*sin(θ))
        //       = (h_real*cos(θ) - h_imag*sin(θ)) + i*(h_real*sin(θ) + h_imag*cos(θ))
        
        let mut distance_squared = 0.0;
        
        for ((((&h_r, &h_i), &phase), &t_r), &t_i) in h_real.iter()
            .zip(h_imag.iter())
            .zip(r_phases.iter())
            .zip(t_real.iter())
            .zip(t_imag.iter()) {
            
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();
            
            // Rotated head entity
            let rotated_real = h_r * cos_phase - h_i * sin_phase;
            let rotated_imag = h_r * sin_phase + h_i * cos_phase;
            
            // Distance components
            let diff_real = rotated_real - t_r;
            let diff_imag = rotated_imag - t_i;
            
            distance_squared += diff_real * diff_real + diff_imag * diff_imag;
        }
        
        // Return negative distance as score (higher is better)
        Ok(-distance_squared.sqrt())
    }
    
    /// Compute gradients for RotatE model
    fn compute_gradients(
        &self,
        pos_triple: (usize, usize, usize),
        neg_triple: (usize, usize, usize),
        pos_score: f64,
        neg_score: f64,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let mut entity_grads_real = Array2::zeros(self.entity_embeddings_real.raw_dim());
        let mut entity_grads_imag = Array2::zeros(self.entity_embeddings_imag.raw_dim());
        let mut relation_grads = Array2::zeros(self.relation_phases.raw_dim());
        
        // Margin-based ranking loss gradients
        let margin = self.base.config.model_params.get("margin").copied().unwrap_or(6.0);
        let loss = margin + (-pos_score) - (-neg_score); // Convert back to distances
        
        if loss > 0.0 {
            // Compute gradients for positive triple (increase distance)
            self.add_triple_gradients(
                pos_triple,
                1.0,
                &mut entity_grads_real,
                &mut entity_grads_imag,
                &mut relation_grads,
            );
            
            // Compute gradients for negative triple (decrease distance)
            self.add_triple_gradients(
                neg_triple,
                -1.0,
                &mut entity_grads_real,
                &mut entity_grads_imag,
                &mut relation_grads,
            );
        }
        
        Ok((entity_grads_real, entity_grads_imag, relation_grads))
    }
    
    /// Add gradients for a single triple
    fn add_triple_gradients(
        &self,
        triple: (usize, usize, usize),
        grad_coeff: f64,
        entity_grads_real: &mut Array2<f64>,
        entity_grads_imag: &mut Array2<f64>,
        relation_grads: &mut Array2<f64>,
    ) {
        let (s, p, o) = triple;
        
        let h_real = self.entity_embeddings_real.row(s);
        let h_imag = self.entity_embeddings_imag.row(s);
        let r_phases = self.relation_phases.row(p);
        let t_real = self.entity_embeddings_real.row(o);
        let t_imag = self.entity_embeddings_imag.row(o);
        
        for (i, ((((&h_r, &h_i), &phase), &t_r), &t_i)) in h_real.iter()
            .zip(h_imag.iter())
            .zip(r_phases.iter())
            .zip(t_real.iter())
            .zip(t_imag.iter())
            .enumerate() {
            
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();
            
            // Rotated head entity
            let rotated_real = h_r * cos_phase - h_i * sin_phase;
            let rotated_imag = h_r * sin_phase + h_i * cos_phase;
            
            // Distance components
            let diff_real = rotated_real - t_r;
            let diff_imag = rotated_imag - t_i;
            
            let distance = (diff_real * diff_real + diff_imag * diff_imag).sqrt();
            
            if distance > 1e-10 {
                let norm_factor = grad_coeff / distance;
                let grad_real = diff_real * norm_factor;
                let grad_imag = diff_imag * norm_factor;
                
                // Gradients w.r.t. head entity (subject)
                entity_grads_real[[s, i]] += grad_real * cos_phase + grad_imag * sin_phase;
                entity_grads_imag[[s, i]] += -grad_real * sin_phase + grad_imag * cos_phase;
                
                // Gradients w.r.t. tail entity (object)
                entity_grads_real[[o, i]] -= grad_real;
                entity_grads_imag[[o, i]] -= grad_imag;
                
                // Gradients w.r.t. relation phases
                let phase_grad = grad_real * (-h_r * sin_phase - h_i * cos_phase) +
                               grad_imag * (h_r * cos_phase - h_i * sin_phase);
                relation_grads[[p, i]] += phase_grad;
            }
        }
    }
    
    /// Generate adversarial negative samples
    fn generate_adversarial_negatives(
        &self,
        positive_triple: (usize, usize, usize),
        num_samples: usize,
        rng: &mut impl Rng,
    ) -> Vec<(usize, usize, usize)> {
        let mut negatives = Vec::new();
        let num_entities = self.base.num_entities();
        
        for _ in 0..num_samples {
            // Choose to corrupt either head or tail
            let corrupt_head = rng.gen_bool(0.5);
            
            if corrupt_head {
                // Sample entity according to adversarial distribution
                let mut candidate_scores = Vec::new();
                for entity_id in 0..num_entities {
                    if entity_id != positive_triple.0 {
                        let neg_triple = (entity_id, positive_triple.1, positive_triple.2);
                        if let Ok(score) = self.score_triple_ids(neg_triple.0, neg_triple.1, neg_triple.2) {
                            candidate_scores.push((entity_id, score));
                        }
                    }
                }
                
                if !candidate_scores.is_empty() {
                    // Use adversarial sampling based on scores
                    let weights: Vec<f64> = candidate_scores.iter()
                        .map(|(_, score)| (-score / self.adversarial_temperature).exp())
                        .collect();
                    
                    let total_weight: f64 = weights.iter().sum();
                    let mut cumulative = 0.0;
                    let threshold = rng.gen::<f64>() * total_weight;
                    
                    for (i, &weight) in weights.iter().enumerate() {
                        cumulative += weight;
                        if cumulative >= threshold {
                            let entity_id = candidate_scores[i].0;
                            negatives.push((entity_id, positive_triple.1, positive_triple.2));
                            break;
                        }
                    }
                }
            } else {
                // Similar logic for corrupting tail
                let mut candidate_scores = Vec::new();
                for entity_id in 0..num_entities {
                    if entity_id != positive_triple.2 {
                        let neg_triple = (positive_triple.0, positive_triple.1, entity_id);
                        if let Ok(score) = self.score_triple_ids(neg_triple.0, neg_triple.1, neg_triple.2) {
                            candidate_scores.push((entity_id, score));
                        }
                    }
                }
                
                if !candidate_scores.is_empty() {
                    let weights: Vec<f64> = candidate_scores.iter()
                        .map(|(_, score)| (-score / self.adversarial_temperature).exp())
                        .collect();
                    
                    let total_weight: f64 = weights.iter().sum();
                    let mut cumulative = 0.0;
                    let threshold = rng.gen::<f64>() * total_weight;
                    
                    for (i, &weight) in weights.iter().enumerate() {
                        cumulative += weight;
                        if cumulative >= threshold {
                            let entity_id = candidate_scores[i].0;
                            negatives.push((positive_triple.0, positive_triple.1, entity_id));
                            break;
                        }
                    }
                }
            }
        }
        
        // Fall back to uniform sampling if adversarial sampling fails
        while negatives.len() < num_samples {
            let corrupt_head = rng.gen_bool(0.5);
            let negative_triple = if corrupt_head {
                let new_head = rng.gen_range(0..num_entities);
                (new_head, positive_triple.1, positive_triple.2)
            } else {
                let new_tail = rng.gen_range(0..num_entities);
                (positive_triple.0, positive_triple.1, new_tail)
            };
            
            if !self.base.has_triple(negative_triple.0, negative_triple.1, negative_triple.2) {
                negatives.push(negative_triple);
            }
        }
        
        negatives
    }
    
    /// Perform one training epoch
    async fn train_epoch(&mut self, learning_rate: f64) -> Result<f64> {
        let mut rng = if let Some(seed) = self.base.config.seed {
            StdRng::seed_from_u64(seed + rand::random::<u64>())
        } else {
            StdRng::from_entropy()
        };
        
        let mut total_loss = 0.0;
        let num_batches = (self.base.triples.len() + self.base.config.batch_size - 1) / self.base.config.batch_size;
        
        let mut shuffled_triples = self.base.triples.clone();
        shuffled_triples.shuffle(&mut rng);
        
        for batch_triples in shuffled_triples.chunks(self.base.config.batch_size) {
            let mut batch_entity_grads_real = Array2::zeros(self.entity_embeddings_real.raw_dim());
            let mut batch_entity_grads_imag = Array2::zeros(self.entity_embeddings_imag.raw_dim());
            let mut batch_relation_grads = Array2::zeros(self.relation_phases.raw_dim());
            let mut batch_loss = 0.0;
            
            for &pos_triple in batch_triples {
                // Use adversarial negative sampling
                let neg_samples = self.generate_adversarial_negatives(
                    pos_triple,
                    self.base.config.negative_samples,
                    &mut rng,
                );
                
                for neg_triple in neg_samples {
                    let pos_score = self.score_triple_ids(pos_triple.0, pos_triple.1, pos_triple.2)?;
                    let neg_score = self.score_triple_ids(neg_triple.0, neg_triple.1, neg_triple.2)?;
                    
                    // Convert scores back to distances for loss computation
                    let pos_distance = -pos_score;
                    let neg_distance = -neg_score;
                    
                    let margin = self.base.config.model_params.get("margin").copied().unwrap_or(6.0);
                    let loss = margin_loss(pos_distance, neg_distance, margin);
                    batch_loss += loss;
                    
                    if loss > 0.0 {
                        let (entity_grads_real, entity_grads_imag, relation_grads) = 
                            self.compute_gradients(pos_triple, neg_triple, pos_score, neg_score)?;
                        
                        batch_entity_grads_real += &entity_grads_real;
                        batch_entity_grads_imag += &entity_grads_imag;
                        batch_relation_grads += &relation_grads;
                    }
                }
            }
            
            // Apply gradients with regularization
            gradient_update(
                &mut self.entity_embeddings_real,
                &batch_entity_grads_real,
                learning_rate,
                self.base.config.l2_reg,
            );
            
            gradient_update(
                &mut self.entity_embeddings_imag,
                &batch_entity_grads_imag,
                learning_rate,
                self.base.config.l2_reg,
            );
            
            gradient_update(
                &mut self.relation_phases,
                &batch_relation_grads,
                learning_rate,
                0.0, // No regularization on phases
            );
            
            // Apply modulus constraint
            if self.modulus_constraint {
                self.apply_modulus_constraint();
            }
            
            // Constrain relation phases to [0, 2π]
            self.relation_phases.mapv_inplace(|x| {
                let mut angle = x % (2.0 * std::f64::consts::PI);
                if angle < 0.0 {
                    angle += 2.0 * std::f64::consts::PI;
                }
                angle
            });
            
            total_loss += batch_loss;
        }
        
        Ok(total_loss / num_batches as f64)
    }
    
    /// Get entity embedding as concatenated real/imaginary vector
    fn get_entity_embedding_vector(&self, entity_id: usize) -> Vector {
        let real_part = self.entity_embeddings_real.row(entity_id);
        let imag_part = self.entity_embeddings_imag.row(entity_id);
        
        let mut values = Vec::with_capacity(real_part.len() * 2);
        for &val in real_part.iter() {
            values.push(val as f32);
        }
        for &val in imag_part.iter() {
            values.push(val as f32);
        }
        
        Vector::new(values)
    }
    
    /// Get relation embedding as phase vector
    fn get_relation_embedding_vector(&self, relation_id: usize) -> Vector {
        let phases = self.relation_phases.row(relation_id);
        let values: Vec<f32> = phases.iter().map(|&x| x as f32).collect();
        Vector::new(values)
    }
}

#[async_trait]
impl EmbeddingModel for RotatE {
    fn config(&self) -> &ModelConfig {
        &self.base.config
    }
    
    fn model_id(&self) -> &Uuid {
        &self.base.model_id
    }
    
    fn model_type(&self) -> &'static str {
        "RotatE"
    }
    
    fn add_triple(&mut self, triple: Triple) -> Result<()> {
        self.base.add_triple(triple)
    }
    
    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let start_time = Instant::now();
        let max_epochs = epochs.unwrap_or(self.base.config.max_epochs);
        
        self.initialize_embeddings();
        
        if !self.embeddings_initialized {
            return Err(anyhow!("No training data available"));
        }
        
        let mut loss_history = Vec::new();
        let learning_rate = self.base.config.learning_rate;
        
        info!("Starting RotatE training for {} epochs", max_epochs);
        
        for epoch in 0..max_epochs {
            let epoch_loss = self.train_epoch(learning_rate).await?;
            loss_history.push(epoch_loss);
            
            if epoch % 100 == 0 {
                debug!("Epoch {}: loss = {:.6}", epoch, epoch_loss);
            }
            
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
        
        let entity_id = self.base.get_entity_id(entity)
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))?;
        
        Ok(self.get_entity_embedding_vector(entity_id))
    }
    
    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }
        
        let relation_id = self.base.get_relation_id(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        
        Ok(self.get_relation_embedding_vector(relation_id))
    }
    
    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let subject_id = self.base.get_entity_id(subject)
            .ok_or_else(|| anyhow!("Subject not found: {}", subject))?;
        let predicate_id = self.base.get_relation_id(predicate)
            .ok_or_else(|| anyhow!("Predicate not found: {}", predicate))?;
        let object_id = self.base.get_entity_id(object)
            .ok_or_else(|| anyhow!("Object not found: {}", object))?;
        
        self.score_triple_ids(subject_id, predicate_id, object_id)
    }
    
    fn predict_objects(&self, subject: &str, predicate: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }
        
        let subject_id = self.base.get_entity_id(subject)
            .ok_or_else(|| anyhow!("Subject not found: {}", subject))?;
        let predicate_id = self.base.get_relation_id(predicate)
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
    
    fn predict_subjects(&self, predicate: &str, object: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }
        
        let predicate_id = self.base.get_relation_id(predicate)
            .ok_or_else(|| anyhow!("Predicate not found: {}", predicate))?;
        let object_id = self.base.get_entity_id(object)
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
    
    fn predict_relations(&self, subject: &str, object: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if !self.embeddings_initialized {
            return Err(anyhow!("Model not trained"));
        }
        
        let subject_id = self.base.get_entity_id(subject)
            .ok_or_else(|| anyhow!("Subject not found: {}", subject))?;
        let object_id = self.base.get_entity_id(object)
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
        self.base.get_stats("RotatE")
    }
    
    fn save(&self, path: &str) -> Result<()> {
        info!("Saving RotatE model to {}", path);
        Ok(())
    }
    
    fn load(&mut self, path: &str) -> Result<()> {
        info!("Loading RotatE model from {}", path);
        Ok(())
    }
    
    fn clear(&mut self) {
        self.base.clear();
        self.entity_embeddings_real = Array2::zeros((0, self.base.config.dimensions));
        self.entity_embeddings_imag = Array2::zeros((0, self.base.config.dimensions));
        self.relation_phases = Array2::zeros((0, self.base.config.dimensions));
        self.embeddings_initialized = false;
    }
    
    fn is_trained(&self) -> bool {
        self.base.is_trained
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::NamedNode;
    
    #[tokio::test]
    async fn test_rotate_basic() -> Result<()> {
        let config = ModelConfig::default()
            .with_dimensions(10)
            .with_max_epochs(5)
            .with_seed(42);
        
        let mut model = RotatE::new(config);
        
        let alice = NamedNode::new("http://example.org/alice")?;
        let knows = NamedNode::new("http://example.org/knows")?;
        let bob = NamedNode::new("http://example.org/bob")?;
        
        model.add_triple(Triple::new(alice.clone(), knows.clone(), bob.clone()))?;
        
        let stats = model.train(Some(3)).await?;
        assert!(stats.epochs_completed > 0);
        
        let alice_emb = model.get_entity_embedding("http://example.org/alice")?;
        assert_eq!(alice_emb.dimensions, 20); // 2 * 10 (real + imaginary)
        
        let score = model.score_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob"
        )?;
        
        assert!(score.is_finite());
        
        Ok(())
    }
}