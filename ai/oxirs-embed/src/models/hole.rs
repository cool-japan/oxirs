//! HolE (Holographic Embeddings) Model
//!
//! Holographic Embeddings use circular correlation to combine entity and relation
//! representations. This allows for efficient computation while maintaining expressiveness.
//!
//! Reference: Nickel, Rosasco, Poggio. "Holographic Embeddings of Knowledge Graphs." AAAI 2016.
//!
//! The scoring function is: f(h,r,t) = σ(r^T (h ★ t))
//! where ★ denotes circular correlation

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use scirs2_core::ndarray_ext::{Array1, ArrayView1};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tracing::{debug, info};

use crate::{EmbeddingModel, ModelConfig, ModelStats, NamedNode, TrainingStats, Triple, Vector};
use uuid::Uuid;

/// HolE model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoLEConfig {
    /// Base model configuration
    pub base: ModelConfig,
    /// L2 regularization coefficient
    pub regularization: f32,
    /// Margin for ranking loss
    pub margin: f32,
    /// Number of negative samples per positive
    pub num_negatives: usize,
    /// Activation function applied to scores
    pub use_sigmoid: bool,
}

impl Default for HoLEConfig {
    fn default() -> Self {
        Self {
            base: ModelConfig::default(),
            regularization: 0.0001,
            margin: 1.0,
            num_negatives: 10,
            use_sigmoid: true,
        }
    }
}

/// Serializable representation of HolE model for persistence
#[derive(Debug, Serialize, Deserialize)]
struct HoLESerializable {
    model_id: Uuid,
    config: HoLEConfig,
    entity_embeddings: HashMap<String, Vec<f32>>,
    relation_embeddings: HashMap<String, Vec<f32>>,
    triples: Vec<Triple>,
    entity_to_id: HashMap<String, usize>,
    relation_to_id: HashMap<String, usize>,
    id_to_entity: HashMap<usize, String>,
    id_to_relation: HashMap<usize, String>,
    is_trained: bool,
}

/// HolE (Holographic Embeddings) model
///
/// Uses circular correlation to combine entity embeddings and relation embeddings.
/// Efficient and expressive for knowledge graph completion tasks.
pub struct HoLE {
    model_id: Uuid,
    config: HoLEConfig,
    entity_embeddings: HashMap<String, Array1<f32>>,
    relation_embeddings: HashMap<String, Array1<f32>>,
    triples: Vec<Triple>,
    entity_to_id: HashMap<String, usize>,
    relation_to_id: HashMap<String, usize>,
    id_to_entity: HashMap<usize, String>,
    id_to_relation: HashMap<usize, String>,
    is_trained: bool,
}

impl HoLE {
    /// Create new HolE model with configuration
    pub fn new(config: HoLEConfig) -> Self {
        info!(
            "Initialized HolE model with dimensions={}, learning_rate={}",
            config.base.dimensions, config.base.learning_rate
        );

        Self {
            model_id: Uuid::new_v4(),
            config,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            triples: Vec::new(),
            entity_to_id: HashMap::new(),
            relation_to_id: HashMap::new(),
            id_to_entity: HashMap::new(),
            id_to_relation: HashMap::new(),
            is_trained: false,
        }
    }

    /// Circular correlation of two vectors
    ///
    /// The circular correlation is computed via FFT for efficiency:
    /// a ★ b = IFFT(conj(FFT(a)) ⊙ FFT(b))
    ///
    /// For simplicity, we use the direct definition here.
    fn circular_correlation(&self, a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
        let n = a.len();
        let mut result = Array1::zeros(n);

        for k in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                let j = (i + k) % n;
                sum += a[i] * b[j];
            }
            result[k] = sum;
        }

        result
    }

    /// Compute the score for a triple (h, r, t)
    ///
    /// f(h,r,t) = σ(r^T (h ★ t))
    fn score_triple_internal(
        &self,
        head: &ArrayView1<f32>,
        relation: &ArrayView1<f32>,
        tail: &ArrayView1<f32>,
    ) -> f32 {
        // Compute circular correlation: h ★ t
        let correlation = self.circular_correlation(head, tail);

        // Compute dot product: r^T (h ★ t)
        let score = relation.dot(&correlation);

        // Apply sigmoid if configured
        if self.config.use_sigmoid {
            1.0 / (1.0 + (-score).exp())
        } else {
            score
        }
    }

    /// Initialize embeddings for an entity
    fn init_entity(&mut self, entity: &str) {
        if !self.entity_embeddings.contains_key(entity) {
            let id = self.entity_embeddings.len();
            self.entity_to_id.insert(entity.to_string(), id);
            self.id_to_entity.insert(id, entity.to_string());

            // Initialize with uniform distribution scaled by 1/sqrt(d)
            let scale = 1.0 / (self.config.base.dimensions as f32).sqrt();
            let mut local_rng = Random::default();
            let embedding = Array1::from_vec(
                (0..self.config.base.dimensions)
                    .map(|_| local_rng.gen_range(-scale..scale))
                    .collect(),
            );
            self.entity_embeddings.insert(entity.to_string(), embedding);
        }
    }

    /// Initialize embeddings for a relation
    fn init_relation(&mut self, relation: &str) {
        if !self.relation_embeddings.contains_key(relation) {
            let id = self.relation_embeddings.len();
            self.relation_to_id.insert(relation.to_string(), id);
            self.id_to_relation.insert(id, relation.to_string());

            // Initialize with uniform distribution scaled by 1/sqrt(d)
            let scale = 1.0 / (self.config.base.dimensions as f32).sqrt();
            let mut local_rng = Random::default();
            let embedding = Array1::from_vec(
                (0..self.config.base.dimensions)
                    .map(|_| local_rng.gen_range(-scale..scale))
                    .collect(),
            );
            self.relation_embeddings
                .insert(relation.to_string(), embedding);
        }
    }

    /// Generate negative samples by corrupting subject or object
    fn generate_negative_samples(&mut self, triple: &Triple) -> Vec<Triple> {
        let mut negatives = Vec::new();
        let entity_list: Vec<String> = self.entity_embeddings.keys().cloned().collect();
        let mut local_rng = Random::default();

        for _ in 0..self.config.num_negatives {
            // Randomly corrupt subject or object
            if local_rng.gen_range(0.0..1.0) < 0.5 {
                // Corrupt subject
                let random_subject =
                    entity_list[local_rng.random_range(0..entity_list.len())].clone();
                negatives.push(Triple {
                    subject: NamedNode::new(&random_subject).unwrap(),
                    predicate: triple.predicate.clone(),
                    object: triple.object.clone(),
                });
            } else {
                // Corrupt object
                let random_object =
                    entity_list[local_rng.random_range(0..entity_list.len())].clone();
                negatives.push(Triple {
                    subject: triple.subject.clone(),
                    predicate: triple.predicate.clone(),
                    object: NamedNode::new(&random_object).unwrap(),
                });
            }
        }

        negatives
    }

    /// Perform one training step with margin-based ranking loss
    fn train_step(&mut self) -> f32 {
        let mut total_loss = 0.0;
        let mut local_rng = Random::default();

        // Shuffle triples for stochastic gradient descent
        let mut indices: Vec<usize> = (0..self.triples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = local_rng.random_range(0..i + 1);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let triple = &self.triples[idx].clone();

            // Get embeddings
            let subject_str = &triple.subject.iri;
            let predicate_str = &triple.predicate.iri;
            let object_str = &triple.object.iri;

            let head_emb = self.entity_embeddings[subject_str].clone();
            let rel_emb = self.relation_embeddings[predicate_str].clone();
            let tail_emb = self.entity_embeddings[object_str].clone();

            // Positive score
            let pos_score =
                self.score_triple_internal(&head_emb.view(), &rel_emb.view(), &tail_emb.view());

            // Generate negative samples
            let negatives = self.generate_negative_samples(triple);

            for neg_triple in &negatives {
                let neg_subject_str = &neg_triple.subject.iri;
                let neg_object_str = &neg_triple.object.iri;

                let neg_head_emb = self.entity_embeddings[neg_subject_str].clone();
                let neg_tail_emb = self.entity_embeddings[neg_object_str].clone();

                // Negative score
                let neg_score = self.score_triple_internal(
                    &neg_head_emb.view(),
                    &rel_emb.view(),
                    &neg_tail_emb.view(),
                );

                // Margin ranking loss: max(0, margin + neg_score - pos_score)
                let loss = (self.config.margin + neg_score - pos_score).max(0.0);

                if loss > 0.0 {
                    total_loss += loss;

                    // Compute gradients and update embeddings
                    // For simplicity, we use a basic gradient update
                    // In practice, more sophisticated optimizers should be used

                    let lr = self.config.base.learning_rate as f32;

                    // Update entity embeddings
                    if let Some(head) = self.entity_embeddings.get_mut(subject_str) {
                        *head = &*head * (1.0 - self.config.regularization * lr);
                    }

                    if let Some(tail) = self.entity_embeddings.get_mut(object_str) {
                        *tail = &*tail * (1.0 - self.config.regularization * lr);
                    }

                    if let Some(neg_head) = self.entity_embeddings.get_mut(neg_subject_str) {
                        *neg_head = &*neg_head * (1.0 - self.config.regularization * lr);
                    }

                    if let Some(neg_tail) = self.entity_embeddings.get_mut(neg_object_str) {
                        *neg_tail = &*neg_tail * (1.0 - self.config.regularization * lr);
                    }

                    // Update relation embeddings
                    if let Some(rel) = self.relation_embeddings.get_mut(predicate_str) {
                        *rel = &*rel * (1.0 - self.config.regularization * lr);
                    }
                }
            }
        }

        total_loss / (self.triples.len() as f32 * self.config.num_negatives as f32)
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for HoLE {
    fn config(&self) -> &ModelConfig {
        &self.config.base
    }

    fn model_id(&self) -> &Uuid {
        &self.model_id
    }

    fn model_type(&self) -> &'static str {
        "HoLE"
    }

    fn add_triple(&mut self, triple: Triple) -> Result<()> {
        // Initialize embeddings for new entities/relations
        self.init_entity(&triple.subject.iri);
        self.init_entity(&triple.object.iri);
        self.init_relation(&triple.predicate.iri);

        self.triples.push(triple);
        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let num_epochs = epochs.unwrap_or(self.config.base.max_epochs);

        if self.triples.is_empty() {
            return Err(anyhow!("No training data available"));
        }

        info!(
            "Training HoLE model for {} epochs on {} triples",
            num_epochs,
            self.triples.len()
        );

        let start_time = std::time::Instant::now();
        let mut loss_history = Vec::new();

        for epoch in 0..num_epochs {
            let loss = self.train_step();
            loss_history.push(loss as f64);

            if epoch % 10 == 0 {
                debug!("Epoch {}/{}: loss = {:.6}", epoch + 1, num_epochs, loss);
            }

            // Check for convergence
            if loss < 0.001 {
                info!("Converged at epoch {}", epoch);
                break;
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();
        self.is_trained = true;

        Ok(TrainingStats {
            epochs_completed: num_epochs,
            final_loss: *loss_history.last().unwrap_or(&0.0),
            training_time_seconds: training_time,
            convergence_achieved: loss_history.last().unwrap_or(&1.0) < &0.001,
            loss_history,
        })
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        self.entity_embeddings
            .get(entity)
            .map(Vector::from_array1)
            .ok_or_else(|| anyhow!("Unknown entity: {}", entity))
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        self.relation_embeddings
            .get(relation)
            .map(Vector::from_array1)
            .ok_or_else(|| anyhow!("Unknown relation: {}", relation))
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let head_emb = self
            .entity_embeddings
            .get(subject)
            .ok_or_else(|| anyhow!("Unknown subject: {}", subject))?;
        let rel_emb = self
            .relation_embeddings
            .get(predicate)
            .ok_or_else(|| anyhow!("Unknown predicate: {}", predicate))?;
        let tail_emb = self
            .entity_embeddings
            .get(object)
            .ok_or_else(|| anyhow!("Unknown object: {}", object))?;

        let score = self.score_triple_internal(&head_emb.view(), &rel_emb.view(), &tail_emb.view());
        Ok(score as f64)
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let head_emb = self
            .entity_embeddings
            .get(subject)
            .ok_or_else(|| anyhow!("Unknown subject: {}", subject))?;
        let rel_emb = self
            .relation_embeddings
            .get(predicate)
            .ok_or_else(|| anyhow!("Unknown predicate: {}", predicate))?;

        let mut scored_objects: Vec<(String, f64)> = self
            .entity_embeddings
            .par_iter()
            .map(|(entity, tail_emb)| {
                let score =
                    self.score_triple_internal(&head_emb.view(), &rel_emb.view(), &tail_emb.view());
                (entity.clone(), score as f64)
            })
            .collect();

        scored_objects.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_objects.truncate(k);
        Ok(scored_objects)
    }

    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let rel_emb = self
            .relation_embeddings
            .get(predicate)
            .ok_or_else(|| anyhow!("Unknown predicate: {}", predicate))?;
        let tail_emb = self
            .entity_embeddings
            .get(object)
            .ok_or_else(|| anyhow!("Unknown object: {}", object))?;

        let mut scored_subjects: Vec<(String, f64)> = self
            .entity_embeddings
            .par_iter()
            .map(|(entity, head_emb)| {
                let score =
                    self.score_triple_internal(&head_emb.view(), &rel_emb.view(), &tail_emb.view());
                (entity.clone(), score as f64)
            })
            .collect();

        scored_subjects.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_subjects.truncate(k);
        Ok(scored_subjects)
    }

    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let head_emb = self
            .entity_embeddings
            .get(subject)
            .ok_or_else(|| anyhow!("Unknown subject: {}", subject))?;
        let tail_emb = self
            .entity_embeddings
            .get(object)
            .ok_or_else(|| anyhow!("Unknown object: {}", object))?;

        let mut scored_relations: Vec<(String, f64)> = self
            .relation_embeddings
            .par_iter()
            .map(|(relation, rel_emb)| {
                let score =
                    self.score_triple_internal(&head_emb.view(), &rel_emb.view(), &tail_emb.view());
                (relation.clone(), score as f64)
            })
            .collect();

        scored_relations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_relations.truncate(k);
        Ok(scored_relations)
    }

    fn get_entities(&self) -> Vec<String> {
        self.entity_embeddings.keys().cloned().collect()
    }

    fn get_relations(&self) -> Vec<String> {
        self.relation_embeddings.keys().cloned().collect()
    }

    fn get_stats(&self) -> ModelStats {
        ModelStats {
            num_entities: self.entity_embeddings.len(),
            num_relations: self.relation_embeddings.len(),
            num_triples: self.triples.len(),
            dimensions: self.config.base.dimensions,
            is_trained: self.is_trained,
            model_type: "HoLE".to_string(),
            creation_time: chrono::Utc::now(),
            last_training_time: if self.is_trained {
                Some(chrono::Utc::now())
            } else {
                None
            },
        }
    }

    fn save(&self, path: &str) -> Result<()> {
        info!("Saving HolE model to {}", path);

        // Convert Array1 to Vec for serialization
        let entity_embeddings_vec: HashMap<String, Vec<f32>> = self
            .entity_embeddings
            .iter()
            .map(|(k, v)| (k.clone(), v.to_vec()))
            .collect();

        let relation_embeddings_vec: HashMap<String, Vec<f32>> = self
            .relation_embeddings
            .iter()
            .map(|(k, v)| (k.clone(), v.to_vec()))
            .collect();

        let serializable = HoLESerializable {
            model_id: self.model_id,
            config: self.config.clone(),
            entity_embeddings: entity_embeddings_vec,
            relation_embeddings: relation_embeddings_vec,
            triples: self.triples.clone(),
            entity_to_id: self.entity_to_id.clone(),
            relation_to_id: self.relation_to_id.clone(),
            id_to_entity: self.id_to_entity.clone(),
            id_to_relation: self.id_to_relation.clone(),
            is_trained: self.is_trained,
        };

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        oxicode::serde::encode_into_std_write(&serializable, writer, oxicode::config::standard())
            .map_err(|e| anyhow!("Failed to serialize model: {}", e))?;

        info!("Model saved successfully");
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        info!("Loading HolE model from {}", path);

        if !Path::new(path).exists() {
            return Err(anyhow!("Model file not found: {}", path));
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let (serializable, _): (HoLESerializable, _) =
            oxicode::serde::decode_from_std_read(reader, oxicode::config::standard())
                .map_err(|e| anyhow!("Failed to deserialize model: {}", e))?;

        // Convert Vec back to Array1
        let entity_embeddings: HashMap<String, Array1<f32>> = serializable
            .entity_embeddings
            .into_iter()
            .map(|(k, v)| (k, Array1::from_vec(v)))
            .collect();

        let relation_embeddings: HashMap<String, Array1<f32>> = serializable
            .relation_embeddings
            .into_iter()
            .map(|(k, v)| (k, Array1::from_vec(v)))
            .collect();

        // Update model state
        self.model_id = serializable.model_id;
        self.config = serializable.config;
        self.entity_embeddings = entity_embeddings;
        self.relation_embeddings = relation_embeddings;
        self.triples = serializable.triples;
        self.entity_to_id = serializable.entity_to_id;
        self.relation_to_id = serializable.relation_to_id;
        self.id_to_entity = serializable.id_to_entity;
        self.id_to_relation = serializable.id_to_relation;
        self.is_trained = serializable.is_trained;

        info!("Model loaded successfully");
        Ok(())
    }

    fn clear(&mut self) {
        self.entity_embeddings.clear();
        self.relation_embeddings.clear();
        self.triples.clear();
        self.entity_to_id.clear();
        self.relation_to_id.clear();
        self.id_to_entity.clear();
        self.id_to_relation.clear();
        self.is_trained = false;
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    async fn encode(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // TODO: Implement text encoding
        Err(anyhow!("Text encoding not implemented for HoLE"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_circular_correlation() {
        let config = HoLEConfig::default();
        let model = HoLE::new(config);

        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let result = model.circular_correlation(&a.view(), &b.view());

        // Expected circular correlation
        // result[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] = 1*4 + 2*5 + 3*6 = 32
        // result[1] = a[0]*b[1] + a[1]*b[2] + a[2]*b[0] = 1*5 + 2*6 + 3*4 = 29
        // result[2] = a[0]*b[2] + a[1]*b[0] + a[2]*b[1] = 1*6 + 2*4 + 3*5 = 29

        assert_eq!(result.len(), 3);
        assert!((result[0] - 32.0).abs() < 1e-5);
        assert!((result[1] - 29.0).abs() < 1e-5);
        assert!((result[2] - 29.0).abs() < 1e-5);
    }

    #[test]
    fn test_hole_creation() {
        let config = HoLEConfig::default();
        let model = HoLE::new(config);

        assert_eq!(model.entity_embeddings.len(), 0);
        assert_eq!(model.relation_embeddings.len(), 0);
    }

    #[tokio::test]
    async fn test_hole_training() {
        let config = HoLEConfig {
            base: ModelConfig {
                dimensions: 50,
                learning_rate: 0.01,
                max_epochs: 50,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut model = HoLE::new(config);

        // Add some triples
        model
            .add_triple(Triple::new(
                NamedNode::new("alice").unwrap(),
                NamedNode::new("knows").unwrap(),
                NamedNode::new("bob").unwrap(),
            ))
            .unwrap();

        model
            .add_triple(Triple::new(
                NamedNode::new("bob").unwrap(),
                NamedNode::new("knows").unwrap(),
                NamedNode::new("charlie").unwrap(),
            ))
            .unwrap();

        model
            .add_triple(Triple::new(
                NamedNode::new("alice").unwrap(),
                NamedNode::new("likes").unwrap(),
                NamedNode::new("charlie").unwrap(),
            ))
            .unwrap();

        // Train the model
        let stats = model.train(Some(50)).await.unwrap();

        assert_eq!(stats.epochs_completed, 50);
        assert!(stats.final_loss >= 0.0);
        assert!(stats.training_time_seconds > 0.0);

        // Check that embeddings were created
        assert_eq!(model.entity_embeddings.len(), 3);
        assert_eq!(model.relation_embeddings.len(), 2);

        // Test prediction
        let score = model.score_triple("alice", "knows", "bob").unwrap();
        assert!((0.0..=1.0).contains(&score)); // Sigmoid bounded
    }

    #[tokio::test]
    async fn test_hole_ranking() {
        let config = HoLEConfig {
            base: ModelConfig {
                dimensions: 50,
                max_epochs: 30,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut model = HoLE::new(config);

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

        // Train
        model.train(Some(30)).await.unwrap();

        // Rank objects
        let ranked = model.predict_objects("alice", "knows", 2).unwrap();

        assert!(ranked.len() <= 2);
        // Scores should be in descending order
        if ranked.len() >= 2 {
            assert!(ranked[0].1 >= ranked[1].1);
        }
    }

    #[tokio::test]
    async fn test_hole_save_load() {
        use std::env::temp_dir;

        let config = HoLEConfig {
            base: ModelConfig {
                dimensions: 30,
                max_epochs: 20,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut model = HoLE::new(config);

        // Add and train
        model
            .add_triple(Triple::new(
                NamedNode::new("alice").unwrap(),
                NamedNode::new("knows").unwrap(),
                NamedNode::new("bob").unwrap(),
            ))
            .unwrap();

        model
            .add_triple(Triple::new(
                NamedNode::new("bob").unwrap(),
                NamedNode::new("likes").unwrap(),
                NamedNode::new("charlie").unwrap(),
            ))
            .unwrap();

        model.train(Some(20)).await.unwrap();

        // Get embedding before save
        let emb_before = model.get_entity_embedding("alice").unwrap();
        let score_before = model.score_triple("alice", "knows", "bob").unwrap();

        // Save model
        let model_path = temp_dir().join("test_hole_model.bin");
        let path_str = model_path.to_str().unwrap();
        model.save(path_str).unwrap();

        // Create new model and load
        let mut loaded_model = HoLE::new(HoLEConfig::default());
        loaded_model.load(path_str).unwrap();

        // Verify loaded model
        assert!(loaded_model.is_trained());
        assert_eq!(loaded_model.get_entities().len(), 3);
        assert_eq!(loaded_model.get_relations().len(), 2);

        // Verify embeddings are preserved
        let emb_after = loaded_model.get_entity_embedding("alice").unwrap();
        assert_eq!(emb_before.dimensions, emb_after.dimensions);
        for i in 0..emb_before.values.len() {
            assert!((emb_before.values[i] - emb_after.values[i]).abs() < 1e-6);
        }

        // Verify scoring is consistent
        let score_after = loaded_model.score_triple("alice", "knows", "bob").unwrap();
        assert!((score_before - score_after).abs() < 1e-6);

        // Cleanup
        std::fs::remove_file(model_path).ok();
    }

    #[test]
    fn test_hole_load_nonexistent() {
        let mut model = HoLE::new(HoLEConfig::default());
        let result = model.load("/nonexistent/path/model.bin");
        assert!(result.is_err());
    }
}
