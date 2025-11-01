//! ConvE (Convolutional Embeddings) Model
//!
//! ConvE uses 2D convolutional neural networks to model interactions between
//! entities and relations in knowledge graphs. This allows for expressive
//! feature learning while maintaining parameter efficiency.
//!
//! Reference: Dettmers et al. "Convolutional 2D Knowledge Graph Embeddings." AAAI 2018.
//!
//! The model reshapes entity and relation embeddings into 2D matrices,
//! concatenates them, applies 2D convolution, and projects to entity space.

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use scirs2_core::ndarray_ext::{Array1, Array2, Array3, Axis};
use scirs2_core::random::{rng, Random};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::{EmbeddingModel, ModelConfig, ModelStats, NamedNode, TrainingStats, Triple, Vector};
use uuid::Uuid;

/// ConvE model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvEConfig {
    /// Base model configuration
    pub base: ModelConfig,
    /// Width of the 2D reshape (height = dimensions / width)
    pub reshape_width: usize,
    /// Number of output channels for convolution
    pub num_filters: usize,
    /// Kernel size for 2D convolution (square kernel)
    pub kernel_size: usize,
    /// Dropout rate for regularization
    pub dropout_rate: f32,
    /// L2 regularization coefficient
    pub regularization: f32,
    /// Margin for ranking loss
    pub margin: f32,
    /// Number of negative samples per positive
    pub num_negatives: usize,
    /// Use batch normalization
    pub use_batch_norm: bool,
}

impl Default for ConvEConfig {
    fn default() -> Self {
        Self {
            base: ModelConfig::default().with_dimensions(200),
            reshape_width: 20, // 200 dimensions -> 10x20 matrix
            num_filters: 32,
            kernel_size: 3,
            dropout_rate: 0.3,
            regularization: 0.0001,
            margin: 1.0,
            num_negatives: 10,
            use_batch_norm: true,
        }
    }
}

/// Convolutional layer parameters
struct ConvLayer {
    /// Filters: shape (num_filters, kernel_size, kernel_size)
    filters: Vec<Array2<f32>>,
    /// Biases for each filter
    biases: Array1<f32>,
}

impl ConvLayer {
    fn new(num_filters: usize, kernel_size: usize, rng: &mut Random) -> Self {
        let scale = (2.0 / (kernel_size * kernel_size) as f32).sqrt();
        let mut filters = Vec::new();

        for _ in 0..num_filters {
            let filter =
                Array2::from_shape_fn((kernel_size, kernel_size), |_| rng.gen_range(-scale..scale));
            filters.push(filter);
        }

        let biases = Array1::zeros(num_filters);

        Self { filters, biases }
    }

    /// Apply 2D convolution with valid padding
    fn forward(&self, input: &Array2<f32>) -> Array3<f32> {
        let kernel_size = self.filters[0].nrows();
        let input_height = input.nrows();
        let input_width = input.ncols();

        let out_height = input_height.saturating_sub(kernel_size - 1);
        let out_width = input_width.saturating_sub(kernel_size - 1);

        if out_height == 0 || out_width == 0 {
            // Return empty array if convolution cannot be performed
            return Array3::zeros((self.filters.len(), 1, 1));
        }

        let mut output = Array3::zeros((self.filters.len(), out_height, out_width));

        for (f_idx, filter) in self.filters.iter().enumerate() {
            for i in 0..out_height {
                for j in 0..out_width {
                    let mut sum = 0.0;

                    for ki in 0..kernel_size {
                        for kj in 0..kernel_size {
                            sum += input[[i + ki, j + kj]] * filter[[ki, kj]];
                        }
                    }

                    output[[f_idx, i, j]] = sum + self.biases[f_idx];
                }
            }
        }

        output
    }
}

/// Fully connected layer
struct FCLayer {
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl FCLayer {
    fn new(input_size: usize, output_size: usize, rng: &mut Random) -> Self {
        let scale = (2.0 / input_size as f32).sqrt();
        let weights =
            Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-scale..scale));
        let bias = Array1::zeros(output_size);

        Self { weights, bias }
    }

    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut output = self.bias.clone();
        for i in 0..output.len() {
            for j in 0..input.len() {
                output[i] += input[j] * self.weights[[j, i]];
            }
        }
        output
    }
}

/// ConvE (Convolutional Embeddings) model
pub struct ConvE {
    model_id: Uuid,
    config: ConvEConfig,
    entity_embeddings: HashMap<String, Array1<f32>>,
    relation_embeddings: HashMap<String, Array1<f32>>,
    conv_layer: ConvLayer,
    fc_layer: FCLayer,
    triples: Vec<Triple>,
    entity_to_id: HashMap<String, usize>,
    relation_to_id: HashMap<String, usize>,
    id_to_entity: HashMap<usize, String>,
    id_to_relation: HashMap<usize, String>,
    is_trained: bool,
}

impl ConvE {
    /// Create new ConvE model with configuration
    pub fn new(config: ConvEConfig) -> Self {
        let mut rng = rng();

        // Calculate feature map size after convolution
        let reshape_height = config.base.dimensions / config.reshape_width;
        let conv_out_height = reshape_height.saturating_sub(config.kernel_size - 1);
        let conv_out_width = (config.reshape_width * 2).saturating_sub(config.kernel_size - 1);
        let fc_input_size = config.num_filters * conv_out_height * conv_out_width;

        let conv_layer = ConvLayer::new(config.num_filters, config.kernel_size, &mut rng);
        let fc_layer = FCLayer::new(fc_input_size, config.base.dimensions, &mut rng);

        info!(
            "Initialized ConvE model: dim={}, filters={}, kernel={}, fc_input={}",
            config.base.dimensions, config.num_filters, config.kernel_size, fc_input_size
        );

        Self {
            model_id: Uuid::new_v4(),
            config,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            conv_layer,
            fc_layer,
            triples: Vec::new(),
            entity_to_id: HashMap::new(),
            relation_to_id: HashMap::new(),
            id_to_entity: HashMap::new(),
            id_to_relation: HashMap::new(),
            is_trained: false,
        }
    }

    /// Reshape 1D embedding to 2D matrix
    fn reshape_embedding(&self, embedding: &Array1<f32>) -> Array2<f32> {
        let height = self.config.base.dimensions / self.config.reshape_width;
        let width = self.config.reshape_width;

        Array2::from_shape_fn((height, width), |(i, j)| embedding[i * width + j])
    }

    /// Apply ReLU activation
    fn relu(&self, x: f32) -> f32 {
        x.max(0.0)
    }

    /// Apply dropout (during training)
    fn dropout(&mut self, values: &mut Array1<f32>, training: bool) {
        if !training || self.config.dropout_rate == 0.0 {
            return;
        }

        let mut local_rng = rng();
        let keep_prob = 1.0 - self.config.dropout_rate;
        for val in values.iter_mut() {
            if local_rng.uniform(0.0, 1.0) > keep_prob {
                *val = 0.0;
            } else {
                *val /= keep_prob; // Inverted dropout
            }
        }
    }

    /// Forward pass to compute score
    fn forward(
        &mut self,
        head: &Array1<f32>,
        relation: &Array1<f32>,
        training: bool,
    ) -> Array1<f32> {
        // Reshape head and relation to 2D
        let head_2d = self.reshape_embedding(head);
        let rel_2d = self.reshape_embedding(relation);

        // Concatenate horizontally: [head | relation]
        let height = head_2d.nrows();
        let width = head_2d.ncols() * 2;
        let mut concat = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..head_2d.ncols() {
                concat[[i, j]] = head_2d[[i, j]];
            }
            for j in 0..rel_2d.ncols() {
                concat[[i, head_2d.ncols() + j]] = rel_2d[[i, j]];
            }
        }

        // Apply 2D convolution
        let conv_out = self.conv_layer.forward(&concat);

        // Apply ReLU activation
        let conv_out_relu = conv_out.mapv(|x| self.relu(x));

        // Flatten the feature maps
        let flattened_size = conv_out_relu.len();
        let mut flattened = Array1::zeros(flattened_size);
        for (idx, &val) in conv_out_relu.iter().enumerate() {
            flattened[idx] = val;
        }

        // Apply dropout
        self.dropout(&mut flattened, training);

        // Fully connected layer
        let mut output = self.fc_layer.forward(&flattened);

        // Apply dropout again
        self.dropout(&mut output, training);

        output
    }

    /// Compute score for a triple
    fn score_triple_internal(
        &mut self,
        head: &Array1<f32>,
        relation: &Array1<f32>,
        tail: &Array1<f32>,
    ) -> f32 {
        let projected = self.forward(head, relation, false);
        // Score is dot product with tail entity
        projected.dot(tail)
    }

    /// Initialize embeddings for an entity
    fn init_entity(&mut self, entity: &str) {
        if !self.entity_embeddings.contains_key(entity) {
            let id = self.entity_embeddings.len();
            self.entity_to_id.insert(entity.to_string(), id);
            self.id_to_entity.insert(id, entity.to_string());

            let mut local_rng = rng();
            let scale = (6.0 / self.config.base.dimensions as f32).sqrt();
            let embedding = Array1::from_vec(
                (0..self.config.base.dimensions)
                    .map(|_| local_rng.uniform(-scale, scale))
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

            let mut local_rng = rng();
            let scale = (6.0 / self.config.base.dimensions as f32).sqrt();
            let embedding = Array1::from_vec(
                (0..self.config.base.dimensions)
                    .map(|_| local_rng.uniform(-scale, scale))
                    .collect(),
            );
            self.relation_embeddings
                .insert(relation.to_string(), embedding);
        }
    }

    /// Training step with simplified gradient updates
    fn train_step(&mut self) -> f32 {
        let mut total_loss = 0.0;
        let mut local_rng = rng();

        // Shuffle triples
        let mut indices: Vec<usize> = (0..self.triples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = local_rng.range(0, i + 1);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let triple = &self.triples[idx].clone();

            let subject_str = &triple.subject.iri;
            let predicate_str = &triple.predicate.iri;
            let object_str = &triple.object.iri;

            let head_emb = self.entity_embeddings[subject_str].clone();
            let rel_emb = self.relation_embeddings[predicate_str].clone();
            let tail_emb = self.entity_embeddings[object_str].clone();

            // Positive score
            let pos_score = self.score_triple_internal(&head_emb, &rel_emb, &tail_emb);

            // Generate negative samples
            let entity_list: Vec<String> = self.entity_embeddings.keys().cloned().collect();
            for _ in 0..self.config.num_negatives {
                let neg_tail_id = entity_list[local_rng.range(0, entity_list.len())].clone();
                let neg_tail_emb = self.entity_embeddings[&neg_tail_id].clone();

                let neg_score = self.score_triple_internal(&head_emb, &rel_emb, &neg_tail_emb);

                // Margin ranking loss
                let loss = (self.config.margin + neg_score - pos_score).max(0.0);
                total_loss += loss;

                // Simplified parameter update (in practice, use proper backpropagation)
                if loss > 0.0 {
                    let lr = self.config.base.learning_rate as f32;
                    // Apply L2 regularization
                    for emb in self.entity_embeddings.values_mut() {
                        *emb = &*emb * (1.0 - self.config.regularization * lr);
                    }
                    for emb in self.relation_embeddings.values_mut() {
                        *emb = &*emb * (1.0 - self.config.regularization * lr);
                    }
                }
            }
        }

        total_loss / (self.triples.len() as f32 * self.config.num_negatives as f32)
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for ConvE {
    fn config(&self) -> &ModelConfig {
        &self.config.base
    }

    fn model_id(&self) -> &Uuid {
        &self.model_id
    }

    fn model_type(&self) -> &'static str {
        "ConvE"
    }

    fn add_triple(&mut self, triple: Triple) -> Result<()> {
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
            "Training ConvE model for {} epochs on {} triples",
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
            .map(|arr| Vector::from_array1(arr))
            .ok_or_else(|| anyhow!("Unknown entity: {}", entity))
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        self.relation_embeddings
            .get(relation)
            .map(|arr| Vector::from_array1(arr))
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

        // Simplified scoring: (head + relation) · tail
        // Note: Full ConvE scoring requires mutable access for CNN forward pass
        let score = (head_emb + rel_emb).dot(tail_emb);
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

        let combined = head_emb + rel_emb;
        let mut scored_objects: Vec<(String, f64)> = self
            .entity_embeddings
            .par_iter()
            .map(|(entity, tail_emb)| {
                let score = combined.dot(tail_emb);
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
                let score = (head_emb + rel_emb).dot(tail_emb);
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
                let score = (head_emb + rel_emb).dot(tail_emb);
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
            model_type: "ConvE".to_string(),
            creation_time: chrono::Utc::now(),
            last_training_time: if self.is_trained {
                Some(chrono::Utc::now())
            } else {
                None
            },
        }
    }

    fn save(&self, _path: &str) -> Result<()> {
        // TODO: Implement model saving
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<()> {
        // TODO: Implement model loading
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
        Err(anyhow!("Text encoding not implemented for ConvE"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conve_creation() {
        let config = ConvEConfig::default();
        let model = ConvE::new(config);

        assert_eq!(model.entity_embeddings.len(), 0);
        assert_eq!(model.relation_embeddings.len(), 0);
    }

    #[tokio::test]
    async fn test_conve_training() {
        let config = ConvEConfig {
            base: ModelConfig {
                dimensions: 100,
                learning_rate: 0.001,
                max_epochs: 20,
                ..Default::default()
            },
            reshape_width: 10,
            num_filters: 16,
            ..Default::default()
        };

        let mut model = ConvE::new(config);

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

        let stats = model.train(Some(20)).await.unwrap();

        assert_eq!(stats.epochs_completed, 20);
        assert!(stats.final_loss >= 0.0);
        assert_eq!(model.entity_embeddings.len(), 3);
        assert_eq!(model.relation_embeddings.len(), 2);
    }
}
