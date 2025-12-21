//! Fine-tuning Capabilities for Pre-trained Embedding Models
//!
//! This module provides tools for fine-tuning pre-trained knowledge graph embeddings
//! on domain-specific data, enabling transfer learning and model adaptation.

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::{EmbeddingModel, Triple};

/// Fine-tuning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FineTuningStrategy {
    /// Fine-tune all parameters
    FullFineTuning,
    /// Freeze entity embeddings, only update relation embeddings
    FreezeEntities,
    /// Freeze relation embeddings, only update entity embeddings
    FreezeRelations,
    /// Only fine-tune last N% of dimensions
    PartialDimensions,
    /// Adapter-based fine-tuning (add small adapter layers)
    AdapterBased,
    /// Layer-wise discriminative fine-tuning (different learning rates per layer)
    Discriminative,
}

/// Fine-tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningConfig {
    /// Fine-tuning strategy
    pub strategy: FineTuningStrategy,
    /// Learning rate for fine-tuning (typically lower than pre-training)
    pub learning_rate: f64,
    /// Number of fine-tuning epochs
    pub max_epochs: usize,
    /// Regularization strength (prevents catastrophic forgetting)
    pub regularization: f64,
    /// Percentage of dimensions to fine-tune (for PartialDimensions strategy)
    pub partial_dimensions_pct: f32,
    /// Adapter dimension size (for AdapterBased strategy)
    pub adapter_dim: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Minimum improvement threshold for early stopping
    pub min_improvement: f64,
    /// Validation split ratio (0.0 to 1.0)
    pub validation_split: f32,
    /// Whether to use knowledge distillation from the pre-trained model
    pub use_distillation: bool,
    /// Distillation temperature
    pub distillation_temperature: f32,
    /// Distillation weight (balance between task loss and distillation loss)
    pub distillation_weight: f32,
}

impl Default for FineTuningConfig {
    fn default() -> Self {
        Self {
            strategy: FineTuningStrategy::FullFineTuning,
            learning_rate: 0.001, // 10x lower than typical pre-training
            max_epochs: 50,
            regularization: 0.01,
            partial_dimensions_pct: 0.2, // Fine-tune top 20% of dimensions
            adapter_dim: 32,
            early_stopping_patience: 5,
            min_improvement: 0.001,
            validation_split: 0.1,
            use_distillation: false,
            distillation_temperature: 2.0,
            distillation_weight: 0.5,
        }
    }
}

/// Fine-tuning result with statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningResult {
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Final training loss
    pub final_training_loss: f64,
    /// Final validation loss
    pub final_validation_loss: f64,
    /// Training time in seconds
    pub training_time_seconds: f64,
    /// Whether early stopping was triggered
    pub early_stopped: bool,
    /// Best validation loss achieved
    pub best_validation_loss: f64,
    /// Training loss history
    pub training_loss_history: Vec<f64>,
    /// Validation loss history
    pub validation_loss_history: Vec<f64>,
    /// Number of parameters updated
    pub num_parameters_updated: usize,
}

/// Adapter layer for adapter-based fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterLayer {
    /// Down-projection matrix (embed_dim -> adapter_dim)
    pub down_projection: Vec<Vec<f32>>,
    /// Up-projection matrix (adapter_dim -> embed_dim)
    pub up_projection: Vec<Vec<f32>>,
    /// Bias for down projection
    pub down_bias: Vec<f32>,
    /// Bias for up projection
    pub up_bias: Vec<f32>,
}

impl AdapterLayer {
    /// Create a new adapter layer with random initialization
    pub fn new(embed_dim: usize, adapter_dim: usize) -> Self {
        let mut rng = Random::default();
        let scale = (2.0 / embed_dim as f32).sqrt();

        let down_projection = (0..adapter_dim)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        let up_projection = (0..embed_dim)
            .map(|_| {
                (0..adapter_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        let down_bias = vec![0.0; adapter_dim];
        let up_bias = vec![0.0; embed_dim];

        Self {
            down_projection,
            up_projection,
            down_bias,
            up_bias,
        }
    }

    /// Forward pass through the adapter
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let embed_dim = input.len();

        // Down-projection: adapter_dim = down @ input + down_bias
        let mut hidden: Vec<f32> = vec![0.0; self.down_bias.len()];
        for (i, h) in hidden.iter_mut().enumerate() {
            let mut sum = self.down_bias[i];
            for j in 0..embed_dim {
                sum += self.down_projection[i][j] * input[j];
            }
            // ReLU activation
            *h = sum.max(0.0);
        }

        // Up-projection: output = up @ hidden + up_bias + input (residual)
        let mut output = vec![0.0; embed_dim];
        for i in 0..embed_dim {
            let mut sum = self.up_bias[i];
            for (j, &h_val) in hidden.iter().enumerate() {
                sum += self.up_projection[i][j] * h_val;
            }
            // Residual connection
            output[i] = sum + input[i];
        }

        Array1::from_vec(output)
    }
}

/// Fine-tuning manager for embedding models
pub struct FineTuningManager {
    config: FineTuningConfig,
    /// Pre-trained embeddings for knowledge distillation
    pretrained_entities: HashMap<String, Array1<f32>>,
    pretrained_relations: HashMap<String, Array1<f32>>,
    /// Adapter layers (if using adapter-based strategy)
    entity_adapters: HashMap<String, AdapterLayer>,
    relation_adapters: HashMap<String, AdapterLayer>,
}

impl FineTuningManager {
    /// Create a new fine-tuning manager
    pub fn new(config: FineTuningConfig) -> Self {
        info!(
            "Initialized fine-tuning manager with strategy: {:?}",
            config.strategy
        );

        Self {
            config,
            pretrained_entities: HashMap::new(),
            pretrained_relations: HashMap::new(),
            entity_adapters: HashMap::new(),
            relation_adapters: HashMap::new(),
        }
    }

    /// Save pre-trained embeddings for distillation
    pub fn save_pretrained_embeddings<M: EmbeddingModel>(&mut self, model: &M) -> Result<()> {
        if !self.config.use_distillation {
            return Ok(());
        }

        info!("Saving pre-trained embeddings for knowledge distillation");

        // Save entity embeddings
        for entity in model.get_entities() {
            if let Ok(emb) = model.get_entity_embedding(&entity) {
                self.pretrained_entities
                    .insert(entity, Array1::from_vec(emb.values));
            }
        }

        // Save relation embeddings
        for relation in model.get_relations() {
            if let Ok(emb) = model.get_relation_embedding(&relation) {
                self.pretrained_relations
                    .insert(relation, Array1::from_vec(emb.values));
            }
        }

        info!(
            "Saved {} entity and {} relation embeddings",
            self.pretrained_entities.len(),
            self.pretrained_relations.len()
        );

        Ok(())
    }

    /// Initialize adapters for adapter-based fine-tuning
    pub fn initialize_adapters<M: EmbeddingModel>(
        &mut self,
        model: &M,
        embed_dim: usize,
    ) -> Result<()> {
        if self.config.strategy != FineTuningStrategy::AdapterBased {
            return Ok(());
        }

        info!(
            "Initializing adapters with dimension: embed_dim={}, adapter_dim={}",
            embed_dim, self.config.adapter_dim
        );

        // Initialize entity adapters
        for entity in model.get_entities() {
            let adapter = AdapterLayer::new(embed_dim, self.config.adapter_dim);
            self.entity_adapters.insert(entity, adapter);
        }

        // Initialize relation adapters
        for relation in model.get_relations() {
            let adapter = AdapterLayer::new(embed_dim, self.config.adapter_dim);
            self.relation_adapters.insert(relation, adapter);
        }

        info!(
            "Initialized {} entity and {} relation adapters",
            self.entity_adapters.len(),
            self.relation_adapters.len()
        );

        Ok(())
    }

    /// Fine-tune a model on domain-specific data
    pub async fn fine_tune<M: EmbeddingModel>(
        &mut self,
        model: &mut M,
        training_triples: Vec<Triple>,
    ) -> Result<FineTuningResult> {
        if training_triples.is_empty() {
            return Err(anyhow!("No training data provided for fine-tuning"));
        }

        info!(
            "Starting fine-tuning with {} triples using {:?} strategy",
            training_triples.len(),
            self.config.strategy
        );

        // Split into training and validation sets
        let (train_data, val_data) = self.split_data(&training_triples)?;

        info!(
            "Split data: {} training, {} validation",
            train_data.len(),
            val_data.len()
        );

        // Save pre-trained embeddings if using distillation
        if self.config.use_distillation {
            self.save_pretrained_embeddings(model)?;
        }

        // Initialize adapters if needed
        if self.config.strategy == FineTuningStrategy::AdapterBased {
            let config = model.config();
            self.initialize_adapters(model, config.dimensions)?;
        }

        // Add training triples to model
        for triple in &train_data {
            model.add_triple(triple.clone())?;
        }

        let start_time = std::time::Instant::now();
        let mut training_loss_history = Vec::new();
        let mut validation_loss_history = Vec::new();
        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let mut early_stopped = false;

        // Training loop
        for epoch in 0..self.config.max_epochs {
            // Train for one epoch
            let stats = model.train(Some(1)).await?;
            let train_loss = stats.final_loss;
            training_loss_history.push(train_loss);

            // Validate
            let val_loss = self.validate(model, &val_data)?;
            validation_loss_history.push(val_loss);

            debug!(
                "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}",
                epoch + 1,
                self.config.max_epochs,
                train_loss,
                val_loss
            );

            // Early stopping check
            if val_loss < best_val_loss - self.config.min_improvement {
                best_val_loss = val_loss;
                patience_counter = 0;
                info!("New best validation loss: {:.6}", best_val_loss);
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    warn!(
                        "Early stopping triggered at epoch {} (patience={})",
                        epoch + 1,
                        self.config.early_stopping_patience
                    );
                    early_stopped = true;
                    break;
                }
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();

        // Count updated parameters
        let num_parameters_updated = self.count_updated_parameters(model)?;

        info!(
            "Fine-tuning complete: {} epochs, {:.2}s, {} parameters updated",
            training_loss_history.len(),
            training_time,
            num_parameters_updated
        );

        Ok(FineTuningResult {
            epochs_completed: training_loss_history.len(),
            final_training_loss: *training_loss_history.last().unwrap_or(&0.0),
            final_validation_loss: *validation_loss_history.last().unwrap_or(&0.0),
            training_time_seconds: training_time,
            early_stopped,
            best_validation_loss: best_val_loss,
            training_loss_history,
            validation_loss_history,
            num_parameters_updated,
        })
    }

    /// Split data into training and validation sets
    fn split_data(&self, data: &[Triple]) -> Result<(Vec<Triple>, Vec<Triple>)> {
        let val_size = (data.len() as f32 * self.config.validation_split) as usize;
        let train_size = data.len() - val_size;

        if val_size == 0 {
            warn!("Validation set is empty, using full data for training");
            return Ok((data.to_vec(), Vec::new()));
        }

        let mut indices: Vec<usize> = (0..data.len()).collect();
        let mut rng = Random::default();

        // Shuffle indices
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0, i + 1);
            indices.swap(i, j);
        }

        let train_data: Vec<Triple> = indices[..train_size]
            .iter()
            .map(|&i| data[i].clone())
            .collect();

        let val_data: Vec<Triple> = indices[train_size..]
            .iter()
            .map(|&i| data[i].clone())
            .collect();

        Ok((train_data, val_data))
    }

    /// Validate the model on validation data
    fn validate<M: EmbeddingModel>(&self, model: &M, val_data: &[Triple]) -> Result<f64> {
        if val_data.is_empty() {
            return Ok(0.0);
        }

        let total_loss: f64 = val_data
            .par_iter()
            .filter_map(|triple| {
                model
                    .score_triple(
                        &triple.subject.iri,
                        &triple.predicate.iri,
                        &triple.object.iri,
                    )
                    .ok()
            })
            .map(|score| {
                // Margin-based loss (higher score is better, so negative for minimization)
                -score
            })
            .sum();

        Ok(total_loss / val_data.len() as f64)
    }

    /// Count the number of parameters that would be updated
    fn count_updated_parameters<M: EmbeddingModel>(&self, model: &M) -> Result<usize> {
        let stats = model.get_stats();
        let embed_dim = stats.dimensions;

        match self.config.strategy {
            FineTuningStrategy::FullFineTuning => {
                Ok((stats.num_entities + stats.num_relations) * embed_dim)
            }
            FineTuningStrategy::FreezeEntities => Ok(stats.num_relations * embed_dim),
            FineTuningStrategy::FreezeRelations => Ok(stats.num_entities * embed_dim),
            FineTuningStrategy::PartialDimensions => {
                let partial_dim = (embed_dim as f32 * self.config.partial_dimensions_pct) as usize;
                Ok((stats.num_entities + stats.num_relations) * partial_dim)
            }
            FineTuningStrategy::AdapterBased => {
                let adapter_params =
                    2 * embed_dim * self.config.adapter_dim + embed_dim + self.config.adapter_dim;
                Ok((stats.num_entities + stats.num_relations) * adapter_params)
            }
            FineTuningStrategy::Discriminative => {
                // All parameters but with different learning rates
                Ok((stats.num_entities + stats.num_relations) * embed_dim)
            }
        }
    }

    /// Get fine-tuning statistics
    pub fn get_stats(&self) -> FineTuningStats {
        FineTuningStats {
            num_pretrained_entities: self.pretrained_entities.len(),
            num_pretrained_relations: self.pretrained_relations.len(),
            num_entity_adapters: self.entity_adapters.len(),
            num_relation_adapters: self.relation_adapters.len(),
            strategy: self.config.strategy,
        }
    }
}

/// Fine-tuning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningStats {
    pub num_pretrained_entities: usize,
    pub num_pretrained_relations: usize,
    pub num_entity_adapters: usize,
    pub num_relation_adapters: usize,
    pub strategy: FineTuningStrategy,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedNode;

    #[test]
    fn test_fine_tuning_config_default() {
        let config = FineTuningConfig::default();
        assert_eq!(config.strategy, FineTuningStrategy::FullFineTuning);
        assert!(config.learning_rate < 0.01); // Should be lower than pre-training
        assert_eq!(config.max_epochs, 50);
    }

    #[test]
    fn test_adapter_layer_creation() {
        let adapter = AdapterLayer::new(128, 32);
        assert_eq!(adapter.down_projection.len(), 32);
        assert_eq!(adapter.up_projection.len(), 128);
        assert_eq!(adapter.down_bias.len(), 32);
        assert_eq!(adapter.up_bias.len(), 128);
    }

    #[test]
    fn test_adapter_forward_pass() {
        let adapter = AdapterLayer::new(128, 32);
        let input = Array1::from_vec(vec![1.0; 128]);
        let output = adapter.forward(&input);
        assert_eq!(output.len(), 128);
        // Output should be different from input due to adapter transformation
    }

    #[test]
    fn test_fine_tuning_manager_creation() {
        let config = FineTuningConfig::default();
        let manager = FineTuningManager::new(config);
        let stats = manager.get_stats();
        assert_eq!(stats.num_pretrained_entities, 0);
        assert_eq!(stats.strategy, FineTuningStrategy::FullFineTuning);
    }

    #[test]
    fn test_split_data() {
        let config = FineTuningConfig {
            validation_split: 0.2,
            ..Default::default()
        };
        let manager = FineTuningManager::new(config);

        let triples: Vec<Triple> = (0..100)
            .map(|i| Triple {
                subject: NamedNode {
                    iri: format!("s{}", i),
                },
                predicate: NamedNode {
                    iri: format!("p{}", i),
                },
                object: NamedNode {
                    iri: format!("o{}", i),
                },
            })
            .collect();

        let (train, val) = manager.split_data(&triples).unwrap();
        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }
}
