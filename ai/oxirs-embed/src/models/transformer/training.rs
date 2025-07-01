//! Training algorithms for transformer embedding models

use super::types::{ModelWeights, TransformerConfig, TransformerTrainingStats};
use crate::{EmbeddingError, Triple, Vector};
use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::{prelude::SliceRandom, Rng};
use std::collections::HashMap;

/// Training manager for transformer embeddings
#[derive(Debug)]
pub struct TransformerTrainer {
    config: TransformerConfig,
    entity_embeddings: HashMap<String, Array1<f32>>,
    relation_embeddings: HashMap<String, Array1<f32>>,
    entity_to_idx: HashMap<String, usize>,
    relation_to_idx: HashMap<String, usize>,
    model_weights: Option<ModelWeights>,
    training_stats: TransformerTrainingStats,
}

impl TransformerTrainer {
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            config,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entity_to_idx: HashMap::new(),
            relation_to_idx: HashMap::new(),
            model_weights: None,
            training_stats: TransformerTrainingStats::default(),
        }
    }

    /// Initialize model weights
    pub fn initialize_weights(&mut self, vocab_size: usize, hidden_size: usize) -> Result<()> {
        self.model_weights = Some(ModelWeights::new(vocab_size, hidden_size));
        Ok(())
    }

    /// Train the model on triples
    pub async fn train(&mut self, triples: &[Triple], epochs: usize) -> Result<()> {
        // Initialize embeddings
        self.initialize_embeddings(triples)?;

        for epoch in 0..epochs {
            self.training_stats.epoch = epoch;

            // Shuffle triples for each epoch
            let mut shuffled_triples = triples.to_vec();
            shuffled_triples.shuffle(&mut rand::thread_rng());

            // Process triples in batches
            let batch_size = 32;
            for (batch_idx, batch) in shuffled_triples.chunks(batch_size).enumerate() {
                self.training_stats.batch_processed = batch_idx;

                // Process each triple in the batch
                for triple in batch {
                    self.process_triple(triple).await?;
                }

                // Apply contrastive learning
                self.contrastive_learning(5).await?;

                // Update training statistics
                self.update_training_stats()?;
            }

            // Apply regularization
            self.apply_regularization()?;
        }

        Ok(())
    }

    /// Initialize embeddings for entities and relations
    fn initialize_embeddings(&mut self, triples: &[Triple]) -> Result<()> {
        let dimensions = self.config.base_config.dimensions;
        let mut rng = rand::thread_rng();

        // Collect unique entities and relations
        let mut entities = std::collections::HashSet::new();
        let mut relations = std::collections::HashSet::new();

        for triple in triples {
            entities.insert(triple.subject.iri.clone());
            entities.insert(triple.object.iri.clone());
            relations.insert(triple.predicate.iri.clone());
        }

        // Initialize entity embeddings
        for (idx, entity) in entities.iter().enumerate() {
            let embedding =
                Array1::from_vec((0..dimensions).map(|_| rng.gen_range(-0.1..0.1)).collect());
            self.entity_embeddings.insert(entity.clone(), embedding);
            self.entity_to_idx.insert(entity.clone(), idx);
        }

        // Initialize relation embeddings
        for (idx, relation) in relations.iter().enumerate() {
            let embedding =
                Array1::from_vec((0..dimensions).map(|_| rng.gen_range(-0.1..0.1)).collect());
            self.relation_embeddings.insert(relation.clone(), embedding);
            self.relation_to_idx.insert(relation.clone(), idx);
        }

        Ok(())
    }

    /// Process a single triple during training
    async fn process_triple(&mut self, triple: &Triple) -> Result<()> {
        let subject_key = &triple.subject.iri;
        let predicate_key = &triple.predicate.iri;
        let object_key = &triple.object.iri;

        // Get embeddings
        let subject_emb = self.entity_embeddings.get(subject_key).cloned();
        let predicate_emb = self.relation_embeddings.get(predicate_key).cloned();
        let object_emb = self.entity_embeddings.get(object_key).cloned();

        if let (Some(s_emb), Some(p_emb), Some(o_emb)) = (subject_emb, predicate_emb, object_emb) {
            // Compute TransE-style loss: ||h + r - t||²
            let predicted = &s_emb + &p_emb;
            let diff = &predicted - &o_emb;
            let loss = diff.mapv(|x| x * x).sum();

            // Apply gradient updates
            let learning_rate = self.config.base_config.learning_rate as f32;
            self.apply_gradient_updates(&s_emb, &p_emb, &o_emb, &diff, learning_rate)?;

            // Update loss statistics
            self.training_stats.reconstruction_loss = loss;
        }

        Ok(())
    }

    /// Apply gradient updates to embeddings
    fn apply_gradient_updates(
        &mut self,
        subject_emb: &Array1<f32>,
        predicate_emb: &Array1<f32>,
        object_emb: &Array1<f32>,
        diff: &Array1<f32>,
        learning_rate: f32,
    ) -> Result<()> {
        // Gradient for subject: 2 * diff
        let subject_gradient = diff * 2.0;

        // Gradient for predicate: 2 * diff
        let predicate_gradient = diff * 2.0;

        // Gradient for object: -2 * diff
        let object_gradient = diff * -2.0;

        // Update embeddings (gradient descent)
        // Note: In practice, you'd want to track which triple corresponds to which embeddings
        // This is a simplified version for demonstration

        // Update statistics
        let gradient_norm = subject_gradient.mapv(|x| x * x).sum().sqrt()
            + predicate_gradient.mapv(|x| x * x).sum().sqrt()
            + object_gradient.mapv(|x| x * x).sum().sqrt();

        self.training_stats.gradient_norm = gradient_norm;
        self.training_stats.learning_rate = learning_rate;

        Ok(())
    }

    /// Advanced contrastive learning for better semantic representations
    pub async fn contrastive_learning(&mut self, negative_samples: usize) -> Result<()> {
        let temperature = 0.07;
        let learning_rate = self.config.base_config.learning_rate as f32 * 0.5;

        // Create a vector of entity keys for negative sampling
        let entity_keys: Vec<String> = self.entity_embeddings.keys().cloned().collect();

        if entity_keys.len() < 2 {
            return Ok(()); // Need at least 2 entities for contrastive learning
        }

        // Process pairs of entities for contrastive learning
        for (i, entity1) in entity_keys.iter().enumerate() {
            for entity2 in entity_keys.iter().skip(i + 1) {
                if let (Some(emb1), Some(emb2)) = (
                    self.entity_embeddings.get(entity1).cloned(),
                    self.entity_embeddings.get(entity2).cloned(),
                ) {
                    // Normalize embeddings for better cosine similarity
                    let norm1 = emb1.mapv(|x| x * x).sum().sqrt();
                    let norm2 = emb2.mapv(|x| x * x).sum().sqrt();

                    if norm1 > 0.0 && norm2 > 0.0 {
                        let norm_factor = norm1 * norm2;

                        // Positive sample score (cosine similarity)
                        let positive_score = (&emb1 * &emb2).sum() / (norm_factor * temperature);

                        // Generate negative samples
                        let mut negative_scores = Vec::new();
                        for _ in 0..negative_samples {
                            if let Some(neg_entity) = entity_keys.choose(&mut rand::thread_rng()) {
                                if neg_entity != entity1 && neg_entity != entity2 {
                                    if let Some(neg_emb) = self.entity_embeddings.get(neg_entity) {
                                        let neg_norm = neg_emb.mapv(|x| x * x).sum().sqrt();
                                        if neg_norm > 0.0 {
                                            let neg_norm_factor = norm1 * neg_norm;
                                            let neg_score = (&emb1 * neg_emb).sum()
                                                / (neg_norm_factor * temperature);
                                            negative_scores.push(neg_score);
                                        }
                                    }
                                }
                            }
                        }

                        // Compute contrastive loss and update embeddings
                        if !negative_scores.is_empty() {
                            let max_neg_score = negative_scores
                                .iter()
                                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let loss_gradient = positive_score - max_neg_score;

                            // Use sigmoid for smoother gradients
                            let gradient_factor = if loss_gradient.abs() < 0.001 {
                                0.01 // Minimum update to ensure embedding changes
                            } else {
                                (loss_gradient / (1.0 + loss_gradient.abs())).clamp(-0.1, 0.1)
                            };

                            // Update embeddings based on contrastive loss
                            let emb1_update = &emb1 + &(&emb2 * learning_rate * gradient_factor);
                            let emb2_update = &emb2 + &(&emb1 * learning_rate * gradient_factor);

                            self.entity_embeddings.insert(entity1.clone(), emb1_update);
                            self.entity_embeddings.insert(entity2.clone(), emb2_update);

                            // Update training statistics
                            self.training_stats.contrastive_loss = loss_gradient.abs();
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply regularization to prevent overfitting
    fn apply_regularization(&mut self) -> Result<()> {
        let reg_strength = 0.01;
        let mut total_reg_loss = 0.0;

        // L2 regularization for entity embeddings
        for (_, embedding) in self.entity_embeddings.iter_mut() {
            let reg_loss = embedding.mapv(|x| x * x).sum() * reg_strength;
            total_reg_loss += reg_loss;

            // Apply regularization gradient
            *embedding = embedding.mapv(|x| x * (1.0 - reg_strength));
        }

        // L2 regularization for relation embeddings
        for (_, embedding) in self.relation_embeddings.iter_mut() {
            let reg_loss = embedding.mapv(|x| x * x).sum() * reg_strength;
            total_reg_loss += reg_loss;

            // Apply regularization gradient
            *embedding = embedding.mapv(|x| x * (1.0 - reg_strength));
        }

        self.training_stats.regularization_loss = total_reg_loss;
        Ok(())
    }

    /// Update training statistics
    fn update_training_stats(&mut self) -> Result<()> {
        // Compute average embedding norms
        let mut entity_norm_sum = 0.0;
        let mut entity_count = 0;

        for embedding in self.entity_embeddings.values() {
            entity_norm_sum += embedding.mapv(|x| x * x).sum().sqrt();
            entity_count += 1;
        }

        if entity_count > 0 {
            let avg_entity_norm = entity_norm_sum / entity_count as f32;
            // Store in some statistics structure if needed
        }

        Ok(())
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> &TransformerTrainingStats {
        &self.training_stats
    }

    /// Get entity embeddings
    pub fn get_entity_embeddings(&self) -> &HashMap<String, Array1<f32>> {
        &self.entity_embeddings
    }

    /// Get relation embeddings
    pub fn get_relation_embeddings(&self) -> &HashMap<String, Array1<f32>> {
        &self.relation_embeddings
    }

    /// Set entity embedding
    pub fn set_entity_embedding(&mut self, entity: String, embedding: Array1<f32>) {
        self.entity_embeddings.insert(entity, embedding);
    }

    /// Set relation embedding
    pub fn set_relation_embedding(&mut self, relation: String, embedding: Array1<f32>) {
        self.relation_embeddings.insert(relation, embedding);
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        !self.entity_embeddings.is_empty() && !self.relation_embeddings.is_empty()
    }

    /// Reset training state
    pub fn reset(&mut self) {
        self.entity_embeddings.clear();
        self.relation_embeddings.clear();
        self.entity_to_idx.clear();
        self.relation_to_idx.clear();
        self.model_weights = None;
        self.training_stats = TransformerTrainingStats::default();
    }

    /// Get model configuration
    pub fn get_config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Update model configuration
    pub fn update_config(&mut self, config: TransformerConfig) {
        self.config = config;
    }
}

/// Advanced training scheduler for learning rate adjustment
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    initial_lr: f32,
    schedule_type: String,
    warmup_steps: usize,
    current_step: usize,
}

impl LearningRateScheduler {
    pub fn new(initial_lr: f32, schedule_type: String, warmup_steps: usize) -> Self {
        Self {
            initial_lr,
            schedule_type,
            warmup_steps,
            current_step: 0,
        }
    }

    pub fn get_learning_rate(&self) -> f32 {
        match self.schedule_type.as_str() {
            "linear" => self.linear_schedule(),
            "cosine" => self.cosine_schedule(),
            "polynomial" => self.polynomial_schedule(),
            _ => self.initial_lr,
        }
    }

    fn linear_schedule(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            self.initial_lr * (self.current_step as f32 / self.warmup_steps as f32)
        } else {
            self.initial_lr
                * (1.0 - (self.current_step - self.warmup_steps) as f32 / 10000.0).max(0.1)
        }
    }

    fn cosine_schedule(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            self.initial_lr * (self.current_step as f32 / self.warmup_steps as f32)
        } else {
            let progress = (self.current_step - self.warmup_steps) as f32 / 10000.0;
            self.initial_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }

    fn polynomial_schedule(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            self.initial_lr * (self.current_step as f32 / self.warmup_steps as f32)
        } else {
            let progress = (self.current_step - self.warmup_steps) as f32 / 10000.0;
            self.initial_lr * (1.0 - progress).powf(2.0).max(0.1)
        }
    }

    pub fn step(&mut self) {
        self.current_step += 1;
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::transformer::types::TransformerType;
    use crate::NamedNode;

    #[tokio::test]
    async fn test_trainer_initialization() {
        let config = TransformerConfig::default();
        let mut trainer = TransformerTrainer::new(config);

        assert!(trainer.initialize_weights(1000, 768).is_ok());
        assert!(!trainer.is_trained());
    }

    #[tokio::test]
    async fn test_contrastive_learning() {
        let config = TransformerConfig::default();
        let mut trainer = TransformerTrainer::new(config);

        // Add some test embeddings
        let emb1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let emb2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        trainer.set_entity_embedding("entity1".to_string(), emb1);
        trainer.set_entity_embedding("entity2".to_string(), emb2);

        assert!(trainer.contrastive_learning(3).await.is_ok());
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let mut scheduler = LearningRateScheduler::new(0.001, "linear".to_string(), 100);

        // Test warmup phase
        let lr_start = scheduler.get_learning_rate();
        assert_eq!(lr_start, 0.0);

        scheduler.step();
        let lr_warmup = scheduler.get_learning_rate();
        assert!(lr_warmup > 0.0 && lr_warmup < 0.001);

        // Skip to end of warmup
        scheduler.current_step = 100;
        let lr_end_warmup = scheduler.get_learning_rate();
        assert_eq!(lr_end_warmup, 0.001);
    }
}
