use super::{
    EmbeddingConfig, KnowledgeGraphEmbedding, KnowledgeGraphMetrics, TrainingConfig,
    TrainingMetrics,
};
use crate::model::Triple;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::{Random, Rng};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// TransE embedding model
pub struct TransE {
    /// Model configuration
    config: EmbeddingConfig,

    /// Entity embeddings
    entity_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,

    /// Relation embeddings
    relation_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,

    /// Entity vocabulary
    entity_vocab: HashMap<String, usize>,

    /// Relation vocabulary
    relation_vocab: HashMap<String, usize>,

    /// Training state
    trained: bool,
}

impl TransE {
    /// Create new TransE model
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            entity_embeddings: Arc::new(RwLock::new(HashMap::new())),
            relation_embeddings: Arc::new(RwLock::new(HashMap::new())),
            entity_vocab: HashMap::new(),
            relation_vocab: HashMap::new(),
            trained: false,
        }
    }

    /// Initialize embeddings from vocabulary
    async fn initialize_embeddings(&mut self, triples: &[Triple]) -> Result<()> {
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();

        // Collect vocabulary
        for triple in triples {
            entities.insert(triple.subject().to_string());
            entities.insert(triple.object().to_string());
            relations.insert(triple.predicate().to_string());
        }

        // Create vocabularies
        self.entity_vocab = entities
            .iter()
            .enumerate()
            .map(|(i, entity)| (entity.clone(), i))
            .collect();

        self.relation_vocab = relations
            .iter()
            .enumerate()
            .map(|(i, relation)| (relation.clone(), i))
            .collect();

        // Initialize embeddings with Xavier initialization
        let mut entity_embs = self.entity_embeddings.write().await;
        let mut relation_embs = self.relation_embeddings.write().await;

        let bound = (6.0 / self.config.embedding_dim as f32).sqrt();

        for entity in entities {
            let embedding = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            entity_embs.insert(entity, embedding);
        }

        for relation in relations {
            let embedding = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            relation_embs.insert(relation, embedding);
        }

        Ok(())
    }

    /// Compute TransE score: ||h + r - t||
    async fn compute_score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;

        let h = entity_embs
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let r = relation_embs
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let t = entity_embs
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;

        // Compute ||h + r - t||_L1 or ||h + r - t||_L2
        let diff = h + r - t;
        let score = diff.mapv(|x| x.abs()).sum(); // L1 norm

        Ok(score)
    }

    /// Generate negative samples
    fn generate_negative_samples(
        &self,
        positive_triples: &[(String, String, String)],
        num_negatives: usize,
    ) -> Vec<(String, String, String)> {
        let mut negatives = Vec::new();
        let entities: Vec<String> = self.entity_vocab.keys().cloned().collect();
        let _relations: Vec<String> = self.relation_vocab.keys().cloned().collect();

        for _ in 0..num_negatives {
            // Randomly corrupt head or tail
            let positive_idx = {
                let mut rng = Random::default();
                rng.random_range(0, positive_triples.len())
            };
            let (h, r, t) = &positive_triples[positive_idx];

            let should_corrupt_head = {
                let mut rng = Random::default();
                rng.random_bool_with_chance(0.5)
            };
            if should_corrupt_head {
                // Corrupt head
                let new_head_idx = {
                    let mut rng = Random::default();
                    rng.random_range(0, entities.len())
                };
                let new_head = &entities[new_head_idx];
                if new_head != h {
                    negatives.push((new_head.clone(), r.clone(), t.clone()));
                }
            } else {
                // Corrupt tail
                let new_tail_idx = {
                    let mut rng = Random::default();
                    rng.random_range(0, entities.len())
                };
                let new_tail = &entities[new_tail_idx];
                if new_tail != t {
                    negatives.push((h.clone(), r.clone(), new_tail.clone()));
                }
            }
        }

        negatives
    }

    /// Calculate accuracy on validation triples
    async fn calculate_accuracy(&self, triples: &[(String, String, String)]) -> Result<f32> {
        if triples.is_empty() {
            return Ok(0.0);
        }

        let mut correct = 0;
        let total = triples.len().min(100); // Sample for efficiency

        for triple in triples.iter().take(total) {
            let positive_score = self.compute_score(&triple.0, &triple.1, &triple.2).await?;

            // Generate a random negative and compare
            let entities: Vec<String> = self.entity_vocab.keys().cloned().collect();
            if entities.len() >= 2 {
                let corrupt_idx = {
                    let mut rng = Random::default();
                    rng.random_range(0, entities.len())
                };
                let corrupt_entity = &entities[corrupt_idx];

                let should_corrupt_head = {
                    let mut rng = Random::default();
                    rng.random_bool_with_chance(0.5)
                };
                let negative_score = if should_corrupt_head {
                    self.compute_score(corrupt_entity, &triple.1, &triple.2)
                        .await?
                } else {
                    self.compute_score(&triple.0, &triple.1, corrupt_entity)
                        .await?
                };

                // For TransE, lower score is better
                if positive_score < negative_score {
                    correct += 1;
                }
            }
        }

        Ok(correct as f32 / total as f32)
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for TransE {
    async fn generate_embeddings(&self, triples: &[Triple]) -> Result<Vec<Vec<f32>>> {
        let entity_embs = self.entity_embeddings.read().await;
        let mut embeddings = Vec::new();

        for triple in triples {
            let subject_str = triple.subject().to_string();
            let object_str = triple.object().to_string();
            let head_emb = entity_embs
                .get(&subject_str)
                .ok_or_else(|| anyhow!("Entity not found"))?;
            let tail_emb = entity_embs
                .get(&object_str)
                .ok_or_else(|| anyhow!("Entity not found"))?;

            // Combine head and tail embeddings
            let combined: Vec<f32> = head_emb
                .iter()
                .zip(tail_emb.iter())
                .map(|(h, t)| (h + t) / 2.0)
                .collect();

            embeddings.push(combined);
        }

        Ok(embeddings)
    }

    async fn score_triple(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        self.compute_score(head, relation, tail).await
    }

    async fn predict_links(
        &self,
        entities: &[String],
        relations: &[String],
    ) -> Result<Vec<(String, String, String, f32)>> {
        let mut predictions = Vec::new();

        // Generate all possible triples and score them
        for head in entities {
            for relation in relations {
                for tail in entities {
                    if head != tail {
                        let score = self.score_triple(head, relation, tail).await?;
                        predictions.push((head.clone(), relation.clone(), tail.clone(), score));
                    }
                }
            }
        }

        // Sort by score (lower is better for TransE)
        predictions.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

        Ok(predictions)
    }

    async fn get_entity_embedding(&self, entity: &str) -> Result<Vec<f32>> {
        let entity_embs = self.entity_embeddings.read().await;
        let embedding = entity_embs
            .get(entity)
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))?;

        Ok(embedding.to_vec())
    }

    async fn get_relation_embedding(&self, relation: &str) -> Result<Vec<f32>> {
        let relation_embs = self.relation_embeddings.read().await;
        let embedding = relation_embs
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;

        Ok(embedding.to_vec())
    }

    async fn train(
        &mut self,
        triples: &[Triple],
        config: &TrainingConfig,
    ) -> Result<TrainingMetrics> {
        use std::time::Instant;

        let start_time = Instant::now();

        // Initialize embeddings
        self.initialize_embeddings(triples).await?;

        // Convert triples to string format
        let triple_strings: Vec<(String, String, String)> = triples
            .iter()
            .map(|t| {
                (
                    t.subject().to_string(),
                    t.predicate().to_string(),
                    t.object().to_string(),
                )
            })
            .collect();

        // Split into training and validation sets
        let val_size = (triple_strings.len() as f32 * config.validation_split) as usize;
        let train_triples = &triple_strings[val_size..];
        let val_triples = &triple_strings[..val_size];

        // Use margin-based loss with fixed margin (can be made configurable later)
        let margin = 1.0;

        let learning_rate = config.learning_rate;
        let batch_size = config.batch_size;

        // Initialize optimizer states for Adam (default optimizer)
        let mut entity_m1: HashMap<String, Array1<f32>> = HashMap::new();
        let mut entity_m2: HashMap<String, Array1<f32>> = HashMap::new();
        let mut relation_m1: HashMap<String, Array1<f32>> = HashMap::new();
        let mut relation_m2: HashMap<String, Array1<f32>> = HashMap::new();

        // Adam hyperparameters (standard defaults)
        let (beta1, beta2, epsilon) = (0.9, 0.999, 1e-8);
        let weight_decay = 0.0001;

        let mut loss_history = Vec::new();
        let mut best_val_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let mut final_epoch = 0;

        for epoch in 0..config.max_epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Shuffle training data using Fisher-Yates shuffle
            let mut shuffled_indices: Vec<usize> = (0..train_triples.len()).collect();
            {
                use scirs2_core::random::Random;
                let mut rng = Random::default();
                for i in (1..shuffled_indices.len()).rev() {
                    let j = rng.random_range(0, i + 1);
                    shuffled_indices.swap(i, j);
                }
            } // RNG dropped here before any await points

            // Process mini-batches
            for batch_start in (0..train_triples.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(train_triples.len());
                let batch_indices = &shuffled_indices[batch_start..batch_end];

                // Collect batch triples
                let batch_triples: Vec<_> =
                    batch_indices.iter().map(|&i| &train_triples[i]).collect();

                // Generate negative samples for this batch
                let negatives = self.generate_negative_samples(
                    &batch_triples
                        .iter()
                        .map(|t| (*t).clone())
                        .collect::<Vec<_>>(),
                    batch_triples.len(),
                );

                // Accumulate gradients for this batch
                let mut entity_gradients: HashMap<String, Array1<f32>> = HashMap::new();
                let mut relation_gradients: HashMap<String, Array1<f32>> = HashMap::new();

                let mut batch_loss = 0.0;

                for (i, positive) in batch_triples.iter().enumerate() {
                    if i >= negatives.len() {
                        continue;
                    }

                    // Get embeddings
                    let entity_embs = self.entity_embeddings.read().await;
                    let relation_embs = self.relation_embeddings.read().await;

                    let h_pos = entity_embs
                        .get(&positive.0)
                        .ok_or_else(|| anyhow!("Entity not found: {}", positive.0))?
                        .clone();
                    let r = relation_embs
                        .get(&positive.1)
                        .ok_or_else(|| anyhow!("Relation not found: {}", positive.1))?
                        .clone();
                    let t_pos = entity_embs
                        .get(&positive.2)
                        .ok_or_else(|| anyhow!("Entity not found: {}", positive.2))?
                        .clone();

                    let (h_neg_key, _, t_neg_key) = &negatives[i];
                    let h_neg = entity_embs
                        .get(h_neg_key)
                        .cloned()
                        .unwrap_or_else(|| h_pos.clone());
                    let t_neg = entity_embs
                        .get(t_neg_key)
                        .cloned()
                        .unwrap_or_else(|| t_pos.clone());

                    drop(entity_embs);
                    drop(relation_embs);

                    // Compute scores
                    let diff_pos = &h_pos + &r - &t_pos;
                    let score_pos = diff_pos.mapv(|x| x.abs()).sum();

                    let diff_neg = &h_neg + &r - &t_neg;
                    let score_neg = diff_neg.mapv(|x| x.abs()).sum();

                    // Margin-based loss: max(0, score_pos - score_neg + margin)
                    let loss = (score_pos - score_neg + margin).max(0.0);
                    batch_loss += loss;

                    // Compute gradients only if loss > 0
                    if loss > 0.0 {
                        // Gradient for positive triple: sign(h + r - t)
                        let grad_pos = diff_pos.mapv(|x| x.signum());

                        // Gradient for negative triple: -sign(h' + r - t')
                        let grad_neg = diff_neg.mapv(|x| -x.signum());

                        // Accumulate gradients for positive triple
                        *entity_gradients
                            .entry(positive.0.clone())
                            .or_insert_with(|| Array1::zeros(self.config.embedding_dim)) +=
                            &grad_pos;
                        *relation_gradients
                            .entry(positive.1.clone())
                            .or_insert_with(|| Array1::zeros(self.config.embedding_dim)) +=
                            &grad_pos;
                        *entity_gradients
                            .entry(positive.2.clone())
                            .or_insert_with(|| Array1::zeros(self.config.embedding_dim)) -=
                            &grad_pos;

                        // Accumulate gradients for negative triple
                        *entity_gradients
                            .entry(h_neg_key.clone())
                            .or_insert_with(|| Array1::zeros(self.config.embedding_dim)) +=
                            &grad_neg;
                        *entity_gradients
                            .entry(t_neg_key.clone())
                            .or_insert_with(|| Array1::zeros(self.config.embedding_dim)) -=
                            &grad_neg;
                    }
                }

                epoch_loss += batch_loss;
                num_batches += 1;

                // Apply gradients to update embeddings
                let mut entity_embs = self.entity_embeddings.write().await;
                let mut relation_embs = self.relation_embeddings.write().await;

                // Update entity embeddings with Adam optimizer
                for (entity, gradient) in entity_gradients {
                    if let Some(embedding) = entity_embs.get_mut(&entity) {
                        // Adam optimizer with bias correction
                        let m1 = entity_m1
                            .entry(entity.clone())
                            .or_insert_with(|| Array1::zeros(self.config.embedding_dim));
                        let m2 = entity_m2
                            .entry(entity.clone())
                            .or_insert_with(|| Array1::zeros(self.config.embedding_dim));

                        // Update biased first moment estimate
                        *m1 = &*m1 * beta1 + &gradient * (1.0 - beta1);

                        // Update biased second raw moment estimate
                        *m2 = &*m2 * beta2 + &gradient.mapv(|g| g * g) * (1.0 - beta2);

                        // Compute bias-corrected moment estimates
                        let t = epoch as f32 + 1.0;
                        let m1_hat = &*m1 / (1.0 - beta1.powf(t));
                        let m2_hat = &*m2 / (1.0 - beta2.powf(t));

                        // Update parameters (element-wise division)
                        for i in 0..embedding.len() {
                            let update = learning_rate * m1_hat[i] / (m2_hat[i].sqrt() + epsilon);
                            embedding[i] -= update;
                        }

                        // Apply weight decay
                        if weight_decay > 0.0 {
                            *embedding = &*embedding * (1.0 - learning_rate * weight_decay);
                        }

                        // Normalize embeddings (important for TransE)
                        let norm = embedding.mapv(|x| x * x).sum().sqrt();
                        if norm > 1.0 {
                            *embedding = &*embedding / norm;
                        }
                    }
                }

                // Update relation embeddings with Adam optimizer
                for (relation, gradient) in relation_gradients {
                    if let Some(embedding) = relation_embs.get_mut(&relation) {
                        let m1 = relation_m1
                            .entry(relation.clone())
                            .or_insert_with(|| Array1::zeros(self.config.embedding_dim));
                        let m2 = relation_m2
                            .entry(relation.clone())
                            .or_insert_with(|| Array1::zeros(self.config.embedding_dim));

                        *m1 = &*m1 * beta1 + &gradient * (1.0 - beta1);
                        *m2 = &*m2 * beta2 + &gradient.mapv(|g| g * g) * (1.0 - beta2);

                        let t = epoch as f32 + 1.0;
                        let m1_hat = &*m1 / (1.0 - beta1.powf(t));
                        let m2_hat = &*m2 / (1.0 - beta2.powf(t));

                        // Update parameters (element-wise division)
                        for i in 0..embedding.len() {
                            let update = learning_rate * m1_hat[i] / (m2_hat[i].sqrt() + epsilon);
                            embedding[i] -= update;
                        }

                        // Apply weight decay
                        if weight_decay > 0.0 {
                            *embedding = &*embedding * (1.0 - learning_rate * weight_decay);
                        }
                    }
                }
            }

            // Calculate average epoch loss
            let avg_epoch_loss = if num_batches > 0 {
                epoch_loss / num_batches as f32
            } else {
                0.0
            };
            loss_history.push(avg_epoch_loss);
            final_epoch = epoch;

            // Validation (run every 10 epochs)
            let validation_frequency = 10;
            if epoch % validation_frequency == 0 && !val_triples.is_empty() {
                let mut val_loss = 0.0;
                for triple in val_triples {
                    let score = self.compute_score(&triple.0, &triple.1, &triple.2).await?;
                    val_loss += score;
                }
                val_loss /= val_triples.len() as f32;

                // Early stopping with patience
                let min_delta = 1e-4;
                if val_loss < best_val_loss - min_delta {
                    best_val_loss = val_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= config.patience {
                        tracing::info!("Early stopping triggered at epoch {}", epoch);
                        break;
                    }
                }

                // Log every 10 epochs
                if epoch % 10 == 0 {
                    tracing::info!(
                        "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}",
                        epoch,
                        config.max_epochs,
                        avg_epoch_loss,
                        val_loss
                    );
                }
            } else if epoch % 10 == 0 {
                tracing::info!(
                    "Epoch {}/{}: train_loss={:.4}",
                    epoch,
                    config.max_epochs,
                    avg_epoch_loss
                );
            }
        }

        self.trained = true;

        // Calculate final accuracy
        let accuracy = if !val_triples.is_empty() {
            self.calculate_accuracy(val_triples).await?
        } else {
            self.calculate_accuracy(train_triples).await?
        };

        Ok(TrainingMetrics {
            loss: loss_history.last().copied().unwrap_or(0.0),
            loss_history,
            accuracy,
            epochs: final_epoch + 1,
            time_elapsed: start_time.elapsed(),
            kg_metrics: KnowledgeGraphMetrics::default(),
        })
    }

    async fn save(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        // Create serializable model state
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;

        let model_state = serde_json::json!({
            "config": self.config,
            "entity_embeddings": entity_embs
                .iter()
                .map(|(k, v)| (k, v.to_vec()))
                .collect::<HashMap<_, _>>(),
            "relation_embeddings": relation_embs
                .iter()
                .map(|(k, v)| (k, v.to_vec()))
                .collect::<HashMap<_, _>>(),
            "entity_vocab": self.entity_vocab,
            "relation_vocab": self.relation_vocab,
            "trained": self.trained,
        });

        let mut file = File::create(path)?;
        let serialized = serde_json::to_string_pretty(&model_state)?;
        file.write_all(serialized.as_bytes())?;

        Ok(())
    }

    async fn load(&mut self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let model_state: serde_json::Value = serde_json::from_str(&contents)?;

        // Load configuration
        self.config = serde_json::from_value(model_state["config"].clone())?;

        // Load vocabularies
        self.entity_vocab = serde_json::from_value(model_state["entity_vocab"].clone())?;
        self.relation_vocab = serde_json::from_value(model_state["relation_vocab"].clone())?;

        // Load embeddings
        let mut entity_embs = self.entity_embeddings.write().await;
        let mut relation_embs = self.relation_embeddings.write().await;

        entity_embs.clear();
        relation_embs.clear();

        let entity_embeddings_data: HashMap<String, Vec<f32>> =
            serde_json::from_value(model_state["entity_embeddings"].clone())?;
        for (entity, embedding) in entity_embeddings_data {
            entity_embs.insert(entity, Array1::from_vec(embedding));
        }

        let relation_embeddings_data: HashMap<String, Vec<f32>> =
            serde_json::from_value(model_state["relation_embeddings"].clone())?;
        for (relation, embedding) in relation_embeddings_data {
            relation_embs.insert(relation, Array1::from_vec(embedding));
        }

        self.trained = model_state["trained"].as_bool().unwrap_or(false);

        Ok(())
    }
}
