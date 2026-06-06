//! Knowledge Graph Embedding Models
//!
//! Contains the concrete model implementations: TransE, DistMult, ComplEx.
//! All models implement the [`EmbeddingModel`] trait defined in the parent module.

use crate::{StarResult, StarTriple};
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use super::{random_uniform, EmbeddingConfig, EmbeddingModel, TrainingStats, Vocabulary};

/// TransE: Translation-based embeddings
///
/// Models relations as translations in embedding space: h + r ≈ t
#[derive(Serialize, Deserialize)]
pub struct TransE {
    pub config: EmbeddingConfig,
    /// Entity embeddings (num_entities × embedding_dim)
    pub(super) entity_embeddings: Array2<f64>,
    /// Relation embeddings (num_relations × embedding_dim)
    pub(super) relation_embeddings: Array2<f64>,
    /// Vocabulary
    pub(super) vocab: Option<Vocabulary>,
    /// RNG seed for reproducibility
    #[allow(dead_code)]
    pub(super) seed: u64,
}

impl TransE {
    /// Create a new TransE model
    pub fn new(config: EmbeddingConfig) -> Self {
        Self::with_seed(config, 42)
    }

    /// Create a new TransE model with a specific seed
    pub fn with_seed(config: EmbeddingConfig, seed: u64) -> Self {
        Self {
            config,
            entity_embeddings: Array2::zeros((0, 0)),
            relation_embeddings: Array2::zeros((0, 0)),
            vocab: None,
            seed,
        }
    }

    /// Initialize embeddings
    fn initialize_embeddings(&mut self, num_entities: usize, num_relations: usize) {
        let dim = self.config.embedding_dim;

        // Xavier initialization
        let scale = (6.0 / (dim as f64)).sqrt();

        self.entity_embeddings = Array2::zeros((num_entities, dim));
        for i in 0..num_entities {
            for j in 0..dim {
                self.entity_embeddings[[i, j]] = random_uniform() * 2.0 * scale - scale;
            }
        }

        self.relation_embeddings = Array2::zeros((num_relations, dim));
        for i in 0..num_relations {
            for j in 0..dim {
                self.relation_embeddings[[i, j]] = random_uniform() * 2.0 * scale - scale;
            }
        }

        // Normalize entity embeddings
        self.normalize_embeddings();

        info!(
            "Initialized TransE embeddings: {} entities, {} relations, dim={}",
            num_entities, num_relations, dim
        );
    }

    /// Normalize entity embeddings to unit length
    fn normalize_embeddings(&mut self) {
        for i in 0..self.entity_embeddings.nrows() {
            let mut row = self.entity_embeddings.row_mut(i);
            let norm = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                row.mapv_inplace(|x| x / norm);
            }
        }
    }

    /// Compute score for a triple (lower is better)
    fn score(&self, head_idx: usize, rel_idx: usize, tail_idx: usize) -> f64 {
        let h = self.entity_embeddings.row(head_idx);
        let r = self.relation_embeddings.row(rel_idx);
        let t = self.entity_embeddings.row(tail_idx);

        // L1 or L2 distance: ||h + r - t||
        let mut distance = 0.0;
        for i in 0..self.config.embedding_dim {
            let diff = h[i] + r[i] - t[i];
            distance += diff.abs(); // L1 norm
        }
        distance
    }

    /// Generate negative sample
    fn negative_sample(&self, num_entities: usize) -> usize {
        (random_uniform() * num_entities as f64) as usize
    }
}

impl EmbeddingModel for TransE {
    #[instrument(skip(self, triples))]
    fn train(&mut self, triples: &[StarTriple], epochs: usize) -> StarResult<TrainingStats> {
        let start = std::time::Instant::now();

        // Build vocabulary
        let vocab = Vocabulary::from_triples(triples);
        let num_entities = vocab.num_entities();
        let num_relations = vocab.num_relations();

        info!(
            "Training TransE on {} triples ({} entities, {} relations) for {} epochs",
            triples.len(),
            num_entities,
            num_relations,
            epochs
        );

        // Initialize embeddings
        self.initialize_embeddings(num_entities, num_relations);
        self.vocab = Some(vocab.clone());

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Shuffle triples (simple random sampling)
            let mut triple_indices: Vec<usize> = (0..triples.len()).collect();
            for i in (1..triple_indices.len()).rev() {
                let j = (random_uniform() * (i + 1) as f64) as usize;
                triple_indices.swap(i, j);
            }

            // Process batches
            for batch_start in (0..triples.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(triples.len());
                let mut batch_loss = 0.0;

                for &triple_idx in &triple_indices[batch_start..batch_end] {
                    let triple = &triples[triple_idx];

                    // Get indices (using same extraction as vocabulary)
                    let h_idx = vocab
                        .entity_idx(&Vocabulary::term_to_string(&triple.subject))
                        .expect("entity should be in vocabulary");
                    let r_idx = vocab
                        .relation_idx(&Vocabulary::term_to_string(&triple.predicate))
                        .expect("relation should be in vocabulary");
                    let t_idx = vocab
                        .entity_idx(&Vocabulary::term_to_string(&triple.object))
                        .expect("entity should be in vocabulary");

                    // Positive score
                    let pos_score = self.score(h_idx, r_idx, t_idx);

                    // Negative sampling
                    for _ in 0..self.config.num_negative_samples {
                        let neg_t_idx = self.negative_sample(num_entities);

                        // Negative score
                        let neg_score = self.score(h_idx, r_idx, neg_t_idx);

                        // Margin ranking loss
                        let loss = (self.config.margin + pos_score - neg_score).max(0.0);
                        batch_loss += loss;

                        // Gradient descent update (simplified)
                        if loss > 0.0 {
                            let lr = self.config.learning_rate;

                            // Update embeddings (simplified gradient)
                            for i in 0..self.config.embedding_dim {
                                let h_grad = if self.entity_embeddings[[h_idx, i]]
                                    + self.relation_embeddings[[r_idx, i]]
                                    > self.entity_embeddings[[t_idx, i]]
                                {
                                    lr
                                } else {
                                    -lr
                                };

                                self.entity_embeddings[[h_idx, i]] -= h_grad;
                                self.relation_embeddings[[r_idx, i]] -= h_grad;
                                self.entity_embeddings[[t_idx, i]] += h_grad;

                                // L2 regularization
                                self.entity_embeddings[[h_idx, i]] *= 1.0 - self.config.l2_reg * lr;
                                self.relation_embeddings[[r_idx, i]] *=
                                    1.0 - self.config.l2_reg * lr;
                                self.entity_embeddings[[t_idx, i]] *= 1.0 - self.config.l2_reg * lr;
                            }
                        }
                    }
                }

                epoch_loss += batch_loss;
                num_batches += 1;
            }

            // Normalize embeddings after each epoch
            self.normalize_embeddings();

            let avg_loss = epoch_loss / num_batches as f64;
            losses.push(avg_loss);

            if epoch % 10 == 0 {
                debug!("Epoch {}/{}: loss = {:.4}", epoch + 1, epochs, avg_loss);
            }
        }

        let training_time = start.elapsed().as_secs_f64();

        info!(
            "Training complete in {:.2}s, final loss: {:.4}",
            training_time,
            losses.last().copied().unwrap_or(0.0)
        );

        Ok(TrainingStats {
            total_epochs: epochs,
            final_loss: losses.last().copied().unwrap_or(0.0),
            losses_per_epoch: losses,
            training_time_secs: training_time,
        })
    }

    fn get_embedding(&self, entity: &str) -> Option<Array1<f64>> {
        let vocab = self.vocab.as_ref()?;
        let idx = vocab.entity_idx(entity)?;
        Some(self.entity_embeddings.row(idx).to_owned())
    }

    fn similarity(&self, entity1: &str, entity2: &str) -> StarResult<f64> {
        let e1 = self
            .get_embedding(entity1)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Entity not found: {}", entity1),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        let e2 = self
            .get_embedding(entity2)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Entity not found: {}", entity2),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        // Cosine similarity
        let dot: f64 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = e1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = e2.iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(dot / (norm1 * norm2))
    }

    fn predict_tail(&self, head: &str, relation: &str, k: usize) -> StarResult<Vec<(String, f64)>> {
        let vocab = self
            .vocab
            .as_ref()
            .ok_or_else(|| crate::StarError::QueryError {
                message: "Model not trained".to_string(),
                query_fragment: None,
                position: None,
                suggestion: Some("Train the model first".to_string()),
            })?;

        let h_idx = vocab
            .entity_idx(head)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Head entity not found: {}", head),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        let r_idx = vocab
            .relation_idx(relation)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Relation not found: {}", relation),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        // Score all possible tails
        let mut scores: Vec<(String, f64)> = Vec::new();
        for t_idx in 0..vocab.num_entities() {
            let score = self.score(h_idx, r_idx, t_idx);
            let entity = vocab
                .entity(t_idx)
                .expect("entity index should be valid")
                .to_string();
            scores.push((entity, -score)); // Negate for ranking (higher is better)
        }

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        Ok(scores.into_iter().take(k).collect())
    }

    fn save(&self, path: &str) -> StarResult<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| crate::StarError::serialization_error(e.to_string()))
    }

    fn load(&mut self, path: &str) -> StarResult<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        let loaded: TransE = serde_json::from_str(&content)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        self.config = loaded.config;
        self.entity_embeddings = loaded.entity_embeddings;
        self.relation_embeddings = loaded.relation_embeddings;
        self.vocab = loaded.vocab;
        self.seed = loaded.seed;
        Ok(())
    }
}

/// DistMult: Bilinear Diagonal Model
///
/// Models relations using element-wise multiplication: score(h, r, t) = Σᵢ hᵢ × rᵢ × tᵢ
///
/// ## Properties
/// - **Symmetric**: score(h, r, t) = score(t, r, h)
/// - **Scalable**: Linear complexity in embedding dimension
/// - **Effective**: Good performance for many knowledge graphs
///
/// ## Limitations
/// - Cannot model asymmetric relations (e.g., "parent_of")
/// - Cannot handle relation inverses
///
/// ## Reference
/// Yang et al., "Embedding Entities and Relations for Learning and Inference in Knowledge Bases", ICLR 2015
#[derive(Serialize, Deserialize)]
pub struct DistMult {
    pub config: EmbeddingConfig,
    /// Entity embeddings (num_entities × embedding_dim)
    pub(super) entity_embeddings: Array2<f64>,
    /// Relation embeddings (num_relations × embedding_dim) - diagonal elements
    pub(super) relation_embeddings: Array2<f64>,
    /// Vocabulary
    pub(super) vocab: Option<Vocabulary>,
    /// RNG seed for reproducibility
    #[allow(dead_code)]
    pub(super) seed: u64,
}

impl DistMult {
    /// Create a new DistMult model
    pub fn new(config: EmbeddingConfig) -> Self {
        Self::with_seed(config, 42)
    }

    /// Create a new DistMult model with a specific seed
    pub fn with_seed(config: EmbeddingConfig, seed: u64) -> Self {
        Self {
            config,
            entity_embeddings: Array2::zeros((0, 0)),
            relation_embeddings: Array2::zeros((0, 0)),
            vocab: None,
            seed,
        }
    }

    /// Initialize embeddings with Xavier initialization
    fn initialize_embeddings(&mut self, num_entities: usize, num_relations: usize) {
        let dim = self.config.embedding_dim;

        // Xavier initialization
        let scale = (6.0 / (dim as f64)).sqrt();

        self.entity_embeddings = Array2::zeros((num_entities, dim));
        for i in 0..num_entities {
            for j in 0..dim {
                self.entity_embeddings[[i, j]] = random_uniform() * 2.0 * scale - scale;
            }
        }

        self.relation_embeddings = Array2::zeros((num_relations, dim));
        for i in 0..num_relations {
            for j in 0..dim {
                self.relation_embeddings[[i, j]] = random_uniform() * 2.0 * scale - scale;
            }
        }

        info!(
            "Initialized DistMult embeddings: {} entities, {} relations, dim={}",
            num_entities, num_relations, dim
        );
    }

    /// Compute score for a triple: s(h, r, t) = Σᵢ hᵢ × rᵢ × tᵢ (higher is better)
    fn score(&self, head_idx: usize, rel_idx: usize, tail_idx: usize) -> f64 {
        let h = self.entity_embeddings.row(head_idx);
        let r = self.relation_embeddings.row(rel_idx);
        let t = self.entity_embeddings.row(tail_idx);

        // Element-wise multiplication and sum
        let mut score = 0.0;
        for i in 0..self.config.embedding_dim {
            score += h[i] * r[i] * t[i];
        }
        score
    }

    /// Generate negative sample
    fn negative_sample(&self, num_entities: usize) -> usize {
        (random_uniform() * num_entities as f64) as usize
    }
}

impl EmbeddingModel for DistMult {
    #[instrument(skip(self, triples))]
    fn train(&mut self, triples: &[StarTriple], epochs: usize) -> StarResult<TrainingStats> {
        let start = std::time::Instant::now();

        // Build vocabulary
        let vocab = Vocabulary::from_triples(triples);
        let num_entities = vocab.num_entities();
        let num_relations = vocab.num_relations();

        info!(
            "Training DistMult on {} triples ({} entities, {} relations) for {} epochs",
            triples.len(),
            num_entities,
            num_relations,
            epochs
        );

        // Initialize embeddings
        self.initialize_embeddings(num_entities, num_relations);
        self.vocab = Some(vocab.clone());

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Shuffle triples
            let mut triple_indices: Vec<usize> = (0..triples.len()).collect();
            for i in (1..triple_indices.len()).rev() {
                let j = (random_uniform() * (i + 1) as f64) as usize;
                triple_indices.swap(i, j);
            }

            // Process batches
            for batch_start in (0..triples.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(triples.len());
                let mut batch_loss = 0.0;

                for &triple_idx in &triple_indices[batch_start..batch_end] {
                    let triple = &triples[triple_idx];

                    // Get indices
                    let h_idx = vocab
                        .entity_idx(&Vocabulary::term_to_string(&triple.subject))
                        .expect("entity should be in vocabulary");
                    let r_idx = vocab
                        .relation_idx(&Vocabulary::term_to_string(&triple.predicate))
                        .expect("relation should be in vocabulary");
                    let t_idx = vocab
                        .entity_idx(&Vocabulary::term_to_string(&triple.object))
                        .expect("entity should be in vocabulary");

                    // Positive score
                    let pos_score = self.score(h_idx, r_idx, t_idx);

                    // Negative sampling
                    for _ in 0..self.config.num_negative_samples {
                        // Corrupt either head or tail (50/50)
                        let corrupt_head = random_uniform() > 0.5;
                        let (neg_h_idx, neg_t_idx) = if corrupt_head {
                            (self.negative_sample(num_entities), t_idx)
                        } else {
                            (h_idx, self.negative_sample(num_entities))
                        };

                        // Negative score
                        let neg_score = self.score(neg_h_idx, r_idx, neg_t_idx);

                        // Logistic loss: log(1 + exp(-y*score)) where y=1 for positive, y=-1 for negative
                        // Simplified: softplus(-(pos_score - neg_score))
                        let margin_diff = pos_score - neg_score;
                        let loss = (1.0 + (-margin_diff).exp()).ln(); // softplus
                        batch_loss += loss;

                        // Gradient descent update
                        if loss > 0.01 {
                            // Gradient threshold
                            let lr = self.config.learning_rate;
                            let sigmoid = 1.0 / (1.0 + margin_diff.exp());

                            // Update embeddings with gradients
                            for i in 0..self.config.embedding_dim {
                                let h_val = self.entity_embeddings[[h_idx, i]];
                                let r_val = self.relation_embeddings[[r_idx, i]];
                                let t_val = self.entity_embeddings[[t_idx, i]];

                                // Positive gradients
                                let grad_h_pos = sigmoid * r_val * t_val;
                                let grad_r_pos = sigmoid * h_val * t_val;
                                let grad_t_pos = sigmoid * h_val * r_val;

                                // Update positive triple embeddings
                                self.entity_embeddings[[h_idx, i]] += lr * grad_h_pos;
                                self.relation_embeddings[[r_idx, i]] += lr * grad_r_pos;
                                self.entity_embeddings[[t_idx, i]] += lr * grad_t_pos;

                                // Negative gradients (opposite direction)
                                if corrupt_head {
                                    let grad_neg_h = -sigmoid * r_val * t_val;
                                    self.entity_embeddings[[neg_h_idx, i]] += lr * grad_neg_h;
                                } else {
                                    let grad_neg_t = -sigmoid * h_val * r_val;
                                    self.entity_embeddings[[neg_t_idx, i]] += lr * grad_neg_t;
                                }

                                // L2 regularization
                                self.entity_embeddings[[h_idx, i]] *= 1.0 - self.config.l2_reg * lr;
                                self.relation_embeddings[[r_idx, i]] *=
                                    1.0 - self.config.l2_reg * lr;
                            }
                        }
                    }
                }

                epoch_loss += batch_loss;
                num_batches += 1;
            }

            let avg_loss = epoch_loss / num_batches as f64;
            losses.push(avg_loss);

            if epoch % 10 == 0 {
                debug!("Epoch {}/{}: loss = {:.4}", epoch + 1, epochs, avg_loss);
            }
        }

        let training_time = start.elapsed().as_secs_f64();

        info!(
            "Training complete in {:.2}s, final loss: {:.4}",
            training_time,
            losses.last().copied().unwrap_or(0.0)
        );

        Ok(TrainingStats {
            total_epochs: epochs,
            final_loss: losses.last().copied().unwrap_or(0.0),
            losses_per_epoch: losses,
            training_time_secs: training_time,
        })
    }

    fn get_embedding(&self, entity: &str) -> Option<Array1<f64>> {
        let vocab = self.vocab.as_ref()?;
        let idx = vocab.entity_idx(entity)?;
        Some(self.entity_embeddings.row(idx).to_owned())
    }

    fn similarity(&self, entity1: &str, entity2: &str) -> StarResult<f64> {
        let e1 = self
            .get_embedding(entity1)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Entity not found: {}", entity1),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        let e2 = self
            .get_embedding(entity2)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Entity not found: {}", entity2),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        // Cosine similarity
        let dot: f64 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = e1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = e2.iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(dot / (norm1 * norm2 + 1e-10))
    }

    fn predict_tail(&self, head: &str, relation: &str, k: usize) -> StarResult<Vec<(String, f64)>> {
        let vocab = self
            .vocab
            .as_ref()
            .ok_or_else(|| crate::StarError::QueryError {
                message: "Model not trained".to_string(),
                query_fragment: None,
                position: None,
                suggestion: Some("Train the model first".to_string()),
            })?;

        let h_idx = vocab
            .entity_idx(head)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Head entity not found: {}", head),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        let r_idx = vocab
            .relation_idx(relation)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Relation not found: {}", relation),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        // Score all possible tails
        let mut scores: Vec<(String, f64)> = Vec::new();
        for t_idx in 0..vocab.num_entities() {
            let score = self.score(h_idx, r_idx, t_idx);
            let entity = vocab
                .entity(t_idx)
                .expect("entity index should be valid")
                .to_string();
            scores.push((entity, score)); // Higher is better for DistMult
        }

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        Ok(scores.into_iter().take(k).collect())
    }

    fn save(&self, path: &str) -> StarResult<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| crate::StarError::serialization_error(e.to_string()))
    }

    fn load(&mut self, path: &str) -> StarResult<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        let loaded: DistMult = serde_json::from_str(&content)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        self.config = loaded.config;
        self.entity_embeddings = loaded.entity_embeddings;
        self.relation_embeddings = loaded.relation_embeddings;
        self.vocab = loaded.vocab;
        self.seed = loaded.seed;
        Ok(())
    }
}

/// ComplEx: Complex-Valued Embeddings
///
/// Models entities and relations as complex-valued vectors with Hermitian dot product.
/// Score: Re(h^T diag(r) conj(t)) = Re(Σᵢ hᵢ × rᵢ × conj(tᵢ))
///
/// ## Properties
/// - **Asymmetric**: Can model asymmetric relations (parent_of ≠ child_of)
/// - **Handles Inverses**: Natural handling of relation inverses
/// - **State-of-the-art**: Top performance on link prediction benchmarks
///
/// ## Complex Arithmetic
/// - h, r, t ∈ ℂᵈ (complex vectors)
/// - Each dimension: xᵢ = aᵢ + bᵢj where j² = -1
/// - Conjugate: conj(a + bj) = a - bj
/// - Re(z): Real part of complex number z
///
/// ## Reference
/// Trouillon et al., "Complex Embeddings for Simple Link Prediction", ICML 2016
#[derive(Serialize, Deserialize)]
pub struct ComplEx {
    pub config: EmbeddingConfig,
    /// Entity embeddings - real part (num_entities × embedding_dim)
    pub(super) entity_embeddings_real: Array2<f64>,
    /// Entity embeddings - imaginary part (num_entities × embedding_dim)
    pub(super) entity_embeddings_imag: Array2<f64>,
    /// Relation embeddings - real part (num_relations × embedding_dim)
    pub(super) relation_embeddings_real: Array2<f64>,
    /// Relation embeddings - imaginary part (num_relations × embedding_dim)
    pub(super) relation_embeddings_imag: Array2<f64>,
    /// Vocabulary
    pub(super) vocab: Option<Vocabulary>,
    /// RNG seed for reproducibility
    #[allow(dead_code)]
    pub(super) seed: u64,
}

impl ComplEx {
    /// Create a new ComplEx model
    pub fn new(config: EmbeddingConfig) -> Self {
        Self::with_seed(config, 42)
    }

    /// Create a new ComplEx model with a specific seed
    pub fn with_seed(config: EmbeddingConfig, seed: u64) -> Self {
        Self {
            config,
            entity_embeddings_real: Array2::zeros((0, 0)),
            entity_embeddings_imag: Array2::zeros((0, 0)),
            relation_embeddings_real: Array2::zeros((0, 0)),
            relation_embeddings_imag: Array2::zeros((0, 0)),
            vocab: None,
            seed,
        }
    }

    /// Initialize complex embeddings with Xavier initialization
    fn initialize_embeddings(&mut self, num_entities: usize, num_relations: usize) {
        let dim = self.config.embedding_dim;

        // Xavier initialization for complex embeddings
        let scale = (6.0 / (2.0 * dim as f64)).sqrt(); // 2x for real+imag

        // Initialize entity embeddings (real and imaginary parts)
        self.entity_embeddings_real = Array2::zeros((num_entities, dim));
        self.entity_embeddings_imag = Array2::zeros((num_entities, dim));
        for i in 0..num_entities {
            for j in 0..dim {
                self.entity_embeddings_real[[i, j]] = random_uniform() * 2.0 * scale - scale;
                self.entity_embeddings_imag[[i, j]] = random_uniform() * 2.0 * scale - scale;
            }
        }

        // Initialize relation embeddings (real and imaginary parts)
        self.relation_embeddings_real = Array2::zeros((num_relations, dim));
        self.relation_embeddings_imag = Array2::zeros((num_relations, dim));
        for i in 0..num_relations {
            for j in 0..dim {
                self.relation_embeddings_real[[i, j]] = random_uniform() * 2.0 * scale - scale;
                self.relation_embeddings_imag[[i, j]] = random_uniform() * 2.0 * scale - scale;
            }
        }

        info!(
            "Initialized ComplEx embeddings: {} entities, {} relations, dim={} (complex)",
            num_entities, num_relations, dim
        );
    }

    /// Compute score for a triple: Re(h^T diag(r) conj(t))
    /// where h, r, t are complex vectors
    fn score(&self, head_idx: usize, rel_idx: usize, tail_idx: usize) -> f64 {
        let h_re = self.entity_embeddings_real.row(head_idx);
        let h_im = self.entity_embeddings_imag.row(head_idx);
        let r_re = self.relation_embeddings_real.row(rel_idx);
        let r_im = self.relation_embeddings_imag.row(rel_idx);
        let t_re = self.entity_embeddings_real.row(tail_idx);
        let t_im = self.entity_embeddings_imag.row(tail_idx);

        // Complex multiplication: (h * r * conj(t))
        // h = h_re + i*h_im
        // r = r_re + i*r_im
        // conj(t) = t_re - i*t_im
        //
        // h * r = (h_re*r_re - h_im*r_im) + i*(h_re*r_im + h_im*r_re)
        // (h*r) * conj(t) = ...
        // Re((h*r) * conj(t)) = sum_i [(h_re*r_re - h_im*r_im)*t_re + (h_re*r_im + h_im*r_re)*t_im]

        let mut score = 0.0;
        for i in 0..self.config.embedding_dim {
            // Intermediate: h * r
            let hr_re = h_re[i] * r_re[i] - h_im[i] * r_im[i];
            let hr_im = h_re[i] * r_im[i] + h_im[i] * r_re[i];

            // Final: (h*r) * conj(t)
            // conj(t) = t_re - i*t_im
            // (hr_re + i*hr_im) * (t_re - i*t_im)
            // = hr_re*t_re + hr_im*t_im + i*(hr_im*t_re - hr_re*t_im)
            // Real part:
            score += hr_re * t_re[i] + hr_im * t_im[i];
        }

        score
    }

    /// Generate negative sample
    fn negative_sample(&self, num_entities: usize) -> usize {
        (random_uniform() * num_entities as f64) as usize
    }
}

impl EmbeddingModel for ComplEx {
    #[instrument(skip(self, triples))]
    fn train(&mut self, triples: &[StarTriple], epochs: usize) -> StarResult<TrainingStats> {
        let start = std::time::Instant::now();

        // Build vocabulary
        let vocab = Vocabulary::from_triples(triples);
        let num_entities = vocab.num_entities();
        let num_relations = vocab.num_relations();

        info!(
            "Training ComplEx on {} triples ({} entities, {} relations) for {} epochs",
            triples.len(),
            num_entities,
            num_relations,
            epochs
        );

        // Initialize embeddings
        self.initialize_embeddings(num_entities, num_relations);
        self.vocab = Some(vocab.clone());

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Shuffle triples
            let mut triple_indices: Vec<usize> = (0..triples.len()).collect();
            for i in (1..triple_indices.len()).rev() {
                let j = (random_uniform() * (i + 1) as f64) as usize;
                triple_indices.swap(i, j);
            }

            // Process batches
            for batch_start in (0..triples.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(triples.len());
                let mut batch_loss = 0.0;

                for &triple_idx in &triple_indices[batch_start..batch_end] {
                    let triple = &triples[triple_idx];

                    // Get indices
                    let h_idx = vocab
                        .entity_idx(&Vocabulary::term_to_string(&triple.subject))
                        .expect("entity should be in vocabulary");
                    let r_idx = vocab
                        .relation_idx(&Vocabulary::term_to_string(&triple.predicate))
                        .expect("relation should be in vocabulary");
                    let t_idx = vocab
                        .entity_idx(&Vocabulary::term_to_string(&triple.object))
                        .expect("entity should be in vocabulary");

                    // Positive score
                    let pos_score = self.score(h_idx, r_idx, t_idx);

                    // Negative sampling
                    for _ in 0..self.config.num_negative_samples {
                        // Corrupt either head or tail
                        let corrupt_head = random_uniform() > 0.5;
                        let (neg_h_idx, neg_t_idx) = if corrupt_head {
                            (self.negative_sample(num_entities), t_idx)
                        } else {
                            (h_idx, self.negative_sample(num_entities))
                        };

                        // Negative score
                        let neg_score = self.score(neg_h_idx, r_idx, neg_t_idx);

                        // Logistic loss with margin
                        let margin_diff = pos_score - neg_score;
                        let loss = (1.0 + (-margin_diff).exp()).ln();
                        batch_loss += loss;

                        // Gradient descent (simplified for complex embeddings)
                        if loss > 0.01 {
                            let lr = self.config.learning_rate;
                            let sigmoid = 1.0 / (1.0 + margin_diff.exp());

                            // Update complex embeddings
                            // Gradient computation is complex (pun intended)
                            // Simplified: update in direction of score improvement
                            for i in 0..self.config.embedding_dim {
                                // Get current values
                                let h_re = self.entity_embeddings_real[[h_idx, i]];
                                let h_im = self.entity_embeddings_imag[[h_idx, i]];
                                let r_re = self.relation_embeddings_real[[r_idx, i]];
                                let r_im = self.relation_embeddings_imag[[r_idx, i]];
                                let t_re = self.entity_embeddings_real[[t_idx, i]];
                                let t_im = self.entity_embeddings_imag[[t_idx, i]];

                                // Gradients (approximate for positive triple)
                                let grad_h_re = sigmoid * (r_re * t_re + r_im * t_im);
                                let grad_h_im = sigmoid * (r_im * t_re - r_re * t_im);
                                let grad_r_re = sigmoid * (h_re * t_re + h_im * t_im);
                                let grad_r_im = sigmoid * (h_re * t_im - h_im * t_re);
                                let grad_t_re = sigmoid * (h_re * r_re - h_im * r_im);
                                let grad_t_im = sigmoid * (h_re * r_im + h_im * r_re);

                                // Update positive embeddings
                                self.entity_embeddings_real[[h_idx, i]] += lr * grad_h_re;
                                self.entity_embeddings_imag[[h_idx, i]] += lr * grad_h_im;
                                self.relation_embeddings_real[[r_idx, i]] += lr * grad_r_re;
                                self.relation_embeddings_imag[[r_idx, i]] += lr * grad_r_im;
                                self.entity_embeddings_real[[t_idx, i]] += lr * grad_t_re;
                                self.entity_embeddings_imag[[t_idx, i]] += lr * grad_t_im;

                                // Update negative embeddings (opposite direction)
                                if corrupt_head {
                                    self.entity_embeddings_real[[neg_h_idx, i]] -=
                                        lr * sigmoid * (r_re * t_re + r_im * t_im);
                                    self.entity_embeddings_imag[[neg_h_idx, i]] -=
                                        lr * sigmoid * (r_im * t_re - r_re * t_im);
                                } else {
                                    self.entity_embeddings_real[[neg_t_idx, i]] -=
                                        lr * sigmoid * (h_re * r_re - h_im * r_im);
                                    self.entity_embeddings_imag[[neg_t_idx, i]] -=
                                        lr * sigmoid * (h_re * r_im + h_im * r_re);
                                }

                                // L2 regularization
                                let reg_factor = 1.0 - self.config.l2_reg * lr;
                                self.entity_embeddings_real[[h_idx, i]] *= reg_factor;
                                self.entity_embeddings_imag[[h_idx, i]] *= reg_factor;
                                self.relation_embeddings_real[[r_idx, i]] *= reg_factor;
                                self.relation_embeddings_imag[[r_idx, i]] *= reg_factor;
                            }
                        }
                    }
                }

                epoch_loss += batch_loss;
                num_batches += 1;
            }

            let avg_loss = epoch_loss / num_batches as f64;
            losses.push(avg_loss);

            if epoch % 10 == 0 {
                debug!("Epoch {}/{}: loss = {:.4}", epoch + 1, epochs, avg_loss);
            }
        }

        let training_time = start.elapsed().as_secs_f64();

        info!(
            "Training complete in {:.2}s, final loss: {:.4}",
            training_time,
            losses.last().copied().unwrap_or(0.0)
        );

        Ok(TrainingStats {
            total_epochs: epochs,
            final_loss: losses.last().copied().unwrap_or(0.0),
            losses_per_epoch: losses,
            training_time_secs: training_time,
        })
    }

    fn get_embedding(&self, entity: &str) -> Option<Array1<f64>> {
        let vocab = self.vocab.as_ref()?;
        let idx = vocab.entity_idx(entity)?;

        // Return concatenated real and imaginary parts as a single vector
        let real = self.entity_embeddings_real.row(idx);
        let imag = self.entity_embeddings_imag.row(idx);

        let mut embedding = Array1::zeros(self.config.embedding_dim * 2);
        for i in 0..self.config.embedding_dim {
            embedding[i] = real[i];
            embedding[i + self.config.embedding_dim] = imag[i];
        }

        Some(embedding)
    }

    fn similarity(&self, entity1: &str, entity2: &str) -> StarResult<f64> {
        let e1 = self
            .get_embedding(entity1)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Entity not found: {}", entity1),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        let e2 = self
            .get_embedding(entity2)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Entity not found: {}", entity2),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        // Cosine similarity on concatenated embeddings
        let dot: f64 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = e1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = e2.iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(dot / (norm1 * norm2 + 1e-10))
    }

    fn predict_tail(&self, head: &str, relation: &str, k: usize) -> StarResult<Vec<(String, f64)>> {
        let vocab = self
            .vocab
            .as_ref()
            .ok_or_else(|| crate::StarError::QueryError {
                message: "Model not trained".to_string(),
                query_fragment: None,
                position: None,
                suggestion: Some("Train the model first".to_string()),
            })?;

        let h_idx = vocab
            .entity_idx(head)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Head entity not found: {}", head),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        let r_idx = vocab
            .relation_idx(relation)
            .ok_or_else(|| crate::StarError::QueryError {
                message: format!("Relation not found: {}", relation),
                query_fragment: None,
                position: None,
                suggestion: None,
            })?;

        // Score all possible tails
        let mut scores: Vec<(String, f64)> = Vec::new();
        for t_idx in 0..vocab.num_entities() {
            let score = self.score(h_idx, r_idx, t_idx);
            let entity = vocab
                .entity(t_idx)
                .expect("entity index should be valid")
                .to_string();
            scores.push((entity, score)); // Higher is better for ComplEx
        }

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        Ok(scores.into_iter().take(k).collect())
    }

    fn save(&self, path: &str) -> StarResult<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| crate::StarError::serialization_error(e.to_string()))
    }

    fn load(&mut self, path: &str) -> StarResult<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        let loaded: ComplEx = serde_json::from_str(&content)
            .map_err(|e| crate::StarError::serialization_error(e.to_string()))?;
        self.config = loaded.config;
        self.entity_embeddings_real = loaded.entity_embeddings_real;
        self.entity_embeddings_imag = loaded.entity_embeddings_imag;
        self.relation_embeddings_real = loaded.relation_embeddings_real;
        self.relation_embeddings_imag = loaded.relation_embeddings_imag;
        self.vocab = loaded.vocab;
        self.seed = loaded.seed;
        Ok(())
    }
}
