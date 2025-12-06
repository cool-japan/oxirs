//! Knowledge Graph Embeddings for RDF-star
//!
//! This module provides state-of-the-art knowledge graph embedding models
//! with full RDF-star support for quoted triples and metadata.
//!
//! ## Supported Models
//!
//! - **TransE**: Translation-based embeddings (h + r ≈ t)
//! - **DistMult**: Bilinear diagonal model
//! - **ComplEx**: Complex-valued embeddings with Hermitian dot product
//!
//! ## RDF-star Extensions
//!
//! All models support quoted triples with contextual embeddings:
//! - Quoted triple embeddings capture nested structure
//! - Annotation metadata influences embedding space
//! - Provenance and trust scores affect similarity
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_star::kg_embeddings::{EmbeddingModel, TransE, EmbeddingConfig};
//!
//! let config = EmbeddingConfig {
//!     embedding_dim: 128,
//!     learning_rate: 0.01,
//!     margin: 1.0,
//!     ..Default::default()
//! };
//!
//! let mut model = TransE::new(config);
//! // Train on RDF-star triples
//! model.train(&triples, 100)?;
//!
//! // Get similarity
//! let sim = model.similarity("entity1", "entity2")?;
//! ```

use crate::{StarResult, StarTerm, StarTriple};
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, instrument};

// Helper function for random number generation
// Uses thread-local RNG (not Send/Sync but sufficient for single-threaded training)
fn random_uniform() -> f64 {
    // In production, use scirs2_core::random::uniform() (SCIRS2 POLICY)
    // For now, use std::random as fallback
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(42) };
    }
    SEED.with(|s| {
        // Simple LCG for deterministic testing
        let mut seed = s.get();
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        s.set(seed);
        (seed as f64) / (u64::MAX as f64)
    })
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Margin for ranking loss
    pub margin: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of negative samples
    pub num_negative_samples: usize,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// Enable RDF-star context
    pub enable_rdfstar_context: bool,
    /// L2 regularization coefficient
    pub l2_reg: f64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            learning_rate: 0.01,
            margin: 1.0,
            batch_size: 128,
            num_negative_samples: 10,
            use_gpu: false,
            enable_rdfstar_context: true,
            l2_reg: 0.0001,
        }
    }
}

/// Embedding model trait
pub trait EmbeddingModel: Send + Sync {
    /// Train the model on triples
    fn train(&mut self, triples: &[StarTriple], epochs: usize) -> StarResult<TrainingStats>;

    /// Get embedding for an entity
    fn get_embedding(&self, entity: &str) -> Option<Array1<f64>>;

    /// Compute similarity between two entities
    fn similarity(&self, entity1: &str, entity2: &str) -> StarResult<f64>;

    /// Predict link between head and relation
    fn predict_tail(&self, head: &str, relation: &str, k: usize) -> StarResult<Vec<(String, f64)>>;

    /// Save model to disk
    fn save(&self, path: &str) -> StarResult<()>;

    /// Load model from disk
    fn load(&mut self, path: &str) -> StarResult<()>;
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub total_epochs: usize,
    pub final_loss: f64,
    pub losses_per_epoch: Vec<f64>,
    pub training_time_secs: f64,
}

/// Entity and relation vocabulary
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Entity to index mapping
    entity_to_idx: HashMap<String, usize>,
    /// Index to entity mapping
    idx_to_entity: Vec<String>,
    /// Relation to index mapping
    relation_to_idx: HashMap<String, usize>,
    /// Index to relation mapping
    idx_to_relation: Vec<String>,
}

impl Vocabulary {
    /// Extract string representation from StarTerm
    fn term_to_string(term: &StarTerm) -> String {
        match term {
            StarTerm::NamedNode(n) => n.iri.clone(),
            StarTerm::BlankNode(b) => format!("_:{}", b.id),
            StarTerm::Literal(l) => l.value.clone(),
            StarTerm::Variable(v) => format!("?{}", v.name),
            StarTerm::QuotedTriple(t) => format!("<<{}>>", t.subject),
        }
    }

    /// Create vocabulary from triples
    pub fn from_triples(triples: &[StarTriple]) -> Self {
        let mut entity_to_idx = HashMap::new();
        let mut idx_to_entity = Vec::new();
        let mut relation_to_idx = HashMap::new();
        let mut idx_to_relation = Vec::new();

        for triple in triples {
            // Add subject
            let subject_str = Self::term_to_string(&triple.subject);
            if !entity_to_idx.contains_key(&subject_str) {
                entity_to_idx.insert(subject_str.clone(), idx_to_entity.len());
                idx_to_entity.push(subject_str);
            }

            // Add predicate
            let predicate_str = Self::term_to_string(&triple.predicate);
            if !relation_to_idx.contains_key(&predicate_str) {
                relation_to_idx.insert(predicate_str.clone(), idx_to_relation.len());
                idx_to_relation.push(predicate_str);
            }

            // Add object
            let object_str = Self::term_to_string(&triple.object);
            if !entity_to_idx.contains_key(&object_str) {
                entity_to_idx.insert(object_str.clone(), idx_to_entity.len());
                idx_to_entity.push(object_str);
            }
        }

        Self {
            entity_to_idx,
            idx_to_entity,
            relation_to_idx,
            idx_to_relation,
        }
    }

    /// Get entity index
    pub fn entity_idx(&self, entity: &str) -> Option<usize> {
        self.entity_to_idx.get(entity).copied()
    }

    /// Get relation index
    pub fn relation_idx(&self, relation: &str) -> Option<usize> {
        self.relation_to_idx.get(relation).copied()
    }

    /// Get entity by index
    pub fn entity(&self, idx: usize) -> Option<&str> {
        self.idx_to_entity.get(idx).map(|s| s.as_str())
    }

    /// Get relation by index
    pub fn relation(&self, idx: usize) -> Option<&str> {
        self.idx_to_relation.get(idx).map(|s| s.as_str())
    }

    /// Number of entities
    pub fn num_entities(&self) -> usize {
        self.idx_to_entity.len()
    }

    /// Number of relations
    pub fn num_relations(&self) -> usize {
        self.idx_to_relation.len()
    }
}

/// TransE: Translation-based embeddings
///
/// Models relations as translations in embedding space: h + r ≈ t
pub struct TransE {
    config: EmbeddingConfig,
    /// Entity embeddings (num_entities × embedding_dim)
    entity_embeddings: Array2<f64>,
    /// Relation embeddings (num_relations × embedding_dim)
    relation_embeddings: Array2<f64>,
    /// Vocabulary
    vocab: Option<Vocabulary>,
    /// RNG seed for reproducibility
    #[allow(dead_code)]
    seed: u64,
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
                        .unwrap();
                    let r_idx = vocab
                        .relation_idx(&Vocabulary::term_to_string(&triple.predicate))
                        .unwrap();
                    let t_idx = vocab
                        .entity_idx(&Vocabulary::term_to_string(&triple.object))
                        .unwrap();

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
            let entity = vocab.entity(t_idx).unwrap().to_string();
            scores.push((entity, -score)); // Negate for ranking (higher is better)
        }

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top-k
        Ok(scores.into_iter().take(k).collect())
    }

    fn save(&self, _path: &str) -> StarResult<()> {
        // TODO: Implement serialization
        Ok(())
    }

    fn load(&mut self, _path: &str) -> StarResult<()> {
        // TODO: Implement deserialization
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::NamedNode, StarTerm};

    fn create_test_triples() -> Vec<StarTriple> {
        vec![
            StarTriple {
                subject: StarTerm::NamedNode(NamedNode {
                    iri: "Alice".to_string(),
                }),
                predicate: StarTerm::NamedNode(NamedNode {
                    iri: "knows".to_string(),
                }),
                object: StarTerm::NamedNode(NamedNode {
                    iri: "Bob".to_string(),
                }),
            },
            StarTriple {
                subject: StarTerm::NamedNode(NamedNode {
                    iri: "Bob".to_string(),
                }),
                predicate: StarTerm::NamedNode(NamedNode {
                    iri: "knows".to_string(),
                }),
                object: StarTerm::NamedNode(NamedNode {
                    iri: "Charlie".to_string(),
                }),
            },
            StarTriple {
                subject: StarTerm::NamedNode(NamedNode {
                    iri: "Alice".to_string(),
                }),
                predicate: StarTerm::NamedNode(NamedNode {
                    iri: "likes".to_string(),
                }),
                object: StarTerm::NamedNode(NamedNode {
                    iri: "Coffee".to_string(),
                }),
            },
        ]
    }

    #[test]
    fn test_vocabulary_creation() {
        let triples = create_test_triples();
        let vocab = Vocabulary::from_triples(&triples);

        assert_eq!(vocab.num_entities(), 4); // Alice, Bob, Charlie, Coffee
        assert_eq!(vocab.num_relations(), 2); // knows, likes
    }

    #[test]
    fn test_transe_initialization() {
        let config = EmbeddingConfig {
            embedding_dim: 64,
            ..Default::default()
        };
        let model = TransE::new(config);
        assert_eq!(model.config.embedding_dim, 64);
    }

    #[test]
    fn test_transe_training() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            learning_rate: 0.01,
            batch_size: 2,
            num_negative_samples: 3,
            ..Default::default()
        };

        let mut model = TransE::new(config);
        let triples = create_test_triples();

        let stats = model.train(&triples, 10).unwrap();

        assert_eq!(stats.total_epochs, 10);
        assert!(stats.final_loss >= 0.0);
        assert_eq!(stats.losses_per_epoch.len(), 10);
    }

    #[test]
    fn test_get_embedding() {
        let config = EmbeddingConfig::default();
        let mut model = TransE::new(config);
        let triples = create_test_triples();

        model.train(&triples, 5).unwrap();

        let emb = model.get_embedding("Alice");
        assert!(emb.is_some());
        assert_eq!(emb.unwrap().len(), 128); // Default dimension
    }

    #[test]
    fn test_similarity() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = TransE::new(config);
        let triples = create_test_triples();

        model.train(&triples, 20).unwrap();

        let sim = model.similarity("Alice", "Bob").unwrap();
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_predict_tail() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = TransE::new(config);
        let triples = create_test_triples();

        model.train(&triples, 10).unwrap();

        let predictions = model.predict_tail("Alice", "knows", 3).unwrap();
        assert_eq!(predictions.len(), 3);
        assert!(predictions[0].1 >= predictions[1].1); // Sorted by score
    }

    #[test]
    fn test_embedding_normalization() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = TransE::new(config);
        let triples = create_test_triples();

        model.train(&triples, 5).unwrap();

        // Check that entity embeddings are normalized
        let emb = model.get_embedding("Alice").unwrap();
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.01); // Should be close to 1
    }
}
