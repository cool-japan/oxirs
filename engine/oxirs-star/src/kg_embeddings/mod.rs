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
//! ```rust,ignore
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

use crate::{StarResult, StarTriple};
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod kg_embeddings_models;

#[cfg(test)]
mod kg_embeddings_tests;

pub use kg_embeddings_models::*;

/// Thread-local LCG random number generator for deterministic, allocation-free sampling.
///
/// Uses a simple LCG (Linear Congruential Generator) seeded at 42, which is sufficient
/// for non-cryptographic use cases like embedding initialization and negative sampling.
///
/// Per the SCIRS2 POLICY, production code should use `scirs2_core::random::uniform()`.
pub(crate) fn random_uniform() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(42) };
    }
    SEED.with(|s| {
        let mut seed = s.get();
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub(crate) fn term_to_string(term: &crate::StarTerm) -> String {
        match term {
            crate::StarTerm::NamedNode(n) => n.iri.clone(),
            crate::StarTerm::BlankNode(b) => format!("_:{}", b.id),
            crate::StarTerm::Literal(l) => l.value.clone(),
            crate::StarTerm::Variable(v) => format!("?{}", v.name),
            crate::StarTerm::QuotedTriple(t) => format!("<<{}>>", t.subject),
        }
    }

    /// Create vocabulary from triples
    pub fn from_triples(triples: &[StarTriple]) -> Self {
        let mut entity_to_idx = HashMap::new();
        let mut idx_to_entity = Vec::new();
        let mut relation_to_idx = HashMap::new();
        let mut idx_to_relation = Vec::new();

        for triple in triples {
            let subject_str = Self::term_to_string(&triple.subject);
            if !entity_to_idx.contains_key(&subject_str) {
                entity_to_idx.insert(subject_str.clone(), idx_to_entity.len());
                idx_to_entity.push(subject_str);
            }

            let predicate_str = Self::term_to_string(&triple.predicate);
            if !relation_to_idx.contains_key(&predicate_str) {
                relation_to_idx.insert(predicate_str.clone(), idx_to_relation.len());
                idx_to_relation.push(predicate_str);
            }

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
