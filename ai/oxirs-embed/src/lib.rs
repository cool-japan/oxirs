//! # OxiRS Embed: Advanced Knowledge Graph Embeddings
//!
//! This crate provides state-of-the-art knowledge graph embedding methods
//! including TransE, DistMult, ComplEx, and RotatE models.

#[cfg(feature = "api-server")]
pub mod api;
pub mod caching;
pub mod evaluation;
pub mod inference;
pub mod integration;
pub mod model_registry;
pub mod models;
pub mod persistence;
pub mod training;
pub mod utils;

// Local type definitions (normally would import from oxirs-core and oxirs-vec)
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Vector for embeddings
#[derive(Debug, Clone)]
pub struct Vector {
    pub values: Vec<f32>,
    pub dimensions: usize,
}

impl Vector {
    pub fn new(values: Vec<f32>) -> Self {
        let dimensions = values.len();
        Self { values, dimensions }
    }
}

/// Triple structure for RDF triples
#[derive(Debug, Clone, PartialEq)]
pub struct Triple {
    pub subject: NamedNode,
    pub predicate: NamedNode,
    pub object: NamedNode,
}

impl Triple {
    pub fn new(subject: NamedNode, predicate: NamedNode, object: NamedNode) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

/// Named node for RDF resources
#[derive(Debug, Clone, PartialEq)]
pub struct NamedNode {
    pub iri: String,
}

impl NamedNode {
    pub fn new(iri: &str) -> Result<Self> {
        Ok(Self {
            iri: iri.to_string(),
        })
    }
}

impl std::fmt::Display for NamedNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.iri)
    }
}

/// Configuration for embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dimensions: usize,
    pub learning_rate: f64,
    pub l2_reg: f64,
    pub max_epochs: usize,
    pub batch_size: usize,
    pub negative_samples: usize,
    pub seed: Option<u64>,
    pub use_gpu: bool,
    pub model_params: HashMap<String, f64>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dimensions: 100,
            learning_rate: 0.01,
            l2_reg: 0.0001,
            max_epochs: 1000,
            batch_size: 1000,
            negative_samples: 10,
            seed: None,
            use_gpu: false,
            model_params: HashMap::new(),
        }
    }
}

impl ModelConfig {
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub epochs_completed: usize,
    pub final_loss: f64,
    pub training_time_seconds: f64,
    pub convergence_achieved: bool,
    pub loss_history: Vec<f64>,
}

/// Model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub num_entities: usize,
    pub num_relations: usize,
    pub num_triples: usize,
    pub dimensions: usize,
    pub is_trained: bool,
    pub model_type: String,
    pub creation_time: DateTime<Utc>,
    pub last_training_time: Option<DateTime<Utc>>,
}

/// Embedding errors
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Model not trained")]
    ModelNotTrained,
    #[error("Entity not found: {entity}")]
    EntityNotFound { entity: String },
    #[error("Relation not found: {relation}")]
    RelationNotFound { relation: String },
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Basic embedding model trait
#[async_trait::async_trait]
pub trait EmbeddingModel: Send + Sync {
    fn config(&self) -> &ModelConfig;
    fn model_id(&self) -> &Uuid;
    fn model_type(&self) -> &'static str;
    fn add_triple(&mut self, triple: Triple) -> Result<()>;
    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats>;
    fn get_entity_embedding(&self, entity: &str) -> Result<Vector>;
    fn get_relation_embedding(&self, relation: &str) -> Result<Vector>;
    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64>;
    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>>;
    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>>;
    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>>;
    fn get_entities(&self) -> Vec<String>;
    fn get_relations(&self) -> Vec<String>;
    fn get_stats(&self) -> ModelStats;
    fn save(&self, path: &str) -> Result<()>;
    fn load(&mut self, path: &str) -> Result<()>;
    fn clear(&mut self);
    fn is_trained(&self) -> bool;
}

// Re-export main types
#[cfg(feature = "api-server")]
pub use api::{ApiState, ApiConfig, start_server};
pub use caching::{CacheManager, CacheConfig, CachedEmbeddingModel};
pub use models::{
    ComplEx, DistMult, RotatE, TransE,
    TransformerEmbedding, TransformerType, TransformerConfig, PoolingStrategy,
    GNNEmbedding, GNNType, GNNConfig, AggregationType
};

#[cfg(feature = "tucker")]
pub use models::TuckER;

#[cfg(feature = "quatd")]
pub use models::QuatD;

// Re-export model registry types
pub use crate::model_registry::{ModelRegistry, ModelVersion, ResourceAllocation};
