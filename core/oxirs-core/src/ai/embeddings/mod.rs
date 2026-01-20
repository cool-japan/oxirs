//! Knowledge Graph Embeddings for RDF
//!
//! This module implements various knowledge graph embedding models including
//! TransE, DistMult, ComplEx, RotatE, and other state-of-the-art approaches.

use crate::model::Triple;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod complex;
pub mod distmult;
pub mod evaluation;
pub mod transe;

pub use complex::ComplEx;
pub use distmult::DistMult;
pub use evaluation::{
    ConfidenceIntervals, KnowledgeGraphMetrics, LinkPredictionMetrics, StatisticalTestResults,
    TaskBreakdownMetrics, TrainingMetrics,
};
pub use transe::TransE;

/// Knowledge graph embedding trait
#[async_trait::async_trait]
pub trait KnowledgeGraphEmbedding: Send + Sync {
    /// Generate embeddings for entities and relations
    async fn generate_embeddings(&self, triples: &[Triple]) -> Result<Vec<Vec<f32>>>;

    /// Score a triple (head, relation, tail)
    async fn score_triple(&self, head: &str, relation: &str, tail: &str) -> Result<f32>;

    /// Predict missing links
    async fn predict_links(
        &self,
        entities: &[String],
        relations: &[String],
    ) -> Result<Vec<(String, String, String, f32)>>;

    /// Get entity embedding
    async fn get_entity_embedding(&self, entity: &str) -> Result<Vec<f32>>;

    /// Get relation embedding
    async fn get_relation_embedding(&self, relation: &str) -> Result<Vec<f32>>;

    /// Train the embedding model
    async fn train(
        &mut self,
        triples: &[Triple],
        config: &TrainingConfig,
    ) -> Result<TrainingMetrics>;

    /// Save model to file
    async fn save(&self, path: &str) -> Result<()>;

    /// Load model from file
    async fn load(&mut self, path: &str) -> Result<()>;
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model type
    pub model_type: EmbeddingModelType,

    /// Embedding dimension
    pub embedding_dim: usize,

    /// Learning rate
    pub learning_rate: f32,

    /// L2 regularization weight
    pub l2_weight: f32,

    /// Negative sampling ratio
    pub negative_sampling_ratio: f32,

    /// Training batch size
    pub batch_size: usize,

    /// Maximum training epochs
    pub max_epochs: usize,

    /// Early stopping patience
    pub patience: usize,

    /// Validation split
    pub validation_split: f32,

    /// Enable GPU acceleration
    pub use_gpu: bool,

    /// Random seed
    pub seed: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_type: EmbeddingModelType::TransE,
            embedding_dim: 100,
            learning_rate: 0.001,
            l2_weight: 1e-5,
            negative_sampling_ratio: 1.0,
            batch_size: 1024,
            max_epochs: 1000,
            patience: 50,
            validation_split: 0.1,
            use_gpu: true,
            seed: 42,
        }
    }
}

/// Embedding model types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingModelType {
    /// Translation-based model (Bordes et al., 2013)
    TransE,

    /// Bilinear model (Yang et al., 2014)
    DistMult,

    /// Complex embeddings (Trouillon et al., 2016)
    ComplEx,

    /// Rotation-based model (Sun et al., 2019)
    RotatE,

    /// Hyperbolic embeddings (Balazevic et al., 2019)
    HypE,

    /// Tucker decomposition (Balazevic et al., 2019)
    TuckER,

    /// Convolutional model (Dettmers et al., 2018)
    ConvE,

    /// Transformer-based model
    KGTransformer,

    /// Neural tensor network (Socher et al., 2013)
    NeuralTensorNetwork,

    /// SimplE (Kazemi & Poole, 2018)
    SimplE,
}

/// Create embedding model based on configuration
pub fn create_embedding_model(
    config: EmbeddingConfig,
) -> anyhow::Result<std::sync::Arc<dyn KnowledgeGraphEmbedding>> {
    match config.model_type {
        EmbeddingModelType::TransE => Ok(std::sync::Arc::new(TransE::new(config))),
        EmbeddingModelType::DistMult => Ok(std::sync::Arc::new(DistMult::new(config))),
        EmbeddingModelType::ComplEx => Ok(std::sync::Arc::new(ComplEx::new(config))),
        _ => Err(anyhow::anyhow!("Embedding model not yet implemented")),
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f32,

    /// Maximum epochs
    pub max_epochs: usize,

    /// Validation split
    pub validation_split: f32,

    /// Early stopping patience
    pub patience: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1024,
            learning_rate: 0.001,
            max_epochs: 1000,
            validation_split: 0.1,
            patience: 50,
        }
    }
}
