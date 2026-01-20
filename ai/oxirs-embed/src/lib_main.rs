//! # OxiRS Embed: Advanced Knowledge Graph Embeddings
//!
//! This crate provides state-of-the-art knowledge graph embedding methods
//! including TransE, DistMult, ComplEx, and RotatE models. It is designed
//! to integrate seamlessly with the OxiRS ecosystem for RDF and SPARQL processing.
//!
//! ## Features
//!
//! - **Multiple Embedding Models**: TransE, DistMult, ComplEx, RotatE
//! - **High Performance**: Optimized inference engine with caching
//! - **Advanced Training**: Multiple optimizers, early stopping, learning rate scheduling
//! - **Comprehensive Evaluation**: Built-in evaluation metrics and benchmarking
//! - **OxiRS Integration**: Seamless integration with other OxiRS components
//! - **Flexible Data Loading**: Support for multiple data formats (TSV, CSV, N-Triples)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxirs_embed::{ModelConfig, models::TransE};
//! use oxirs_core::{Triple, NamedNode};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create a model configuration
//!     let config = ModelConfig::default()
//!         .with_dimensions(100)
//!         .with_learning_rate(0.01);
//!     
//!     // Initialize a TransE model
//!     let mut model = TransE::new(config);
//!     
//!     // Add training data
//!     let alice = NamedNode::new("http://example.org/alice")?;
//!     let knows = NamedNode::new("http://example.org/knows")?;
//!     let bob = NamedNode::new("http://example.org/bob")?;
//!     
//!     model.add_triple(Triple::new(alice, knows, bob))?;
//!     
//!     // Train the model
//!     let stats = model.train(Some(100)).await?;
//!     println!("Training completed in {:.2}s", stats.training_time_seconds);
//!     
//!     // Generate embeddings
//!     let alice_embedding = model.get_entity_embedding("http://example.org/alice")?;
//!     println!("Alice embedding dimensions: {}", alice_embedding.dimensions);
//!     
//!     Ok(())
//! }
//! ```

pub mod models;
pub mod training;
pub mod evaluation;
pub mod inference;
pub mod persistence;
pub mod integration;
pub mod utils;

// Re-export core types and traits
pub use crate::models::{ModelConfig, TrainingStats, ModelStats, EmbeddingError, EmbeddingModel};

// Re-export model implementations
pub use crate::models::{TransE, DistMult, ComplEx, RotatE};

// Re-export commonly used types
pub use oxirs_core::{Triple, NamedNode, Literal};
pub use oxirs_vec::Vector;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

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
    /// Set embedding dimensions
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }
    
    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    
    /// Set L2 regularization parameter
    pub fn with_l2_reg(mut self, l2_reg: f64) -> Self {
        self.l2_reg = l2_reg;
        self
    }
    
    /// Set maximum number of training epochs
    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }
    
    /// Set batch size for training
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    
    /// Set number of negative samples per positive triple
    pub fn with_negative_samples(mut self, negative_samples: usize) -> Self {
        self.negative_samples = negative_samples;
        self
    }
    
    /// Set random seed for reproducible results
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Enable or disable GPU acceleration
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }
    
    /// Add model-specific parameter
    pub fn with_param(mut self, key: &str, value: f64) -> Self {
        self.model_params.insert(key.to_string(), value);
        self
    }
}

/// Training statistics returned after model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub epochs_completed: usize,
    pub final_loss: f64,
    pub training_time_seconds: f64,
    pub convergence_achieved: bool,
    pub loss_history: Vec<f64>,
}

/// Model statistics and metadata
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

/// Custom error types for embedding operations
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Model not trained")]
    ModelNotTrained,
    
    #[error("Entity not found: {entity}")]
    EntityNotFound { entity: String },
    
    #[error("Relation not found: {relation}")]
    RelationNotFound { relation: String },
    
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },
    
    #[error("Training failed: {message}")]
    TrainingFailed { message: String },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Main trait for embedding models
#[async_trait::async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Get model configuration
    fn config(&self) -> &ModelConfig;
    
    /// Get unique model identifier
    fn model_id(&self) -> &Uuid;
    
    /// Get model type name
    fn model_type(&self) -> &'static str;
    
    /// Add a triple to the training data
    fn add_triple(&mut self, triple: Triple) -> Result<()>;
    
    /// Train the model for specified number of epochs
    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats>;
    
    /// Get embedding for an entity
    fn get_entity_embedding(&self, entity: &str) -> Result<Vector>;
    
    /// Get embedding for a relation
    fn get_relation_embedding(&self, relation: &str) -> Result<Vector>;
    
    /// Score a triple (higher score means more likely to be true)
    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64>;
    
    /// Predict top-k most likely objects for a given subject-predicate pair
    fn predict_objects(&self, subject: &str, predicate: &str, k: usize) -> Result<Vec<(String, f64)>>;
    
    /// Predict top-k most likely subjects for a given predicate-object pair
    fn predict_subjects(&self, predicate: &str, object: &str, k: usize) -> Result<Vec<(String, f64)>>;
    
    /// Predict top-k most likely relations for a given subject-object pair
    fn predict_relations(&self, subject: &str, object: &str, k: usize) -> Result<Vec<(String, f64)>>;
    
    /// Get all entities in the model
    fn get_entities(&self) -> Vec<String>;
    
    /// Get all relations in the model
    fn get_relations(&self) -> Vec<String>;
    
    /// Get model statistics
    fn get_stats(&self) -> ModelStats;
    
    /// Save model to file
    fn save(&self, path: &str) -> Result<()>;
    
    /// Load model from file
    fn load(&mut self, path: &str) -> Result<()>;
    
    /// Clear all training data and reset model
    fn clear(&mut self);
    
    /// Check if model has been trained
    fn is_trained(&self) -> bool;
}