//! # OxiRS Embed: Advanced Knowledge Graph Embeddings (Standalone Version)
//!
//! This is a standalone version for demonstration purposes.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Simple vector for embeddings
#[derive(Debug, Clone)]
pub struct Vector {
    pub values: Vec<f32>,
}

impl Vector {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }
}

/// Simple triple structure
#[derive(Debug, Clone, PartialEq)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    pub fn new(subject: String, predicate: String, object: String) -> Self {
        Self { subject, predicate, object }
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

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub epochs_completed: usize,
    pub final_loss: f64,
    pub training_time_seconds: f64,
    pub convergence_achieved: bool,
    pub loss_history: Vec<f64>,
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
    fn get_entities(&self) -> Vec<String>;
    fn get_relations(&self) -> Vec<String>;
    fn clear(&mut self);
    fn is_trained(&self) -> bool;
}

// Export types
pub use {ModelConfig, TrainingStats, EmbeddingModel, Vector, Triple};