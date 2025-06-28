//! Machine learning models for shape learning
//!
//! This module provides advanced ML capabilities including Graph Neural Networks,
//! decision trees, association rule learning, and reinforcement learning.

pub mod association_rules;
pub mod decision_tree;
pub mod feature_extraction;
pub mod gnn;
pub mod model_selection;
pub mod reinforcement;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Common trait for all ML models in shape learning
pub trait ShapeLearningModel: Send + Sync + std::fmt::Debug {
    /// Train the model on shape learning data
    fn train(&mut self, data: &ShapeTrainingData) -> Result<ModelMetrics, ModelError>;

    /// Predict shapes from graph data
    fn predict(&self, graph_data: &GraphData) -> Result<Vec<LearnedShape>, ModelError>;

    /// Evaluate model performance
    fn evaluate(&self, test_data: &ShapeTrainingData) -> Result<ModelMetrics, ModelError>;

    /// Get model parameters
    fn get_params(&self) -> ModelParams;

    /// Set model parameters
    fn set_params(&mut self, params: ModelParams) -> Result<(), ModelError>;

    /// Save model to disk
    fn save(&self, path: &str) -> Result<(), ModelError>;

    /// Load model from disk
    fn load(&mut self, path: &str) -> Result<(), ModelError>;
}

/// Training data for shape learning models
#[derive(Debug, Clone)]
pub struct ShapeTrainingData {
    pub graph_features: Vec<GraphFeatures>,
    pub shape_labels: Vec<ShapeLabel>,
    pub metadata: TrainingMetadata,
}

/// Graph data for prediction
#[derive(Debug, Clone)]
pub struct GraphData {
    pub nodes: Vec<NodeFeatures>,
    pub edges: Vec<EdgeFeatures>,
    pub global_features: GlobalFeatures,
}

/// Learned shape representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedShape {
    pub shape_id: String,
    pub constraints: Vec<LearnedConstraint>,
    pub confidence: f64,
    pub feature_importance: HashMap<String, f64>,
}

/// Learned constraint representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedConstraint {
    pub constraint_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub confidence: f64,
    pub support: f64,
}

/// Node features for graph neural networks
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    pub node_id: String,
    pub node_type: Option<String>,
    pub properties: HashMap<String, f64>,
    pub embedding: Option<Vec<f64>>,
}

/// Edge features for graph neural networks
#[derive(Debug, Clone)]
pub struct EdgeFeatures {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: String,
    pub properties: HashMap<String, f64>,
}

/// Global graph features
#[derive(Debug, Clone)]
pub struct GlobalFeatures {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub density: f64,
    pub clustering_coefficient: f64,
    pub diameter: Option<usize>,
    pub properties: HashMap<String, f64>,
}

/// Graph features for training
#[derive(Debug, Clone)]
pub struct GraphFeatures {
    pub graph_id: String,
    pub node_features: Vec<NodeFeatures>,
    pub edge_features: Vec<EdgeFeatures>,
    pub global_features: GlobalFeatures,
}

/// Shape label for supervised learning
#[derive(Debug, Clone)]
pub struct ShapeLabel {
    pub graph_id: String,
    pub shapes: Vec<LabeledShape>,
}

/// Labeled shape for training
#[derive(Debug, Clone)]
pub struct LabeledShape {
    pub shape_type: String,
    pub constraints: Vec<LabeledConstraint>,
    pub quality_score: f64,
}

/// Labeled constraint for training
#[derive(Debug, Clone)]
pub struct LabeledConstraint {
    pub constraint_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Training metadata
#[derive(Debug, Clone)]
pub struct TrainingMetadata {
    pub dataset_name: String,
    pub creation_date: chrono::DateTime<chrono::Utc>,
    pub num_examples: usize,
    pub split_ratio: SplitRatio,
    pub feature_stats: FeatureStatistics,
}

/// Train/validation/test split ratio
#[derive(Debug, Clone)]
pub struct SplitRatio {
    pub train: f64,
    pub validation: f64,
    pub test: f64,
}

/// Feature statistics
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    pub node_feature_dims: usize,
    pub edge_feature_dims: usize,
    pub global_feature_dims: usize,
    pub avg_nodes_per_graph: f64,
    pub avg_edges_per_graph: f64,
}

/// Model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub early_stopping_patience: usize,
    pub regularization: RegularizationParams,
    pub optimizer: OptimizerParams,
    pub model_specific: HashMap<String, serde_json::Value>,
}

/// Regularization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationParams {
    pub l1_lambda: f64,
    pub l2_lambda: f64,
    pub dropout_rate: f64,
}

/// Optimizer parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerParams {
    pub optimizer_type: OptimizerType,
    pub momentum: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
}

/// Model evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_roc: f64,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub per_class_metrics: HashMap<String, ClassMetrics>,
    pub training_time: std::time::Duration,
}

/// Per-class metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

/// Model errors
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Prediction error: {0}")]
    PredictionError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Model ensemble for combining multiple models
#[derive(Debug)]
pub struct ModelEnsemble {
    models: Vec<Box<dyn ShapeLearningModel>>,
    weights: Vec<f64>,
    voting_strategy: VotingStrategy,
}

/// Voting strategies for ensemble models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    /// Simple majority voting
    Majority,
    /// Weighted voting based on model confidence
    Weighted,
    /// Stacking with meta-learner
    Stacking,
}

impl ModelEnsemble {
    /// Create a new model ensemble
    pub fn new(voting_strategy: VotingStrategy) -> Self {
        Self {
            models: Vec::new(),
            weights: Vec::new(),
            voting_strategy,
        }
    }

    /// Add a model to the ensemble
    pub fn add_model(&mut self, model: Box<dyn ShapeLearningModel>, weight: f64) {
        self.models.push(model);
        self.weights.push(weight);
    }

    /// Predict using ensemble
    pub fn predict_ensemble(
        &self,
        graph_data: &GraphData,
    ) -> Result<Vec<LearnedShape>, ModelError> {
        let mut all_predictions = Vec::new();

        for (model, weight) in self.models.iter().zip(&self.weights) {
            let predictions = model.predict(graph_data)?;
            all_predictions.push((predictions, *weight));
        }

        match &self.voting_strategy {
            VotingStrategy::Majority => self.majority_vote(all_predictions),
            VotingStrategy::Weighted => self.weighted_vote(all_predictions),
            VotingStrategy::Stacking => {
                // For stacking, we would need a separate meta-learner
                // For now, default to weighted voting
                self.weighted_vote(all_predictions)
            }
        }
    }

    fn majority_vote(
        &self,
        predictions: Vec<(Vec<LearnedShape>, f64)>,
    ) -> Result<Vec<LearnedShape>, ModelError> {
        // Implement majority voting logic
        // For now, return first prediction
        Ok(predictions
            .first()
            .map(|(shapes, _)| shapes.clone())
            .unwrap_or_default())
    }

    fn weighted_vote(
        &self,
        predictions: Vec<(Vec<LearnedShape>, f64)>,
    ) -> Result<Vec<LearnedShape>, ModelError> {
        // Implement weighted voting logic
        // For now, return first prediction
        Ok(predictions
            .first()
            .map(|(shapes, _)| shapes.clone())
            .unwrap_or_default())
    }

    fn stacking_vote(
        &self,
        predictions: Vec<(Vec<LearnedShape>, f64)>,
        meta_learner: &dyn ShapeLearningModel,
        graph_data: &GraphData,
    ) -> Result<Vec<LearnedShape>, ModelError> {
        // Implement stacking logic with meta-learner
        // For now, use meta-learner directly
        meta_learner.predict(graph_data)
    }
}

/// Default model parameters
impl Default for ModelParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            num_epochs: 100,
            early_stopping_patience: 10,
            regularization: RegularizationParams {
                l1_lambda: 0.0,
                l2_lambda: 0.01,
                dropout_rate: 0.1,
            },
            optimizer: OptimizerParams {
                optimizer_type: OptimizerType::Adam,
                momentum: 0.9,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            model_specific: HashMap::new(),
        }
    }
}
