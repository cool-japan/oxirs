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
        if predictions.is_empty() {
            return Ok(Vec::new());
        }

        // Collect all unique shape IDs across all predictions
        let mut shape_votes: HashMap<String, Vec<(LearnedShape, f64)>> = HashMap::new();

        for (shapes, _weight) in &predictions {
            for shape in shapes {
                shape_votes
                    .entry(shape.shape_id.clone())
                    .or_insert_with(Vec::new)
                    .push((shape.clone(), 1.0)); // Each model gets one vote
            }
        }

        // Apply majority voting: keep shapes that appear in majority of models
        let majority_threshold = (predictions.len() as f64 / 2.0).ceil() as usize;
        let mut final_shapes = Vec::new();

        for (shape_id, votes) in shape_votes {
            if votes.len() >= majority_threshold {
                // Aggregate the shape from majority votes
                let aggregated_shape = self.aggregate_shapes_majority(&votes)?;
                final_shapes.push(aggregated_shape);
            }
        }

        // Sort by aggregated confidence
        final_shapes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(final_shapes)
    }

    fn weighted_vote(
        &self,
        predictions: Vec<(Vec<LearnedShape>, f64)>,
    ) -> Result<Vec<LearnedShape>, ModelError> {
        if predictions.is_empty() {
            return Ok(Vec::new());
        }

        // Normalize weights to sum to 1.0
        let total_weight: f64 = predictions.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return self.majority_vote(predictions);
        }

        // Collect weighted votes for each shape
        let mut shape_votes: HashMap<String, Vec<(LearnedShape, f64)>> = HashMap::new();

        for (shapes, weight) in &predictions {
            let normalized_weight = weight / total_weight;
            for shape in shapes {
                shape_votes
                    .entry(shape.shape_id.clone())
                    .or_insert_with(Vec::new)
                    .push((shape.clone(), normalized_weight));
            }
        }

        // Aggregate shapes using weighted averaging
        let mut final_shapes = Vec::new();
        for (shape_id, votes) in shape_votes {
            let aggregated_shape = self.aggregate_shapes_weighted(&votes)?;

            // Only include shapes with sufficient weighted support
            let total_vote_weight: f64 = votes.iter().map(|(_, w)| w).sum();
            if total_vote_weight >= 0.3 {
                // Require at least 30% total weight support
                final_shapes.push(aggregated_shape);
            }
        }

        // Sort by weighted confidence
        final_shapes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(final_shapes)
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

    /// Aggregate shapes from majority voting
    fn aggregate_shapes_majority(
        &self,
        votes: &[(LearnedShape, f64)],
    ) -> Result<LearnedShape, ModelError> {
        if votes.is_empty() {
            return Err(ModelError::InvalidParams(
                "No votes to aggregate".to_string(),
            ));
        }

        let first_shape = &votes[0].0;
        let mut aggregated_shape = LearnedShape {
            shape_id: first_shape.shape_id.clone(),
            constraints: Vec::new(),
            confidence: 0.0,
            feature_importance: HashMap::new(),
        };

        // Aggregate confidence as simple average
        aggregated_shape.confidence =
            votes.iter().map(|(shape, _)| shape.confidence).sum::<f64>() / votes.len() as f64;

        // Aggregate constraints by finding most common ones
        let mut constraint_counts: HashMap<String, Vec<LearnedConstraint>> = HashMap::new();
        for (shape, _) in votes {
            for constraint in &shape.constraints {
                constraint_counts
                    .entry(constraint.constraint_type.clone())
                    .or_insert_with(Vec::new)
                    .push(constraint.clone());
            }
        }

        // Include constraints that appear in majority of votes
        let majority_threshold = (votes.len() as f64 / 2.0).ceil() as usize;
        for (constraint_type, constraints) in constraint_counts {
            if constraints.len() >= majority_threshold {
                // Create aggregated constraint
                let avg_confidence = constraints.iter().map(|c| c.confidence).sum::<f64>()
                    / constraints.len() as f64;
                let avg_support =
                    constraints.iter().map(|c| c.support).sum::<f64>() / constraints.len() as f64;

                // Use parameters from first constraint (could be improved with averaging)
                let parameters = constraints[0].parameters.clone();

                aggregated_shape.constraints.push(LearnedConstraint {
                    constraint_type,
                    parameters,
                    confidence: avg_confidence,
                    support: avg_support,
                });
            }
        }

        // Aggregate feature importance
        for (shape, _) in votes {
            for (feature, importance) in &shape.feature_importance {
                let current = aggregated_shape
                    .feature_importance
                    .get(feature)
                    .unwrap_or(&0.0);
                aggregated_shape
                    .feature_importance
                    .insert(feature.clone(), current + importance / votes.len() as f64);
            }
        }

        Ok(aggregated_shape)
    }

    /// Aggregate shapes from weighted voting
    fn aggregate_shapes_weighted(
        &self,
        votes: &[(LearnedShape, f64)],
    ) -> Result<LearnedShape, ModelError> {
        if votes.is_empty() {
            return Err(ModelError::InvalidParams(
                "No votes to aggregate".to_string(),
            ));
        }

        let first_shape = &votes[0].0;
        let mut aggregated_shape = LearnedShape {
            shape_id: first_shape.shape_id.clone(),
            constraints: Vec::new(),
            confidence: 0.0,
            feature_importance: HashMap::new(),
        };

        let total_weight: f64 = votes.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return self.aggregate_shapes_majority(votes);
        }

        // Weighted average of confidence
        aggregated_shape.confidence = votes
            .iter()
            .map(|(shape, weight)| shape.confidence * weight)
            .sum::<f64>()
            / total_weight;

        // Aggregate constraints with weighted voting
        let mut constraint_weights: HashMap<String, (Vec<LearnedConstraint>, f64)> = HashMap::new();
        for (shape, weight) in votes {
            for constraint in &shape.constraints {
                let entry = constraint_weights
                    .entry(constraint.constraint_type.clone())
                    .or_insert_with(|| (Vec::new(), 0.0));
                entry.0.push(constraint.clone());
                entry.1 += weight;
            }
        }

        // Include constraints with sufficient weighted support (>= 50% of total weight)
        let weight_threshold = total_weight * 0.5;
        for (constraint_type, (constraints, constraint_weight)) in constraint_weights {
            if constraint_weight >= weight_threshold {
                // Weighted average of constraint properties
                let weighted_confidence = constraints
                    .iter()
                    .zip(votes.iter().map(|(_, w)| w))
                    .map(|(constraint, weight)| constraint.confidence * weight)
                    .sum::<f64>()
                    / constraint_weight;

                let weighted_support = constraints
                    .iter()
                    .zip(votes.iter().map(|(_, w)| w))
                    .map(|(constraint, weight)| constraint.support * weight)
                    .sum::<f64>()
                    / constraint_weight;

                // Use parameters from highest-weighted constraint
                let best_constraint = constraints
                    .iter()
                    .zip(votes.iter().map(|(_, w)| w))
                    .max_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap())
                    .map(|(c, _)| c)
                    .unwrap();

                aggregated_shape.constraints.push(LearnedConstraint {
                    constraint_type,
                    parameters: best_constraint.parameters.clone(),
                    confidence: weighted_confidence,
                    support: weighted_support,
                });
            }
        }

        // Weighted average of feature importance
        for (shape, weight) in votes {
            for (feature, importance) in &shape.feature_importance {
                let current = aggregated_shape
                    .feature_importance
                    .get(feature)
                    .unwrap_or(&0.0);
                aggregated_shape.feature_importance.insert(
                    feature.clone(),
                    current + (importance * weight) / total_weight,
                );
            }
        }

        Ok(aggregated_shape)
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
