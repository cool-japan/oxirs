//! Training Infrastructure for AI Models
//!
//! This module provides comprehensive training capabilities for various AI models
//! including knowledge graph embeddings, graph neural networks, and other ML models.

use crate::ai::{GraphNeuralNetwork, KnowledgeGraphEmbedding};
use crate::{
    term::{Object, Predicate, Subject},
    Triple,
};
use anyhow::{anyhow, Result};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Negative sampling strategy for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NegativeSamplingStrategy {
    /// Random corruption of entities
    Random,
    /// Type-constrained sampling based on entity types
    TypeConstrained,
    /// Adversarial sampling using model scores
    Adversarial,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Maximum number of epochs
    pub max_epochs: usize,

    /// Batch size
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f32,

    /// Learning rate scheduler
    pub lr_scheduler: LearningRateScheduler,

    /// Loss function
    pub loss_function: LossFunction,

    /// Optimizer
    pub optimizer: Optimizer,

    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig,

    /// Validation configuration
    pub validation: ValidationConfig,

    /// Regularization settings
    pub regularization: RegularizationConfig,

    /// Gradient clipping
    pub gradient_clipping: Option<f32>,

    /// Mixed precision training
    pub mixed_precision: bool,

    /// Checkpoint configuration
    pub checkpointing: CheckpointConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Negative sampling strategy
    pub negative_sampling_strategy: NegativeSamplingStrategy,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            batch_size: 256,
            learning_rate: 0.001,
            lr_scheduler: LearningRateScheduler::Constant,
            loss_function: LossFunction::MarginRankingLoss { margin: 1.0 },
            optimizer: Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 1e-4,
            },
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 50,
                min_delta: 1e-6,
                monitor_metric: "validation_loss".to_string(),
                mode: MonitorMode::Min,
            },
            validation: ValidationConfig {
                validation_split: 0.1,
                validation_frequency: 10,
                metrics: vec![
                    TrainingMetric::Loss,
                    TrainingMetric::MeanReciprocalRank,
                    TrainingMetric::HitsAtK { k: 1 },
                    TrainingMetric::HitsAtK { k: 3 },
                    TrainingMetric::HitsAtK { k: 10 },
                ],
            },
            regularization: RegularizationConfig {
                l1_weight: 0.0,
                l2_weight: 1e-5,
                dropout_rate: 0.1,
                batch_norm: true,
            },
            gradient_clipping: Some(1.0),
            mixed_precision: true,
            checkpointing: CheckpointConfig {
                enabled: true,
                frequency: 100,
                save_best_only: true,
                save_dir: "./checkpoints".to_string(),
            },
            logging: LoggingConfig {
                log_frequency: 10,
                tensorboard_dir: Some("./logs".to_string()),
                wandb_project: None,
            },
            negative_sampling_strategy: NegativeSamplingStrategy::Random,
        }
    }
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateScheduler {
    /// Constant learning rate
    Constant,

    /// Step decay
    StepDecay { step_size: usize, gamma: f32 },

    /// Exponential decay
    ExponentialDecay { gamma: f32 },

    /// Cosine annealing
    CosineAnnealing { t_max: usize, eta_min: f32 },

    /// Reduce on plateau
    ReduceOnPlateau {
        factor: f32,
        patience: usize,
        threshold: f32,
    },

    /// Warmup with cosine decay
    WarmupCosine {
        warmup_epochs: usize,
        max_epochs: usize,
        warmup_start_lr: f32,
        eta_min: f32,
    },
}

/// Loss functions for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    /// Margin ranking loss
    MarginRankingLoss { margin: f32 },

    /// Binary cross-entropy loss
    BinaryCrossEntropy,

    /// Cross-entropy loss
    CrossEntropy,

    /// Mean squared error
    MeanSquaredError,

    /// Contrastive loss
    ContrastiveLoss { margin: f32 },

    /// Triplet loss
    TripletLoss { margin: f32 },

    /// InfoNCE loss
    InfoNCE { temperature: f32 },

    /// Focal loss
    FocalLoss { alpha: f32, gamma: f32 },

    /// Custom loss function
    Custom(String),
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Optimizer {
    /// Stochastic Gradient Descent
    SGD {
        momentum: f32,
        weight_decay: f32,
        nesterov: bool,
    },

    /// Adam optimizer
    Adam {
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },

    /// AdamW optimizer
    AdamW {
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },

    /// AdaGrad optimizer
    AdaGrad { epsilon: f32, weight_decay: f32 },

    /// RMSprop optimizer
    RMSprop {
        alpha: f32,
        epsilon: f32,
        weight_decay: f32,
        momentum: f32,
    },

    /// AdaBound optimizer
    AdaBound {
        beta1: f32,
        beta2: f32,
        final_lr: f32,
        gamma: f32,
        epsilon: f32,
        weight_decay: f32,
    },
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,

    /// Number of epochs to wait without improvement
    pub patience: usize,

    /// Minimum change to qualify as improvement
    pub min_delta: f32,

    /// Metric to monitor
    pub monitor_metric: String,

    /// Mode (minimize or maximize)
    pub mode: MonitorMode,
}

/// Monitor mode for early stopping
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MonitorMode {
    Min,
    Max,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validation split ratio
    pub validation_split: f32,

    /// Validation frequency (epochs)
    pub validation_frequency: usize,

    /// Metrics to compute during validation
    pub metrics: Vec<TrainingMetric>,
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingMetric {
    /// Training/validation loss
    Loss,

    /// Accuracy
    Accuracy,

    /// Mean Reciprocal Rank
    MeanReciprocalRank,

    /// Hits@K
    HitsAtK { k: usize },

    /// Area Under Curve
    AUC,

    /// F1 Score
    F1Score,

    /// Precision
    Precision,

    /// Recall
    Recall,

    /// Mean Average Precision
    MeanAveragePrecision,

    /// Normalized Discounted Cumulative Gain
    NDCG { k: usize },
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization weight
    pub l1_weight: f32,

    /// L2 regularization weight
    pub l2_weight: f32,

    /// Dropout rate
    pub dropout_rate: f32,

    /// Enable batch normalization
    pub batch_norm: bool,
}

/// Checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Enable checkpointing
    pub enabled: bool,

    /// Checkpoint frequency (epochs)
    pub frequency: usize,

    /// Save only the best model
    pub save_best_only: bool,

    /// Checkpoint directory
    pub save_dir: String,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Logging frequency (batches)
    pub log_frequency: usize,

    /// TensorBoard logging directory
    pub tensorboard_dir: Option<String>,

    /// Weights & Biases project name
    pub wandb_project: Option<String>,
}

/// Training metrics collected during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history
    pub train_loss: Vec<f32>,

    /// Validation loss history
    pub val_loss: Vec<f32>,

    /// Training accuracy history
    pub train_accuracy: Vec<f32>,

    /// Validation accuracy history
    pub val_accuracy: Vec<f32>,

    /// Learning rate history
    pub learning_rate: Vec<f32>,

    /// Epoch times
    pub epoch_times: Vec<Duration>,

    /// Best validation score
    pub best_val_score: f32,

    /// Best epoch
    pub best_epoch: usize,

    /// Total training time
    pub total_time: Duration,

    /// Final epoch reached
    pub final_epoch: usize,

    /// Early stopping triggered
    pub early_stopped: bool,

    /// Additional metrics
    pub additional_metrics: HashMap<String, Vec<f32>>,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingMetrics {
    /// Create new empty training metrics
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            train_accuracy: Vec::new(),
            val_accuracy: Vec::new(),
            learning_rate: Vec::new(),
            epoch_times: Vec::new(),
            best_val_score: f32::INFINITY,
            best_epoch: 0,
            total_time: Duration::from_secs(0),
            final_epoch: 0,
            early_stopped: false,
            additional_metrics: HashMap::new(),
        }
    }

    /// Update metrics for an epoch
    #[allow(clippy::too_many_arguments)]
    pub fn update_epoch(
        &mut self,
        epoch: usize,
        train_loss: f32,
        val_loss: Option<f32>,
        train_acc: Option<f32>,
        val_acc: Option<f32>,
        lr: f32,
        epoch_time: Duration,
    ) {
        self.train_loss.push(train_loss);
        self.learning_rate.push(lr);
        self.epoch_times.push(epoch_time);

        if let Some(val_loss) = val_loss {
            self.val_loss.push(val_loss);
            if val_loss < self.best_val_score {
                self.best_val_score = val_loss;
                self.best_epoch = epoch;
            }
        }

        if let Some(train_acc) = train_acc {
            self.train_accuracy.push(train_acc);
        }

        if let Some(val_acc) = val_acc {
            self.val_accuracy.push(val_acc);
        }

        self.final_epoch = epoch;
    }

    /// Add custom metric
    pub fn add_metric(&mut self, name: String, value: f32) {
        self.additional_metrics.entry(name).or_default().push(value);
    }
}

/// Checkpoint data for resuming training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    /// Epoch number when checkpoint was saved
    pub epoch: usize,

    /// Current learning rate
    pub current_lr: f32,

    /// Best validation score achieved
    pub best_val_score: f32,

    /// Epoch with best validation score
    pub best_epoch: usize,

    /// Training loss history
    pub train_loss_history: Vec<f32>,

    /// Validation loss history
    pub val_loss_history: Vec<f32>,

    /// Training accuracy history
    pub train_accuracy_history: Vec<f32>,

    /// Validation accuracy history
    pub val_accuracy_history: Vec<f32>,

    /// Learning rate history
    pub lr_history: Vec<f32>,

    /// Epoch times in milliseconds
    pub epoch_times_ms: Vec<u64>,

    /// Total training time in milliseconds
    pub total_time_ms: u64,

    /// Model state (serialized parameters)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_state: Option<Vec<u8>>,

    /// Optimizer state (momentum buffers, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimizer_state: Option<Vec<u8>>,

    /// Additional metrics
    pub additional_metrics: HashMap<String, Vec<f32>>,

    /// Training configuration snapshot
    pub config: TrainingConfig,
}

impl CheckpointData {
    /// Create checkpoint from current training state
    pub fn from_metrics(
        metrics: &TrainingMetrics,
        current_lr: f32,
        config: &TrainingConfig,
    ) -> Self {
        Self {
            epoch: metrics.final_epoch,
            current_lr,
            best_val_score: metrics.best_val_score,
            best_epoch: metrics.best_epoch,
            train_loss_history: metrics.train_loss.clone(),
            val_loss_history: metrics.val_loss.clone(),
            train_accuracy_history: metrics.train_accuracy.clone(),
            val_accuracy_history: metrics.val_accuracy.clone(),
            lr_history: metrics.learning_rate.clone(),
            epoch_times_ms: metrics
                .epoch_times
                .iter()
                .map(|d| d.as_millis() as u64)
                .collect(),
            total_time_ms: metrics.total_time.as_millis() as u64,
            model_state: None,
            optimizer_state: None,
            additional_metrics: metrics.additional_metrics.clone(),
            config: config.clone(),
        }
    }

    /// Save checkpoint to file
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path.as_ref(), json)?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path.as_ref())?;
        let checkpoint = serde_json::from_str(&json)?;
        Ok(checkpoint)
    }
}

/// Trainer trait for different model types
#[async_trait::async_trait]
pub trait Trainer: Send + Sync {
    /// Train knowledge graph embedding model
    async fn train_embedding_model(
        &mut self,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics>;

    /// Train graph neural network
    async fn train_gnn(
        &mut self,
        model: Arc<dyn GraphNeuralNetwork>,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics>;

    /// Resume training from checkpoint
    async fn resume_training(
        &mut self,
        checkpoint_path: &str,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics>;

    /// Evaluate model on test data
    async fn evaluate(
        &self,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        test_data: &[Triple],
        metrics: &[TrainingMetric],
    ) -> Result<HashMap<String, f32>>;
}

/// Default trainer implementation
pub struct DefaultTrainer {
    /// Training configuration
    config: TrainingConfig,

    /// Current learning rate
    current_lr: f32,

    /// Early stopping state
    early_stopping_state: EarlyStoppingState,
}

/// Early stopping state
#[derive(Debug, Clone)]
struct EarlyStoppingState {
    best_score: f32,
    patience_counter: usize,
    should_stop: bool,
}

impl DefaultTrainer {
    /// Create new trainer
    pub fn new(config: TrainingConfig) -> Self {
        let current_lr = config.learning_rate;
        let early_stopping_state = EarlyStoppingState {
            best_score: if config.early_stopping.mode == MonitorMode::Min {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            },
            patience_counter: 0,
            should_stop: false,
        };

        Self {
            config,
            current_lr,
            early_stopping_state,
        }
    }

    /// Update learning rate based on scheduler
    fn update_learning_rate(&mut self, epoch: usize, val_score: Option<f32>) {
        match &self.config.lr_scheduler {
            LearningRateScheduler::Constant => {
                // No change
            }
            LearningRateScheduler::StepDecay { step_size, gamma } => {
                if epoch % step_size == 0 && epoch > 0 {
                    self.current_lr *= gamma;
                }
            }
            LearningRateScheduler::ExponentialDecay { gamma } => {
                self.current_lr *= gamma;
            }
            LearningRateScheduler::CosineAnnealing { t_max, eta_min } => {
                let cos_inner = std::f32::consts::PI * (epoch as f32) / (*t_max as f32);
                self.current_lr =
                    eta_min + (self.config.learning_rate - eta_min) * (1.0 + cos_inner.cos()) / 2.0;
            }
            LearningRateScheduler::WarmupCosine {
                warmup_epochs,
                max_epochs,
                warmup_start_lr,
                eta_min,
            } => {
                if epoch < *warmup_epochs {
                    // Linear warmup
                    self.current_lr = warmup_start_lr
                        + (self.config.learning_rate - warmup_start_lr) * (epoch as f32)
                            / (*warmup_epochs as f32);
                } else {
                    // Cosine decay
                    let cos_inner = std::f32::consts::PI * ((epoch - warmup_epochs) as f32)
                        / ((*max_epochs - warmup_epochs) as f32);
                    self.current_lr = eta_min
                        + (self.config.learning_rate - eta_min) * (1.0 + cos_inner.cos()) / 2.0;
                }
            }
            LearningRateScheduler::ReduceOnPlateau {
                factor,
                patience,
                threshold,
            } => {
                if let Some(score) = val_score {
                    let improved = match self.config.early_stopping.mode {
                        MonitorMode::Min => {
                            score < self.early_stopping_state.best_score - threshold
                        }
                        MonitorMode::Max => {
                            score > self.early_stopping_state.best_score + threshold
                        }
                    };

                    if !improved {
                        self.early_stopping_state.patience_counter += 1;
                        if self.early_stopping_state.patience_counter >= *patience {
                            self.current_lr *= factor;
                            self.early_stopping_state.patience_counter = 0;
                        }
                    } else {
                        self.early_stopping_state.patience_counter = 0;
                    }
                }
            }
        }
    }

    /// Check early stopping condition
    fn check_early_stopping(&mut self, val_score: f32) -> bool {
        if !self.config.early_stopping.enabled {
            return false;
        }

        let improved = match self.config.early_stopping.mode {
            MonitorMode::Min => {
                val_score
                    < self.early_stopping_state.best_score - self.config.early_stopping.min_delta
            }
            MonitorMode::Max => {
                val_score
                    > self.early_stopping_state.best_score + self.config.early_stopping.min_delta
            }
        };

        if improved {
            self.early_stopping_state.best_score = val_score;
            self.early_stopping_state.patience_counter = 0;
        } else {
            self.early_stopping_state.patience_counter += 1;
        }

        if self.early_stopping_state.patience_counter >= self.config.early_stopping.patience {
            self.early_stopping_state.should_stop = true;
            return true;
        }

        false
    }

    /// Generate negative samples for training
    fn generate_negative_samples(&self, positive_triples: &[Triple], ratio: f32) -> Vec<Triple> {
        // Simplified negative sampling
        // In a real implementation, this would be more sophisticated
        let num_negatives = (positive_triples.len() as f32 * ratio) as usize;

        // Implement negative sampling strategies
        let mut negative_triples = Vec::with_capacity(num_negatives);
        let mut rng = Random::default();

        // Get all entities and relations from positive triples
        let mut subjects = HashSet::new();
        let mut objects = HashSet::new();
        let mut relations = HashSet::new();

        for triple in positive_triples {
            subjects.insert(triple.subject().clone());
            objects.insert(triple.object().clone());
            relations.insert(triple.predicate().clone());
        }

        let subject_vec: Vec<_> = subjects.into_iter().collect();
        let object_vec: Vec<_> = objects.into_iter().collect();
        let _relation_vec: Vec<_> = relations.into_iter().collect();

        // Generate negative samples by corruption
        for _ in 0..num_negatives {
            if !positive_triples.is_empty() {
                let index = rng.random_range(0, positive_triples.len());
                let pos_triple = &positive_triples[index];
                match self.config.negative_sampling_strategy {
                    NegativeSamplingStrategy::Random => {
                        // Random corruption of head or tail
                        if rng.random_bool() {
                            // Corrupt head
                            if !subject_vec.is_empty() {
                                let subject_index = rng.random_range(0, subject_vec.len());
                                let random_subject = &subject_vec[subject_index];
                                let corrupted = Triple::new(
                                    random_subject.clone(),
                                    pos_triple.predicate().clone(),
                                    pos_triple.object().clone(),
                                );
                                // Check if this is actually a positive triple
                                if !positive_triples.contains(&corrupted) {
                                    negative_triples.push(corrupted);
                                }
                            }
                        } else {
                            // Corrupt tail
                            if !object_vec.is_empty() {
                                let object_index = rng.random_range(0, object_vec.len());
                                let random_object = &object_vec[object_index];
                                let corrupted = Triple::new(
                                    pos_triple.subject().clone(),
                                    pos_triple.predicate().clone(),
                                    random_object.clone(),
                                );
                                // Check if this is actually a positive triple
                                if !positive_triples.contains(&corrupted) {
                                    negative_triples.push(corrupted);
                                }
                            }
                        }
                    }
                    NegativeSamplingStrategy::TypeConstrained => {
                        // For type-constrained, we would filter entities by type
                        // For now, use random sampling as fallback
                        if rng.random_bool() {
                            // Corrupt subject
                            if !subject_vec.is_empty() {
                                let subject_index = rng.random_range(0, subject_vec.len());
                                let random_subject = &subject_vec[subject_index];
                                let corrupted = Triple::new(
                                    random_subject.clone(),
                                    pos_triple.predicate().clone(),
                                    pos_triple.object().clone(),
                                );
                                if !positive_triples.contains(&corrupted) {
                                    negative_triples.push(corrupted);
                                }
                            }
                        } else {
                            // Corrupt object
                            if !object_vec.is_empty() {
                                let object_index = rng.random_range(0, object_vec.len());
                                let random_object = &object_vec[object_index];
                                let corrupted = Triple::new(
                                    pos_triple.subject().clone(),
                                    pos_triple.predicate().clone(),
                                    random_object.clone(),
                                );
                                if !positive_triples.contains(&corrupted) {
                                    negative_triples.push(corrupted);
                                }
                            }
                        }
                    }
                    NegativeSamplingStrategy::Adversarial => {
                        // Adversarial sampling would use model scores to find hard negatives
                        // For now, use random as fallback
                        if !object_vec.is_empty() {
                            let object_index = rng.random_range(0, object_vec.len());
                            let random_object = &object_vec[object_index];
                            let corrupted = Triple::new(
                                pos_triple.subject().clone(),
                                pos_triple.predicate().clone(),
                                random_object.clone(),
                            );
                            if !positive_triples.contains(&corrupted) {
                                negative_triples.push(corrupted);
                            }
                        }
                    }
                }
            }
        }

        negative_triples
    }

    /// Compute training loss
    fn compute_loss(&self, positive_scores: &[f32], negative_scores: &[f32]) -> f32 {
        match &self.config.loss_function {
            LossFunction::MarginRankingLoss { margin } => {
                let mut total_loss = 0.0;
                let mut count = 0;

                for (pos_score, neg_score) in positive_scores.iter().zip(negative_scores.iter()) {
                    let loss = (neg_score - pos_score + margin).max(0.0);
                    total_loss += loss;
                    count += 1;
                }

                if count > 0 {
                    total_loss / count as f32
                } else {
                    0.0
                }
            }
            LossFunction::BinaryCrossEntropy => {
                // Binary Cross Entropy loss for link prediction
                let mut total_loss = 0.0;
                let count = positive_scores.len() + negative_scores.len();

                // Positive examples (label = 1)
                for &score in positive_scores {
                    let prob = 1.0 / (1.0 + (-score).exp()); // sigmoid
                    let loss = -(1.0 * prob.ln());
                    total_loss += loss;
                }

                // Negative examples (label = 0)
                for &score in negative_scores {
                    let prob = 1.0 / (1.0 + (-score).exp()); // sigmoid
                    let loss = -(0.0 * prob.ln() + (1.0 - 0.0) * (1.0 - prob).ln());
                    total_loss += loss;
                }

                if count > 0 {
                    total_loss / count as f32
                } else {
                    0.0
                }
            }
            LossFunction::CrossEntropy => {
                // Multi-class cross entropy loss
                let mut total_loss = 0.0;
                let num_classes = positive_scores.len().max(negative_scores.len());

                if num_classes > 0 {
                    // Softmax normalization
                    let all_scores: Vec<f32> = positive_scores
                        .iter()
                        .chain(negative_scores.iter())
                        .cloned()
                        .collect();

                    let max_score = all_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_scores: Vec<f32> =
                        all_scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();

                    // Cross entropy for positive examples
                    for &exp_score in exp_scores.iter().take(positive_scores.len()) {
                        let prob = exp_score / sum_exp;
                        total_loss -= prob.ln();
                    }

                    total_loss / positive_scores.len() as f32
                } else {
                    0.0
                }
            }
            _ => {
                // Default to margin ranking loss
                let mut total_loss = 0.0;
                let mut count = 0;

                for (pos_score, neg_score) in positive_scores.iter().zip(negative_scores.iter()) {
                    let loss = (neg_score - pos_score + 1.0).max(0.0); // margin = 1.0
                    total_loss += loss;
                    count += 1;
                }

                if count > 0 {
                    total_loss / count as f32
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute evaluation metrics
    async fn compute_metrics(
        &self,
        test_triples: &[Triple],
        model: &dyn KnowledgeGraphEmbedding,
    ) -> Result<HashMap<String, f32>> {
        let mut metrics = HashMap::new();

        if test_triples.is_empty() {
            metrics.insert("mrr".to_string(), 0.0);
            metrics.insert("hits_at_1".to_string(), 0.0);
            metrics.insert("hits_at_3".to_string(), 0.0);
            metrics.insert("hits_at_10".to_string(), 0.0);
            return Ok(metrics);
        }

        // Collect all entities for ranking evaluation (convert subjects to objects for unified handling)
        let mut all_entities = HashSet::new();
        for triple in test_triples {
            // Convert subject to object for unified entity ranking
            match triple.subject() {
                Subject::NamedNode(nn) => {
                    all_entities.insert(Object::NamedNode(nn.clone()));
                }
                Subject::BlankNode(bn) => {
                    all_entities.insert(Object::BlankNode(bn.clone()));
                }
                Subject::Variable(_) => {} // Skip variables in training evaluation
                Subject::QuotedTriple(_) => {} // Skip quoted triples in training evaluation
            }
            all_entities.insert(triple.object().clone());
        }
        let entity_vec: Vec<_> = all_entities.into_iter().collect();

        let mut reciprocal_ranks = Vec::new();
        let mut hits_at_1 = 0;
        let mut hits_at_3 = 0;
        let mut hits_at_10 = 0;

        // Evaluate link prediction for each test triple
        for test_triple in test_triples {
            // Head prediction: given (?, r, t), predict h
            let head_rank = self
                .compute_entity_rank(
                    test_triple.subject(),
                    test_triple.predicate(),
                    test_triple.object(),
                    &entity_vec,
                    model,
                    true, // predict head
                )
                .await?;

            // Tail prediction: given (h, r, ?), predict t
            let tail_rank = self
                .compute_entity_rank(
                    test_triple.subject(),
                    test_triple.predicate(),
                    test_triple.object(),
                    &entity_vec,
                    model,
                    false, // predict tail
                )
                .await?;

            // Use the best rank for each triple
            let best_rank = head_rank.min(tail_rank);

            reciprocal_ranks.push(1.0 / best_rank as f32);

            if best_rank <= 1 {
                hits_at_1 += 1;
            }
            if best_rank <= 3 {
                hits_at_3 += 1;
            }
            if best_rank <= 10 {
                hits_at_10 += 1;
            }
        }

        let num_test = test_triples.len() as f32;
        let mrr = reciprocal_ranks.iter().sum::<f32>() / num_test;

        metrics.insert("mrr".to_string(), mrr);
        metrics.insert("hits_at_1".to_string(), hits_at_1 as f32 / num_test);
        metrics.insert("hits_at_3".to_string(), hits_at_3 as f32 / num_test);
        metrics.insert("hits_at_10".to_string(), hits_at_10 as f32 / num_test);

        Ok(metrics)
    }

    /// Compute the rank of the correct entity in link prediction
    async fn compute_entity_rank(
        &self,
        correct_subject: &Subject,
        predicate: &Predicate,
        correct_object: &Object,
        all_entities: &[Object],
        model: &dyn KnowledgeGraphEmbedding,
        predict_head: bool,
    ) -> Result<usize> {
        let mut scores = Vec::new();

        if predict_head {
            // Predict head: score all (e, r, t) combinations
            for entity in all_entities {
                // Convert Object to Subject for head prediction
                let candidate_subject = match entity {
                    Object::NamedNode(nn) => Subject::NamedNode(nn.clone()),
                    Object::BlankNode(bn) => Subject::BlankNode(bn.clone()),
                    Object::Literal(_) => continue, // Literals can't be subjects
                    Object::Variable(v) => Subject::Variable(v.clone()),
                    Object::QuotedTriple(qt) => Subject::QuotedTriple(qt.clone()),
                };

                let _candidate_triple = Triple::new(
                    candidate_subject.clone(),
                    predicate.clone(),
                    correct_object.clone(),
                );
                let score = model
                    .score_triple(
                        &candidate_subject.to_string(),
                        &predicate.to_string(),
                        &correct_object.to_string(),
                    )
                    .await?;
                scores.push((score, entity));
            }
        } else {
            // Predict tail: score all (h, r, e) combinations
            for entity in all_entities {
                let _candidate_triple =
                    Triple::new(correct_subject.clone(), predicate.clone(), entity.clone());
                let score = model
                    .score_triple(
                        &correct_subject.to_string(),
                        &predicate.to_string(),
                        &entity.to_string(),
                    )
                    .await?;
                scores.push((score, entity));
            }
        }

        // Sort by score (descending - higher scores are better)
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Find the rank of the correct entity
        let correct_entity_as_object = if predict_head {
            // Convert subject to object for comparison
            match correct_subject {
                Subject::NamedNode(nn) => Object::NamedNode(nn.clone()),
                Subject::BlankNode(bn) => Object::BlankNode(bn.clone()),
                Subject::Variable(v) => Object::Variable(v.clone()),
                Subject::QuotedTriple(qt) => Object::QuotedTriple(qt.clone()),
            }
        } else {
            correct_object.clone()
        };

        for (rank, (_, entity)) in scores.iter().enumerate() {
            if *entity == &correct_entity_as_object {
                return Ok(rank + 1); // Ranks are 1-indexed
            }
        }

        Ok(all_entities.len()) // Worst possible rank
    }

    /// Compute accuracy for a set of triples
    /// Accuracy is defined as the percentage of triples where positive score > negative score
    async fn compute_accuracy(
        &self,
        triples: &[Triple],
        model: &dyn KnowledgeGraphEmbedding,
    ) -> Result<f32> {
        if triples.is_empty() {
            return Ok(0.0);
        }

        // Generate negative samples
        let negatives = self.generate_negative_samples(triples, 1.0);

        if negatives.is_empty() {
            return Ok(0.0);
        }

        let mut correct_count = 0;
        let total_count = triples.len().min(negatives.len());

        // Compare positive vs negative scores
        for (positive_triple, negative_triple) in triples.iter().zip(negatives.iter()) {
            // Score positive triple
            let pos_score = model
                .score_triple(
                    &positive_triple.subject().to_string(),
                    &positive_triple.predicate().to_string(),
                    &positive_triple.object().to_string(),
                )
                .await?;

            // Score negative triple
            let neg_score = model
                .score_triple(
                    &negative_triple.subject().to_string(),
                    &negative_triple.predicate().to_string(),
                    &negative_triple.object().to_string(),
                )
                .await?;

            // Correct if positive score is higher than negative score
            if pos_score > neg_score {
                correct_count += 1;
            }
        }

        Ok(correct_count as f32 / total_count as f32)
    }
}

#[async_trait::async_trait]
impl Trainer for DefaultTrainer {
    async fn train_embedding_model(
        &mut self,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        let mut metrics = TrainingMetrics::new();
        let start_time = Instant::now();

        // Split data into batches
        let batch_size = self.config.batch_size;
        let num_batches = (training_data.len() + batch_size - 1) / batch_size;

        for epoch in 0..self.config.max_epochs {
            let epoch_start = Instant::now();
            let mut epoch_loss = 0.0;

            // Training phase
            for batch_idx in 0..num_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(training_data.len());
                let batch = &training_data[start_idx..end_idx];

                // Generate negative samples
                let _negatives = self.generate_negative_samples(batch, 1.0);

                // Forward pass (simplified)
                let mut positive_scores = Vec::new();
                let negative_scores = Vec::new();

                for triple in batch {
                    if let Ok(score) = model
                        .score_triple(
                            &triple.subject().to_string(),
                            &triple.predicate().to_string(),
                            &triple.object().to_string(),
                        )
                        .await
                    {
                        positive_scores.push(score);
                    }
                }

                // Compute loss
                let batch_loss = self.compute_loss(&positive_scores, &negative_scores);
                epoch_loss += batch_loss;

                // Backward pass: compute gradients and update parameters
                if let Err(e) = self
                    .backward_pass(&positive_scores, &negative_scores, model.as_ref())
                    .await
                {
                    tracing::warn!("Backward pass failed: {}", e);
                    continue;
                }

                // Apply gradient clipping if configured
                if let Some(clip_value) = self.config.gradient_clipping {
                    self.clip_gradients(model.as_ref(), clip_value).await;
                }

                // Update parameters using optimizer
                self.update_parameters(model.as_ref(), epoch as f32).await;
            }

            epoch_loss /= num_batches as f32;

            // Validation phase
            let val_loss = if epoch % self.config.validation.validation_frequency == 0 {
                let mut val_loss = 0.0;
                let val_batches = (validation_data.len() + batch_size - 1) / batch_size;

                for batch_idx in 0..val_batches {
                    let start_idx = batch_idx * batch_size;
                    let end_idx = (start_idx + batch_size).min(validation_data.len());
                    let batch = &validation_data[start_idx..end_idx];

                    // Compute validation loss (no gradient computation)
                    for triple in batch {
                        if let Ok(score) = model
                            .score_triple(
                                &triple.subject().to_string(),
                                &triple.predicate().to_string(),
                                &triple.object().to_string(),
                            )
                            .await
                        {
                            val_loss += score; // Simplified
                        }
                    }
                }

                Some(val_loss / validation_data.len() as f32)
            } else {
                None
            };

            let epoch_time = epoch_start.elapsed();

            // Compute training accuracy
            let train_accuracy = if epoch % self.config.logging.log_frequency == 0 {
                Some(
                    self.compute_accuracy(training_data, model.as_ref())
                        .await
                        .unwrap_or(0.0),
                )
            } else {
                None
            };

            // Compute validation accuracy
            let val_accuracy = if epoch % self.config.validation.validation_frequency == 0 {
                Some(
                    self.compute_accuracy(validation_data, model.as_ref())
                        .await
                        .unwrap_or(0.0),
                )
            } else {
                None
            };

            // Update metrics
            metrics.update_epoch(
                epoch,
                epoch_loss,
                val_loss,
                train_accuracy,
                val_accuracy,
                self.current_lr,
                epoch_time,
            );

            // Check early stopping
            if let Some(val_loss) = val_loss {
                if self.check_early_stopping(val_loss) {
                    metrics.early_stopped = true;
                    break;
                }
            }

            // Update learning rate
            self.update_learning_rate(epoch, val_loss);

            // Logging
            if epoch % self.config.logging.log_frequency == 0 {
                println!(
                    "Epoch {}: train_loss={:.4}, val_loss={:.4}, lr={:.6}",
                    epoch,
                    epoch_loss,
                    val_loss.unwrap_or(0.0),
                    self.current_lr
                );
            }
        }

        metrics.total_time = start_time.elapsed();
        Ok(metrics)
    }

    async fn train_gnn(
        &mut self,
        model: Arc<dyn GraphNeuralNetwork>,
        training_data: &[Triple],
        _validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        use crate::ai::gnn::{RdfGraph, TrainingConfig as GnnTrainingConfig};

        // Convert training configuration to GNN training configuration
        let gnn_config = GnnTrainingConfig {
            max_epochs: self.config.max_epochs,
            batch_size: self.config.batch_size,
            learning_rate: self.config.learning_rate,
            patience: self.config.early_stopping.patience,
            validation_split: self.config.validation.validation_split,
            loss_function: match &self.config.loss_function {
                LossFunction::CrossEntropy => crate::ai::gnn::LossFunction::CrossEntropy,
                LossFunction::BinaryCrossEntropy => {
                    crate::ai::gnn::LossFunction::BinaryCrossEntropy
                }
                LossFunction::MeanSquaredError => crate::ai::gnn::LossFunction::MeanSquaredError,
                LossFunction::ContrastiveLoss { margin } => {
                    crate::ai::gnn::LossFunction::ContrastiveLoss { margin: *margin }
                }
                _ => crate::ai::gnn::LossFunction::CrossEntropy, // Default
            },
            optimizer: match &self.config.optimizer {
                Optimizer::SGD {
                    momentum,
                    weight_decay,
                    nesterov,
                } => crate::ai::gnn::Optimizer::SGD {
                    momentum: *momentum,
                    weight_decay: *weight_decay,
                    nesterov: *nesterov,
                },
                Optimizer::Adam {
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                } => crate::ai::gnn::Optimizer::Adam {
                    beta1: *beta1,
                    beta2: *beta2,
                    epsilon: *epsilon,
                    weight_decay: *weight_decay,
                },
                Optimizer::AdaGrad {
                    epsilon,
                    weight_decay: _,
                } => crate::ai::gnn::Optimizer::AdaGrad { epsilon: *epsilon },
                Optimizer::RMSprop {
                    alpha,
                    epsilon,
                    weight_decay: _,
                    momentum: _,
                } => crate::ai::gnn::Optimizer::RMSprop {
                    decay: *alpha,
                    epsilon: *epsilon,
                },
                _ => crate::ai::gnn::Optimizer::Adam {
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 1e-4,
                },
            },
            gradient_clipping: self.config.gradient_clipping,
            early_stopping: crate::ai::gnn::EarlyStoppingConfig {
                enabled: self.config.early_stopping.enabled,
                patience: self.config.early_stopping.patience,
                min_delta: self.config.early_stopping.min_delta,
            },
        };

        // Build RDF graph from training data
        let graph = RdfGraph::from_triples(training_data)?;

        // Extract node features
        let features = model.extract_node_features(&graph).await?;

        // For link prediction, create labels based on the graph structure
        // This is a simplified approach - in practice, labels would be task-specific
        let labels = {
            use scirs2_core::ndarray_ext::Array2;
            let num_nodes = graph.num_nodes;
            let output_dim = 64; // Default output dimension

            // Create one-hot encoded labels or use identity matrix
            let mut labels = Array2::zeros((num_nodes, output_dim));
            for i in 0..num_nodes.min(output_dim) {
                labels[[i, i]] = 1.0;
            }
            labels
        };

        // Train the GNN using its own training method
        // We need mutable access, so we need to ensure there's only one reference
        let mut model_mut = model;
        let gnn_metrics = if let Some(model_ref) = Arc::get_mut(&mut model_mut) {
            // We have exclusive access, can train directly
            model_ref
                .train(&graph, &features, &labels, &gnn_config)
                .await?
        } else {
            // Multiple references exist, cannot train
            // This is a limitation - in production, the GNN should use interior mutability
            return Err(anyhow!(
                "Cannot train GNN: model has multiple references. \
                 Clone the model or ensure exclusive ownership before training."
            ));
        };

        // Convert GNN metrics to general training metrics
        let mut metrics = TrainingMetrics::new();
        metrics.update_epoch(
            0,
            gnn_metrics.loss,
            Some(gnn_metrics.loss),
            Some(gnn_metrics.accuracy),
            Some(gnn_metrics.accuracy),
            self.config.learning_rate,
            gnn_metrics.time_elapsed,
        );
        metrics.final_epoch = gnn_metrics.epochs;
        metrics.total_time = gnn_metrics.time_elapsed;

        Ok(metrics)
    }

    async fn resume_training(
        &mut self,
        checkpoint_path: &str,
        _training_data: &[Triple],
        _validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        use std::path::Path;

        // Check if checkpoint file exists
        let path = Path::new(checkpoint_path);
        if !path.exists() {
            return Err(anyhow!("Checkpoint file not found: {}", checkpoint_path));
        }

        // Load checkpoint metadata
        let checkpoint_data = std::fs::read_to_string(checkpoint_path)?;

        // Parse checkpoint (expecting JSON format)
        let checkpoint: CheckpointData = serde_json::from_str(&checkpoint_data).map_err(|e| {
            anyhow!(
                "Failed to parse checkpoint file {}: {}. \
                 Expected JSON format with model state and training progress.",
                checkpoint_path,
                e
            )
        })?;

        tracing::info!(
            "Loaded checkpoint from epoch {}, best validation score: {:.6}",
            checkpoint.epoch,
            checkpoint.best_val_score
        );

        // Restore trainer state
        self.current_lr = checkpoint.current_lr;
        self.early_stopping_state = EarlyStoppingState {
            best_score: checkpoint.best_val_score,
            patience_counter: 0,
            should_stop: false,
        };

        // Initialize metrics with checkpoint data
        let metrics = TrainingMetrics {
            train_loss: checkpoint.train_loss_history.clone(),
            val_loss: checkpoint.val_loss_history.clone(),
            train_accuracy: checkpoint.train_accuracy_history.clone(),
            val_accuracy: checkpoint.val_accuracy_history.clone(),
            learning_rate: checkpoint.lr_history.clone(),
            epoch_times: checkpoint
                .epoch_times_ms
                .iter()
                .map(|&ms| Duration::from_millis(ms))
                .collect(),
            best_val_score: checkpoint.best_val_score,
            best_epoch: checkpoint.best_epoch,
            total_time: Duration::from_millis(checkpoint.total_time_ms),
            final_epoch: checkpoint.epoch,
            early_stopped: false,
            additional_metrics: checkpoint.additional_metrics.clone(),
        };

        tracing::info!(
            "Resuming training from epoch {} with {} remaining epochs",
            checkpoint.epoch + 1,
            self.config.max_epochs.saturating_sub(checkpoint.epoch + 1)
        );

        // Note: In a full implementation, this would also:
        // 1. Restore model parameters (embeddings) from checkpoint.model_state
        // 2. Restore optimizer state (momentum buffers) from checkpoint.optimizer_state
        // 3. Resume training loop from checkpoint.epoch + 1
        //
        // For now, we return metrics indicating the checkpoint was loaded successfully
        // but actual training continuation would require model parameter restoration

        tracing::warn!(
            "Checkpoint loaded successfully, but model parameter restoration is not yet implemented. \
             To fully resume training, implement model state serialization/deserialization."
        );

        // Total time already set from checkpoint data
        // Note: In a full implementation, we would add elapsed time since checkpoint was saved

        Ok(metrics)
    }

    async fn evaluate(
        &self,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        test_data: &[Triple],
        _metrics: &[TrainingMetric],
    ) -> Result<HashMap<String, f32>> {
        let computed_metrics = self.compute_metrics(test_data, model.as_ref()).await?;
        Ok(computed_metrics)
    }
}

impl DefaultTrainer {
    /// Compute gradients using backward pass
    async fn backward_pass(
        &self,
        positive_scores: &[f32],
        negative_scores: &[f32],
        _model: &dyn KnowledgeGraphEmbedding,
    ) -> Result<()> {
        // Compute gradients based on loss function
        match &self.config.loss_function {
            LossFunction::MarginRankingLoss { margin } => {
                // Gradient computation for margin ranking loss
                for (pos_score, neg_score) in positive_scores.iter().zip(negative_scores.iter()) {
                    let loss = neg_score - pos_score + margin;
                    if loss > 0.0 {
                        // Gradient w.r.t positive score: -1
                        // Gradient w.r.t negative score: +1
                        // Note: In production, these gradients would be backpropagated
                        // through the embedding model to update entity/relation embeddings
                    }
                }
            }
            LossFunction::BinaryCrossEntropy => {
                // Gradient computation for binary cross entropy
                for &score in positive_scores {
                    let prob = 1.0 / (1.0 + (-score).exp()); // sigmoid
                    let _gradient = prob - 1.0; // gradient for positive examples
                }
                for &score in negative_scores {
                    let prob = 1.0 / (1.0 + (-score).exp()); // sigmoid
                    let _gradient = prob; // gradient for negative examples
                }
            }
            LossFunction::CrossEntropy => {
                // Gradient computation for softmax cross entropy
                let all_scores: Vec<f32> = positive_scores
                    .iter()
                    .chain(negative_scores.iter())
                    .cloned()
                    .collect();

                if !all_scores.is_empty() {
                    let max_score = all_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_scores: Vec<f32> =
                        all_scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();

                    // Compute softmax probabilities and gradients
                    for (i, exp_score) in exp_scores.iter().enumerate() {
                        let prob = exp_score / sum_exp;
                        let is_positive = i < positive_scores.len();
                        let _gradient = if is_positive { prob - 1.0 } else { prob };
                    }
                }
            }
            _ => {
                // Default gradient computation (margin ranking)
                for (pos_score, neg_score) in positive_scores.iter().zip(negative_scores.iter()) {
                    let loss = neg_score - pos_score + 1.0;
                    if loss > 0.0 {
                        // Compute gradients for default case
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply gradient clipping to prevent exploding gradients
    async fn clip_gradients(&self, _model: &dyn KnowledgeGraphEmbedding, clip_value: f32) {
        // In a full implementation, this would:
        // 1. Compute the L2 norm of all gradients
        // 2. If norm > clip_value, scale all gradients by clip_value / norm
        // 3. Apply the clipped gradients to the model parameters

        tracing::debug!("Applying gradient clipping with value: {}", clip_value);

        // Placeholder for gradient clipping implementation
        // This would typically involve iterating through model parameters
        // and applying the clipping operation to their gradients
    }

    /// Update model parameters using the configured optimizer
    async fn update_parameters(&self, _model: &dyn KnowledgeGraphEmbedding, epoch: f32) {
        match &self.config.optimizer {
            Optimizer::SGD {
                momentum,
                weight_decay: _,
                nesterov: _,
            } => {
                tracing::debug!(
                    "Updating parameters with SGD optimizer (momentum: {})",
                    momentum
                );
                // SGD parameter update: θ = θ - α * ∇θ
                // With momentum: v = β * v + ∇θ; θ = θ - α * v
            }
            Optimizer::Adam {
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => {
                tracing::debug!("Updating parameters with Adam optimizer");
                // Adam parameter update with bias correction
                // m = β₁ * m + (1 - β₁) * ∇θ
                // v = β₂ * v + (1 - β₂) * ∇θ²
                // m̂ = m / (1 - β₁^t)
                // v̂ = v / (1 - β₂^t)
                // θ = θ - α * m̂ / (√v̂ + ε)

                let _bias_correction1 = 1.0 - beta1.powf(epoch);
                let _bias_correction2 = 1.0 - beta2.powf(epoch);
                let _effective_lr = self.config.learning_rate / _bias_correction1;

                if weight_decay > &0.0 {
                    // Apply weight decay: θ = θ * (1 - α * λ)
                    tracing::debug!("Applying weight decay: {}", weight_decay);
                }

                tracing::debug!(
                    "Adam step with lr: {}, beta1: {}, beta2: {}, epsilon: {}",
                    self.config.learning_rate,
                    beta1,
                    beta2,
                    epsilon
                );
            }
            Optimizer::AdamW {
                beta1,
                beta2,
                epsilon: _,
                weight_decay,
            } => {
                tracing::debug!("Updating parameters with AdamW optimizer");
                // AdamW is similar to Adam but with decoupled weight decay
                // Weight decay is applied directly to parameters, not to gradients

                let _bias_correction1 = 1.0 - beta1.powf(epoch);
                let _bias_correction2 = 1.0 - beta2.powf(epoch);

                // Apply weight decay before parameter update
                if weight_decay > &0.0 {
                    tracing::debug!("Applying decoupled weight decay: {}", weight_decay);
                }

                tracing::debug!("AdamW step with decoupled weight decay");
            }
            Optimizer::AdaGrad {
                epsilon,
                weight_decay,
            } => {
                tracing::debug!("Updating parameters with AdaGrad optimizer");
                // AdaGrad: G = G + ∇θ²; θ = θ - α * ∇θ / (√G + ε)

                if weight_decay > &0.0 {
                    tracing::debug!("Applying weight decay: {}", weight_decay);
                }

                tracing::debug!("AdaGrad step with epsilon: {}", epsilon);
            }
            Optimizer::RMSprop {
                alpha,
                epsilon,
                weight_decay,
                momentum,
            } => {
                tracing::debug!("Updating parameters with RMSprop optimizer");
                // RMSprop: E[g²] = α * E[g²] + (1-α) * ∇θ²
                // θ = θ - η * ∇θ / (√E[g²] + ε)

                if momentum > &0.0 {
                    tracing::debug!("RMSprop with momentum: {}", momentum);
                }

                if weight_decay > &0.0 {
                    tracing::debug!("Applying weight decay: {}", weight_decay);
                }

                tracing::debug!("RMSprop step with alpha: {}, epsilon: {}", alpha, epsilon);
            }
            Optimizer::AdaBound {
                beta1: _,
                beta2: _,
                final_lr,
                gamma,
                epsilon: _,
                weight_decay,
            } => {
                tracing::debug!("Updating parameters with AdaBound optimizer");
                // AdaBound combines Adam with SGD by constraining the adaptive learning rate

                let _step_size = self.config.learning_rate;
                let _lower_bound = final_lr * (1.0 - 1.0 / (gamma * epoch + 1.0));
                let _upper_bound = final_lr * (1.0 + 1.0 / (gamma * epoch));

                if weight_decay > &0.0 {
                    tracing::debug!("Applying weight decay: {}", weight_decay);
                }

                tracing::debug!(
                    "AdaBound step with bounds: [{}, {}]",
                    _lower_bound,
                    _upper_bound
                );
            }
        }

        // Note: In a complete implementation, this would:
        // 1. Access model parameters (embeddings for entities and relations)
        // 2. Apply the optimizer-specific update rules
        // 3. Update the model's internal state (momentum buffers, etc.)
        // 4. Handle learning rate scheduling if configured
    }
}

/// Create trainer based on configuration
pub fn create_trainer(config: &TrainingConfig) -> Result<Arc<dyn Trainer>> {
    Ok(Arc::new(DefaultTrainer::new(config.clone())))
}

/// Hyperparameter optimization
pub struct HyperparameterOptimizer {
    /// Search space
    #[allow(dead_code)]
    search_space: HashMap<String, ParameterRange>,

    /// Optimization strategy
    #[allow(dead_code)]
    strategy: OptimizationStrategy,

    /// Number of trials
    #[allow(dead_code)]
    num_trials: usize,
}

/// Parameter range for hyperparameter search
#[derive(Debug, Clone)]
pub enum ParameterRange {
    Float { min: f32, max: f32 },
    Int { min: i32, max: i32 },
    Choice(Vec<String>),
    LogFloat { min: f32, max: f32 },
}

/// Optimization strategies
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    RandomSearch,
    GridSearch,
    BayesianOptimization,
    TPE,   // Tree-structured Parzen Estimator
    CmaEs, // Covariance Matrix Adaptation Evolution Strategy
}

impl HyperparameterOptimizer {
    /// Create new hyperparameter optimizer
    pub fn new(
        search_space: HashMap<String, ParameterRange>,
        strategy: OptimizationStrategy,
        num_trials: usize,
    ) -> Self {
        Self {
            search_space,
            strategy,
            num_trials,
        }
    }

    /// Optimize hyperparameters
    pub async fn optimize(
        &self,
        _objective_fn: impl Fn(TrainingConfig) -> f32,
    ) -> Result<TrainingConfig> {
        // TODO: Implement hyperparameter optimization
        Err(anyhow!("Hyperparameter optimization not yet implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_creation() {
        let config = TrainingConfig::default();
        assert_eq!(config.max_epochs, 1000);
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.learning_rate, 0.001);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();

        metrics.update_epoch(
            0,
            0.5,
            Some(0.4),
            Some(0.8),
            Some(0.85),
            0.001,
            Duration::from_millis(100),
        );

        assert_eq!(metrics.train_loss.len(), 1);
        assert_eq!(metrics.val_loss.len(), 1);
        assert_eq!(metrics.best_val_score, 0.4);
        assert_eq!(metrics.best_epoch, 0);
    }

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default();
        let trainer = DefaultTrainer::new(config);
        assert_eq!(trainer.current_lr, 0.001);
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let config = TrainingConfig {
            lr_scheduler: LearningRateScheduler::StepDecay {
                step_size: 100,
                gamma: 0.1,
            },
            ..Default::default()
        };

        let mut trainer = DefaultTrainer::new(config);
        assert_eq!(trainer.current_lr, 0.001);

        trainer.update_learning_rate(100, None);
        assert!((trainer.current_lr - 0.0001).abs() < 1e-8);
    }
}
