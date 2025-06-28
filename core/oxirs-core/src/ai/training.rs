//! Training Infrastructure for AI Models
//!
//! This module provides comprehensive training capabilities for various AI models
//! including knowledge graph embeddings, graph neural networks, and other ML models.

use crate::ai::{GraphNeuralNetwork, KnowledgeGraphEmbedding};
use crate::model::Triple;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
        self.additional_metrics
            .entry(name)
            .or_insert_with(Vec::new)
            .push(value);
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
        let mut negatives = Vec::with_capacity(num_negatives);

        // TODO: Implement proper negative sampling strategies
        // - Random corruption of head/tail entities
        // - Type-constrained negative sampling
        // - Adversarial negative sampling

        negatives
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
                // TODO: Implement BCE loss
                0.0
            }
            LossFunction::CrossEntropy => {
                // TODO: Implement CE loss
                0.0
            }
            _ => {
                // TODO: Implement other loss functions
                0.0
            }
        }
    }

    /// Compute evaluation metrics
    fn compute_metrics(
        &self,
        test_triples: &[Triple],
        model: &dyn KnowledgeGraphEmbedding,
    ) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();

        // TODO: Implement proper metric computation
        // - Mean Reciprocal Rank (MRR)
        // - Hits@K for K=1,3,10
        // - Link prediction accuracy

        metrics.insert("mrr".to_string(), 0.0);
        metrics.insert("hits_at_1".to_string(), 0.0);
        metrics.insert("hits_at_3".to_string(), 0.0);
        metrics.insert("hits_at_10".to_string(), 0.0);

        metrics
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
                let negatives = self.generate_negative_samples(batch, 1.0);

                // Forward pass (simplified)
                let mut positive_scores = Vec::new();
                let mut negative_scores = Vec::new();

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

                // TODO: Implement backward pass and parameter updates
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

            // Update metrics
            metrics.update_epoch(
                epoch,
                epoch_loss,
                val_loss,
                None, // TODO: Compute accuracy
                None, // TODO: Compute validation accuracy
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
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        // TODO: Implement GNN training
        Ok(TrainingMetrics::new())
    }

    async fn resume_training(
        &mut self,
        checkpoint_path: &str,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        // TODO: Implement checkpoint loading and resume training
        Err(anyhow!("Resume training not yet implemented"))
    }

    async fn evaluate(
        &self,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        test_data: &[Triple],
        metrics: &[TrainingMetric],
    ) -> Result<HashMap<String, f32>> {
        let computed_metrics = self.compute_metrics(test_data, model.as_ref());
        Ok(computed_metrics)
    }
}

/// Create trainer based on configuration
pub fn create_trainer(config: &TrainingConfig) -> Result<Arc<dyn Trainer>> {
    Ok(Arc::new(DefaultTrainer::new(config.clone())))
}

/// Hyperparameter optimization
pub struct HyperparameterOptimizer {
    /// Search space
    search_space: HashMap<String, ParameterRange>,

    /// Optimization strategy
    strategy: OptimizationStrategy,

    /// Number of trials
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
        objective_fn: impl Fn(TrainingConfig) -> f32,
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
        let mut config = TrainingConfig::default();
        config.lr_scheduler = LearningRateScheduler::StepDecay {
            step_size: 100,
            gamma: 0.1,
        };

        let mut trainer = DefaultTrainer::new(config);
        assert_eq!(trainer.current_lr, 0.001);

        trainer.update_learning_rate(100, None);
        assert!((trainer.current_lr - 0.0001).abs() < 1e-8);
    }
}
