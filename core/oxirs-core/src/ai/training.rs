//! Training Infrastructure for AI Models
//!
//! This module provides comprehensive training capabilities for various AI models
//! including knowledge graph embeddings, graph neural networks, and other ML models.

use crate::ai::{GraphNeuralNetwork, KnowledgeGraphEmbedding};
use crate::model::{Triple, Subject, Predicate, Object};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use rand::seq::SliceRandom;
use rand::Rng;

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

impl Default for NegativeSamplingStrategy {
    fn default() -> Self {
        NegativeSamplingStrategy::Random
    }
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
        let mut rng = rand::thread_rng();
        
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
        let relation_vec: Vec<_> = relations.into_iter().collect();
        
        // Generate negative samples by corruption
        for _ in 0..num_negatives {
            if let Some(pos_triple) = positive_triples.choose(&mut rng) {
                match self.config.negative_sampling_strategy {
                    NegativeSamplingStrategy::Random => {
                        // Random corruption of head or tail
                        if rng.gen_bool(0.5) {
                            // Corrupt head
                            if let Some(random_subject) = subject_vec.choose(&mut rng) {
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
                            if let Some(random_object) = object_vec.choose(&mut rng) {
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
                        if rng.gen_bool(0.5) {
                            // Corrupt subject
                            if let Some(random_subject) = subject_vec.choose(&mut rng) {
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
                            if let Some(random_object) = object_vec.choose(&mut rng) {
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
                        if let Some(random_object) = object_vec.choose(&mut rng) {
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
                    let all_scores: Vec<f32> = positive_scores.iter()
                        .chain(negative_scores.iter())
                        .cloned()
                        .collect();
                    
                    let max_score = all_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_scores: Vec<f32> = all_scores.iter()
                        .map(|&s| (s - max_score).exp())
                        .collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    
                    // Cross entropy for positive examples
                    for i in 0..positive_scores.len() {
                        let prob = exp_scores[i] / sum_exp;
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
                Subject::NamedNode(nn) => { all_entities.insert(Object::NamedNode(nn.clone())); },
                Subject::BlankNode(bn) => { all_entities.insert(Object::BlankNode(bn.clone())); },
                Subject::Variable(_) => {}, // Skip variables in training evaluation
                Subject::QuotedTriple(_) => {}, // Skip quoted triples in training evaluation
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
            let head_rank = self.compute_entity_rank(
                &test_triple.subject(),
                &test_triple.predicate(),
                &test_triple.object(),
                &entity_vec,
                model,
                true, // predict head
            ).await?;

            // Tail prediction: given (h, r, ?), predict t  
            let tail_rank = self.compute_entity_rank(
                &test_triple.subject(),
                &test_triple.predicate(),
                &test_triple.object(),
                &entity_vec,
                model,
                false, // predict tail
            ).await?;

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
                
                let candidate_triple = Triple::new(
                    candidate_subject.clone(),
                    predicate.clone(),
                    correct_object.clone(),
                );
                let score = model.score_triple(
                    &candidate_subject.to_string(),
                    &predicate.to_string(),
                    &correct_object.to_string(),
                ).await?;
                scores.push((score, entity));
            }
        } else {
            // Predict tail: score all (h, r, e) combinations
            for entity in all_entities {
                let candidate_triple = Triple::new(
                    correct_subject.clone(),
                    predicate.clone(),
                    entity.clone(),
                );
                let score = model.score_triple(
                    &correct_subject.to_string(),
                    &predicate.to_string(),
                    &entity.to_string(),
                ).await?;
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
        _model: Arc<dyn GraphNeuralNetwork>,
        _training_data: &[Triple],
        _validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        // TODO: Implement GNN training
        Ok(TrainingMetrics::new())
    }

    async fn resume_training(
        &mut self,
        _checkpoint_path: &str,
        _training_data: &[Triple],
        _validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        // TODO: Implement checkpoint loading and resume training
        Err(anyhow!("Resume training not yet implemented"))
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
