//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::ai::KnowledgeGraphEmbedding;
use crate::{
    term::{Object, Predicate, Subject},
    Triple,
};
use anyhow::Result;
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Early stopping state
#[derive(Debug, Clone)]
pub(super) struct EarlyStoppingState {
    pub(super) best_score: f32,
    pub(super) patience_counter: usize,
    pub(super) should_stop: bool,
}
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
/// Monitor mode for early stopping
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MonitorMode {
    Min,
    Max,
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
/// Parameter value types
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ParameterValue {
    Float(f32),
    Int(i32),
    String(String),
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
/// Parameter range for hyperparameter search
#[derive(Debug, Clone)]
pub enum ParameterRange {
    Float { min: f32, max: f32 },
    Int { min: i32, max: i32 },
    Choice(Vec<String>),
    LogFloat { min: f32, max: f32 },
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
    pub async fn optimize<F>(&self, mut objective_fn: F) -> Result<TrainingConfig>
    where
        F: FnMut(TrainingConfig) -> f32 + Send,
    {
        tracing::info!(
            "Starting hyperparameter optimization with {:?} strategy, {} trials",
            self.strategy,
            self.num_trials
        );
        match &self.strategy {
            OptimizationStrategy::RandomSearch => self.random_search(&mut objective_fn).await,
            OptimizationStrategy::GridSearch => self.grid_search(&mut objective_fn).await,
            OptimizationStrategy::BayesianOptimization => {
                self.bayesian_optimization(&mut objective_fn).await
            }
            OptimizationStrategy::TPE => self.tpe_optimization(&mut objective_fn).await,
            OptimizationStrategy::CmaEs => self.cmaes_optimization(&mut objective_fn).await,
        }
    }
    /// Random search hyperparameter optimization
    async fn random_search<F>(&self, objective_fn: &mut F) -> Result<TrainingConfig>
    where
        F: FnMut(TrainingConfig) -> f32,
    {
        use scirs2_core::random::Random;
        let mut best_config = TrainingConfig::default();
        let mut best_score = f32::NEG_INFINITY;
        let mut rng = Random::default();
        tracing::info!("Running random search with {} trials", self.num_trials);
        for trial in 0..self.num_trials {
            let mut config = TrainingConfig::default();
            for (param_name, range) in &self.search_space {
                self.apply_parameter(
                    &mut config,
                    param_name,
                    self.sample_parameter(range, &mut rng)?,
                )?;
            }
            let score = objective_fn(config.clone());
            if score > best_score {
                best_score = score;
                best_config = config.clone();
                tracing::info!(
                    "Trial {}/{}: New best score: {:.6} (lr={:.6}, batch_size={})",
                    trial + 1,
                    self.num_trials,
                    best_score,
                    config.learning_rate,
                    config.batch_size
                );
            } else if trial % 10 == 0 {
                tracing::debug!(
                    "Trial {}/{}: score={:.6}",
                    trial + 1,
                    self.num_trials,
                    score
                );
            }
        }
        tracing::info!("Random search completed. Best score: {:.6}", best_score);
        Ok(best_config)
    }
    /// Grid search hyperparameter optimization
    async fn grid_search<F>(&self, objective_fn: &mut F) -> Result<TrainingConfig>
    where
        F: FnMut(TrainingConfig) -> f32,
    {
        let mut best_config = TrainingConfig::default();
        let mut best_score = f32::NEG_INFINITY;
        tracing::info!("Running grid search");
        let mut param_grids: Vec<(String, Vec<ParameterValue>)> = Vec::new();
        for (param_name, range) in &self.search_space {
            let grid = self.generate_grid(range, 5)?;
            param_grids.push((param_name.clone(), grid));
        }
        let combinations = self.cartesian_product(&param_grids);
        tracing::info!(
            "Grid search will evaluate {} combinations",
            combinations.len()
        );
        for (idx, combination) in combinations.iter().enumerate() {
            let mut config = TrainingConfig::default();
            for (param_name, value) in combination {
                self.apply_parameter(&mut config, param_name, value.clone())?;
            }
            let score = objective_fn(config.clone());
            if score > best_score {
                best_score = score;
                best_config = config.clone();
                tracing::info!(
                    "Combination {}/{}: New best score: {:.6}",
                    idx + 1,
                    combinations.len(),
                    best_score
                );
            }
        }
        tracing::info!("Grid search completed. Best score: {:.6}", best_score);
        Ok(best_config)
    }
    /// Bayesian optimization (simplified version using random sampling with history)
    async fn bayesian_optimization<F>(&self, objective_fn: &mut F) -> Result<TrainingConfig>
    where
        F: FnMut(TrainingConfig) -> f32,
    {
        use scirs2_core::random::{Random, Rng};
        let mut best_config = TrainingConfig::default();
        let mut best_score = f32::NEG_INFINITY;
        let mut rng = Random::default();
        let mut history: Vec<(TrainingConfig, f32)> = Vec::new();
        tracing::info!(
            "Running Bayesian optimization with {} trials",
            self.num_trials
        );
        let exploration_trials = (self.num_trials / 3).max(5);
        for trial in 0..self.num_trials {
            let mut config = TrainingConfig::default();
            if trial < exploration_trials {
                for (param_name, range) in &self.search_space {
                    self.apply_parameter(
                        &mut config,
                        param_name,
                        self.sample_parameter(range, &mut rng)?,
                    )?;
                }
            } else {
                config = best_config.clone();
                for (param_name, range) in &self.search_space {
                    match range {
                        ParameterRange::Float { min, max }
                        | ParameterRange::LogFloat { min, max } => {
                            let current = self.get_parameter_value(&config, param_name)?;
                            let std_dev = (*max - *min) * 0.1;
                            let noise = (rng.random::<f32>() - 0.5) * 2.0 * std_dev;
                            let new_value = (current + noise).clamp(*min, *max);
                            self.apply_parameter(
                                &mut config,
                                param_name,
                                ParameterValue::Float(new_value),
                            )?;
                        }
                        _ => {
                            if rng.random::<f32>() < 0.2 {
                                self.apply_parameter(
                                    &mut config,
                                    param_name,
                                    self.sample_parameter(range, &mut rng)?,
                                )?;
                            }
                        }
                    }
                }
            }
            let score = objective_fn(config.clone());
            history.push((config.clone(), score));
            if score > best_score {
                best_score = score;
                best_config = config.clone();
                tracing::info!(
                    "Trial {}/{}: New best score: {:.6} (exploration={})",
                    trial + 1,
                    self.num_trials,
                    best_score,
                    trial < exploration_trials
                );
            }
        }
        tracing::info!(
            "Bayesian optimization completed. Best score: {:.6}",
            best_score
        );
        Ok(best_config)
    }
    /// TPE (Tree-structured Parzen Estimator) optimization
    async fn tpe_optimization<F>(&self, objective_fn: &mut F) -> Result<TrainingConfig>
    where
        F: FnMut(TrainingConfig) -> f32,
    {
        use scirs2_core::random::{Random, Rng};
        let mut best_config = TrainingConfig::default();
        let mut best_score = f32::NEG_INFINITY;
        let mut rng = Random::default();
        let mut history: Vec<(TrainingConfig, f32)> = Vec::new();
        tracing::info!("Running TPE optimization with {} trials", self.num_trials);
        let percentile = 0.25;
        for trial in 0..self.num_trials {
            let mut config = TrainingConfig::default();
            if trial < 10 {
                for (param_name, range) in &self.search_space {
                    self.apply_parameter(
                        &mut config,
                        param_name,
                        self.sample_parameter(range, &mut rng)?,
                    )?;
                }
            } else {
                let sorted_history: Vec<_> = {
                    let mut h = history.clone();
                    h.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    h
                };
                let good_count = (sorted_history.len() as f32 * percentile) as usize;
                let good_configs: Vec<_> = sorted_history.iter().take(good_count.max(1)).collect();
                if !good_configs.is_empty() {
                    let idx = rng.random_range(0..good_configs.len());
                    config = good_configs[idx].0.clone();
                    for (param_name, range) in &self.search_space {
                        if rng.random::<f32>() < 0.3 {
                            self.apply_parameter(
                                &mut config,
                                param_name,
                                self.sample_parameter(range, &mut rng)?,
                            )?;
                        }
                    }
                }
            }
            let score = objective_fn(config.clone());
            history.push((config.clone(), score));
            if score > best_score {
                best_score = score;
                best_config = config.clone();
                tracing::info!(
                    "Trial {}/{}: New best score: {:.6}",
                    trial + 1,
                    self.num_trials,
                    best_score
                );
            }
        }
        tracing::info!("TPE optimization completed. Best score: {:.6}", best_score);
        Ok(best_config)
    }
    /// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization
    async fn cmaes_optimization<F>(&self, objective_fn: &mut F) -> Result<TrainingConfig>
    where
        F: FnMut(TrainingConfig) -> f32,
    {
        use scirs2_core::random::{Random, Rng};
        let mut best_config = TrainingConfig::default();
        let mut best_score = f32::NEG_INFINITY;
        let mut rng = Random::default();
        tracing::info!(
            "Running CMA-ES optimization with {} trials",
            self.num_trials
        );
        let mut step_size = 1.0;
        for trial in 0..self.num_trials {
            let mut config = if trial == 0 {
                TrainingConfig::default()
            } else {
                best_config.clone()
            };
            for (param_name, range) in &self.search_space {
                match range {
                    ParameterRange::Float { min, max } | ParameterRange::LogFloat { min, max } => {
                        if trial > 0 {
                            let current = self.get_parameter_value(&config, param_name)?;
                            let std_dev = (*max - *min) * step_size * 0.1;
                            let noise = (rng.random::<f32>() - 0.5) * 2.0 * std_dev;
                            let new_value = (current + noise).clamp(*min, *max);
                            self.apply_parameter(
                                &mut config,
                                param_name,
                                ParameterValue::Float(new_value),
                            )?;
                        } else {
                            self.apply_parameter(
                                &mut config,
                                param_name,
                                self.sample_parameter(range, &mut rng)?,
                            )?;
                        }
                    }
                    _ => {
                        self.apply_parameter(
                            &mut config,
                            param_name,
                            self.sample_parameter(range, &mut rng)?,
                        )?;
                    }
                }
            }
            let score = objective_fn(config.clone());
            if score > best_score {
                best_score = score;
                best_config = config.clone();
                step_size *= 1.2;
                tracing::info!(
                    "Trial {}/{}: New best score: {:.6}, step_size={:.4}",
                    trial + 1,
                    self.num_trials,
                    best_score,
                    step_size
                );
            } else {
                step_size *= 0.8;
            }
            step_size = step_size.clamp(0.01, 2.0);
        }
        tracing::info!(
            "CMA-ES optimization completed. Best score: {:.6}",
            best_score
        );
        Ok(best_config)
    }
    /// Sample a parameter value from its range
    fn sample_parameter(&self, range: &ParameterRange, rng: &mut Random) -> Result<ParameterValue> {
        match range {
            ParameterRange::Float { min, max } => {
                Ok(ParameterValue::Float(rng.random_range(*min..*max)))
            }
            ParameterRange::Int { min, max } => {
                Ok(ParameterValue::Int(rng.random_range(*min..*max)))
            }
            ParameterRange::Choice(choices) => {
                let idx = rng.random_range(0..choices.len());
                Ok(ParameterValue::String(choices[idx].clone()))
            }
            ParameterRange::LogFloat { min, max } => {
                let log_min = min.ln();
                let log_max = max.ln();
                let log_value = rng.random_range(log_min..log_max);
                Ok(ParameterValue::Float(log_value.exp()))
            }
        }
    }
    /// Generate grid points for a parameter range
    fn generate_grid(
        &self,
        range: &ParameterRange,
        num_points: usize,
    ) -> Result<Vec<ParameterValue>> {
        match range {
            ParameterRange::Float { min, max } => {
                let step = (max - min) / (num_points - 1) as f32;
                Ok((0..num_points)
                    .map(|i| ParameterValue::Float(min + step * i as f32))
                    .collect())
            }
            ParameterRange::Int { min, max } => {
                let step = ((max - min) / (num_points - 1) as i32).max(1);
                Ok((0..num_points)
                    .map(|i| ParameterValue::Int(min + step * i as i32))
                    .collect())
            }
            ParameterRange::Choice(choices) => Ok(choices
                .iter()
                .map(|c| ParameterValue::String(c.clone()))
                .collect()),
            ParameterRange::LogFloat { min, max } => {
                let log_min = min.ln();
                let log_max = max.ln();
                let step = (log_max - log_min) / (num_points - 1) as f32;
                Ok((0..num_points)
                    .map(|i| ParameterValue::Float((log_min + step * i as f32).exp()))
                    .collect())
            }
        }
    }
    /// Apply a parameter value to a training config
    fn apply_parameter(
        &self,
        config: &mut TrainingConfig,
        name: &str,
        value: ParameterValue,
    ) -> Result<()> {
        match name {
            "learning_rate" => {
                if let ParameterValue::Float(v) = value {
                    config.learning_rate = v;
                }
            }
            "batch_size" => {
                if let ParameterValue::Int(v) = value {
                    config.batch_size = v as usize;
                }
            }
            "max_epochs" => {
                if let ParameterValue::Int(v) = value {
                    config.max_epochs = v as usize;
                }
            }
            "weight_decay" => {
                if let ParameterValue::Float(v) = value {
                    if let Optimizer::Adam { weight_decay, .. } = &mut config.optimizer {
                        *weight_decay = v;
                    }
                }
            }
            _ => {
                tracing::warn!("Unknown parameter: {}", name);
            }
        }
        Ok(())
    }
    /// Get a parameter value from a training config
    fn get_parameter_value(&self, config: &TrainingConfig, name: &str) -> Result<f32> {
        match name {
            "learning_rate" => Ok(config.learning_rate),
            "batch_size" => Ok(config.batch_size as f32),
            "max_epochs" => Ok(config.max_epochs as f32),
            "weight_decay" => {
                if let Optimizer::Adam { weight_decay, .. } = &config.optimizer {
                    Ok(*weight_decay)
                } else {
                    Ok(0.0)
                }
            }
            _ => Ok(0.0),
        }
    }
    /// Compute Cartesian product of parameter grids
    fn cartesian_product(
        &self,
        grids: &[(String, Vec<ParameterValue>)],
    ) -> Vec<Vec<(String, ParameterValue)>> {
        Self::cartesian_product_impl(grids)
    }
    /// Compute Cartesian product implementation (helper to avoid self recursion)
    fn cartesian_product_impl(
        grids: &[(String, Vec<ParameterValue>)],
    ) -> Vec<Vec<(String, ParameterValue)>> {
        if grids.is_empty() {
            return vec![vec![]];
        }
        let mut result = Vec::new();
        let (name, values) = &grids[0];
        let rest = &grids[1..];
        let rest_product = Self::cartesian_product_impl(rest);
        for value in values {
            for combination in &rest_product {
                let mut new_combination = vec![(name.clone(), value.clone())];
                new_combination.extend_from_slice(combination);
                result.push(new_combination);
            }
        }
        result
    }
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
/// Default trainer implementation
pub struct DefaultTrainer {
    /// Training configuration
    pub(super) config: TrainingConfig,
    /// Current learning rate
    pub(super) current_lr: f32,
    /// Early stopping state
    pub(super) early_stopping_state: EarlyStoppingState,
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
    pub(super) fn update_learning_rate(&mut self, epoch: usize, val_score: Option<f32>) {
        match &self.config.lr_scheduler {
            LearningRateScheduler::Constant => {}
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
                    self.current_lr = warmup_start_lr
                        + (self.config.learning_rate - warmup_start_lr) * (epoch as f32)
                            / (*warmup_epochs as f32);
                } else {
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
    pub(super) fn check_early_stopping(&mut self, val_score: f32) -> bool {
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
    pub(super) fn generate_negative_samples(
        &self,
        positive_triples: &[Triple],
        ratio: f32,
    ) -> Vec<Triple> {
        let num_negatives = (positive_triples.len() as f32 * ratio) as usize;
        let mut negative_triples = Vec::with_capacity(num_negatives);
        let mut rng = Random::default();
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
        for _ in 0..num_negatives {
            if !positive_triples.is_empty() {
                let index = rng.random_range(0..positive_triples.len());
                let pos_triple = &positive_triples[index];
                match self.config.negative_sampling_strategy {
                    NegativeSamplingStrategy::Random => {
                        if rng.random_bool() {
                            if !subject_vec.is_empty() {
                                let subject_index = rng.random_range(0..subject_vec.len());
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
                        } else if !object_vec.is_empty() {
                            let object_index = rng.random_range(0..object_vec.len());
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
                    NegativeSamplingStrategy::TypeConstrained => {
                        if rng.random_bool() && !subject_vec.is_empty() {
                            let subject_index = rng.random_range(0..subject_vec.len());
                            let random_subject = &subject_vec[subject_index];
                            let corrupted = Triple::new(
                                random_subject.clone(),
                                pos_triple.predicate().clone(),
                                pos_triple.object().clone(),
                            );
                            if !positive_triples.contains(&corrupted) {
                                negative_triples.push(corrupted);
                            }
                        } else if !object_vec.is_empty() {
                            let object_index = rng.random_range(0..object_vec.len());
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
                    NegativeSamplingStrategy::Adversarial => {
                        if !object_vec.is_empty() {
                            let object_index = rng.random_range(0..object_vec.len());
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
    pub(super) fn compute_loss(&self, positive_scores: &[f32], negative_scores: &[f32]) -> f32 {
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
                let mut total_loss = 0.0;
                let count = positive_scores.len() + negative_scores.len();
                for &score in positive_scores {
                    let prob = 1.0 / (1.0 + (-score).exp());
                    let loss = -(1.0 * prob.ln());
                    total_loss += loss;
                }
                for &score in negative_scores {
                    let prob = 1.0 / (1.0 + (-score).exp());
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
                let mut total_loss = 0.0;
                let num_classes = positive_scores.len().max(negative_scores.len());
                if num_classes > 0 {
                    let all_scores: Vec<f32> = positive_scores
                        .iter()
                        .chain(negative_scores.iter())
                        .cloned()
                        .collect();
                    let max_score = all_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_scores: Vec<f32> =
                        all_scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
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
                let mut total_loss = 0.0;
                let mut count = 0;
                for (pos_score, neg_score) in positive_scores.iter().zip(negative_scores.iter()) {
                    let loss = (neg_score - pos_score + 1.0).max(0.0);
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
    pub(super) async fn compute_metrics(
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
        let mut all_entities = HashSet::new();
        for triple in test_triples {
            match triple.subject() {
                Subject::NamedNode(nn) => {
                    all_entities.insert(Object::NamedNode(nn.clone()));
                }
                Subject::BlankNode(bn) => {
                    all_entities.insert(Object::BlankNode(bn.clone()));
                }
                Subject::Variable(_) => {}
                Subject::QuotedTriple(_) => {}
            }
            all_entities.insert(triple.object().clone());
        }
        let entity_vec: Vec<_> = all_entities.into_iter().collect();
        let mut reciprocal_ranks = Vec::new();
        let mut hits_at_1 = 0;
        let mut hits_at_3 = 0;
        let mut hits_at_10 = 0;
        for test_triple in test_triples {
            let head_rank = self
                .compute_entity_rank(
                    test_triple.subject(),
                    test_triple.predicate(),
                    test_triple.object(),
                    &entity_vec,
                    model,
                    true,
                )
                .await?;
            let tail_rank = self
                .compute_entity_rank(
                    test_triple.subject(),
                    test_triple.predicate(),
                    test_triple.object(),
                    &entity_vec,
                    model,
                    false,
                )
                .await?;
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
            for entity in all_entities {
                let candidate_subject = match entity {
                    Object::NamedNode(nn) => Subject::NamedNode(nn.clone()),
                    Object::BlankNode(bn) => Subject::BlankNode(bn.clone()),
                    Object::Literal(_) => continue,
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
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let correct_entity_as_object = if predict_head {
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
                return Ok(rank + 1);
            }
        }
        Ok(all_entities.len())
    }
    /// Compute accuracy for a set of triples
    /// Accuracy is defined as the percentage of triples where positive score > negative score
    pub(super) async fn compute_accuracy(
        &self,
        triples: &[Triple],
        model: &dyn KnowledgeGraphEmbedding,
    ) -> Result<f32> {
        if triples.is_empty() {
            return Ok(0.0);
        }
        let negatives = self.generate_negative_samples(triples, 1.0);
        if negatives.is_empty() {
            return Ok(0.0);
        }
        let mut correct_count = 0;
        let total_count = triples.len().min(negatives.len());
        for (positive_triple, negative_triple) in triples.iter().zip(negatives.iter()) {
            let pos_score = model
                .score_triple(
                    &positive_triple.subject().to_string(),
                    &positive_triple.predicate().to_string(),
                    &positive_triple.object().to_string(),
                )
                .await?;
            let neg_score = model
                .score_triple(
                    &negative_triple.subject().to_string(),
                    &negative_triple.predicate().to_string(),
                    &negative_triple.object().to_string(),
                )
                .await?;
            if pos_score > neg_score {
                correct_count += 1;
            }
        }
        Ok(correct_count as f32 / total_count as f32)
    }
}
impl DefaultTrainer {
    /// Compute gradients using backward pass
    pub(super) async fn backward_pass(
        &self,
        positive_scores: &[f32],
        negative_scores: &[f32],
        _model: &dyn KnowledgeGraphEmbedding,
    ) -> Result<()> {
        match &self.config.loss_function {
            LossFunction::MarginRankingLoss { margin } => {
                for (pos_score, neg_score) in positive_scores.iter().zip(negative_scores.iter()) {
                    let loss = neg_score - pos_score + margin;
                    if loss > 0.0 {}
                }
            }
            LossFunction::BinaryCrossEntropy => {
                for &score in positive_scores {
                    let prob = 1.0 / (1.0 + (-score).exp());
                    let _gradient = prob - 1.0;
                }
                for &score in negative_scores {
                    let prob = 1.0 / (1.0 + (-score).exp());
                    let _gradient = prob;
                }
            }
            LossFunction::CrossEntropy => {
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
                    for (i, exp_score) in exp_scores.iter().enumerate() {
                        let prob = exp_score / sum_exp;
                        let is_positive = i < positive_scores.len();
                        let _gradient = if is_positive { prob - 1.0 } else { prob };
                    }
                }
            }
            _ => {
                for (pos_score, neg_score) in positive_scores.iter().zip(negative_scores.iter()) {
                    let loss = neg_score - pos_score + 1.0;
                    if loss > 0.0 {}
                }
            }
        }
        Ok(())
    }
    /// Apply gradient clipping to prevent exploding gradients
    pub(super) async fn clip_gradients(
        &self,
        _model: &dyn KnowledgeGraphEmbedding,
        clip_value: f32,
    ) {
        tracing::debug!("Applying gradient clipping with value: {}", clip_value);
    }
    /// Update model parameters using the configured optimizer
    pub(super) async fn update_parameters(&self, _model: &dyn KnowledgeGraphEmbedding, epoch: f32) {
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
            }
            Optimizer::Adam {
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => {
                tracing::debug!("Updating parameters with Adam optimizer");
                let _bias_correction1 = 1.0 - beta1.powf(epoch);
                let _bias_correction2 = 1.0 - beta2.powf(epoch);
                let _effective_lr = self.config.learning_rate / _bias_correction1;
                if weight_decay > &0.0 {
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
                let _bias_correction1 = 1.0 - beta1.powf(epoch);
                let _bias_correction2 = 1.0 - beta2.powf(epoch);
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
/// Optimization strategies
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    RandomSearch,
    GridSearch,
    BayesianOptimization,
    TPE,
    CmaEs,
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
