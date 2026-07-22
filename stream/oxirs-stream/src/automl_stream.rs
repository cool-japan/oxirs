//! # AutoML for Stream Processing
//!
//! This module provides automated machine learning capabilities for streaming data,
//! including automatic algorithm selection, hyperparameter optimization, and model
//! ensembling with minimal manual intervention.
//!
//! ## Features
//! - Automatic algorithm selection from a pool of candidates
//! - Hyperparameter optimization using Bayesian optimization
//! - Adaptive model selection based on data drift
//! - Ensemble methods for improved robustness
//! - Online performance tracking and model swapping
//! - Meta-learning for quick adaptation to new tasks
//!
//! ## Example Usage
//! ```rust,ignore
//! use oxirs_stream::automl_stream::{AutoML, AutoMLConfig, TaskType};
//!
//! let config = AutoMLConfig {
//!     task_type: TaskType::Classification,
//!     max_training_time_secs: 300,
//!     ..Default::default()
//! };
//!
//! let mut automl = AutoML::new(config)?;
//! automl.fit(&training_data).await?;
//! let prediction = automl.predict(&features).await?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{Random, RngExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::info;

/// Numerically-stable logistic sigmoid.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Machine learning task type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Binary or multi-class classification
    Classification,
    /// Regression (continuous values)
    Regression,
    /// Time series forecasting
    TimeSeries,
    /// Anomaly detection
    AnomalyDetection,
    /// Clustering
    Clustering,
}

/// Algorithm candidates for AutoML
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Algorithm {
    /// Linear regression
    LinearRegression,
    /// Logistic regression
    LogisticRegression,
    /// Decision tree
    DecisionTree,
    /// Random forest
    RandomForest,
    /// Gradient boosting
    GradientBoosting,
    /// Neural network
    NeuralNetwork,
    /// K-Nearest Neighbors
    KNN,
    /// Support Vector Machine
    SVM,
    /// Naive Bayes
    NaiveBayes,
    /// Online learning (SGD)
    OnlineSGD,
    /// ARIMA for time series
    ARIMA,
    /// Isolation Forest for anomaly detection
    IsolationForest,
    /// K-Means for clustering
    KMeans,
}

impl Algorithm {
    /// Get compatible algorithms for a task type
    pub fn for_task(task: TaskType) -> Vec<Algorithm> {
        match task {
            TaskType::Classification => vec![
                Algorithm::LogisticRegression,
                Algorithm::DecisionTree,
                Algorithm::RandomForest,
                Algorithm::GradientBoosting,
                Algorithm::NeuralNetwork,
                Algorithm::KNN,
                Algorithm::NaiveBayes,
            ],
            TaskType::Regression => vec![
                Algorithm::LinearRegression,
                Algorithm::DecisionTree,
                Algorithm::RandomForest,
                Algorithm::GradientBoosting,
                Algorithm::NeuralNetwork,
                Algorithm::KNN,
                Algorithm::SVM,
            ],
            TaskType::TimeSeries => vec![
                Algorithm::ARIMA,
                Algorithm::LinearRegression,
                Algorithm::NeuralNetwork,
                Algorithm::GradientBoosting,
            ],
            TaskType::AnomalyDetection => vec![
                Algorithm::IsolationForest,
                Algorithm::OnlineSGD,
                Algorithm::NeuralNetwork,
            ],
            TaskType::Clustering => vec![Algorithm::KMeans],
        }
    }
}

/// Hyperparameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParameters {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of estimators (trees, epochs, etc.)
    pub n_estimators: usize,
    /// Maximum depth (for tree-based models)
    pub max_depth: Option<usize>,
    /// Regularization strength
    pub regularization: f64,
    /// Number of neighbors (for KNN)
    pub n_neighbors: usize,
    /// Batch size (for neural networks)
    pub batch_size: usize,
    /// Random seed
    pub random_seed: u64,
}

impl Default for HyperParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            n_estimators: 100,
            max_depth: Some(5),
            regularization: 0.1,
            n_neighbors: 5,
            batch_size: 32,
            random_seed: 42,
        }
    }
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Algorithm used
    pub algorithm: Algorithm,
    /// Hyperparameters
    pub hyperparameters: HyperParameters,
    /// Accuracy (for classification)
    pub accuracy: Option<f64>,
    /// Precision
    pub precision: Option<f64>,
    /// Recall
    pub recall: Option<f64>,
    /// F1 score
    pub f1_score: Option<f64>,
    /// Mean squared error (for regression)
    pub mse: Option<f64>,
    /// R² score
    pub r_squared: Option<f64>,
    /// Training time (seconds)
    pub training_time_secs: f64,
    /// Inference time (milliseconds)
    pub inference_time_ms: f64,
    /// Model complexity score
    pub complexity_score: f64,
    /// Cross-validation score
    pub cv_score: f64,
}

impl ModelPerformance {
    /// Get overall score for model selection
    pub fn overall_score(&self) -> f64 {
        // Weighted combination of metrics
        let perf_score = self.cv_score;
        let time_penalty = (-self.training_time_secs / 60.0).exp(); // Penalize long training
        let complexity_penalty = (-self.complexity_score / 100.0).exp(); // Penalize complexity

        perf_score * time_penalty * complexity_penalty
    }
}

/// AutoML configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLConfig {
    /// Task type
    pub task_type: TaskType,
    /// Maximum training time (seconds) for AutoML search
    pub max_training_time_secs: u64,
    /// Number of hyperparameter optimization trials
    pub n_trials: usize,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Enable ensemble methods
    pub enable_ensemble: bool,
    /// Enable meta-learning
    pub enable_meta_learning: bool,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Metric to optimize
    pub optimization_metric: String,
    /// Enable automatic feature engineering
    pub auto_feature_engineering: bool,
    /// Maximum number of models to keep in ensemble
    pub max_ensemble_size: usize,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            task_type: TaskType::Classification,
            max_training_time_secs: 600,
            n_trials: 50,
            cv_folds: 5,
            enable_ensemble: true,
            enable_meta_learning: false,
            early_stopping_patience: 10,
            optimization_metric: "cv_score".to_string(),
            auto_feature_engineering: true,
            max_ensemble_size: 5,
        }
    }
}

/// Trained model representation
#[derive(Debug, Clone)]
pub struct TrainedModel {
    /// Algorithm used
    pub algorithm: Algorithm,
    /// Hyperparameters
    pub hyperparameters: HyperParameters,
    /// Model weights/parameters
    pub parameters: ModelParameters,
    /// Performance metrics
    pub performance: ModelPerformance,
}

/// Model parameters (simplified)
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Weight vector
    pub weights: Vec<f64>,
    /// Bias term
    pub bias: f64,
    /// Additional parameters (algorithm-specific)
    pub extra: HashMap<String, Vec<f64>>,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            extra: HashMap::new(),
        }
    }
}

/// AutoML statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLStats {
    /// Total trials executed
    pub total_trials: u64,
    /// Best model score
    pub best_score: f64,
    /// Total training time (seconds)
    pub total_training_time_secs: f64,
    /// Number of models in ensemble
    pub ensemble_size: usize,
    /// Current best algorithm
    pub best_algorithm: Option<Algorithm>,
    /// Number of predictions made
    pub predictions_count: u64,
    /// Average prediction time (ms)
    pub avg_prediction_time_ms: f64,
}

impl Default for AutoMLStats {
    fn default() -> Self {
        Self {
            total_trials: 0,
            best_score: 0.0,
            total_training_time_secs: 0.0,
            ensemble_size: 0,
            best_algorithm: None,
            predictions_count: 0,
            avg_prediction_time_ms: 0.0,
        }
    }
}

/// Main AutoML engine
pub struct AutoML {
    config: AutoMLConfig,
    /// Best model found
    best_model: Arc<RwLock<Option<TrainedModel>>>,
    /// Ensemble of models
    ensemble: Arc<RwLock<Vec<TrainedModel>>>,
    /// Trial history
    trial_history: Arc<RwLock<Vec<ModelPerformance>>>,
    /// Statistics
    stats: Arc<RwLock<AutoMLStats>>,
    /// Random number generator
    #[allow(clippy::arc_with_non_send_sync)]
    rng: Arc<Mutex<Random>>,
}

impl AutoML {
    /// Create a new AutoML instance
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(config: AutoMLConfig) -> Result<Self> {
        Ok(Self {
            config,
            best_model: Arc::new(RwLock::new(None)),
            ensemble: Arc::new(RwLock::new(Vec::new())),
            trial_history: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(AutoMLStats::default())),
            rng: Arc::new(Mutex::new(Random::default())),
        })
    }

    /// Fit AutoML on training data
    pub async fn fit(&mut self, features: &Array2<f64>, labels: &Array1<f64>) -> Result<()> {
        info!(
            "Starting AutoML training with task {:?}, {} samples, {} features",
            self.config.task_type,
            features.shape()[0],
            features.shape()[1]
        );

        let start_time = std::time::Instant::now();
        // Only search over algorithms we actually implement a learner for. This
        // avoids fabricating scores for unimplemented algorithms: the search
        // never claims to have trained something it cannot train.
        let candidate_algorithms: Vec<Algorithm> = Algorithm::for_task(self.config.task_type)
            .into_iter()
            .filter(|algorithm| Self::is_supported(*algorithm))
            .collect();
        if candidate_algorithms.is_empty() {
            return Err(anyhow!(
                "AutoML has no implemented learner for task {:?}; supported algorithms are \
                 LinearRegression, LogisticRegression and OnlineSGD",
                self.config.task_type
            ));
        }

        let mut best_overall_score = f64::NEG_INFINITY;
        let mut trials_without_improvement = 0;

        for trial in 0..self.config.n_trials {
            // Check time budget
            if start_time.elapsed().as_secs() >= self.config.max_training_time_secs {
                info!("Time budget exhausted, stopping AutoML");
                break;
            }

            // Select algorithm
            let algorithm = {
                let mut rng = self.rng.lock().await;
                let idx = rng.random_range(0..candidate_algorithms.len());
                candidate_algorithms[idx]
            };

            // Generate hyperparameters
            let hyperparams = self.generate_hyperparameters(algorithm).await?;

            // Train and evaluate model
            let performance = self
                .train_and_evaluate(algorithm, &hyperparams, features, labels)
                .await?;

            // Record trial
            self.trial_history.write().await.push(performance.clone());

            let overall_score = performance.overall_score();

            info!(
                "Trial {}: {:?} - CV score: {:.4}, Overall score: {:.4}",
                trial, algorithm, performance.cv_score, overall_score
            );

            // Update best model
            if overall_score > best_overall_score {
                best_overall_score = overall_score;
                trials_without_improvement = 0;

                let model = TrainedModel {
                    algorithm,
                    hyperparameters: hyperparams.clone(),
                    parameters: self
                        .train_final_model(algorithm, &hyperparams, features, labels)
                        .await?,
                    performance: performance.clone(),
                };

                *self.best_model.write().await = Some(model.clone());

                // Update ensemble if enabled
                if self.config.enable_ensemble {
                    self.update_ensemble(model).await?;
                }

                // Update stats
                let mut stats = self.stats.write().await;
                stats.best_score = best_overall_score;
                stats.best_algorithm = Some(algorithm);
            } else {
                trials_without_improvement += 1;
            }

            // Early stopping
            if trials_without_improvement >= self.config.early_stopping_patience {
                info!(
                    "Early stopping triggered after {} trials without improvement",
                    trials_without_improvement
                );
                break;
            }

            // Update stats
            let mut stats = self.stats.write().await;
            stats.total_trials = trial as u64 + 1;
        }

        // Final stats update
        let mut stats = self.stats.write().await;
        stats.total_training_time_secs = start_time.elapsed().as_secs_f64();
        stats.ensemble_size = self.ensemble.read().await.len();

        info!(
            "AutoML training complete: {} trials, best score: {:.4}, algorithm: {:?}",
            stats.total_trials, stats.best_score, stats.best_algorithm
        );

        Ok(())
    }

    /// Generate hyperparameters for an algorithm
    async fn generate_hyperparameters(&self, algorithm: Algorithm) -> Result<HyperParameters> {
        let mut rng = self.rng.lock().await;

        // Use meta-learning to initialize if enabled
        let _base = if self.config.enable_meta_learning {
            self.get_meta_learning_initialization(algorithm).await
        } else {
            HyperParameters::default()
        };

        // Apply random perturbations
        Ok(HyperParameters {
            learning_rate: rng.random_range(0.0001..0.1),
            n_estimators: rng.random_range(10..500),
            max_depth: Some(rng.random_range(3..20)),
            regularization: rng.random_range(0.0..1.0),
            n_neighbors: rng.random_range(3..20),
            batch_size: rng.random_range(16..256),
            random_seed: rng.random::<u64>(),
        })
    }

    /// Get meta-learning initialization (placeholder)
    async fn get_meta_learning_initialization(&self, _algorithm: Algorithm) -> HyperParameters {
        // In production, this would use historical performance data
        HyperParameters::default()
    }

    /// Train and evaluate a model with cross-validation
    async fn train_and_evaluate(
        &self,
        algorithm: Algorithm,
        hyperparams: &HyperParameters,
        features: &Array2<f64>,
        labels: &Array1<f64>,
    ) -> Result<ModelPerformance> {
        let start_time = std::time::Instant::now();

        // Perform cross-validation
        let cv_scores = self
            .cross_validate(algorithm, hyperparams, features, labels)
            .await?;
        let cv_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;

        // Compute additional metrics
        let (accuracy, precision, recall, f1, mse, r_squared) = self
            .compute_metrics(algorithm, hyperparams, features, labels)
            .await?;

        let training_time = start_time.elapsed().as_secs_f64();

        // Estimate complexity (simplified)
        let complexity_score = match algorithm {
            Algorithm::LinearRegression | Algorithm::LogisticRegression => 10.0,
            Algorithm::DecisionTree => 30.0,
            Algorithm::RandomForest | Algorithm::GradientBoosting => 60.0,
            Algorithm::NeuralNetwork => 80.0,
            _ => 40.0,
        };

        Ok(ModelPerformance {
            algorithm,
            hyperparameters: hyperparams.clone(),
            accuracy,
            precision,
            recall,
            f1_score: f1,
            mse,
            r_squared,
            training_time_secs: training_time,
            inference_time_ms: 1.0, // Placeholder
            complexity_score,
            cv_score,
        })
    }

    /// Perform k-fold cross-validation
    async fn cross_validate(
        &self,
        algorithm: Algorithm,
        hyperparams: &HyperParameters,
        features: &Array2<f64>,
        labels: &Array1<f64>,
    ) -> Result<Vec<f64>> {
        let n_samples = features.shape()[0];
        let fold_size = n_samples / self.config.cv_folds;

        let mut scores = Vec::new();

        for fold in 0..self.config.cv_folds {
            let val_start = fold * fold_size;
            let val_end = ((fold + 1) * fold_size).min(n_samples);

            // Simple train/val split (in production, use proper indexing)
            let score = self
                .evaluate_fold(algorithm, hyperparams, features, labels, val_start, val_end)
                .await?;
            scores.push(score);
        }

        Ok(scores)
    }

    /// Whether AutoML has a real learner implemented for this algorithm.
    ///
    /// The model representation used throughout this module is a linear model
    /// (weights + bias, with an optional sigmoid activation). Only the
    /// algorithms that map onto that representation are supported; everything
    /// else is rejected rather than being faked.
    fn is_supported(algorithm: Algorithm) -> bool {
        matches!(
            algorithm,
            Algorithm::LinearRegression | Algorithm::LogisticRegression | Algorithm::OnlineSGD
        )
    }

    /// Whether the task uses a sigmoid (classification-style) activation.
    fn is_classification_task(task: TaskType) -> bool {
        matches!(task, TaskType::Classification | TaskType::AnomalyDetection)
    }

    /// Fit a linear (or logistic) model by gradient descent.
    ///
    /// Features are standardized internally for numerical stability regardless
    /// of scale, and the learned parameters are converted back to raw feature
    /// space so callers can apply them directly to un-normalized inputs. When
    /// `skip` is `Some((start, end))` the rows in that half-open range are held
    /// out (used for cross-validation folds).
    fn fit_linear(
        features: &Array2<f64>,
        labels: &Array1<f64>,
        hyperparams: &HyperParameters,
        is_classification: bool,
        skip: Option<(usize, usize)>,
    ) -> (Vec<f64>, f64) {
        let n_samples = features.shape()[0];
        let n_features = features.shape()[1];

        // Compute mean/std per feature over the included rows for standardization.
        let mut mean = vec![0.0f64; n_features];
        let mut included: f64 = 0.0;
        for i in 0..n_samples {
            if let Some((start, end)) = skip {
                if i >= start && i < end {
                    continue;
                }
            }
            for j in 0..n_features {
                mean[j] += features[[i, j]];
            }
            included += 1.0;
        }
        if included == 0.0 {
            return (vec![0.0; n_features], 0.0);
        }
        for m in mean.iter_mut() {
            *m /= included;
        }
        let mut std = vec![0.0f64; n_features];
        for i in 0..n_samples {
            if let Some((start, end)) = skip {
                if i >= start && i < end {
                    continue;
                }
            }
            for j in 0..n_features {
                let diff = features[[i, j]] - mean[j];
                std[j] += diff * diff;
            }
        }
        for s in std.iter_mut() {
            *s = (*s / included).sqrt();
            if *s < 1e-8 {
                *s = 1.0;
            }
        }

        let lr = hyperparams.learning_rate.max(1e-6);
        let l2 = hyperparams.regularization.max(0.0);
        let epochs = hyperparams.n_estimators.clamp(10, 300);

        // Gradient descent in standardized feature space.
        let mut std_weights = vec![0.0f64; n_features];
        let mut std_bias = 0.0f64;
        for _ in 0..epochs {
            let mut grad_w = vec![0.0f64; n_features];
            let mut grad_b = 0.0f64;
            for i in 0..n_samples {
                if let Some((start, end)) = skip {
                    if i >= start && i < end {
                        continue;
                    }
                }
                let mut z = std_bias;
                for j in 0..n_features {
                    let x = (features[[i, j]] - mean[j]) / std[j];
                    z += std_weights[j] * x;
                }
                let pred = if is_classification { sigmoid(z) } else { z };
                let error = pred - labels[i];
                for j in 0..n_features {
                    let x = (features[[i, j]] - mean[j]) / std[j];
                    grad_w[j] += error * x;
                }
                grad_b += error;
            }
            let inv = 1.0 / included;
            for j in 0..n_features {
                std_weights[j] -= lr * (grad_w[j] * inv + l2 * std_weights[j]);
            }
            std_bias -= lr * grad_b * inv;
        }

        // Convert standardized-space parameters back to raw feature space:
        //   z = b' + Σ w'_j (x_j - mean_j)/std_j
        //     = (b' - Σ w'_j mean_j/std_j) + Σ (w'_j/std_j) x_j
        let mut weights = vec![0.0f64; n_features];
        let mut bias = std_bias;
        for j in 0..n_features {
            weights[j] = std_weights[j] / std[j];
            bias -= std_weights[j] * mean[j] / std[j];
        }

        (weights, bias)
    }

    /// Score a single row using raw-space parameters.
    fn linear_score(weights: &[f64], bias: f64, features: &Array2<f64>, row: usize) -> f64 {
        let n_features = features.shape()[1].min(weights.len());
        let mut z = bias;
        for j in 0..n_features {
            z += weights[j] * features[[row, j]];
        }
        z
    }

    /// Evaluate a single fold by training on the complement and scoring the
    /// held-out validation rows.
    async fn evaluate_fold(
        &self,
        algorithm: Algorithm,
        hyperparams: &HyperParameters,
        features: &Array2<f64>,
        labels: &Array1<f64>,
        val_start: usize,
        val_end: usize,
    ) -> Result<f64> {
        if !Self::is_supported(algorithm) {
            return Err(anyhow!(
                "AutoML training for algorithm {:?} is not implemented",
                algorithm
            ));
        }

        let is_classification = Self::is_classification_task(self.config.task_type);
        let (weights, bias) = Self::fit_linear(
            features,
            labels,
            hyperparams,
            is_classification,
            Some((val_start, val_end)),
        );

        if val_end <= val_start {
            return Ok(0.0);
        }

        if is_classification {
            let mut correct = 0usize;
            let mut total = 0usize;
            for i in val_start..val_end {
                let z = Self::linear_score(&weights, bias, features, i);
                let predicted = sigmoid(z) >= 0.5;
                let actual = labels[i] >= 0.5;
                if predicted == actual {
                    correct += 1;
                }
                total += 1;
            }
            Ok(if total > 0 {
                correct as f64 / total as f64
            } else {
                0.0
            })
        } else {
            // Coefficient of determination (R²) on the validation fold.
            let mut sum = 0.0;
            let mut count = 0.0;
            for i in val_start..val_end {
                sum += labels[i];
                count += 1.0;
            }
            if count == 0.0 {
                return Ok(0.0);
            }
            let mean = sum / count;
            let mut ss_res = 0.0;
            let mut ss_tot = 0.0;
            for i in val_start..val_end {
                let z = Self::linear_score(&weights, bias, features, i);
                ss_res += (labels[i] - z).powi(2);
                ss_tot += (labels[i] - mean).powi(2);
            }
            let r2 = if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            };
            // Clamp to [0, 1] for use as a selection score.
            Ok(r2.clamp(0.0, 1.0))
        }
    }

    /// Compute real performance metrics from actual predictions vs. labels.
    async fn compute_metrics(
        &self,
        algorithm: Algorithm,
        hyperparams: &HyperParameters,
        features: &Array2<f64>,
        labels: &Array1<f64>,
    ) -> Result<(
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
    )> {
        if !Self::is_supported(algorithm) {
            return Err(anyhow!(
                "AutoML training for algorithm {:?} is not implemented",
                algorithm
            ));
        }

        let is_classification = Self::is_classification_task(self.config.task_type);
        let (weights, bias) =
            Self::fit_linear(features, labels, hyperparams, is_classification, None);
        let n_samples = features.shape()[0];

        match self.config.task_type {
            TaskType::Classification | TaskType::AnomalyDetection => {
                let (mut tp, mut fp, mut fn_, mut tn) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for i in 0..n_samples {
                    let z = Self::linear_score(&weights, bias, features, i);
                    let predicted = sigmoid(z) >= 0.5;
                    let actual = labels[i] >= 0.5;
                    match (predicted, actual) {
                        (true, true) => tp += 1.0,
                        (true, false) => fp += 1.0,
                        (false, true) => fn_ += 1.0,
                        (false, false) => tn += 1.0,
                    }
                }
                let total = tp + fp + fn_ + tn;
                let accuracy = if total > 0.0 { (tp + tn) / total } else { 0.0 };
                let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
                let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };
                Ok((
                    Some(accuracy),
                    Some(precision),
                    Some(recall),
                    Some(f1),
                    None,
                    None,
                ))
            }
            TaskType::Regression | TaskType::TimeSeries => {
                let mut ss_res = 0.0;
                let mut sum = 0.0;
                for i in 0..n_samples {
                    sum += labels[i];
                }
                let mean = if n_samples > 0 {
                    sum / n_samples as f64
                } else {
                    0.0
                };
                let mut ss_tot = 0.0;
                for i in 0..n_samples {
                    let z = Self::linear_score(&weights, bias, features, i);
                    ss_res += (labels[i] - z).powi(2);
                    ss_tot += (labels[i] - mean).powi(2);
                }
                let mse = if n_samples > 0 {
                    ss_res / n_samples as f64
                } else {
                    0.0
                };
                let r_squared = if ss_tot > 0.0 {
                    1.0 - ss_res / ss_tot
                } else {
                    0.0
                };
                Ok((None, None, None, None, Some(mse), Some(r_squared)))
            }
            _ => Ok((None, None, None, None, None, None)),
        }
    }

    /// Train the final model with the given hyperparameters on all data.
    async fn train_final_model(
        &self,
        algorithm: Algorithm,
        hyperparams: &HyperParameters,
        features: &Array2<f64>,
        labels: &Array1<f64>,
    ) -> Result<ModelParameters> {
        if !Self::is_supported(algorithm) {
            return Err(anyhow!(
                "AutoML training for algorithm {:?} is not implemented",
                algorithm
            ));
        }

        let is_classification = Self::is_classification_task(self.config.task_type);
        let (weights, bias) =
            Self::fit_linear(features, labels, hyperparams, is_classification, None);

        Ok(ModelParameters {
            weights,
            bias,
            extra: HashMap::new(),
        })
    }

    /// Update ensemble with new model
    async fn update_ensemble(&self, model: TrainedModel) -> Result<()> {
        let mut ensemble = self.ensemble.write().await;

        // Add model to ensemble
        ensemble.push(model);

        // Keep only top models
        if ensemble.len() > self.config.max_ensemble_size {
            ensemble.sort_by(|a, b| {
                b.performance
                    .overall_score()
                    .partial_cmp(&a.performance.overall_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ensemble.truncate(self.config.max_ensemble_size);
        }

        Ok(())
    }

    /// Make prediction using the best model or ensemble
    pub async fn predict(&self, features: &Array1<f64>) -> Result<f64> {
        let start_time = std::time::Instant::now();

        let prediction = if self.config.enable_ensemble {
            self.ensemble_predict(features).await?
        } else {
            self.single_model_predict(features).await?
        };

        // Update stats
        let mut stats = self.stats.write().await;
        stats.predictions_count += 1;
        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        stats.avg_prediction_time_ms =
            (stats.avg_prediction_time_ms * (stats.predictions_count - 1) as f64 + elapsed_ms)
                / stats.predictions_count as f64;

        Ok(prediction)
    }

    /// Predict using single best model
    async fn single_model_predict(&self, features: &Array1<f64>) -> Result<f64> {
        let model = self.best_model.read().await;

        match &*model {
            Some(m) => {
                // Simple linear prediction
                let mut pred = m.parameters.bias;
                for (i, &weight) in m.parameters.weights.iter().enumerate() {
                    if i < features.len() {
                        pred += weight * features[i];
                    }
                }

                // Apply activation for classification
                if matches!(self.config.task_type, TaskType::Classification) {
                    pred = 1.0 / (1.0 + (-pred).exp()); // Sigmoid
                }

                Ok(pred)
            }
            None => Err(anyhow!("No trained model available")),
        }
    }

    /// Predict using ensemble (averaging)
    async fn ensemble_predict(&self, features: &Array1<f64>) -> Result<f64> {
        let ensemble = self.ensemble.read().await;

        if ensemble.is_empty() {
            return self.single_model_predict(features).await;
        }

        let mut predictions = Vec::new();
        let mut weights = Vec::new();

        for model in ensemble.iter() {
            let mut pred = model.parameters.bias;
            for (i, &weight) in model.parameters.weights.iter().enumerate() {
                if i < features.len() {
                    pred += weight * features[i];
                }
            }

            if matches!(self.config.task_type, TaskType::Classification) {
                pred = 1.0 / (1.0 + (-pred).exp());
            }

            predictions.push(pred);
            weights.push(model.performance.overall_score());
        }

        // Weighted average
        let total_weight: f64 = weights.iter().sum();
        let weighted_pred = predictions
            .iter()
            .zip(&weights)
            .map(|(p, w)| p * w)
            .sum::<f64>()
            / total_weight;

        Ok(weighted_pred)
    }

    /// Get AutoML statistics
    pub async fn get_stats(&self) -> AutoMLStats {
        self.stats.read().await.clone()
    }

    /// Get trial history
    pub async fn get_trial_history(&self) -> Vec<ModelPerformance> {
        self.trial_history.read().await.clone()
    }

    /// Get best model information
    pub async fn get_best_model_info(
        &self,
    ) -> Option<(Algorithm, HyperParameters, ModelPerformance)> {
        let model = self.best_model.read().await;
        model.as_ref().map(|m| {
            (
                m.algorithm,
                m.hyperparameters.clone(),
                m.performance.clone(),
            )
        })
    }

    /// Export best model for deployment
    pub async fn export_model(&self) -> Result<String> {
        let model = self.best_model.read().await;

        match &*model {
            Some(m) => {
                let export = serde_json::json!({
                    "algorithm": format!("{:?}", m.algorithm),
                    "hyperparameters": m.hyperparameters,
                    "parameters": {
                        "weights": m.parameters.weights,
                        "bias": m.parameters.bias,
                    },
                    "performance": m.performance,
                });
                Ok(serde_json::to_string_pretty(&export)?)
            }
            None => Err(anyhow!("No model to export")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_for_task() {
        let classifiers = Algorithm::for_task(TaskType::Classification);
        assert!(!classifiers.is_empty());
        assert!(classifiers.contains(&Algorithm::LogisticRegression));

        let regressors = Algorithm::for_task(TaskType::Regression);
        assert!(regressors.contains(&Algorithm::LinearRegression));

        let ts_algorithms = Algorithm::for_task(TaskType::TimeSeries);
        assert!(ts_algorithms.contains(&Algorithm::ARIMA));
    }

    #[test]
    fn test_hyperparameters_default() {
        let params = HyperParameters::default();
        assert_eq!(params.learning_rate, 0.01);
        assert_eq!(params.n_estimators, 100);
        assert_eq!(params.max_depth, Some(5));
    }

    #[test]
    fn test_model_performance_overall_score() {
        let perf = ModelPerformance {
            algorithm: Algorithm::LinearRegression,
            hyperparameters: HyperParameters::default(),
            accuracy: None,
            precision: None,
            recall: None,
            f1_score: None,
            mse: Some(0.5),
            r_squared: Some(0.9),
            training_time_secs: 10.0,
            inference_time_ms: 1.0,
            complexity_score: 20.0,
            cv_score: 0.85,
        };

        let score = perf.overall_score();
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[tokio::test]
    async fn test_automl_creation() {
        let config = AutoMLConfig::default();
        let automl = AutoML::new(config);
        assert!(automl.is_ok());
    }

    #[tokio::test]
    async fn test_automl_generate_hyperparameters() {
        let config = AutoMLConfig::default();
        let automl = AutoML::new(config).unwrap();

        let params = automl
            .generate_hyperparameters(Algorithm::LinearRegression)
            .await;
        assert!(params.is_ok());

        let p = params.unwrap();
        assert!(p.learning_rate > 0.0);
        assert!(p.n_estimators > 0);
    }

    #[tokio::test]
    async fn test_automl_fit_small_dataset() {
        let config = AutoMLConfig {
            task_type: TaskType::Regression,
            max_training_time_secs: 5,
            n_trials: 3,
            cv_folds: 2,
            enable_ensemble: false,
            ..Default::default()
        };

        let mut automl = AutoML::new(config).unwrap();

        // Small synthetic dataset
        let features = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
                9.0, 10.0, 10.0, 11.0,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]);

        let result = automl.fit(&features, &labels).await;
        assert!(result.is_ok());

        let stats = automl.get_stats().await;
        assert!(stats.total_trials > 0);
        assert!(stats.total_trials <= 3);
    }

    #[tokio::test]
    async fn test_automl_prediction() {
        let config = AutoMLConfig {
            task_type: TaskType::Regression,
            max_training_time_secs: 5,
            n_trials: 2,
            ..Default::default()
        };

        let mut automl = AutoML::new(config).unwrap();

        let features = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
                9.0, 10.0, 10.0, 11.0,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]);

        automl.fit(&features, &labels).await.unwrap();

        let test_features = Array1::from_vec(vec![5.5, 6.5]);
        let prediction = automl.predict(&test_features).await;
        assert!(prediction.is_ok());
    }

    #[tokio::test]
    async fn test_ensemble_prediction() {
        let config = AutoMLConfig {
            task_type: TaskType::Classification,
            enable_ensemble: true,
            max_ensemble_size: 3,
            n_trials: 5,
            max_training_time_secs: 10,
            ..Default::default()
        };

        let mut automl = AutoML::new(config).unwrap();

        let features =
            Array2::from_shape_vec((20, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let labels = Array1::from_vec((0..20).map(|x| (x % 2) as f64).collect());

        automl.fit(&features, &labels).await.unwrap();

        let test_features = Array1::from_vec(vec![5.0, 10.0]);
        let prediction = automl.predict(&test_features).await;
        assert!(prediction.is_ok());

        let pred = prediction.unwrap();
        assert!((0.0..=1.0).contains(&pred)); // Should be probability for classification
    }

    #[tokio::test]
    async fn regression_linear_fit_is_data_dependent() {
        // y = 2x + 1 over a clean single-feature dataset.
        let config = AutoMLConfig {
            task_type: TaskType::Regression,
            max_training_time_secs: 10,
            n_trials: 25,
            cv_folds: 3,
            enable_ensemble: false,
            ..Default::default()
        };
        let mut automl = AutoML::new(config).unwrap();

        let xs: Vec<f64> = (0..30).map(|x| x as f64).collect();
        let features = Array2::from_shape_vec((30, 1), xs.clone()).unwrap();
        let labels = Array1::from_vec(xs.iter().map(|x| 2.0 * x + 1.0).collect());

        automl.fit(&features, &labels).await.unwrap();

        // Prediction must track the real linear relationship, not random noise.
        let prediction = automl.predict(&Array1::from_vec(vec![10.0])).await.unwrap();
        assert!(
            (prediction - 21.0).abs() < 3.0,
            "expected ~21 for y=2x+1 at x=10, got {prediction}"
        );

        // The best model should have a strong fit (real R²), not a fabricated score.
        let (_, _, perf) = automl.get_best_model_info().await.unwrap();
        assert!(perf.r_squared.unwrap_or(0.0) > 0.8);
    }

    #[tokio::test]
    async fn regression_unsupported_task_fails_loud() {
        let config = AutoMLConfig {
            task_type: TaskType::Clustering,
            n_trials: 2,
            max_training_time_secs: 2,
            ..Default::default()
        };
        let mut automl = AutoML::new(config).unwrap();
        let features = Array2::from_shape_vec((4, 2), (0..8).map(|x| x as f64).collect()).unwrap();
        let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

        // Clustering has no implemented learner => fit must error, not fake it.
        assert!(automl.fit(&features, &labels).await.is_err());
    }

    #[tokio::test]
    async fn test_get_best_model_info() {
        let config = AutoMLConfig {
            n_trials: 2,
            max_training_time_secs: 5,
            ..Default::default()
        };

        let mut automl = AutoML::new(config).unwrap();

        let features =
            Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let labels = Array1::from_vec((0..10).map(|x| x as f64).collect());

        automl.fit(&features, &labels).await.unwrap();

        let best_info = automl.get_best_model_info().await;
        assert!(best_info.is_some());

        let (_algorithm, _hyperparams, performance) = best_info.unwrap();
        assert!(performance.cv_score >= 0.0);
    }

    #[tokio::test]
    async fn test_export_model() {
        let config = AutoMLConfig {
            n_trials: 1,
            max_training_time_secs: 5,
            ..Default::default()
        };

        let mut automl = AutoML::new(config).unwrap();

        let features =
            Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let labels = Array1::from_vec((0..10).map(|x| x as f64).collect());

        automl.fit(&features, &labels).await.unwrap();

        let export = automl.export_model().await;
        assert!(export.is_ok());

        let json_str = export.unwrap();
        assert!(json_str.contains("algorithm"));
        assert!(json_str.contains("hyperparameters"));
    }

    #[tokio::test]
    async fn test_trial_history() {
        let config = AutoMLConfig {
            n_trials: 3,
            max_training_time_secs: 5,
            ..Default::default()
        };

        let mut automl = AutoML::new(config).unwrap();

        let features =
            Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let labels = Array1::from_vec((0..10).map(|x| x as f64).collect());

        automl.fit(&features, &labels).await.unwrap();

        let history = automl.get_trial_history().await;
        assert!(!history.is_empty());
        assert!(history.len() <= 3);
    }

    #[tokio::test]
    async fn test_early_stopping() {
        let config = AutoMLConfig {
            n_trials: 100, // Large number
            max_training_time_secs: 60,
            early_stopping_patience: 3,
            ..Default::default()
        };

        let mut automl = AutoML::new(config).unwrap();

        let features =
            Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let labels = Array1::from_vec((0..10).map(|x| x as f64).collect());

        automl.fit(&features, &labels).await.unwrap();

        let stats = automl.get_stats().await;
        // Should stop early, not run all 100 trials
        assert!(stats.total_trials < 100);
    }
}
