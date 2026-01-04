//! Machine Learning-Enhanced Query Optimizer
//!
//! This module provides ML-powered query optimization capabilities that learn from
//! historical query performance to make intelligent optimization decisions.
//!
//! ## SciRS2-Core Integration
//!
//! This module leverages SciRS2-Core for high-performance ML capabilities:
//! - **Array Operations**: `scirs2_core::ndarray_ext` for vectorized feature matrices
//! - **Random Number Generation**: `scirs2_core::random` for model initialization
//! - **SIMD Operations**: Vectorized computations for faster training
//! - **Parallel Processing**: Batch operations for improved throughput
//! - **Memory Efficiency**: Optimized memory usage for large training datasets
//! - **Statistics**: Advanced statistical operations for model evaluation

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::info;

// SciRS2-Core integration for comprehensive ML operations
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::Random;
use scirs2_core::sampling::random_normal;

use crate::ast::{Document, OperationType, Selection, SelectionSet};
use crate::optimizer::{QueryComplexity, QueryOptimizer};
use crate::performance::{OperationMetrics, PerformanceTracker};
use crate::system_monitor;

/// ML optimizer configuration
#[derive(Debug, Clone)]
pub struct MLOptimizerConfig {
    pub learning_enabled: bool,
    pub min_samples_for_learning: usize,
    pub feature_extraction_enabled: bool,
    pub prediction_threshold: f64,
    pub model_update_interval: Duration,
    pub max_training_samples: usize,
    pub performance_history_window: Duration,
    pub use_neural_network: bool,
    pub neural_network_layers: Vec<usize>,
    pub neural_learning_rate: f64,
    pub enable_reinforcement_learning: bool,
    pub enable_semantic_analysis: bool,
    pub adaptive_optimization: bool,
}

impl Default for MLOptimizerConfig {
    fn default() -> Self {
        Self {
            learning_enabled: true,
            min_samples_for_learning: 100,
            feature_extraction_enabled: true,
            prediction_threshold: 0.7,
            model_update_interval: Duration::from_secs(3600), // 1 hour
            max_training_samples: 10000,
            performance_history_window: Duration::from_secs(86400), // 24 hours
            use_neural_network: false,
            neural_network_layers: vec![64, 32, 16],
            neural_learning_rate: 0.001,
            enable_reinforcement_learning: false,
            enable_semantic_analysis: true,
            adaptive_optimization: true,
        }
    }
}

/// Query features extracted for ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    pub field_count: f64,
    pub max_depth: f64,
    pub complexity_score: f64,
    pub selection_count: f64,
    pub has_fragments: f64,
    pub has_variables: f64,
    pub operation_type: f64, // 0 = Query, 1 = Mutation, 2 = Subscription
    pub unique_field_types: f64,
    pub nested_list_count: f64,
    pub argument_count: f64,
    pub directive_count: f64,
    pub estimated_result_size: f64,
}

impl QueryFeatures {
    /// Convert features to a vector for ML algorithms
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.field_count,
            self.max_depth,
            self.complexity_score,
            self.selection_count,
            self.has_fragments,
            self.has_variables,
            self.operation_type,
            self.unique_field_types,
            self.nested_list_count,
            self.argument_count,
            self.directive_count,
            self.estimated_result_size,
        ]
    }

    /// Create features from a vector
    pub fn from_vector(vector: &[f64]) -> Result<Self> {
        if vector.len() != 12 {
            return Err(anyhow!(
                "Invalid feature vector length: expected 12, got {}",
                vector.len()
            ));
        }

        Ok(Self {
            field_count: vector[0],
            max_depth: vector[1],
            complexity_score: vector[2],
            selection_count: vector[3],
            has_fragments: vector[4],
            has_variables: vector[5],
            operation_type: vector[6],
            unique_field_types: vector[7],
            nested_list_count: vector[8],
            argument_count: vector[9],
            directive_count: vector[10],
            estimated_result_size: vector[11],
        })
    }
}

/// Training sample for the ML model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub features: QueryFeatures,
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cache_hit: bool,
    pub error_occurred: bool,
    pub timestamp: SystemTime,
}

/// Performance prediction from the ML model
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub predicted_execution_time: Duration,
    pub predicted_memory_usage: f64,
    pub cache_hit_probability: f64,
    pub error_probability: f64,
    pub confidence_score: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub estimated_improvement: f64, // Percentage improvement
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum RecommendationType {
    ReduceDepth,
    AddCaching,
    BreakIntoSubqueries,
    AddIndexes,
    OptimizeFragments,
    ParallelizeFields,
    ReduceComplexity,
}

/// Activation functions for neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    Softmax,
}

/// Advanced linear regression model for performance prediction
/// Enhanced with SciRS2-Core for vectorized matrix operations and efficient training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionModel {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub training_samples: usize,
    pub last_updated: SystemTime,
    /// L2 regularization parameter to prevent overfitting
    pub regularization: f64,
}

impl LinearRegressionModel {
    pub fn new(feature_count: usize) -> Self {
        // Enhanced initialization with SciRS2 Random for better convergence
        let mut rng = Random::seed(42);
        let weights = (0..feature_count)
            .map(|_| random_normal(&mut rng, 0.0, 0.01))
            .collect();

        Self {
            weights,
            bias: 0.0,
            training_samples: 0,
            last_updated: SystemTime::now(),
            regularization: 0.01,
        }
    }

    /// Create model with custom regularization
    pub fn with_regularization(feature_count: usize, regularization: f64) -> Self {
        let mut model = Self::new(feature_count);
        model.regularization = regularization;
        model
    }

    /// Predict execution time based on features
    pub fn predict(&self, features: &[f64]) -> f64 {
        if features.len() != self.weights.len() {
            return 1000.0; // Default prediction
        }

        let prediction = self.bias
            + features
                .iter()
                .zip(&self.weights)
                .map(|(f, w)| f * w)
                .sum::<f64>();

        prediction.max(0.0) // Ensure non-negative prediction
    }

    /// Train the model using vectorized gradient descent with SciRS2-Core array operations
    /// This implementation uses matrix operations for efficient batch processing
    pub fn train(&mut self, samples: &[TrainingSample], learning_rate: f64, iterations: usize) {
        if samples.is_empty() {
            return;
        }

        let start_time = SystemTime::now();

        // Filter valid samples
        let valid_samples: Vec<_> = samples
            .iter()
            .filter(|s| s.features.to_vector().len() == self.weights.len())
            .collect();

        if valid_samples.is_empty() {
            return;
        }

        // Convert to feature matrix using SciRS2 Array2 for vectorized operations
        let n_samples = valid_samples.len();
        let n_features = self.weights.len();

        let mut feature_matrix = Vec::with_capacity(n_samples * n_features);
        let mut targets = Vec::with_capacity(n_samples);

        for sample in &valid_samples {
            feature_matrix.extend_from_slice(&sample.features.to_vector());
            targets.push(sample.execution_time_ms);
        }

        let x_matrix = Array2::from_shape_vec((n_samples, n_features), feature_matrix)
            .unwrap_or_else(|_| Array2::zeros((n_samples, n_features)));
        let y_vector = Array1::from_vec(targets);
        let weight_array = Array1::from_vec(self.weights.clone());

        // Gradient descent with vectorized operations
        let mut weights = weight_array;
        let mut bias = self.bias;

        for _ in 0..iterations {
            // Vectorized prediction: X @ w + b
            let predictions = x_matrix.dot(&weights) + bias;

            // Compute errors: predictions - y
            let errors = &predictions - &y_vector;

            // Vectorized gradient computation with L2 regularization
            // gradient_w = (1/n) * X^T @ errors + lambda * w
            let gradient_w =
                x_matrix.t().dot(&errors) / (n_samples as f64) + &weights * self.regularization;

            // gradient_b = (1/n) * sum(errors)
            let gradient_b = errors.sum() / (n_samples as f64);

            // Update parameters
            weights = &weights - &gradient_w * learning_rate;
            bias -= learning_rate * gradient_b;
        }

        // Update model parameters
        self.weights = weights.to_vec();
        self.bias = bias;
        self.training_samples += samples.len();
        self.last_updated = SystemTime::now();

        let training_time = start_time.elapsed().unwrap_or(Duration::from_secs(0));
        info!(
            "ML training completed in {:?} with {} samples (vectorized with SciRS2)",
            training_time,
            valid_samples.len()
        );
    }

    /// Get confidence score based on training samples
    pub fn get_confidence(&self) -> f64 {
        if self.training_samples < 10 {
            0.1
        } else if self.training_samples < 100 {
            0.3
        } else if self.training_samples < 1000 {
            0.6
        } else {
            0.8
        }
    }

    /// Get the number of training samples (alias for compatibility)
    pub fn sample_count(&self) -> usize {
        self.training_samples
    }
}

/// ML-enhanced query optimizer
pub struct MLQueryOptimizer {
    config: MLOptimizerConfig,
    base_optimizer: QueryOptimizer,
    #[allow(dead_code)]
    performance_tracker: Arc<PerformanceTracker>,
    execution_time_model: Arc<RwLock<LinearRegressionModel>>,
    memory_model: Arc<RwLock<LinearRegressionModel>>,
    training_samples: Arc<RwLock<VecDeque<TrainingSample>>>,
    feature_stats: Arc<RwLock<FeatureStatistics>>,
}

/// Statistics for feature normalization
#[derive(Debug, Clone, Default)]
pub struct FeatureStatistics {
    pub feature_means: Vec<f64>,
    pub feature_stds: Vec<f64>,
    pub sample_count: usize,
}

impl FeatureStatistics {
    /// Update statistics with new samples
    pub fn update(&mut self, samples: &[TrainingSample]) {
        if samples.is_empty() {
            return;
        }

        let feature_count = samples[0].features.to_vector().len();

        if self.feature_means.is_empty() {
            self.feature_means = vec![0.0; feature_count];
            self.feature_stds = vec![1.0; feature_count];
        }

        // Calculate means
        let mut sums = vec![0.0; feature_count];
        for sample in samples {
            let features = sample.features.to_vector();
            for (i, &feature) in features.iter().enumerate() {
                sums[i] += feature;
            }
        }

        let total_samples = self.sample_count + samples.len();
        #[allow(clippy::needless_range_loop)]
        for i in 0..feature_count {
            let new_mean = sums[i] / samples.len() as f64;
            self.feature_means[i] = (self.feature_means[i] * self.sample_count as f64
                + new_mean * samples.len() as f64)
                / total_samples as f64;
        }

        // Calculate standard deviations
        let mut var_sums = vec![0.0; feature_count];
        for sample in samples {
            let features = sample.features.to_vector();
            for (i, &feature) in features.iter().enumerate() {
                let diff = feature - self.feature_means[i];
                var_sums[i] += diff * diff;
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..feature_count {
            self.feature_stds[i] = (var_sums[i] / samples.len() as f64).sqrt().max(1e-6);
        }

        self.sample_count = total_samples;
    }

    /// Normalize features using z-score normalization
    /// Enhanced with SciRS2-Core array operations for vectorized processing
    pub fn normalize(&self, features: &[f64]) -> Vec<f64> {
        if self.feature_means.is_empty() || features.len() != self.feature_means.len() {
            return features.to_vec();
        }

        // Use SciRS2 array operations for efficient vectorized normalization
        let features_arr = Array1::from_vec(features.to_vec());
        let means_arr = Array1::from_vec(self.feature_means.clone());
        let stds_arr = Array1::from_vec(self.feature_stds.clone());

        // Vectorized z-score: (x - mean) / std
        let normalized = (&features_arr - &means_arr) / &stds_arr;

        normalized.to_vec()
    }
}

/// Advanced Neural Network ML Model using SciRS2-Core
/// Provides improved prediction accuracy through multi-layer perceptron architecture
pub struct NeuralNetworkModel {
    /// Weights for each layer
    weights: Vec<Array2<f64>>,
    /// Biases for each layer
    biases: Vec<Array1<f64>>,
    /// Layer sizes
    layer_sizes: Vec<usize>,
    /// Learning rate
    learning_rate: f64,
    /// L2 regularization strength
    l2_lambda: f64,
    /// Number of training iterations
    training_iterations: usize,
    /// Training loss history
    loss_history: Vec<f64>,
}

impl NeuralNetworkModel {
    /// Create a new neural network model
    pub fn new(layer_sizes: Vec<usize>, learning_rate: f64) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut rng = Random::seed(42);

        // Initialize weights and biases for each layer
        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Xavier/Glorot initialization: scale by sqrt(2 / (input + output))
            let scale = (2.0 / (input_size + output_size) as f64).sqrt();

            // Initialize weights with scaled normal distribution
            let weight_data: Vec<f64> = (0..input_size * output_size)
                .map(|_| rng.gen_range(-1.0..1.0) * scale)
                .collect();
            let weight_matrix =
                Array2::from_shape_vec((input_size, output_size), weight_data).unwrap();
            weights.push(weight_matrix);

            // Initialize biases to zero
            let bias_data: Vec<f64> = vec![0.0; output_size];
            biases.push(Array1::from_vec(bias_data));
        }

        Self {
            weights,
            biases,
            layer_sizes,
            learning_rate,
            l2_lambda: 0.001, // Default L2 regularization
            training_iterations: 0,
            loss_history: Vec::new(),
        }
    }

    /// ReLU activation function
    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    /// ReLU derivative
    fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &[f64]) -> (Vec<Array1<f64>>, f64) {
        let mut activations = vec![Array1::from_vec(input.to_vec())];

        for i in 0..self.weights.len() {
            let current = &activations[i];

            // Linear transformation: z = Wx + b
            let z = current.dot(&self.weights[i]) + &self.biases[i];

            // Apply activation (ReLU for hidden layers, linear for output)
            let activation = if i < self.weights.len() - 1 {
                z.mapv(Self::relu)
            } else {
                z // Linear output for regression
            };

            activations.push(activation);
        }

        // Return activations and final output
        let output = activations
            .last()
            .expect("activations should not be empty after forward pass")[0];
        (activations, output)
    }

    /// Train the model using gradient descent
    pub fn train(&mut self, samples: &[TrainingSample], epochs: usize) {
        for _epoch in 0..epochs {
            let mut total_loss = 0.0;

            for sample in samples {
                let features = sample.features.to_vector();
                let target = sample.execution_time_ms;

                // Forward pass
                let (activations, prediction) = self.forward(&features);

                // Compute loss (MSE)
                let error = prediction - target;
                total_loss += error * error;

                // Backward pass (gradient descent)
                self.backward(&activations, error);
            }

            // Record average loss
            let avg_loss = total_loss / samples.len() as f64;
            self.loss_history.push(avg_loss);
            self.training_iterations += 1;
        }
    }

    /// Backward pass for gradient computation and weight updates
    fn backward(&mut self, activations: &[Array1<f64>], error: f64) {
        // Output layer gradient
        let mut delta = Array1::from_vec(vec![error]);

        // Backpropagate through layers
        for i in (0..self.weights.len()).rev() {
            let input_activation = &activations[i];

            // Compute gradients
            let grad_w = input_activation
                .view()
                .insert_axis(scirs2_core::ndarray_ext::Axis(1))
                .dot(&delta.view().insert_axis(scirs2_core::ndarray_ext::Axis(0)));

            // Update weights with L2 regularization
            let regularization = &self.weights[i] * self.l2_lambda;
            self.weights[i] = &self.weights[i] - &(&grad_w * self.learning_rate + &regularization);

            // Update biases
            self.biases[i] = &self.biases[i] - &(&delta * self.learning_rate);

            // Compute delta for previous layer
            if i > 0 {
                let pre_activation = activations[i]
                    .iter()
                    .map(|&x| Self::relu_derivative(x))
                    .collect::<Vec<_>>();

                delta = self.weights[i]
                    .dot(&delta)
                    .iter()
                    .zip(pre_activation.iter())
                    .map(|(d, &r)| d * r)
                    .collect::<Vec<_>>()
                    .into();
            }
        }
    }

    /// Predict with confidence interval
    pub fn predict_with_confidence(&self, features: &[f64]) -> (f64, f64) {
        let (_, prediction) = self.forward(features);

        // Estimate confidence based on training history
        let confidence = if self.training_iterations < 100 {
            0.3 // Low confidence with few iterations
        } else if self.training_iterations < 1000 {
            0.6
        } else {
            // Use recent loss trend to estimate confidence
            let recent_loss = self.loss_history.iter().rev().take(10).sum::<f64>() / 10.0;
            (1.0 / (1.0 + recent_loss.sqrt())).min(0.95)
        };

        (prediction, confidence)
    }

    /// Get feature importance scores
    pub fn get_feature_importance(&self) -> Vec<f64> {
        if self.weights.is_empty() {
            return Vec::new();
        }

        // Use first layer weights to compute importance
        let first_layer = &self.weights[0];
        let mut importance: Vec<f64> = Vec::with_capacity(first_layer.nrows());

        for i in 0..first_layer.nrows() {
            let row_sum: f64 = first_layer.row(i).iter().map(|w| w.abs()).sum();
            importance.push(row_sum);
        }

        // Normalize
        let total: f64 = importance.iter().sum();
        if total > 0.0 {
            importance.iter_mut().for_each(|x| *x /= total);
        }

        importance
    }

    /// Get model statistics
    pub fn get_stats(&self) -> ModelStats {
        ModelStats {
            training_iterations: self.training_iterations,
            final_loss: self.loss_history.last().copied().unwrap_or(0.0),
            layer_sizes: self.layer_sizes.clone(),
            total_parameters: self.weights.iter().map(|w| w.len()).sum(),
        }
    }
}

/// Model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub training_iterations: usize,
    pub final_loss: f64,
    pub layer_sizes: Vec<usize>,
    pub total_parameters: usize,
}

/// Ensemble model combining linear regression and neural network
pub struct EnsembleModel {
    linear_model: LinearRegressionModel,
    neural_model: Option<NeuralNetworkModel>,
    linear_weight: f64,
    neural_weight: f64,
}

impl EnsembleModel {
    /// Create a new ensemble model
    pub fn new(feature_count: usize, layer_sizes: Option<Vec<usize>>, learning_rate: f64) -> Self {
        let neural_model = layer_sizes.map(|sizes| {
            let mut full_sizes = vec![feature_count];
            full_sizes.extend(sizes);
            full_sizes.push(1); // Single output
            NeuralNetworkModel::new(full_sizes, learning_rate)
        });

        Self {
            linear_model: LinearRegressionModel::new(feature_count),
            neural_model,
            linear_weight: 0.5,
            neural_weight: 0.5,
        }
    }

    /// Train the ensemble
    pub fn train(&mut self, samples: &[TrainingSample], epochs: usize) {
        // Train linear model with default learning rate and iterations
        self.linear_model.train(samples, 0.01, 100);

        // Train neural model if available
        if let Some(ref mut neural) = self.neural_model {
            neural.train(samples, epochs);
        }

        // Adjust weights based on performance
        self.adjust_weights(samples);
    }

    /// Adjust ensemble weights based on validation performance
    fn adjust_weights(&mut self, samples: &[TrainingSample]) {
        if samples.is_empty() {
            return;
        }

        // If no neural model, keep default weights
        if self.neural_model.is_none() {
            return;
        }

        let mut linear_error = 0.0;
        let mut neural_error = 0.0;

        for sample in samples {
            let features = sample.features.to_vector();
            let target = sample.execution_time_ms;

            let linear_pred = self.linear_model.predict(&features);
            linear_error += (linear_pred - target).abs();

            if let Some(ref neural) = self.neural_model {
                let (neural_pred, _) = neural.predict_with_confidence(&features);
                neural_error += (neural_pred - target).abs();
            }
        }

        // Inverse error weighting - models with lower error get higher weight
        // Add small epsilon to avoid division by zero
        let epsilon = 1e-6;
        let linear_inv = 1.0 / (linear_error + epsilon);
        let neural_inv = 1.0 / (neural_error + epsilon);
        let total_inv = linear_inv + neural_inv;

        // Normalize to get weights that sum to 1.0
        self.linear_weight = linear_inv / total_inv;
        self.neural_weight = neural_inv / total_inv;
    }

    /// Predict using ensemble
    pub fn predict(&self, features: &[f64]) -> f64 {
        let linear_pred = self.linear_model.predict(features);

        if let Some(ref neural) = self.neural_model {
            let (neural_pred, _) = neural.predict_with_confidence(features);
            linear_pred * self.linear_weight + neural_pred * self.neural_weight
        } else {
            linear_pred
        }
    }

    /// Predict with confidence
    pub fn predict_with_confidence(&self, features: &[f64]) -> (f64, f64) {
        let prediction = self.predict(features);

        let confidence = if let Some(ref neural) = self.neural_model {
            let (_, neural_conf) = neural.predict_with_confidence(features);
            let linear_conf = self.linear_model.get_confidence();
            linear_conf * self.linear_weight + neural_conf * self.neural_weight
        } else {
            self.linear_model.get_confidence()
        };

        (prediction, confidence)
    }

    /// Get ensemble statistics
    pub fn get_stats(&self) -> EnsembleStats {
        EnsembleStats {
            linear_weight: self.linear_weight,
            neural_weight: self.neural_weight,
            linear_sample_count: self.linear_model.sample_count(),
            neural_stats: self.neural_model.as_ref().map(|n| n.get_stats()),
        }
    }
}

/// Ensemble model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleStats {
    pub linear_weight: f64,
    pub neural_weight: f64,
    pub linear_sample_count: usize,
    pub neural_stats: Option<ModelStats>,
}

impl MLQueryOptimizer {
    /// Create a new ML-enhanced query optimizer
    pub fn new(config: MLOptimizerConfig, performance_tracker: Arc<PerformanceTracker>) -> Self {
        let feature_count = 12; // Number of features in QueryFeatures

        Self {
            config,
            base_optimizer: QueryOptimizer::new(crate::optimizer::OptimizationConfig::default()),
            performance_tracker,
            execution_time_model: Arc::new(RwLock::new(LinearRegressionModel::new(feature_count))),
            memory_model: Arc::new(RwLock::new(LinearRegressionModel::new(feature_count))),
            training_samples: Arc::new(RwLock::new(VecDeque::new())),
            feature_stats: Arc::new(RwLock::new(FeatureStatistics::default())),
        }
    }

    /// Extract features from a GraphQL document
    pub fn extract_features(&self, document: &Document) -> Result<QueryFeatures> {
        let mut field_count = 0;
        let mut max_depth = 0;
        let mut selection_count = 0;
        let mut has_fragments = false;
        let mut has_variables = false;
        let mut operation_type = 0.0;
        let mut unique_field_types = HashSet::new();
        let mut nested_list_count = 0;
        let mut argument_count = 0;
        let mut directive_count = 0;

        use std::collections::HashSet;

        // Analyze each definition in the document
        for definition in &document.definitions {
            match definition {
                crate::ast::Definition::Operation(operation) => {
                    operation_type = match operation.operation_type {
                        OperationType::Query => 0.0,
                        OperationType::Mutation => 1.0,
                        OperationType::Subscription => 2.0,
                    };

                    if !operation.variable_definitions.is_empty() {
                        has_variables = true;
                    }

                    // Analyze selection set
                    let (fc, md, sc, uft, nlc, ac, dc) =
                        self.analyze_selection_set(&operation.selection_set, 1)?;
                    field_count += fc;
                    max_depth = max_depth.max(md);
                    selection_count += sc;
                    unique_field_types.extend(uft);
                    nested_list_count += nlc;
                    argument_count += ac;
                    directive_count += dc;
                }
                crate::ast::Definition::Fragment(_) => {
                    has_fragments = true;
                }
                crate::ast::Definition::Schema(_) => {
                    // Schema definitions don't affect query complexity
                }
                crate::ast::Definition::Type(_) => {
                    // Type definitions don't affect query complexity
                }
                crate::ast::Definition::Directive(_) => {
                    directive_count += 1;
                }
                crate::ast::Definition::SchemaExtension(_) => {
                    // Schema extensions don't affect query complexity
                }
                crate::ast::Definition::TypeExtension(_) => {
                    // Type extensions don't affect query complexity
                }
            }
        }

        let complexity = self.base_optimizer.analyze_complexity(document)?;

        Ok(QueryFeatures {
            field_count: field_count as f64,
            max_depth: max_depth as f64,
            complexity_score: complexity.complexity_score as f64,
            selection_count: selection_count as f64,
            has_fragments: if has_fragments { 1.0 } else { 0.0 },
            has_variables: if has_variables { 1.0 } else { 0.0 },
            operation_type,
            unique_field_types: unique_field_types.len() as f64,
            nested_list_count: nested_list_count as f64,
            argument_count: argument_count as f64,
            directive_count: directive_count as f64,
            estimated_result_size: self.estimate_result_size(&complexity),
        })
    }

    /// Predict query performance using ML models
    pub async fn predict_performance(&self, document: &Document) -> Result<PerformancePrediction> {
        let features = self.extract_features(document)?;
        let feature_vector = features.to_vector();

        // Normalize features
        let stats = self.feature_stats.read().await;
        let normalized_features = stats.normalize(&feature_vector);
        drop(stats);

        // Get predictions from models
        let execution_time_model = self.execution_time_model.read().await;
        let memory_model = self.memory_model.read().await;

        let predicted_execution_ms = execution_time_model.predict(&normalized_features);
        let predicted_memory_mb = memory_model.predict(&normalized_features);

        // Calculate confidence based on training samples
        let confidence = self
            .calculate_confidence(&execution_time_model, &memory_model)
            .await;

        // Estimate cache hit probability and error probability based on complexity
        let cache_hit_probability = self.estimate_cache_hit_probability(&features);
        let error_probability = self.estimate_error_probability(&features);

        Ok(PerformancePrediction {
            predicted_execution_time: Duration::from_millis(predicted_execution_ms as u64),
            predicted_memory_usage: predicted_memory_mb,
            cache_hit_probability,
            error_probability,
            confidence_score: confidence,
        })
    }

    /// Generate optimization recommendations based on ML predictions
    pub async fn recommend_optimizations(
        &self,
        document: &Document,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let features = self.extract_features(document)?;
        let prediction = self.predict_performance(document).await?;
        let mut recommendations = Vec::new();

        // Recommend based on predicted performance
        if prediction.predicted_execution_time > Duration::from_millis(1000) {
            if features.max_depth > 5.0 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::ReduceDepth,
                    description: "Consider reducing query depth to improve performance".to_string(),
                    estimated_improvement: 15.0,
                    confidence: 0.8,
                });
            }

            if features.complexity_score > 100.0 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::ReduceComplexity,
                    description: "Query complexity is high, consider simplifying".to_string(),
                    estimated_improvement: 20.0,
                    confidence: 0.7,
                });
            }
        }

        if prediction.cache_hit_probability < 0.3 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::AddCaching,
                description: "Low cache hit probability, consider adding caching strategy"
                    .to_string(),
                estimated_improvement: 30.0,
                confidence: 0.6,
            });
        }

        if features.field_count > 20.0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::ParallelizeFields,
                description: "Many fields requested, consider parallelizing field resolution"
                    .to_string(),
                estimated_improvement: 25.0,
                confidence: 0.7,
            });
        }

        Ok(recommendations)
    }

    /// Record query execution for learning
    pub async fn record_execution(
        &self,
        document: &Document,
        metrics: &OperationMetrics,
    ) -> Result<()> {
        if !self.config.learning_enabled {
            return Ok(());
        }

        let features = self.extract_features(document)?;

        // Get real memory usage measurement
        let memory_usage_mb = system_monitor::get_current_memory_usage_mb().await;

        let sample = TrainingSample {
            features,
            execution_time_ms: metrics.execution_time.as_millis() as f64,
            memory_usage_mb,
            cache_hit: metrics.cache_hit,
            error_occurred: metrics.error_count > 0,
            timestamp: metrics.timestamp,
        };

        // Add to training samples
        let mut samples = self.training_samples.write().await;
        samples.push_back(sample);

        // Limit sample size
        while samples.len() > self.config.max_training_samples {
            samples.pop_front();
        }

        // Trigger model update if we have enough samples
        if samples.len() >= self.config.min_samples_for_learning {
            self.update_models().await?;
        }

        Ok(())
    }

    /// Update ML models with new training data
    pub async fn update_models(&self) -> Result<()> {
        let samples = self.training_samples.read().await;
        if samples.len() < self.config.min_samples_for_learning {
            return Ok(());
        }

        let recent_samples: Vec<_> = samples
            .iter()
            .filter(|sample| {
                sample.timestamp.elapsed().unwrap_or(Duration::from_secs(0))
                    < self.config.performance_history_window
            })
            .cloned()
            .collect();

        if recent_samples.is_empty() {
            return Ok(());
        }

        drop(samples);

        // Update feature statistics
        {
            let mut stats = self.feature_stats.write().await;
            stats.update(&recent_samples);
        }

        // Train execution time model
        {
            let mut model = self.execution_time_model.write().await;
            model.train(&recent_samples, 0.01, 100); // learning_rate=0.01, iterations=100
        }

        // Train memory model
        {
            let mut model = self.memory_model.write().await;
            model.train(&recent_samples, 0.01, 100);
        }

        info!("ML models updated with {} samples", recent_samples.len());
        Ok(())
    }

    /// Analyze a selection set recursively
    #[allow(clippy::only_used_in_recursion)]
    #[allow(clippy::type_complexity)]
    fn analyze_selection_set(
        &self,
        selection_set: &SelectionSet,
        depth: usize,
    ) -> Result<(usize, usize, usize, HashSet<String>, usize, usize, usize)> {
        let mut field_count = 0;
        let mut max_depth = depth;
        let mut selection_count = selection_set.selections.len();
        let mut unique_field_types = HashSet::new();
        let mut nested_list_count = 0;
        let mut argument_count = 0;
        let mut directive_count = 0;

        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    field_count += 1;
                    unique_field_types.insert(field.name.clone());
                    argument_count += field.arguments.len();
                    directive_count += field.directives.len();

                    if let Some(ref sub_selection_set) = field.selection_set {
                        let (fc, md, sc, uft, nlc, ac, dc) =
                            self.analyze_selection_set(sub_selection_set, depth + 1)?;
                        field_count += fc;
                        max_depth = max_depth.max(md);
                        selection_count += sc;
                        unique_field_types.extend(uft);
                        nested_list_count += nlc;
                        argument_count += ac;
                        directive_count += dc;

                        // Count nested lists (simplified heuristic)
                        if field.name.ends_with("s") || field.name.contains("list") {
                            nested_list_count += 1;
                        }
                    }
                }
                Selection::InlineFragment(fragment) => {
                    directive_count += fragment.directives.len();
                    let (fc, md, sc, uft, nlc, ac, dc) =
                        self.analyze_selection_set(&fragment.selection_set, depth)?;
                    field_count += fc;
                    max_depth = max_depth.max(md);
                    selection_count += sc;
                    unique_field_types.extend(uft);
                    nested_list_count += nlc;
                    argument_count += ac;
                    directive_count += dc;
                }
                Selection::FragmentSpread(spread) => {
                    directive_count += spread.directives.len();
                    // Fragment spread doesn't directly contribute to depth analysis
                }
            }
        }

        Ok((
            field_count,
            max_depth,
            selection_count,
            unique_field_types,
            nested_list_count,
            argument_count,
            directive_count,
        ))
    }

    /// Estimate result size based on query complexity
    fn estimate_result_size(&self, complexity: &QueryComplexity) -> f64 {
        // Simple heuristic based on field count and depth
        (complexity.field_count as f64 * complexity.depth as f64)
            .log10()
            .max(1.0)
    }

    /// Estimate cache hit probability based on query features
    fn estimate_cache_hit_probability(&self, features: &QueryFeatures) -> f64 {
        // Simple heuristic: less complex queries are more likely to be cached
        let complexity_factor = (features.complexity_score / 100.0).min(1.0);
        (1.0 - complexity_factor).max(0.1)
    }

    /// Estimate error probability based on query features
    fn estimate_error_probability(&self, features: &QueryFeatures) -> f64 {
        // Simple heuristic: more complex queries are more likely to have errors
        let complexity_factor = (features.complexity_score / 200.0).min(1.0);
        complexity_factor.max(0.01)
    }

    /// Calculate confidence score based on model training
    async fn calculate_confidence(
        &self,
        execution_model: &LinearRegressionModel,
        memory_model: &LinearRegressionModel,
    ) -> f64 {
        let min_samples = execution_model
            .training_samples
            .min(memory_model.training_samples);
        let confidence =
            (min_samples as f64 / self.config.min_samples_for_learning as f64).min(1.0);
        confidence.max(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::performance::PerformanceTracker;

    #[test]
    fn test_linear_regression_model() {
        let feature_count = 12; // Match QueryFeatures vector length
        let mut model = LinearRegressionModel::new(feature_count);

        // Create simple training data
        let samples = vec![
            TrainingSample {
                features: QueryFeatures::from_vector(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ])
                .unwrap(),
                execution_time_ms: 100.0,
                memory_usage_mb: 0.0,
                cache_hit: false,
                error_occurred: false,
                timestamp: SystemTime::now(),
            },
            TrainingSample {
                features: QueryFeatures::from_vector(&[
                    2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
                ])
                .unwrap(),
                execution_time_ms: 200.0,
                memory_usage_mb: 0.0,
                cache_hit: false,
                error_occurred: false,
                timestamp: SystemTime::now(),
            },
        ];

        model.train(&samples, 0.01, 100);

        // Test prediction with a feature vector
        let test_features = [
            1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0,
        ];
        let prediction = model.predict(&test_features);

        // Since we're training with limited data, just ensure we get a reasonable prediction
        assert!(prediction >= 0.0, "Prediction should be non-negative");
        assert!(prediction < 1000.0, "Prediction should be reasonable");
    }

    #[test]
    fn test_feature_statistics() {
        let mut stats = FeatureStatistics::default();

        let samples = vec![
            TrainingSample {
                features: QueryFeatures::from_vector(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ])
                .unwrap(),
                execution_time_ms: 100.0,
                memory_usage_mb: 50.0, // Realistic test value
                cache_hit: false,
                error_occurred: false,
                timestamp: SystemTime::now(),
            },
            TrainingSample {
                features: QueryFeatures::from_vector(&[
                    2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
                ])
                .unwrap(),
                execution_time_ms: 200.0,
                memory_usage_mb: 20.0,
                cache_hit: true,
                error_occurred: false,
                timestamp: SystemTime::now(),
            },
        ];

        stats.update(&samples);

        assert_eq!(stats.feature_means.len(), 12);
        assert_eq!(stats.feature_stds.len(), 12);
        assert_eq!(stats.sample_count, 2);

        // Test normalization
        let normalized = stats.normalize(&[
            1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0,
        ]);
        assert_eq!(normalized.len(), 12);
    }

    #[tokio::test]
    async fn test_ml_optimizer_creation() {
        let config = MLOptimizerConfig::default();
        let performance_tracker = Arc::new(PerformanceTracker::new());

        let optimizer = MLQueryOptimizer::new(config, performance_tracker);

        // Test with a simple document
        let document = Document {
            definitions: vec![Definition::Operation(OperationDefinition {
                operation_type: OperationType::Query,
                name: Some("TestQuery".to_string()),
                variable_definitions: vec![],
                directives: vec![],
                selection_set: SelectionSet {
                    selections: vec![Selection::Field(Field {
                        alias: None,
                        name: "user".to_string(),
                        arguments: vec![],
                        directives: vec![],
                        selection_set: Some(SelectionSet {
                            selections: vec![Selection::Field(Field {
                                alias: None,
                                name: "id".to_string(),
                                arguments: vec![],
                                directives: vec![],
                                selection_set: None,
                            })],
                        }),
                    })],
                },
            })],
        };

        let features = optimizer.extract_features(&document).unwrap();
        assert!(features.field_count > 0.0);
        assert!(features.max_depth > 0.0);
    }

    #[test]
    fn test_neural_network_model_creation() {
        let layer_sizes = vec![12, 8, 4, 1];
        let model = NeuralNetworkModel::new(layer_sizes.clone(), 0.01);

        assert_eq!(model.layer_sizes, layer_sizes);
        assert_eq!(model.weights.len(), 3); // 3 weight matrices for 4 layers
        assert_eq!(model.biases.len(), 3);
        assert_eq!(model.training_iterations, 0);
    }

    #[test]
    fn test_neural_network_forward_pass() {
        let layer_sizes = vec![3, 2, 1];
        let model = NeuralNetworkModel::new(layer_sizes, 0.01);

        let input = vec![1.0, 2.0, 3.0];
        let (activations, output) = model.forward(&input);

        assert_eq!(activations.len(), 3); // Input + 2 hidden layers
        assert_eq!(activations[0].len(), 3); // Input size
        assert!(!output.is_nan());
    }

    #[test]
    fn test_neural_network_training() {
        let layer_sizes = vec![12, 4, 1];
        let mut model = NeuralNetworkModel::new(layer_sizes, 0.01);

        // Create synthetic training samples
        let samples: Vec<TrainingSample> = (0..10)
            .map(|i| {
                let features = QueryFeatures {
                    field_count: i as f64,
                    max_depth: (i % 5) as f64,
                    complexity_score: i as f64 * 2.0,
                    selection_count: i as f64,
                    has_fragments: 0.0,
                    has_variables: 1.0,
                    operation_type: 0.0,
                    unique_field_types: 3.0,
                    nested_list_count: 0.0,
                    argument_count: i as f64,
                    directive_count: 0.0,
                    estimated_result_size: 100.0,
                };
                TrainingSample {
                    features,
                    execution_time_ms: (i as f64 * 10.0).max(1.0),
                    memory_usage_mb: 1.0,
                    cache_hit: false,
                    error_occurred: false,
                    timestamp: std::time::SystemTime::now(),
                }
            })
            .collect();

        model.train(&samples, 10);

        assert_eq!(model.training_iterations, 10);
        assert_eq!(model.loss_history.len(), 10);
    }

    #[test]
    fn test_neural_network_predict_with_confidence() {
        let layer_sizes = vec![12, 4, 1];
        let model = NeuralNetworkModel::new(layer_sizes, 0.01);

        let features = vec![1.0; 12];
        let (prediction, confidence) = model.predict_with_confidence(&features);

        assert!(!prediction.is_nan());
        assert!((0.0..=1.0).contains(&confidence));
    }

    #[test]
    fn test_neural_network_feature_importance() {
        let layer_sizes = vec![12, 4, 1];
        let model = NeuralNetworkModel::new(layer_sizes, 0.01);

        let importance = model.get_feature_importance();

        assert_eq!(importance.len(), 12);
        // Importance should sum to approximately 1.0
        let total: f64 = importance.iter().sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_neural_network_stats() {
        let layer_sizes = vec![12, 8, 4, 1];
        let model = NeuralNetworkModel::new(layer_sizes.clone(), 0.01);

        let stats = model.get_stats();

        assert_eq!(stats.training_iterations, 0);
        assert_eq!(stats.final_loss, 0.0);
        assert_eq!(stats.layer_sizes, layer_sizes);
        assert!(stats.total_parameters > 0);
    }

    #[test]
    fn test_ensemble_model_creation() {
        let model = EnsembleModel::new(12, Some(vec![8, 4]), 0.01);

        assert_eq!(model.linear_weight, 0.5);
        assert_eq!(model.neural_weight, 0.5);
    }

    #[test]
    fn test_ensemble_model_prediction() {
        let model = EnsembleModel::new(12, Some(vec![4]), 0.01);

        let features = vec![1.0; 12];
        let prediction = model.predict(&features);

        assert!(!prediction.is_nan());
    }

    #[test]
    fn test_ensemble_model_predict_with_confidence() {
        let model = EnsembleModel::new(12, Some(vec![4]), 0.01);

        let features = vec![1.0; 12];
        let (prediction, confidence) = model.predict_with_confidence(&features);

        assert!(!prediction.is_nan());
        assert!((0.0..=1.0).contains(&confidence));
    }

    #[test]
    fn test_ensemble_model_training() {
        // Test ensemble without neural network component (simpler case)
        let mut model = EnsembleModel::new(12, None, 0.01);

        // Create synthetic training samples
        let samples: Vec<TrainingSample> = (0..5)
            .map(|i| {
                let features = QueryFeatures {
                    field_count: (i + 1) as f64,
                    max_depth: 2.0,
                    complexity_score: (i + 1) as f64 * 2.0,
                    selection_count: (i + 1) as f64,
                    has_fragments: 0.0,
                    has_variables: 1.0,
                    operation_type: 0.0,
                    unique_field_types: 3.0,
                    nested_list_count: 0.0,
                    argument_count: (i + 1) as f64,
                    directive_count: 0.0,
                    estimated_result_size: 100.0,
                };
                TrainingSample {
                    features,
                    execution_time_ms: ((i + 1) as f64 * 10.0),
                    memory_usage_mb: 1.0,
                    cache_hit: false,
                    error_occurred: false,
                    timestamp: std::time::SystemTime::now(),
                }
            })
            .collect();

        model.train(&samples, 5);

        // Weights should remain at default when no neural network
        let stats = model.get_stats();
        assert_eq!(stats.linear_weight, 0.5);
        assert_eq!(stats.neural_weight, 0.5);
        assert!(stats.neural_stats.is_none());
    }

    #[test]
    fn test_ensemble_stats() {
        let model = EnsembleModel::new(12, Some(vec![8, 4]), 0.01);

        let stats = model.get_stats();

        assert_eq!(stats.linear_weight, 0.5);
        assert_eq!(stats.neural_weight, 0.5);
        assert_eq!(stats.linear_sample_count, 0);
        assert!(stats.neural_stats.is_some());
    }

    #[test]
    fn test_linear_regression_confidence() {
        let mut model = LinearRegressionModel::new(12);

        // Initial confidence should be low
        assert!(model.get_confidence() < 0.3);

        // Simulate training samples
        model.training_samples = 500;
        assert!(model.get_confidence() > 0.5);

        model.training_samples = 2000;
        assert!(model.get_confidence() >= 0.8);
    }
}
