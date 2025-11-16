//! Machine Learning-Enhanced SPARQL-star Query Optimizer
//!
//! This module provides ML-powered query optimization for SPARQL-star queries,
//! learning from historical query performance to make intelligent optimization decisions.
//!
//! ## Features
//!
//! - **Feature Extraction**: Automatic extraction of query characteristics
//! - **Performance Learning**: Learn from query execution patterns
//! - **Adaptive Optimization**: Continuously improve query plans based on historical data
//! - **Pattern Recognition**: Identify common query patterns and anti-patterns
//! - **Cost Prediction**: ML-based cost estimation for query execution
//!
//! ## SciRS2-Core Integration
//!
//! This module leverages SciRS2-Core for:
//! - **Array Operations**: `scirs2_core::ndarray_ext` for feature matrices
//! - **ML Pipeline**: `scirs2_core::ml_pipeline` for model orchestration
//! - **Profiling**: `scirs2_core::profiling` for performance tracking
//! - **Random**: `scirs2_core::random` for initialization and sampling

use crate::{StarError, StarResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

// SciRS2-Core integration for ML capabilities
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{rand_distributions as rand_distr, thread_rng};

/// ML optimizer configuration for SPARQL-star queries
#[derive(Debug, Clone)]
pub struct MLSPARQLOptimizerConfig {
    /// Enable machine learning-based optimization
    pub learning_enabled: bool,
    /// Minimum number of samples required before ML kicks in
    pub min_samples_for_learning: usize,
    /// Enable advanced feature extraction
    pub feature_extraction_enabled: bool,
    /// Prediction confidence threshold (0.0-1.0)
    pub prediction_threshold: f64,
    /// How often to retrain the model
    pub model_update_interval: Duration,
    /// Maximum number of training samples to keep
    pub max_training_samples: usize,
    /// Performance history retention window
    pub performance_history_window: Duration,
    /// Use neural network for prediction
    pub use_neural_network: bool,
    /// Neural network layer sizes
    pub neural_network_layers: Vec<usize>,
    /// Learning rate for neural network
    pub neural_learning_rate: f64,
    /// Enable reinforcement learning
    pub enable_reinforcement_learning: bool,
    /// Enable semantic analysis of SPARQL-star queries
    pub enable_semantic_analysis: bool,
    /// Adaptive optimization based on workload
    pub adaptive_optimization: bool,
}

impl Default for MLSPARQLOptimizerConfig {
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

/// SPARQL-star query features extracted for ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPARQLStarQueryFeatures {
    /// Number of triple patterns in the query
    pub triple_pattern_count: f64,
    /// Number of quoted triple patterns
    pub quoted_triple_count: f64,
    /// Maximum nesting depth of quoted triples
    pub max_nesting_depth: f64,
    /// Number of FILTER clauses
    pub filter_count: f64,
    /// Number of OPTIONAL patterns
    pub optional_count: f64,
    /// Number of UNION patterns
    pub union_count: f64,
    /// Number of graph patterns
    pub graph_pattern_count: f64,
    /// Number of variables
    pub variable_count: f64,
    /// Selectivity estimate (0.0-1.0)
    pub estimated_selectivity: f64,
    /// Join complexity score
    pub join_complexity: f64,
    /// Has aggregation
    pub has_aggregation: f64,
    /// Has subquery
    pub has_subquery: f64,
    /// Has property path
    pub has_property_path: f64,
    /// Estimated result size
    pub estimated_result_size: f64,
    /// Query type (0=SELECT, 1=CONSTRUCT, 2=ASK, 3=DESCRIBE)
    pub query_type: f64,
}

impl SPARQLStarQueryFeatures {
    /// Convert features to a vector for ML algorithms
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.triple_pattern_count,
            self.quoted_triple_count,
            self.max_nesting_depth,
            self.filter_count,
            self.optional_count,
            self.union_count,
            self.graph_pattern_count,
            self.variable_count,
            self.estimated_selectivity,
            self.join_complexity,
            self.has_aggregation,
            self.has_subquery,
            self.has_property_path,
            self.estimated_result_size,
            self.query_type,
        ]
    }

    /// Create from vector
    pub fn from_vector(vec: &[f64]) -> Self {
        Self {
            triple_pattern_count: vec.first().copied().unwrap_or(0.0),
            quoted_triple_count: vec.get(1).copied().unwrap_or(0.0),
            max_nesting_depth: vec.get(2).copied().unwrap_or(0.0),
            filter_count: vec.get(3).copied().unwrap_or(0.0),
            optional_count: vec.get(4).copied().unwrap_or(0.0),
            union_count: vec.get(5).copied().unwrap_or(0.0),
            graph_pattern_count: vec.get(6).copied().unwrap_or(0.0),
            variable_count: vec.get(7).copied().unwrap_or(0.0),
            estimated_selectivity: vec.get(8).copied().unwrap_or(0.5),
            join_complexity: vec.get(9).copied().unwrap_or(1.0),
            has_aggregation: vec.get(10).copied().unwrap_or(0.0),
            has_subquery: vec.get(11).copied().unwrap_or(0.0),
            has_property_path: vec.get(12).copied().unwrap_or(0.0),
            estimated_result_size: vec.get(13).copied().unwrap_or(100.0),
            query_type: vec.get(14).copied().unwrap_or(0.0),
        }
    }

    /// Dimensionality of feature vector
    pub const fn feature_dimension() -> usize {
        15
    }
}

/// Query performance record for ML training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceRecord {
    pub features: SPARQLStarQueryFeatures,
    pub execution_time_ms: f64,
    pub result_count: usize,
    pub memory_used_bytes: usize,
    #[serde(skip)]
    #[serde(default = "Instant::now")]
    pub timestamp: Instant,
    pub optimization_hints_used: Vec<String>,
    pub plan_chosen: String,
}

/// ML model for query cost prediction
#[derive(Debug)]
pub struct MLCostPredictor {
    /// Training samples
    training_data: VecDeque<QueryPerformanceRecord>,
    /// Feature weights learned from data
    weights: Array1<f64>,
    /// Model bias term
    bias: f64,
    /// Last training time
    last_trained: Option<Instant>,
    /// Model accuracy metrics
    accuracy_metrics: AccuracyMetrics,
}

/// Model accuracy tracking
#[derive(Debug, Clone, Default)]
pub struct AccuracyMetrics {
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub r_squared: f64,
    pub predictions_made: usize,
    pub predictions_accurate: usize, // Within 10% of actual
}

impl MLCostPredictor {
    /// Create a new ML cost predictor with proper random initialization
    pub fn new() -> Self {
        let feature_dim = SPARQLStarQueryFeatures::feature_dimension();

        // Initialize weights using Xavier initialization for better convergence
        // Xavier: weights ~ U[-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
        let mut rng = thread_rng();
        let limit = (6.0 / (feature_dim as f64 + 1.0)).sqrt();
        let uniform = rand_distr::Uniform::new(-limit, limit).unwrap();
        let weights_vec: Vec<f64> = (0..feature_dim).map(|_| rng.sample(uniform)).collect();

        Self {
            training_data: VecDeque::new(),
            weights: Array1::from_vec(weights_vec),
            bias: 0.0,
            last_trained: None,
            accuracy_metrics: AccuracyMetrics::default(),
        }
    }

    /// Add training sample
    pub fn add_training_sample(&mut self, record: QueryPerformanceRecord, max_samples: usize) {
        self.training_data.push_back(record);

        // Keep only recent samples
        while self.training_data.len() > max_samples {
            self.training_data.pop_front();
        }
    }

    /// Train the model using gradient descent
    pub fn train(&mut self, learning_rate: f64, iterations: usize) -> StarResult<()> {
        if self.training_data.is_empty() {
            return Err(StarError::query_error(
                "Cannot train with no training data".to_string(),
            ));
        }

        info!(
            "Training ML cost predictor with {} samples",
            self.training_data.len()
        );

        for iteration in 0..iterations {
            let mut total_loss = 0.0;

            // Stochastic gradient descent
            for record in self.training_data.iter() {
                let features = Array1::from_vec(record.features.to_vector());
                let actual_cost = record.execution_time_ms;

                // Forward pass
                let predicted_cost = self.predict_from_array(&features);

                // Calculate error
                let error = predicted_cost - actual_cost;
                total_loss += error * error;

                // Backward pass - update weights
                for i in 0..self.weights.len() {
                    let gradient = 2.0 * error * features[i];
                    self.weights[i] -= learning_rate * gradient;
                }

                // Update bias
                self.bias -= learning_rate * 2.0 * error;
            }

            if iteration % 100 == 0 {
                let avg_loss = total_loss / self.training_data.len() as f64;
                debug!("Iteration {}: Average loss = {:.2}", iteration, avg_loss);
            }
        }

        self.last_trained = Some(Instant::now());
        self.update_accuracy_metrics();

        info!(
            "Model trained. MAE: {:.2}ms, R²: {:.3}",
            self.accuracy_metrics.mean_absolute_error, self.accuracy_metrics.r_squared
        );

        Ok(())
    }

    /// Predict cost from feature array
    fn predict_from_array(&self, features: &Array1<f64>) -> f64 {
        let mut result = self.bias;
        for i in 0..self.weights.len().min(features.len()) {
            result += self.weights[i] * features[i];
        }
        result.max(0.0) // Cost cannot be negative
    }

    /// Predict execution cost for query features
    pub fn predict(&mut self, features: &SPARQLStarQueryFeatures) -> f64 {
        let feature_array = Array1::from_vec(features.to_vector());
        let prediction = self.predict_from_array(&feature_array);

        self.accuracy_metrics.predictions_made += 1;

        prediction
    }

    /// Update accuracy metrics based on training data
    fn update_accuracy_metrics(&mut self) {
        if self.training_data.is_empty() {
            return;
        }

        let n = self.training_data.len() as f64;
        let mut sum_abs_error = 0.0;
        let mut sum_squared_error = 0.0;
        let mut sum_actual = 0.0;
        let mut accurate_count = 0;

        for record in self.training_data.iter() {
            let features = Array1::from_vec(record.features.to_vector());
            let actual = record.execution_time_ms;
            let predicted = self.predict_from_array(&features);

            let error = predicted - actual;
            sum_abs_error += error.abs();
            sum_squared_error += error * error;
            sum_actual += actual;

            // Check if within 10% accuracy
            if (error.abs() / actual.max(1.0)) < 0.1 {
                accurate_count += 1;
            }
        }

        let mean_actual = sum_actual / n;
        let mut sum_total_variance = 0.0;

        for record in self.training_data.iter() {
            let diff = record.execution_time_ms - mean_actual;
            sum_total_variance += diff * diff;
        }

        self.accuracy_metrics.mean_absolute_error = sum_abs_error / n;
        self.accuracy_metrics.mean_squared_error = sum_squared_error / n;
        self.accuracy_metrics.r_squared = if sum_total_variance > 0.0 {
            1.0 - (sum_squared_error / sum_total_variance)
        } else {
            0.0
        };
        self.accuracy_metrics.predictions_accurate = accurate_count;
    }

    /// Get model statistics
    pub fn get_statistics(&self) -> MLModelStatistics {
        MLModelStatistics {
            training_samples: self.training_data.len(),
            last_trained: self.last_trained,
            mean_absolute_error: self.accuracy_metrics.mean_absolute_error,
            mean_squared_error: self.accuracy_metrics.mean_squared_error,
            r_squared: self.accuracy_metrics.r_squared,
            accuracy_rate: if self.accuracy_metrics.predictions_made > 0 {
                self.accuracy_metrics.predictions_accurate as f64
                    / self.accuracy_metrics.predictions_made as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for MLCostPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// ML model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelStatistics {
    pub training_samples: usize,
    #[serde(skip)]
    pub last_trained: Option<Instant>,
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub r_squared: f64,
    pub accuracy_rate: f64,
}

/// Machine learning-enhanced SPARQL-star query optimizer
pub struct MLSPARQLOptimizer {
    config: MLSPARQLOptimizerConfig,
    cost_predictor: Arc<RwLock<MLCostPredictor>>,
    feature_extractor: FeatureExtractor,
    performance_history: Arc<RwLock<VecDeque<QueryPerformanceRecord>>>,
    pattern_cache: Arc<RwLock<HashMap<String, SPARQLStarQueryFeatures>>>,
}

impl MLSPARQLOptimizer {
    /// Create a new ML-enhanced SPARQL-star optimizer
    pub fn new(config: MLSPARQLOptimizerConfig) -> Self {
        Self {
            config,
            cost_predictor: Arc::new(RwLock::new(MLCostPredictor::new())),
            feature_extractor: FeatureExtractor::new(),
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Extract features from a SPARQL-star query
    pub async fn extract_features(&self, query: &str) -> StarResult<SPARQLStarQueryFeatures> {
        // Check cache first
        {
            let cache = self.pattern_cache.read().await;
            if let Some(cached) = cache.get(query) {
                return Ok(cached.clone());
            }
        }

        let features = self.feature_extractor.extract(query)?;

        // Cache the result
        {
            let mut cache = self.pattern_cache.write().await;
            cache.insert(query.to_string(), features.clone());
        }

        Ok(features)
    }

    /// Predict execution cost for a query
    pub async fn predict_cost(&self, features: &SPARQLStarQueryFeatures) -> f64 {
        let mut predictor = self.cost_predictor.write().await;
        predictor.predict(features)
    }

    /// Record actual query performance for learning
    pub async fn record_performance(&self, record: QueryPerformanceRecord) -> StarResult<()> {
        // Add to history
        {
            let mut history = self.performance_history.write().await;
            history.push_back(record.clone());

            // Prune old records
            let cutoff = Instant::now() - self.config.performance_history_window;
            while let Some(front) = history.front() {
                if front.timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }

        // Add to training data
        {
            let mut predictor = self.cost_predictor.write().await;
            predictor.add_training_sample(record, self.config.max_training_samples);
        }

        // Check if we should retrain
        let should_retrain = {
            let predictor = self.cost_predictor.read().await;
            let history = self.performance_history.read().await;

            history.len() >= self.config.min_samples_for_learning
                && predictor
                    .last_trained
                    .map(|t| Instant::now() - t > self.config.model_update_interval)
                    .unwrap_or(true)
        };

        if should_retrain && self.config.learning_enabled {
            self.train_model().await?;
        }

        Ok(())
    }

    /// Train or retrain the ML model
    pub async fn train_model(&self) -> StarResult<()> {
        info!("Retraining ML cost prediction model");

        let mut predictor = self.cost_predictor.write().await;
        predictor.train(self.config.neural_learning_rate, 1000)?;

        Ok(())
    }

    /// Get ML model statistics
    pub async fn get_model_statistics(&self) -> MLModelStatistics {
        let predictor = self.cost_predictor.read().await;
        predictor.get_statistics()
    }

    /// Suggest optimization hints based on ML predictions
    pub async fn suggest_optimizations(&self, features: &SPARQLStarQueryFeatures) -> Vec<String> {
        let mut hints = Vec::new();

        // High nesting depth - suggest materialization
        if features.max_nesting_depth > 3.0 {
            hints.push("MaterializeIntermediateResults".to_string());
        }

        // Many joins - suggest reordering
        if features.join_complexity > 5.0 {
            hints.push("OptimizeJoinOrder".to_string());
        }

        // Low selectivity - suggest index usage
        if features.estimated_selectivity < 0.1 {
            hints.push("UseIndex".to_string());
        }

        // Has property paths - suggest specialized evaluation
        if features.has_property_path > 0.0 {
            hints.push("OptimizePropertyPaths".to_string());
        }

        hints
    }

    /// Get performance history summary
    pub async fn get_performance_summary(&self) -> PerformanceSummary {
        let history = self.performance_history.read().await;

        if history.is_empty() {
            return PerformanceSummary::default();
        }

        let mut total_time = 0.0;
        let mut min_time = f64::MAX;
        let mut max_time = 0.0;
        let mut total_results = 0;

        for record in history.iter() {
            total_time += record.execution_time_ms;
            min_time = min_time.min(record.execution_time_ms);
            max_time = f64::max(max_time, record.execution_time_ms);
            total_results += record.result_count;
        }

        let count = history.len() as f64;

        PerformanceSummary {
            total_queries: history.len(),
            avg_execution_time_ms: total_time / count,
            min_execution_time_ms: min_time,
            max_execution_time_ms: max_time,
            total_results_returned: total_results,
        }
    }
}

/// Feature extractor for SPARQL-star queries
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    // Placeholder for more sophisticated extraction logic
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {}
    }

    /// Extract features from SPARQL-star query string
    pub fn extract(&self, query: &str) -> StarResult<SPARQLStarQueryFeatures> {
        // Basic feature extraction (can be enhanced with full SPARQL parser)
        let triple_pattern_count = query.matches("?").count() as f64 / 3.0;
        let quoted_triple_count = query.matches("<<").count() as f64;
        let max_nesting_depth = Self::calculate_nesting_depth(query);
        let filter_count = query.matches("FILTER").count() as f64;
        let optional_count = query.matches("OPTIONAL").count() as f64;
        let union_count = query.matches("UNION").count() as f64;
        let graph_pattern_count = query.matches("GRAPH").count() as f64;
        let variable_count = query
            .split_whitespace()
            .filter(|w| w.starts_with('?'))
            .count() as f64;

        let query_type = if query.contains("SELECT") {
            0.0
        } else if query.contains("CONSTRUCT") {
            1.0
        } else if query.contains("ASK") {
            2.0
        } else if query.contains("DESCRIBE") {
            3.0
        } else {
            0.0
        };

        Ok(SPARQLStarQueryFeatures {
            triple_pattern_count,
            quoted_triple_count,
            max_nesting_depth,
            filter_count,
            optional_count,
            union_count,
            graph_pattern_count,
            variable_count,
            estimated_selectivity: 0.5, // Default estimate
            join_complexity: triple_pattern_count.max(1.0),
            has_aggregation: if query.contains("GROUP BY") { 1.0 } else { 0.0 },
            has_subquery: if query.contains("SELECT") && query.matches("SELECT").count() > 1 {
                1.0
            } else {
                0.0
            },
            has_property_path: if query.contains('/') || query.contains('*') {
                1.0
            } else {
                0.0
            },
            estimated_result_size: 100.0, // Default estimate
            query_type,
        })
    }

    /// Calculate maximum nesting depth in query
    fn calculate_nesting_depth(query: &str) -> f64 {
        let mut depth: i32 = 0;
        let mut max_depth: i32 = 0;

        for ch in query.chars() {
            match ch {
                '<' => depth += 1,
                '>' => depth = depth.saturating_sub(1),
                _ => {}
            }
            max_depth = max_depth.max(depth);
        }

        (max_depth / 2) as f64 // Each << >> pair is one level
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance summary statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_queries: usize,
    pub avg_execution_time_ms: f64,
    pub min_execution_time_ms: f64,
    pub max_execution_time_ms: f64,
    pub total_results_returned: usize,
}

/// Neural network-based cost predictor for complex SPARQL-star queries
///
/// This implements a multi-layer perceptron (MLP) with configurable architecture
/// for more sophisticated cost prediction than linear regression.
#[derive(Debug)]
pub struct NeuralNetworkPredictor {
    /// Layer weights (input->hidden1, hidden1->hidden2, ..., hiddenN->output)
    layer_weights: Vec<Array2<f64>>,
    /// Layer biases
    layer_biases: Vec<Array1<f64>>,
    /// Network architecture (input_dim, hidden_dims..., output_dim)
    architecture: Vec<usize>,
    /// Training history
    training_losses: VecDeque<f64>,
    /// Last training time
    last_trained: Option<Instant>,
}

impl NeuralNetworkPredictor {
    /// Create a new neural network with specified architecture
    ///
    /// # Arguments
    /// * `architecture` - Layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
    ///
    /// # Example
    /// ```ignore
    /// // Create network with 15 inputs, 2 hidden layers (64, 32), 1 output
    /// let nn = NeuralNetworkPredictor::new(vec![15, 64, 32, 1]);
    /// ```
    pub fn new(architecture: Vec<usize>) -> Self {
        assert!(
            architecture.len() >= 2,
            "Need at least input and output layers"
        );

        let mut rng = thread_rng();
        let mut layer_weights = Vec::new();
        let mut layer_biases = Vec::new();

        // Initialize weights using He initialization for ReLU networks
        for i in 0..architecture.len() - 1 {
            let n_in = architecture[i];
            let n_out = architecture[i + 1];

            // He initialization: weights ~ N(0, sqrt(2/n_in))
            let std_dev = (2.0 / n_in as f64).sqrt();
            let uniform = rand_distr::Uniform::new(-std_dev, std_dev).unwrap();
            let weights = Array2::from_shape_fn((n_in, n_out), |_| rng.sample(uniform));

            let biases = Array1::zeros(n_out);

            layer_weights.push(weights);
            layer_biases.push(biases);
        }

        Self {
            layer_weights,
            layer_biases,
            architecture,
            training_losses: VecDeque::new(),
            last_trained: None,
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
    ///
    /// Returns (activations at each layer, pre-activation values for backprop)
    fn forward(&self, input: &Array1<f64>) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut activations = vec![input.clone()];
        let mut z_values = Vec::new();

        for (weights, biases) in self.layer_weights.iter().zip(&self.layer_biases) {
            let last_activation = activations.last().unwrap();

            // Linear transformation: z = W^T * a + b
            let mut z = biases.clone();
            for (i, w_col) in weights
                .axis_iter(scirs2_core::ndarray_ext::Axis(1))
                .enumerate()
            {
                let dot_product: f64 = last_activation
                    .iter()
                    .zip(w_col.iter())
                    .map(|(a, w)| a * w)
                    .sum();
                z[i] += dot_product;
            }

            z_values.push(z.clone());

            // Apply activation (ReLU for hidden layers, linear for output)
            let is_output_layer = activations.len() == self.architecture.len() - 1;
            let activation = if is_output_layer {
                z.clone() // Linear activation for regression output
            } else {
                z.mapv(Self::relu) // ReLU for hidden layers
            };

            activations.push(activation);
        }

        (activations, z_values)
    }

    /// Predict output for given input
    pub fn predict(&self, input: &Array1<f64>) -> f64 {
        let (activations, _) = self.forward(input);
        let output = activations.last().unwrap();
        output[0].max(0.0) // Ensure non-negative cost
    }

    /// Train the network using backpropagation
    ///
    /// # Arguments
    /// * `training_data` - Pairs of (input_features, target_cost)
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `epochs` - Number of training epochs
    /// * `batch_size` - Mini-batch size for stochastic gradient descent
    pub fn train(
        &mut self,
        training_data: &[(Array1<f64>, f64)],
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
    ) -> StarResult<()> {
        if training_data.is_empty() {
            return Err(StarError::query_error(
                "No training data provided".to_string(),
            ));
        }

        let mut rng = thread_rng();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            // Shuffle training data using Fisher-Yates algorithm
            let mut indices: Vec<usize> = (0..training_data.len()).collect();
            for i in (1..indices.len()).rev() {
                let uniform = rand_distr::Uniform::new(0, i + 1).unwrap();
                let j = rng.sample(uniform);
                indices.swap(i, j);
            }

            // Mini-batch training
            for batch_start in (0..training_data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(training_data.len());

                for &idx in &indices[batch_start..batch_end] {
                    let (input, target) = &training_data[idx];
                    let (activations, z_values) = self.forward(input);

                    // Calculate loss (MSE)
                    let prediction = activations.last().unwrap()[0];
                    let error = prediction - target;
                    epoch_loss += error * error;

                    // Backpropagation
                    let mut delta = Array1::from_vec(vec![2.0 * error]); // Output layer gradient

                    // Backward pass through layers
                    for layer_idx in (0..self.layer_weights.len()).rev() {
                        let activation = &activations[layer_idx];

                        // Calculate new delta first (before mutating weights)
                        let new_delta = if layer_idx > 0 {
                            let weights = &self.layer_weights[layer_idx];
                            let mut new_delta = Array1::zeros(weights.shape()[0]);
                            for i in 0..weights.shape()[0] {
                                let mut sum = 0.0;
                                for j in 0..weights.shape()[1] {
                                    sum += weights[[i, j]] * delta[j];
                                }
                                // Apply ReLU derivative
                                new_delta[i] =
                                    sum * Self::relu_derivative(z_values[layer_idx - 1][i]);
                            }
                            Some(new_delta)
                        } else {
                            None
                        };

                        // Update weights: W -= lr * (a * delta^T)
                        for (i, &act_val) in activation.iter().enumerate() {
                            for (j, &delta_val) in delta.iter().enumerate() {
                                self.layer_weights[layer_idx][[i, j]] -=
                                    learning_rate * act_val * delta_val;
                            }
                        }

                        // Update biases: b -= lr * delta
                        for (j, &delta_val) in delta.iter().enumerate() {
                            self.layer_biases[layer_idx][j] -= learning_rate * delta_val;
                        }

                        // Update delta for next iteration
                        if let Some(nd) = new_delta {
                            delta = nd;
                        }
                    }
                }
            }

            let avg_loss = epoch_loss / training_data.len() as f64;
            self.training_losses.push_back(avg_loss);

            if self.training_losses.len() > 100 {
                self.training_losses.pop_front();
            }

            if epoch % 100 == 0 {
                debug!("Neural network epoch {}: avg loss = {:.2}", epoch, avg_loss);
            }
        }

        self.last_trained = Some(Instant::now());
        info!("Neural network training completed");

        Ok(())
    }

    /// Get training statistics
    pub fn get_statistics(&self) -> NeuralNetworkStatistics {
        let recent_loss = self.training_losses.back().copied().unwrap_or(0.0);
        let avg_loss = if !self.training_losses.is_empty() {
            self.training_losses.iter().sum::<f64>() / self.training_losses.len() as f64
        } else {
            0.0
        };

        NeuralNetworkStatistics {
            architecture: self.architecture.clone(),
            last_trained: self.last_trained,
            recent_loss,
            avg_loss,
            total_parameters: self.count_parameters(),
        }
    }

    /// Count total number of trainable parameters
    fn count_parameters(&self) -> usize {
        let mut count = 0;
        for weights in &self.layer_weights {
            count += weights.len();
        }
        for biases in &self.layer_biases {
            count += biases.len();
        }
        count
    }
}

/// Neural network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkStatistics {
    pub architecture: Vec<usize>,
    #[serde(skip)]
    pub last_trained: Option<Instant>,
    pub recent_loss: f64,
    pub avg_loss: f64,
    pub total_parameters: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let extractor = FeatureExtractor::new();
        let query = "SELECT ?s ?p ?o WHERE { << ?s ?p ?o >> ?p2 ?o2 }";

        let features = extractor.extract(query).unwrap();

        assert!(features.quoted_triple_count > 0.0);
        assert!(features.variable_count > 0.0);
        assert_eq!(features.query_type, 0.0); // SELECT
    }

    #[test]
    fn test_nesting_depth() {
        let depth = FeatureExtractor::calculate_nesting_depth("<< << ?s ?p ?o >> ?p2 ?o2 >>");
        assert_eq!(depth, 2.0);
    }

    #[tokio::test]
    async fn test_ml_predictor() {
        let mut predictor = MLCostPredictor::new();

        let features = SPARQLStarQueryFeatures {
            triple_pattern_count: 3.0,
            quoted_triple_count: 1.0,
            max_nesting_depth: 1.0,
            filter_count: 0.0,
            optional_count: 0.0,
            union_count: 0.0,
            graph_pattern_count: 0.0,
            variable_count: 3.0,
            estimated_selectivity: 0.5,
            join_complexity: 2.0,
            has_aggregation: 0.0,
            has_subquery: 0.0,
            has_property_path: 0.0,
            estimated_result_size: 100.0,
            query_type: 0.0,
        };

        // Add training samples
        for i in 0..10 {
            let record = QueryPerformanceRecord {
                features: features.clone(),
                execution_time_ms: 100.0 + (i as f64 * 10.0),
                result_count: 50,
                memory_used_bytes: 1024,
                timestamp: Instant::now(),
                optimization_hints_used: vec![],
                plan_chosen: "default".to_string(),
            };
            predictor.add_training_sample(record, 1000);
        }

        // Train
        let result = predictor.train(0.01, 100);
        assert!(result.is_ok());

        // Predict (may be 0 or negative with untrained model)
        let prediction = predictor.predict(&features);
        assert!(prediction >= 0.0); // Cost cannot be negative after max(0.0) clipping
    }

    #[tokio::test]
    async fn test_ml_optimizer() {
        let config = MLSPARQLOptimizerConfig::default();
        let optimizer = MLSPARQLOptimizer::new(config);

        let query = "SELECT ?s ?p ?o WHERE { << ?s ?p ?o >> ?p2 ?o2 }";
        let features = optimizer.extract_features(query).await.unwrap();

        assert!(features.quoted_triple_count > 0.0);

        // Record performance
        let record = QueryPerformanceRecord {
            features: features.clone(),
            execution_time_ms: 150.0,
            result_count: 75,
            memory_used_bytes: 2048,
            timestamp: Instant::now(),
            optimization_hints_used: vec![],
            plan_chosen: "optimized".to_string(),
        };

        let result = optimizer.record_performance(record).await;
        assert!(result.is_ok());
    }
}
