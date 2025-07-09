//! Machine Learning-Driven Query Optimization Module
//!
//! This module implements advanced ML algorithms for federated query optimization,
//! including performance prediction, source selection learning, join order optimization,
//! caching strategy learning, and anomaly detection.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// ML model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Enable performance prediction
    pub enable_performance_prediction: bool,
    /// Enable source selection learning
    pub enable_source_selection_learning: bool,
    /// Enable join order optimization
    pub enable_join_order_optimization: bool,
    /// Enable caching strategy learning
    pub enable_caching_strategy_learning: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Model training interval
    pub training_interval: Duration,
    /// Feature history size
    pub feature_history_size: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Regularization parameter
    pub regularization: f64,
    /// Confidence threshold
    pub confidence_threshold: f64,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            enable_performance_prediction: true,
            enable_source_selection_learning: true,
            enable_join_order_optimization: true,
            enable_caching_strategy_learning: true,
            enable_anomaly_detection: true,
            training_interval: Duration::from_secs(3600), // 1 hour
            feature_history_size: 10000,
            learning_rate: 0.01,
            regularization: 0.001,
            confidence_threshold: 0.7,
        }
    }
}

/// Query features for ML training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    /// Number of triple patterns
    pub pattern_count: usize,
    /// Number of joins
    pub join_count: usize,
    /// Number of filters
    pub filter_count: usize,
    /// Query complexity score
    pub complexity_score: f64,
    /// Estimated selectivity
    pub selectivity: f64,
    /// Number of services involved
    pub service_count: usize,
    /// Average service latency
    pub avg_service_latency: f64,
    /// Data size estimate
    pub data_size_estimate: u64,
    /// Query depth (nested patterns)
    pub query_depth: usize,
    /// Has optional patterns
    pub has_optional: bool,
    /// Has union patterns
    pub has_union: bool,
    /// Has aggregation
    pub has_aggregation: bool,
    /// Variable count
    pub variable_count: usize,
}

/// Performance outcome for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOutcome {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Network I/O time
    pub network_io_ms: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Success rate
    pub success_rate: f64,
    /// Error count
    pub error_count: u32,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Training sample for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Input features
    pub features: QueryFeatures,
    /// Target outcome
    pub outcome: PerformanceOutcome,
    /// Service selection decisions
    pub service_selections: Vec<String>,
    /// Join order used
    pub join_order: Vec<String>,
    /// Caching decisions
    pub caching_decisions: HashMap<String, bool>,
    /// Query identifier
    pub query_id: String,
}

/// Source selection prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSelectionPrediction {
    /// Recommended services
    pub recommended_services: Vec<String>,
    /// Confidence scores for each service
    pub confidence_scores: HashMap<String, f64>,
    /// Expected performance
    pub expected_performance: PerformanceOutcome,
    /// Alternative options
    pub alternatives: Vec<SourceAlternative>,
}

/// Alternative source selection option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAlternative {
    /// Service IDs
    pub services: Vec<String>,
    /// Expected performance
    pub expected_performance: PerformanceOutcome,
    /// Confidence score
    pub confidence: f64,
    /// Risk assessment
    pub risk_score: f64,
}

/// Join order optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinOrderOptimization {
    /// Recommended join order
    pub recommended_order: Vec<String>,
    /// Expected cost
    pub expected_cost: f64,
    /// Alternative orders
    pub alternatives: Vec<JoinOrderAlternative>,
    /// Optimization confidence
    pub confidence: f64,
}

/// Alternative join order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinOrderAlternative {
    /// Join order
    pub order: Vec<String>,
    /// Expected cost
    pub cost: f64,
    /// Risk score
    pub risk: f64,
}

/// Caching strategy recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingStrategy {
    /// Items to cache
    pub cache_items: HashMap<String, CacheRecommendation>,
    /// Cache eviction order
    pub eviction_order: Vec<String>,
    /// Expected cache hit rate
    pub expected_hit_rate: f64,
    /// Memory requirements
    pub memory_requirements: u64,
}

/// Cache recommendation for specific item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheRecommendation {
    /// Should cache this item
    pub should_cache: bool,
    /// Priority score
    pub priority: f64,
    /// Expected benefit
    pub expected_benefit: f64,
    /// TTL recommendation
    pub ttl_seconds: u64,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Is anomalous
    pub is_anomalous: bool,
    /// Anomaly score (0.0 to 1.0)
    pub anomaly_score: f64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Confidence in detection
    pub confidence: f64,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Performance degradation
    PerformanceDegradation,
    /// Unusual resource usage
    ResourceAnomaly,
    /// Service behavior anomaly
    ServiceAnomaly,
    /// Pattern anomaly
    PatternAnomaly,
    /// Data quality issue
    DataQualityIssue,
    /// Security concern
    SecurityAnomaly,
}

/// Linear regression model for performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionModel {
    /// Feature weights
    pub weights: Vec<f64>,
    /// Bias term
    pub bias: f64,
    /// Training iterations
    pub iterations: u32,
    /// Model accuracy
    pub accuracy: f64,
    /// Last training time
    pub last_trained: SystemTime,
}

/// Neural network model for advanced performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkModel {
    /// Hidden layer weights (input to hidden)
    pub weights_input_hidden: Vec<Vec<f64>>,
    /// Hidden layer biases
    pub bias_hidden: Vec<f64>,
    /// Output layer weights (hidden to output)
    pub weights_hidden_output: Vec<f64>,
    /// Output bias
    pub bias_output: f64,
    /// Training iterations
    pub iterations: u32,
    /// Model accuracy
    pub accuracy: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Last training time
    pub last_trained: SystemTime,
}

impl NeuralNetworkModel {
    /// Create new neural network model
    pub fn new(input_size: usize, hidden_size: usize, learning_rate: f64) -> Self {
        let mut weights_input_hidden = Vec::new();
        for _ in 0..hidden_size {
            let mut layer_weights = Vec::new();
            for _ in 0..input_size {
                // Improved Xavier/Glorot initialization with better stability
                let limit = (2.0 / (input_size + hidden_size) as f64).sqrt();
                let weight = (rand::random::<f64>() * 2.0 - 1.0) * limit;
                layer_weights.push(weight);
            }
            weights_input_hidden.push(layer_weights);
        }

        let mut weights_hidden_output = Vec::new();
        for _ in 0..hidden_size {
            // Better initialization for output layer
            let limit = (2.0 / (hidden_size + 1) as f64).sqrt();
            let weight = (rand::random::<f64>() * 2.0 - 1.0) * limit;
            weights_hidden_output.push(weight);
        }

        Self {
            weights_input_hidden,
            bias_hidden: vec![0.01; hidden_size], // Small positive bias to avoid dead neurons
            weights_hidden_output,
            bias_output: 0.0,
            iterations: 0,
            accuracy: 0.0,
            learning_rate: learning_rate.min(0.01), // Cap learning rate for stability
            last_trained: SystemTime::now(),
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, features: &[f64]) -> f64 {
        // Input to hidden layer
        let mut hidden_activations = Vec::new();
        for (i, neuron_weights) in self.weights_input_hidden.iter().enumerate() {
            let mut activation = self.bias_hidden[i];
            for (j, &feature) in features.iter().enumerate() {
                if j < neuron_weights.len() {
                    activation += neuron_weights[j] * feature;
                }
            }
            // ReLU activation
            hidden_activations.push(activation.max(0.0));
        }

        // Hidden to output layer
        let mut output = self.bias_output;
        for (i, &hidden_val) in hidden_activations.iter().enumerate() {
            if i < self.weights_hidden_output.len() {
                output += self.weights_hidden_output[i] * hidden_val;
            }
        }

        output.max(0.0) // Ensure non-negative prediction
    }

    /// Train the neural network using backpropagation
    pub fn train(&mut self, samples: &[TrainingSample]) {
        if samples.is_empty() {
            return;
        }

        let epochs = 100; // Reduced epochs for stability
        let mut prev_loss = f64::INFINITY;
        let convergence_threshold = 0.001;
        let mut convergence_count = 0;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for sample in samples {
                let features = self.extract_features(&sample.features);
                let target = sample.outcome.execution_time_ms;

                // Forward pass
                let prediction = self.forward(&features);
                let error = prediction - target;

                // Backward pass with gradient clipping
                self.backpropagate(&features, error, target);

                total_loss += error * error;
            }

            total_loss /= samples.len() as f64;

            // Early stopping for convergence
            if (prev_loss - total_loss).abs() < convergence_threshold {
                convergence_count += 1;
                if convergence_count >= 5 {
                    debug!("NN training converged at epoch {}", epoch);
                    break;
                }
            } else {
                convergence_count = 0;
            }
            prev_loss = total_loss;

            if epoch % 20 == 0 {
                debug!("NN training epoch {}: loss = {:.2}", epoch, total_loss);
            }
        }

        self.iterations += epochs;
        self.last_trained = SystemTime::now();
        self.accuracy = self.calculate_accuracy(samples);
    }

    /// Backpropagation algorithm
    fn backpropagate(&mut self, features: &[f64], output_error: f64, _target: f64) {
        // Forward pass to get hidden activations
        let mut hidden_activations = Vec::new();
        for (i, neuron_weights) in self.weights_input_hidden.iter().enumerate() {
            let mut activation = self.bias_hidden[i];
            for (j, &feature) in features.iter().enumerate() {
                if j < neuron_weights.len() {
                    activation += neuron_weights[j] * feature;
                }
            }
            hidden_activations.push(activation.max(0.0));
        }

        // Output layer gradients with clipping
        let output_gradient = output_error.max(-1.0).min(1.0); // Gradient clipping

        // Update output layer weights
        for (i, &hidden_val) in hidden_activations.iter().enumerate() {
            if i < self.weights_hidden_output.len() {
                let weight_update = self.learning_rate * output_gradient * hidden_val;
                let clipped_update = weight_update.max(-0.5).min(0.5); // Clip weight updates
                self.weights_hidden_output[i] -= clipped_update;
            }
        }
        self.bias_output -= self.learning_rate * output_gradient;

        // Hidden layer gradients
        for (i, neuron_weights) in self.weights_input_hidden.iter_mut().enumerate() {
            if i < self.weights_hidden_output.len() {
                let hidden_gradient = if hidden_activations[i] > 0.0 {
                    output_gradient * self.weights_hidden_output[i]
                } else {
                    0.0 // ReLU derivative
                };

                // Update hidden layer weights with gradient clipping
                for (j, &feature) in features.iter().enumerate() {
                    if j < neuron_weights.len() {
                        let weight_update = self.learning_rate * hidden_gradient * feature;
                        let clipped_update = weight_update.max(-0.5).min(0.5); // Clip weight updates
                        neuron_weights[j] -= clipped_update;
                    }
                }
                self.bias_hidden[i] -= self.learning_rate * hidden_gradient;
            }
        }
    }

    /// Extract features as vector (same as LinearRegressionModel)
    fn extract_features(&self, features: &QueryFeatures) -> Vec<f64> {
        vec![
            features.pattern_count as f64,
            features.join_count as f64,
            features.filter_count as f64,
            features.complexity_score,
            features.selectivity,
            features.service_count as f64,
            features.avg_service_latency,
            (features.data_size_estimate as f64).log10(),
            features.query_depth as f64,
            if features.has_optional { 1.0 } else { 0.0 },
            if features.has_union { 1.0 } else { 0.0 },
            if features.has_aggregation { 1.0 } else { 0.0 },
            features.variable_count as f64,
        ]
    }

    /// Calculate model accuracy
    fn calculate_accuracy(&self, samples: &[TrainingSample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut total_error = 0.0;
        let mut total_actual = 0.0;

        for sample in samples {
            let features = self.extract_features(&sample.features);
            let prediction = self.forward(&features);
            let actual = sample.outcome.execution_time_ms;

            total_error += (prediction - actual).abs();
            total_actual += actual;
        }

        let mean_absolute_error = total_error / samples.len() as f64;
        let mean_actual = total_actual / samples.len() as f64;

        // Accuracy as 1 - normalized MAE
        1.0 - (mean_absolute_error / mean_actual).min(1.0)
    }

    /// Predict performance for features
    pub fn predict(&self, features: &QueryFeatures) -> f64 {
        let feature_vec = self.extract_features(features);
        self.forward(&feature_vec)
    }
}

impl LinearRegressionModel {
    /// Create new linear regression model
    pub fn new(feature_count: usize) -> Self {
        Self {
            weights: vec![0.0; feature_count],
            bias: 0.0,
            iterations: 0,
            accuracy: 0.0,
            last_trained: SystemTime::now(),
        }
    }

    /// Train model with samples
    pub fn train(&mut self, samples: &[TrainingSample], learning_rate: f64, regularization: f64) {
        if samples.is_empty() {
            return;
        }

        let epochs = 100;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for sample in samples {
                let features = self.extract_features(&sample.features);
                let target = sample.outcome.execution_time_ms;

                // Forward pass
                let prediction = self.predict_value(&features);
                let error = prediction - target;

                // Backward pass
                for i in 0..self.weights.len() {
                    if i < features.len() {
                        let gradient = error * features[i] + regularization * self.weights[i];
                        self.weights[i] -= learning_rate * gradient;
                    }
                }

                self.bias -= learning_rate * error;
                total_loss += error * error;
            }

            total_loss /= samples.len() as f64;

            if epoch % 10 == 0 {
                debug!("Training epoch {}: loss = {:.2}", epoch, total_loss);
            }
        }

        self.iterations += epochs;
        self.last_trained = SystemTime::now();

        // Calculate accuracy
        self.accuracy = self.calculate_accuracy(samples);
    }

    /// Predict performance for features
    pub fn predict(&self, features: &QueryFeatures) -> f64 {
        let feature_vec = self.extract_features(features);
        self.predict_value(&feature_vec)
    }

    /// Predict value from feature vector
    fn predict_value(&self, features: &[f64]) -> f64 {
        let mut prediction = self.bias;

        for (i, &weight) in self.weights.iter().enumerate() {
            if i < features.len() {
                prediction += weight * features[i];
            }
        }

        prediction.max(0.0) // Ensure non-negative prediction
    }

    /// Extract features as vector
    fn extract_features(&self, features: &QueryFeatures) -> Vec<f64> {
        vec![
            features.pattern_count as f64,
            features.join_count as f64,
            features.filter_count as f64,
            features.complexity_score,
            features.selectivity,
            features.service_count as f64,
            features.avg_service_latency,
            (features.data_size_estimate as f64).log10(),
            features.query_depth as f64,
            if features.has_optional { 1.0 } else { 0.0 },
            if features.has_union { 1.0 } else { 0.0 },
            if features.has_aggregation { 1.0 } else { 0.0 },
            features.variable_count as f64,
        ]
    }

    /// Calculate model accuracy
    fn calculate_accuracy(&self, samples: &[TrainingSample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut total_error = 0.0;
        let mut total_actual = 0.0;

        for sample in samples {
            let prediction = self.predict(&sample.features);
            let actual = sample.outcome.execution_time_ms;

            total_error += (prediction - actual).abs();
            total_actual += actual;
        }

        let mean_absolute_error = total_error / samples.len() as f64;
        let mean_actual = total_actual / samples.len() as f64;

        // Improved accuracy calculation with better bounds checking
        if mean_actual <= 0.0 {
            return 0.0;
        }

        let relative_error = mean_absolute_error / mean_actual;
        // Use exponential decay for accuracy to provide more realistic scores
        (1.0 - relative_error).max(0.0).min(1.0)
    }
}

/// ML-driven query optimizer
#[derive(Clone)]
pub struct MLOptimizer {
    /// Configuration
    config: MLConfig,
    /// Linear regression performance model
    linear_model: Arc<RwLock<LinearRegressionModel>>,
    /// Neural network performance model
    neural_model: Arc<RwLock<NeuralNetworkModel>>,
    /// Source selection model
    source_selection_model: Arc<RwLock<HashMap<String, f64>>>,
    /// Join order optimization model
    join_order_model: Arc<RwLock<HashMap<String, f64>>>,
    /// Caching strategy model
    caching_model: Arc<RwLock<HashMap<String, f64>>>,
    /// Training samples
    training_samples: Arc<RwLock<VecDeque<TrainingSample>>>,
    /// Anomaly detection baseline
    anomaly_baseline: Arc<RwLock<HashMap<String, f64>>>,
    /// Model statistics
    statistics: Arc<RwLock<MLStatistics>>,
}

/// ML optimizer statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MLStatistics {
    /// Total predictions made
    pub total_predictions: u64,
    /// Successful predictions
    pub successful_predictions: u64,
    /// Training samples collected
    pub training_samples_count: u64,
    /// Model accuracy
    pub model_accuracy: f64,
    /// Last training time
    pub last_training: Option<SystemTime>,
    /// Anomalies detected
    pub anomalies_detected: u64,
    /// Cache hit improvement
    pub cache_hit_improvement: f64,
}

impl Default for MLOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MLOptimizer {
    /// Create new ML optimizer
    pub fn new() -> Self {
        Self::with_config(MLConfig::default())
    }

    /// Create ML optimizer with configuration
    pub fn with_config(config: MLConfig) -> Self {
        let feature_count = 13; // Number of features in QueryFeatures
        let hidden_size = 16; // Neural network hidden layer size

        Self {
            config: config.clone(),
            linear_model: Arc::new(RwLock::new(LinearRegressionModel::new(feature_count))),
            neural_model: Arc::new(RwLock::new(NeuralNetworkModel::new(
                feature_count,
                hidden_size,
                config.learning_rate,
            ))),
            source_selection_model: Arc::new(RwLock::new(HashMap::new())),
            join_order_model: Arc::new(RwLock::new(HashMap::new())),
            caching_model: Arc::new(RwLock::new(HashMap::new())),
            training_samples: Arc::new(RwLock::new(VecDeque::new())),
            anomaly_baseline: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(MLStatistics::default())),
        }
    }

    /// Predict query performance using ensemble of linear and neural network models
    pub async fn predict_performance(&self, features: &QueryFeatures) -> Result<f64> {
        if !self.config.enable_performance_prediction {
            return Ok(0.0);
        }

        // Get predictions from both models
        let linear_prediction = {
            let model = self.linear_model.read().await;
            model.predict(features)
        };

        let neural_prediction = {
            let model = self.neural_model.read().await;
            model.predict(features)
        };

        // Ensemble prediction: weighted average based on model accuracies
        let linear_accuracy = {
            let model = self.linear_model.read().await;
            model.accuracy
        };

        let neural_accuracy = {
            let model = self.neural_model.read().await;
            model.accuracy
        };

        let ensemble_prediction = if linear_accuracy + neural_accuracy > 0.0 {
            let linear_weight = linear_accuracy / (linear_accuracy + neural_accuracy);
            let neural_weight = neural_accuracy / (linear_accuracy + neural_accuracy);
            linear_prediction * linear_weight + neural_prediction * neural_weight
        } else {
            // If no accuracy data, use simple average
            (linear_prediction + neural_prediction) / 2.0
        };

        // Provide fallback prediction if models haven't been trained or return 0.0
        let final_prediction = if ensemble_prediction <= 0.0 {
            // Fallback: estimate based on query features using heuristics
            let base_time = 50.0; // Base execution time in ms
            let pattern_complexity = features.pattern_count as f64 * 20.0;
            let join_complexity = features.join_count as f64 * 100.0;
            let filter_complexity = features.filter_count as f64 * 10.0;
            let service_latency =
                features.avg_service_latency * (features.service_count as f64).max(1.0);

            base_time + pattern_complexity + join_complexity + filter_complexity + service_latency
        } else {
            ensemble_prediction
        };

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_predictions += 1;
            stats.model_accuracy = (linear_accuracy + neural_accuracy) / 2.0;
        }

        debug!("Performance prediction: {:.2}ms (linear: {:.2}, neural: {:.2}, final: {:.2}) for query with {} patterns", 
               ensemble_prediction, linear_prediction, neural_prediction, final_prediction, features.pattern_count);

        Ok(final_prediction)
    }

    /// Recommend source selection
    pub async fn recommend_source_selection(
        &self,
        features: &QueryFeatures,
        available_services: &[String],
    ) -> Result<SourceSelectionPrediction> {
        if !self.config.enable_source_selection_learning {
            // Fallback to simple selection
            return Ok(SourceSelectionPrediction {
                recommended_services: available_services.to_vec(),
                confidence_scores: HashMap::new(),
                expected_performance: PerformanceOutcome::default(),
                alternatives: vec![],
            });
        }

        let model = self.source_selection_model.read().await;
        let mut service_scores = HashMap::new();

        // Score each service based on learned patterns
        for service in available_services {
            // Use confidence threshold as default score for new services to ensure they get considered
            let default_score = self.config.confidence_threshold + 0.1;
            let score = model.get(service).copied().unwrap_or(default_score);

            // Adjust score based on query features
            let adjusted_score = self.adjust_service_score(service, features, score).await;
            service_scores.insert(service.clone(), adjusted_score);
        }

        // Select top services above confidence threshold
        let mut recommended: Vec<_> = service_scores
            .iter()
            .filter(|&(_, &score)| score > self.config.confidence_threshold)
            .map(|(service, &score)| (service.clone(), score))
            .collect();

        recommended.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut recommended_services: Vec<String> = recommended
            .iter()
            .take(3) // Top 3 services
            .map(|(service, _)| service.clone())
            .collect();

        // If no services meet the threshold, fall back to top services by score
        if recommended_services.is_empty() {
            let mut all_services: Vec<_> = service_scores
                .iter()
                .map(|(service, &score)| (service.clone(), score))
                .collect();
            all_services.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            recommended_services = all_services
                .iter()
                .take(3)
                .map(|(service, _)| service.clone())
                .collect();
        }

        // Generate alternatives
        let alternatives = self
            .generate_source_alternatives(available_services, &service_scores)
            .await;

        // Predict expected performance
        let expected_performance = self
            .predict_service_performance(&recommended_services, features)
            .await;

        Ok(SourceSelectionPrediction {
            recommended_services,
            confidence_scores: service_scores,
            expected_performance,
            alternatives,
        })
    }

    /// Optimize join order
    pub async fn optimize_join_order(
        &self,
        join_patterns: &[String],
        features: &QueryFeatures,
    ) -> Result<JoinOrderOptimization> {
        if !self.config.enable_join_order_optimization || join_patterns.is_empty() {
            return Ok(JoinOrderOptimization {
                recommended_order: (0..join_patterns.len()).map(|i| i.to_string()).collect(),
                expected_cost: 1.0,
                alternatives: vec![],
                confidence: 0.5,
            });
        }

        let model = self.join_order_model.read().await;

        // Generate permutations and score them
        let mut order_scores = Vec::new();
        let permutations = if join_patterns.len() <= 6 {
            self.generate_permutations(join_patterns)
        } else {
            // For larger sets, use heuristic approach
            self.generate_heuristic_orders(join_patterns)
        };

        for order in permutations {
            let cost = self.calculate_join_cost(&order, features, &model).await;
            order_scores.push((order, cost));
        }

        // Sort by cost (lower is better)
        order_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let best_order = order_scores
            .first()
            .map(|(order, cost)| (order.clone(), *cost))
            .unwrap_or_else(|| (join_patterns.to_vec(), 1.0));

        // Convert pattern strings back to indices
        let pattern_to_index: std::collections::HashMap<String, usize> = join_patterns
            .iter()
            .enumerate()
            .map(|(i, pattern)| (pattern.clone(), i))
            .collect();

        let recommended_indices: Vec<String> = best_order
            .0
            .iter()
            .map(|pattern| {
                pattern_to_index
                    .get(pattern)
                    .map(|&i| i.to_string())
                    .unwrap_or_else(|| "0".to_string())
            })
            .collect();

        // Generate alternatives
        let alternatives: Vec<JoinOrderAlternative> = order_scores
            .iter()
            .skip(1)
            .take(3)
            .map(|(order, cost)| JoinOrderAlternative {
                order: order
                    .iter()
                    .map(|pattern| {
                        pattern_to_index
                            .get(pattern)
                            .map(|&i| i.to_string())
                            .unwrap_or_else(|| "0".to_string())
                    })
                    .collect(),
                cost: *cost,
                risk: self.calculate_risk_score(*cost, best_order.1),
            })
            .collect();

        Ok(JoinOrderOptimization {
            recommended_order: recommended_indices,
            expected_cost: best_order.1,
            alternatives,
            confidence: self.calculate_join_confidence(&order_scores),
        })
    }

    /// Recommend caching strategy
    pub async fn recommend_caching_strategy(
        &self,
        query_patterns: &[String],
        features: &QueryFeatures,
    ) -> Result<CachingStrategy> {
        if !self.config.enable_caching_strategy_learning {
            return Ok(CachingStrategy {
                cache_items: HashMap::new(),
                eviction_order: vec![],
                expected_hit_rate: 0.5,
                memory_requirements: 0,
            });
        }

        let model = self.caching_model.read().await;
        let mut cache_items = HashMap::new();
        let mut total_memory = 0u64;

        for pattern in query_patterns {
            let cache_score = model.get(pattern).copied().unwrap_or(0.3);

            // Adjust score based on query characteristics
            let adjusted_score = self
                .adjust_cache_score(pattern, features, cache_score)
                .await;

            if adjusted_score > 0.5 {
                let estimated_size = self.estimate_cache_size(pattern, features);
                let ttl = self.calculate_optimal_ttl(pattern, adjusted_score);

                cache_items.insert(
                    pattern.clone(),
                    CacheRecommendation {
                        should_cache: true,
                        priority: adjusted_score,
                        expected_benefit: adjusted_score * 100.0, // Percentage improvement
                        ttl_seconds: ttl,
                    },
                );

                total_memory += estimated_size;
            }
        }

        // Generate eviction order based on priority
        let mut eviction_order: Vec<_> = cache_items
            .iter()
            .map(|(pattern, rec)| (pattern.clone(), rec.priority))
            .collect();
        eviction_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let eviction_order: Vec<String> = eviction_order.into_iter().map(|(p, _)| p).collect();

        // Estimate hit rate
        let expected_hit_rate = self.estimate_cache_hit_rate(&cache_items, features).await;

        Ok(CachingStrategy {
            cache_items,
            eviction_order,
            expected_hit_rate,
            memory_requirements: total_memory,
        })
    }

    /// Detect anomalies in query execution
    pub async fn detect_anomalies(
        &self,
        features: &QueryFeatures,
        outcome: &PerformanceOutcome,
    ) -> Result<AnomalyDetection> {
        if !self.config.enable_anomaly_detection {
            return Ok(AnomalyDetection {
                is_anomalous: false,
                anomaly_score: 0.0,
                anomaly_type: AnomalyType::PerformanceDegradation,
                confidence: 0.0,
                recommendations: vec![],
            });
        }

        let baseline = self.anomaly_baseline.read().await;

        // Calculate anomaly scores for different aspects
        let performance_score = self.calculate_performance_anomaly_score(outcome, &baseline);
        let resource_score = self.calculate_resource_anomaly_score(outcome, &baseline);
        let pattern_score = self.calculate_pattern_anomaly_score(features, &baseline);

        let max_score = performance_score.max(resource_score).max(pattern_score);
        let is_anomalous = max_score > 0.7;

        let anomaly_type = if performance_score == max_score {
            AnomalyType::PerformanceDegradation
        } else if resource_score == max_score {
            AnomalyType::ResourceAnomaly
        } else {
            AnomalyType::PatternAnomaly
        };

        let recommendations = self.generate_anomaly_recommendations(&anomaly_type, max_score);

        if is_anomalous {
            let mut stats = self.statistics.write().await;
            stats.anomalies_detected += 1;
        }

        Ok(AnomalyDetection {
            is_anomalous,
            anomaly_score: max_score,
            anomaly_type,
            confidence: max_score,
            recommendations,
        })
    }

    /// Add training sample
    pub async fn add_training_sample(&self, sample: TrainingSample) {
        let mut samples = self.training_samples.write().await;
        samples.push_back(sample);

        // Limit history size
        while samples.len() > self.config.feature_history_size {
            samples.pop_front();
        }

        let mut stats = self.statistics.write().await;
        stats.training_samples_count += 1;

        // Trigger retraining if enough samples accumulated
        if samples.len() % 100 == 0 {
            drop(samples);
            drop(stats);
            let _ = self.retrain_models().await;
        }
    }

    /// Retrain all models
    pub async fn retrain_models(&self) -> Result<()> {
        info!("Starting ML model retraining");

        let samples: Vec<TrainingSample> = {
            let samples_guard = self.training_samples.read().await;
            samples_guard.iter().cloned().collect()
        };

        if samples.is_empty() {
            warn!("No training samples available for retraining");
            return Ok(());
        }

        // Retrain both performance models
        {
            let mut linear_model = self.linear_model.write().await;
            linear_model.train(
                &samples,
                self.config.learning_rate,
                self.config.regularization,
            );
        }

        {
            let mut neural_model = self.neural_model.write().await;
            neural_model.train(&samples);
        }

        // Update source selection model
        self.update_source_selection_model(&samples).await;

        // Update join order model
        self.update_join_order_model(&samples).await;

        // Update caching model
        self.update_caching_model(&samples).await;

        // Update anomaly baseline
        self.update_anomaly_baseline(&samples).await;

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.last_training = Some(SystemTime::now());
            let linear_model = self.linear_model.read().await;
            let neural_model = self.neural_model.read().await;
            stats.model_accuracy = (linear_model.accuracy + neural_model.accuracy) / 2.0;
        }

        info!(
            "ML model retraining completed with {} samples",
            samples.len()
        );
        Ok(())
    }

    /// Get ML optimizer statistics
    pub async fn get_statistics(&self) -> MLStatistics {
        self.statistics.read().await.clone()
    }

    // Private helper methods

    async fn adjust_service_score(
        &self,
        _service: &str,
        features: &QueryFeatures,
        base_score: f64,
    ) -> f64 {
        // Adjust score based on query complexity (less aggressive reduction)
        let complexity_factor = 1.0 - (features.complexity_score / 20.0).min(0.2);
        let service_factor = 1.0 - (features.service_count as f64 / 20.0).min(0.15);

        let adjusted_score = base_score * complexity_factor * service_factor;

        // Ensure the adjusted score doesn't fall below a minimum threshold
        // for new services to still be considered
        adjusted_score.max(self.config.confidence_threshold + 0.01)
    }

    async fn generate_source_alternatives(
        &self,
        available_services: &[String],
        scores: &HashMap<String, f64>,
    ) -> Vec<SourceAlternative> {
        // Generate alternative service combinations
        let mut alternatives = Vec::new();

        for combination_size in 1..=3.min(available_services.len()) {
            if let Some(combination) =
                self.select_service_combination(available_services, scores, combination_size)
            {
                alternatives.push(SourceAlternative {
                    services: combination,
                    expected_performance: PerformanceOutcome::default(),
                    confidence: 0.7,
                    risk_score: 0.3,
                });
            }
        }

        alternatives
    }

    fn select_service_combination(
        &self,
        services: &[String],
        scores: &HashMap<String, f64>,
        size: usize,
    ) -> Option<Vec<String>> {
        if size > services.len() {
            return None;
        }

        let mut scored_services: Vec<_> = services
            .iter()
            .map(|s| (s.clone(), scores.get(s).copied().unwrap_or(0.0)))
            .collect();

        scored_services.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Some(
            scored_services
                .into_iter()
                .take(size)
                .map(|(s, _)| s)
                .collect(),
        )
    }

    async fn predict_service_performance(
        &self,
        _services: &[String],
        _features: &QueryFeatures,
    ) -> PerformanceOutcome {
        // Predict performance for given service combination
        PerformanceOutcome::default()
    }

    fn generate_permutations(&self, items: &[String]) -> Vec<Vec<String>> {
        if items.is_empty() {
            return vec![vec![]];
        }

        let mut permutations = Vec::new();

        for (i, item) in items.iter().enumerate() {
            let mut remaining = items.to_vec();
            remaining.remove(i);

            for mut perm in self.generate_permutations(&remaining) {
                perm.insert(0, item.clone());
                permutations.push(perm);
            }
        }

        permutations
    }

    fn generate_heuristic_orders(&self, items: &[String]) -> Vec<Vec<String>> {
        // For large sets, generate a few heuristic orders
        vec![
            items.to_vec(), // Original order
            {
                let mut reversed = items.to_vec();
                reversed.reverse();
                reversed
            }, // Reversed order
            {
                let mut random = items.to_vec();
                // Simple shuffle approximation
                let len = random.len();
                if len > 1 {
                    random.swap(0, len - 1);
                }
                random
            }, // "Random" order
        ]
    }

    async fn calculate_join_cost(
        &self,
        order: &[String],
        features: &QueryFeatures,
        model: &HashMap<String, f64>,
    ) -> f64 {
        // Calculate estimated cost for join order using more sophisticated model
        let mut cost = 1.0;

        // Base cost factors
        let pattern_cost = order.len() as f64 * 0.5;
        let selectivity_factor = (1.0 - features.selectivity).max(0.1);
        let complexity_factor = features.complexity_score + 1.0;

        // Position-based cost (later joins are more expensive)
        for (i, pattern) in order.iter().enumerate() {
            let position_cost = (i + 1) as f64 * 0.2;

            // Check if model has specific cost for this pattern
            let pattern_specific_cost = model.get(pattern).unwrap_or(&1.0);

            cost += position_cost * pattern_specific_cost * selectivity_factor;
        }

        // Apply complexity and service count factors
        cost *= complexity_factor;
        cost *= (features.service_count as f64).max(1.0);

        // Ensure reasonable bounds
        cost.max(0.1).min(100.0)
    }

    fn calculate_risk_score(&self, cost: f64, best_cost: f64) -> f64 {
        if best_cost == 0.0 {
            return 0.0;
        }
        ((cost - best_cost) / best_cost).min(1.0)
    }

    fn calculate_join_confidence(&self, scores: &[(Vec<String>, f64)]) -> f64 {
        if scores.len() < 2 {
            return 0.5;
        }

        let best_cost = scores[0].1;
        let second_cost = scores[1].1;

        if second_cost == 0.0 {
            return 1.0;
        }

        1.0 - (best_cost / second_cost).min(1.0)
    }

    async fn adjust_cache_score(
        &self,
        _pattern: &str,
        features: &QueryFeatures,
        base_score: f64,
    ) -> f64 {
        // Adjust based on query characteristics
        let frequency_factor = if features.pattern_count > 5 { 1.2 } else { 1.0 };
        let complexity_factor = if features.complexity_score > 5.0 {
            1.1
        } else {
            1.0
        };

        base_score * frequency_factor * complexity_factor
    }

    fn estimate_cache_size(&self, _pattern: &str, features: &QueryFeatures) -> u64 {
        // Estimate cache size based on pattern and query features
        (features.data_size_estimate / features.pattern_count as u64).max(1024)
    }

    fn calculate_optimal_ttl(&self, _pattern: &str, priority: f64) -> u64 {
        // Higher priority items get longer TTL
        (3600.0 * priority) as u64 // 1 hour base * priority
    }

    async fn estimate_cache_hit_rate(
        &self,
        cache_items: &HashMap<String, CacheRecommendation>,
        _features: &QueryFeatures,
    ) -> f64 {
        if cache_items.is_empty() {
            return 0.0;
        }

        let avg_priority: f64 =
            cache_items.values().map(|r| r.priority).sum::<f64>() / cache_items.len() as f64;
        avg_priority.min(0.9) // Cap at 90%
    }

    fn calculate_performance_anomaly_score(
        &self,
        outcome: &PerformanceOutcome,
        baseline: &HashMap<String, f64>,
    ) -> f64 {
        let baseline_time = baseline.get("execution_time").copied().unwrap_or(1000.0);
        let current_time = outcome.execution_time_ms;

        if baseline_time == 0.0 {
            return 0.0;
        }

        ((current_time - baseline_time) / baseline_time)
            .max(0.0)
            .min(1.0)
    }

    fn calculate_resource_anomaly_score(
        &self,
        outcome: &PerformanceOutcome,
        baseline: &HashMap<String, f64>,
    ) -> f64 {
        let baseline_memory = baseline
            .get("memory_usage")
            .copied()
            .unwrap_or(1024.0 * 1024.0);
        let current_memory = outcome.memory_usage_bytes as f64;

        if baseline_memory == 0.0 {
            return 0.0;
        }

        ((current_memory - baseline_memory) / baseline_memory)
            .max(0.0)
            .min(1.0)
    }

    fn calculate_pattern_anomaly_score(
        &self,
        _features: &QueryFeatures,
        _baseline: &HashMap<String, f64>,
    ) -> f64 {
        // Simplified pattern anomaly detection
        0.1
    }

    fn generate_anomaly_recommendations(
        &self,
        anomaly_type: &AnomalyType,
        score: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match anomaly_type {
            AnomalyType::PerformanceDegradation => {
                recommendations.push("Consider optimizing query patterns".to_string());
                if score > 0.8 {
                    recommendations.push("Review service selection strategy".to_string());
                }
            }
            AnomalyType::ResourceAnomaly => {
                recommendations.push("Monitor memory usage patterns".to_string());
                recommendations.push("Consider implementing result streaming".to_string());
            }
            AnomalyType::PatternAnomaly => {
                recommendations.push("Review query structure".to_string());
            }
            _ => {
                recommendations.push("Monitor system behavior".to_string());
            }
        }

        recommendations
    }

    async fn update_source_selection_model(&self, samples: &[TrainingSample]) {
        let mut model = self.source_selection_model.write().await;

        for sample in samples {
            for service in &sample.service_selections {
                let success_rate = sample.outcome.success_rate;
                let current_score = model.get(service).copied().unwrap_or(0.5);
                let new_score = current_score * 0.9 + success_rate * 0.1;
                model.insert(service.clone(), new_score);
            }
        }
    }

    async fn update_join_order_model(&self, samples: &[TrainingSample]) {
        let mut model = self.join_order_model.write().await;

        for sample in samples {
            for (i, pattern) in sample.join_order.iter().enumerate() {
                let position_score = 1.0 - (i as f64 / sample.join_order.len() as f64);
                let performance_factor = 1.0 / sample.outcome.execution_time_ms.max(1.0);
                let score = position_score * performance_factor * 1000.0;

                let current_score = model.get(pattern).copied().unwrap_or(0.5);
                let new_score = current_score * 0.9 + score * 0.1;
                model.insert(pattern.clone(), new_score);
            }
        }
    }

    async fn update_caching_model(&self, samples: &[TrainingSample]) {
        let mut model = self.caching_model.write().await;

        for sample in samples {
            for (item, &should_cache) in &sample.caching_decisions {
                let cache_benefit = if should_cache {
                    sample.outcome.cache_hit_rate
                } else {
                    1.0 - sample.outcome.cache_hit_rate
                };

                let current_score = model.get(item).copied().unwrap_or(0.5);
                let new_score = current_score * 0.9 + cache_benefit * 0.1;
                model.insert(item.clone(), new_score);
            }
        }
    }

    async fn update_anomaly_baseline(&self, samples: &[TrainingSample]) {
        let mut baseline = self.anomaly_baseline.write().await;

        if samples.is_empty() {
            return;
        }

        let avg_execution_time = samples
            .iter()
            .map(|s| s.outcome.execution_time_ms)
            .sum::<f64>()
            / samples.len() as f64;

        let avg_memory_usage = samples
            .iter()
            .map(|s| s.outcome.memory_usage_bytes as f64)
            .sum::<f64>()
            / samples.len() as f64;

        baseline.insert("execution_time".to_string(), avg_execution_time);
        baseline.insert("memory_usage".to_string(), avg_memory_usage);
    }
}

impl Default for PerformanceOutcome {
    fn default() -> Self {
        Self {
            execution_time_ms: 0.0,
            memory_usage_bytes: 0,
            network_io_ms: 0.0,
            cpu_usage_percent: 0.0,
            success_rate: 1.0,
            error_count: 0,
            cache_hit_rate: 0.0,
            timestamp: SystemTime::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ml_optimizer_creation() {
        let optimizer = MLOptimizer::new();
        let stats = optimizer.get_statistics().await;
        assert_eq!(stats.total_predictions, 0);
    }

    #[tokio::test]
    async fn test_performance_prediction() {
        let optimizer = MLOptimizer::new();
        let features = QueryFeatures {
            pattern_count: 5,
            join_count: 2,
            filter_count: 1,
            complexity_score: 3.0,
            selectivity: 0.5,
            service_count: 2,
            avg_service_latency: 100.0,
            data_size_estimate: 1024,
            query_depth: 2,
            has_optional: false,
            has_union: false,
            has_aggregation: false,
            variable_count: 3,
        };

        let prediction = optimizer.predict_performance(&features).await.unwrap();
        assert!(prediction >= 0.0);
    }

    #[tokio::test]
    async fn test_source_selection_recommendation() {
        let optimizer = MLOptimizer::new();
        let features = QueryFeatures {
            pattern_count: 3,
            join_count: 1,
            filter_count: 1,
            complexity_score: 2.0,
            selectivity: 0.8,
            service_count: 3,
            avg_service_latency: 50.0,
            data_size_estimate: 512,
            query_depth: 1,
            has_optional: false,
            has_union: false,
            has_aggregation: false,
            variable_count: 2,
        };

        let services = vec![
            "service1".to_string(),
            "service2".to_string(),
            "service3".to_string(),
        ];
        let recommendation = optimizer
            .recommend_source_selection(&features, &services)
            .await
            .unwrap();

        assert!(!recommendation.recommended_services.is_empty());
    }

    #[tokio::test]
    async fn test_anomaly_detection() {
        let optimizer = MLOptimizer::new();
        let features = QueryFeatures {
            pattern_count: 10,
            join_count: 5,
            filter_count: 3,
            complexity_score: 8.0,
            selectivity: 0.1,
            service_count: 5,
            avg_service_latency: 200.0,
            data_size_estimate: 10240,
            query_depth: 3,
            has_optional: true,
            has_union: true,
            has_aggregation: true,
            variable_count: 8,
        };

        let outcome = PerformanceOutcome {
            execution_time_ms: 5000.0, // Very high execution time
            memory_usage_bytes: 100 * 1024 * 1024,
            network_io_ms: 1000.0,
            cpu_usage_percent: 80.0,
            success_rate: 0.9,
            error_count: 1,
            cache_hit_rate: 0.2,
            timestamp: SystemTime::now(),
        };

        let detection = optimizer
            .detect_anomalies(&features, &outcome)
            .await
            .unwrap();
        assert!(detection.anomaly_score >= 0.0);
    }

    #[test]
    fn test_linear_regression_model() {
        let mut model = LinearRegressionModel::new(5);
        assert_eq!(model.weights.len(), 5);
        assert_eq!(model.bias, 0.0);
        assert_eq!(model.iterations, 0);
    }

    #[test]
    fn test_neural_network_model() {
        let model = NeuralNetworkModel::new(5, 3, 0.01);
        assert_eq!(model.weights_input_hidden.len(), 3); // 3 hidden neurons
        assert_eq!(model.weights_input_hidden[0].len(), 5); // 5 input features
        assert_eq!(model.weights_hidden_output.len(), 3); // 3 hidden to 1 output
        assert_eq!(model.bias_hidden.len(), 3);
        assert_eq!(model.iterations, 0);
        assert_eq!(model.learning_rate, 0.01);
    }

    #[tokio::test]
    async fn test_ensemble_prediction() {
        let optimizer = MLOptimizer::new();
        let features = QueryFeatures {
            pattern_count: 3,
            join_count: 1,
            filter_count: 1,
            complexity_score: 2.5,
            selectivity: 0.6,
            service_count: 2,
            avg_service_latency: 75.0,
            data_size_estimate: 2048,
            query_depth: 1,
            has_optional: false,
            has_union: false,
            has_aggregation: false,
            variable_count: 4,
        };

        let prediction = optimizer.predict_performance(&features).await.unwrap();
        assert!(prediction >= 0.0);

        // Test that we get consistent predictions
        let prediction2 = optimizer.predict_performance(&features).await.unwrap();
        assert_eq!(prediction, prediction2);
    }
}
