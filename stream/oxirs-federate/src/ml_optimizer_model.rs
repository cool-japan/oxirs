//! ML Optimizer — Model implementations
//!
//! Feature extraction, prediction, and training for LinearRegressionModel
//! and NeuralNetworkModel.

use super::ml_optimizer_types::{
    LinearRegressionModel, NeuralNetworkModel, QueryFeatures, TrainingSample,
};
use scirs2_core::random::Random;
use tracing::debug;

// ─── NeuralNetworkModel ───────────────────────────────────────────────────────

impl NeuralNetworkModel {
    /// Create new neural network model with scirs2-optimized initialization
    pub fn new(input_size: usize, hidden_size: usize, learning_rate: f64) -> Self {
        let mut rng = Random::default();

        let mut weights_input_hidden = Vec::with_capacity(hidden_size);
        for _ in 0..hidden_size {
            let mut layer_weights = Vec::with_capacity(input_size);
            // Improved Xavier/Glorot initialization
            let limit = (2.0 / (input_size + hidden_size) as f64).sqrt();
            for _ in 0..input_size {
                let weight = (rng.random_f64() * 2.0 - 1.0) * limit;
                layer_weights.push(weight);
            }
            weights_input_hidden.push(layer_weights);
        }

        let mut weights_hidden_output = Vec::with_capacity(hidden_size);
        let limit = (2.0 / (hidden_size + 1) as f64).sqrt();
        for _ in 0..hidden_size {
            let weight = (rng.random_f64() * 2.0 - 1.0) * limit;
            weights_hidden_output.push(weight);
        }

        Self {
            weights_input_hidden,
            bias_hidden: vec![0.01; hidden_size],
            weights_hidden_output,
            bias_output: 0.0,
            iterations: 0,
            accuracy: 0.0,
            learning_rate: learning_rate.min(0.01),
            last_trained: std::time::SystemTime::now(),
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, features: &[f64]) -> f64 {
        let mut hidden_activations = Vec::new();
        for (i, neuron_weights) in self.weights_input_hidden.iter().enumerate() {
            let mut activation = self.bias_hidden[i];
            for (j, &feature) in features.iter().enumerate() {
                if j < neuron_weights.len() {
                    activation += neuron_weights[j] * feature;
                }
            }
            hidden_activations.push(activation.max(0.0)); // ReLU
        }

        let mut output = self.bias_output;
        for (i, &hidden_val) in hidden_activations.iter().enumerate() {
            if i < self.weights_hidden_output.len() {
                output += self.weights_hidden_output[i] * hidden_val;
            }
        }
        output.max(0.0)
    }

    /// Train the neural network using backpropagation
    pub fn train(&mut self, samples: &[TrainingSample]) {
        if samples.is_empty() {
            return;
        }

        let epochs = 100;
        let mut prev_loss = f64::INFINITY;
        let convergence_threshold = 0.001;
        let mut convergence_count = 0;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for sample in samples {
                let features = self.extract_features(&sample.features);
                let target = sample.outcome.execution_time_ms;
                let prediction = self.forward(&features);
                let error = prediction - target;
                self.backpropagate(&features, error, target);
                total_loss += error * error;
            }

            total_loss /= samples.len() as f64;

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
        self.last_trained = std::time::SystemTime::now();
        self.accuracy = self.calculate_accuracy(samples);
    }

    /// Backpropagation algorithm
    fn backpropagate(&mut self, features: &[f64], output_error: f64, _target: f64) {
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

        let output_gradient = output_error.clamp(-1.0, 1.0);

        for (i, &hidden_val) in hidden_activations.iter().enumerate() {
            if i < self.weights_hidden_output.len() {
                let weight_update = self.learning_rate * output_gradient * hidden_val;
                let clipped_update = weight_update.clamp(-0.5, 0.5);
                self.weights_hidden_output[i] -= clipped_update;
            }
        }
        self.bias_output -= self.learning_rate * output_gradient;

        for (i, neuron_weights) in self.weights_input_hidden.iter_mut().enumerate() {
            if i < self.weights_hidden_output.len() {
                let hidden_gradient = if hidden_activations[i] > 0.0 {
                    output_gradient * self.weights_hidden_output[i]
                } else {
                    0.0
                };
                for (j, &feature) in features.iter().enumerate() {
                    if j < neuron_weights.len() {
                        let weight_update = self.learning_rate * hidden_gradient * feature;
                        let clipped_update = weight_update.clamp(-0.5, 0.5);
                        neuron_weights[j] -= clipped_update;
                    }
                }
                self.bias_hidden[i] -= self.learning_rate * hidden_gradient;
            }
        }
    }

    /// Extract features as vector
    pub(crate) fn extract_features(&self, features: &QueryFeatures) -> Vec<f64> {
        extract_feature_vec(features)
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
        1.0 - (mean_absolute_error / mean_actual).min(1.0)
    }

    /// Predict performance for features
    pub fn predict(&self, features: &QueryFeatures) -> f64 {
        let feature_vec = self.extract_features(features);
        self.forward(&feature_vec)
    }
}

// ─── LinearRegressionModel ────────────────────────────────────────────────────

impl LinearRegressionModel {
    /// Create new linear regression model
    pub fn new(feature_count: usize) -> Self {
        Self {
            weights: vec![0.0; feature_count],
            bias: 0.0,
            iterations: 0,
            accuracy: 0.0,
            last_trained: std::time::SystemTime::now(),
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
                let prediction = self.predict_value(&features);
                let error = prediction - target;

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
        self.last_trained = std::time::SystemTime::now();
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
        prediction.max(0.0)
    }

    /// Extract features as vector
    fn extract_features(&self, features: &QueryFeatures) -> Vec<f64> {
        extract_feature_vec(features)
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
        if mean_actual <= 0.0 {
            return 0.0;
        }
        let relative_error = mean_absolute_error / mean_actual;
        (1.0 - relative_error).clamp(0.0, 1.0)
    }
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

/// Shared feature extraction for both model types
pub(crate) fn extract_feature_vec(features: &QueryFeatures) -> Vec<f64> {
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
