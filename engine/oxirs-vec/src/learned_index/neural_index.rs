//! Neural network-based learned index

use super::config::LearnedIndexConfig;
use super::training::{IndexTrainer, TrainingStats};
use super::types::{
    IndexStatistics, LearnedIndexError, LearnedIndexResult, PredictionBounds, TrainingExample,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Neural vector index using learned models
#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralVectorIndex {
    /// Configuration
    config: LearnedIndexConfig,

    /// Model weights (simplified representation)
    weights: Vec<Vec<f32>>,

    /// Bias terms
    biases: Vec<f32>,

    /// Sorted keys for binary search fallback
    sorted_keys: Vec<Vec<f32>>,

    /// Key to position mapping
    key_positions: HashMap<String, usize>,

    /// Error bounds statistics
    error_stats: ErrorStatistics,

    /// Performance statistics
    stats: IndexStatistics,

    /// Is model trained
    is_trained: bool,
}

#[derive(Clone, Serialize, Deserialize)]
struct ErrorStatistics {
    mean_error: f64,
    std_error: f64,
    max_error: usize,
}

impl NeuralVectorIndex {
    /// Create new neural index
    pub fn new(config: LearnedIndexConfig) -> LearnedIndexResult<Self> {
        config
            .validate()
            .map_err(|e| LearnedIndexError::InvalidConfiguration { message: e })?;

        Ok(Self {
            config,
            weights: Vec::new(),
            biases: Vec::new(),
            sorted_keys: Vec::new(),
            key_positions: HashMap::new(),
            error_stats: ErrorStatistics {
                mean_error: 0.0,
                std_error: 0.0,
                max_error: 0,
            },
            stats: IndexStatistics::new(),
            is_trained: false,
        })
    }

    /// Train the index on data
    pub fn train(&mut self, examples: Vec<TrainingExample>) -> LearnedIndexResult<TrainingStats> {
        if examples.len() < self.config.min_training_examples {
            return Err(LearnedIndexError::InsufficientData {
                min_required: self.config.min_training_examples,
                actual: examples.len(),
            });
        }

        tracing::info!("Training learned index on {} examples", examples.len());

        // Initialize trainer
        let trainer = IndexTrainer::new(self.config.training.clone());

        // Train model
        let training_stats = trainer.train(&mut self.weights, &mut self.biases, &examples)?;

        // Build sorted keys for fallback
        let mut sorted_examples = examples.clone();
        sorted_examples.sort_by(|a, b| {
            a.target_position
                .partial_cmp(&b.target_position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.sorted_keys = sorted_examples
            .iter()
            .map(|ex| ex.features.clone())
            .collect();

        // Build key-position mapping
        for (idx, example) in sorted_examples.iter().enumerate() {
            let key = self.key_to_string(&example.features);
            self.key_positions.insert(key, idx);
        }

        // Compute error statistics
        self.compute_error_statistics(&examples);

        self.is_trained = true;

        tracing::info!(
            "Training complete: loss={:.4}, accuracy={:.2}%",
            training_stats.final_loss,
            training_stats.final_accuracy * 100.0
        );

        Ok(training_stats)
    }

    /// Predict position for a key
    pub fn predict(&mut self, key: &[f32]) -> LearnedIndexResult<PredictionBounds> {
        if !self.is_trained {
            return Err(LearnedIndexError::ModelNotTrained);
        }

        let start = Instant::now();

        // Forward pass through neural network
        let prediction = self.forward(key);

        // Normalize to [0, 1] range
        let normalized_pred = Self::sigmoid(prediction);

        // Scale to index size
        let predicted_pos = (normalized_pred * self.sorted_keys.len() as f32) as usize;
        let predicted_pos = predicted_pos.min(self.sorted_keys.len().saturating_sub(1));

        // Compute error bounds
        let error_bound =
            (self.error_stats.std_error * self.config.error_bound_multiplier as f64) as usize;
        let lower = predicted_pos.saturating_sub(error_bound);
        let upper = (predicted_pos + error_bound).min(self.sorted_keys.len());

        // Confidence based on prediction certainty
        let confidence = 1.0 - (normalized_pred - 0.5).abs() * 2.0;

        let bounds = PredictionBounds::new(predicted_pos, lower, upper, confidence);

        // Record statistics
        let elapsed = start.elapsed().as_micros() as f64;
        self.stats.record_lookup(bounds.error_magnitude, elapsed);

        Ok(bounds)
    }

    /// Lookup exact position using learned index + binary search
    pub fn lookup(&mut self, key: &[f32]) -> LearnedIndexResult<usize> {
        // First check if key exists in mapping
        let key_str = self.key_to_string(key);
        if let Some(&pos) = self.key_positions.get(&key_str) {
            return Ok(pos);
        }

        // Use learned model to narrow search
        let bounds = self.predict(key)?;

        // Binary search within predicted range
        let search_range = bounds.search_range();
        let result = self.binary_search_range(key, search_range);

        // Record prediction accuracy
        if let Ok(actual_pos) = result {
            let within_bounds = actual_pos >= bounds.lower_bound && actual_pos < bounds.upper_bound;
            self.stats
                .record_prediction(bounds.predicted, actual_pos, within_bounds);
        }

        result
    }

    /// Forward pass through neural network
    fn forward(&self, input: &[f32]) -> f32 {
        if self.weights.is_empty() {
            // Fallback: simple linear interpolation
            return input.iter().sum::<f32>() / input.len() as f32;
        }

        let mut activation = input.to_vec();

        // Process through layers
        for (layer_idx, layer_weights) in self.weights.iter().enumerate() {
            let mut next_activation = Vec::new();

            let output_size = layer_weights.len() / activation.len();
            for i in 0..output_size {
                let mut sum = 0.0;
                for (j, &input_val) in activation.iter().enumerate() {
                    let weight_idx = j * output_size + i;
                    if weight_idx < layer_weights.len() {
                        sum += input_val * layer_weights[weight_idx];
                    }
                }

                // Add bias
                if layer_idx < self.biases.len() {
                    sum += self.biases[layer_idx];
                }

                // Apply activation
                next_activation.push(Self::relu(sum));
            }

            activation = next_activation;
        }

        // Output layer (single value)
        activation.first().copied().unwrap_or(0.5)
    }

    /// Binary search within range
    fn binary_search_range(
        &self,
        key: &[f32],
        range: std::ops::Range<usize>,
    ) -> LearnedIndexResult<usize> {
        let mut left = range.start;
        let mut right = range.end;

        while left < right {
            let mid = (left + right) / 2;
            if mid >= self.sorted_keys.len() {
                break;
            }

            let mid_key = &self.sorted_keys[mid];
            match Self::compare_keys(key, mid_key) {
                std::cmp::Ordering::Less => right = mid,
                std::cmp::Ordering::Greater => left = mid + 1,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }

        if left < self.sorted_keys.len() {
            Ok(left)
        } else {
            Err(LearnedIndexError::PredictionOutOfBounds {
                predicted: left,
                actual_size: self.sorted_keys.len(),
            })
        }
    }

    /// Compute error statistics from training data
    fn compute_error_statistics(&mut self, examples: &[TrainingExample]) {
        let mut errors = Vec::new();

        for example in examples.iter().take(1000) {
            // Sample for efficiency
            let predicted = self.forward(&example.features);
            let normalized = Self::sigmoid(predicted);
            let predicted_pos = (normalized * self.sorted_keys.len() as f32) as usize;

            let error = predicted_pos.abs_diff(example.target_position);
            errors.push(error as f64);
        }

        if !errors.is_empty() {
            let mean = errors.iter().sum::<f64>() / errors.len() as f64;
            let variance =
                errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / errors.len() as f64;

            self.error_stats = ErrorStatistics {
                mean_error: mean,
                std_error: variance.sqrt(),
                max_error: errors.iter().map(|&e| e as usize).max().unwrap_or(0),
            };
        }
    }

    /// Get statistics
    pub fn statistics(&self) -> &IndexStatistics {
        &self.stats
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    // Helper functions
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    fn key_to_string(&self, key: &[f32]) -> String {
        key.iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn compare_keys(a: &[f32], b: &[f32]) -> std::cmp::Ordering {
        for (av, bv) in a.iter().zip(b.iter()) {
            match av.partial_cmp(bv) {
                Some(std::cmp::Ordering::Equal) => continue,
                Some(ord) => return ord,
                None => return std::cmp::Ordering::Equal,
            }
        }
        a.len().cmp(&b.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::Random;

    fn create_test_examples(n: usize) -> Vec<TrainingExample> {
        let mut rng = Random::seed(42);
        (0..n)
            .map(|i| {
                let features = vec![i as f32 / n as f32, rng.gen_range(0.0..1.0)];
                TrainingExample::new(features, i)
            })
            .collect()
    }

    #[test]
    fn test_neural_index_creation() {
        let config = LearnedIndexConfig::default_config();
        let index = NeuralVectorIndex::new(config);
        assert!(index.is_ok());
    }

    #[test]
    fn test_training_insufficient_data() {
        let config = LearnedIndexConfig::default_config();
        let mut index = NeuralVectorIndex::new(config).unwrap();

        let examples = create_test_examples(10);
        let result = index.train(examples);
        assert!(result.is_err());
    }

    #[test]
    fn test_prediction_before_training() {
        let config = LearnedIndexConfig::default_config();
        let mut index = NeuralVectorIndex::new(config).unwrap();

        let key = vec![0.5, 0.5];
        let result = index.predict(&key);
        assert!(result.is_err());
    }

    #[test]
    fn test_training_and_prediction() {
        let mut config = LearnedIndexConfig::speed_optimized();
        config.min_training_examples = 100;

        let mut index = NeuralVectorIndex::new(config).unwrap();

        let examples = create_test_examples(100);
        let stats = index.train(examples).unwrap();

        assert!(stats.final_loss >= 0.0);
        assert!(index.is_trained());

        let key = vec![0.5, 0.5];
        let bounds = index.predict(&key).unwrap();

        assert!(bounds.predicted < 100);
        assert!(bounds.lower_bound <= bounds.predicted);
        assert!(bounds.upper_bound > bounds.predicted);
    }
}
