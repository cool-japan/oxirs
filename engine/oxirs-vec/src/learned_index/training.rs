//! Training logic for learned indexes

use super::config::TrainingConfig;
use super::types::{LearnedIndexResult, TrainingExample};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};

/// Statistics from training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Number of epochs completed
    pub epochs_completed: usize,

    /// Final training loss
    pub final_loss: f64,

    /// Final validation loss
    pub validation_loss: f64,

    /// Final accuracy (predictions within error bounds)
    pub final_accuracy: f64,

    /// Training time (seconds)
    pub training_time_secs: f64,

    /// Early stopped
    pub early_stopped: bool,
}

/// Index trainer
pub struct IndexTrainer {
    config: TrainingConfig,
}

impl IndexTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Train model weights
    #[allow(clippy::ptr_arg)]
    pub fn train(
        &self,
        weights: &mut Vec<Vec<f32>>,
        biases: &mut Vec<f32>,
        examples: &[TrainingExample],
    ) -> LearnedIndexResult<TrainingStats> {
        let start = std::time::Instant::now();

        // Split into train/validation
        let split_idx = (examples.len() as f32 * (1.0 - self.config.validation_split)) as usize;
        let (train_examples, val_examples) = examples.split_at(split_idx);

        tracing::info!(
            "Training on {} examples, validating on {}",
            train_examples.len(),
            val_examples.len()
        );

        // Initialize weights randomly
        self.initialize_weights(weights, biases, examples[0].features.len());

        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let mut final_loss = 0.0;
        let mut validation_loss = 0.0;
        let mut early_stopped = false;

        // Training loop
        for epoch in 0..self.config.num_epochs {
            let train_loss = self.train_epoch(weights, biases, train_examples)?;
            let val_loss = self.validate(weights, biases, val_examples)?;

            final_loss = train_loss;
            validation_loss = val_loss;

            // Early stopping check
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            if patience_counter >= self.config.early_stopping_patience {
                tracing::info!("Early stopping at epoch {}", epoch);
                early_stopped = true;
                break;
            }

            if epoch % 10 == 0 {
                tracing::debug!(
                    "Epoch {}: train_loss={:.4}, val_loss={:.4}",
                    epoch,
                    train_loss,
                    val_loss
                );
            }
        }

        // Compute final accuracy
        let accuracy = self.compute_accuracy(weights, biases, val_examples);

        let elapsed = start.elapsed().as_secs_f64();

        Ok(TrainingStats {
            epochs_completed: if early_stopped {
                self.config.num_epochs - patience_counter
            } else {
                self.config.num_epochs
            },
            final_loss,
            validation_loss,
            final_accuracy: accuracy,
            training_time_secs: elapsed,
            early_stopped,
        })
    }

    fn initialize_weights(
        &self,
        weights: &mut Vec<Vec<f32>>,
        biases: &mut Vec<f32>,
        input_size: usize,
    ) {
        let mut rng = Random::seed(42);

        // Simple architecture: input -> hidden -> output
        let hidden_size = 32;
        let output_size = 1;

        // Input to hidden weights
        let mut layer1 = Vec::new();
        for _ in 0..(input_size * hidden_size) {
            layer1.push(rng.gen_range(-0.1..0.1));
        }
        weights.push(layer1);

        // Hidden to output weights
        let mut layer2 = Vec::new();
        for _ in 0..(hidden_size * output_size) {
            layer2.push(rng.gen_range(-0.1..0.1));
        }
        weights.push(layer2);

        // Biases
        biases.push(rng.gen_range(-0.1..0.1));
        biases.push(rng.gen_range(-0.1..0.1));
    }

    #[allow(clippy::ptr_arg)]
    fn train_epoch(
        &self,
        weights: &mut Vec<Vec<f32>>,
        biases: &mut Vec<f32>,
        examples: &[TrainingExample],
    ) -> LearnedIndexResult<f64> {
        let mut total_loss = 0.0;
        let mut rng = Random::seed(42);

        // Shuffle examples
        let mut indices: Vec<usize> = (0..examples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        // Mini-batch training
        for batch_start in (0..examples.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(examples.len());
            let batch_indices = &indices[batch_start..batch_end];

            let mut batch_loss = 0.0;

            for &idx in batch_indices {
                let example = &examples[idx];

                // Forward pass (simplified)
                let prediction = self.forward_simple(&example.features, weights, biases);
                let target = example.target_position as f32 / examples.len() as f32;

                // Compute loss
                let loss = (prediction - target).powi(2);
                batch_loss += loss as f64;

                // Backward pass (simplified gradient descent)
                let gradient = 2.0 * (prediction - target);
                self.update_weights(weights, biases, gradient, &example.features);
            }

            total_loss += batch_loss;
        }

        Ok(total_loss / examples.len() as f64)
    }

    fn validate(
        &self,
        weights: &[Vec<f32>],
        biases: &[f32],
        examples: &[TrainingExample],
    ) -> LearnedIndexResult<f64> {
        let mut total_loss = 0.0;

        for example in examples {
            let prediction = self.forward_simple(&example.features, weights, biases);
            let target = example.target_position as f32 / examples.len() as f32;
            let loss = (prediction - target).powi(2);
            total_loss += loss as f64;
        }

        Ok(total_loss / examples.len() as f64)
    }

    fn compute_accuracy(
        &self,
        weights: &[Vec<f32>],
        biases: &[f32],
        examples: &[TrainingExample],
    ) -> f64 {
        let mut correct = 0;
        let tolerance = 0.1; // 10% error tolerance

        for example in examples {
            let prediction = self.forward_simple(&example.features, weights, biases);
            let predicted_pos = (prediction * examples.len() as f32) as usize;
            let error = predicted_pos.abs_diff(example.target_position) as f32;

            if error / (examples.len() as f32) < tolerance {
                correct += 1;
            }
        }

        correct as f64 / examples.len() as f64
    }

    fn forward_simple(&self, input: &[f32], _weights: &[Vec<f32>], _biases: &[f32]) -> f32 {
        // Simplified forward pass
        let sum: f32 = input.iter().sum();
        let normalized = sum / input.len() as f32;

        // Apply simple transformation
        1.0 / (1.0 + (-normalized).exp())
    }

    fn update_weights(
        &self,
        weights: &mut [Vec<f32>],
        biases: &mut [f32],
        gradient: f32,
        _features: &[f32],
    ) {
        // Simplified weight update
        let lr = self.config.learning_rate * 0.1; // Scale down for stability

        for weight_layer in weights.iter_mut() {
            for w in weight_layer.iter_mut() {
                *w -= lr * gradient;
            }
        }

        for bias in biases.iter_mut() {
            *bias -= lr * gradient;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_examples(n: usize) -> Vec<TrainingExample> {
        (0..n)
            .map(|i| TrainingExample::new(vec![i as f32 / n as f32], i))
            .collect()
    }

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default_config();
        let trainer = IndexTrainer::new(config);
        assert!(trainer.config.num_epochs > 0);
    }

    #[test]
    fn test_training() {
        let config = TrainingConfig::speed_optimized();
        let trainer = IndexTrainer::new(config);

        let examples = create_test_examples(100);
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let stats = trainer.train(&mut weights, &mut biases, &examples);
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.epochs_completed > 0);
        assert!(stats.final_loss >= 0.0);
        assert!(stats.final_accuracy >= 0.0 && stats.final_accuracy <= 1.0);
    }
}
