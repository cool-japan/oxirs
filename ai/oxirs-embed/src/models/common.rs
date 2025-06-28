//! Common utilities and functions used across embedding models

// Removed unused import
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Distribution, Uniform};

/// Initialize embeddings with Xavier/Glorot initialization
pub fn xavier_init(
    shape: (usize, usize),
    fan_in: usize,
    fan_out: usize,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let uniform = Uniform::new(-limit, limit);
    Array2::from_shape_fn(shape, |_| uniform.sample(rng))
}

/// Initialize embeddings with uniform distribution
pub fn uniform_init(shape: (usize, usize), low: f64, high: f64, rng: &mut impl Rng) -> Array2<f64> {
    let uniform = Uniform::new(low, high);
    Array2::from_shape_fn(shape, |_| uniform.sample(rng))
}

/// Initialize embeddings with normal distribution
pub fn normal_init(shape: (usize, usize), mean: f64, std: f64, rng: &mut impl Rng) -> Array2<f64> {
    let normal = rand_distr::Normal::new(mean, std).unwrap();
    Array2::from_shape_fn(shape, |_| normal.sample(rng))
}

/// Normalize embeddings to unit length (L2 normalization)
pub fn normalize_embeddings(embeddings: &mut Array2<f64>) {
    for mut row in embeddings.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 1e-10 {
            row /= norm;
        }
    }
}

/// Normalize a single embedding vector
pub fn normalize_vector(vector: &mut Array1<f64>) {
    let norm = vector.dot(vector).sqrt();
    if norm > 1e-10 {
        *vector /= norm;
    }
}

/// Compute L2 distance between two vectors
pub fn l2_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).mapv(|x| x * x).sum().sqrt()
}

/// Compute L1 distance between two vectors
pub fn l1_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).mapv(|x| x.abs()).sum()
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Clamp embeddings to a maximum norm
pub fn clamp_embeddings(embeddings: &mut Array2<f64>, max_norm: f64) {
    for mut row in embeddings.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > max_norm {
            row *= max_norm / norm;
        }
    }
}

/// Apply gradient descent update with L2 regularization
pub fn gradient_update(
    embeddings: &mut Array2<f64>,
    gradients: &Array2<f64>,
    learning_rate: f64,
    l2_reg: f64,
) {
    *embeddings = embeddings.clone() - learning_rate * (gradients + l2_reg * &*embeddings);
}

/// Apply gradient descent update for a single embedding
pub fn gradient_update_single(
    embedding: &mut Array1<f64>,
    gradient: &Array1<f64>,
    learning_rate: f64,
    l2_reg: f64,
) {
    *embedding = embedding.clone() - learning_rate * (gradient + l2_reg * &*embedding);
}

/// Sigmoid activation function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU activation function
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Tanh activation function
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Compute margin-based ranking loss
pub fn margin_loss(positive_score: f64, negative_score: f64, margin: f64) -> f64 {
    (margin + negative_score - positive_score).max(0.0)
}

/// Compute logistic loss
pub fn logistic_loss(score: f64, label: f64) -> f64 {
    (1.0 + (-label * score).exp()).ln()
}

/// Batch shuffle utility
pub fn shuffle_batch<T>(batch: &mut [T], rng: &mut impl Rng) {
    batch.shuffle(rng);
}

/// Create batches from data
pub fn create_batches<T: Clone>(data: &[T], batch_size: usize) -> Vec<Vec<T>> {
    data.chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Convert ndarray to Vector
pub fn ndarray_to_vector(array: &Array1<f64>) -> crate::Vector {
    let values: Vec<f32> = array.iter().map(|&x| x as f32).collect();
    crate::Vector::new(values)
}

/// Convert Vector to ndarray
pub fn vector_to_ndarray(vector: &crate::Vector) -> Array1<f64> {
    let values: Vec<f64> = vector.values.iter().map(|&x| x as f64).collect();
    Array1::from_vec(values)
}

/// Learning rate scheduling
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant(f64),
    /// Exponential decay: lr * decay_rate^(epoch / decay_steps)
    ExponentialDecay {
        initial_lr: f64,
        decay_rate: f64,
        decay_steps: usize,
    },
    /// Step decay: lr * factor^(epoch / step_size)
    StepDecay {
        initial_lr: f64,
        step_size: usize,
        factor: f64,
    },
    /// Polynomial decay
    PolynomialDecay {
        initial_lr: f64,
        final_lr: f64,
        decay_steps: usize,
        power: f64,
    },
}

impl LearningRateSchedule {
    /// Get learning rate for a given epoch
    pub fn get_lr(&self, epoch: usize) -> f64 {
        match self {
            LearningRateSchedule::Constant(lr) => *lr,
            LearningRateSchedule::ExponentialDecay {
                initial_lr,
                decay_rate,
                decay_steps,
            } => initial_lr * decay_rate.powf(epoch as f64 / *decay_steps as f64),
            LearningRateSchedule::StepDecay {
                initial_lr,
                step_size,
                factor,
            } => initial_lr * factor.powf((epoch / step_size) as f64),
            LearningRateSchedule::PolynomialDecay {
                initial_lr,
                final_lr,
                decay_steps,
                power,
            } => {
                if epoch >= *decay_steps {
                    *final_lr
                } else {
                    let decay_factor = (1.0 - epoch as f64 / *decay_steps as f64).powf(*power);
                    final_lr + (initial_lr - final_lr) * decay_factor
                }
            }
        }
    }
}

/// Early stopping utility
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    best_loss: f64,
    wait_count: usize,
    stopped: bool,
}

impl EarlyStopping {
    /// Create new early stopping monitor
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait_count: 0,
            stopped: false,
        }
    }

    /// Update with current loss and check if should stop
    pub fn update(&mut self, current_loss: f64) -> bool {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.wait_count = 0;
        } else {
            self.wait_count += 1;
            if self.wait_count > self.patience {
                self.stopped = true;
            }
        }

        self.stopped
    }

    /// Check if training should stop
    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    /// Get best loss seen so far
    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_distance_functions() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let l2_dist = l2_distance(&a, &b);
        assert!((l2_dist - 5.196152422706632).abs() < 1e-10);

        let l1_dist = l1_distance(&a, &b);
        assert!((l1_dist - 9.0).abs() < 1e-10);

        let cos_sim = cosine_similarity(&a, &b);
        assert!(cos_sim > 0.0 && cos_sim < 1.0);
    }

    #[test]
    fn test_normalization() {
        let mut vec = Array1::from_vec(vec![3.0, 4.0]);
        normalize_vector(&mut vec);
        let norm = vec.dot(&vec).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_learning_rate_schedule() {
        let schedule = LearningRateSchedule::ExponentialDecay {
            initial_lr: 0.1,
            decay_rate: 0.9,
            decay_steps: 10,
        };

        let lr0 = schedule.get_lr(0);
        let lr10 = schedule.get_lr(10);
        let lr20 = schedule.get_lr(20);

        assert!((lr0 - 0.1).abs() < 1e-10);
        assert!(lr10 < lr0);
        assert!(lr20 < lr10);
    }

    #[test]
    fn test_early_stopping() {
        let mut early_stop = EarlyStopping::new(3, 0.01);

        assert!(!early_stop.update(1.0));
        assert!(!early_stop.update(0.5));
        assert!(!early_stop.update(0.51));
        assert!(!early_stop.update(0.52));
        assert!(!early_stop.update(0.53));
        assert!(early_stop.update(0.54)); // Should stop now
    }
}
