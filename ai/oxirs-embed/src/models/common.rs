//! Common utilities and functions used across embedding models

// Removed unused import
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{Rng, Random};

/// Initialize embeddings with Xavier/Glorot initialization (optimized)
pub fn xavier_init(
    shape: (usize, usize),
    fan_in: usize,
    fan_out: usize,
    rng: &mut Random,
) -> Array2<f64> {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let scale = 2.0 * limit;
    Array2::from_shape_fn(shape, |_| rng.random_f64() * scale - limit)
}

/// Batch Xavier initialization for multiple layers (memory efficient)
pub fn batch_xavier_init(
    shapes: &[(usize, usize)],
    fan_in: usize,
    fan_out: usize,
    rng: &mut Random,
) -> Vec<Array2<f64>> {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let scale = 2.0 * limit;

    shapes
        .iter()
        .map(|&shape| Array2::from_shape_fn(shape, |_| rng.random_f64() * scale - limit))
        .collect()
}

/// Initialize embeddings with uniform distribution
pub fn uniform_init(shape: (usize, usize), low: f64, high: f64, rng: &mut Random) -> Array2<f64> {
    Array2::from_shape_fn(shape, |_| rng.random_f64() * (high - low) + low)
}

/// Initialize embeddings with normal distribution
pub fn normal_init(shape: (usize, usize), mean: f64, std: f64, rng: &mut Random) -> Array2<f64> {
    Array2::from_shape_fn(shape, |_| {
        // Box-Muller transform for normal distribution
        let u1 = rng.random_f64();
        let u2 = rng.random_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z0
    })
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

/// Compute L2 distance between two vectors (optimized)
pub fn l2_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    // Use scirs2-core's zip for better performance
    scirs2_core::ndarray_ext::Zip::from(a)
        .and(b)
        .fold(0.0, |acc, &a_val, &b_val| {
            let diff = a_val - b_val;
            acc + diff * diff
        })
        .sqrt()
}

/// Compute L1 distance between two vectors (optimized)
pub fn l1_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    scirs2_core::ndarray_ext::Zip::from(a)
        .and(b)
        .fold(0.0, |acc, &a_val, &b_val| acc + (a_val - b_val).abs())
}

/// Compute cosine similarity between two vectors (optimized)
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let (dot_product, norm_a_sq, norm_b_sq) = scirs2_core::ndarray_ext::Zip::from(a).and(b).fold(
        (0.0, 0.0, 0.0),
        |(dot, norm_a, norm_b), &a_val, &b_val| {
            (
                dot + a_val * b_val,
                norm_a + a_val * a_val,
                norm_b + b_val * b_val,
            )
        },
    );

    let norm_product = (norm_a_sq * norm_b_sq).sqrt();
    if norm_product > 1e-10 {
        dot_product / norm_product
    } else {
        0.0
    }
}

/// Batch distance computation for multiple vector pairs
pub fn batch_l2_distances(vectors_a: &[Array1<f64>], vectors_b: &[Array1<f64>]) -> Vec<f64> {
    // Compute all pairwise distances between vectors_a and vectors_b
    let mut distances = Vec::with_capacity(vectors_a.len() * vectors_b.len());

    for a in vectors_a {
        for b in vectors_b {
            distances.push(l2_distance(a, b));
        }
    }

    distances
}

/// Efficient pairwise distance matrix computation
pub fn pairwise_distances(vectors: &[Array1<f64>]) -> Array2<f64> {
    let n = vectors.len();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = l2_distance(&vectors[i], &vectors[j]);
            distances[[i, j]] = dist;
            distances[[j, i]] = dist; // Matrix is symmetric
        }
    }

    distances
}

// F32 versions for transformer training compatibility
/// Compute cosine similarity between two f32 vectors (optimized)
pub fn cosine_similarity_f32(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let (dot_product, norm_a_sq, norm_b_sq) = scirs2_core::ndarray_ext::Zip::from(a).and(b).fold(
        (0.0_f32, 0.0_f32, 0.0_f32),
        |(dot, norm_a, norm_b), &a_val, &b_val| {
            (
                dot + a_val * b_val,
                norm_a + a_val * a_val,
                norm_b + b_val * b_val,
            )
        },
    );

    let norm_product = (norm_a_sq * norm_b_sq).sqrt();
    if norm_product > 1e-10 {
        dot_product / norm_product
    } else {
        0.0
    }
}

/// Compute L2 distance between two f32 vectors (optimized)
pub fn l2_distance_f32(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    scirs2_core::ndarray_ext::Zip::from(a)
        .and(b)
        .fold(0.0_f32, |acc, &a_val, &b_val| {
            let diff = a_val - b_val;
            acc + diff * diff
        })
        .sqrt()
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

/// Apply gradient descent update with L2 regularization (optimized)
pub fn gradient_update(
    embeddings: &mut Array2<f64>,
    gradients: &Array2<f64>,
    learning_rate: f64,
    l2_reg: f64,
) {
    // Vectorized in-place update to avoid temporary allocations
    scirs2_core::ndarray_ext::Zip::from(embeddings)
        .and(gradients)
        .for_each(|embed, &grad| {
            *embed = *embed - learning_rate * (grad + l2_reg * *embed);
        });
}

/// Batch gradient update for multiple embedding matrices
pub fn batch_gradient_update(
    embeddings: &mut [Array2<f64>],
    gradients: &[Array2<f64>],
    learning_rate: f64,
    l2_reg: f64,
) {
    for (embedding, gradient) in embeddings.iter_mut().zip(gradients.iter()) {
        gradient_update(embedding, gradient, learning_rate, l2_reg);
    }
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

/// Batch shuffle utility (optimized for performance)
pub fn shuffle_batch<T>(batch: &mut [T], rng: &mut Random) {
    // Fisher-Yates shuffle with early termination for small batches
    if batch.len() <= 1 {
        return;
    }

    for i in (1..batch.len()).rev() {
        let j = rng.random_range(0, i + 1);
        if i != j {
            batch.swap(i, j);
        }
    }
}

/// High-performance batch shuffling for multiple arrays
pub fn shuffle_multiple_batches<T: Clone>(batches: &mut [Vec<T>], rng: &mut Random) {
    for batch in batches.iter_mut() {
        shuffle_batch(batch, rng);
    }
}

/// Optimized random sampling without replacement
pub fn sample_without_replacement<T: Clone>(
    data: &[T],
    sample_size: usize,
    rng: &mut Random,
) -> Vec<T> {
    if sample_size >= data.len() {
        return data.to_vec();
    }

    let mut indices: Vec<usize> = (0..data.len()).collect();
    shuffle_batch(&mut indices, rng);

    indices[..sample_size]
        .iter()
        .map(|&i| data[i].clone())
        .collect()
}

/// Create batches from data (optimized to avoid unnecessary cloning)
pub fn create_batches<T: Clone>(data: &[T], batch_size: usize) -> Vec<Vec<T>> {
    let mut batches = Vec::with_capacity((data.len() + batch_size - 1) / batch_size);
    for chunk in data.chunks(batch_size) {
        batches.push(chunk.to_vec());
    }
    batches
}

/// Create batch references (zero-copy alternative)
pub fn create_batch_refs<T>(data: &[T], batch_size: usize) -> impl Iterator<Item = &[T]> {
    data.chunks(batch_size)
}

/// Convert ndarray to Vector (optimized with pre-allocation)
pub fn ndarray_to_vector(array: &Array1<f64>) -> crate::Vector {
    let mut values = Vec::with_capacity(array.len());
    values.extend(array.iter().map(|&x| x as f32));
    crate::Vector::new(values)
}

/// Convert Vector to ndarray (optimized with pre-allocation)
pub fn vector_to_ndarray(vector: &crate::Vector) -> Array1<f64> {
    let mut values = Vec::with_capacity(vector.values.len());
    values.extend(vector.values.iter().map(|&x| x as f64));
    Array1::from_vec(values)
}

/// Batch convert multiple ndarrays to vectors (SIMD-friendly)
pub fn batch_ndarray_to_vectors(arrays: &[Array1<f64>]) -> Vec<crate::Vector> {
    arrays.iter().map(ndarray_to_vector).collect()
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
    use scirs2_core::ndarray_ext::Array1;

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
