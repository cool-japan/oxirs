//! Core types for distributed training
//!
//! Contains the fundamental data structures shared between distributed and
//! federated training: parameter vectors, gradient accumulators, the
//! AllReduce synchroniser, and per-worker configuration.

use std::sync::{Arc, Barrier, Mutex};

use serde::{Deserialize, Serialize};

use crate::ShaclAiError;

// ---------------------------------------------------------------------------
// Parameter vector abstraction
// ---------------------------------------------------------------------------

/// A flat vector of f64 parameters representing all model weights in a single
/// contiguous buffer.  Used for gradient communication between workers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParameterVector {
    /// Raw parameter values.
    pub values: Vec<f64>,
}

impl ParameterVector {
    /// Create a parameter vector of length `n` initialised to zero.
    pub fn zeros(n: usize) -> Self {
        Self {
            values: vec![0.0_f64; n],
        }
    }

    /// Create with explicit values.
    pub fn from_vec(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Element-wise add `other` to `self`.
    pub fn add_assign(&mut self, other: &ParameterVector) {
        let n = self.values.len().min(other.values.len());
        for i in 0..n {
            self.values[i] += other.values[i];
        }
    }

    /// Scale all parameters by `scalar`.
    pub fn scale(&mut self, scalar: f64) {
        for v in &mut self.values {
            *v *= scalar;
        }
    }

    /// Clip gradient norms to `max_norm` (in-place).
    pub fn clip_norm(&mut self, max_norm: f64) {
        let norm: f64 = self.values.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            self.scale(scale);
        }
    }

    /// L2 norm of the parameter vector.
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    /// Compute the mean squared difference between `self` and `other` (for
    /// convergence checks).
    pub fn mse_diff(&self, other: &ParameterVector) -> f64 {
        let n = self.values.len().min(other.values.len());
        if n == 0 {
            return 0.0;
        }
        let sq_sum: f64 = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        sq_sum / n as f64
    }
}

// ---------------------------------------------------------------------------
// Gradient accumulator (per worker)
// ---------------------------------------------------------------------------

/// Accumulates gradients for a single worker during one mini-batch.
#[derive(Debug, Clone, Default)]
pub struct GradientAccumulator {
    /// Accumulated gradient vector.
    pub grads: ParameterVector,
    /// Number of samples contributing to these gradients.
    pub sample_count: usize,
}

impl GradientAccumulator {
    /// Create an accumulator for a parameter vector of size `n`.
    pub fn new(n: usize) -> Self {
        Self {
            grads: ParameterVector::zeros(n),
            sample_count: 0,
        }
    }

    /// Add a gradient vector for one sample (finite-difference approximation).
    pub fn accumulate(&mut self, grad: &ParameterVector) {
        self.grads.add_assign(grad);
        self.sample_count += 1;
    }

    /// Average accumulated gradients over the number of samples.
    pub fn average(&mut self) {
        if self.sample_count > 0 {
            self.grads.scale(1.0 / self.sample_count as f64);
            self.sample_count = 0;
        }
    }

    /// Reset to zero without reallocating.
    pub fn reset(&mut self) {
        for v in &mut self.grads.values {
            *v = 0.0;
        }
        self.sample_count = 0;
    }
}

// ---------------------------------------------------------------------------
// AllReduce synchroniser
// ---------------------------------------------------------------------------

/// Strategy for combining gradients from multiple workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllReduceStrategy {
    /// Simple average of all worker gradients (default for data-parallel SGD).
    Mean,
    /// Sum of all worker gradients (useful when batch sizes differ).
    Sum,
    /// Momentum-based smoothing (workers contribute weighted by sample count).
    WeightedMean,
}

/// In-process AllReduce for gradient synchronisation.
///
/// Each worker calls [`AllReduceSync::reduce`] once per step.  The call
/// blocks until all workers have submitted their gradients; the master then
/// computes the aggregate and broadcasts it back.
pub struct AllReduceSync {
    strategy: AllReduceStrategy,
    num_workers: usize,
    barrier: Arc<Barrier>,
    /// Shared gradient accumulator (protected by Mutex).
    shared_grads: Arc<Mutex<ParameterVector>>,
    /// Shared sample count (for weighted mean).
    shared_counts: Arc<Mutex<usize>>,
    /// The averaged gradient result, written by the last worker to arrive.
    result: Arc<Mutex<Option<ParameterVector>>>,
}

impl AllReduceSync {
    /// Create a new AllReduceSync for `num_workers` participants.
    pub fn new(strategy: AllReduceStrategy, num_workers: usize, param_count: usize) -> Self {
        Self {
            strategy,
            num_workers,
            barrier: Arc::new(Barrier::new(num_workers)),
            shared_grads: Arc::new(Mutex::new(ParameterVector::zeros(param_count))),
            shared_counts: Arc::new(Mutex::new(0)),
            result: Arc::new(Mutex::new(None)),
        }
    }

    /// Submit local gradients and block until all workers have contributed.
    ///
    /// Returns the averaged gradient vector.
    pub fn reduce(
        &self,
        local_grads: &GradientAccumulator,
    ) -> Result<ParameterVector, ShaclAiError> {
        // Phase 1: accumulate into shared buffer
        {
            let mut shared = self
                .shared_grads
                .lock()
                .map_err(|_| ShaclAiError::ModelTraining("AllReduce mutex poisoned".to_string()))?;
            shared.add_assign(&local_grads.grads);

            if self.strategy == AllReduceStrategy::WeightedMean {
                let mut counts = self.shared_counts.lock().map_err(|_| {
                    ShaclAiError::ModelTraining("AllReduce count mutex poisoned".to_string())
                })?;
                *counts += local_grads.sample_count;
            }
        }

        // Phase 2: barrier – wait for all workers
        self.barrier.wait();

        // Phase 3: last worker to pass barrier computes the result
        // (We use a double-check: try to take ownership of `result`)
        {
            let mut result_guard = self.result.lock().map_err(|_| {
                ShaclAiError::ModelTraining("AllReduce result mutex poisoned".to_string())
            })?;

            if result_guard.is_none() {
                let mut shared = self.shared_grads.lock().map_err(|_| {
                    ShaclAiError::ModelTraining("AllReduce mutex poisoned in finalize".to_string())
                })?;

                match self.strategy {
                    AllReduceStrategy::Mean => {
                        shared.scale(1.0 / self.num_workers as f64);
                    }
                    AllReduceStrategy::Sum => {
                        // already summed
                    }
                    AllReduceStrategy::WeightedMean => {
                        let total = *self.shared_counts.lock().map_err(|_| {
                            ShaclAiError::ModelTraining("count mutex poisoned".to_string())
                        })?;
                        if total > 0 {
                            shared.scale(1.0 / total as f64);
                        }
                    }
                }
                *result_guard = Some(shared.clone());
            }
        }

        // Phase 4: second barrier – ensure result is written before anyone reads it
        self.barrier.wait();

        // Read result
        let result = self
            .result
            .lock()
            .map_err(|_| ShaclAiError::ModelTraining("AllReduce result read poisoned".to_string()))?
            .clone()
            .ok_or_else(|| ShaclAiError::ModelTraining("AllReduce result not set".to_string()))?;

        // Phase 5: third barrier – ensure all workers have read before reset
        self.barrier.wait();

        // First worker resets shared buffer
        {
            let mut shared = self.shared_grads.lock().map_err(|_| {
                ShaclAiError::ModelTraining("AllReduce reset mutex poisoned".to_string())
            })?;
            for v in &mut shared.values {
                *v = 0.0;
            }
            if let Ok(mut counts) = self.shared_counts.lock() {
                *counts = 0;
            }
            if let Ok(mut r) = self.result.lock() {
                *r = None;
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Worker configuration
// ---------------------------------------------------------------------------

/// Configuration for a single distributed training worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Worker index (0 to num_workers-1).
    pub rank: usize,
    /// Total number of workers.
    pub world_size: usize,
    /// Local batch size processed by this worker per step.
    pub local_batch_size: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Gradient clipping max norm (0.0 = disabled).
    pub grad_clip_norm: f64,
    /// Weight decay (L2 regularisation).
    pub weight_decay: f64,
    /// Log interval (steps).
    pub log_interval: usize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            local_batch_size: 32,
            learning_rate: 1e-3,
            grad_clip_norm: 1.0,
            weight_decay: 1e-4,
            log_interval: 10,
        }
    }
}
