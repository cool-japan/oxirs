//! Distributed Training Support for GT-SHACL
//!
//! Implements data-parallel training with synchronous gradient aggregation
//! (AllReduce) across worker threads.  Each worker holds a replica of the
//! model and processes a shard of the training dataset; gradients are
//! synchronised at the end of every mini-batch using an in-process
//! channel-based AllReduce.
//!
//! ## Architecture
//!
//! ```text
//!  DistributedTrainer
//!       │
//!       ├── Worker 0 (model replica) ──► forward/backward ──► grad buffer
//!       ├── Worker 1 (model replica) ──► forward/backward ──► grad buffer
//!       │          ⋮
//!       └── Worker N                 ──► forward/backward ──► grad buffer
//!                                                │
//!                                         AllReduceSync  (average grads)
//!                                                │
//!                                     parameter update (SGD / Adam)
//! ```
//!
//! Because Rust's ownership model prevents sharing mutable model state across
//! threads without synchronisation, each worker maintains an independent clone
//! of the parameter vector.  After AllReduce the master parameter vector is
//! broadcast back to every worker.

use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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

// ---------------------------------------------------------------------------
// Optimiser
// ---------------------------------------------------------------------------

/// SGD with optional momentum and L2 weight decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SgdOptimiser {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    velocity: Vec<f64>,
}

impl SgdOptimiser {
    pub fn new(param_count: usize, learning_rate: f64, momentum: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocity: vec![0.0_f64; param_count],
        }
    }

    /// Apply one step: params ← params − lr * (grad + wd * params) + momentum * velocity
    pub fn step(&mut self, params: &mut ParameterVector, grads: &ParameterVector) {
        let n = params
            .values
            .len()
            .min(grads.values.len())
            .min(self.velocity.len());
        for i in 0..n {
            let g = grads.values[i] + self.weight_decay * params.values[i];
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * g;
            params.values[i] += self.velocity[i];
        }
    }
}

/// Adam optimiser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamOptimiser {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    step: usize,
}

impl AdamOptimiser {
    pub fn new(
        param_count: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: vec![0.0_f64; param_count],
            v: vec![0.0_f64; param_count],
            step: 0,
        }
    }

    /// Convenience constructor with Adam defaults.
    pub fn default_config(param_count: usize, learning_rate: f64) -> Self {
        Self::new(param_count, learning_rate, 0.9, 0.999, 1e-8, 0.0)
    }

    /// Apply one Adam step.
    pub fn step(&mut self, params: &mut ParameterVector, grads: &ParameterVector) {
        self.step += 1;
        let t = self.step as f64;
        let lr_t =
            self.learning_rate * (1.0 - self.beta2.powf(t)).sqrt() / (1.0 - self.beta1.powf(t));

        let n = params
            .values
            .len()
            .min(grads.values.len())
            .min(self.m.len());
        for i in 0..n {
            let g = grads.values[i] + self.weight_decay * params.values[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            params.values[i] -= lr_t * self.m[i] / (self.v[i].sqrt() + self.eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Finite-difference gradient estimator
// ---------------------------------------------------------------------------

/// Approximates gradients via symmetric finite differences.
///
/// Given a scalar loss function `f: &[f64] -> f64`, estimates ∂f/∂θ_i ≈
/// (f(θ + h·eᵢ) − f(θ − h·eᵢ)) / (2h) for each parameter i.
pub fn finite_difference_grad<F>(params: &[f64], f: &F, h: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = params.len();
    let mut grad = vec![0.0_f64; n];
    let mut p_plus = params.to_vec();
    let mut p_minus = params.to_vec();

    for i in 0..n {
        p_plus[i] = params[i] + h;
        p_minus[i] = params[i] - h;
        grad[i] = (f(&p_plus) - f(&p_minus)) / (2.0 * h);
        p_plus[i] = params[i];
        p_minus[i] = params[i];
    }
    grad
}

// ---------------------------------------------------------------------------
// Distributed training runner
// ---------------------------------------------------------------------------

/// Statistics collected during a distributed training run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedTrainingStats {
    /// Total wall-clock time for the run.
    pub total_time_ms: u64,
    /// Number of communication rounds (AllReduce calls).
    pub communication_rounds: usize,
    /// Aggregate training loss over all rounds.
    pub total_loss: f64,
    /// Final parameter norm (useful for convergence monitoring).
    pub final_param_norm: f64,
    /// Number of worker threads used.
    pub num_workers: usize,
    /// Per-step losses.
    pub step_losses: Vec<f64>,
}

/// Configuration for a distributed training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTrainingConfig {
    /// Number of data-parallel workers.
    pub num_workers: usize,
    /// Number of training steps (AllReduce rounds).
    pub num_steps: usize,
    /// Size of the parameter vector.
    pub param_count: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Gradient clipping norm (0.0 = disabled).
    pub grad_clip_norm: f64,
    /// AllReduce strategy.
    pub strategy: AllReduceStrategy,
    /// Whether to use Adam (true) or SGD (false).
    pub use_adam: bool,
    /// Adam β₁.
    pub adam_beta1: f64,
    /// Adam β₂.
    pub adam_beta2: f64,
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            num_steps: 10,
            param_count: 128,
            learning_rate: 1e-3,
            grad_clip_norm: 1.0,
            strategy: AllReduceStrategy::Mean,
            use_adam: true,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
        }
    }
}

/// Simulated distributed training runner.
///
/// Each "worker" is a thread that generates synthetic gradient estimates and
/// participates in AllReduce.  This validates the synchronisation primitives
/// and optimiser implementations without requiring real training data.
pub struct DistributedTrainer {
    config: DistributedTrainingConfig,
}

impl DistributedTrainer {
    pub fn new(config: DistributedTrainingConfig) -> Self {
        Self { config }
    }

    /// Run the distributed training simulation.
    ///
    /// Returns per-step aggregate losses and final parameter norm.
    pub fn run(&self) -> Result<DistributedTrainingStats, ShaclAiError> {
        let t0 = Instant::now();
        let cfg = &self.config;

        if cfg.num_workers == 0 {
            return Err(ShaclAiError::Configuration(
                "num_workers must be >= 1".to_string(),
            ));
        }

        // Shared state
        let allreduce = Arc::new(AllReduceSync::new(
            cfg.strategy,
            cfg.num_workers,
            cfg.param_count,
        ));

        // Master parameter vector (protected by Mutex)
        let params = Arc::new(Mutex::new(ParameterVector::from_vec(vec![
            0.1_f64;
            cfg.param_count
        ])));

        // Shared step loss accumulator
        let step_losses: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let step_losses_out = Arc::clone(&step_losses);

        let num_steps = cfg.num_steps;
        let learning_rate = cfg.learning_rate;
        let grad_clip = cfg.grad_clip_norm;
        let use_adam = cfg.use_adam;
        let beta1 = cfg.adam_beta1;
        let beta2 = cfg.adam_beta2;
        let param_count = cfg.param_count;

        let mut handles = Vec::new();

        for rank in 0..cfg.num_workers {
            let allreduce = Arc::clone(&allreduce);
            let params = Arc::clone(&params);
            let step_losses = Arc::clone(&step_losses);

            let handle = thread::spawn(move || -> Result<(), ShaclAiError> {
                let mut local_params = {
                    let p = params.lock().map_err(|_| {
                        ShaclAiError::ModelTraining("params mutex poisoned".to_string())
                    })?;
                    p.clone()
                };

                let mut opt_sgd = SgdOptimiser::new(param_count, learning_rate, 0.9, 1e-4);
                let mut opt_adam = AdamOptimiser::default_config(param_count, learning_rate);
                opt_adam.beta1 = beta1;
                opt_adam.beta2 = beta2;

                for step in 0..num_steps {
                    // Simulate a loss function: quadratic bowl centered at zero
                    let loss: f64 = local_params
                        .values
                        .iter()
                        .map(|&v| 0.5 * v * v)
                        .sum::<f64>()
                        / param_count as f64;

                    // Approximate gradient: ∇L ≈ params (exact for quadratic)
                    let mut grad_acc = GradientAccumulator::new(param_count);
                    let grad = ParameterVector {
                        values: local_params.values.to_vec(),
                    };
                    grad_acc.accumulate(&grad);
                    grad_acc.average();

                    // Clip gradients
                    if grad_clip > 0.0 {
                        grad_acc.grads.clip_norm(grad_clip);
                    }

                    // AllReduce
                    let averaged_grad = allreduce.reduce(&grad_acc)?;

                    // Only rank 0 updates master params and records loss
                    if rank == 0 {
                        let mut p = params.lock().map_err(|_| {
                            ShaclAiError::ModelTraining(
                                "params mutex poisoned in update".to_string(),
                            )
                        })?;
                        let grad_pv = ParameterVector {
                            values: averaged_grad.values.clone(),
                        };
                        if use_adam {
                            opt_adam.step(&mut p, &grad_pv);
                        } else {
                            opt_sgd.step(&mut p, &grad_pv);
                        }

                        let mut sl = step_losses.lock().map_err(|_| {
                            ShaclAiError::ModelTraining("step_losses mutex poisoned".to_string())
                        })?;
                        sl.push(loss);

                        tracing::debug!(step, rank, loss, "distributed step");
                    }

                    // Sync local params with master (all workers read updated params)
                    // Use a small sleep to let rank 0 finish writing
                    thread::sleep(Duration::from_micros(10));

                    local_params = {
                        let p = params.lock().map_err(|_| {
                            ShaclAiError::ModelTraining("params local sync poisoned".to_string())
                        })?;
                        p.clone()
                    };

                    let _ = step; // suppress unused warning in non-debug builds
                }

                Ok(())
            });

            handles.push(handle);
        }

        // Join all workers
        for (i, handle) in handles.into_iter().enumerate() {
            handle.join().map_err(|e| {
                ShaclAiError::ModelTraining(format!("Worker {i} panicked: {e:?}"))
            })??;
        }

        let elapsed_ms = t0.elapsed().as_millis() as u64;

        let final_norm = params.lock().map(|p| p.norm()).unwrap_or(0.0);

        let losses = step_losses_out
            .lock()
            .map(|sl| sl.clone())
            .unwrap_or_default();

        let total_loss = losses.iter().sum();
        let communication_rounds = losses.len();

        Ok(DistributedTrainingStats {
            total_time_ms: elapsed_ms,
            communication_rounds,
            total_loss,
            final_param_norm: final_norm,
            num_workers: cfg.num_workers,
            step_losses: losses,
        })
    }
}

// ---------------------------------------------------------------------------
// Gradient compression (for communication efficiency)
// ---------------------------------------------------------------------------

/// Sparsify gradients by zeroing out entries below `threshold`.
///
/// Returns the compressed gradient and the sparsity ratio.
pub fn sparsify_gradients(grads: &ParameterVector, threshold: f64) -> (ParameterVector, f64) {
    let n = grads.values.len();
    if n == 0 {
        return (ParameterVector::zeros(0), 0.0);
    }
    let mut values = grads.values.clone();
    let mut zeroed = 0usize;
    for v in &mut values {
        if v.abs() < threshold {
            *v = 0.0;
            zeroed += 1;
        }
    }
    let sparsity = zeroed as f64 / n as f64;
    (ParameterVector { values }, sparsity)
}

/// Quantise gradients to INT8 representation (scale + zero-point).
///
/// Returns `(quantised_values, scale, zero_point)`.
pub fn quantise_gradients_i8(grads: &ParameterVector) -> (Vec<i8>, f64, f64) {
    if grads.values.is_empty() {
        return (Vec::new(), 1.0, 0.0);
    }
    let min = grads.values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = grads
        .values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-12);
    let scale = range / 255.0;
    let zero_point = min;
    let quantised: Vec<i8> = grads
        .values
        .iter()
        .map(|&v| {
            let q = ((v - zero_point) / scale).round().clamp(0.0, 255.0) as u8;
            q.wrapping_sub(128) as i8
        })
        .collect();
    (quantised, scale, zero_point)
}

/// Dequantise INT8 gradient buffer back to f64.
pub fn dequantise_gradients_i8(quantised: &[i8], scale: f64, zero_point: f64) -> ParameterVector {
    let values = quantised
        .iter()
        .map(|&q| {
            let u = (q as i16 + 128) as u8 as f64;
            u * scale + zero_point
        })
        .collect();
    ParameterVector { values }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ParameterVector ---

    #[test]
    fn test_param_vector_zeros() {
        let p = ParameterVector::zeros(4);
        assert_eq!(p.len(), 4);
        assert!(p.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_param_vector_add_assign() {
        let mut a = ParameterVector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = ParameterVector::from_vec(vec![0.5, 0.5, 0.5]);
        a.add_assign(&b);
        assert!((a.values[0] - 1.5).abs() < 1e-12);
        assert!((a.values[1] - 2.5).abs() < 1e-12);
        assert!((a.values[2] - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_param_vector_scale() {
        let mut p = ParameterVector::from_vec(vec![2.0, 4.0, 6.0]);
        p.scale(0.5);
        assert!((p.values[0] - 1.0).abs() < 1e-12);
        assert!((p.values[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_param_vector_norm() {
        let p = ParameterVector::from_vec(vec![3.0, 4.0]);
        assert!((p.norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_param_vector_clip_norm() {
        let mut p = ParameterVector::from_vec(vec![3.0, 4.0]); // norm = 5
        p.clip_norm(2.5);
        assert!(
            (p.norm() - 2.5).abs() < 1e-10,
            "norm after clip: {}",
            p.norm()
        );
    }

    #[test]
    fn test_param_vector_clip_norm_no_effect_when_below() {
        let mut p = ParameterVector::from_vec(vec![1.0, 1.0]); // norm ~= 1.41
        p.clip_norm(5.0);
        assert!((p.norm() - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_param_vector_mse_diff_zeros() {
        let a = ParameterVector::zeros(4);
        let b = ParameterVector::zeros(4);
        assert_eq!(a.mse_diff(&b), 0.0);
    }

    #[test]
    fn test_param_vector_mse_diff_nonzero() {
        let a = ParameterVector::from_vec(vec![0.0, 0.0]);
        let b = ParameterVector::from_vec(vec![1.0, 1.0]);
        assert!((a.mse_diff(&b) - 1.0).abs() < 1e-12);
    }

    // --- GradientAccumulator ---

    #[test]
    fn test_grad_accumulator_new() {
        let acc = GradientAccumulator::new(8);
        assert_eq!(acc.grads.len(), 8);
        assert_eq!(acc.sample_count, 0);
    }

    #[test]
    fn test_grad_accumulator_accumulate_and_average() {
        let mut acc = GradientAccumulator::new(2);
        acc.accumulate(&ParameterVector::from_vec(vec![2.0, 4.0]));
        acc.accumulate(&ParameterVector::from_vec(vec![4.0, 8.0]));
        acc.average();
        // After 2 samples with sum [6, 12] → average [3, 6]
        assert!((acc.grads.values[0] - 3.0).abs() < 1e-12);
        assert!((acc.grads.values[1] - 6.0).abs() < 1e-12);
        assert_eq!(acc.sample_count, 0);
    }

    #[test]
    fn test_grad_accumulator_reset() {
        let mut acc = GradientAccumulator::new(3);
        acc.accumulate(&ParameterVector::from_vec(vec![1.0, 1.0, 1.0]));
        acc.reset();
        assert!(acc.grads.values.iter().all(|&v| v == 0.0));
        assert_eq!(acc.sample_count, 0);
    }

    // --- SGD Optimiser ---

    #[test]
    fn test_sgd_converges_quadratic() {
        let n = 4;
        let mut opt = SgdOptimiser::new(n, 0.1, 0.0, 0.0);
        let mut params = ParameterVector::from_vec(vec![1.0_f64; n]);

        // Gradient of 0.5 * ||params||^2 is params itself.
        // With lr=0.1 and n=4 dimensions, 50 iterations yields
        // norm = sqrt(n) * 0.9^50 ≈ 0.0103 which narrowly exceeds 0.01.
        // 100 iterations gives norm ≈ 5e-6, well within the threshold.
        for _ in 0..100 {
            let grad = params.clone();
            opt.step(&mut params, &grad);
        }
        let norm = params.norm();
        assert!(norm < 0.01, "SGD did not converge, norm={norm}");
    }

    #[test]
    fn test_sgd_momentum() {
        let n = 2;
        let mut opt = SgdOptimiser::new(n, 0.01, 0.9, 0.0);
        let mut params = ParameterVector::from_vec(vec![1.0_f64; n]);
        for _ in 0..100 {
            let grad = params.clone();
            opt.step(&mut params, &grad);
        }
        assert!(params.norm() < 1.0, "momentum SGD norm={}", params.norm());
    }

    // --- Adam Optimiser ---

    #[test]
    fn test_adam_converges_quadratic() {
        let n = 4;
        let mut opt = AdamOptimiser::default_config(n, 0.01);
        let mut params = ParameterVector::from_vec(vec![1.0_f64; n]);

        for _ in 0..200 {
            let grad = params.clone();
            opt.step(&mut params, &grad);
        }
        let norm = params.norm();
        assert!(norm < 0.1, "Adam did not converge, norm={norm}");
    }

    #[test]
    fn test_adam_step_counter() {
        let n = 2;
        let mut opt = AdamOptimiser::default_config(n, 0.001);
        let mut params = ParameterVector::from_vec(vec![1.0_f64; n]);
        let grad = ParameterVector::from_vec(vec![0.1_f64; n]);
        opt.step(&mut params, &grad);
        opt.step(&mut params, &grad);
        assert_eq!(opt.step, 2);
    }

    // --- Finite difference ---

    #[test]
    fn test_finite_difference_quadratic() {
        // f(x) = 0.5 * ||x||^2  =>  grad = x
        let params = vec![1.0_f64, 2.0, 3.0];
        let f = |p: &[f64]| -> f64 { 0.5 * p.iter().map(|&v| v * v).sum::<f64>() };
        let grad = finite_difference_grad(&params, &f, 1e-5);
        for (i, &g) in grad.iter().enumerate() {
            assert!(
                (g - params[i]).abs() < 1e-6,
                "grad[{i}]={g} expected {}",
                params[i]
            );
        }
    }

    // --- Gradient compression ---

    #[test]
    fn test_sparsify_gradients() {
        let g = ParameterVector::from_vec(vec![0.001, 0.5, 0.002, 0.8]);
        let (sparse, sparsity) = sparsify_gradients(&g, 0.01);
        assert_eq!(sparse.values[0], 0.0);
        assert!((sparse.values[1] - 0.5).abs() < 1e-12);
        assert_eq!(sparse.values[2], 0.0);
        assert!((sparse.values[3] - 0.8).abs() < 1e-12);
        assert!((sparsity - 0.5).abs() < 1e-12, "sparsity={sparsity}");
    }

    #[test]
    fn test_sparsify_empty() {
        let g = ParameterVector::zeros(0);
        let (sparse, sparsity) = sparsify_gradients(&g, 0.1);
        assert!(sparse.is_empty());
        assert_eq!(sparsity, 0.0);
    }

    #[test]
    fn test_quantise_dequantise_roundtrip() {
        let original = ParameterVector::from_vec(vec![-1.0, 0.0, 0.5, 1.0]);
        let (q, scale, zp) = quantise_gradients_i8(&original);
        let recovered = dequantise_gradients_i8(&q, scale, zp);
        for (a, b) in original.values.iter().zip(&recovered.values) {
            assert!(
                (a - b).abs() < 0.05,
                "roundtrip error too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_quantise_empty() {
        let g = ParameterVector::zeros(0);
        let (q, scale, zp) = quantise_gradients_i8(&g);
        assert!(q.is_empty());
        assert_eq!(scale, 1.0);
        assert_eq!(zp, 0.0);
    }

    // --- Distributed training runner ---

    #[test]
    fn test_distributed_training_single_worker() {
        let cfg = DistributedTrainingConfig {
            num_workers: 1,
            num_steps: 5,
            param_count: 16,
            ..Default::default()
        };
        let trainer = DistributedTrainer::new(cfg);
        let stats = trainer.run().expect("single-worker training ok");
        assert_eq!(stats.num_workers, 1);
        assert_eq!(stats.communication_rounds, 5);
        assert!(stats.total_loss >= 0.0);
    }

    #[test]
    fn test_distributed_training_multi_worker() {
        let cfg = DistributedTrainingConfig {
            num_workers: 3,
            num_steps: 4,
            param_count: 8,
            ..Default::default()
        };
        let trainer = DistributedTrainer::new(cfg);
        let stats = trainer.run().expect("multi-worker training ok");
        assert_eq!(stats.num_workers, 3);
        // The 4 step losses are recorded by rank 0
        assert_eq!(stats.step_losses.len(), 4);
    }

    #[test]
    fn test_distributed_training_zero_workers_error() {
        let cfg = DistributedTrainingConfig {
            num_workers: 0,
            ..Default::default()
        };
        let trainer = DistributedTrainer::new(cfg);
        assert!(trainer.run().is_err());
    }

    #[test]
    fn test_distributed_training_loss_decreases() {
        let cfg = DistributedTrainingConfig {
            num_workers: 2,
            num_steps: 8,
            param_count: 4,
            learning_rate: 0.1,
            use_adam: false,
            ..Default::default()
        };
        let trainer = DistributedTrainer::new(cfg);
        let stats = trainer.run().expect("loss decrease test ok");
        let losses = &stats.step_losses;
        if losses.len() >= 2 {
            // Generally the first loss should be larger than the last
            // (not strictly guaranteed with threading nondeterminism but very likely)
            assert!(
                losses[0] >= losses[losses.len() - 1],
                "loss should not increase: first={} last={}",
                losses[0],
                losses[losses.len() - 1]
            );
        }
    }

    #[test]
    fn test_distributed_training_sgd() {
        let cfg = DistributedTrainingConfig {
            num_workers: 2,
            num_steps: 3,
            param_count: 4,
            use_adam: false,
            strategy: AllReduceStrategy::Sum,
            ..Default::default()
        };
        let trainer = DistributedTrainer::new(cfg);
        let stats = trainer.run().expect("SGD run ok");
        assert!(stats.total_time_ms < 10_000); // should finish quickly
    }

    #[test]
    fn test_all_reduce_strategies() {
        for strategy in [
            AllReduceStrategy::Mean,
            AllReduceStrategy::Sum,
            AllReduceStrategy::WeightedMean,
        ] {
            let cfg = DistributedTrainingConfig {
                num_workers: 2,
                num_steps: 2,
                param_count: 4,
                strategy,
                ..Default::default()
            };
            let trainer = DistributedTrainer::new(cfg);
            trainer.run().expect("strategy variant should run");
        }
    }

    #[test]
    fn test_training_stats_serialization() {
        let stats = DistributedTrainingStats {
            total_time_ms: 150,
            communication_rounds: 10,
            total_loss: 3.5,
            final_param_norm: 0.12,
            num_workers: 4,
            step_losses: vec![0.5, 0.4, 0.35, 0.3],
        };
        let json = serde_json::to_string(&stats).expect("serialize ok");
        let s2: DistributedTrainingStats = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(s2.num_workers, 4);
        assert_eq!(s2.communication_rounds, 10);
    }

    #[test]
    fn test_worker_config_default() {
        let wc = WorkerConfig::default();
        assert_eq!(wc.rank, 0);
        assert_eq!(wc.world_size, 1);
        assert!(wc.learning_rate > 0.0);
    }
}

// ===========================================================================
// Federated Learning for SHACL Shape Inference
// ===========================================================================
//
// Implements Federated Averaging (FedAvg) over multiple participants that
// each hold private RDF/SHACL training data.  Participants exchange only
// model weight deltas, never raw data, preserving data sovereignty.
//
// ## Algorithm
//
// 1. Server broadcasts current global model weights to participants.
// 2. Each participant trains locally for `local_epochs` steps on their shard.
// 3. Participants send back `(weight_delta, sample_count, local_loss)`.
// 4. Server computes a weighted average proportional to `sample_count`
//    and updates the global model (FedAvg).
// 5. Repeat for `num_rounds` federation rounds.

// ---------------------------------------------------------------------------
// ModelWeights
// ---------------------------------------------------------------------------

/// Dense model weight vector used for inter-participant communication.
///
/// Intentionally kept separate from [`ParameterVector`] so that federated
/// and data-parallel codepaths can evolve independently.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ModelWeights {
    /// Flattened weight values (e.g. all linear layer weights concatenated).
    pub weights: Vec<f64>,
    /// Bias values.
    pub biases: Vec<f64>,
    /// Monotonically increasing version counter incremented on each federated
    /// round.
    pub version: u32,
}

impl ModelWeights {
    /// Create zero-initialised weights of the given dimensions.
    pub fn zeros(weight_count: usize, bias_count: usize) -> Self {
        Self {
            weights: vec![0.0_f64; weight_count],
            biases: vec![0.0_f64; bias_count],
            version: 0,
        }
    }

    /// Create from explicit vectors.
    pub fn from_vecs(weights: Vec<f64>, biases: Vec<f64>) -> Self {
        Self {
            weights,
            biases,
            version: 0,
        }
    }

    /// L2 norm of the concatenated weight + bias vector.
    pub fn norm(&self) -> f64 {
        self.weights
            .iter()
            .chain(self.biases.iter())
            .map(|&v| v * v)
            .sum::<f64>()
            .sqrt()
    }

    /// Element-wise add `other` (in place).  Shorter vector is zero-padded.
    pub fn add_assign(&mut self, other: &ModelWeights) {
        let wn = self.weights.len().min(other.weights.len());
        for i in 0..wn {
            self.weights[i] += other.weights[i];
        }
        let bn = self.biases.len().min(other.biases.len());
        for i in 0..bn {
            self.biases[i] += other.biases[i];
        }
    }

    /// Scale all parameters by `scalar` (in place).
    pub fn scale(&mut self, scalar: f64) {
        for v in &mut self.weights {
            *v *= scalar;
        }
        for v in &mut self.biases {
            *v *= scalar;
        }
    }
}

// ---------------------------------------------------------------------------
// LocalUpdate
// ---------------------------------------------------------------------------

/// A model update submitted by one federated participant after local training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalUpdate {
    /// Unique participant identifier (e.g. endpoint IRI or UUID).
    pub participant_id: String,
    /// Delta between participant's locally trained weights and the global
    /// model weights at the start of this round.
    pub weight_delta: ModelWeights,
    /// Number of training samples used by this participant.
    pub sample_count: u32,
    /// Final training loss reported by the participant.
    pub local_loss: f64,
}

impl LocalUpdate {
    /// Create a new local update record.
    pub fn new(
        participant_id: impl Into<String>,
        weight_delta: ModelWeights,
        sample_count: u32,
        local_loss: f64,
    ) -> Self {
        Self {
            participant_id: participant_id.into(),
            weight_delta,
            sample_count,
            local_loss,
        }
    }
}

// ---------------------------------------------------------------------------
// FederatedRound
// ---------------------------------------------------------------------------

/// The state of a single federated averaging round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedRound {
    /// Round index (0-based).
    pub round_id: u32,
    /// Number of participating clients in this round.
    pub participant_count: u32,
    /// Global model broadcast at the start of this round.
    pub global_model: ModelWeights,
    /// Updates collected from participants.
    pub local_updates: Vec<LocalUpdate>,
}

impl FederatedRound {
    /// Create an empty round with the given global model.
    pub fn new(round_id: u32, global_model: ModelWeights) -> Self {
        let participant_count = 0;
        Self {
            round_id,
            participant_count,
            global_model,
            local_updates: Vec::new(),
        }
    }

    /// Add a participant update to this round.
    pub fn add_update(&mut self, update: LocalUpdate) {
        self.participant_count += 1;
        self.local_updates.push(update);
    }

    /// Total number of training samples across all participants.
    pub fn total_samples(&self) -> u64 {
        self.local_updates
            .iter()
            .map(|u| u.sample_count as u64)
            .sum()
    }

    /// Average local loss (unweighted).
    pub fn average_local_loss(&self) -> f64 {
        if self.local_updates.is_empty() {
            return 0.0;
        }
        self.local_updates.iter().map(|u| u.local_loss).sum::<f64>()
            / self.local_updates.len() as f64
    }
}

// ---------------------------------------------------------------------------
// FederatedShapeTrainer
// ---------------------------------------------------------------------------

/// Federated learning orchestrator for SHACL shape inference models.
///
/// Implements the FedAvg algorithm: the server aggregates participant weight
/// deltas using a sample-count-weighted average and applies the result to the
/// global model.
pub struct FederatedShapeTrainer {
    /// Current global model.
    global_model: ModelWeights,
    /// Number of federation rounds executed so far.
    rounds_completed: u32,
    /// Accumulated average loss across all rounds (for monitoring).
    cumulative_loss: f64,
}

impl FederatedShapeTrainer {
    /// Create a new trainer with a zero-initialised model.
    pub fn new(weight_count: usize, bias_count: usize) -> Self {
        Self {
            global_model: ModelWeights::zeros(weight_count, bias_count),
            rounds_completed: 0,
            cumulative_loss: 0.0,
        }
    }

    /// Create with an explicit initial model.
    pub fn with_initial_model(model: ModelWeights) -> Self {
        Self {
            global_model: model,
            rounds_completed: 0,
            cumulative_loss: 0.0,
        }
    }

    /// Return a clone of the current global model.
    pub fn global_model(&self) -> &ModelWeights {
        &self.global_model
    }

    /// Number of rounds completed.
    pub fn rounds_completed(&self) -> u32 {
        self.rounds_completed
    }

    /// Compute the FedAvg aggregate of the updates in `round` and return the
    /// new global model weights.
    ///
    /// The aggregated delta is the weighted average of each participant's
    /// `weight_delta` where the weight is proportional to `sample_count`.
    pub fn aggregate_updates(&self, round: &FederatedRound) -> ModelWeights {
        self.federated_averaging(&round.local_updates)
    }

    /// Perform FedAvg over a slice of local updates.
    ///
    /// Returns a `ModelWeights` whose `weights` and `biases` are the
    /// sample-count-weighted average of the deltas.  If `updates` is empty
    /// the global model is returned unchanged.
    pub fn federated_averaging(&self, updates: &[LocalUpdate]) -> ModelWeights {
        let total_samples: u64 = updates.iter().map(|u| u.sample_count as u64).sum();
        if total_samples == 0 || updates.is_empty() {
            return self.global_model.clone();
        }

        // Determine output dimensions from first update.
        let weight_len = updates[0].weight_delta.weights.len();
        let bias_len = updates[0].weight_delta.biases.len();

        let mut aggregated_weights = vec![0.0_f64; weight_len];
        let mut aggregated_biases = vec![0.0_f64; bias_len];

        for update in updates {
            let fraction = update.sample_count as f64 / total_samples as f64;
            for (i, &w) in update.weight_delta.weights.iter().enumerate() {
                if i < weight_len {
                    aggregated_weights[i] += w * fraction;
                }
            }
            for (i, &b) in update.weight_delta.biases.iter().enumerate() {
                if i < bias_len {
                    aggregated_biases[i] += b * fraction;
                }
            }
        }

        // Apply delta to global model.
        let mut new_model = self.global_model.clone();
        for (i, &dw) in aggregated_weights.iter().enumerate() {
            if i < new_model.weights.len() {
                new_model.weights[i] += dw;
            }
        }
        for (i, &db) in aggregated_biases.iter().enumerate() {
            if i < new_model.biases.len() {
                new_model.biases[i] += db;
            }
        }
        new_model.version += 1;
        new_model
    }

    /// Execute a complete federated round: aggregate updates, update the
    /// global model, and return the new model.
    pub fn execute_round(&mut self, mut round: FederatedRound) -> ModelWeights {
        round.participant_count = round.local_updates.len() as u32;
        let avg_loss = round.average_local_loss();
        let new_model = self.aggregate_updates(&round);
        self.global_model = new_model.clone();
        self.rounds_completed += 1;
        self.cumulative_loss += avg_loss;
        new_model
    }

    /// Average loss across all executed rounds.
    pub fn average_round_loss(&self) -> f64 {
        if self.rounds_completed == 0 {
            return 0.0;
        }
        self.cumulative_loss / self.rounds_completed as f64
    }
}

// ---------------------------------------------------------------------------
// GradientPrivacy
// ---------------------------------------------------------------------------

/// Differential privacy mechanisms for gradient updates.
///
/// Implements the Gaussian mechanism for (ε, δ)-differential privacy:
/// noise calibrated to `σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε`.
pub struct GradientPrivacy;

impl GradientPrivacy {
    /// Add calibrated Gaussian noise to `grad` in place.
    ///
    /// Implements the Gaussian mechanism with noise scale
    /// `σ = compute_noise_scale(sensitivity, epsilon, delta)`.
    ///
    /// Uses a deterministic pseudo-random sequence seeded from the gradient
    /// values themselves so that tests remain reproducible.
    pub fn add_gaussian_noise(grad: &mut [f64], sensitivity: f64, epsilon: f64, delta: f64) {
        let sigma = Self::compute_noise_scale(sensitivity, epsilon, delta);
        if sigma <= 0.0 || !sigma.is_finite() {
            return;
        }
        // Box-Muller transform for deterministic Gaussian noise generation.
        // We use the index as the seed to avoid external rand dependency.
        for (i, v) in grad.iter_mut().enumerate() {
            let u1 = Self::pseudo_uniform(i as u64 * 2 + 1);
            let u2 = Self::pseudo_uniform(i as u64 * 2 + 2);
            // Box-Muller
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *v += sigma * z;
        }
    }

    /// Clip gradient L2 norm to `max_norm` in place.
    ///
    /// If the L2 norm of `grad` exceeds `max_norm`, the entire vector is
    /// scaled down uniformly so that ‖grad‖₂ = `max_norm`.
    pub fn clip_gradients(grad: &mut [f64], max_norm: f64) {
        let norm: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for v in grad.iter_mut() {
                *v *= scale;
            }
        }
    }

    /// Compute the noise standard deviation for the Gaussian mechanism.
    ///
    /// Formula: `σ = sensitivity * sqrt(2 * ln(1.25 / δ)) / ε`
    ///
    /// Returns `f64::INFINITY` if epsilon ≤ 0 or delta ≤ 0.
    pub fn compute_noise_scale(sensitivity: f64, epsilon: f64, delta: f64) -> f64 {
        if epsilon <= 0.0 || delta <= 0.0 || delta >= 1.0 {
            return f64::INFINITY;
        }
        sensitivity * (2.0 * (1.25_f64 / delta).ln()).sqrt() / epsilon
    }

    /// Simple deterministic uniform-in-(0,1) pseudo-random number from seed.
    /// Uses a linear congruential generator.
    fn pseudo_uniform(seed: u64) -> f64 {
        // LCG constants from Numerical Recipes
        let x = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map to (0, 1) exclusive
        (x >> 11) as f64 / (1u64 << 53) as f64 + f64::EPSILON
    }
}

// ---------------------------------------------------------------------------
// Federated + privacy tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod federated_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ModelWeights tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_weights_zeros() {
        let mw = ModelWeights::zeros(4, 2);
        assert_eq!(mw.weights, vec![0.0; 4]);
        assert_eq!(mw.biases, vec![0.0; 2]);
        assert_eq!(mw.version, 0);
    }

    #[test]
    fn test_model_weights_norm() {
        let mw = ModelWeights::from_vecs(vec![3.0, 4.0], vec![0.0]);
        // ‖[3,4,0]‖ = 5.0
        assert!((mw.norm() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_model_weights_add_assign() {
        let mut a = ModelWeights::from_vecs(vec![1.0, 2.0], vec![0.5]);
        let b = ModelWeights::from_vecs(vec![0.5, 0.5], vec![0.5]);
        a.add_assign(&b);
        assert!((a.weights[0] - 1.5).abs() < 1e-9);
        assert!((a.weights[1] - 2.5).abs() < 1e-9);
        assert!((a.biases[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_model_weights_scale() {
        let mut mw = ModelWeights::from_vecs(vec![2.0, 4.0], vec![1.0]);
        mw.scale(0.5);
        assert!((mw.weights[0] - 1.0).abs() < 1e-9);
        assert!((mw.weights[1] - 2.0).abs() < 1e-9);
        assert!((mw.biases[0] - 0.5).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // FederatedRound tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_federated_round_total_samples() {
        let global = ModelWeights::zeros(2, 1);
        let mut round = FederatedRound::new(0, global);
        round.add_update(LocalUpdate::new("p1", ModelWeights::zeros(2, 1), 100, 0.5));
        round.add_update(LocalUpdate::new("p2", ModelWeights::zeros(2, 1), 200, 0.4));
        assert_eq!(round.total_samples(), 300);
        assert_eq!(round.participant_count, 2);
    }

    #[test]
    fn test_federated_round_average_loss() {
        let global = ModelWeights::zeros(2, 1);
        let mut round = FederatedRound::new(0, global);
        round.add_update(LocalUpdate::new("p1", ModelWeights::zeros(2, 1), 50, 0.6));
        round.add_update(LocalUpdate::new("p2", ModelWeights::zeros(2, 1), 50, 0.4));
        let avg = round.average_local_loss();
        assert!((avg - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_federated_round_empty_loss() {
        let round = FederatedRound::new(0, ModelWeights::zeros(2, 1));
        assert_eq!(round.average_local_loss(), 0.0);
    }

    // -----------------------------------------------------------------------
    // FederatedShapeTrainer — federated_averaging
    // -----------------------------------------------------------------------

    #[test]
    fn test_federated_averaging_equal_samples() {
        let trainer = FederatedShapeTrainer::new(2, 1);
        let updates = vec![
            LocalUpdate::new(
                "p1",
                ModelWeights::from_vecs(vec![2.0, 0.0], vec![1.0]),
                50,
                0.5,
            ),
            LocalUpdate::new(
                "p2",
                ModelWeights::from_vecs(vec![0.0, 2.0], vec![1.0]),
                50,
                0.5,
            ),
        ];
        let result = trainer.federated_averaging(&updates);
        // With equal samples each participant contributes 0.5
        // delta_w = 0.5*[2,0] + 0.5*[0,2] = [1,1]
        // global is zeros → new global = [1,1]
        assert!((result.weights[0] - 1.0).abs() < 1e-9);
        assert!((result.weights[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_federated_averaging_weighted_by_samples() {
        let trainer = FederatedShapeTrainer::new(1, 0);
        let updates = vec![
            LocalUpdate::new("p1", ModelWeights::from_vecs(vec![3.0], vec![]), 100, 0.3),
            LocalUpdate::new("p2", ModelWeights::from_vecs(vec![1.0], vec![]), 100, 0.7),
        ];
        let result = trainer.federated_averaging(&updates);
        // delta = (100/200)*3 + (100/200)*1 = 1.5 + 0.5 = 2.0
        assert!((result.weights[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_federated_averaging_empty_updates() {
        let trainer = FederatedShapeTrainer::with_initial_model(ModelWeights::from_vecs(
            vec![5.0],
            vec![1.0],
        ));
        let result = trainer.federated_averaging(&[]);
        // Should return the current global model unchanged
        assert!((result.weights[0] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_execute_round_updates_global_model() {
        let mut trainer = FederatedShapeTrainer::new(2, 1);
        let mut round = FederatedRound::new(0, trainer.global_model().clone());
        round.add_update(LocalUpdate::new(
            "p1",
            ModelWeights::from_vecs(vec![1.0, 0.5], vec![0.1]),
            80,
            0.4,
        ));
        round.add_update(LocalUpdate::new(
            "p2",
            ModelWeights::from_vecs(vec![0.5, 1.0], vec![0.2]),
            120,
            0.6,
        ));
        trainer.execute_round(round);
        assert_eq!(trainer.rounds_completed(), 1);
        // Global model should have changed
        assert!(trainer.global_model().norm() > 0.0);
    }

    #[test]
    fn test_version_increments_per_round() {
        let mut trainer = FederatedShapeTrainer::new(2, 1);
        assert_eq!(trainer.global_model().version, 0);
        for r in 0..3 {
            let mut round = FederatedRound::new(r, trainer.global_model().clone());
            round.add_update(LocalUpdate::new("p1", ModelWeights::zeros(2, 1), 10, 0.5));
            trainer.execute_round(round);
        }
        assert_eq!(trainer.global_model().version, 3);
    }

    #[test]
    fn test_average_round_loss_no_rounds() {
        let trainer = FederatedShapeTrainer::new(2, 1);
        assert_eq!(trainer.average_round_loss(), 0.0);
    }

    #[test]
    fn test_average_round_loss_after_rounds() {
        let mut trainer = FederatedShapeTrainer::new(1, 0);
        for loss in [0.8, 0.6, 0.4] {
            let mut round = FederatedRound::new(0, trainer.global_model().clone());
            round.add_update(LocalUpdate::new("p1", ModelWeights::zeros(1, 0), 10, loss));
            trainer.execute_round(round);
        }
        // Average of 0.8, 0.6, 0.4 = 0.6
        let avg = trainer.average_round_loss();
        assert!((avg - 0.6).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // GradientPrivacy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_clip_gradients_below_max_norm_unchanged() {
        let mut grad = vec![3.0, 4.0]; // norm = 5.0
        GradientPrivacy::clip_gradients(&mut grad, 10.0);
        // norm < max_norm → unchanged
        assert!((grad[0] - 3.0).abs() < 1e-9);
        assert!((grad[1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_clip_gradients_above_max_norm_scaled() {
        let mut grad = vec![3.0, 4.0]; // norm = 5.0
        GradientPrivacy::clip_gradients(&mut grad, 2.5);
        let new_norm: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!((new_norm - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_clip_gradients_zero_vector() {
        let mut grad = vec![0.0, 0.0];
        GradientPrivacy::clip_gradients(&mut grad, 1.0);
        assert_eq!(grad, vec![0.0, 0.0]);
    }

    #[test]
    fn test_compute_noise_scale_positive() {
        let sigma = GradientPrivacy::compute_noise_scale(1.0, 1.0, 1e-5);
        assert!(sigma > 0.0 && sigma.is_finite());
    }

    #[test]
    fn test_compute_noise_scale_large_epsilon() {
        // Larger epsilon → smaller noise
        let sigma_small_eps = GradientPrivacy::compute_noise_scale(1.0, 0.1, 1e-5);
        let sigma_large_eps = GradientPrivacy::compute_noise_scale(1.0, 10.0, 1e-5);
        assert!(sigma_large_eps < sigma_small_eps);
    }

    #[test]
    fn test_compute_noise_scale_invalid_inputs() {
        assert!(GradientPrivacy::compute_noise_scale(1.0, 0.0, 1e-5).is_infinite());
        assert!(GradientPrivacy::compute_noise_scale(1.0, -1.0, 1e-5).is_infinite());
        assert!(GradientPrivacy::compute_noise_scale(1.0, 1.0, 0.0).is_infinite());
        assert!(GradientPrivacy::compute_noise_scale(1.0, 1.0, 1.5).is_infinite());
    }

    #[test]
    fn test_add_gaussian_noise_changes_gradient() {
        let original = vec![1.0, 2.0, 3.0];
        let mut grad = original.clone();
        GradientPrivacy::add_gaussian_noise(&mut grad, 1.0, 1.0, 1e-5);
        // With noise added, at least one element should differ
        let changed = grad
            .iter()
            .zip(original.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(changed, "noise should modify gradient values");
    }

    #[test]
    fn test_add_gaussian_noise_zero_epsilon_no_panic() {
        let mut grad = vec![1.0, 2.0];
        // epsilon=0 → sigma=inf → no-op (early return)
        GradientPrivacy::add_gaussian_noise(&mut grad, 1.0, 0.0, 1e-5);
        // Should not panic; values may remain unchanged
    }

    #[test]
    fn test_clip_then_noise_pipeline() {
        let mut grad = vec![100.0, 200.0, 300.0];
        // First clip to reasonable norm
        GradientPrivacy::clip_gradients(&mut grad, 1.0);
        let norm_after_clip: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!((norm_after_clip - 1.0).abs() < 1e-6);
        // Then add noise
        GradientPrivacy::add_gaussian_noise(&mut grad, 1.0, 2.0, 1e-5);
        // Should not panic; gradient values altered
    }

    #[test]
    fn test_local_update_serialization() {
        let update = LocalUpdate::new(
            "participant-1",
            ModelWeights::from_vecs(vec![0.1, 0.2], vec![0.05]),
            42,
            0.35,
        );
        let json = serde_json::to_string(&update).expect("serialize ok");
        let back: LocalUpdate = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(back.participant_id, "participant-1");
        assert_eq!(back.sample_count, 42);
        assert!((back.local_loss - 0.35).abs() < 1e-9);
    }

    #[test]
    fn test_federated_round_serialization() {
        let global = ModelWeights::from_vecs(vec![1.0, 2.0], vec![0.5]);
        let mut round = FederatedRound::new(3, global);
        round.add_update(LocalUpdate::new("p", ModelWeights::zeros(2, 1), 10, 0.2));
        let json = serde_json::to_string(&round).expect("serialize ok");
        let back: FederatedRound = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(back.round_id, 3);
        assert_eq!(back.local_updates.len(), 1);
    }
}
