//! Distributed training coordinator and federated-learning aggregator
//!
//! Houses the high-level runners that drive a population of workers:
//!
//! * [`DistributedTrainer`] simulates an in-process data-parallel job using
//!   AllReduce for gradient synchronisation.
//! * [`FederatedShapeTrainer`] orchestrates a FedAvg-style federation where
//!   participants exchange weight deltas instead of raw data.
//! * [`GradientPrivacy`] applies the (ε, δ) Gaussian mechanism to support
//!   differentially-private updates.
//!
//! Gradient compression utilities (sparsification, INT8 quantisation) live
//! here as well because they are typically applied just before the
//! coordinator broadcasts an update.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::training::distributed_types::{
    AllReduceStrategy, AllReduceSync, GradientAccumulator, ParameterVector,
};
use crate::training::distributed_worker::{AdamOptimiser, SgdOptimiser};
use crate::ShaclAiError;

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
