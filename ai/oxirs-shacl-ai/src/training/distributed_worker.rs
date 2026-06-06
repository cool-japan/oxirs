//! Worker-side training primitives: optimisers and gradient estimators
//!
//! Each distributed worker uses an optimiser (SGD or Adam) to apply the
//! aggregated gradient to its local parameter replica.  Gradients themselves
//! can either come from analytical backprop or, for unit tests and
//! integration sketches, from a symmetric finite-difference estimator.

use serde::{Deserialize, Serialize};

use crate::training::distributed_types::ParameterVector;

// ---------------------------------------------------------------------------
// Optimiser
// ---------------------------------------------------------------------------

/// SGD with optional momentum and L2 weight decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SgdOptimiser {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub(crate) velocity: Vec<f64>,
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
    pub(crate) m: Vec<f64>,
    pub(crate) v: Vec<f64>,
    pub(crate) step: usize,
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

    /// Current step counter (number of `step` invocations).
    pub fn step_count(&self) -> usize {
        self.step
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
