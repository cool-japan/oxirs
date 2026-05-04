//! # Online regression models for streaming
//!
//! Implements two regression models that learn incrementally from a stream of
//! `(features, target)` observations and produce predictions in real time.
//!
//! ## Models
//!
//! * [`OnlineLinearRegressor`] — least-mean-squares (LMS) online linear regression
//!   with optional L2 regularisation. Numerically stable using Welford's
//!   algorithm to track per-feature mean/variance for adaptive normalisation.
//! * [`StreamingGradientBoostedRegressor`] — Gradient-boosted tree-style
//!   regressor for streams. Uses a finite ensemble of small piece-wise constant
//!   "decision stumps" (single split, two leaves) trained on the residuals of
//!   the running ensemble. The ensemble is a fixed-capacity ring; the oldest
//!   tree is replaced by a freshly fit one once the ring is full.
//!
//! Both models implement the [`StreamRegressor`] trait so they can be plugged
//! into a higher-level operator (see `ml/mod.rs::StreamingModelRunner`).

use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::debug;

use scirs2_core::ndarray_ext::Array1;

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by the regression models.
#[derive(Debug, Error)]
pub enum RegressionError {
    /// Feature vector length did not match the model's configured dimension.
    #[error("feature dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    /// The model has not seen enough samples yet to make a prediction.
    #[error("model not ready: only {observed} of {required} samples observed")]
    NotReady { observed: usize, required: usize },
    /// A configuration value was out of range.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Convenience alias.
pub type RegressionResult<T> = std::result::Result<T, RegressionError>;

// ─── StreamRegressor trait ──────────────────────────────────────────────────

/// Common interface for online regressors used in stream operators.
pub trait StreamRegressor: Send + Sync {
    /// Configured number of input features.
    fn n_features(&self) -> usize;
    /// Number of `(features, target)` samples observed so far.
    fn n_observed(&self) -> u64;
    /// Update the model with a single observation.
    fn observe(&self, features: &Array1<f64>, target: f64) -> RegressionResult<()>;
    /// Produce a prediction for `features` if the model is ready.
    fn predict(&self, features: &Array1<f64>) -> RegressionResult<f64>;
}

// ─── OnlineLinearRegressor ──────────────────────────────────────────────────

/// Configuration for [`OnlineLinearRegressor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearConfig {
    /// Number of input features.
    pub n_features: usize,
    /// Learning rate of the LMS update step.
    pub learning_rate: f64,
    /// L2 regularisation strength (`0.0` disables it).
    pub l2: f64,
    /// Whether to standardise inputs using a running mean / variance estimate.
    pub standardise_inputs: bool,
    /// Minimum number of samples before [`StreamRegressor::predict`] returns.
    pub min_samples: u64,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            n_features: 4,
            learning_rate: 0.01,
            l2: 0.0,
            standardise_inputs: true,
            min_samples: 5,
        }
    }
}

impl LinearConfig {
    fn validate(&self) -> RegressionResult<()> {
        if self.n_features == 0 {
            return Err(RegressionError::InvalidConfig(
                "n_features must be > 0".into(),
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(RegressionError::InvalidConfig(
                "learning_rate must be positive".into(),
            ));
        }
        if !self.l2.is_finite() || self.l2 < 0.0 {
            return Err(RegressionError::InvalidConfig(
                "l2 must be non-negative".into(),
            ));
        }
        Ok(())
    }
}

/// Per-feature running statistics computed via Welford's algorithm.
///
/// Stable in single precision over very long streams.
#[derive(Debug, Clone, Default)]
struct FeatureStats {
    count: u64,
    mean: Array1<f64>,
    m2: Array1<f64>,
}

impl FeatureStats {
    fn new(n: usize) -> Self {
        Self {
            count: 0,
            mean: Array1::zeros(n),
            m2: Array1::zeros(n),
        }
    }

    /// Standardise `x` in place: `(x - mean) / sqrt(var + eps)`.
    fn standardise(&self, x: &Array1<f64>) -> Array1<f64> {
        const EPS: f64 = 1.0e-8;
        if self.count < 2 {
            return x.clone();
        }
        let n = x.len();
        let mut out = Array1::zeros(n);
        for i in 0..n {
            let var = if self.count > 1 {
                self.m2[i] / (self.count as f64 - 1.0)
            } else {
                0.0
            };
            let denom = (var + EPS).sqrt();
            out[i] = (x[i] - self.mean[i]) / denom;
        }
        out
    }

    fn update(&mut self, x: &Array1<f64>) {
        self.count += 1;
        let count_f = self.count as f64;
        for i in 0..x.len() {
            let delta = x[i] - self.mean[i];
            self.mean[i] += delta / count_f;
            let delta2 = x[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }
}

/// Inner mutable state of the linear regressor.
struct LinearState {
    weights: Array1<f64>,
    bias: f64,
    feature_stats: FeatureStats,
    samples: u64,
    last_loss: f64,
}

/// Online least-mean-squares linear regressor.
pub struct OnlineLinearRegressor {
    config: LinearConfig,
    state: Arc<RwLock<LinearState>>,
}

impl std::fmt::Debug for OnlineLinearRegressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.state.read();
        f.debug_struct("OnlineLinearRegressor")
            .field("config", &self.config)
            .field("samples", &st.samples)
            .field("last_loss", &st.last_loss)
            .finish()
    }
}

impl OnlineLinearRegressor {
    /// Build a regressor from a config.
    pub fn new(config: LinearConfig) -> RegressionResult<Self> {
        config.validate()?;
        let state = LinearState {
            weights: Array1::zeros(config.n_features),
            bias: 0.0,
            feature_stats: FeatureStats::new(config.n_features),
            samples: 0,
            last_loss: 0.0,
        };
        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Snapshot of the current weights (for diagnostics or persistence).
    pub fn weights(&self) -> Array1<f64> {
        self.state.read().weights.clone()
    }

    /// Current bias term.
    pub fn bias(&self) -> f64 {
        self.state.read().bias
    }

    /// Last per-sample squared loss.
    pub fn last_loss(&self) -> f64 {
        self.state.read().last_loss
    }

    fn check_dim(&self, x: &Array1<f64>) -> RegressionResult<()> {
        if x.len() != self.config.n_features {
            return Err(RegressionError::DimensionMismatch {
                expected: self.config.n_features,
                actual: x.len(),
            });
        }
        Ok(())
    }

    fn forward(&self, normalised: &Array1<f64>) -> f64 {
        let st = self.state.read();
        let mut y = st.bias;
        for i in 0..normalised.len() {
            y += st.weights[i] * normalised[i];
        }
        y
    }
}

impl StreamRegressor for OnlineLinearRegressor {
    fn n_features(&self) -> usize {
        self.config.n_features
    }

    fn n_observed(&self) -> u64 {
        self.state.read().samples
    }

    fn observe(&self, features: &Array1<f64>, target: f64) -> RegressionResult<()> {
        self.check_dim(features)?;
        if !target.is_finite() {
            return Err(RegressionError::InvalidConfig(
                "target must be finite".into(),
            ));
        }

        let normalised = if self.config.standardise_inputs {
            let stats = self.state.read().feature_stats.clone();
            stats.standardise(features)
        } else {
            features.clone()
        };

        let pred = self.forward(&normalised);
        let err = pred - target;

        {
            let mut st = self.state.write();
            // LMS update: w := w - lr * (err * x + l2 * w)
            for i in 0..self.config.n_features {
                let grad = err * normalised[i] + self.config.l2 * st.weights[i];
                st.weights[i] -= self.config.learning_rate * grad;
            }
            // Bias does not get L2 regularised.
            st.bias -= self.config.learning_rate * err;
            st.feature_stats.update(features);
            st.samples += 1;
            st.last_loss = 0.5 * err * err;
        }
        Ok(())
    }

    fn predict(&self, features: &Array1<f64>) -> RegressionResult<f64> {
        self.check_dim(features)?;
        let observed = self.state.read().samples;
        if observed < self.config.min_samples {
            return Err(RegressionError::NotReady {
                observed: observed as usize,
                required: self.config.min_samples as usize,
            });
        }

        let normalised = if self.config.standardise_inputs {
            let stats = self.state.read().feature_stats.clone();
            stats.standardise(features)
        } else {
            features.clone()
        };
        Ok(self.forward(&normalised))
    }
}

// ─── StreamingGradientBoostedRegressor ─────────────────────────────────────

/// Configuration for [`StreamingGradientBoostedRegressor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbtConfig {
    /// Number of input features.
    pub n_features: usize,
    /// Maximum number of stumps in the ensemble.
    pub max_trees: usize,
    /// Learning rate (shrinkage) applied to each tree's contribution.
    pub learning_rate: f64,
    /// Number of buffered observations used to fit each new stump.
    pub fit_buffer_size: usize,
    /// Minimum number of total observations before [`predict`] returns.
    pub min_samples: u64,
}

impl Default for GbtConfig {
    fn default() -> Self {
        Self {
            n_features: 4,
            max_trees: 16,
            learning_rate: 0.1,
            fit_buffer_size: 32,
            min_samples: 16,
        }
    }
}

impl GbtConfig {
    fn validate(&self) -> RegressionResult<()> {
        if self.n_features == 0 || self.max_trees == 0 || self.fit_buffer_size == 0 {
            return Err(RegressionError::InvalidConfig(
                "n_features, max_trees and fit_buffer_size must all be > 0".into(),
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(RegressionError::InvalidConfig(
                "learning_rate must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// A single decision stump with one feature index, one threshold and two leaves.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionStump {
    feature: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl DecisionStump {
    fn predict(&self, features: &Array1<f64>) -> f64 {
        if features[self.feature] <= self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }
}

/// Mutable state of [`StreamingGradientBoostedRegressor`].
struct GbtState {
    bias: f64,
    trees: Vec<DecisionStump>,
    /// Buffered observations awaiting a new stump fit.
    buffer: Vec<(Array1<f64>, f64)>,
    samples: u64,
    last_loss: f64,
}

/// Streaming gradient-boosted regressor.
pub struct StreamingGradientBoostedRegressor {
    config: GbtConfig,
    state: Arc<RwLock<GbtState>>,
}

impl std::fmt::Debug for StreamingGradientBoostedRegressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.state.read();
        f.debug_struct("StreamingGradientBoostedRegressor")
            .field("config", &self.config)
            .field("samples", &st.samples)
            .field("ensemble_size", &st.trees.len())
            .field("last_loss", &st.last_loss)
            .finish()
    }
}

impl StreamingGradientBoostedRegressor {
    /// Build a regressor from config.
    pub fn new(config: GbtConfig) -> RegressionResult<Self> {
        config.validate()?;
        let state = GbtState {
            bias: 0.0,
            trees: Vec::with_capacity(config.max_trees),
            buffer: Vec::with_capacity(config.fit_buffer_size),
            samples: 0,
            last_loss: 0.0,
        };
        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Number of stumps currently in the ensemble.
    pub fn ensemble_size(&self) -> usize {
        self.state.read().trees.len()
    }

    /// Last per-batch mean squared error.
    pub fn last_loss(&self) -> f64 {
        self.state.read().last_loss
    }

    fn check_dim(&self, features: &Array1<f64>) -> RegressionResult<()> {
        if features.len() != self.config.n_features {
            return Err(RegressionError::DimensionMismatch {
                expected: self.config.n_features,
                actual: features.len(),
            });
        }
        Ok(())
    }

    fn ensemble_predict(&self, features: &Array1<f64>) -> f64 {
        let st = self.state.read();
        let mut y = st.bias;
        for tree in &st.trees {
            y += self.config.learning_rate * tree.predict(features);
        }
        y
    }

    /// Fit a stump on the buffered residuals using the per-feature variance reduction
    /// criterion. The chosen threshold is the median value of the candidate feature
    /// (a robust, deterministic split point that does not require sorting all candidates).
    fn fit_stump_from_buffer(buffer: &[(Array1<f64>, f64)], n_features: usize) -> DecisionStump {
        debug_assert!(
            !buffer.is_empty(),
            "fit_stump_from_buffer called with empty buffer"
        );
        let n = buffer.len() as f64;
        let mean_target = buffer.iter().map(|(_, t)| *t).sum::<f64>() / n;
        let total_ss: f64 = buffer.iter().map(|(_, t)| (t - mean_target).powi(2)).sum();

        let mut best = DecisionStump {
            feature: 0,
            threshold: 0.0,
            left_value: mean_target,
            right_value: mean_target,
        };
        let mut best_gain = -1.0;

        for f in 0..n_features {
            // Threshold = sample median for feature `f` (robust to outliers).
            let mut values: Vec<f64> = buffer.iter().map(|(x, _)| x[f]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = values.len() / 2;
            let threshold = if values.len() % 2 == 0 && mid > 0 {
                0.5 * (values[mid - 1] + values[mid])
            } else if !values.is_empty() {
                values[mid]
            } else {
                continue;
            };

            let mut left_sum = 0.0;
            let mut left_count = 0usize;
            let mut right_sum = 0.0;
            let mut right_count = 0usize;
            for (x, t) in buffer {
                if x[f] <= threshold {
                    left_sum += *t;
                    left_count += 1;
                } else {
                    right_sum += *t;
                    right_count += 1;
                }
            }
            if left_count == 0 || right_count == 0 {
                continue;
            }
            let left_mean = left_sum / left_count as f64;
            let right_mean = right_sum / right_count as f64;
            let mut split_ss = 0.0;
            for (x, t) in buffer {
                if x[f] <= threshold {
                    split_ss += (t - left_mean).powi(2);
                } else {
                    split_ss += (t - right_mean).powi(2);
                }
            }
            let gain = total_ss - split_ss;
            if gain > best_gain {
                best_gain = gain;
                best = DecisionStump {
                    feature: f,
                    threshold,
                    left_value: left_mean,
                    right_value: right_mean,
                };
            }
        }
        best
    }
}

impl StreamRegressor for StreamingGradientBoostedRegressor {
    fn n_features(&self) -> usize {
        self.config.n_features
    }

    fn n_observed(&self) -> u64 {
        self.state.read().samples
    }

    fn observe(&self, features: &Array1<f64>, target: f64) -> RegressionResult<()> {
        self.check_dim(features)?;
        if !target.is_finite() {
            return Err(RegressionError::InvalidConfig(
                "target must be finite".into(),
            ));
        }
        let pred = self.ensemble_predict(features);
        let residual = target - pred;

        let mut st = self.state.write();
        st.samples += 1;
        st.last_loss = 0.5 * residual * residual;
        st.buffer.push((features.clone(), residual));

        if st.buffer.len() >= self.config.fit_buffer_size {
            let buffer = std::mem::take(&mut st.buffer);
            let stump = Self::fit_stump_from_buffer(&buffer, self.config.n_features);
            let feature = stump.feature;
            let threshold = stump.threshold;
            if st.trees.len() < self.config.max_trees {
                st.trees.push(stump);
            } else {
                // Ring-replace the oldest tree.
                st.trees.remove(0);
                st.trees.push(stump);
            }
            // Refresh bias: simple moving average of observed targets stays stable
            // but we also drift it toward the buffer's mean.
            let avg_target = buffer.iter().map(|(_, t)| *t).sum::<f64>() / buffer.len() as f64;
            st.bias += self.config.learning_rate * avg_target;
            debug!(
                "GBT fit a new stump: feature={}, threshold={:.4}, ensemble={}",
                feature,
                threshold,
                st.trees.len()
            );
        }
        Ok(())
    }

    fn predict(&self, features: &Array1<f64>) -> RegressionResult<f64> {
        self.check_dim(features)?;
        let observed = self.state.read().samples;
        if observed < self.config.min_samples {
            return Err(RegressionError::NotReady {
                observed: observed as usize,
                required: self.config.min_samples as usize,
            });
        }
        Ok(self.ensemble_predict(features))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lin_target(x: &Array1<f64>) -> f64 {
        // y = 2 x0 - 3 x1 + 5 x2 + 7
        2.0 * x[0] - 3.0 * x[1] + 5.0 * x[2] + 7.0
    }

    #[test]
    fn linear_regressor_converges_on_linear_target() {
        let cfg = LinearConfig {
            n_features: 3,
            learning_rate: 0.05,
            l2: 0.0,
            standardise_inputs: true,
            min_samples: 5,
        };
        let model = OnlineLinearRegressor::new(cfg).expect("ok");
        for i in 0..2000 {
            let x = Array1::from_vec(vec![
                ((i % 13) as f64) * 0.5,
                ((i % 7) as f64) * 0.25,
                ((i % 5) as f64) * 1.0,
            ]);
            let y = lin_target(&x);
            model.observe(&x, y).expect("observe");
        }
        let probe = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let pred = model.predict(&probe).expect("ready");
        let want = lin_target(&probe); // 2 - 3 + 5 + 7 = 11
        assert!(
            (pred - want).abs() < 1.5,
            "linear regressor diverged: pred={pred}, want={want}"
        );
    }

    #[test]
    fn linear_regressor_dimension_mismatch() {
        let cfg = LinearConfig {
            n_features: 3,
            ..Default::default()
        };
        let model = OnlineLinearRegressor::new(cfg).expect("ok");
        let bad = Array1::from_vec(vec![1.0, 2.0]);
        let err = model.observe(&bad, 1.0).expect_err("should fail");
        assert!(matches!(err, RegressionError::DimensionMismatch { .. }));
    }

    #[test]
    fn linear_regressor_not_ready() {
        let cfg = LinearConfig {
            n_features: 2,
            min_samples: 10,
            ..Default::default()
        };
        let model = OnlineLinearRegressor::new(cfg).expect("ok");
        let probe = Array1::from_vec(vec![0.0, 0.0]);
        let err = model.predict(&probe).expect_err("should fail");
        assert!(matches!(err, RegressionError::NotReady { .. }));
    }

    #[test]
    fn linear_regressor_invalid_config() {
        let cfg = LinearConfig {
            n_features: 0,
            ..Default::default()
        };
        let err = OnlineLinearRegressor::new(cfg).expect_err("should fail");
        assert!(matches!(err, RegressionError::InvalidConfig(_)));
    }

    fn nonlinear_target(x: &Array1<f64>) -> f64 {
        // y = x0^2 + 0.5 sin(x1)
        x[0] * x[0] + 0.5 * x[1].sin()
    }

    #[test]
    fn gbt_regressor_learns_nonlinear_pattern() {
        let cfg = GbtConfig {
            n_features: 2,
            max_trees: 24,
            learning_rate: 0.1,
            fit_buffer_size: 16,
            min_samples: 32,
        };
        let model = StreamingGradientBoostedRegressor::new(cfg).expect("ok");
        for i in 0..1500 {
            let x0 = ((i % 11) as f64) * 0.2 - 1.0;
            let x1 = ((i % 17) as f64) * 0.3 - 2.5;
            let x = Array1::from_vec(vec![x0, x1]);
            let y = nonlinear_target(&x);
            model.observe(&x, y).expect("observe");
        }
        let probe = Array1::from_vec(vec![0.5, 1.0]);
        let pred = model.predict(&probe).expect("ready");
        let want = nonlinear_target(&probe);
        // Stumps cannot perfectly represent a quadratic, but should track within
        // a generous error band.
        assert!(
            (pred - want).abs() < 1.5,
            "gbt regressor too far off: pred={pred}, want={want}"
        );
        assert!(model.ensemble_size() > 0, "ensemble should be non-empty");
    }

    #[test]
    fn gbt_ensemble_size_capped() {
        let cfg = GbtConfig {
            n_features: 1,
            max_trees: 4,
            learning_rate: 0.1,
            fit_buffer_size: 4,
            min_samples: 4,
        };
        let model = StreamingGradientBoostedRegressor::new(cfg).expect("ok");
        // Push enough observations to trigger many fits beyond the cap.
        for i in 0..200 {
            let x = Array1::from_vec(vec![i as f64]);
            model.observe(&x, x[0] * 0.3).expect("observe");
        }
        assert!(model.ensemble_size() <= 4);
    }

    #[test]
    fn gbt_invalid_config() {
        let cfg = GbtConfig {
            max_trees: 0,
            ..Default::default()
        };
        let err = StreamingGradientBoostedRegressor::new(cfg).expect_err("should fail");
        assert!(matches!(err, RegressionError::InvalidConfig(_)));
    }
}
