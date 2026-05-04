//! Ensemble Embedding Methods (v0.3.0)
//!
//! Aggregates multiple embedding models using three strategies:
//! - **Voting** (mean pooling): average embeddings across all models element-wise.
//! - **WeightedAverage**: performance-weighted mean (weights from validation cosine similarity).
//! - **Stacking**: a two-layer meta-learner (linear + ReLU + linear) trained on concatenated
//!   model outputs; requires a held-out validation set.
//!
//! ## Design
//!
//! Each "model" in the ensemble is represented as a boxed function `Fn(&str) -> Vec<f64>`
//! (same contract as `ModelVariant` in the A/B testing module).  This keeps the
//! ensemble decoupled from specific KGE implementations.
//!
//! ```rust,no_run
//! use oxirs_embed::ensemble::{EnsembleConfig, EnsembleEmbedder, EnsembleStrategy};
//!
//! let models: Vec<Box<dyn Fn(&str) -> Vec<f64> + Send + Sync>> = vec![
//!     Box::new(|_key: &str| vec![1.0f64; 16]),
//!     Box::new(|_key: &str| vec![2.0f64; 16]),
//! ];
//! let config = EnsembleConfig {
//!     strategy: EnsembleStrategy::Voting,
//!     output_dim: 16,
//!     ..Default::default()
//! };
//! let embedder = EnsembleEmbedder::new(models, config).expect("valid config");
//! let embedding = embedder.embed("entity:Alice").expect("embedding");
//! assert_eq!(embedding.len(), 16);
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// EnsembleStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregation strategy for combining model embeddings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EnsembleStrategy {
    /// Element-wise mean over all model outputs.
    Voting,
    /// Weighted element-wise mean. Weights must be supplied via
    /// [`EnsembleEmbedder::set_weights`]; default is uniform.
    WeightedAverage,
    /// Two-layer meta-learner trained on concatenated model outputs.
    /// Requires calling [`EnsembleEmbedder::fit_stacking`] before inference.
    Stacking,
}

/// Configuration for [`EnsembleEmbedder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Aggregation strategy.
    pub strategy: EnsembleStrategy,
    /// Dimensionality of each individual model's output.
    pub output_dim: usize,
    /// Hidden dimension of the stacking meta-learner.
    pub stacking_hidden_dim: usize,
    /// Learning rate for stacking meta-learner training.
    pub stacking_lr: f64,
    /// Number of gradient-descent epochs for stacking.
    pub stacking_epochs: usize,
    /// L2-normalize final embedding (all strategies).
    pub normalize: bool,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            strategy: EnsembleStrategy::Voting,
            output_dim: 64,
            stacking_hidden_dim: 128,
            stacking_lr: 0.01,
            stacking_epochs: 50,
            normalize: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stacking meta-learner (linear → ReLU → linear)
// ─────────────────────────────────────────────────────────────────────────────

/// A minimal two-layer MLP meta-learner for the stacking strategy.
///
/// Architecture: `[concat_dim] → hidden → ReLU → output_dim`
struct StackingMLP {
    /// Weight matrix W1: shape [hidden, concat_dim]
    w1: Vec<Vec<f64>>,
    /// Bias b1: shape [hidden]
    b1: Vec<f64>,
    /// Weight matrix W2: shape [output_dim, hidden]
    w2: Vec<Vec<f64>>,
    /// Bias b2: shape [output_dim]
    b2: Vec<f64>,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl StackingMLP {
    /// Initialise weights with Xavier uniform (±sqrt(6/(fan_in+fan_out))).
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut state = seed.wrapping_add(1);
        let mut lcg = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // map to (-1, 1)
            (((state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
        };

        let xavier1 = (6.0_f64 / (input_dim + hidden_dim) as f64).sqrt();
        let xavier2 = (6.0_f64 / (hidden_dim + output_dim) as f64).sqrt();

        let w1 = (0..hidden_dim)
            .map(|_| (0..input_dim).map(|_| lcg() * xavier1).collect())
            .collect();
        let b1 = vec![0.0; hidden_dim];
        let w2 = (0..output_dim)
            .map(|_| (0..hidden_dim).map(|_| lcg() * xavier2).collect())
            .collect();
        let b2 = vec![0.0; output_dim];

        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Forward pass: returns hidden activations and output.
    fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Hidden layer
        let mut h = vec![0.0; self.hidden_dim];
        for (i, hi) in h.iter_mut().enumerate() {
            let dot: f64 = self.w1[i].iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
            *hi = (dot + self.b1[i]).max(0.0); // ReLU
        }
        // Output layer
        let mut out = vec![0.0; self.output_dim];
        for (i, oi) in out.iter_mut().enumerate() {
            let dot: f64 = self.w2[i].iter().zip(h.iter()).map(|(w, hi)| w * hi).sum();
            *oi = dot + self.b2[i];
        }
        (h, out)
    }

    /// Single SGD step with MSE loss given target `y`.
    fn backward_step(&mut self, x: &[f64], y: &[f64], lr: f64) {
        let (h, out) = self.forward(x);

        // Output layer gradients (MSE derivative = 2*(out - y))
        let d_out: Vec<f64> = out
            .iter()
            .zip(y.iter())
            .map(|(o, t)| 2.0 * (o - t))
            .collect();

        // Gradient w2 and b2
        for (i, di) in d_out.iter().enumerate() {
            for (j, hj) in h.iter().enumerate() {
                self.w2[i][j] -= lr * di * hj;
            }
            self.b2[i] -= lr * di;
        }

        // Backprop through ReLU into hidden layer
        let mut d_h = vec![0.0; self.hidden_dim];
        for (j, dj) in d_h.iter_mut().enumerate() {
            let back: f64 = (0..self.output_dim).map(|i| d_out[i] * self.w2[i][j]).sum();
            *dj = if h[j] > 0.0 { back } else { 0.0 };
        }

        // Gradient w1 and b1
        for (i, di) in d_h.iter().enumerate() {
            for (j, xj) in x.iter().enumerate() {
                self.w1[i][j] -= lr * di * xj;
            }
            self.b1[i] -= lr * di;
        }
    }

    fn predict(&self, x: &[f64]) -> Vec<f64> {
        self.forward(x).1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EnsembleEmbedder
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregates multiple embedding model functions under a single interface.
///
/// Each model is a boxed `Fn(&str) -> Vec<f64>` that maps an entity key to
/// an embedding vector.  All models must produce embeddings of the same
/// dimensionality (`config.output_dim`).
/// Type alias for an ensemble model function: maps an entity key to its embedding.
type EnsembleModel = Box<dyn Fn(&str) -> Vec<f64> + Send + Sync>;

pub struct EnsembleEmbedder {
    models: Vec<EnsembleModel>,
    config: EnsembleConfig,
    /// Per-model weights (only used for `WeightedAverage`).
    weights: Vec<f64>,
    /// Trained stacking meta-learner (only for `Stacking` strategy).
    stacking_mlp: Option<StackingMLP>,
}

impl std::fmt::Debug for EnsembleEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsembleEmbedder")
            .field("num_models", &self.models.len())
            .field("config", &self.config)
            .field("strategy", &self.config.strategy)
            .finish()
    }
}

impl EnsembleEmbedder {
    /// Create a new ensemble embedder.
    ///
    /// # Errors
    /// Returns an error if `models` is empty.
    pub fn new(models: Vec<EnsembleModel>, config: EnsembleConfig) -> Result<Self> {
        if models.is_empty() {
            return Err(anyhow!("EnsembleEmbedder requires at least one model"));
        }
        let n = models.len();
        let weights = vec![1.0 / n as f64; n]; // uniform by default
        Ok(Self {
            models,
            config,
            weights,
            stacking_mlp: None,
        })
    }

    /// Return the number of models in the ensemble.
    pub fn num_models(&self) -> usize {
        self.models.len()
    }

    /// Set per-model weights for `WeightedAverage`.
    ///
    /// Weights are automatically normalised to sum to 1.
    ///
    /// # Errors
    /// Returns an error if the weight vector length does not match the model count,
    /// or if any weight is negative or not finite, or if the sum is zero.
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<()> {
        if weights.len() != self.models.len() {
            return Err(anyhow!(
                "weight vector length {} != model count {}",
                weights.len(),
                self.models.len()
            ));
        }
        for &w in &weights {
            if !w.is_finite() || w < 0.0 {
                return Err(anyhow!("all weights must be non-negative and finite"));
            }
        }
        let sum: f64 = weights.iter().sum();
        if sum == 0.0 {
            return Err(anyhow!("weight sum must be > 0"));
        }
        self.weights = weights.iter().map(|w| w / sum).collect();
        Ok(())
    }

    /// Collect raw embeddings from every model for `key`.
    fn collect_embeddings(&self, key: &str) -> Result<Vec<Vec<f64>>> {
        let embeddings: Vec<Vec<f64>> = self.models.iter().map(|m| m(key)).collect();
        // Validate dimensionality
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != self.config.output_dim {
                return Err(anyhow!(
                    "model {} returned embedding of dimension {} but config.output_dim is {}",
                    i,
                    emb.len(),
                    self.config.output_dim
                ));
            }
        }
        Ok(embeddings)
    }

    /// L2-normalize a vector in-place. No-op if the norm is zero.
    fn l2_normalize(v: &mut [f64]) {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Produce an ensemble embedding for the given entity key.
    ///
    /// # Errors
    /// - Dimensionality mismatch between model output and `config.output_dim`.
    /// - `Stacking` strategy called before [`EnsembleEmbedder::fit_stacking`].
    pub fn embed(&self, key: &str) -> Result<Vec<f64>> {
        let embeddings = self.collect_embeddings(key)?;
        let mut result = match self.config.strategy {
            EnsembleStrategy::Voting => {
                let dim = self.config.output_dim;
                let mut agg = vec![0.0; dim];
                for emb in &embeddings {
                    for (a, e) in agg.iter_mut().zip(emb.iter()) {
                        *a += e;
                    }
                }
                let n = embeddings.len() as f64;
                agg.iter_mut().for_each(|a| *a /= n);
                agg
            }
            EnsembleStrategy::WeightedAverage => {
                let dim = self.config.output_dim;
                let mut agg = vec![0.0; dim];
                for (emb, &w) in embeddings.iter().zip(self.weights.iter()) {
                    for (a, e) in agg.iter_mut().zip(emb.iter()) {
                        *a += w * e;
                    }
                }
                agg
            }
            EnsembleStrategy::Stacking => {
                let mlp = self.stacking_mlp.as_ref().ok_or_else(|| {
                    anyhow!("Stacking strategy requires calling fit_stacking() first")
                })?;
                // Concatenate all embeddings into a single input vector
                let concat: Vec<f64> = embeddings.into_iter().flatten().collect();
                mlp.predict(&concat)
            }
        };
        if self.config.normalize {
            Self::l2_normalize(&mut result);
        }
        Ok(result)
    }

    /// Train the stacking meta-learner on a validation set.
    ///
    /// `validation_pairs` is a slice of `(key, target_embedding)` pairs where
    /// `target_embedding` is the ground-truth embedding (e.g. from a stronger
    /// reference model or link-prediction evaluation).
    ///
    /// # Errors
    /// Returns an error if the validation set is empty, or if any target
    /// embedding has the wrong dimensionality.
    pub fn fit_stacking(&mut self, validation_pairs: &[(&str, Vec<f64>)]) -> Result<()> {
        if validation_pairs.is_empty() {
            return Err(anyhow!("validation set must not be empty for stacking"));
        }
        let concat_dim = self.models.len() * self.config.output_dim;
        if self.stacking_mlp.is_none() {
            self.stacking_mlp = Some(StackingMLP::new(
                concat_dim,
                self.config.stacking_hidden_dim,
                self.config.output_dim,
                42,
            ));
        }
        for _epoch in 0..self.config.stacking_epochs {
            for (key, target) in validation_pairs {
                if target.len() != self.config.output_dim {
                    return Err(anyhow!(
                        "target embedding dimension {} != config.output_dim {}",
                        target.len(),
                        self.config.output_dim
                    ));
                }
                let embeddings = self.collect_embeddings(key)?;
                let concat: Vec<f64> = embeddings.into_iter().flatten().collect();
                if let Some(mlp) = &mut self.stacking_mlp {
                    mlp.backward_step(&concat, target, self.config.stacking_lr);
                }
            }
        }
        Ok(())
    }

    /// Compute the average cosine similarity between ensemble output and a reference
    /// model on the given validation keys. Useful for tuning weights.
    pub fn eval_cosine(
        &self,
        reference: &impl Fn(&str) -> Vec<f64>,
        validation_keys: &[&str],
    ) -> Result<f64> {
        if validation_keys.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0.0;
        for &key in validation_keys {
            let pred = self.embed(key)?;
            let ref_emb = reference(key);
            let dot: f64 = pred.iter().zip(ref_emb.iter()).map(|(a, b)| a * b).sum();
            let norm_pred: f64 = pred.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_ref: f64 = ref_emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            let cos = if norm_pred > 1e-10 && norm_ref > 1e-10 {
                (dot / (norm_pred * norm_ref)).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            total += cos;
        }
        Ok(total / validation_keys.len() as f64)
    }

    /// Derive performance-based weights from cosine similarity scores on a validation set.
    ///
    /// Each model's weight is proportional to its mean cosine similarity to a reference
    /// embedding. Models with zero weight (similarity ≤ 0) are assigned a small ε = 1e-6
    /// to keep them in the ensemble.
    pub fn derive_weights(
        &mut self,
        reference: &impl Fn(&str) -> Vec<f64>,
        validation_keys: &[&str],
    ) -> Result<()> {
        let mut scores = vec![0.0_f64; self.models.len()];
        for &key in validation_keys {
            let ref_emb = reference(key);
            for (i, model) in self.models.iter().enumerate() {
                let emb = model(key);
                let dot: f64 = emb.iter().zip(ref_emb.iter()).map(|(a, b)| a * b).sum();
                let na: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
                let nb: f64 = ref_emb.iter().map(|x| x * x).sum::<f64>().sqrt();
                let cos = if na > 1e-10 && nb > 1e-10 {
                    (dot / (na * nb)).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                scores[i] += cos;
            }
        }
        // Average and clamp to ε minimum
        let n = validation_keys.len().max(1) as f64;
        let weights: Vec<f64> = scores.iter().map(|s| (s / n).max(1e-6)).collect();
        self.set_weights(weights)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(value: f64, dim: usize) -> EnsembleModel {
        Box::new(move |_key: &str| vec![value; dim])
    }

    #[test]
    fn test_voting_mean() {
        let models: Vec<EnsembleModel> = vec![make_model(1.0, 4), make_model(3.0, 4)];
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::Voting,
            output_dim: 4,
            normalize: false,
            ..Default::default()
        };
        let embedder = EnsembleEmbedder::new(models, config).unwrap();
        let emb = embedder.embed("e1").unwrap();
        for v in &emb {
            assert!((v - 2.0).abs() < 1e-9, "expected 2.0 got {v}");
        }
    }

    #[test]
    fn test_weighted_average() {
        let models: Vec<EnsembleModel> = vec![make_model(0.0, 4), make_model(4.0, 4)];
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::WeightedAverage,
            output_dim: 4,
            normalize: false,
            ..Default::default()
        };
        let mut embedder = EnsembleEmbedder::new(models, config).unwrap();
        // 25% model-0 + 75% model-1 → expected = 3.0
        embedder.set_weights(vec![1.0, 3.0]).unwrap();
        let emb = embedder.embed("e1").unwrap();
        for v in &emb {
            assert!((v - 3.0).abs() < 1e-9, "expected 3.0 got {v}");
        }
    }

    #[test]
    fn test_zero_weight_model_excluded() {
        let models: Vec<EnsembleModel> = vec![
            make_model(100.0, 4), // would skew result if weighted
            make_model(1.0, 4),
        ];
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::WeightedAverage,
            output_dim: 4,
            normalize: false,
            ..Default::default()
        };
        let mut embedder = EnsembleEmbedder::new(models, config).unwrap();
        // Weight first model to near-zero (not exactly 0 — must be non-negative)
        embedder.set_weights(vec![1e-10, 1.0]).unwrap();
        let emb = embedder.embed("e1").unwrap();
        // Result should be ≈1.0 (dominated by second model)
        for v in &emb {
            assert!((v - 1.0).abs() < 0.01, "expected ≈1.0 got {v}");
        }
    }

    #[test]
    fn test_stacking_convergence() {
        let dim = 8;
        // Both models output constant vectors; target = 0.5*vec
        let models: Vec<EnsembleModel> = vec![make_model(1.0, dim), make_model(0.0, dim)];
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::Stacking,
            output_dim: dim,
            stacking_hidden_dim: 32,
            stacking_lr: 0.01,
            stacking_epochs: 200,
            normalize: false,
        };
        let mut embedder = EnsembleEmbedder::new(models, config).unwrap();
        // Validation: target = 0.5 for each dimension
        let targets: Vec<(&str, Vec<f64>)> = (0..20)
            .map(|i| {
                let key = Box::leak(format!("e{i}").into_boxed_str()) as &str;
                (key, vec![0.5; dim])
            })
            .collect();
        embedder.fit_stacking(&targets).unwrap();
        let emb = embedder.embed("e0").unwrap();
        // After training should converge towards 0.5
        for v in &emb {
            assert!(
                (v - 0.5).abs() < 0.2,
                "expected ≈0.5 after stacking, got {v}"
            );
        }
    }

    #[test]
    fn test_derive_weights() {
        let dim = 4;
        // Model 0 matches reference perfectly; model 1 is zeros (cosine = 0)
        let models: Vec<EnsembleModel> = vec![
            Box::new(move |_| vec![1.0; dim]),
            Box::new(move |_| vec![0.0; dim]),
        ];
        let reference = |_key: &str| vec![1.0f64; dim];
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::WeightedAverage,
            output_dim: dim,
            normalize: false,
            ..Default::default()
        };
        let mut embedder = EnsembleEmbedder::new(models, config).unwrap();
        let keys = vec!["e0", "e1", "e2"];
        embedder.derive_weights(&reference, &keys).unwrap();
        // Model 0 weight should be >> model 1 weight
        assert!(embedder.weights[0] > embedder.weights[1] * 100.0);
    }

    #[test]
    fn test_empty_models_rejected() {
        let models: Vec<EnsembleModel> = vec![];
        let config = EnsembleConfig::default();
        assert!(EnsembleEmbedder::new(models, config).is_err());
    }

    #[test]
    fn test_stacking_requires_fit() {
        let models: Vec<EnsembleModel> = vec![make_model(1.0, 4)];
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::Stacking,
            output_dim: 4,
            ..Default::default()
        };
        let embedder = EnsembleEmbedder::new(models, config).unwrap();
        assert!(embedder.embed("e1").is_err());
    }

    #[test]
    fn test_normalize_output() {
        let models: Vec<EnsembleModel> = vec![make_model(3.0, 4)];
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::Voting,
            output_dim: 4,
            normalize: true,
            ..Default::default()
        };
        let embedder = EnsembleEmbedder::new(models, config).unwrap();
        let emb = embedder.embed("e1").unwrap();
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "norm should be 1.0 after normalize, got {norm}"
        );
    }
}
