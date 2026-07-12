//! # Online classification models for streaming
//!
//! Two complementary models for binary and multi-class classification on
//! streams:
//!
//! * [`OnlineLogisticClassifier`] — multinomial logistic regression with
//!   per-class weight vectors trained via stochastic gradient descent on the
//!   cross-entropy loss. Suitable when classes are roughly linearly separable
//!   in feature space.
//! * [`StreamingKnnClassifier`] — sliding-window k-nearest-neighbour
//!   classifier that keeps the latest `window_size` labelled observations.
//!   The implementation is exact (Euclidean distance, no approximation) and
//!   appropriate for low-dimensional features where the working set fits in
//!   memory.
//!
//! Both classifiers implement [`StreamClassifier`].

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::debug;

use scirs2_core::ndarray_ext::Array1;

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by the classification models.
#[derive(Debug, Error)]
pub enum ClassificationError {
    /// Feature vector length did not match the model's configured dimension.
    #[error("feature dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    /// The requested class label is outside the configured range.
    #[error("class label {label} out of range (n_classes = {n_classes})")]
    LabelOutOfRange { label: usize, n_classes: usize },
    /// Model has not seen enough samples yet.
    #[error("model not ready: only {observed} of {required} samples observed")]
    NotReady { observed: usize, required: usize },
    /// Configuration value out of range.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Convenience alias.
pub type ClassificationResult<T> = std::result::Result<T, ClassificationError>;

// ─── StreamClassifier trait ─────────────────────────────────────────────────

/// Result of a single classification call.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassPrediction {
    /// Predicted class label (0-indexed).
    pub label: usize,
    /// Per-class scores; for probabilistic models these are normalised to a
    /// proper distribution. For [`StreamingKnnClassifier`] they are vote counts
    /// divided by `k` so still in `[0, 1]`.
    pub scores: Vec<f64>,
}

/// Common interface for streaming classifiers.
pub trait StreamClassifier: Send + Sync {
    /// Number of input features.
    fn n_features(&self) -> usize;
    /// Number of classes.
    fn n_classes(&self) -> usize;
    /// Number of `(features, label)` samples observed so far.
    fn n_observed(&self) -> u64;
    /// Update the model with a single labelled observation.
    fn observe(&self, features: &Array1<f64>, label: usize) -> ClassificationResult<()>;
    /// Produce a class prediction.
    fn predict(&self, features: &Array1<f64>) -> ClassificationResult<ClassPrediction>;
}

// ─── OnlineLogisticClassifier ───────────────────────────────────────────────

/// Configuration for [`OnlineLogisticClassifier`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticConfig {
    /// Number of input features.
    pub n_features: usize,
    /// Number of classes.
    pub n_classes: usize,
    /// SGD learning rate.
    pub learning_rate: f64,
    /// L2 regularisation strength (`0.0` disables).
    pub l2: f64,
    /// Min samples required before [`OnlineLogisticClassifier::predict`] returns.
    pub min_samples: u64,
}

impl Default for LogisticConfig {
    fn default() -> Self {
        Self {
            n_features: 4,
            n_classes: 2,
            learning_rate: 0.05,
            l2: 0.0,
            min_samples: 5,
        }
    }
}

impl LogisticConfig {
    fn validate(&self) -> ClassificationResult<()> {
        if self.n_features == 0 {
            return Err(ClassificationError::InvalidConfig(
                "n_features must be > 0".into(),
            ));
        }
        if self.n_classes < 2 {
            return Err(ClassificationError::InvalidConfig(
                "n_classes must be >= 2".into(),
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(ClassificationError::InvalidConfig(
                "learning_rate must be positive".into(),
            ));
        }
        if !self.l2.is_finite() || self.l2 < 0.0 {
            return Err(ClassificationError::InvalidConfig(
                "l2 must be non-negative".into(),
            ));
        }
        Ok(())
    }
}

/// Inner mutable state of [`OnlineLogisticClassifier`].
struct LogisticState {
    /// Shape `(n_classes, n_features)` flattened row-major.
    weights: Vec<Array1<f64>>,
    bias: Array1<f64>,
    samples: u64,
    last_loss: f64,
}

/// Multinomial logistic regression trained via online SGD.
pub struct OnlineLogisticClassifier {
    config: LogisticConfig,
    state: Arc<RwLock<LogisticState>>,
}

impl std::fmt::Debug for OnlineLogisticClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.state.read();
        f.debug_struct("OnlineLogisticClassifier")
            .field("config", &self.config)
            .field("samples", &st.samples)
            .field("last_loss", &st.last_loss)
            .finish()
    }
}

impl OnlineLogisticClassifier {
    /// Build from config.
    pub fn new(config: LogisticConfig) -> ClassificationResult<Self> {
        config.validate()?;
        let weights = (0..config.n_classes)
            .map(|_| Array1::zeros(config.n_features))
            .collect::<Vec<_>>();
        let state = LogisticState {
            weights,
            bias: Array1::zeros(config.n_classes),
            samples: 0,
            last_loss: 0.0,
        };
        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Last per-sample cross-entropy.
    pub fn last_loss(&self) -> f64 {
        self.state.read().last_loss
    }

    fn check_dim(&self, x: &Array1<f64>) -> ClassificationResult<()> {
        if x.len() != self.config.n_features {
            return Err(ClassificationError::DimensionMismatch {
                expected: self.config.n_features,
                actual: x.len(),
            });
        }
        Ok(())
    }

    fn check_label(&self, label: usize) -> ClassificationResult<()> {
        if label >= self.config.n_classes {
            return Err(ClassificationError::LabelOutOfRange {
                label,
                n_classes: self.config.n_classes,
            });
        }
        Ok(())
    }

    /// Forward pass: per-class score vector (logits → softmax).
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let st = self.state.read();
        let mut logits = Array1::zeros(self.config.n_classes);
        for c in 0..self.config.n_classes {
            let mut z = st.bias[c];
            for i in 0..self.config.n_features {
                z += st.weights[c][i] * x[i];
            }
            logits[c] = z;
        }
        softmax(&logits)
    }
}

fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let mut m = f64::NEG_INFINITY;
    for v in logits.iter() {
        if *v > m {
            m = *v;
        }
    }
    let mut sum = 0.0;
    let mut out = Array1::zeros(logits.len());
    for i in 0..logits.len() {
        let e = (logits[i] - m).exp();
        out[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        for i in 0..out.len() {
            out[i] /= sum;
        }
    }
    out
}

fn argmax(scores: &Array1<f64>) -> usize {
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, v) in scores.iter().enumerate() {
        if *v > best_score {
            best_score = *v;
            best = i;
        }
    }
    best
}

impl StreamClassifier for OnlineLogisticClassifier {
    fn n_features(&self) -> usize {
        self.config.n_features
    }

    fn n_classes(&self) -> usize {
        self.config.n_classes
    }

    fn n_observed(&self) -> u64 {
        self.state.read().samples
    }

    fn observe(&self, features: &Array1<f64>, label: usize) -> ClassificationResult<()> {
        self.check_dim(features)?;
        self.check_label(label)?;
        let probs = self.forward(features);

        // Compute cross-entropy loss.
        let p_label = probs[label].max(1e-12);
        let loss = -p_label.ln();

        let lr = self.config.learning_rate;
        let l2 = self.config.l2;

        let mut st = self.state.write();
        for c in 0..self.config.n_classes {
            let target = if c == label { 1.0 } else { 0.0 };
            let err = probs[c] - target;
            for i in 0..self.config.n_features {
                let grad = err * features[i] + l2 * st.weights[c][i];
                st.weights[c][i] -= lr * grad;
            }
            st.bias[c] -= lr * err;
        }
        st.samples += 1;
        st.last_loss = loss;
        Ok(())
    }

    fn predict(&self, features: &Array1<f64>) -> ClassificationResult<ClassPrediction> {
        self.check_dim(features)?;
        let observed = self.state.read().samples;
        if observed < self.config.min_samples {
            return Err(ClassificationError::NotReady {
                observed: observed as usize,
                required: self.config.min_samples as usize,
            });
        }
        let probs = self.forward(features);
        let label = argmax(&probs);
        Ok(ClassPrediction {
            label,
            scores: probs.to_vec(),
        })
    }
}

// ─── StreamingKnnClassifier ────────────────────────────────────────────────

/// Configuration for [`StreamingKnnClassifier`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnConfig {
    /// Number of input features.
    pub n_features: usize,
    /// Number of classes (used for score vector sizing).
    pub n_classes: usize,
    /// Number of neighbours queried per prediction.
    pub k: usize,
    /// Sliding window of the last N observations.
    pub window_size: usize,
    /// If true, weight votes inversely with distance.
    pub distance_weighted: bool,
    /// Minimum number of samples required before [`StreamingKnnClassifier::predict`] returns.
    pub min_samples: u64,
}

impl Default for KnnConfig {
    fn default() -> Self {
        Self {
            n_features: 4,
            n_classes: 2,
            k: 5,
            window_size: 200,
            distance_weighted: false,
            min_samples: 5,
        }
    }
}

impl KnnConfig {
    fn validate(&self) -> ClassificationResult<()> {
        if self.n_features == 0 {
            return Err(ClassificationError::InvalidConfig(
                "n_features must be > 0".into(),
            ));
        }
        if self.n_classes < 2 {
            return Err(ClassificationError::InvalidConfig(
                "n_classes must be >= 2".into(),
            ));
        }
        if self.k == 0 || self.window_size == 0 {
            return Err(ClassificationError::InvalidConfig(
                "k and window_size must be > 0".into(),
            ));
        }
        if self.k > self.window_size {
            return Err(ClassificationError::InvalidConfig(
                "k must be <= window_size".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct LabeledPoint {
    features: Array1<f64>,
    label: usize,
}

struct KnnState {
    window: VecDeque<LabeledPoint>,
    samples: u64,
}

/// Sliding-window k-nearest-neighbour classifier.
pub struct StreamingKnnClassifier {
    config: KnnConfig,
    state: Arc<RwLock<KnnState>>,
}

impl std::fmt::Debug for StreamingKnnClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.state.read();
        f.debug_struct("StreamingKnnClassifier")
            .field("config", &self.config)
            .field("samples", &st.samples)
            .field("window_count", &st.window.len())
            .finish()
    }
}

impl StreamingKnnClassifier {
    /// Build from config.
    pub fn new(config: KnnConfig) -> ClassificationResult<Self> {
        config.validate()?;
        let state = KnnState {
            window: VecDeque::with_capacity(config.window_size),
            samples: 0,
        };
        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Number of points currently held in the window.
    pub fn window_count(&self) -> usize {
        self.state.read().window.len()
    }

    fn check_dim(&self, x: &Array1<f64>) -> ClassificationResult<()> {
        if x.len() != self.config.n_features {
            return Err(ClassificationError::DimensionMismatch {
                expected: self.config.n_features,
                actual: x.len(),
            });
        }
        Ok(())
    }

    fn check_label(&self, label: usize) -> ClassificationResult<()> {
        if label >= self.config.n_classes {
            return Err(ClassificationError::LabelOutOfRange {
                label,
                n_classes: self.config.n_classes,
            });
        }
        Ok(())
    }
}

fn euclid(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

impl StreamClassifier for StreamingKnnClassifier {
    fn n_features(&self) -> usize {
        self.config.n_features
    }

    fn n_classes(&self) -> usize {
        self.config.n_classes
    }

    fn n_observed(&self) -> u64 {
        self.state.read().samples
    }

    fn observe(&self, features: &Array1<f64>, label: usize) -> ClassificationResult<()> {
        self.check_dim(features)?;
        self.check_label(label)?;
        let mut st = self.state.write();
        if st.window.len() >= self.config.window_size {
            st.window.pop_front();
        }
        st.window.push_back(LabeledPoint {
            features: features.clone(),
            label,
        });
        st.samples += 1;
        Ok(())
    }

    fn predict(&self, features: &Array1<f64>) -> ClassificationResult<ClassPrediction> {
        self.check_dim(features)?;
        let st = self.state.read();
        if st.samples < self.config.min_samples {
            return Err(ClassificationError::NotReady {
                observed: st.samples as usize,
                required: self.config.min_samples as usize,
            });
        }

        // Compute distances to every point in the window.
        let mut candidates: Vec<(f64, usize)> = st
            .window
            .iter()
            .map(|p| (euclid(&p.features, features), p.label))
            .collect();
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.config.k.min(candidates.len());
        let neighbours = &candidates[..k];

        let mut votes: HashMap<usize, f64> = HashMap::new();
        for (dist, label) in neighbours {
            let weight = if self.config.distance_weighted {
                1.0 / (dist + 1e-9)
            } else {
                1.0
            };
            *votes.entry(*label).or_insert(0.0) += weight;
        }

        let mut scores = vec![0.0; self.config.n_classes];
        let total: f64 = votes.values().copied().sum();
        for (label, weight) in &votes {
            if *label < self.config.n_classes {
                scores[*label] = if total > 0.0 { *weight / total } else { 0.0 };
            }
        }
        let mut best = 0usize;
        let mut best_score = -1.0;
        for (i, s) in scores.iter().enumerate() {
            if *s > best_score {
                best_score = *s;
                best = i;
            }
        }
        debug!("knn predict: votes={votes:?}");
        Ok(ClassPrediction {
            label: best,
            scores,
        })
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linsep_label(x: &Array1<f64>) -> usize {
        // Class 1 if x0 + x1 >= 0, else class 0.
        if x[0] + x[1] >= 0.0 {
            1
        } else {
            0
        }
    }

    #[test]
    fn logistic_classifier_learns_linear_separation() {
        let cfg = LogisticConfig {
            n_features: 2,
            n_classes: 2,
            learning_rate: 0.1,
            l2: 0.0,
            min_samples: 5,
        };
        let model = OnlineLogisticClassifier::new(cfg).expect("ok");
        for i in 0..2000 {
            let x0 = ((i % 31) as f64) * 0.2 - 3.0;
            let x1 = ((i % 23) as f64) * 0.3 - 3.0;
            let x = Array1::from_vec(vec![x0, x1]);
            let y = linsep_label(&x);
            model.observe(&x, y).expect("observe");
        }
        let mut correct = 0;
        let mut total = 0;
        for i in 0..400 {
            let x0 = ((i * 7) as f64).sin();
            let x1 = ((i * 11) as f64).cos();
            let x = Array1::from_vec(vec![x0, x1]);
            let pred = model.predict(&x).expect("ready");
            if pred.label == linsep_label(&x) {
                correct += 1;
            }
            total += 1;
        }
        let accuracy = correct as f64 / total as f64;
        assert!(
            accuracy > 0.9,
            "logistic regressor accuracy too low: {accuracy}"
        );
    }

    #[test]
    fn logistic_classifier_label_out_of_range() {
        let cfg = LogisticConfig {
            n_features: 2,
            n_classes: 3,
            ..Default::default()
        };
        let model = OnlineLogisticClassifier::new(cfg).expect("ok");
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let err = model.observe(&x, 5).expect_err("should fail");
        assert!(matches!(err, ClassificationError::LabelOutOfRange { .. }));
    }

    #[test]
    fn logistic_classifier_dimension_mismatch() {
        let cfg = LogisticConfig {
            n_features: 3,
            ..Default::default()
        };
        let model = OnlineLogisticClassifier::new(cfg).expect("ok");
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let err = model.observe(&x, 0).expect_err("should fail");
        assert!(matches!(err, ClassificationError::DimensionMismatch { .. }));
    }

    #[test]
    fn logistic_classifier_invalid_config() {
        let cfg = LogisticConfig {
            n_classes: 1,
            ..Default::default()
        };
        let err = OnlineLogisticClassifier::new(cfg).expect_err("should fail");
        assert!(matches!(err, ClassificationError::InvalidConfig(_)));
    }

    fn three_cluster_label(x: &Array1<f64>) -> usize {
        // Three radial clusters around (0,0), (5,5), (-5, 5).
        let centres = [(0.0, 0.0), (5.0, 5.0), (-5.0, 5.0)];
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for (i, c) in centres.iter().enumerate() {
            let dx = x[0] - c.0;
            let dy = x[1] - c.1;
            let d = (dx * dx + dy * dy).sqrt();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    #[test]
    fn knn_classifier_three_clusters() {
        let cfg = KnnConfig {
            n_features: 2,
            n_classes: 3,
            k: 5,
            window_size: 300,
            distance_weighted: false,
            min_samples: 30,
        };
        let model = StreamingKnnClassifier::new(cfg).expect("ok");
        // Train: feed cluster centroids with deterministic offsets.
        let centres = [(0.0, 0.0), (5.0, 5.0), (-5.0, 5.0)];
        for i in 0..300 {
            let cluster = i % 3;
            let c = centres[cluster];
            let dx = ((i / 3) as f64).sin() * 0.2;
            let dy = ((i / 3) as f64).cos() * 0.2;
            let x = Array1::from_vec(vec![c.0 + dx, c.1 + dy]);
            model.observe(&x, cluster).expect("observe");
        }

        // Test points near each centre.
        let probes = [
            (Array1::from_vec(vec![0.1, -0.1]), 0),
            (Array1::from_vec(vec![5.1, 4.9]), 1),
            (Array1::from_vec(vec![-5.1, 5.2]), 2),
        ];
        for (probe, expected) in probes {
            let pred = model.predict(&probe).expect("ready");
            assert_eq!(
                pred.label, expected,
                "knn misclassified probe {probe:?} as {} (want {expected})",
                pred.label
            );
            let total: f64 = pred.scores.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-6 || total == 0.0,
                "scores should normalise to 1: {:?}",
                pred.scores
            );
        }
    }

    #[test]
    fn knn_distance_weighted_label_check() {
        let cfg = KnnConfig {
            n_features: 1,
            n_classes: 2,
            k: 3,
            window_size: 5,
            distance_weighted: true,
            min_samples: 3,
        };
        let model = StreamingKnnClassifier::new(cfg).expect("ok");
        // 1 close-by class-0 point + 2 far class-1 points.
        model.observe(&Array1::from_vec(vec![0.0]), 0).expect("ok");
        model.observe(&Array1::from_vec(vec![10.0]), 1).expect("ok");
        model.observe(&Array1::from_vec(vec![11.0]), 1).expect("ok");

        let probe = Array1::from_vec(vec![0.1]);
        let pred = model.predict(&probe).expect("ready");
        assert_eq!(
            pred.label, 0,
            "with distance-weighted votes the close class-0 should dominate"
        );
    }

    #[test]
    fn knn_invalid_config() {
        let cfg = KnnConfig {
            k: 10,
            window_size: 5,
            ..Default::default()
        };
        let err = StreamingKnnClassifier::new(cfg).expect_err("should fail");
        assert!(matches!(err, ClassificationError::InvalidConfig(_)));
    }

    #[test]
    fn knn_window_eviction() {
        let cfg = KnnConfig {
            n_features: 1,
            n_classes: 2,
            k: 3,
            window_size: 3,
            distance_weighted: false,
            min_samples: 3,
        };
        let model = StreamingKnnClassifier::new(cfg).expect("ok");
        for i in 0..100 {
            let x = Array1::from_vec(vec![i as f64]);
            model.observe(&x, (i % 2) as usize).expect("observe");
        }
        assert_eq!(model.window_count(), 3);
    }
}
