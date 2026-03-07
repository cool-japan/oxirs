//! Anomaly detection for time-series data.
//!
//! Provides multiple anomaly detection algorithms:
//! - Z-score based detection
//! - Interquartile Range (IQR) detection
//! - Exponentially Weighted Moving Average (EWMA) detection
//! - Simplified Isolation Forest detection

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// The type of detector that raised an anomaly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectorKind {
    /// Z-score (standard-score) based detection.
    ZScore,
    /// Interquartile Range outlier detection.
    Iqr,
    /// Exponentially Weighted Moving Average deviation.
    MovingAverage,
    /// Isolation Forest path-length scoring.
    IsolationForest,
}

/// A single detected anomalous point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyPoint {
    /// Index in the original slice.
    pub index: usize,
    /// Timestamp in milliseconds since epoch (0 if not provided).
    pub timestamp: i64,
    /// The original value at this position.
    pub value: f64,
    /// Anomaly score (higher = more anomalous).
    pub score: f64,
    /// Which detector produced this point.
    pub detector_type: DetectorKind,
}

/// Detection method selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyMethod {
    /// Use Z-score algorithm.
    ZScore,
    /// Use IQR algorithm.
    Iqr,
    /// Use EWMA algorithm.
    MovingAverage,
    /// Use Isolation Forest algorithm.
    IsolationForest,
}

/// Shared configuration for anomaly detectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyConfig {
    /// Score threshold above which a point is labelled anomalous.
    pub threshold: f64,
    /// Rolling window size used by window-based detectors.
    pub window_size: usize,
    /// Algorithm to use (informational; the concrete detector struct overrides).
    pub method: AnomalyMethod,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            threshold: 3.0,
            window_size: 20,
            method: AnomalyMethod::ZScore,
        }
    }
}

/// Core trait that every anomaly detector must implement.
pub trait AnomalyDetector: Send + Sync {
    /// Detect anomalous points in the supplied slice of values.
    ///
    /// Returns the subset of points considered anomalous together with their
    /// scores. Returns an error if the slice is too short for the algorithm.
    fn detect(&self, values: &[f64]) -> TsdbResult<Vec<AnomalyPoint>>;

    /// Same as `detect` but associates timestamps (milliseconds since epoch)
    /// with each value. The `timestamps` slice must have the same length as
    /// `values` when supplied.
    fn detect_with_timestamps(
        &self,
        values: &[f64],
        timestamps: &[i64],
    ) -> TsdbResult<Vec<AnomalyPoint>>;
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal statistical helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the mean of a slice.  Returns 0.0 for empty input.
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().copied().sum::<f64>() / xs.len() as f64
}

/// Compute the population standard deviation of a slice.
fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|v| (v - m).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Return the value at the given percentile using linear interpolation.
/// `xs` **must** be pre-sorted in ascending order.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ──────────────────────────────────────────────────────────────────────────────
// Z-Score Detector
// ──────────────────────────────────────────────────────────────────────────────

/// Flags points whose absolute Z-score exceeds the configured threshold.
///
/// The Z-score for value *v* is `|v − μ| / σ` where μ and σ are the mean and
/// standard deviation of the whole series.  This is a global, non-windowed
/// algorithm and works best for stationary series.
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    /// Threshold (number of standard deviations) above which a point is
    /// considered anomalous.  Default: 3.0.
    pub threshold: f64,
}

impl ZScoreDetector {
    /// Create a detector with the given threshold.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Default for ZScoreDetector {
    fn default() -> Self {
        Self::new(3.0)
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn detect(&self, values: &[f64]) -> TsdbResult<Vec<AnomalyPoint>> {
        self.detect_with_timestamps(values, &[])
    }

    fn detect_with_timestamps(
        &self,
        values: &[f64],
        timestamps: &[i64],
    ) -> TsdbResult<Vec<AnomalyPoint>> {
        if values.len() < 2 {
            return Err(TsdbError::Query(
                "Z-score detector requires at least 2 data points".to_string(),
            ));
        }
        if !timestamps.is_empty() && timestamps.len() != values.len() {
            return Err(TsdbError::Query(
                "timestamps length must match values length".to_string(),
            ));
        }

        let mu = mean(values);
        let sigma = std_dev(values);

        let mut anomalies = Vec::new();

        for (i, &v) in values.iter().enumerate() {
            let score = if sigma < f64::EPSILON {
                0.0
            } else {
                (v - mu).abs() / sigma
            };

            if score > self.threshold {
                anomalies.push(AnomalyPoint {
                    index: i,
                    timestamp: timestamps.get(i).copied().unwrap_or(0),
                    value: v,
                    score,
                    detector_type: DetectorKind::ZScore,
                });
            }
        }

        Ok(anomalies)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IQR Detector
// ──────────────────────────────────────────────────────────────────────────────

/// Flags outliers whose value falls outside `[Q1 − k·IQR, Q3 + k·IQR]`.
///
/// The multiplier *k* is the configured `threshold` (default 1.5 for mild
/// outliers; 3.0 is the conventional "extreme outlier" cutoff).
#[derive(Debug, Clone)]
pub struct IqrDetector {
    /// IQR multiplier (default 1.5).
    pub threshold: f64,
}

impl IqrDetector {
    /// Create a detector with the given IQR multiplier.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Default for IqrDetector {
    fn default() -> Self {
        Self::new(1.5)
    }
}

impl AnomalyDetector for IqrDetector {
    fn detect(&self, values: &[f64]) -> TsdbResult<Vec<AnomalyPoint>> {
        self.detect_with_timestamps(values, &[])
    }

    fn detect_with_timestamps(
        &self,
        values: &[f64],
        timestamps: &[i64],
    ) -> TsdbResult<Vec<AnomalyPoint>> {
        if values.len() < 4 {
            return Err(TsdbError::Query(
                "IQR detector requires at least 4 data points".to_string(),
            ));
        }
        if !timestamps.is_empty() && timestamps.len() != values.len() {
            return Err(TsdbError::Query(
                "timestamps length must match values length".to_string(),
            ));
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = percentile_sorted(&sorted, 25.0);
        let q3 = percentile_sorted(&sorted, 75.0);
        let iqr = q3 - q1;

        let lower = q1 - self.threshold * iqr;
        let upper = q3 + self.threshold * iqr;

        let mut anomalies = Vec::new();

        for (i, &v) in values.iter().enumerate() {
            if v < lower || v > upper {
                // Score: normalised distance outside the fence
                let score = if iqr < f64::EPSILON {
                    0.0
                } else if v < lower {
                    (lower - v) / iqr
                } else {
                    (v - upper) / iqr
                };

                anomalies.push(AnomalyPoint {
                    index: i,
                    timestamp: timestamps.get(i).copied().unwrap_or(0),
                    value: v,
                    score,
                    detector_type: DetectorKind::Iqr,
                });
            }
        }

        Ok(anomalies)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Moving Average (EWMA) Detector
// ──────────────────────────────────────────────────────────────────────────────

/// Detects anomalies as large deviations from an Exponentially Weighted
/// Moving Average (EWMA).
///
/// At each step *t* the EWMA is updated as:
///   `ewma_t = α · v_t + (1 − α) · ewma_{t−1}`
///
/// A point is anomalous when `|v_t − ewma_{t−1}|` exceeds `threshold` times
/// the running EWMA standard deviation.
#[derive(Debug, Clone)]
pub struct MovingAverageDetector {
    /// Smoothing factor in (0, 1).  Closer to 1 = more reactive.
    pub alpha: f64,
    /// Deviation threshold (multiplier of the EWMA std-dev).
    pub threshold: f64,
}

impl MovingAverageDetector {
    /// Create a detector with the given alpha and threshold.
    pub fn new(alpha: f64, threshold: f64) -> TsdbResult<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(TsdbError::Config(
                "EWMA alpha must be in the open interval (0, 1)".to_string(),
            ));
        }
        Ok(Self { alpha, threshold })
    }
}

impl Default for MovingAverageDetector {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            threshold: 3.0,
        }
    }
}

impl AnomalyDetector for MovingAverageDetector {
    fn detect(&self, values: &[f64]) -> TsdbResult<Vec<AnomalyPoint>> {
        self.detect_with_timestamps(values, &[])
    }

    fn detect_with_timestamps(
        &self,
        values: &[f64],
        timestamps: &[i64],
    ) -> TsdbResult<Vec<AnomalyPoint>> {
        if values.len() < 3 {
            return Err(TsdbError::Query(
                "EWMA detector requires at least 3 data points".to_string(),
            ));
        }
        if !timestamps.is_empty() && timestamps.len() != values.len() {
            return Err(TsdbError::Query(
                "timestamps length must match values length".to_string(),
            ));
        }

        // Initialise EWMA with the first value, variance with global variance.
        let mut ewma = values[0];
        let global_var = {
            let m = mean(values);
            values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64
        };
        let mut ewma_var = global_var.max(f64::EPSILON);

        let mut anomalies = Vec::new();

        for (i, &v) in values.iter().enumerate() {
            if i == 0 {
                // Seed – not scored.
                ewma = v;
                continue;
            }

            let deviation = (v - ewma).abs();
            let ewma_std = ewma_var.sqrt();
            let score = if ewma_std < f64::EPSILON {
                0.0
            } else {
                deviation / ewma_std
            };

            if score > self.threshold {
                anomalies.push(AnomalyPoint {
                    index: i,
                    timestamp: timestamps.get(i).copied().unwrap_or(0),
                    value: v,
                    score,
                    detector_type: DetectorKind::MovingAverage,
                });
            }

            // Update EWMA and variance estimate.
            ewma = self.alpha * v + (1.0 - self.alpha) * ewma;
            let new_dev = v - ewma;
            ewma_var = self.alpha * new_dev * new_dev + (1.0 - self.alpha) * ewma_var;
        }

        Ok(anomalies)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Isolation Forest Detector
// ──────────────────────────────────────────────────────────────────────────────

/// A lightweight, single-feature Isolation Forest for 1-D time-series.
///
/// The Isolation Forest principle: anomalies are rare and different, so they
/// are isolated near the root of a random binary partition tree.  The anomaly
/// score for a point is its average normalised path length across `n_trees`
/// random isolation trees.
///
/// This implementation operates on scalar values only (1-D feature space),
/// which is appropriate for univariate time-series anomaly detection.
#[derive(Debug, Clone)]
pub struct IsolationForestDetector {
    /// Number of isolation trees to build.
    pub n_trees: usize,
    /// Sub-sample size used when constructing each tree.
    pub sample_size: usize,
    /// Score threshold (0–1 scale; higher = more anomalous).
    pub threshold: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl IsolationForestDetector {
    /// Create a new detector.
    pub fn new(n_trees: usize, sample_size: usize, threshold: f64, seed: u64) -> TsdbResult<Self> {
        if n_trees == 0 {
            return Err(TsdbError::Config("n_trees must be > 0".to_string()));
        }
        if sample_size < 2 {
            return Err(TsdbError::Config("sample_size must be >= 2".to_string()));
        }
        Ok(Self {
            n_trees,
            sample_size,
            threshold,
            seed,
        })
    }
}

impl Default for IsolationForestDetector {
    fn default() -> Self {
        Self {
            n_trees: 100,
            sample_size: 256,
            threshold: 0.6,
            seed: 42,
        }
    }
}

// ── Isolation tree internals ──────────────────────────────────────────────────

/// A node in an isolation tree.
#[derive(Debug, Clone)]
enum ITreeNode {
    /// Internal split node.
    Split {
        split_value: f64,
        left: Box<ITreeNode>,
        right: Box<ITreeNode>,
    },
    /// External leaf (isolated sub-sample).
    Leaf { size: usize },
}

/// A minimal linear-congruential PRNG (no external crates).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 6_364_136_223_846_793_005,
        }
    }

    /// Next pseudo-random u64.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform usize in [0, n).
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Build a single isolation tree over `values`.
fn build_itree(values: &[f64], max_depth: usize, rng: &mut Lcg) -> ITreeNode {
    if values.len() <= 1 || max_depth == 0 {
        return ITreeNode::Leaf { size: values.len() };
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < f64::EPSILON {
        return ITreeNode::Leaf { size: values.len() };
    }

    // Random split in [min, max).
    let split = min + rng.next_f64() * (max - min);

    let left_vals: Vec<f64> = values.iter().cloned().filter(|&x| x < split).collect();
    let right_vals: Vec<f64> = values.iter().cloned().filter(|&x| x >= split).collect();

    ITreeNode::Split {
        split_value: split,
        left: Box::new(build_itree(&left_vals, max_depth - 1, rng)),
        right: Box::new(build_itree(&right_vals, max_depth - 1, rng)),
    }
}

/// Compute path length of `x` in a tree.
fn path_length(node: &ITreeNode, x: f64, current_depth: f64) -> f64 {
    match node {
        ITreeNode::Leaf { size } => current_depth + c_factor(*size),
        ITreeNode::Split {
            split_value,
            left,
            right,
        } => {
            if x < *split_value {
                path_length(left, x, current_depth + 1.0)
            } else {
                path_length(right, x, current_depth + 1.0)
            }
        }
    }
}

/// Expected path length for a BST of *n* nodes (harmonic approximation).
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n_f = n as f64;
    2.0 * (n_f.ln() + 0.5772156649) - 2.0 * (n_f - 1.0) / n_f
}

impl AnomalyDetector for IsolationForestDetector {
    fn detect(&self, values: &[f64]) -> TsdbResult<Vec<AnomalyPoint>> {
        self.detect_with_timestamps(values, &[])
    }

    fn detect_with_timestamps(
        &self,
        values: &[f64],
        timestamps: &[i64],
    ) -> TsdbResult<Vec<AnomalyPoint>> {
        if values.len() < self.sample_size.min(4) {
            return Err(TsdbError::Query(format!(
                "IsolationForest needs at least {} data points",
                self.sample_size.min(4)
            )));
        }
        if !timestamps.is_empty() && timestamps.len() != values.len() {
            return Err(TsdbError::Query(
                "timestamps length must match values length".to_string(),
            ));
        }

        let n = values.len();
        let sub = self.sample_size.min(n);
        let max_depth = (sub as f64).log2().ceil() as usize;
        let c = c_factor(sub);

        let mut rng = Lcg::new(self.seed);

        // Build forest on a (possibly sub-sampled) version of the data.
        let mut trees: Vec<ITreeNode> = Vec::with_capacity(self.n_trees);
        for _ in 0..self.n_trees {
            // Draw a sub-sample without replacement via partial Fisher-Yates.
            let mut indices: Vec<usize> = (0..n).collect();
            for j in 0..sub {
                let k = j + rng.next_usize(n - j);
                indices.swap(j, k);
            }
            let sample: Vec<f64> = indices[..sub].iter().map(|&i| values[i]).collect();
            trees.push(build_itree(&sample, max_depth, &mut rng));
        }

        // Score every point.
        let mut anomalies = Vec::new();
        for (i, &v) in values.iter().enumerate() {
            let avg_path =
                trees.iter().map(|t| path_length(t, v, 0.0)).sum::<f64>() / self.n_trees as f64;

            // Normalise to [0, 1]: closer to 1 = more anomalous.
            let score = if c < f64::EPSILON {
                0.0
            } else {
                2.0_f64.powf(-avg_path / c)
            };

            if score > self.threshold {
                anomalies.push(AnomalyPoint {
                    index: i,
                    timestamp: timestamps.get(i).copied().unwrap_or(0),
                    value: v,
                    score,
                    detector_type: DetectorKind::IsolationForest,
                });
            }
        }

        Ok(anomalies)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a clean sinusoidal series with a few spike anomalies injected at
    /// known positions.
    fn synthetic_series(n: usize, anomaly_positions: &[usize], spike: f64) -> Vec<f64> {
        let mut v: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.3).sin() * 5.0 + 50.0)
            .collect();
        for &pos in anomaly_positions {
            if pos < n {
                v[pos] += spike;
            }
        }
        v
    }

    #[test]
    fn zscore_detects_known_anomalies() {
        let anomaly_indices = [20usize, 60, 90];
        let data = synthetic_series(100, &anomaly_indices, 80.0);
        let detector = ZScoreDetector::new(3.0);
        let anomalies = detector.detect(&data).expect("zscore detect failed");
        assert!(!anomalies.is_empty(), "should detect at least one anomaly");
        for ap in &anomalies {
            assert_eq!(ap.detector_type, DetectorKind::ZScore);
            assert!(ap.score > 3.0);
        }
        let detected_indices: Vec<usize> = anomalies.iter().map(|a| a.index).collect();
        for &known in &anomaly_indices {
            assert!(
                detected_indices.contains(&known),
                "expected anomaly at index {known} to be detected"
            );
        }
    }

    #[test]
    fn zscore_error_on_too_few_points() {
        let detector = ZScoreDetector::new(3.0);
        assert!(detector.detect(&[1.0]).is_err());
    }

    #[test]
    fn zscore_with_timestamps() {
        let data = synthetic_series(50, &[25], 100.0);
        let timestamps: Vec<i64> = (0..50).map(|i| i as i64 * 1000).collect();
        let detector = ZScoreDetector::new(3.0);
        let anomalies = detector
            .detect_with_timestamps(&data, &timestamps)
            .expect("detect_with_timestamps failed");
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].timestamp, 25 * 1000);
    }

    #[test]
    fn iqr_detects_outliers() {
        // A series whose tail contains extreme outliers.
        let mut data: Vec<f64> = (0..80).map(|i| i as f64).collect();
        data.push(10_000.0); // extreme high
        data.push(-10_000.0); // extreme low
        let detector = IqrDetector::new(1.5);
        let anomalies = detector.detect(&data).expect("iqr detect failed");
        assert!(!anomalies.is_empty());
        for ap in &anomalies {
            assert_eq!(ap.detector_type, DetectorKind::Iqr);
            assert!(ap.score > 0.0);
        }
    }

    #[test]
    fn iqr_error_on_too_few_points() {
        let detector = IqrDetector::new(1.5);
        assert!(detector.detect(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn iqr_with_constant_series_no_anomalies() {
        let data = vec![5.0_f64; 100];
        let detector = IqrDetector::new(1.5);
        let anomalies = detector.detect(&data).expect("iqr const detect failed");
        assert!(
            anomalies.is_empty(),
            "constant series should yield no anomalies"
        );
    }

    #[test]
    fn ewma_detects_step_change() {
        // Stable series that abruptly jumps.
        let mut data: Vec<f64> = vec![10.0_f64; 50];
        data.extend(vec![10_000.0_f64; 1]); // single spike at index 50
        data.extend(vec![10.0_f64; 49]);
        let detector = MovingAverageDetector::new(0.2, 3.0).expect("ctor failed");
        let anomalies = detector.detect(&data).expect("ewma detect failed");
        assert!(!anomalies.is_empty());
        for ap in &anomalies {
            assert_eq!(ap.detector_type, DetectorKind::MovingAverage);
        }
    }

    #[test]
    fn ewma_invalid_alpha() {
        assert!(MovingAverageDetector::new(0.0, 3.0).is_err());
        assert!(MovingAverageDetector::new(1.0, 3.0).is_err());
    }

    #[test]
    fn ewma_error_on_too_few_points() {
        let detector = MovingAverageDetector::default();
        assert!(detector.detect(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn isolation_forest_detects_anomalies() {
        // Tightly clustered series with two extreme outliers.
        let mut data: Vec<f64> = (0..200).map(|i| (i as f64 % 5.0) + 10.0).collect();
        data[50] = 1_000.0;
        data[150] = -1_000.0;

        let detector =
            IsolationForestDetector::new(50, 64, 0.55, 12345).expect("iforest ctor failed");
        let anomalies = detector.detect(&data).expect("iforest detect failed");
        assert!(!anomalies.is_empty(), "should detect iforest anomalies");
        let detected_indices: Vec<usize> = anomalies.iter().map(|a| a.index).collect();
        assert!(
            detected_indices.contains(&50) || detected_indices.contains(&150),
            "at least one known anomaly must be detected"
        );
    }

    #[test]
    fn isolation_forest_invalid_params() {
        assert!(IsolationForestDetector::new(0, 64, 0.6, 1).is_err());
        assert!(IsolationForestDetector::new(10, 1, 0.6, 1).is_err());
    }

    #[test]
    fn isolation_forest_scores_in_range() {
        let data: Vec<f64> = (0..150).map(|i| i as f64).collect();
        let detector = IsolationForestDetector::new(20, 32, 0.0, 99).expect("ctor");
        let anomalies = detector.detect(&data).expect("detect");
        for ap in &anomalies {
            assert!(
                ap.score >= 0.0 && ap.score <= 1.0,
                "score must be in [0,1], got {}",
                ap.score
            );
        }
    }

    #[test]
    fn anomaly_config_default() {
        let cfg = AnomalyConfig::default();
        assert_eq!(cfg.threshold, 3.0);
        assert_eq!(cfg.window_size, 20);
    }
}
