//! Statistical Anomaly Detection for Time-Series Data.
//!
//! Implements four classical anomaly-detection algorithms:
//!
//! - **ZScore** – (x − μ) / σ > threshold
//! - **IQR** – x < Q1 − k×IQR or x > Q3 + k×IQR
//! - **MovingAverage** – deviation from rolling mean in sigma units
//! - **CUSUM** – cumulative sum control chart
//!
//! Each algorithm produces a list of `Anomaly` records and supports
//! multi-series batch detection.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_tsdb::anomaly_detection::{AnomalyDetector, Algorithm, DataPoint};
//!
//! let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
//! let data: Vec<DataPoint> = (0..100)
//!     .map(|i| DataPoint { timestamp: i, value: if i == 50 { 999.0 } else { 1.0 } })
//!     .collect();
//! let anomalies = detector.detect(&data);
//! assert_eq!(anomalies.len(), 1);
//! assert_eq!(anomalies[0].point.timestamp, 50);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// A single time-series observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DataPoint {
    /// Unix timestamp (or any monotonically increasing counter).
    pub timestamp: u64,
    /// Observed value.
    pub value: f64,
}

/// A detected anomalous data point together with its score and originating
/// algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// The data point flagged as anomalous.
    pub point: DataPoint,
    /// Anomaly score (higher = more anomalous; interpretation varies by
    /// algorithm).
    pub score: f64,
    /// Name of the algorithm that produced this anomaly.
    pub algorithm: String,
}

/// Summary statistics over a collection of detected anomalies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnomalyStats {
    /// Total number of anomalies detected.
    pub count: usize,
    /// Minimum anomaly score (or NaN if no anomalies).
    pub min_score: f64,
    /// Maximum anomaly score (or NaN if no anomalies).
    pub max_score: f64,
    /// Mean anomaly score (or NaN if no anomalies).
    pub mean_score: f64,
}

impl AnomalyStats {
    /// Compute statistics from a slice of anomalies.
    pub fn from_anomalies(anomalies: &[Anomaly]) -> Self {
        let count = anomalies.len();
        if count == 0 {
            return Self {
                count: 0,
                min_score: f64::NAN,
                max_score: f64::NAN,
                mean_score: f64::NAN,
            };
        }
        let mut min_score = f64::INFINITY;
        let mut max_score = f64::NEG_INFINITY;
        let mut total = 0.0f64;
        for a in anomalies {
            if a.score < min_score {
                min_score = a.score;
            }
            if a.score > max_score {
                max_score = a.score;
            }
            total += a.score;
        }
        Self {
            count,
            min_score,
            max_score,
            mean_score: total / count as f64,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm enum
// ─────────────────────────────────────────────────────────────────────────────

/// Anomaly detection algorithm selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Algorithm {
    /// Standard Z-score outlier detection.
    ///
    /// A point is anomalous when `|x - mean| / std > threshold`.
    ZScore {
        /// Deviation threshold in standard deviation units (e.g. 3.0).
        threshold: f64,
    },

    /// Interquartile Range outlier detection.
    ///
    /// A point is anomalous when `x < Q1 - multiplier×IQR` or
    /// `x > Q3 + multiplier×IQR`.
    IQR {
        /// IQR multiplier (e.g. 1.5 for mild, 3.0 for extreme outliers).
        multiplier: f64,
    },

    /// Moving-average deviation in sigma units.
    ///
    /// A point is anomalous when `|x - rolling_mean| / rolling_std > sigma`.
    MovingAverage {
        /// Size of the rolling window.
        window: usize,
        /// Deviation threshold in standard deviation units.
        sigma: f64,
    },

    /// Cumulative Sum (CUSUM) control chart.
    ///
    /// - `k` – reference value (allowable slack, usually `0.5 × sigma`)
    /// - `h` – decision threshold (chart alarm limit)
    CUSUM {
        /// Allowable slack parameter.
        k: f64,
        /// Decision threshold.
        h: f64,
    },
}

impl Algorithm {
    /// Human-readable name of the algorithm.
    pub fn name(&self) -> &'static str {
        match self {
            Algorithm::ZScore { .. } => "ZScore",
            Algorithm::IQR { .. } => "IQR",
            Algorithm::MovingAverage { .. } => "MovingAverage",
            Algorithm::CUSUM { .. } => "CUSUM",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistical helpers
// ─────────────────────────────────────────────────────────────────────────────

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64], mean_val: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance =
        values.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Anomaly detector configured with a single algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetector {
    /// The algorithm to use.
    pub algorithm: Algorithm,
}

impl AnomalyDetector {
    /// Create a new detector with the given algorithm.
    pub fn new(algorithm: Algorithm) -> Self {
        Self { algorithm }
    }

    /// Detect anomalies in a slice of data points.
    ///
    /// Returns a `Vec<Anomaly>` sorted by timestamp.  An empty slice always
    /// produces no anomalies.
    pub fn detect(&self, data: &[DataPoint]) -> Vec<Anomaly> {
        if data.is_empty() {
            return vec![];
        }
        match &self.algorithm {
            Algorithm::ZScore { threshold } => detect_zscore(data, *threshold),
            Algorithm::IQR { multiplier } => detect_iqr(data, *multiplier),
            Algorithm::MovingAverage { window, sigma } => {
                detect_moving_average(data, *window, *sigma)
            }
            Algorithm::CUSUM { k, h } => detect_cusum(data, *k, *h),
        }
    }

    /// Detect anomalies in multiple time series simultaneously.
    ///
    /// The input is a map of series ID (`u64`) → slice of data points.
    /// Returns a map of series ID → detected anomalies.
    pub fn batch_detect(&self, series: &[(u64, &[DataPoint])]) -> HashMap<u64, Vec<Anomaly>> {
        series
            .iter()
            .map(|(id, data)| (*id, self.detect(data)))
            .collect()
    }

    /// Compute [`AnomalyStats`] for a given data series.
    pub fn stats(&self, data: &[DataPoint]) -> AnomalyStats {
        let anomalies = self.detect(data);
        AnomalyStats::from_anomalies(&anomalies)
    }

    /// Convenience: detect anomalies and return only the timestamps.
    pub fn anomalous_timestamps(&self, data: &[DataPoint]) -> Vec<u64> {
        self.detect(data)
            .into_iter()
            .map(|a| a.point.timestamp)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ZScore detection
// ─────────────────────────────────────────────────────────────────────────────

fn detect_zscore(data: &[DataPoint], threshold: f64) -> Vec<Anomaly> {
    let values: Vec<f64> = data.iter().map(|p| p.value).collect();
    let m = mean(&values);
    let s = std_dev(&values, m);

    if s == 0.0 {
        // All values are identical — no anomalies
        return vec![];
    }

    data.iter()
        .filter_map(|p| {
            let z = (p.value - m).abs() / s;
            if z > threshold {
                Some(Anomaly {
                    point: *p,
                    score: z,
                    algorithm: "ZScore".to_owned(),
                })
            } else {
                None
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// IQR detection
// ─────────────────────────────────────────────────────────────────────────────

fn detect_iqr(data: &[DataPoint], multiplier: f64) -> Vec<Anomaly> {
    if data.is_empty() {
        return vec![];
    }

    let mut sorted: Vec<f64> = data.iter().map(|p| p.value).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = percentile_sorted(&sorted, 25.0);
    let q3 = percentile_sorted(&sorted, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - multiplier * iqr;
    let upper = q3 + multiplier * iqr;

    data.iter()
        .filter_map(|p| {
            if p.value < lower || p.value > upper {
                // Score = distance from nearest fence, normalized by IQR
                let dist = if p.value < lower {
                    lower - p.value
                } else {
                    p.value - upper
                };
                let score = if iqr > 0.0 { dist / iqr } else { dist.abs() };
                Some(Anomaly {
                    point: *p,
                    score,
                    algorithm: "IQR".to_owned(),
                })
            } else {
                None
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Moving Average detection
// ─────────────────────────────────────────────────────────────────────────────

fn detect_moving_average(data: &[DataPoint], window: usize, sigma: f64) -> Vec<Anomaly> {
    // window == 0 or window == 1 leaves no room for a baseline; skip.
    if window <= 1 || data.len() < window {
        return vec![];
    }

    let values: Vec<f64> = data.iter().map(|p| p.value).collect();
    let mut anomalies = Vec::new();

    for i in (window - 1)..values.len() {
        // Use the preceding `window - 1` points as the baseline so that the
        // current point does not influence its own mean/std.  This ensures a
        // sharp spike is measured against a stable reference window rather
        // than a diluted one that already includes the spike itself.
        let baseline = &values[i + 1 - window..i];
        let m = mean(baseline);
        let s = std_dev(baseline, m);

        let current = values[i];
        let deviation = if s > 1e-10 {
            (current - m).abs() / s
        } else if (current - m).abs() > 1e-10 {
            // Baseline is perfectly flat but current value differs — treat as
            // a definite anomaly by capping the score at a large finite value
            // so that downstream consumers always receive a finite score.
            f64::MAX.sqrt()
        } else {
            0.0
        };

        if deviation > sigma {
            anomalies.push(Anomaly {
                point: data[i],
                score: deviation,
                algorithm: "MovingAverage".to_owned(),
            });
        }
    }
    anomalies
}

// ─────────────────────────────────────────────────────────────────────────────
// CUSUM detection
// ─────────────────────────────────────────────────────────────────────────────

fn detect_cusum(data: &[DataPoint], k: f64, h: f64) -> Vec<Anomaly> {
    if data.is_empty() {
        return vec![];
    }

    let values: Vec<f64> = data.iter().map(|p| p.value).collect();
    let m = mean(&values);

    // Two-sided CUSUM: tracks upward (s_pos) and downward (s_neg) sums
    let mut s_pos = 0.0f64;
    let mut s_neg = 0.0f64;
    let mut anomalies = Vec::new();

    for (i, &x) in values.iter().enumerate() {
        let delta = x - m;
        s_pos = (s_pos + delta - k).max(0.0);
        s_neg = (s_neg - delta - k).max(0.0);

        let score = s_pos.max(s_neg);
        if score > h {
            anomalies.push(Anomaly {
                point: data[i],
                score,
                algorithm: "CUSUM".to_owned(),
            });
            // Reset both charts after alarm (restart CUSUM)
            s_pos = 0.0;
            s_neg = 0.0;
        }
    }
    anomalies
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Create a flat time series with an outlier at `spike_idx` with value `spike`.
    fn flat_with_spike(n: usize, baseline: f64, spike_idx: usize, spike: f64) -> Vec<DataPoint> {
        (0..n)
            .map(|i| DataPoint {
                timestamp: i as u64,
                value: if i == spike_idx { spike } else { baseline },
            })
            .collect()
    }

    // ── ZScore ───────────────────────────────────────────────────────────────

    #[test]
    fn test_zscore_detects_spike() {
        let data = flat_with_spike(100, 1.0, 50, 1000.0);
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        let anomalies = detector.detect(&data);
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].point.timestamp, 50);
    }

    #[test]
    fn test_zscore_no_anomalies_flat_data() {
        let data: Vec<DataPoint> = (0..50)
            .map(|i| DataPoint {
                timestamp: i,
                value: 5.0,
            })
            .collect();
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        assert!(detector.detect(&data).is_empty());
    }

    #[test]
    fn test_zscore_score_above_threshold() {
        let data = flat_with_spike(100, 1.0, 50, 1000.0);
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        let anomalies = detector.detect(&data);
        assert!(anomalies[0].score > 3.0);
    }

    #[test]
    fn test_zscore_algorithm_name() {
        let data = flat_with_spike(100, 1.0, 50, 1000.0);
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        let anomalies = detector.detect(&data);
        assert_eq!(anomalies[0].algorithm, "ZScore");
    }

    #[test]
    fn test_zscore_empty_data() {
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        assert!(detector.detect(&[]).is_empty());
    }

    #[test]
    fn test_zscore_high_threshold_no_anomalies() {
        let data = flat_with_spike(100, 1.0, 50, 10.0);
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 100.0 });
        assert!(detector.detect(&data).is_empty());
    }

    // ── IQR ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_iqr_detects_extreme_outlier() {
        let data = flat_with_spike(100, 1.0, 50, 9999.0);
        let detector = AnomalyDetector::new(Algorithm::IQR { multiplier: 1.5 });
        let anomalies = detector.detect(&data);
        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.point.timestamp == 50));
    }

    #[test]
    fn test_iqr_no_anomalies_flat_data() {
        let data: Vec<DataPoint> = (0..50)
            .map(|i| DataPoint {
                timestamp: i,
                value: 2.0,
            })
            .collect();
        let detector = AnomalyDetector::new(Algorithm::IQR { multiplier: 1.5 });
        assert!(detector.detect(&data).is_empty());
    }

    #[test]
    fn test_iqr_algorithm_name() {
        let data = flat_with_spike(100, 1.0, 50, 9999.0);
        let detector = AnomalyDetector::new(Algorithm::IQR { multiplier: 1.5 });
        let anomalies = detector.detect(&data);
        if !anomalies.is_empty() {
            assert_eq!(anomalies[0].algorithm, "IQR");
        }
    }

    #[test]
    fn test_iqr_empty_data() {
        let detector = AnomalyDetector::new(Algorithm::IQR { multiplier: 1.5 });
        assert!(detector.detect(&[]).is_empty());
    }

    #[test]
    fn test_iqr_high_multiplier_fewer_anomalies() {
        let data = flat_with_spike(100, 1.0, 50, 10.0);
        let strict = AnomalyDetector::new(Algorithm::IQR { multiplier: 1.5 });
        let lenient = AnomalyDetector::new(Algorithm::IQR { multiplier: 10.0 });
        let strict_count = strict.detect(&data).len();
        let lenient_count = lenient.detect(&data).len();
        assert!(strict_count >= lenient_count);
    }

    #[test]
    fn test_iqr_negative_outlier() {
        let data = flat_with_spike(100, 1.0, 25, -9999.0);
        let detector = AnomalyDetector::new(Algorithm::IQR { multiplier: 1.5 });
        let anomalies = detector.detect(&data);
        assert!(anomalies.iter().any(|a| a.point.timestamp == 25));
    }

    // ── Moving Average ────────────────────────────────────────────────────────

    #[test]
    fn test_moving_average_detects_spike() {
        let data = flat_with_spike(50, 1.0, 40, 999.0);
        let detector = AnomalyDetector::new(Algorithm::MovingAverage {
            window: 5,
            sigma: 3.0,
        });
        let anomalies = detector.detect(&data);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_moving_average_no_anomalies_flat() {
        let data: Vec<DataPoint> = (0..50)
            .map(|i| DataPoint {
                timestamp: i,
                value: 3.0,
            })
            .collect();
        let detector = AnomalyDetector::new(Algorithm::MovingAverage {
            window: 5,
            sigma: 3.0,
        });
        assert!(detector.detect(&data).is_empty());
    }

    #[test]
    fn test_moving_average_window_larger_than_data_returns_empty() {
        let data = flat_with_spike(3, 1.0, 1, 100.0);
        let detector = AnomalyDetector::new(Algorithm::MovingAverage {
            window: 10,
            sigma: 3.0,
        });
        assert!(detector.detect(&data).is_empty());
    }

    #[test]
    fn test_moving_average_zero_window_returns_empty() {
        let data = flat_with_spike(50, 1.0, 25, 999.0);
        let detector = AnomalyDetector::new(Algorithm::MovingAverage {
            window: 0,
            sigma: 3.0,
        });
        assert!(detector.detect(&data).is_empty());
    }

    #[test]
    fn test_moving_average_algorithm_name() {
        let data = flat_with_spike(50, 1.0, 40, 999.0);
        let detector = AnomalyDetector::new(Algorithm::MovingAverage {
            window: 5,
            sigma: 3.0,
        });
        let anomalies = detector.detect(&data);
        if !anomalies.is_empty() {
            assert_eq!(anomalies[0].algorithm, "MovingAverage");
        }
    }

    #[test]
    fn test_moving_average_empty_data() {
        let detector = AnomalyDetector::new(Algorithm::MovingAverage {
            window: 5,
            sigma: 3.0,
        });
        assert!(detector.detect(&[]).is_empty());
    }

    // ── CUSUM ────────────────────────────────────────────────────────────────

    #[test]
    fn test_cusum_detects_sustained_shift() {
        // Data is flat at 0.0 then shifts to 10.0
        let mut data: Vec<DataPoint> = (0..50)
            .map(|i| DataPoint {
                timestamp: i as u64,
                value: 0.0,
            })
            .collect();
        for dp in data[30..50].iter_mut() {
            dp.value = 10.0;
        }
        let detector = AnomalyDetector::new(Algorithm::CUSUM { k: 0.5, h: 5.0 });
        let anomalies = detector.detect(&data);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_cusum_no_anomalies_flat_data() {
        let data: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint {
                timestamp: i,
                value: 0.0,
            })
            .collect();
        let detector = AnomalyDetector::new(Algorithm::CUSUM { k: 0.5, h: 5.0 });
        assert!(detector.detect(&data).is_empty());
    }

    #[test]
    fn test_cusum_algorithm_name() {
        let mut data: Vec<DataPoint> = (0..50)
            .map(|i| DataPoint {
                timestamp: i as u64,
                value: 0.0,
            })
            .collect();
        for dp in data[30..50].iter_mut() {
            dp.value = 20.0;
        }
        let detector = AnomalyDetector::new(Algorithm::CUSUM { k: 0.5, h: 5.0 });
        let anomalies = detector.detect(&data);
        if !anomalies.is_empty() {
            assert_eq!(anomalies[0].algorithm, "CUSUM");
        }
    }

    #[test]
    fn test_cusum_empty_data() {
        let detector = AnomalyDetector::new(Algorithm::CUSUM { k: 0.5, h: 5.0 });
        assert!(detector.detect(&[]).is_empty());
    }

    #[test]
    fn test_cusum_high_threshold_no_anomalies() {
        let data = flat_with_spike(100, 1.0, 50, 5.0);
        let detector = AnomalyDetector::new(Algorithm::CUSUM { k: 0.5, h: 9999.0 });
        assert!(detector.detect(&data).is_empty());
    }

    // ── AnomalyStats ─────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_no_anomalies() {
        let stats = AnomalyStats::from_anomalies(&[]);
        assert_eq!(stats.count, 0);
        assert!(stats.min_score.is_nan());
        assert!(stats.max_score.is_nan());
        assert!(stats.mean_score.is_nan());
    }

    #[test]
    fn test_stats_single_anomaly() {
        let anomaly = Anomaly {
            point: DataPoint {
                timestamp: 0,
                value: 999.0,
            },
            score: 5.0,
            algorithm: "ZScore".to_owned(),
        };
        let stats = AnomalyStats::from_anomalies(&[anomaly]);
        assert_eq!(stats.count, 1);
        assert!((stats.min_score - 5.0).abs() < 1e-9);
        assert!((stats.max_score - 5.0).abs() < 1e-9);
        assert!((stats.mean_score - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_multiple_anomalies() {
        let anomalies: Vec<Anomaly> = vec![3.0, 5.0, 7.0]
            .into_iter()
            .enumerate()
            .map(|(i, s)| Anomaly {
                point: DataPoint {
                    timestamp: i as u64,
                    value: 0.0,
                },
                score: s,
                algorithm: "ZScore".to_owned(),
            })
            .collect();
        let stats = AnomalyStats::from_anomalies(&anomalies);
        assert_eq!(stats.count, 3);
        assert!((stats.min_score - 3.0).abs() < 1e-9);
        assert!((stats.max_score - 7.0).abs() < 1e-9);
        assert!((stats.mean_score - 5.0).abs() < 1e-9);
    }

    // ── Detector helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_stats_via_detector() {
        let data = flat_with_spike(100, 1.0, 50, 1000.0);
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        let stats = detector.stats(&data);
        assert_eq!(stats.count, 1);
    }

    #[test]
    fn test_anomalous_timestamps() {
        let data = flat_with_spike(100, 1.0, 50, 1000.0);
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        let ts = detector.anomalous_timestamps(&data);
        assert_eq!(ts, vec![50u64]);
    }

    // ── Batch detection ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_detect_multiple_series() {
        let s1 = flat_with_spike(100, 1.0, 10, 1000.0);
        let s2 = flat_with_spike(100, 2.0, 80, 9999.0);
        let series: Vec<(u64, &[DataPoint])> = vec![(1, &s1), (2, &s2)];
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        let results = detector.batch_detect(&series);
        assert_eq!(results.len(), 2);
        assert!(!results[&1].is_empty());
        assert!(!results[&2].is_empty());
    }

    #[test]
    fn test_batch_detect_empty_input() {
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        let results = detector.batch_detect(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_detect_series_with_no_anomalies() {
        let s1: Vec<DataPoint> = (0..50)
            .map(|i| DataPoint {
                timestamp: i,
                value: 1.0,
            })
            .collect();
        let series: Vec<(u64, &[DataPoint])> = vec![(42, &s1)];
        let detector = AnomalyDetector::new(Algorithm::ZScore { threshold: 3.0 });
        let results = detector.batch_detect(&series);
        assert!(results[&42].is_empty());
    }

    // ── Algorithm::name() ────────────────────────────────────────────────────

    #[test]
    fn test_algorithm_name_zscore() {
        assert_eq!(Algorithm::ZScore { threshold: 3.0 }.name(), "ZScore");
    }

    #[test]
    fn test_algorithm_name_iqr() {
        assert_eq!(Algorithm::IQR { multiplier: 1.5 }.name(), "IQR");
    }

    #[test]
    fn test_algorithm_name_moving_average() {
        assert_eq!(
            Algorithm::MovingAverage {
                window: 5,
                sigma: 3.0
            }
            .name(),
            "MovingAverage"
        );
    }

    #[test]
    fn test_algorithm_name_cusum() {
        assert_eq!(Algorithm::CUSUM { k: 0.5, h: 5.0 }.name(), "CUSUM");
    }

    // ── DataPoint ─────────────────────────────────────────────────────────────

    #[test]
    fn test_data_point_copy_trait() {
        let p = DataPoint {
            timestamp: 1,
            value: 2.0,
        };
        let q = p; // Copy
        assert_eq!(q.timestamp, 1);
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    #[test]
    fn test_data_point_serde() {
        let p = DataPoint {
            timestamp: 42,
            value: 3.15,
        };
        let json = serde_json::to_string(&p).unwrap();
        let q: DataPoint = serde_json::from_str(&json).unwrap();
        assert_eq!(q.timestamp, 42);
        assert!((q.value - 3.15).abs() < 1e-9);
    }

    #[test]
    fn test_anomaly_serde() {
        let a = Anomaly {
            point: DataPoint {
                timestamp: 1,
                value: 99.9,
            },
            score: 5.5,
            algorithm: "ZScore".to_owned(),
        };
        let json = serde_json::to_string(&a).unwrap();
        let b: Anomaly = serde_json::from_str(&json).unwrap();
        assert!((b.score - 5.5).abs() < 1e-9);
        assert_eq!(b.algorithm, "ZScore");
    }

    #[test]
    fn test_algorithm_serde() {
        let alg = Algorithm::ZScore { threshold: 3.0 };
        let json = serde_json::to_string(&alg).unwrap();
        let restored: Algorithm = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.name(), "ZScore");
    }

    // ── Percentile helpers ────────────────────────────────────────────────────

    #[test]
    fn test_percentile_sorted_midpoint() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q2 = percentile_sorted(&sorted, 50.0);
        assert!((q2 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_sorted_first() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q0 = percentile_sorted(&sorted, 0.0);
        assert!((q0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_sorted_last() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q100 = percentile_sorted(&sorted, 100.0);
        assert!((q100 - 5.0).abs() < 1e-9);
    }
}
