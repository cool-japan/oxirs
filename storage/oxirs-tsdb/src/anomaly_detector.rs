//! Time-series anomaly detection algorithms.
//!
//! Provides multiple statistical methods for detecting anomalous data points
//! in time-series data: Z-Score, IQR, Moving Average, and Percentile Range.

/// The algorithm used for anomaly detection.
#[derive(Debug, Clone)]
pub enum AnomalyMethod {
    /// Z-Score method: flag values where `|v - mean| / std_dev > threshold`.
    ZScore {
        /// Number of standard deviations beyond which a point is an anomaly.
        threshold: f64,
    },
    /// Interquartile Range method: flag values outside `[Q1 - mult*IQR, Q3 + mult*IQR]`.
    Iqr {
        /// IQR multiplier (typically 1.5 for outliers, 3.0 for extreme outliers).
        multiplier: f64,
    },
    /// Moving average method: flag values where the deviation from the local mean exceeds
    /// `threshold` times the local standard deviation.
    MovingAverage {
        /// Number of preceding data points to include in the moving window.
        window: usize,
        /// Deviation threshold in units of local standard deviation.
        threshold: f64,
    },
    /// Percentile range method: flag values outside the `[lower, upper]` percentile range.
    PercentileRange {
        /// Lower percentile boundary (0.0–100.0).
        lower: f64,
        /// Upper percentile boundary (0.0–100.0).
        upper: f64,
    },
}

/// A single time-series observation.
#[derive(Debug, Clone, PartialEq)]
pub struct DataPoint {
    /// Millisecond timestamp (monotonic or wall-clock).
    pub timestamp_ms: u64,
    /// The observed value.
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point.
    pub fn new(timestamp_ms: u64, value: f64) -> Self {
        Self {
            timestamp_ms,
            value,
        }
    }
}

/// A detected anomaly in the time series.
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// The anomalous data point.
    pub point: DataPoint,
    /// The anomaly score (higher = more anomalous).
    pub score: f64,
    /// The name of the detection method that flagged this point.
    pub method: String,
    /// The upper bound of the normal range, if applicable.
    pub upper_bound: Option<f64>,
    /// The lower bound of the normal range, if applicable.
    pub lower_bound: Option<f64>,
}

/// Anomaly detector supporting multiple statistical detection methods.
pub struct AnomalyDetector {
    /// The detection method to apply.
    pub method: AnomalyMethod,
}

impl AnomalyDetector {
    /// Create a new detector with the given method.
    pub fn new(method: AnomalyMethod) -> Self {
        Self { method }
    }

    /// Detect anomalies in a time-series slice.
    ///
    /// Returns a list of `Anomaly` entries for each flagged data point.
    pub fn detect(&self, data: &[DataPoint]) -> Vec<Anomaly> {
        match &self.method {
            AnomalyMethod::ZScore { threshold } => self.detect_zscore(data, *threshold),
            AnomalyMethod::Iqr { multiplier } => self.detect_iqr(data, *multiplier),
            AnomalyMethod::MovingAverage { window, threshold } => {
                self.detect_moving_average(data, *window, *threshold)
            }
            AnomalyMethod::PercentileRange { lower, upper } => {
                self.detect_percentile_range(data, *lower, *upper)
            }
        }
    }

    /// Return the indices of anomalous values from a plain value slice.
    pub fn detect_by_value(&self, values: &[f64]) -> Vec<usize> {
        let points: Vec<DataPoint> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| DataPoint::new(i as u64, v))
            .collect();
        self.detect(&points)
            .iter()
            .map(|a| a.point.timestamp_ms as usize)
            .collect()
    }

    /// Score all data points and return each point paired with its anomaly score.
    pub fn score_all(&self, data: &[DataPoint]) -> Vec<(DataPoint, f64)> {
        match &self.method {
            AnomalyMethod::ZScore { .. } => {
                if data.is_empty() {
                    return Vec::new();
                }
                let values: Vec<f64> = data.iter().map(|p| p.value).collect();
                let m = mean(&values);
                let sd = std_dev(&values);
                data.iter()
                    .map(|p| {
                        let score = if sd == 0.0 {
                            0.0
                        } else {
                            (p.value - m).abs() / sd
                        };
                        (p.clone(), score)
                    })
                    .collect()
            }
            AnomalyMethod::Iqr { multiplier } => {
                if data.is_empty() {
                    return Vec::new();
                }
                let mut sorted: Vec<f64> = data.iter().map(|p| p.value).collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q1 = percentile(&sorted, 25.0);
                let q3 = percentile(&sorted, 75.0);
                let iqr = q3 - q1;
                let lower = q1 - multiplier * iqr;
                let upper = q3 + multiplier * iqr;
                data.iter()
                    .map(|p| {
                        let score = if p.value < lower {
                            lower - p.value
                        } else if p.value > upper {
                            p.value - upper
                        } else {
                            0.0
                        };
                        (p.clone(), score)
                    })
                    .collect()
            }
            AnomalyMethod::MovingAverage {
                window,
                threshold: _,
            } => data
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    if i == 0 {
                        return (p.clone(), 0.0);
                    }
                    let start = i.saturating_sub(*window);
                    let window_vals: Vec<f64> = data[start..i].iter().map(|q| q.value).collect();
                    let ma = mean(&window_vals);
                    let ms = std_dev(&window_vals);
                    let score = if ms == 0.0 {
                        (p.value - ma).abs()
                    } else {
                        (p.value - ma).abs() / ms
                    };
                    (p.clone(), score)
                })
                .collect(),
            AnomalyMethod::PercentileRange { lower, upper } => {
                if data.is_empty() {
                    return Vec::new();
                }
                let mut sorted: Vec<f64> = data.iter().map(|p| p.value).collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let lo = percentile(&sorted, *lower);
                let hi = percentile(&sorted, *upper);
                data.iter()
                    .map(|p| {
                        let score = if p.value < lo {
                            lo - p.value
                        } else if p.value > hi {
                            p.value - hi
                        } else {
                            0.0
                        };
                        (p.clone(), score)
                    })
                    .collect()
            }
        }
    }

    // ── Private detection helpers ─────────────────────────────────────────────

    fn detect_zscore(&self, data: &[DataPoint], threshold: f64) -> Vec<Anomaly> {
        if data.is_empty() {
            return Vec::new();
        }
        let values: Vec<f64> = data.iter().map(|p| p.value).collect();
        let m = mean(&values);
        let sd = std_dev(&values);

        if sd == 0.0 {
            // All values identical → no anomalies
            return Vec::new();
        }

        let upper = m + threshold * sd;
        let lower = m - threshold * sd;

        data.iter()
            .filter_map(|p| {
                let score = (p.value - m).abs() / sd;
                if score > threshold {
                    Some(Anomaly {
                        point: p.clone(),
                        score,
                        method: "ZScore".to_string(),
                        upper_bound: Some(upper),
                        lower_bound: Some(lower),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    fn detect_iqr(&self, data: &[DataPoint], multiplier: f64) -> Vec<Anomaly> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut sorted: Vec<f64> = data.iter().map(|p| p.value).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = percentile(&sorted, 25.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;
        let lower = q1 - multiplier * iqr;
        let upper = q3 + multiplier * iqr;

        data.iter()
            .filter_map(|p| {
                if p.value < lower || p.value > upper {
                    let score = if p.value < lower {
                        lower - p.value
                    } else {
                        p.value - upper
                    };
                    Some(Anomaly {
                        point: p.clone(),
                        score,
                        method: "IQR".to_string(),
                        upper_bound: Some(upper),
                        lower_bound: Some(lower),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    fn detect_moving_average(
        &self,
        data: &[DataPoint],
        window: usize,
        threshold: f64,
    ) -> Vec<Anomaly> {
        if data.is_empty() || window == 0 {
            return Vec::new();
        }

        let mut anomalies = Vec::new();

        for i in 1..data.len() {
            let start = i.saturating_sub(window);
            let window_vals: Vec<f64> = data[start..i].iter().map(|p| p.value).collect();

            if window_vals.is_empty() {
                continue;
            }

            let ma = mean(&window_vals);
            let ms = std_dev(&window_vals);

            let score = if ms == 0.0 {
                // If all window values are identical, flag any deviation
                (data[i].value - ma).abs()
            } else {
                (data[i].value - ma).abs() / ms
            };

            let flag = if ms == 0.0 {
                // Flag if any deviation exists
                score > 0.0
            } else {
                score > threshold
            };

            if flag {
                let upper = ma + threshold * ms;
                let lower = ma - threshold * ms;
                anomalies.push(Anomaly {
                    point: data[i].clone(),
                    score,
                    method: "MovingAverage".to_string(),
                    upper_bound: Some(upper),
                    lower_bound: Some(lower),
                });
            }
        }

        anomalies
    }

    fn detect_percentile_range(
        &self,
        data: &[DataPoint],
        lower_pct: f64,
        upper_pct: f64,
    ) -> Vec<Anomaly> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut sorted: Vec<f64> = data.iter().map(|p| p.value).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lo = percentile(&sorted, lower_pct);
        let hi = percentile(&sorted, upper_pct);

        data.iter()
            .filter_map(|p| {
                if p.value < lo || p.value > hi {
                    let score = if p.value < lo {
                        lo - p.value
                    } else {
                        p.value - hi
                    };
                    Some(Anomaly {
                        point: p.clone(),
                        score,
                        method: "PercentileRange".to_string(),
                        upper_bound: Some(hi),
                        lower_bound: Some(lo),
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

// ── Public helper functions ───────────────────────────────────────────────────

/// Compute the `p`-th percentile of a **sorted** slice.
///
/// Uses linear interpolation between the two neighbouring values.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let p = p.clamp(0.0, 100.0);
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }
}

/// Compute the arithmetic mean of a slice.
///
/// Returns 0.0 for an empty slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the population standard deviation of a slice.
///
/// Returns 0.0 for slices with fewer than 2 elements.
pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(values: &[f64]) -> Vec<DataPoint> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| DataPoint::new(i as u64, v))
            .collect()
    }

    // ── Helper functions ──────────────────────────────────────────────────────

    #[test]
    fn test_mean_empty() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_mean_single() {
        assert_eq!(mean(&[5.0]), 5.0);
    }

    #[test]
    fn test_mean_multiple() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev_empty() {
        assert_eq!(std_dev(&[]), 0.0);
    }

    #[test]
    fn test_std_dev_single() {
        assert_eq!(std_dev(&[42.0]), 0.0);
    }

    #[test]
    fn test_std_dev_uniform() {
        assert_eq!(std_dev(&[3.0, 3.0, 3.0, 3.0]), 0.0);
    }

    #[test]
    fn test_std_dev_known() {
        // Population std dev of [2,4,4,4,5,5,7,9] = 2.0
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev(&data) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[7.0], 50.0), 7.0);
    }

    #[test]
    fn test_percentile_median() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 50.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_0_and_100() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&sorted, 100.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_interpolation() {
        let sorted = [0.0, 10.0];
        // 25th percentile = 0 + 0.25 * (10 - 0) = 2.5
        assert!((percentile(&sorted, 25.0) - 2.5).abs() < 1e-10);
    }

    // ── ZScore ────────────────────────────────────────────────────────────────

    #[test]
    fn test_zscore_empty() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 2.0 });
        assert!(det.detect(&[]).is_empty());
    }

    #[test]
    fn test_zscore_all_same() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 2.0 });
        let data = pts(&[5.0, 5.0, 5.0, 5.0]);
        assert!(det.detect(&data).is_empty());
    }

    #[test]
    fn test_zscore_detects_outlier() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 2.0 });
        let mut values = vec![1.0; 100];
        values.push(1000.0); // extreme outlier
        let data = pts(&values);
        let anomalies = det.detect(&data);
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].method, "ZScore");
        assert!(anomalies[0].score > 2.0);
    }

    #[test]
    fn test_zscore_bounds_present() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 2.0 });
        let mut values = vec![1.0; 50];
        values.push(100.0);
        let data = pts(&values);
        let anomalies = det.detect(&data);
        assert!(!anomalies.is_empty());
        assert!(anomalies[0].upper_bound.is_some());
        assert!(anomalies[0].lower_bound.is_some());
    }

    #[test]
    fn test_zscore_no_false_positives_uniform() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 3.0 });
        let data = pts(&[10.0, 11.0, 10.5, 10.2, 10.8, 11.1, 9.9, 10.3]);
        let anomalies = det.detect(&data);
        assert!(anomalies.is_empty());
    }

    // ── IQR ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_iqr_empty() {
        let det = AnomalyDetector::new(AnomalyMethod::Iqr { multiplier: 1.5 });
        assert!(det.detect(&[]).is_empty());
    }

    #[test]
    fn test_iqr_detects_outlier() {
        let det = AnomalyDetector::new(AnomalyMethod::Iqr { multiplier: 1.5 });
        let mut values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        values.push(1000.0);
        let data = pts(&values);
        let anomalies = det.detect(&data);
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].method, "IQR");
    }

    #[test]
    fn test_iqr_bounds_present() {
        let det = AnomalyDetector::new(AnomalyMethod::Iqr { multiplier: 1.5 });
        let mut values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        values.push(999.0);
        let data = pts(&values);
        let anomalies = det.detect(&data);
        assert!(!anomalies.is_empty());
        assert!(anomalies[0].upper_bound.is_some());
        assert!(anomalies[0].lower_bound.is_some());
    }

    #[test]
    fn test_iqr_all_same_no_anomalies() {
        let det = AnomalyDetector::new(AnomalyMethod::Iqr { multiplier: 1.5 });
        let data = pts(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        // All same → IQR = 0, bounds = [5, 5], all values on boundary (not strictly outside)
        let anomalies = det.detect(&data);
        assert!(anomalies.is_empty());
    }

    // ── MovingAverage ─────────────────────────────────────────────────────────

    #[test]
    fn test_moving_average_empty() {
        let det = AnomalyDetector::new(AnomalyMethod::MovingAverage {
            window: 5,
            threshold: 2.0,
        });
        assert!(det.detect(&[]).is_empty());
    }

    #[test]
    fn test_moving_average_single_point() {
        let det = AnomalyDetector::new(AnomalyMethod::MovingAverage {
            window: 5,
            threshold: 2.0,
        });
        let data = pts(&[42.0]);
        assert!(det.detect(&data).is_empty());
    }

    #[test]
    fn test_moving_average_detects_spike() {
        let det = AnomalyDetector::new(AnomalyMethod::MovingAverage {
            window: 5,
            threshold: 2.0,
        });
        let mut values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        values.push(100.0); // spike
        let data = pts(&values);
        let anomalies = det.detect(&data);
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].method, "MovingAverage");
    }

    #[test]
    fn test_moving_average_window_zero_empty() {
        let det = AnomalyDetector::new(AnomalyMethod::MovingAverage {
            window: 0,
            threshold: 2.0,
        });
        let data = pts(&[1.0, 2.0, 100.0, 1.0]);
        assert!(det.detect(&data).is_empty());
    }

    // ── PercentileRange ───────────────────────────────────────────────────────

    #[test]
    fn test_percentile_range_empty() {
        let det = AnomalyDetector::new(AnomalyMethod::PercentileRange {
            lower: 5.0,
            upper: 95.0,
        });
        assert!(det.detect(&[]).is_empty());
    }

    #[test]
    fn test_percentile_range_detects_extremes() {
        let det = AnomalyDetector::new(AnomalyMethod::PercentileRange {
            lower: 10.0,
            upper: 90.0,
        });
        let mut values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        values.push(0.0); // below 10th percentile
        values.push(200.0); // above 90th percentile
        let data = pts(&values);
        let anomalies = det.detect(&data);
        assert!(!anomalies.is_empty());
        for a in &anomalies {
            assert_eq!(a.method, "PercentileRange");
        }
    }

    #[test]
    fn test_percentile_range_bounds_present() {
        let det = AnomalyDetector::new(AnomalyMethod::PercentileRange {
            lower: 5.0,
            upper: 95.0,
        });
        let mut values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        values.push(999.0);
        let data = pts(&values);
        let anomalies = det.detect(&data);
        assert!(!anomalies.is_empty());
        assert!(anomalies
            .iter()
            .all(|a| a.upper_bound.is_some() && a.lower_bound.is_some()));
    }

    // ── detect_by_value ───────────────────────────────────────────────────────

    #[test]
    fn test_detect_by_value_returns_indices() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 2.0 });
        let mut values = vec![1.0f64; 50];
        values.push(1000.0);
        let indices = det.detect_by_value(&values);
        assert!(indices.contains(&50));
    }

    #[test]
    fn test_detect_by_value_empty() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 2.0 });
        assert!(det.detect_by_value(&[]).is_empty());
    }

    // ── score_all ─────────────────────────────────────────────────────────────

    #[test]
    fn test_score_all_length_matches() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 2.0 });
        let data = pts(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let scores = det.score_all(&data);
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_score_all_empty() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 2.0 });
        assert!(det.score_all(&[]).is_empty());
    }

    #[test]
    fn test_score_all_outlier_has_high_score() {
        let det = AnomalyDetector::new(AnomalyMethod::ZScore { threshold: 2.0 });
        let mut values = vec![1.0f64; 50];
        values.push(1000.0);
        let data = pts(&values);
        let scores = det.score_all(&data);
        let last_score = scores.last().expect("should succeed").1;
        assert!(
            last_score > 2.0,
            "Outlier should have score > 2.0, got {last_score}"
        );
    }

    // ── DataPoint ─────────────────────────────────────────────────────────────

    #[test]
    fn test_data_point_new() {
        let p = DataPoint::new(12345, 2.71);
        assert_eq!(p.timestamp_ms, 12345);
        assert!((p.value - 2.71).abs() < 1e-10);
    }

    #[test]
    fn test_data_point_clone_eq() {
        let p = DataPoint::new(1, 2.0);
        assert_eq!(p.clone(), p);
    }
}
