//! Kalman filter for time-series smoothing and prediction.
//!
//! Provides:
//! - [`KalmanFilter`] – Classic scalar Kalman filter
//! - [`AdaptiveKalmanFilter`] – Adapts process noise based on recent innovation variance
//! - [`KalmanAnomaly`] – Anomaly detector built on top of [`AdaptiveKalmanFilter`]

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// AnomalyEvent (shared with the caller)
// ──────────────────────────────────────────────────────────────────────────────

/// An anomaly event detected by the Kalman-based detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyEvent {
    /// The raw measurement that triggered the anomaly.
    pub value: f64,
    /// The Kalman filter's predicted value at this step.
    pub predicted: f64,
    /// The innovation (measurement - prediction).
    pub innovation: f64,
    /// The sigma threshold used for detection.
    pub threshold_sigma: f64,
    /// The computed standard deviation of recent innovations.
    pub innovation_std: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// KalmanFilter
// ──────────────────────────────────────────────────────────────────────────────

/// Classic scalar Kalman filter for 1-D time series.
///
/// State model: `x_{k+1} = x_k + w_k` (random walk)
/// Measurement model: `z_k = x_k + v_k`
///
/// where `w_k ~ N(0, process_noise)` and `v_k ~ N(0, measurement_noise)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilter {
    /// Current state estimate.
    pub state_estimate: f64,
    /// Current error covariance `P`.
    pub error_covariance: f64,
    /// Process noise variance `Q`.
    pub process_noise: f64,
    /// Measurement noise variance `R`.
    pub measurement_noise: f64,
    /// Most recently computed Kalman gain `K`.
    pub kalman_gain: f64,
}

impl KalmanFilter {
    /// Create a new Kalman filter with the given noise parameters and initial state.
    ///
    /// # Arguments
    /// * `process_noise` – Variance of the state transition noise `Q`.
    /// * `measurement_noise` – Variance of the measurement noise `R`.
    /// * `initial_estimate` – Prior state estimate `x_0`.
    pub fn new(process_noise: f64, measurement_noise: f64, initial_estimate: f64) -> Self {
        Self {
            state_estimate: initial_estimate,
            // Start with a relatively large initial covariance so the filter
            // trusts the first observation heavily.
            error_covariance: 1.0,
            process_noise,
            measurement_noise,
            kalman_gain: 0.0,
        }
    }

    /// Ingest a new measurement and return the filtered (posterior) estimate.
    ///
    /// The update is performed in-place; subsequent calls to `predict` will use
    /// the posterior state.
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Prediction step: propagate covariance by process noise.
        let p_prior = self.error_covariance + self.process_noise;

        // Update step: compute Kalman gain.
        let k = p_prior / (p_prior + self.measurement_noise);
        self.kalman_gain = k;

        // Update state estimate.
        self.state_estimate += k * (measurement - self.state_estimate);

        // Update error covariance (Joseph form is numerically more stable but
        // the simplified form suffices for scalar case).
        self.error_covariance = (1.0 - k) * p_prior;

        self.state_estimate
    }

    /// Predict the state `steps_ahead` steps into the future.
    ///
    /// Under the random-walk model the expected value stays constant, so this
    /// simply returns the current posterior estimate.  The error covariance
    /// grows by `steps_ahead * process_noise` but the mean is unchanged.
    pub fn predict(&self, _steps_ahead: usize) -> f64 {
        // For a random-walk model E[x_{k+n} | x_k] = x_k.
        self.state_estimate
    }

    /// Smooth an entire series and return the filtered values.
    ///
    /// A new filter is constructed internally so the method is stateless from
    /// the caller's perspective.
    pub fn smooth_series(values: &[f64], process_noise: f64, measurement_noise: f64) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }
        let mut filter = KalmanFilter::new(process_noise, measurement_noise, values[0]);
        values.iter().map(|&v| filter.update(v)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AdaptiveKalmanFilter
// ──────────────────────────────────────────────────────────────────────────────

/// Adaptive Kalman filter that adjusts `process_noise` based on recent
/// innovation variance.
///
/// The innovation window stores the last `window_size` residuals (measurement
/// minus prior prediction).  Their sample variance is used to rescale `Q` so
/// that the filter responds faster during periods of high variability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveKalmanFilter {
    /// Underlying Kalman filter.
    pub base: KalmanFilter,
    /// Sliding window of recent innovations.
    pub innovation_window: Vec<f64>,
    /// Maximum number of innovations kept.
    pub window_size: usize,
}

impl AdaptiveKalmanFilter {
    /// Create a new adaptive filter with the given window size.
    ///
    /// Initial noise parameters are chosen conservatively; they adapt after
    /// the first `window_size` observations.
    pub fn new(window_size: usize) -> Self {
        let safe_window = window_size.max(2);
        Self {
            base: KalmanFilter::new(0.01, 1.0, 0.0),
            innovation_window: Vec::with_capacity(safe_window),
            window_size: safe_window,
        }
    }

    /// Compute the sample variance of the innovation window.
    fn innovation_variance(&self) -> f64 {
        let n = self.innovation_window.len();
        if n < 2 {
            return self.base.process_noise;
        }
        let mean = self.innovation_window.iter().sum::<f64>() / n as f64;
        let var = self
            .innovation_window
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        var
    }

    /// Ingest a new measurement, adapt `process_noise`, and return the filtered
    /// value.
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Prediction before updating: prior = current state estimate.
        let prior = self.base.state_estimate;

        // Compute innovation (residual) using the prior state.
        let innovation = measurement - prior;

        // Maintain the sliding window of innovations.
        if self.innovation_window.len() == self.window_size {
            self.innovation_window.remove(0);
        }
        self.innovation_window.push(innovation);

        // Adapt process noise to the empirical innovation variance.
        let empirical_var = self.innovation_variance();
        // Clip to a sensible range to avoid degeneracy.
        self.base.process_noise = empirical_var.clamp(1e-10, 1e6);

        self.base.update(measurement)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// KalmanAnomaly
// ──────────────────────────────────────────────────────────────────────────────

/// Anomaly detector that flags measurements whose innovation exceeds
/// `threshold_sigma` standard deviations of the recent innovation distribution.
///
/// Internally uses an [`AdaptiveKalmanFilter`] to track the normal behaviour of
/// the series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanAnomaly {
    /// Adaptive filter tracking normal behaviour.
    pub filter: AdaptiveKalmanFilter,
    /// Number of standard deviations beyond which a point is anomalous.
    pub threshold_sigma: f64,
}

impl KalmanAnomaly {
    /// Create a new detector with the given sigma threshold.
    ///
    /// A common choice is `3.0` (3-sigma rule).  Smaller values produce more
    /// alerts; larger values are more conservative.
    pub fn new(threshold_sigma: f64) -> Self {
        Self {
            filter: AdaptiveKalmanFilter::new(20),
            threshold_sigma,
        }
    }

    /// Process a new measurement.
    ///
    /// Returns `Some(AnomalyEvent)` if the measurement is anomalous, or `None`
    /// if it is within the expected range.
    pub fn check(&mut self, value: f64) -> Option<AnomalyEvent> {
        let predicted = self.filter.base.state_estimate;
        let innovation = value - predicted;

        // Update filter with the new observation.
        self.filter.update(value);

        // Compute innovation std from the window.
        let var = self.filter.innovation_variance();
        let std_dev = var.sqrt();

        // Need enough data before we can make meaningful comparisons.
        if self.filter.innovation_window.len() < 3 {
            return None;
        }

        let threshold = self.threshold_sigma * std_dev;
        if innovation.abs() > threshold {
            Some(AnomalyEvent {
                value,
                predicted,
                innovation,
                threshold_sigma: self.threshold_sigma,
                innovation_std: std_dev,
            })
        } else {
            None
        }
    }

    /// Batch-check a slice, returning all detected anomaly events with their
    /// indices.
    pub fn check_series(&mut self, values: &[f64]) -> TsdbResult<Vec<(usize, AnomalyEvent)>> {
        if values.is_empty() {
            return Err(TsdbError::Query(
                "empty input slice for Kalman anomaly detection".into(),
            ));
        }
        let mut results = Vec::new();
        for (idx, &v) in values.iter().enumerate() {
            if let Some(event) = self.check(v) {
                results.push((idx, event));
            }
        }
        Ok(results)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── KalmanFilter ──────────────────────────────────────────────────────────

    #[test]
    fn test_kalman_new_state() {
        let kf = KalmanFilter::new(0.1, 1.0, 5.0);
        assert_eq!(kf.state_estimate, 5.0);
        assert_eq!(kf.process_noise, 0.1);
        assert_eq!(kf.measurement_noise, 1.0);
    }

    #[test]
    fn test_kalman_update_converges_to_constant() {
        let mut kf = KalmanFilter::new(0.01, 0.1, 0.0);
        // Feed a constant signal; the estimate should converge toward 10.0.
        for _ in 0..200 {
            kf.update(10.0);
        }
        let estimate = kf.state_estimate;
        assert!((estimate - 10.0).abs() < 0.1, "estimate={estimate}");
    }

    #[test]
    fn test_kalman_update_tracks_ramp() {
        let mut kf = KalmanFilter::new(1.0, 0.5, 0.0);
        let mut last = 0.0;
        for i in 0..50 {
            last = kf.update(i as f64);
        }
        // Estimate should be roughly tracking the last few values.
        assert!(last > 30.0, "last estimate={last}");
    }

    #[test]
    fn test_kalman_predict_returns_current_state() {
        let mut kf = KalmanFilter::new(0.1, 1.0, 0.0);
        kf.update(7.0);
        let pred1 = kf.predict(1);
        let pred5 = kf.predict(5);
        // Both should equal the posterior state (random-walk model).
        assert_eq!(pred1, pred5);
        assert_eq!(pred1, kf.state_estimate);
    }

    #[test]
    fn test_kalman_smooth_series_length() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let smoothed = KalmanFilter::smooth_series(&values, 0.1, 1.0);
        assert_eq!(smoothed.len(), values.len());
    }

    #[test]
    fn test_kalman_smooth_series_empty() {
        let smoothed = KalmanFilter::smooth_series(&[], 0.1, 1.0);
        assert!(smoothed.is_empty());
    }

    #[test]
    fn test_kalman_smooth_reduces_noise() {
        // Noisy constant signal around 50.0.
        let noisy: Vec<f64> = (0..100)
            .map(|i| 50.0 + (i as f64 % 3.0) * 2.0 - 2.0)
            .collect();
        let smoothed = KalmanFilter::smooth_series(&noisy, 0.01, 1.0);
        // The smoothed values should all be within ±10 of 50.
        for &s in &smoothed {
            assert!(
                (s - 50.0).abs() < 10.0,
                "smoothed value {s} too far from 50"
            );
        }
    }

    #[test]
    fn test_kalman_error_covariance_decreases() {
        let mut kf = KalmanFilter::new(0.0, 0.5, 0.0);
        let p_before = kf.error_covariance;
        kf.update(1.0);
        // With zero process noise the posterior covariance drops below prior.
        assert!(
            kf.error_covariance < p_before,
            "covariance should shrink: {} >= {}",
            kf.error_covariance,
            p_before
        );
    }

    #[test]
    fn test_kalman_kalman_gain_in_range() {
        let mut kf = KalmanFilter::new(0.1, 1.0, 0.0);
        kf.update(5.0);
        assert!(
            kf.kalman_gain > 0.0 && kf.kalman_gain < 1.0,
            "Kalman gain out of (0,1): {}",
            kf.kalman_gain
        );
    }

    // ── AdaptiveKalmanFilter ──────────────────────────────────────────────────

    #[test]
    fn test_adaptive_kalman_new() {
        let akf = AdaptiveKalmanFilter::new(10);
        assert_eq!(akf.window_size, 10);
        assert!(akf.innovation_window.is_empty());
    }

    #[test]
    fn test_adaptive_kalman_window_enforced() {
        let mut akf = AdaptiveKalmanFilter::new(5);
        for i in 0..20 {
            akf.update(i as f64);
        }
        assert!(
            akf.innovation_window.len() <= 5,
            "window grew beyond limit: {}",
            akf.innovation_window.len()
        );
    }

    #[test]
    fn test_adaptive_kalman_update_returns_value() {
        let mut akf = AdaptiveKalmanFilter::new(10);
        let v = akf.update(42.0);
        assert!(v.is_finite(), "update returned non-finite: {v}");
    }

    #[test]
    fn test_adaptive_kalman_converges_constant() {
        let mut akf = AdaptiveKalmanFilter::new(20);
        let mut last = 0.0;
        for _ in 0..300 {
            last = akf.update(100.0);
        }
        assert!(
            (last - 100.0).abs() < 5.0,
            "adaptive filter did not converge: {last}"
        );
    }

    #[test]
    fn test_adaptive_kalman_min_window_size() {
        // window_size=1 is clamped to 2 internally.
        let akf = AdaptiveKalmanFilter::new(1);
        assert!(akf.window_size >= 2);
    }

    // ── KalmanAnomaly ─────────────────────────────────────────────────────────

    #[test]
    fn test_kalman_anomaly_new() {
        let ka = KalmanAnomaly::new(3.0);
        assert_eq!(ka.threshold_sigma, 3.0);
    }

    #[test]
    fn test_kalman_anomaly_no_anomaly_for_constant() {
        let mut ka = KalmanAnomaly::new(3.0);
        // Warm up with 50 constant observations.
        for _ in 0..50 {
            ka.check(10.0);
        }
        // After warmup the constant signal should not trigger anomalies.
        for _ in 0..10 {
            let ev = ka.check(10.0);
            assert!(
                ev.is_none(),
                "unexpected anomaly on constant signal: {ev:?}"
            );
        }
    }

    #[test]
    fn test_kalman_anomaly_detects_spike() {
        let mut ka = KalmanAnomaly::new(3.0);
        // Warm up.
        for _ in 0..50 {
            ka.check(1.0);
        }
        // Introduce a large spike.
        let ev = ka.check(1000.0);
        assert!(ev.is_some(), "spike should be detected as anomaly");
    }

    #[test]
    fn test_kalman_anomaly_event_fields() {
        let mut ka = KalmanAnomaly::new(2.0);
        for _ in 0..50 {
            ka.check(5.0);
        }
        let ev = ka.check(5000.0).expect("spike should produce an event");
        assert!(ev.innovation > 0.0);
        assert_eq!(ev.threshold_sigma, 2.0);
        assert!(ev.innovation_std >= 0.0);
    }

    #[test]
    fn test_kalman_anomaly_check_series_empty_error() {
        let mut ka = KalmanAnomaly::new(3.0);
        let result = ka.check_series(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_kalman_anomaly_check_series_detects_multiple() {
        let mut ka = KalmanAnomaly::new(3.0);
        // Warmup with steady values, then insert two spikes.
        let mut values: Vec<f64> = vec![1.0; 100];
        values[70] = 9999.0;
        values[80] = 9999.0;
        let result = ka.check_series(&values).expect("should succeed");
        // At least one spike should be caught.
        assert!(!result.is_empty(), "expected anomalies in spiked series");
    }
}
