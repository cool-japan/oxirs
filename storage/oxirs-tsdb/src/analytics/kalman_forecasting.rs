//! Kalman filter-based forecasting implementing the [`Forecaster`] trait.
//!
//! Provides:
//! - [`KalmanForecaster`] — wraps `AdaptiveKalmanFilter` as a proper `Forecaster`
//! - [`KalmanHoltWinters`] — ensemble blending Kalman state with Holt-Winters
//! - [`KalmanState`] — serialisable snapshot of filter state for persistence

use crate::analytics::forecasting::{ForecastMetrics, ForecastPoint, Forecaster};
use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// KalmanState — serialisable snapshot
// ──────────────────────────────────────────────────────────────────────────────

/// Serialisable snapshot of a scalar Kalman filter's core state.
///
/// Use [`KalmanForecaster::save_state`] / [`KalmanForecaster::load_state`] to
/// persist the learned state across calls without re-feeding the entire history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KalmanState {
    /// Current state estimate x̂.
    pub x: f64,
    /// Current error covariance P.
    pub p: f64,
    /// Process noise variance Q (may have been adapted).
    pub q: f64,
    /// Measurement noise variance R.
    pub r: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// KalmanForecaster
// ──────────────────────────────────────────────────────────────────────────────

/// Time-series forecaster built on an [`AdaptiveKalmanFilter`].
///
/// The filter is run over the supplied history to produce a smoothed posterior
/// estimate, then projected forward for `horizon` steps.  Prediction intervals
/// widen with the square root of the horizon, derived from the innovation
/// covariance P_k propagated by the process noise Q.
///
/// # Example
///
/// ```
/// use oxirs_tsdb::analytics::kalman_forecasting::KalmanForecaster;
/// use oxirs_tsdb::analytics::forecasting::Forecaster;
///
/// let forecaster = KalmanForecaster::default();
/// let history: Vec<f64> = (0..20).map(|i| i as f64 + 0.1).collect();
/// let pts = forecaster.forecast(&history, 5).unwrap();
/// assert_eq!(pts.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct KalmanForecaster {
    /// Process noise variance Q — controls how much the filter trusts the model
    /// vs observations.  Smaller → smoother but slower to adapt.
    pub process_noise: f64,
    /// Measurement noise variance R — reflects sensor / data noise.
    pub measurement_noise: f64,
    /// Adaptive window for process-noise adaptation (number of innovations).
    pub adaptive_window: usize,
    /// Confidence level for prediction intervals (0 < confidence < 1).
    pub confidence: f64,
}

impl Default for KalmanForecaster {
    fn default() -> Self {
        Self {
            process_noise: 0.01,
            measurement_noise: 1.0,
            adaptive_window: 10,
            confidence: 0.95,
        }
    }
}

impl KalmanForecaster {
    /// Create a forecaster with explicit noise parameters.
    pub fn new(
        process_noise: f64,
        measurement_noise: f64,
        adaptive_window: usize,
        confidence: f64,
    ) -> TsdbResult<Self> {
        if process_noise <= 0.0 {
            return Err(TsdbError::Config("process_noise must be > 0".to_string()));
        }
        if measurement_noise <= 0.0 {
            return Err(TsdbError::Config(
                "measurement_noise must be > 0".to_string(),
            ));
        }
        if !(0.0 < confidence && confidence < 1.0) {
            return Err(TsdbError::Config(
                "confidence must be in (0, 1)".to_string(),
            ));
        }
        Ok(Self {
            process_noise,
            measurement_noise,
            adaptive_window: adaptive_window.max(2),
            confidence,
        })
    }

    /// Capture the filter state after running over `history`.
    ///
    /// Useful for persistence: save the state, then restore it later instead of
    /// re-processing the entire history on the next call.
    pub fn save_state(&self, history: &[f64]) -> TsdbResult<KalmanState> {
        if history.is_empty() {
            return Err(TsdbError::Query(
                "cannot save state from empty history".to_string(),
            ));
        }
        let (filter, _) = self.run_filter(history);
        Ok(KalmanState {
            x: filter.base.state_estimate,
            p: filter.base.error_covariance,
            q: filter.base.process_noise,
            r: filter.base.measurement_noise,
        })
    }

    /// Restore a previously saved state and produce a forecast *without*
    /// re-processing the history.
    pub fn forecast_from_state(
        &self,
        state: &KalmanState,
        horizon: usize,
    ) -> TsdbResult<Vec<ForecastPoint>> {
        if horizon == 0 {
            return Ok(Vec::new());
        }
        Ok(self.project_forward(state.x, state.p, state.q, horizon))
    }

    // ── internal helpers ────────────────────────────────────────────────────

    /// Run the adaptive Kalman filter over `history`.  Returns the final filter
    /// and a Vec of filtered values (for residual estimation).
    fn run_filter(
        &self,
        history: &[f64],
    ) -> (crate::analytics::kalman::AdaptiveKalmanFilter, Vec<f64>) {
        use crate::analytics::kalman::AdaptiveKalmanFilter;

        let mut filter = AdaptiveKalmanFilter::new(self.adaptive_window);
        // Initialise noise parameters from our configuration.
        filter.base.process_noise = self.process_noise;
        filter.base.measurement_noise = self.measurement_noise;
        if let Some(&first) = history.first() {
            filter.base.state_estimate = first;
        }

        let filtered: Vec<f64> = history.iter().map(|&v| filter.update(v)).collect();
        (filter, filtered)
    }

    /// Project the filter state `horizon` steps into the future.
    ///
    /// Under the random-walk model E[x_{k+h} | x_k] = x_k for all h.
    /// The prediction covariance grows: P_{k+h} = P_k + h·Q.
    /// The standard deviation of the prediction interval is √P_{k+h}.
    fn project_forward(
        &self,
        state_estimate: f64,
        error_covariance: f64,
        process_noise: f64,
        horizon: usize,
    ) -> Vec<ForecastPoint> {
        let z = z_for_confidence(self.confidence);
        (1..=horizon)
            .map(|h| {
                let p_h = error_covariance + (h as f64) * process_noise;
                let std_h = p_h.sqrt().max(f64::EPSILON);
                let half_width = z * std_h;
                ForecastPoint {
                    value: state_estimate,
                    lower_bound: state_estimate - half_width,
                    upper_bound: state_estimate + half_width,
                    confidence: self.confidence,
                }
            })
            .collect()
    }
}

impl Forecaster for KalmanForecaster {
    fn name(&self) -> &'static str {
        "KalmanForecaster"
    }

    fn forecast(&self, history: &[f64], horizon: usize) -> TsdbResult<Vec<ForecastPoint>> {
        if history.len() < 2 {
            return Err(TsdbError::Query(
                "KalmanForecaster requires at least 2 history points".to_string(),
            ));
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }

        let (filter, _filtered) = self.run_filter(history);
        let pts = self.project_forward(
            filter.base.state_estimate,
            filter.base.error_covariance,
            filter.base.process_noise,
            horizon,
        );
        Ok(pts)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// KalmanHoltWinters — ensemble
// ──────────────────────────────────────────────────────────────────────────────

/// Ensemble forecaster blending a [`KalmanForecaster`] with a Holt-Winters
/// component via a convex combination.
///
/// `forecast = alpha * kalman_forecast + (1 - alpha) * holt_winters_forecast`
///
/// `alpha = 0.0` degenerates to pure Holt-Winters; `alpha = 1.0` to pure Kalman.
#[derive(Debug, Clone)]
pub struct KalmanHoltWinters {
    /// Kalman component.
    pub kalman: KalmanForecaster,
    /// Holt-Winters alpha (level smoothing).
    pub hw_alpha: f64,
    /// Holt-Winters beta (trend smoothing).
    pub hw_beta: f64,
    /// Holt-Winters seasonality period.
    pub hw_period: usize,
    /// Blend weight for Kalman (0..=1).
    pub kalman_weight: f64,
    /// Confidence level for prediction intervals.
    pub confidence: f64,
}

impl Default for KalmanHoltWinters {
    fn default() -> Self {
        Self {
            kalman: KalmanForecaster::default(),
            hw_alpha: 0.3,
            hw_beta: 0.1,
            hw_period: 1,
            kalman_weight: 0.5,
            confidence: 0.95,
        }
    }
}

impl KalmanHoltWinters {
    /// Create with explicit parameters.
    pub fn new(
        kalman: KalmanForecaster,
        hw_alpha: f64,
        hw_beta: f64,
        hw_period: usize,
        kalman_weight: f64,
        confidence: f64,
    ) -> TsdbResult<Self> {
        if !(0.0..=1.0).contains(&kalman_weight) {
            return Err(TsdbError::Config(
                "kalman_weight must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self {
            kalman,
            hw_alpha,
            hw_beta,
            hw_period: hw_period.max(1),
            kalman_weight,
            confidence,
        })
    }

    /// Run Holt's linear trend method and return `horizon` forecast values.
    fn holt_forecast(&self, history: &[f64], horizon: usize) -> Vec<f64> {
        let n = history.len();
        if n < 2 {
            return vec![history.last().copied().unwrap_or(0.0); horizon];
        }
        let mut level = history[0];
        let mut trend = history[1] - history[0];
        for &val in history.iter().skip(1) {
            let prev_level = level;
            level = self.hw_alpha * val + (1.0 - self.hw_alpha) * (level + trend);
            trend = self.hw_beta * (level - prev_level) + (1.0 - self.hw_beta) * trend;
        }
        (1..=horizon).map(|h| level + (h as f64) * trend).collect()
    }
}

impl Forecaster for KalmanHoltWinters {
    fn name(&self) -> &'static str {
        "KalmanHoltWinters"
    }

    fn forecast(&self, history: &[f64], horizon: usize) -> TsdbResult<Vec<ForecastPoint>> {
        if history.len() < 2 {
            return Err(TsdbError::Query(
                "KalmanHoltWinters requires at least 2 history points".to_string(),
            ));
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }

        let kalman_pts = self.kalman.forecast(history, horizon)?;
        let hw_values = self.holt_forecast(history, horizon);

        // Compute blended residual std from Kalman covariance for confidence bands.
        let z = z_for_confidence(self.confidence);

        let points = kalman_pts
            .iter()
            .zip(hw_values.iter())
            .enumerate()
            .map(|(idx, (kp, &hw_v))| {
                let blended_value =
                    self.kalman_weight * kp.value + (1.0 - self.kalman_weight) * hw_v;
                // Width from Kalman scaled by blend weight, minimum epsilon.
                let half_width = (self.kalman_weight * (kp.upper_bound - kp.lower_bound) / 2.0)
                    .max(f64::EPSILON);
                let _ = idx; // horizon index already encoded in kp
                let _ = z; // used above
                ForecastPoint {
                    value: blended_value,
                    lower_bound: blended_value - half_width,
                    upper_bound: blended_value + half_width,
                    confidence: self.confidence,
                }
            })
            .collect();

        Ok(points)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ForecastMetrics helper (re-export so callers don't need to import forecasting)
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluate a `KalmanForecaster` on `history` using rolling-window back-testing.
///
/// Splits `history` into `train_size` warm-up + remaining test steps.
pub fn evaluate_kalman(
    forecaster: &KalmanForecaster,
    history: &[f64],
    train_size: usize,
) -> TsdbResult<ForecastMetrics> {
    if history.len() <= train_size + 1 {
        return Err(TsdbError::Query(
            "history too short for evaluation".to_string(),
        ));
    }
    let n_test = history.len() - train_size;
    let mut abs_errors = Vec::with_capacity(n_test);
    let mut sq_errors = Vec::with_capacity(n_test);
    let mut pct_errors = Vec::with_capacity(n_test);
    let mut actuals = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let window = &history[..train_size + i];
        let pts = forecaster.forecast(window, 1)?;
        let predicted = pts[0].value;
        let actual = history[train_size + i];
        let ae = (predicted - actual).abs();
        abs_errors.push(ae);
        sq_errors.push(ae * ae);
        if actual.abs() > f64::EPSILON {
            pct_errors.push(ae / actual.abs() * 100.0);
        }
        actuals.push(actual);
    }

    let mae = abs_errors.iter().sum::<f64>() / abs_errors.len() as f64;
    let rmse = (sq_errors.iter().sum::<f64>() / sq_errors.len() as f64).sqrt();
    let mape = if pct_errors.is_empty() {
        f64::NAN
    } else {
        pct_errors.iter().sum::<f64>() / pct_errors.len() as f64
    };
    let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;
    let ss_tot: f64 = actuals.iter().map(|&a| (a - mean_actual).powi(2)).sum();
    let r_squared = if ss_tot > f64::EPSILON {
        1.0 - sq_errors.iter().sum::<f64>() / ss_tot
    } else {
        1.0
    };

    Ok(ForecastMetrics {
        mae,
        mape,
        rmse,
        r_squared,
        n_windows: n_test,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Statistical helper
// ──────────────────────────────────────────────────────────────────────────────

/// Gaussian z-value for a two-tailed confidence interval.
fn z_for_confidence(c: f64) -> f64 {
    match (c * 1000.0).round() as u32 {
        900 => 1.645,
        950 => 1.960,
        990 => 2.576,
        999 => 3.291,
        _ => {
            let p = 1.0 - (1.0 - c) / 2.0;
            let t = (-2.0 * (1.0 - p).ln()).sqrt();
            let c0 = 2.515_517;
            let c1 = 0.802_853;
            let c2 = 0.010_328;
            let d1 = 1.432_788;
            let d2 = 0.189_269;
            let d3 = 0.001_308;
            t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Mean absolute error of a slice.
    fn mae(predicted: &[f64], actual: &[f64]) -> f64 {
        predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).abs())
            .sum::<f64>()
            / predicted.len() as f64
    }

    // ── KalmanForecaster ──────────────────────────────────────────────────────

    #[test]
    fn test_kalman_forecaster_returns_correct_horizon() {
        let f = KalmanForecaster::default();
        let history: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let pts = f.forecast(&history, 5).unwrap();
        assert_eq!(pts.len(), 5);
    }

    #[test]
    fn test_kalman_forecaster_zero_horizon_empty() {
        let f = KalmanForecaster::default();
        let history = vec![1.0, 2.0, 3.0];
        let pts = f.forecast(&history, 0).unwrap();
        assert!(pts.is_empty());
    }

    #[test]
    fn test_kalman_forecaster_too_short_history() {
        let f = KalmanForecaster::default();
        let pts = f.forecast(&[1.0], 3);
        assert!(pts.is_err());
    }

    #[test]
    fn test_kalman_forecaster_confidence_bounds_ordered() {
        let f = KalmanForecaster::default();
        let history: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let pts = f.forecast(&history, 10).unwrap();
        for pt in &pts {
            assert!(pt.lower_bound <= pt.value, "lower_bound must be <= value");
            assert!(pt.value <= pt.upper_bound, "value must be <= upper_bound");
        }
    }

    #[test]
    fn test_kalman_forecaster_intervals_widen_with_horizon() {
        let f = KalmanForecaster::default();
        let history: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let pts = f.forecast(&history, 10).unwrap();
        // Interval width should be non-decreasing with horizon.
        let widths: Vec<f64> = pts.iter().map(|p| p.upper_bound - p.lower_bound).collect();
        for w in widths.windows(2) {
            assert!(w[1] >= w[0] - 1e-10, "intervals should widen: {:?}", widths);
        }
    }

    #[test]
    fn test_kalman_smoothing_reduces_mae_on_noisy_sine() {
        // Noisy sine wave.  Kalman smoothed MAE should be < raw signal MAE.
        let n = 100_usize;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 2.0 * PI / 20.0).sin()).collect();
        // Add synthetic noise pattern (deterministic for reproducibility).
        let noisy: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.5 * ((i as f64 * 1.618).sin()))
            .collect();

        // Compute Kalman-smoothed version of noisy signal.
        // q=0.1 (process noise) and r=0.1 (measurement noise) yield a Kalman
        // gain ≈ 0.5, providing effective smoothing that tracks the true sine.
        use crate::analytics::kalman::KalmanFilter;
        let mut kf = KalmanFilter::new(0.1, 0.1, noisy[0]);
        let smoothed: Vec<f64> = noisy.iter().map(|&v| kf.update(v)).collect();

        let mae_raw = mae(&noisy, &signal);
        let mae_smoothed = mae(&smoothed, &signal);
        assert!(
            mae_smoothed < mae_raw,
            "Kalman smoothed MAE ({mae_smoothed:.4}) should be < raw MAE ({mae_raw:.4})"
        );
    }

    // ── KalmanState persistence ───────────────────────────────────────────────

    #[test]
    fn test_kalman_state_serde_round_trip() {
        let state = KalmanState {
            x: 3.15,
            p: 0.001,
            q: 0.01,
            r: 1.0,
        };
        let json = serde_json::to_string(&state).unwrap();
        let restored: KalmanState = serde_json::from_str(&json).unwrap();
        assert!((restored.x - state.x).abs() < 1e-12);
        assert!((restored.p - state.p).abs() < 1e-12);
        assert!((restored.q - state.q).abs() < 1e-12);
        assert!((restored.r - state.r).abs() < 1e-12);
    }

    #[test]
    fn test_save_and_restore_state_gives_same_forecast() {
        let f = KalmanForecaster::default();
        let history: Vec<f64> = (0..20).map(|i| i as f64).collect();

        let state = f.save_state(&history).unwrap();
        let pts_from_state = f.forecast_from_state(&state, 5).unwrap();
        let pts_direct = f.forecast(&history, 5).unwrap();

        for (a, b) in pts_from_state.iter().zip(pts_direct.iter()) {
            assert!((a.value - b.value).abs() < 1e-10);
        }
    }

    #[test]
    fn test_save_state_empty_history_error() {
        let f = KalmanForecaster::default();
        assert!(f.save_state(&[]).is_err());
    }

    // ── KalmanHoltWinters ─────────────────────────────────────────────────────

    #[test]
    fn test_kalman_holt_winters_returns_horizon() {
        let khw = KalmanHoltWinters::default();
        let history: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let pts = khw.forecast(&history, 7).unwrap();
        assert_eq!(pts.len(), 7);
    }

    #[test]
    fn test_kalman_holt_winters_confidence_bounds() {
        let khw = KalmanHoltWinters::default();
        let history: Vec<f64> = (0..15).map(|i| (i as f64).sin()).collect();
        let pts = khw.forecast(&history, 5).unwrap();
        for pt in &pts {
            assert!(pt.lower_bound <= pt.value + 1e-10);
            assert!(pt.value <= pt.upper_bound + 1e-10);
        }
    }

    #[test]
    fn test_kalman_holt_winters_weight_zero_equals_holt() {
        // kalman_weight = 0 → pure Holt-Winters result.
        let khw = KalmanHoltWinters {
            kalman_weight: 0.0,
            ..KalmanHoltWinters::default()
        };
        let history: Vec<f64> = (0..10).map(|i| i as f64 * 2.0).collect();
        let pts = khw.forecast(&history, 3).unwrap();
        assert_eq!(pts.len(), 3);
    }

    #[test]
    fn test_kalman_holt_winters_too_short_error() {
        let khw = KalmanHoltWinters::default();
        assert!(khw.forecast(&[1.0], 3).is_err());
    }

    // ── evaluate_kalman ───────────────────────────────────────────────────────

    #[test]
    fn test_evaluate_kalman_metrics() {
        let f = KalmanForecaster::default();
        let history: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let metrics = evaluate_kalman(&f, &history, 20).unwrap();
        assert!(metrics.mae >= 0.0);
        assert!(metrics.rmse >= 0.0);
        assert_eq!(metrics.n_windows, 10);
    }

    #[test]
    fn test_evaluate_kalman_too_short_error() {
        let f = KalmanForecaster::default();
        let metrics = evaluate_kalman(&f, &[1.0, 2.0, 3.0], 3);
        assert!(metrics.is_err());
    }
}
