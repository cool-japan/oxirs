//! Time-series forecasting algorithms.
//!
//! Provides multiple forecasting methods:
//! - Naive (seasonal naive baseline)
//! - Simple Exponential Smoothing (SES)
//! - Double Exponential Smoothing / Holt's linear trend
//! - Triple Exponential Smoothing / Holt-Winters seasonal
//!
//! All methods implement the `Forecaster` trait, which produces a horizon of
//! `ForecastPoint` values with confidence intervals, plus a `ForecastEvaluator`
//! that performs rolling-window back-testing and returns `ForecastMetrics`.

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A single forecast value with associated uncertainty bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastPoint {
    /// Point forecast (best estimate).
    pub value: f64,
    /// Lower bound of the prediction interval.
    pub lower_bound: f64,
    /// Upper bound of the prediction interval.
    pub upper_bound: f64,
    /// Confidence level used to compute the bounds (e.g. 0.95).
    pub confidence: f64,
}

impl ForecastPoint {
    fn new(value: f64, residual_std: f64, confidence: f64) -> Self {
        // z-value lookup (Gaussian approximation).
        let z = z_value_for_confidence(confidence);
        let half_width = z * residual_std;
        Self {
            value,
            lower_bound: value - half_width,
            upper_bound: value + half_width,
            confidence,
        }
    }
}

/// Error metrics from a back-testing evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastMetrics {
    /// Mean Absolute Error.
    pub mae: f64,
    /// Mean Absolute Percentage Error (%).
    pub mape: f64,
    /// Root Mean Squared Error.
    pub rmse: f64,
    /// R² coefficient of determination.
    pub r_squared: f64,
    /// Number of test windows evaluated.
    pub n_windows: usize,
}

/// Core trait for forecasting models.
pub trait Forecaster: Send + Sync {
    /// Produce a `horizon`-step-ahead forecast given the `history` slice.
    ///
    /// The returned `Vec<ForecastPoint>` has exactly `horizon` elements, where
    /// element `[0]` is the one-step-ahead forecast.
    fn forecast(&self, history: &[f64], horizon: usize) -> TsdbResult<Vec<ForecastPoint>>;

    /// Name of this forecasting method (for logging / metrics labels).
    fn name(&self) -> &'static str;
}

// ──────────────────────────────────────────────────────────────────────────────
// Statistical helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Mean of a slice (0.0 for empty).
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().copied().sum::<f64>() / xs.len() as f64
}

/// Population standard deviation.
fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|v| (v - m).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Approximate z-value for a two-tailed confidence interval.
fn z_value_for_confidence(c: f64) -> f64 {
    // Standard lookup table for common levels; linear interpolation otherwise.
    match (c * 1000.0).round() as u32 {
        900 => 1.645,
        950 => 1.960,
        990 => 2.576,
        999 => 3.291,
        _ => {
            // Rational approximation of the inverse-normal CDF (Abramowitz & Stegun 26.2.17).
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
// Naive (seasonal naive baseline)
// ──────────────────────────────────────────────────────────────────────────────

/// Seasonal Naive forecaster.
///
/// Each future step *h* is predicted as the last observed value at the same
/// position in the seasonal cycle: `ŷ_{n+h} = y_{n + h − period}`.
///
/// When `period == 1` this degenerates to the simple last-value naive forecast.
#[derive(Debug, Clone)]
pub struct NaiveForecast {
    /// Seasonality period (1 = no seasonality).
    pub period: usize,
    /// Confidence level for prediction intervals.
    pub confidence: f64,
}

impl NaiveForecast {
    /// Create a naive forecaster.
    pub fn new(period: usize, confidence: f64) -> TsdbResult<Self> {
        if period == 0 {
            return Err(TsdbError::Config("period must be > 0".to_string()));
        }
        Ok(Self { period, confidence })
    }
}

impl Default for NaiveForecast {
    fn default() -> Self {
        Self {
            period: 1,
            confidence: 0.95,
        }
    }
}

impl Forecaster for NaiveForecast {
    fn name(&self) -> &'static str {
        "NaiveForecast"
    }

    fn forecast(&self, history: &[f64], horizon: usize) -> TsdbResult<Vec<ForecastPoint>> {
        let n = history.len();
        if n < self.period {
            return Err(TsdbError::Query(format!(
                "NaiveForecast needs at least {} history points",
                self.period
            )));
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }

        // Residual std estimated from one-step naive errors.
        let residuals: Vec<f64> = (self.period..n)
            .map(|i| history[i] - history[i - self.period])
            .collect();
        let resid_std = std_dev(&residuals).max(f64::EPSILON);

        let mut points = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            // Seasonal index: look back `period` steps into history + already-forecast.
            let src_idx = (n + h - 1) - self.period;
            let value = if src_idx < n {
                history[src_idx]
            } else {
                // Look up from previously forecast values.
                let fi = src_idx - n;
                points
                    .get(fi)
                    .map(|p: &ForecastPoint| p.value)
                    .unwrap_or(history[n - 1])
            };
            // Prediction interval widens with horizon.
            let horizon_std = resid_std * (h as f64).sqrt();
            points.push(ForecastPoint::new(value, horizon_std, self.confidence));
        }

        Ok(points)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Simple Exponential Smoothing
// ──────────────────────────────────────────────────────────────────────────────

/// Simple Exponential Smoothing (SES).
///
/// Level equation:
///   `l_t = α · y_t + (1 − α) · l_{t−1}`
///
/// All horizon forecasts are the same: `ŷ_{n+h} = l_n`.
#[derive(Debug, Clone)]
pub struct SimpleExponentialSmoothing {
    /// Level smoothing parameter in (0, 1).
    pub alpha: f64,
    /// Confidence level for prediction intervals.
    pub confidence: f64,
}

impl SimpleExponentialSmoothing {
    /// Create a SES forecaster.
    pub fn new(alpha: f64, confidence: f64) -> TsdbResult<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(TsdbError::Config("SES alpha must be in (0, 1)".to_string()));
        }
        Ok(Self { alpha, confidence })
    }
}

impl Default for SimpleExponentialSmoothing {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            confidence: 0.95,
        }
    }
}

impl Forecaster for SimpleExponentialSmoothing {
    fn name(&self) -> &'static str {
        "SimpleExponentialSmoothing"
    }

    fn forecast(&self, history: &[f64], horizon: usize) -> TsdbResult<Vec<ForecastPoint>> {
        let n = history.len();
        if n < 2 {
            return Err(TsdbError::Query(
                "SES requires at least 2 history points".to_string(),
            ));
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }

        // Initialise level with the first observation.
        let mut level = history[0];
        let mut one_step_errors: Vec<f64> = Vec::with_capacity(n - 1);

        for &y in &history[1..] {
            let fitted = level;
            one_step_errors.push(y - fitted);
            level = self.alpha * y + (1.0 - self.alpha) * level;
        }

        let resid_std = std_dev(&one_step_errors).max(f64::EPSILON);

        Ok((1..=horizon)
            .map(|h| ForecastPoint::new(level, resid_std * (h as f64).sqrt(), self.confidence))
            .collect())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Double Exponential Smoothing (Holt's linear trend)
// ──────────────────────────────────────────────────────────────────────────────

/// Double Exponential Smoothing (Holt's method) with additive trend.
///
/// Level:  `l_t = α · y_t + (1 − α) · (l_{t−1} + b_{t−1})`
/// Trend:  `b_t = β · (l_t − l_{t−1}) + (1 − β) · b_{t−1}`
/// Forecast: `ŷ_{n+h} = l_n + h · b_n`
#[derive(Debug, Clone)]
pub struct DoubleExponentialSmoothing {
    /// Level smoothing parameter in (0, 1).
    pub alpha: f64,
    /// Trend smoothing parameter in (0, 1).
    pub beta: f64,
    /// Confidence level for prediction intervals.
    pub confidence: f64,
}

impl DoubleExponentialSmoothing {
    /// Create a Holt's linear trend forecaster.
    pub fn new(alpha: f64, beta: f64, confidence: f64) -> TsdbResult<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(TsdbError::Config(
                "Holt alpha must be in (0, 1)".to_string(),
            ));
        }
        if !(0.0 < beta && beta < 1.0) {
            return Err(TsdbError::Config("Holt beta must be in (0, 1)".to_string()));
        }
        Ok(Self {
            alpha,
            beta,
            confidence,
        })
    }
}

impl Default for DoubleExponentialSmoothing {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.1,
            confidence: 0.95,
        }
    }
}

impl Forecaster for DoubleExponentialSmoothing {
    fn name(&self) -> &'static str {
        "DoubleExponentialSmoothing"
    }

    fn forecast(&self, history: &[f64], horizon: usize) -> TsdbResult<Vec<ForecastPoint>> {
        let n = history.len();
        if n < 2 {
            return Err(TsdbError::Query(
                "Holt method requires at least 2 history points".to_string(),
            ));
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }

        // Initialise level and trend.
        let mut level = history[0];
        let mut trend = history[1] - history[0];

        let mut one_step_errors: Vec<f64> = Vec::with_capacity(n - 1);

        for &y in &history[1..] {
            let fitted = level + trend;
            one_step_errors.push(y - fitted);
            let prev_level = level;
            level = self.alpha * y + (1.0 - self.alpha) * (level + trend);
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
        }

        let resid_std = std_dev(&one_step_errors).max(f64::EPSILON);

        Ok((1..=horizon)
            .map(|h| {
                let forecast = level + h as f64 * trend;
                ForecastPoint::new(forecast, resid_std * (h as f64).sqrt(), self.confidence)
            })
            .collect())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Triple Exponential Smoothing (Holt-Winters)
// ──────────────────────────────────────────────────────────────────────────────

/// Additive Holt-Winters seasonal model.
///
/// Level:   `l_t = α · (y_t − s_{t−m}) + (1 − α) · (l_{t−1} + b_{t−1})`
/// Trend:   `b_t = β · (l_t − l_{t−1}) + (1 − β) · b_{t−1}`
/// Season:  `s_t = γ · (y_t − l_t) + (1 − γ) · s_{t−m}`
/// Forecast: `ŷ_{n+h} = l_n + h · b_n + s_{n+h−m·⌈h/m⌉}`
#[derive(Debug, Clone)]
pub struct TripleExponentialSmoothing {
    /// Level smoothing parameter in (0, 1).
    pub alpha: f64,
    /// Trend smoothing parameter in (0, 1).
    pub beta: f64,
    /// Seasonal smoothing parameter in (0, 1).
    pub gamma: f64,
    /// Season length (number of periods per cycle).
    pub period: usize,
    /// Confidence level for prediction intervals.
    pub confidence: f64,
}

impl TripleExponentialSmoothing {
    /// Create a Holt-Winters additive forecaster.
    pub fn new(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
        confidence: f64,
    ) -> TsdbResult<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(TsdbError::Config("HW alpha must be in (0,1)".to_string()));
        }
        if !(0.0 < beta && beta < 1.0) {
            return Err(TsdbError::Config("HW beta must be in (0,1)".to_string()));
        }
        if !(0.0 < gamma && gamma < 1.0) {
            return Err(TsdbError::Config("HW gamma must be in (0,1)".to_string()));
        }
        if period < 2 {
            return Err(TsdbError::Config("HW period must be >= 2".to_string()));
        }
        Ok(Self {
            alpha,
            beta,
            gamma,
            period,
            confidence,
        })
    }
}

impl Default for TripleExponentialSmoothing {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.2,
            period: 12,
            confidence: 0.95,
        }
    }
}

impl Forecaster for TripleExponentialSmoothing {
    fn name(&self) -> &'static str {
        "TripleExponentialSmoothing"
    }

    fn forecast(&self, history: &[f64], horizon: usize) -> TsdbResult<Vec<ForecastPoint>> {
        let n = history.len();
        let m = self.period;

        if n < 2 * m {
            return Err(TsdbError::Query(format!(
                "Holt-Winters requires at least {} history points (2 × period)",
                2 * m
            )));
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }

        // ── Initialisation ────────────────────────────────────────────────────
        // Level: mean of first season.
        let level_init = mean(&history[..m]);

        // Trend: average slope over the first two seasons.
        let trend_init = {
            let sum: f64 = (0..m)
                .map(|i| (history[m + i] - history[i]) / m as f64)
                .sum();
            sum / m as f64
        };

        // Seasonal indices: mean-centred values of each seasonal position.
        let n_seasons = n / m;
        let mut season = vec![0.0_f64; m];
        for k in 0..m {
            let vals: Vec<f64> = (0..n_seasons).map(|s| history[s * m + k]).collect();
            season[k] = mean(&vals) - level_init;
        }

        // ── Smoothing loop ────────────────────────────────────────────────────
        let mut level = level_init;
        let mut trend = trend_init;
        let mut seasonal = season.clone();
        let mut one_step_errors: Vec<f64> = Vec::with_capacity(n);

        for (t, &y) in history.iter().enumerate() {
            let s_idx = t % m;
            let fitted = level + trend + seasonal[s_idx];
            one_step_errors.push(y - fitted);

            let prev_level = level;
            level = self.alpha * (y - seasonal[s_idx]) + (1.0 - self.alpha) * (level + trend);
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
            seasonal[s_idx] = self.gamma * (y - level) + (1.0 - self.gamma) * seasonal[s_idx];
        }

        let resid_std = std_dev(&one_step_errors).max(f64::EPSILON);

        // ── Forecast ──────────────────────────────────────────────────────────
        Ok((1..=horizon)
            .map(|h| {
                // Seasonal index for step h (wraps within the last m seasonal values).
                let s_idx = (n + h - 1) % m;
                let value = level + h as f64 * trend + seasonal[s_idx];
                ForecastPoint::new(value, resid_std * (h as f64).sqrt(), self.confidence)
            })
            .collect())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ForecastEvaluator – rolling window back-testing
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluates a `Forecaster` using rolling-origin (expanding window) cross-
/// validation and returns summary error metrics.
pub struct ForecastEvaluator<'f> {
    forecaster: &'f dyn Forecaster,
    /// Minimum number of history points before the first evaluation window.
    min_history: usize,
    /// Number of steps ahead to forecast in each window.
    horizon: usize,
    /// Step size between consecutive windows.
    step: usize,
}

impl<'f> ForecastEvaluator<'f> {
    /// Create an evaluator.
    ///
    /// - `min_history`: minimum training set size before evaluation begins.
    /// - `horizon`: steps-ahead to forecast per window.
    /// - `step`: how many observations to advance the training set each time.
    pub fn new(
        forecaster: &'f dyn Forecaster,
        min_history: usize,
        horizon: usize,
        step: usize,
    ) -> TsdbResult<Self> {
        if min_history < 2 {
            return Err(TsdbError::Config("min_history must be >= 2".to_string()));
        }
        if horizon == 0 {
            return Err(TsdbError::Config("horizon must be >= 1".to_string()));
        }
        if step == 0 {
            return Err(TsdbError::Config("step must be >= 1".to_string()));
        }
        Ok(Self {
            forecaster,
            min_history,
            horizon,
            step,
        })
    }

    /// Run back-testing over the `data` slice and return aggregate metrics.
    pub fn evaluate(&self, data: &[f64]) -> TsdbResult<ForecastMetrics> {
        let n = data.len();
        if n < self.min_history + self.horizon {
            return Err(TsdbError::Query(format!(
                "Data too short: need >= {} points for evaluation",
                self.min_history + self.horizon
            )));
        }

        let mut actuals: Vec<f64> = Vec::new();
        let mut forecasts: Vec<f64> = Vec::new();

        let mut train_end = self.min_history;
        while train_end + self.horizon <= n {
            let history = &data[..train_end];
            let fcast = self.forecaster.forecast(history, self.horizon)?;
            let test = &data[train_end..train_end + self.horizon];

            for (fp, &actual) in fcast.iter().zip(test.iter()) {
                actuals.push(actual);
                forecasts.push(fp.value);
            }

            train_end += self.step;
        }

        if actuals.is_empty() {
            return Err(TsdbError::Query(
                "No evaluation windows could be formed".to_string(),
            ));
        }

        let n_windows = actuals.len();
        let y_mean = mean(&actuals);

        let mut mae = 0.0_f64;
        let mut mape = 0.0_f64;
        let mut ss_res = 0.0_f64;
        let mut ss_tot = 0.0_f64;

        for (&a, &f) in actuals.iter().zip(forecasts.iter()) {
            let e = a - f;
            mae += e.abs();
            mape += if a.abs() > f64::EPSILON {
                (e / a).abs() * 100.0
            } else {
                0.0
            };
            ss_res += e * e;
            ss_tot += (a - y_mean).powi(2);
        }

        let count = n_windows as f64;
        let rmse = (ss_res / count).sqrt();
        let r_squared = if ss_tot < f64::EPSILON {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

        Ok(ForecastMetrics {
            mae: mae / count,
            mape: mape / count,
            rmse,
            r_squared,
            n_windows,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a periodic series with additive trend and seasonal component.
    fn build_periodic(
        n: usize,
        period: usize,
        amplitude: f64,
        trend: f64,
        noise_scale: f64,
    ) -> Vec<f64> {
        // Deterministic synthetic series (no external rand dependency).
        (0..n)
            .map(|i| {
                let seasonal =
                    amplitude * ((2.0 * std::f64::consts::PI * i as f64 / period as f64).sin());
                let trend_component = trend * i as f64;
                // Tiny deterministic "noise" via bit-manipulation to keep tests reproducible.
                let noise = noise_scale * ((i as f64 * 1.6180339887).fract() - 0.5);
                100.0 + seasonal + trend_component + noise
            })
            .collect()
    }

    #[test]
    fn naive_forecast_constant_series() {
        let data = vec![5.0_f64; 30];
        let forecaster = NaiveForecast::new(1, 0.95).expect("ctor");
        let result = forecaster.forecast(&data, 5).expect("forecast");
        assert_eq!(result.len(), 5);
        for fp in &result {
            assert!((fp.value - 5.0).abs() < 1e-10);
            assert!(fp.lower_bound <= fp.value);
            assert!(fp.upper_bound >= fp.value);
        }
    }

    #[test]
    fn naive_seasonal_forecast() {
        let period = 4;
        let data: Vec<f64> = (0..20).map(|i| (i % period) as f64).collect();
        let forecaster = NaiveForecast::new(period, 0.95).expect("ctor");
        let result = forecaster.forecast(&data, period).expect("forecast");
        // The next cycle should match the previous one.
        for (h, fp) in result.iter().enumerate() {
            let expected = ((20 + h) % period) as f64;
            assert!(
                (fp.value - expected).abs() < 1e-10,
                "h={h}: got {}, expected {expected}",
                fp.value
            );
        }
    }

    #[test]
    fn naive_error_on_too_short_history() {
        let f = NaiveForecast::new(5, 0.95).expect("ctor");
        assert!(f.forecast(&[1.0, 2.0, 3.0], 2).is_err());
    }

    #[test]
    fn naive_zero_horizon_returns_empty() {
        let data = vec![1.0_f64; 10];
        let f = NaiveForecast::default();
        let r = f.forecast(&data, 0).expect("forecast");
        assert!(r.is_empty());
    }

    #[test]
    fn ses_forecast_captures_level() {
        // A step change: first 50 values at 0.0, next 50 at 100.0.
        let data: Vec<f64> = (0..100).map(|i| if i < 50 { 0.0 } else { 100.0 }).collect();
        let forecaster = SimpleExponentialSmoothing::new(0.5, 0.95).expect("ctor");
        let result = forecaster.forecast(&data, 3).expect("forecast");
        assert_eq!(result.len(), 3);
        // After seeing 50 values at 100.0, the level should be close to 100.
        for fp in &result {
            assert!(
                fp.value > 50.0,
                "level should have risen above 50, got {}",
                fp.value
            );
        }
    }

    #[test]
    fn ses_invalid_alpha() {
        assert!(SimpleExponentialSmoothing::new(0.0, 0.95).is_err());
        assert!(SimpleExponentialSmoothing::new(1.0, 0.95).is_err());
    }

    #[test]
    fn ses_error_on_too_few_points() {
        let f = SimpleExponentialSmoothing::default();
        assert!(f.forecast(&[1.0], 2).is_err());
    }

    #[test]
    fn ses_intervals_widen_with_horizon() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let f = SimpleExponentialSmoothing::new(0.3, 0.95).expect("ctor");
        let result = f.forecast(&data, 5).expect("forecast");
        // Width at h=5 should be wider than at h=1.
        let width = |fp: &ForecastPoint| fp.upper_bound - fp.lower_bound;
        assert!(
            width(&result[4]) > width(&result[0]),
            "intervals must widen with horizon"
        );
    }

    #[test]
    fn double_es_trend_direction() {
        // Monotonically increasing series.
        let data: Vec<f64> = (0..30).map(|i| i as f64 * 2.0).collect();
        let f = DoubleExponentialSmoothing::new(0.4, 0.2, 0.95).expect("ctor");
        let result = f.forecast(&data, 3).expect("forecast");
        assert_eq!(result.len(), 3);
        // Forecasts should be increasing.
        assert!(result[1].value > result[0].value);
        assert!(result[2].value > result[1].value);
    }

    #[test]
    fn double_es_invalid_params() {
        assert!(DoubleExponentialSmoothing::new(0.0, 0.3, 0.95).is_err());
        assert!(DoubleExponentialSmoothing::new(0.3, 1.0, 0.95).is_err());
    }

    #[test]
    fn triple_es_seasonal_pattern() {
        let period = 12;
        let n = 48;
        let data = build_periodic(n, period, 10.0, 0.0, 0.5);
        let f = TripleExponentialSmoothing::new(0.3, 0.1, 0.2, period, 0.95).expect("ctor");
        let result = f.forecast(&data, period).expect("forecast");
        assert_eq!(result.len(), period);
        // All forecast values should be finite.
        for fp in &result {
            assert!(
                fp.value.is_finite(),
                "forecast must be finite: {}",
                fp.value
            );
            assert!(fp.lower_bound < fp.upper_bound);
        }
    }

    #[test]
    fn triple_es_invalid_period() {
        assert!(TripleExponentialSmoothing::new(0.3, 0.1, 0.2, 1, 0.95).is_err());
    }

    #[test]
    fn triple_es_too_short_history() {
        let f = TripleExponentialSmoothing::new(0.3, 0.1, 0.2, 12, 0.95).expect("ctor");
        assert!(f.forecast(&[1.0; 20], 3).is_err()); // needs 2*12=24
    }

    #[test]
    fn forecast_metrics_structure() {
        let data = build_periodic(120, 12, 5.0, 0.1, 0.2);
        let f = SimpleExponentialSmoothing::new(0.3, 0.95).expect("ctor");
        let evaluator = ForecastEvaluator::new(&f, 24, 6, 6).expect("eval ctor");
        let metrics = evaluator.evaluate(&data).expect("evaluate");
        assert!(metrics.mae >= 0.0);
        assert!(metrics.mape >= 0.0);
        assert!(metrics.rmse >= 0.0);
        assert!(metrics.n_windows > 0);
    }

    #[test]
    fn forecast_evaluator_holt_winters() {
        let period = 12;
        let data = build_periodic(120, period, 15.0, 0.05, 0.1);
        let f = TripleExponentialSmoothing::new(0.2, 0.05, 0.1, period, 0.95).expect("ctor");
        let evaluator = ForecastEvaluator::new(&f, 2 * period, 3, 3).expect("eval ctor");
        let metrics = evaluator.evaluate(&data).expect("evaluate");
        assert!(metrics.rmse.is_finite());
        assert!(metrics.r_squared <= 1.0);
    }

    #[test]
    fn forecast_evaluator_error_on_too_short_data() {
        let f = SimpleExponentialSmoothing::default();
        let evaluator = ForecastEvaluator::new(&f, 10, 5, 2).expect("eval ctor");
        assert!(evaluator.evaluate(&[1.0_f64; 10]).is_err());
    }

    #[test]
    fn forecast_point_bounds_valid() {
        let fp = ForecastPoint::new(50.0, 5.0, 0.95);
        assert!(fp.lower_bound < fp.value);
        assert!(fp.upper_bound > fp.value);
        assert!((fp.upper_bound - fp.lower_bound) > 0.0);
    }

    #[test]
    fn all_forecasters_have_names() {
        let naive = NaiveForecast::default();
        let ses = SimpleExponentialSmoothing::default();
        let des = DoubleExponentialSmoothing::default();
        let tes = TripleExponentialSmoothing::default();
        assert!(!naive.name().is_empty());
        assert!(!ses.name().is_empty());
        assert!(!des.name().is_empty());
        assert!(!tes.name().is_empty());
    }
}
